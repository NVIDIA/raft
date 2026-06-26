/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <int warpSize, int tpb, int rpw, bool noLoop = false>
struct ReductionThinPolicy {
  static_assert(tpb % warpSize == 0);

  static constexpr int LogicalWarpSize    = warpSize;
  static constexpr int ThreadsPerBlock    = tpb;
  static constexpr int RowsPerLogicalWarp = rpw;
  static constexpr int NumLogicalWarps    = ThreadsPerBlock / LogicalWarpSize;
  static constexpr int RowsPerBlock       = NumLogicalWarps * RowsPerLogicalWarp;

  // Whether D (run-time arg) will be smaller than warpSize (compile-time parameter)
  static constexpr bool NoSequentialReduce = noLoop;
};

template <typename Type>
DI void KahanBabushkaNeumaierSum(Type& sum, Type& compensation, const Type& cur_value)
{
  const Type t = sum + cur_value;
  if (abs(sum) >= abs(cur_value)) {
    compensation += (sum - t) + cur_value;
  } else {
    compensation += (cur_value - t) + sum;
  }
  sum = t;
}

template <typename Policy,
          typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
RAFT_KERNEL __launch_bounds__(Policy::ThreadsPerBlock)
  coalescedReductionThinKernel(OutType* dots,
                               const InType* data,
                               IdxType D,
                               IdxType N,
                               OutType init,
                               MainLambda main_op,
                               ReduceLambda reduce_op,
                               FinalLambda final_op,
                               bool inplace = false)
{
  /* The strategy to achieve near-SOL memory bandwidth differs based on D:
   *  - For small D, we need to process multiple rows per logical warp in order to have
   *    multiple loads per thread and increase bytes in flight and amortize latencies.
   *  - For large D, we start with a sequential reduction. The compiler partially unrolls
   *    that loop (e.g. first a loop of stride 16, then 8, 4, and 1).
   */
  IdxType i0 = threadIdx.y + (Policy::RowsPerBlock * static_cast<IdxType>(blockIdx.x));
  if (i0 >= N) return;

  OutType acc[Policy::RowsPerLogicalWarp];
#pragma unroll
  for (int k = 0; k < Policy::RowsPerLogicalWarp; k++) {
    acc[k] = init;
  }

  if constexpr (Policy::NoSequentialReduce) {
    IdxType j = threadIdx.x;
    if (j < D) {
#pragma unroll
      for (IdxType k = 0; k < Policy::RowsPerLogicalWarp; k++) {
        // Only the first row is known to be within bounds. Clamp to avoid out-of-mem read.
        const IdxType i = raft::min(i0 + k * Policy::NumLogicalWarps, N - 1);
        acc[k]          = reduce_op(acc[k], main_op(data[j + (D * i)], j));
      }
    }
  } else {
    for (IdxType j = threadIdx.x; j < D; j += Policy::LogicalWarpSize) {
#pragma unroll
      for (IdxType k = 0; k < Policy::RowsPerLogicalWarp; k++) {
        const IdxType i = raft::min(i0 + k * Policy::NumLogicalWarps, N - 1);
        acc[k]          = reduce_op(acc[k], main_op(data[j + (D * i)], j));
      }
    }
  }

  /* This vector reduction has two benefits compared to naive separate reductions:
   * - It avoids the LSU bottleneck when the number of columns is around 32 (e.g. for 32, 5 shuffles
   *   are required and there is no initial sequential reduction to amortize that cost).
   * - It distributes the outputs to multiple threads, enabling a coalesced store when the number of
   *   rows per logical warp and logical warp size are equal.
   */
  raft::logicalWarpReduceVector<Policy::LogicalWarpSize, Policy::RowsPerLogicalWarp>(
    acc, threadIdx.x, reduce_op);

  constexpr int reducOutVecWidth =
    std::max(1, Policy::RowsPerLogicalWarp / Policy::LogicalWarpSize);
  constexpr int reducOutGroupSize =
    std::max(1, Policy::LogicalWarpSize / Policy::RowsPerLogicalWarp);
  constexpr int reducNumGroups = Policy::LogicalWarpSize / reducOutGroupSize;

  if (threadIdx.x % reducOutGroupSize == 0) {
    const int groupId = threadIdx.x / reducOutGroupSize;
    if (inplace) {
#pragma unroll
      for (int k = 0; k < reducOutVecWidth; k++) {
        const int reductionId = k * reducNumGroups + groupId;
        const IdxType i       = i0 + reductionId * Policy::NumLogicalWarps;
        if (i < N) { dots[i] = final_op(reduce_op(dots[i], acc[k])); }
      }
    } else {
#pragma unroll
      for (int k = 0; k < reducOutVecWidth; k++) {
        const int reductionId = k * reducNumGroups + groupId;
        const IdxType i       = i0 + reductionId * Policy::NumLogicalWarps;
        if (i < N) { dots[i] = final_op(acc[k]); }
      }
    }
  }
}

template <typename Policy,
          typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename FinalLambda>
RAFT_KERNEL __launch_bounds__(Policy::ThreadsPerBlock) coalescedSumThinKernel(OutType* dots,
                                                                              const InType* data,
                                                                              IdxType D,
                                                                              IdxType N,
                                                                              OutType init,
                                                                              MainLambda main_op,
                                                                              FinalLambda final_op,
                                                                              bool inplace = false)
{
  /* The strategy to achieve near-SOL memory bandwidth differs based on D:
   *  - For small D, we need to process multiple rows per logical warp in order to have
   *    multiple loads per thread and increase bytes in flight and amortize latencies.
   *  - For large D, we start with a sequential reduction. The compiler partially unrolls
   *    that loop (e.g. first a loop of stride 16, then 8, 4, and 1).
   */
  IdxType i0 = threadIdx.y + (Policy::RowsPerBlock * static_cast<IdxType>(blockIdx.x));
  if (i0 >= N) return;

  OutType acc[Policy::RowsPerLogicalWarp];
  OutType thread_c[Policy::RowsPerLogicalWarp];

#pragma unroll
  for (int k = 0; k < Policy::RowsPerLogicalWarp; k++) {
    acc[k]      = init;
    thread_c[k] = 0;
  }

  if constexpr (Policy::NoSequentialReduce) {
    IdxType j = threadIdx.x;
    if (j < D) {
#pragma unroll
      for (IdxType k = 0; k < Policy::RowsPerLogicalWarp; k++) {
        // Only the first row is known to be within bounds. Clamp to avoid out-of-mem read.
        const IdxType i = raft::min(i0 + k * Policy::NumLogicalWarps, N - 1);
        // acc[k]          = reduce_op(acc[k], main_op(data[j + (D * i)], j));
        KahanBabushkaNeumaierSum<OutType>(acc[k], thread_c[k], main_op(data[j + (D * i)], j));
      }
    }
  } else {
    for (IdxType j = threadIdx.x; j < D; j += Policy::LogicalWarpSize) {
#pragma unroll
      for (IdxType k = 0; k < Policy::RowsPerLogicalWarp; k++) {
        const IdxType i = raft::min(i0 + k * Policy::NumLogicalWarps, N - 1);
        // acc[k]          = reduce_op(acc[k], main_op(data[j + (D * i)], j));
        KahanBabushkaNeumaierSum<OutType>(acc[k], thread_c[k], main_op(data[j + (D * i)], j));
      }
    }
  }

  /* This vector reduction has two benefits compared to naive separate reductions:
   * - It avoids the LSU bottleneck when the number of columns is around 32 (e.g. for 32, 5 shuffles
   *   are required and there is no initial sequential reduction to amortize that cost).
   * - It distributes the outputs to multiple threads, enabling a coalesced store when the number of
   *   rows per logical warp and logical warp size are equal.
   */
  raft::logicalWarpReduceVector<Policy::LogicalWarpSize, Policy::RowsPerLogicalWarp>(
    acc, threadIdx.x, raft::add_op());

  raft::logicalWarpReduceVector<Policy::LogicalWarpSize, Policy::RowsPerLogicalWarp>(
    thread_c, threadIdx.x, raft::add_op());

  constexpr int reducOutVecWidth =
    std::max(1, Policy::RowsPerLogicalWarp / Policy::LogicalWarpSize);
  constexpr int reducOutGroupSize =
    std::max(1, Policy::LogicalWarpSize / Policy::RowsPerLogicalWarp);
  constexpr int reducNumGroups = Policy::LogicalWarpSize / reducOutGroupSize;

  if (threadIdx.x % reducOutGroupSize == 0) {
    const int groupId = threadIdx.x / reducOutGroupSize;
    if (inplace) {
#pragma unroll
      for (int k = 0; k < reducOutVecWidth; k++) {
        const int reductionId = k * reducNumGroups + groupId;
        const IdxType i       = i0 + reductionId * Policy::NumLogicalWarps;
        if (i < N) { dots[i] = final_op(dots[i] + acc[k] + thread_c[k]); }
      }
    } else {
#pragma unroll
      for (int k = 0; k < reducOutVecWidth; k++) {
        const int reductionId = k * reducNumGroups + groupId;
        const IdxType i       = i0 + reductionId * Policy::NumLogicalWarps;
        if (i < N) { dots[i] = final_op(acc[k] + thread_c[k]); }
      }
    }
  }
}

template <typename Policy,
          typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionThin(OutType* dots,
                            const InType* data,
                            IdxType D,
                            IdxType N,
                            OutType init,
                            cudaStream_t stream,
                            bool inplace           = false,
                            MainLambda main_op     = raft::identity_op(),
                            ReduceLambda reduce_op = raft::add_op(),
                            FinalLambda final_op   = raft::identity_op())
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "coalescedReductionThin<%d,%d,%d,%d>",
    Policy::LogicalWarpSize,
    Policy::ThreadsPerBlock,
    Policy::RowsPerLogicalWarp,
    static_cast<int>(Policy::NoSequentialReduce));
  dim3 threads(Policy::LogicalWarpSize, Policy::NumLogicalWarps, 1);
  dim3 blocks(ceildiv<IdxType>(N, Policy::RowsPerBlock), 1, 1);
  if constexpr (std::is_same_v<ReduceLambda, raft::add_op>) {
    coalescedSumThinKernel<Policy>
      <<<blocks, threads, 0, stream>>>(dots, data, D, N, init, main_op, final_op, inplace);
  } else {
    coalescedReductionThinKernel<Policy><<<blocks, threads, 0, stream>>>(
      dots, data, D, N, init, main_op, reduce_op, final_op, inplace);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionThinDispatcher(OutType* dots,
                                      const InType* data,
                                      IdxType D,
                                      IdxType N,
                                      OutType init,
                                      cudaStream_t stream,
                                      bool inplace           = false,
                                      MainLambda main_op     = raft::identity_op(),
                                      ReduceLambda reduce_op = raft::add_op(),
                                      FinalLambda final_op   = raft::identity_op())
{
  if (D <= IdxType(2)) {
    coalescedReductionThin<ReductionThinPolicy<2, 128, 8, true>>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if (D <= IdxType(4)) {
    coalescedReductionThin<ReductionThinPolicy<4, 128, 8, true>>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if (D <= IdxType(8)) {
    coalescedReductionThin<ReductionThinPolicy<8, 128, 8, true>>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if (D <= IdxType(16)) {
    coalescedReductionThin<ReductionThinPolicy<16, 128, 8, true>>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if (D <= IdxType(32)) {
    coalescedReductionThin<ReductionThinPolicy<32, 128, 8, true>>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if (D < IdxType(128)) {
    coalescedReductionThin<ReductionThinPolicy<32, 128, 4, false>>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else {
    // For D=128 (included) and above, the 4x-unrolled loading loop is used
    // and multiple rows per warp are counter-productive in terms of cache-friendliness
    // and register use.
    coalescedReductionThin<ReductionThinPolicy<32, 128, 1, false>>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  }
}

template <int TPB,
          typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
RAFT_KERNEL __launch_bounds__(TPB) coalescedReductionMediumKernel(OutType* dots,
                                                                  const InType* data,
                                                                  IdxType D,
                                                                  IdxType N,
                                                                  OutType init,
                                                                  MainLambda main_op,
                                                                  ReduceLambda reduce_op,
                                                                  FinalLambda final_op,
                                                                  bool inplace = false)
{
  typedef cub::BlockReduce<OutType, TPB, cub::BLOCK_REDUCE_RAKING> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  OutType thread_data = init;
  IdxType rowStart    = blockIdx.x * D;
  for (IdxType i = threadIdx.x; i < D; i += TPB) {
    IdxType idx = rowStart + i;
    thread_data = reduce_op(thread_data, main_op(data[idx], i));
  }
  OutType acc = BlockReduce(temp_storage).Reduce(thread_data, reduce_op);
  if (threadIdx.x == 0) {
    if (inplace) {
      dots[blockIdx.x] = final_op(reduce_op(dots[blockIdx.x], acc));
    } else {
      dots[blockIdx.x] = final_op(acc);
    }
  }
}

template <int TPB,
          typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename FinalLambda>
RAFT_KERNEL __launch_bounds__(TPB) coalescedSumMediumKernel(OutType* dots,
                                                            const InType* data,
                                                            IdxType D,
                                                            IdxType N,
                                                            OutType init,
                                                            MainLambda main_op,
                                                            FinalLambda final_op,
                                                            bool inplace = false)
{
  typedef cub::BlockReduce<OutType, TPB, cub::BLOCK_REDUCE_RAKING> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;
  OutType thread_data = init;
  OutType thread_c    = (OutType)0;

  IdxType rowStart = blockIdx.x * D;
  for (IdxType i = threadIdx.x; i < D; i += TPB) {
    IdxType idx = rowStart + i;
    KahanBabushkaNeumaierSum<OutType>(thread_data, thread_c, main_op(data[idx], i));
  }
  OutType block_acc = BlockReduce(temp_storage1).Sum(thread_data);
  OutType block_c   = BlockReduce(temp_storage2).Sum(thread_c);

  if (threadIdx.x == 0) {
    if (inplace) {
      dots[blockIdx.x] = final_op(dots[blockIdx.x] + block_acc + block_c);
    } else {
      dots[blockIdx.x] = final_op(block_acc + block_c);
    }
  }
}

template <int TPB,
          typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionMedium(OutType* dots,
                              const InType* data,
                              IdxType D,
                              IdxType N,
                              OutType init,
                              cudaStream_t stream,
                              bool inplace           = false,
                              MainLambda main_op     = raft::identity_op(),
                              ReduceLambda reduce_op = raft::add_op(),
                              FinalLambda final_op   = raft::identity_op())
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("coalescedReductionMedium<%d>", TPB);
  if constexpr (std::is_same_v<ReduceLambda, raft::add_op>) {
    coalescedSumMediumKernel<TPB>
      <<<N, TPB, 0, stream>>>(dots, data, D, N, init, main_op, final_op, inplace);
  } else {
    coalescedReductionMediumKernel<TPB>
      <<<N, TPB, 0, stream>>>(dots, data, D, N, init, main_op, reduce_op, final_op, inplace);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionMediumDispatcher(OutType* dots,
                                        const InType* data,
                                        IdxType D,
                                        IdxType N,
                                        OutType init,
                                        cudaStream_t stream,
                                        bool inplace           = false,
                                        MainLambda main_op     = raft::identity_op(),
                                        ReduceLambda reduce_op = raft::add_op(),
                                        FinalLambda final_op   = raft::identity_op())
{
  // Note: for now, this kernel is only used when D > 256. If this changes in the future, use
  // smaller block sizes when relevant.
  coalescedReductionMedium<256>(
    dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
}

template <int tpb, int bpr>
struct ReductionThickPolicy {
  static constexpr int ThreadsPerBlock = tpb;
  static constexpr int BlocksPerRow    = bpr;
  static constexpr int BlockStride     = tpb * bpr;
};

template <typename Policy,
          typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda>
RAFT_KERNEL __launch_bounds__(Policy::ThreadsPerBlock)
  coalescedReductionThickKernel(OutType* buffer,
                                const InType* data,
                                IdxType D,
                                IdxType N,
                                OutType init,
                                MainLambda main_op,
                                ReduceLambda reduce_op)
{
  typedef cub::BlockReduce<OutType, Policy::ThreadsPerBlock, cub::BLOCK_REDUCE_RAKING> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  OutType thread_data = init;
  IdxType rowStart    = blockIdx.x * D;
  for (IdxType i = blockIdx.y * Policy::ThreadsPerBlock + threadIdx.x; i < D;
       i += Policy::BlockStride) {
    IdxType idx = rowStart + i;
    thread_data = reduce_op(thread_data, main_op(data[idx], i));
  }
  OutType acc = BlockReduce(temp_storage).Reduce(thread_data, reduce_op);
  if (threadIdx.x == 0) { buffer[Policy::BlocksPerRow * blockIdx.x + blockIdx.y] = acc; }
}

template <typename Policy, typename InType, typename OutType, typename IdxType, typename MainLambda>
RAFT_KERNEL __launch_bounds__(Policy::ThreadsPerBlock) coalescedSumThickKernel(
  OutType* buffer, const InType* data, IdxType D, IdxType N, OutType init, MainLambda main_op)
{
  typedef cub::BlockReduce<OutType, Policy::ThreadsPerBlock, cub::BLOCK_REDUCE_RAKING> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage1;
  __shared__ typename BlockReduce::TempStorage temp_storage2;

  OutType thread_data = init;
  OutType thread_c    = (OutType)0;

  IdxType rowStart = blockIdx.x * D;
  for (IdxType i = blockIdx.y * Policy::ThreadsPerBlock + threadIdx.x; i < D;
       i += Policy::BlockStride) {
    IdxType idx = rowStart + i;
    KahanBabushkaNeumaierSum<OutType>(thread_data, thread_c, main_op(data[idx], i));
  }

  OutType block_acc = BlockReduce(temp_storage1).Sum(thread_data);
  OutType block_c   = BlockReduce(temp_storage2).Sum(thread_c);

  if (threadIdx.x == 0) {
    buffer[Policy::BlocksPerRow * blockIdx.x + blockIdx.y] = block_acc + block_c;
  }
}

template <typename ThickPolicy,
          typename ThinPolicy,
          typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionThick(bool dry_run,
                             OutType* dots,
                             const InType* data,
                             IdxType D,
                             IdxType N,
                             OutType init,
                             cudaStream_t stream,
                             bool inplace           = false,
                             MainLambda main_op     = raft::identity_op(),
                             ReduceLambda reduce_op = raft::add_op(),
                             FinalLambda final_op   = raft::identity_op())
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "coalescedReductionThick<%d,%d>", ThickPolicy::ThreadsPerBlock, ThickPolicy::BlocksPerRow);

  dim3 threads(ThickPolicy::ThreadsPerBlock, 1, 1);
  dim3 blocks(N, ThickPolicy::BlocksPerRow, 1);

  rmm::device_uvector<OutType> buffer(N * ThickPolicy::BlocksPerRow, stream);

  if (dry_run) { return; }

  /* We apply a two-step reduction:
   *  1. coalescedReductionThickKernel reduces the [N x D] input data to [N x BlocksPerRow]. It
   *     applies the main_op but not the final op.
   *  2. coalescedReductionThinKernel reduces [N x BlocksPerRow] to [N x 1]. It doesn't apply any
   *     main_op but applies final_op. If in-place, the existing and new values are reduced.
   */
  if constexpr (std::is_same_v<ReduceLambda, raft::add_op>) {
    coalescedSumThickKernel<ThickPolicy>
      <<<blocks, threads, 0, stream>>>(buffer.data(), data, D, N, init, main_op);
  } else {
    coalescedReductionThickKernel<ThickPolicy>
      <<<blocks, threads, 0, stream>>>(buffer.data(), data, D, N, init, main_op, reduce_op);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  coalescedReductionThin<ThinPolicy>(dots,
                                     buffer.data(),
                                     static_cast<IdxType>(ThickPolicy::BlocksPerRow),
                                     N,
                                     init,
                                     stream,
                                     inplace,
                                     raft::identity_op(),
                                     reduce_op,
                                     final_op);
}

template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionThickDispatcher(bool dry_run,
                                       OutType* dots,
                                       const InType* data,
                                       IdxType D,
                                       IdxType N,
                                       OutType init,
                                       cudaStream_t stream,
                                       bool inplace           = false,
                                       MainLambda main_op     = raft::identity_op(),
                                       ReduceLambda reduce_op = raft::add_op(),
                                       FinalLambda final_op   = raft::identity_op())
{
  // Note: multiple elements per thread to take advantage of the sequential reduction and loop
  // unrolling
  coalescedReductionThick<ReductionThickPolicy<256, 64>, ReductionThinPolicy<32, 128, 1>>(
    dry_run, dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
}

// Primitive to perform reductions along the coalesced dimension of the matrix, i.e. reduce along
// rows for row major or reduce along columns for column major layout. Can do an inplace reduction
// adding to original values of dots if requested.
// In case of an add-reduction, a compensated summation will be performed in order to reduce
// numerical error. Note that the compensation will only be performed 'per-thread' for performance
// reasons and therefore not be equivalent to a sequential compensation.
template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReduction(bool dry_run,
                        OutType* dots,
                        const InType* data,
                        IdxType D,
                        IdxType N,
                        OutType init,
                        cudaStream_t stream,
                        bool inplace           = false,
                        MainLambda main_op     = raft::identity_op(),
                        ReduceLambda reduce_op = raft::add_op(),
                        FinalLambda final_op   = raft::identity_op())
{
  /* The primitive selects one of three implementations based on heuristics:
   *  - Thin: very efficient when D is small and/or N is large
   *    At most one warp is processing each row
   *  - Thick: used when N is very small and D very large
   *    Multiple blocks (32/64) processing each row
   *  - Medium: everything in between
   *    One block is processing each row
   */
  const IdxType numSMs = raft::getMultiProcessorCount();
  if (D <= IdxType(512) || (N >= IdxType(16) * numSMs && D < IdxType(2048))) {
    if (dry_run) { return; }
    coalescedReductionThinDispatcher(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if (N < numSMs && D >= IdxType(1 << 17)) {
    // Must call through to coalescedReductionThick even in dry-run so workspace
    // allocations are recorded (coalescedReductionThick allocates before guarding).
    coalescedReductionThickDispatcher(
      dry_run, dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else {
    if (dry_run) { return; }
    coalescedReductionMediumDispatcher(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  }
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft
