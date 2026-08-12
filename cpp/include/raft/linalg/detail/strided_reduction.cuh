/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

#include <type_traits>

namespace raft {
namespace linalg {
namespace detail {

// Kernel to perform summation along the strided dimension
// of the matrix, i.e. reduce along columns for row major or reduce along rows
// for column major layout
// A compensated summation will be performed in order to reduce numerical error.
// Note that the compensation will only be performed 'per-block' for performance
// reasons and therefore not be equivalent to a sequential compensation.

template <typename Type, typename IdxType, typename MainLambda>
RAFT_KERNEL stridedSummationKernel(
  Type* out, const Type* data, IdxType D, IdxType N, Type init, MainLambda main_op)
{
  // Thread reduction
  Type thread_sum  = Type(init);
  Type thread_c    = Type(0);
  IdxType colStart = blockIdx.x * blockDim.x + threadIdx.x;
  if (colStart < D) {
    IdxType rowStart = blockIdx.y * blockDim.y + threadIdx.y;
    IdxType stride   = blockDim.y * gridDim.y;
    for (IdxType j = rowStart; j < N; j += stride) {
      auto idx = colStart + j * D;

      // KahanBabushkaNeumaierSum
      const Type cur_value = main_op(data[idx], j);
      const Type t         = thread_sum + cur_value;
      if (abs(thread_sum) >= abs(cur_value)) {
        thread_c += (thread_sum - t) + cur_value;
      } else {
        thread_c += (cur_value - t) + thread_sum;
      }
      thread_sum = t;
    }
  }

  // Block reduction
  extern __shared__ char tmp[];
  auto* block_sum = (Type*)tmp;
  auto* block_c   = block_sum + blockDim.x;

  if (threadIdx.y == 0) {
    block_sum[threadIdx.x] = Type(0);
    block_c[threadIdx.x]   = Type(0);
  }
  __syncthreads();
  // also compute compensation for block-sum
  const Type old_sum = atomicAdd(block_sum + threadIdx.x, thread_sum);
  const Type t       = old_sum + thread_sum;
  if (abs(old_sum) >= abs(thread_sum)) {
    thread_c += (old_sum - t) + thread_sum;
  } else {
    thread_c += (thread_sum - t) + old_sum;
  }
  raft::myAtomicAdd(block_c + threadIdx.x, thread_c);
  __syncthreads();

  // Grid reduction
  if (colStart < D && (threadIdx.y == 0))
    raft::myAtomicAdd(out + colStart, block_sum[threadIdx.x] + block_c[threadIdx.x]);
}

// Kernel to perform reductions along the strided dimension
// of the matrix, i.e. reduce along columns for row major or reduce along rows
// for column major layout
template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda>
RAFT_KERNEL stridedReductionKernel(OutType* dots,
                                   const InType* data,
                                   IdxType D,
                                   IdxType N,
                                   OutType init,
                                   MainLambda main_op,
                                   ReduceLambda reduce_op)
{
  // Thread reduction
  OutType thread_data = init;
  IdxType colStart    = blockIdx.x * blockDim.x + threadIdx.x;
  if (colStart < D) {
    IdxType rowStart = blockIdx.y * blockDim.y + threadIdx.y;
    IdxType stride   = blockDim.y * gridDim.y;
    for (IdxType j = rowStart; j < N; j += stride) {
      IdxType idx = colStart + j * D;
      thread_data = reduce_op(thread_data, main_op(data[idx], j));
    }
  }

  // Block reduction
  extern __shared__ char tmp[];   // One element per thread in block
  auto* temp    = (OutType*)tmp;  // Cast to desired type
  IdxType myidx = threadIdx.x + ((IdxType)blockDim.x * (IdxType)threadIdx.y);
  temp[myidx]   = thread_data;
  __syncthreads();
  for (int j = blockDim.y / 2; j > 0; j /= 2) {
    if (threadIdx.y < j) temp[myidx] = reduce_op(temp[myidx], temp[myidx + j * blockDim.x]);
    __syncthreads();
  }

  // Grid reduction
  if ((colStart < D) && (threadIdx.y == 0))
    raft::myAtomicReduce(dots + colStart, temp[myidx], reduce_op);
}

template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void stridedReduction(OutType* dots,
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
  ///@todo: this extra should go away once we have eliminated the need
  /// for atomics in stridedKernel (redesign for this is already underway)
  if (!inplace) raft::linalg::unaryOp(dots, dots, D, raft::const_op(init), stream);

  ///@todo: this complication should go away once we have eliminated the need
  /// for atomics in stridedKernel (redesign for this is already underway)
  if constexpr (std::is_same<ReduceLambda, raft::add_op>::value &&
                std::is_same<InType, OutType>::value) {
    constexpr int TPB        = 256;
    constexpr int ColsPerBlk = 8;
    constexpr dim3 Block(ColsPerBlk, TPB / ColsPerBlk);
    constexpr int MinRowsPerThread = 16;
    constexpr int MinRowsPerBlk    = Block.y * MinRowsPerThread;
    constexpr int MaxBlocksDimY    = 8192;

    const dim3 grid(raft::ceildiv(D, (IdxType)ColsPerBlk),
                    raft::min((IdxType)MaxBlocksDimY, raft::ceildiv(N, (IdxType)MinRowsPerBlk)));
    const size_t shmemSize = sizeof(OutType) * Block.x * 2;

    raft::launch_kernel({stream, shmemSize},
                        grid,
                        Block,
                        stridedSummationKernel<InType, IdxType>,
                        dots,
                        data,
                        D,
                        N,
                        init,
                        main_op);
  } else {
    // Arbitrary numbers for now, probably need to tune
    const dim3 thrds(32, 16);
    IdxType elemsPerThread = raft::ceildiv(N, (IdxType)thrds.y);
    elemsPerThread         = (elemsPerThread > 8) ? 8 : elemsPerThread;
    const dim3 nblks(raft::ceildiv(D, (IdxType)thrds.x),
                     raft::ceildiv(N, (IdxType)thrds.y * elemsPerThread));
    const size_t shmemSize = sizeof(OutType) * thrds.x * thrds.y;

    raft::launch_kernel({stream, shmemSize},
                        nblks,
                        thrds,
                        stridedReductionKernel<InType, OutType, IdxType>,
                        dots,
                        data,
                        D,
                        N,
                        init,
                        main_op,
                        reduce_op);
  }

  ///@todo: this complication should go away once we have eliminated the need
  /// for atomics in stridedKernel (redesign for this is already underway)
  // Perform final op on output data
  if (!std::is_same<FinalLambda, raft::identity_op>::value)
    raft::linalg::unaryOp(dots, dots, D, final_op, stream);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
