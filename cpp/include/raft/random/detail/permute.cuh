/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>
#include <raft/util/vectorized.cuh>

#include <cooperative_groups.h>
#include <cuda/iterator>
#include <cuda/std/random>

namespace raft {
namespace random {
namespace detail {

template <typename IntType,
          typename IdxType,
          int TPB,
          int ITEMS_PER_THREAD,
          typename ShuffleIterator>
RAFT_KERNEL permsOnlyKernel(IntType* perms, ShuffleIterator shuffled_indices, IdxType N)
{
  IdxType base = IdxType(blockIdx.x) * IdxType(TPB * ITEMS_PER_THREAD) + threadIdx.x;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    IdxType idx = base + IdxType(i * TPB);
    if (idx < N) { perms[idx] = IntType(shuffled_indices[idx]); }
  }
}

template <typename Type,
          typename IntType,
          typename IdxType,
          int TPB,
          bool rowMajor,
          typename ShuffleIterator>
RAFT_KERNEL permuteKernel(
  IntType* perms, Type* out, const Type* in, ShuffleIterator shuffled_indices, IdxType N, IdxType D)
{
  namespace cg        = cooperative_groups;
  const int WARP_SIZE = 32;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  IntType outIdx = tid;
  IntType inIdx  = (tid < N) ? IntType(shuffled_indices[tid]) : IntType(0);

  if (perms != nullptr && tid < N) { perms[outIdx] = inIdx; }

  if (out == nullptr || in == nullptr) { return; }

  if (rowMajor) {
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

    // The warp cooperatively copies its 32 rows one at a time so that the 32
    // lanes stride together along D, giving coalesced global loads and stores.
    // Copying row i needs lane i's (inIdx, outIdx);
    int laneID = threadIdx.x % WARP_SIZE;
    for (int i = 0; i < WARP_SIZE; ++i) {
      IntType inIdxI  = warp.shfl(inIdx, i);
      IntType outIdxI = warp.shfl(outIdx, i);
      if (outIdxI < N) {
#pragma unroll
        for (int j = laneID; j < D; j += WARP_SIZE) {
          out[outIdxI * D + j] = in[inIdxI * D + j];
        }
      }
    }
  } else {
#pragma unroll
    for (int j = 0; j < D; ++j) {
      if (tid < N) { out[outIdx + j * N] = in[inIdx + j * N]; }
    }
  }
}

// This is wrapped in a type to allow for partial template specialization
template <typename Type, typename IntType, typename IdxType, int TPB, bool rowMajor, int VLen>
struct permute_impl_t {
  template <typename ShuffleIterator>
  static void permuteImpl(IntType* perms,
                          Type* out,
                          const Type* in,
                          IdxType N,
                          IdxType D,
                          int nblks,
                          ShuffleIterator shuffled_indices,
                          cudaStream_t stream)
  {
    // determine vector type and set new pointers
    typedef typename raft::IOType<Type, VLen>::Type VType;
    VType* vout      = reinterpret_cast<VType*>(out);
    const VType* vin = reinterpret_cast<const VType*>(in);

    // check if we can execute at this vector length
    if (D % VLen == 0 && raft::is_aligned(vout, sizeof(VType)) &&
        raft::is_aligned(vin, sizeof(VType))) {
      raft::launch_kernel(stream,
                          nblks,
                          TPB,
                          permuteKernel<VType, IntType, IdxType, TPB, rowMajor>,
                          perms,
                          vout,
                          vin,
                          fp,
                          N,
                          D / VLen);
    } else {  // otherwise try the next lower vector length
      permute_impl_t<Type, IntType, IdxType, TPB, rowMajor, VLen / 2>::permuteImpl(
        perms, out, in, N, D, nblks, shuffled_indices, stream);
    }
  }
};

// at vector length 1 we just execute a scalar version to break the recursion
template <typename Type, typename IntType, typename IdxType, int TPB, bool rowMajor>
struct permute_impl_t<Type, IntType, IdxType, TPB, rowMajor, 1> {
  template <typename ShuffleIterator>
  static void permuteImpl(IntType* perms,
                          Type* out,
                          const Type* in,
                          IdxType N,
                          IdxType D,
                          int nblks,
                          ShuffleIterator shuffled_indices,
                          cudaStream_t stream)
  {
    raft::launch_kernel(stream,
                        nblks,
                        TPB,
                        permuteKernel<Type, IntType, IdxType, TPB, rowMajor>,
                        perms,
                        out,
                        in,
                        fp,
                        N,
                        D);
  }
};

template <typename Type, typename IntType = int, typename IdxType = int, int TPB = 256>
void permute(IntType* perms,
             Type* out,
             const Type* in,
             IntType D,
             IntType N,
             bool rowMajor,
             cudaStream_t stream,
             uint64_t key)
{
  if (N <= 0) { return; }

  cuda::shuffle_iterator shuffled_indices{cuda::random_bijection{N, cuda::std::minstd_rand{key}}};

  if (out == nullptr) {
    constexpr int ITEMS_PER_THREAD = 8;
    auto nblks                     = raft::ceildiv(N, IntType(TPB * ITEMS_PER_THREAD));
    permsOnlyKernel<IntType, IntType, TPB, ITEMS_PER_THREAD>
      <<<nblks, TPB, 0, stream>>>(perms, shuffled_indices, N);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    return;
  }

  auto nblks = raft::ceildiv(N, (IntType)TPB);

  if (rowMajor) {
    permute_impl_t<Type,
                   IntType,
                   IdxType,
                   TPB,
                   true,
                   (16 / sizeof(Type) > 0) ? 16 / sizeof(Type) : 1>::permuteImpl(perms,
                                                                                 out,
                                                                                 in,
                                                                                 N,
                                                                                 D,
                                                                                 nblks,
                                                                                 shuffled_indices,
                                                                                 stream);
  } else {
    permute_impl_t<Type, IntType, IdxType, TPB, false, 1>::permuteImpl(
      perms, out, in, N, D, nblks, shuffled_indices, stream);
  }
}

};  // end namespace detail
};  // namespace random
}  // namespace raft
