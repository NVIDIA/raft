/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/vectorized.cuh>

#include <cooperative_groups.h>

#include <memory>
#include <random>

namespace raft {
namespace random {
namespace detail {

template <typename Type, typename IntType, typename IdxType, int TPB, bool rowMajor>
RAFT_KERNEL permuteKernel(
  IntType* perms, Type* out, const Type* in, IdxType a, IdxType b, IdxType N, IdxType D)
{
  namespace cg        = cooperative_groups;
  const int WARP_SIZE = 32;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // having shuffled input indices and coalesced output indices appears
  // to be preferable to the reverse, especially for column major
  IntType inIdx  = ((a * int64_t(tid)) + b) % N;
  IntType outIdx = tid;

  if (perms != nullptr && tid < N) { perms[outIdx] = inIdx; }

  if (out == nullptr || in == nullptr) { return; }

  if (rowMajor) {
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

    __shared__ IntType inIdxShm[TPB];
    __shared__ IntType outIdxShm[TPB];
    inIdxShm[threadIdx.x]  = inIdx;
    outIdxShm[threadIdx.x] = outIdx;
    warp.sync();

    int warpID = threadIdx.x / WARP_SIZE;
    int laneID = threadIdx.x % WARP_SIZE;
    for (int i = warpID * WARP_SIZE; i < warpID * WARP_SIZE + WARP_SIZE; ++i) {
      if (outIdxShm[i] < N) {
#pragma unroll
        for (int j = laneID; j < D; j += WARP_SIZE) {
          out[outIdxShm[i] * D + j] = in[inIdxShm[i] * D + j];
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
  static void permuteImpl(IntType* perms,
                          Type* out,
                          const Type* in,
                          IdxType N,
                          IdxType D,
                          int nblks,
                          IdxType a,
                          IdxType b,
                          cudaStream_t stream)
  {
    // determine vector type and set new pointers
    typedef typename raft::IOType<Type, VLen>::Type VType;
    VType* vout      = reinterpret_cast<VType*>(out);
    const VType* vin = reinterpret_cast<const VType*>(in);

    // check if we can execute at this vector length
    if (D % VLen == 0 && raft::is_aligned(vout, sizeof(VType)) &&
        raft::is_aligned(vin, sizeof(VType))) {
      permuteKernel<VType, IntType, IdxType, TPB, rowMajor>
        <<<nblks, TPB, 0, stream>>>(perms, vout, vin, a, b, N, D / VLen);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {  // otherwise try the next lower vector length
      permute_impl_t<Type, IntType, IdxType, TPB, rowMajor, VLen / 2>::permuteImpl(
        perms, out, in, N, D, nblks, a, b, stream);
    }
  }
};

// at vector length 1 we just execute a scalar version to break the recursion
template <typename Type, typename IntType, typename IdxType, int TPB, bool rowMajor>
struct permute_impl_t<Type, IntType, IdxType, TPB, rowMajor, 1> {
  static void permuteImpl(IntType* perms,
                          Type* out,
                          const Type* in,
                          IdxType N,
                          IdxType D,
                          int nblks,
                          IdxType a,
                          IdxType b,
                          cudaStream_t stream)
  {
    permuteKernel<Type, IntType, IdxType, TPB, rowMajor>
      <<<nblks, TPB, 0, stream>>>(perms, out, in, a, b, N, D);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
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
             uint64_t seed)
{
  auto nblks = raft::ceildiv(N, (IntType)TPB);

  // gen seeded once, deterministically, from seed param
  std::mt19937_64 gen(seed);
  // maps gen's raw bit stream to a uniform integer
  std::uniform_int_distribution<IdxType> dist(0, N - 1);

  // a must be coprime to N, otherwise (a * tid) % N does not hit every row exactly once
  IdxType a = dist(gen);
  while (raft::gcd(a, N) != 1)
    a = (a + 1) % N;
  // additive offset
  IdxType b = dist(gen);

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
                                                                                 a,
                                                                                 b,
                                                                                 stream);
  } else {
    permute_impl_t<Type, IntType, IdxType, TPB, false, 1>::permuteImpl(
      perms, out, in, N, D, nblks, a, b, stream);
  }
}

};  // end namespace detail
};  // namespace random
}  // namespace raft
