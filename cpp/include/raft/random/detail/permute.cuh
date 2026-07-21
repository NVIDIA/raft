/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/vectorized.cuh>

#include <cooperative_groups.h>

#include <cstdint>
#include <memory>
#include <type_traits>

namespace raft {
namespace random {
namespace detail {

/*
 * Keyed index permutation for permute(), built from a small Feistel network
 * whose round function is the MurmurHash3 32-bit finalizer (fmix32).
 *
 * permute() needs a bijection f: [0, N) -> [0, N) so that mapping every output
 * index to a distinct input index yields a true permutation (no duplicated or
 * dropped rows).
 *
 * Because N is usually not a power of two, we use *cycle walking*: choose the
 * smallest n with 2^n >= N, permute over the 2^n domain, and if the result is
 * >= N, permute again until it lands in [0, N). Restricting a permutation of a
 * finite set to a subset by cycle walking is itself a permutation of that
 * subset, and it always terminates. With a good mixer the expected number of
 * applications is 2^n / N < 2.
 *
 * Reference: https://en.wikipedia.org/wiki/Feistel_cipher
 */

// Simplified version of MurmurHash3 32-bit finalizer.
// Diffuses all input bits into all output bits;
// used as the Feistel round function. Reference:
// https://github.com/aappleby/smhasher/blob/07bb4de10a63e8cc2e1724865454eba635742383/src/MurmurHash3.cpp#L68
HDI uint32_t fmix32(uint32_t h)
{
  // h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  // h *= 0xc2b2ae35u;
  // h ^= h >> 16;
  return h;
}

// Precomputed schedule for the keyed Feistel permutation of [0, N). Every field
// here depends only on (N, key), so it is pre-computed on the host.
//
// The round count is fixed at ROUNDS (4), which is enough for near-ideal
// avalanche at these widths.
struct kperm_params {
  static constexpr int ROUNDS = 4;

  uint64_t N;               // N (domain size); N <= 1 selects the identity permutation
  uint64_t mask_n;          // low-n-bit mask, n = ceil(log2(N))
  uint32_t b_bits;          // width of the low part (shift to extract H = m >> b_bits)
  uint32_t a;               // low-bit mask for the high part: (1 << a_bits) - 1
  uint32_t b;               // low-bit mask for the low part:  (1 << b_bits) - 1
  uint32_t prefix[ROUNDS];  // per-round key-derived mixing constant
};

// Build the Feistel schedule for permuting [0, N) with the given 64-bit key.
// Runs once on the host per permute() call.
HDI kperm_params make_kperm_params(uint64_t N, uint64_t key)
{
  kperm_params p{};
  p.N = N;
  if (N <= 1) return p;  // identity domain; remaining fields go unused

  // n = ceil(log2(N)): smallest n with 2^n >= N (the cycle-walking domain).
  int n         = 0;
  uint64_t pow2 = 1;
  while (pow2 < N && n < 64) {
    pow2 <<= 1;
    ++n;
  }

  // Clamp the width to at least 2 bits so the low half always has >= 1 bit
  // This makes b == 0 impossible, so the device mixer needs no per-round
  // guard against a zero-width low half.
  if (n < 2) n = 2;

  const uint32_t a_bits = uint32_t((n + 1) / 2);  // high part width
  const uint32_t b_bits = uint32_t(n / 2);        // low part width
  p.b_bits              = b_bits;
  p.a                   = (a_bits >= 32) ? ~uint32_t(0) : ((uint32_t(1) << a_bits) - 1);
  p.b      = (b_bits >= 32) ? ~uint32_t(0) : ((uint32_t(1) << b_bits) - 1);  // b_bits >= 1 here
  p.mask_n = (n >= 64) ? ~uint64_t(0) : ((uint64_t(1) << n) - 1);

  // Per-round key schedule: diffuse the 64-bit key once, then derive a distinct
  // prefix per round with a golden-ratio round spread. Identical across threads.
  const uint32_t klo          = uint32_t(key);
  const uint32_t khi          = uint32_t(key >> 32);
  const uint32_t golden_ratio = 0x9e3779b9u;

  const uint32_t kbase = fmix32(fmix32(klo) ^ khi);
  for (int i = 0; i < kperm_params::ROUNDS; i++) {
    p.prefix[i] = fmix32(kbase ^ (uint32_t(i) + golden_ratio));
  }
  return p;
}

// Feistel permutation on m.
// The 4 rounds are unrolled by hand: even rounds mix the low part into the high
// part (a bits), odd rounds mix the high part into the low part (b bits).
// WordType is uint32_t for 32-bit domains and uint64_t for larger ones.
template <typename WordType>
HDI WordType kperm_mix_nbits(WordType m, const kperm_params& p)
{
  m &= WordType(p.mask_n);
  uint32_t L = uint32_t(m & p.b);        // low b_bits
  uint32_t H = uint32_t(m >> p.b_bits);  // high a_bits

  H ^= fmix32(L ^ p.prefix[0]) & p.a;  // round 0 (even): H from L
  L ^= fmix32(H ^ p.prefix[1]) & p.b;  // round 1 (odd):  L from H
  H ^= fmix32(L ^ p.prefix[2]) & p.a;  // round 2 (even): H from L
  L ^= fmix32(H ^ p.prefix[3]) & p.b;  // round 3 (odd):  L from H
  return (WordType(H) << p.b_bits) | WordType(L);
}

// Keyed permutation of [0, N): returns the index that output slot idx pulls
// from, using the precomputed schedule p. A bijection over [0, N) for any key.
//
// idx MUST lie in [0, N) for the cycle walk to halt.
template <typename IdxType>
HDI IdxType kperm_index(IdxType idx, const kperm_params& p)
{
  if (p.N <= 1) return idx;  // 0- or 1-element domain: identity

  // Use a 32-bit word for 32-bit (or narrower) index types to avoid unnecessary
  // 32->64->32 promotion. For larger types, stay in 64 bits.
  using WordType = std::conditional_t<sizeof(IdxType) <= 4, uint32_t, uint64_t>;

  // Cycle walk: re-permute over [0, 2^n) until the result falls in [0, N). When
  // N is a power of two this is a single application (the loop body runs once).
  WordType x = WordType(idx);
  do {
    x = kperm_mix_nbits(x, p);
  } while (x >= WordType(p.N));
  return IdxType(x);
}

template <typename IntType, typename IdxType, int TPB, int ITEMS_PER_THREAD>
RAFT_KERNEL permsOnlyKernel(IntType* perms, kperm_params fp, IdxType N)
{
  IdxType base = IdxType(blockIdx.x) * IdxType(TPB * ITEMS_PER_THREAD) + threadIdx.x;
#pragma unroll
  for (int i = 0; i < ITEMS_PER_THREAD; i++) {
    IdxType idx = base + IdxType(i * TPB);
    if (idx < N) { perms[idx] = IntType(kperm_index<IdxType>(idx, fp)); }
  }
}

template <typename Type, typename IntType, typename IdxType, int TPB, bool rowMajor>
RAFT_KERNEL permuteKernel(
  IntType* perms, Type* out, const Type* in, kperm_params fp, IdxType N, IdxType D)
{
  namespace cg        = cooperative_groups;
  const int WARP_SIZE = 32;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  IntType outIdx = tid;
  IntType inIdx  = (tid < N) ? kperm_index<IntType>(IntType(tid), fp) : IntType(0);

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
  static void permuteImpl(IntType* perms,
                          Type* out,
                          const Type* in,
                          IdxType N,
                          IdxType D,
                          int nblks,
                          kperm_params fp,
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
        <<<nblks, TPB, 0, stream>>>(perms, vout, vin, fp, N, D / VLen);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    } else {  // otherwise try the next lower vector length
      permute_impl_t<Type, IntType, IdxType, TPB, rowMajor, VLen / 2>::permuteImpl(
        perms, out, in, N, D, nblks, fp, stream);
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
                          kperm_params fp,
                          cudaStream_t stream)
  {
    permuteKernel<Type, IntType, IdxType, TPB, rowMajor>
      <<<nblks, TPB, 0, stream>>>(perms, out, in, fp, N, D);
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
             uint64_t key)
{
  if (N <= 0) { return; }

  if (out == nullptr) {
    constexpr int ITEMS_PER_THREAD = 8;
    kperm_params fp                = make_kperm_params(uint64_t(N), key);
    auto nblks                     = raft::ceildiv(N, IntType(TPB * ITEMS_PER_THREAD));
    permsOnlyKernel<IntType, IntType, TPB, ITEMS_PER_THREAD>
      <<<nblks, TPB, 0, stream>>>(perms, fp, N);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    return;
  }

  auto nblks = raft::ceildiv(N, (IntType)TPB);

  // build the keyed Feistel schedule for [0, N) once on the host from the
  // caller-supplied key; the same schedule is passed as an argument
  kperm_params fp = make_kperm_params(uint64_t(N), key);

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
                                                                                 fp,
                                                                                 stream);
  } else {
    permute_impl_t<Type, IntType, IdxType, TPB, false, 1>::permuteImpl(
      perms, out, in, N, D, nblks, fp, stream);
  }
}

};  // end namespace detail
};  // namespace random
}  // namespace raft
