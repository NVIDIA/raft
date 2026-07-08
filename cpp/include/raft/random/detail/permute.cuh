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

namespace raft {
namespace random {
namespace detail {

/*
 * Keyed index permutation for permute(), built from a small Feistel network
 * whose round function is the MurmurHash3 32-bit finalizer (fmix32).
 *
 * permute() needs a bijection f: [0, N) -> [0, N) so that mapping every output
 * index to a distinct input index yields a true permutation (no duplicated or
 * dropped rows). The previous affine map out = (a*x + b) mod N is a valid
 * bijection but is *linear*: a single input-bit flip reaches only the higher
 * output bits (poor avalanche), so nearby indices map to structured, correlated
 * destinations. The Feistel construction below is a non-linear bijection with
 * near-ideal strict-avalanche behavior, so the permutation looks random while
 * remaining exactly collision-free.
 *
 * A balanced Feistel network is a permutation over n-bit words for *any* round
 * function -- invertibility does not depend on the round function being a
 * bijection, which is why an fmix32-keyed mixer is a valid round function. We
 * run 4 rounds (empirically enough for near-ideal avalanche at these widths).
 *
 * Because N is usually not a power of two, we use *cycle walking*: choose the
 * smallest n with 2^n >= N, permute over the 2^n domain, and if the result is
 * >= N, permute again until it lands in [0, N). Restricting a permutation of a
 * finite set to a subset by cycle walking is itself a permutation of that
 * subset, and it always terminates. With a good mixer the expected number of
 * applications is 2^n / N < 2.
 */

// MurmurHash3 32-bit finalizer. Diffuses all input bits into all output bits;
// used as the (non-invertible, but that is fine) Feistel round function.
HDI uint32_t feistel_fmix32(uint32_t h)
{
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return h;
}

// Precomputed schedule for the keyed Feistel permutation of [0, N). Every field
// here depends only on (N, key) and not on the index being permuted, so it is
// derived once on the host and passed by value to every thread. This avoids
// re-deriving the key schedule and masks in each thread -- and on each
// cycle-walk retry -- which would otherwise be identical redundant work.
//
// The round count is fixed at ROUNDS (4), which is enough for near-ideal
// avalanche at these widths. Keeping it a compile-time constant lets the round
// loop in feistel_mix_nbits unroll fully with the even/odd branch folded out.
struct feistel_permute_params {
  static constexpr int ROUNDS = 4;

  uint64_t U;               // N (domain size); U <= 1 selects the identity permutation
  uint64_t mask_n;          // low-n-bit mask, n = ceil(log2(N)) (the cycle-walk domain)
  uint32_t mask_b;          // low-half mask (b bits); high half needs no mask (see below)
  uint32_t a;               // high part width, ceil(n/2), 1..32
  uint32_t b;               // low  part width, floor(n/2), 1..32 (n is clamped to >= 2)
  uint32_t prefix[ROUNDS];  // per-round key-derived mixing constant
};

// Build the Feistel schedule for permuting [0, N) with the given 64-bit key.
// Runs once on the host per permute() call.
inline feistel_permute_params make_feistel_permute_params(uint64_t N, uint64_t key)
{
  feistel_permute_params p{};
  p.U = N;
  if (N <= 1) return p;  // identity domain; remaining fields go unused

  // n = ceil(log2(N)): smallest n with 2^n >= N (the cycle-walking domain).
  int n         = 0;
  uint64_t pow2 = 1;
  while (pow2 < N) {
    pow2 <<= 1;
    ++n;
  }

  // Clamp the width to at least 2 bits so the low half always has >= 1 bit
  // (b = floor(n/2) >= 1). Only N == 2 is affected (n = 1 -> 2): it now permutes
  // over the 4-element domain [0, 4) and cycle-walks the two dead values, still a
  // valid, terminating permutation of {0, 1}. This makes b == 0 impossible, so
  // the device mixer needs no per-round guard against a zero-width low half.
  if (n < 2) n = 2;

  const uint32_t a = uint32_t((n + 1) / 2);  // high part width, 1..32
  const uint32_t b = uint32_t(n / 2);        // low  part width, 1..32
  p.a              = a;
  p.b              = b;
  p.mask_n         = (n >= 64) ? ~uint64_t(0) : ((uint64_t(1) << n) - 1);
  p.mask_b         = (b >= 32) ? ~uint32_t(0) : ((uint32_t(1) << b) - 1);  // b >= 1 here

  // Per-round key schedule: diffuse the 64-bit key once, then derive a distinct
  // prefix per round with a golden-ratio round spread. Identical across threads.
  const uint32_t klo   = uint32_t(key);
  const uint32_t khi   = uint32_t(key >> 32);
  const uint32_t kbase = feistel_fmix32(feistel_fmix32(klo) ^ khi);
  for (int i = 0; i < feistel_permute_params::ROUNDS; i++) {
    p.prefix[i] = feistel_fmix32(kbase ^ (uint32_t(i) + 0x9e3779b9u));
  }
  return p;
}

// One balanced Feistel permutation over the low n bits of m, using the schedule
// precomputed in p. The n bits split into a high part of ceil(n/2) bits and a
// low part of floor(n/2) bits (each <= 32, so each half fits fmix32). Each round
// XORs an fmix32 mix of one half, keyed per round, into the other half,
// alternating which half is updated. Every step is invertible, so the whole map
// is a bijection over [0, 2^n).
//
// The 4 rounds are unrolled by hand: even rounds mix the low part into the high
// part (a bits), odd rounds mix the high part into the low part (b bits). Since
// n is clamped to >= 2, both halves always have >= 1 bit, so every round is
// unconditional -- no branch and no loop overhead.
HDI uint64_t feistel_mix_nbits(uint64_t m, const feistel_permute_params& p)
{
  m &= p.mask_n;
  uint32_t L = uint32_t(m & p.mask_b);  // low  floor(n/2) bits
  uint32_t H = uint32_t(m >> p.b);      // high ceil(n/2) bits (n - b = a bits, already masked)

  // Each round's mix is fmix32(...) >> (32 - width), so it already has <= a (or
  // <= b) bits; XORing into a same-width half keeps it within that width, so no
  // per-round masking is needed. H stays a bits and L stays b bits throughout.
  H ^= feistel_fmix32(L ^ p.prefix[0]) >> (32 - p.a);  // round 0 (even): H from L
  L ^= feistel_fmix32(H ^ p.prefix[1]) >> (32 - p.b);  // round 1 (odd):  L from H
  H ^= feistel_fmix32(L ^ p.prefix[2]) >> (32 - p.a);  // round 2 (even): H from L
  L ^= feistel_fmix32(H ^ p.prefix[3]) >> (32 - p.b);  // round 3 (odd):  L from H
  return (uint64_t(H) << p.b) | uint64_t(L);
}

// Keyed permutation of [0, N): returns the index that output slot idx pulls
// from, using the precomputed schedule p. A bijection over [0, N) for any key.
//
// idx MUST lie in [0, N). Cycle walking is guaranteed to terminate only when it
// starts inside the target set: an in-range idx sits on an orbit that returns to
// it (in range), so the walk always halts. An out-of-range idx can land on a
// cycle contained entirely in the dead zone [N, 2^n) and loop forever, so callers
// must not pass idx >= N (e.g. gate padding threads whose tid >= N).
template <typename IdxType>
HDI IdxType feistel_permute_index(IdxType idx, const feistel_permute_params& p)
{
  if (p.U <= 1) return idx;  // 0- or 1-element domain: identity

  // Cycle walk: re-permute over [0, 2^n) until the result falls in [0, N). When
  // N is a power of two this is a single application (the loop body runs once).
  uint64_t x = uint64_t(idx);
  do {
    x = feistel_mix_nbits(x, p);
  } while (x >= p.U);
  return IdxType(x);
}

template <typename Type, typename IntType, typename IdxType, int TPB, bool rowMajor>
RAFT_KERNEL permuteKernel(
  IntType* perms, Type* out, const Type* in, feistel_permute_params fp, IdxType N, IdxType D)
{
  namespace cg        = cooperative_groups;
  const int WARP_SIZE = 32;

  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  // having shuffled input indices and coalesced output indices appears
  // to be preferable to the reverse, especially for column major.
  // Only in-range slots get a permuted source index: feistel_permute_index
  // requires idx in [0, N) to guarantee the cycle walk terminates, and padding
  // threads (tid >= N) never have their inIdx read (their row is skipped below).
  IntType outIdx = tid;
  IntType inIdx  = (tid < N) ? feistel_permute_index<IntType>(IntType(tid), fp) : IntType(0);

  if (perms != nullptr && tid < N) { perms[outIdx] = inIdx; }

  if (out == nullptr || in == nullptr) { return; }

  if (rowMajor) {
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(cg::this_thread_block());

    // The warp cooperatively copies its 32 rows one at a time so that the 32
    // lanes stride together along D, giving coalesced global loads and stores.
    // Copying row i needs lane i's (inIdx, outIdx); that exchange is purely
    // intra-warp, so a warp shuffle broadcasts it register-to-register -- no
    // shared memory or block barrier required.
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
                          feistel_permute_params fp,
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
                          feistel_permute_params fp,
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
  auto nblks = raft::ceildiv(N, (IntType)TPB);

  // build the keyed Feistel schedule for [0, N) once on the host from the
  // caller-supplied key; the same schedule is broadcast (by value) to every
  // thread
  feistel_permute_params fp = make_feistel_permute_params(uint64_t(N), key);

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
