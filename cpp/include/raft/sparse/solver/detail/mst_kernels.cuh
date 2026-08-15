/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>

#include <cstdint>
#include <type_traits>

/*
 * Based on ECL-MST (Fallin, Gonzalez, Seo, Burtscher, SC'23). Per-component
 * minima are selected on the packed key (order_key(weight), edge index):
 * ties break deterministically by edge index.
 */

// __CUDA_ARCH_LIST__ is sorted ascending and visible to all compile passes:
// its first element is the min target arch, so mixed fatbins stay consistent.
#if defined(__CUDA_ARCH_LIST__)
#define RAFT_MST_FIRST_ARCH_(a, ...) a
#define RAFT_MST_FIRST_ARCH(...)     RAFT_MST_FIRST_ARCH_(__VA_ARGS__)
#define RAFT_MST_MIN_ARCH            RAFT_MST_FIRST_ARCH(__CUDA_ARCH_LIST__)
#else
#define RAFT_MST_MIN_ARCH 0  // unknown toolchain: take the portable path
#endif
// RAFT_MST_FORCE_TWOPASS: testing knob, compiles the portable wide path on
// any target. Must be defined consistently across all TUs of a binary (ODR).
#if defined(RAFT_MST_FORCE_TWOPASS)
#define RAFT_MST_HAS_CAS128 0
#elif RAFT_MST_MIN_ARCH >= 900
#define RAFT_MST_HAS_CAS128 1
#else
#define RAFT_MST_HAS_CAS128 0
#endif
#undef RAFT_MST_FIRST_ARCH_
#undef RAFT_MST_FIRST_ARCH
#undef RAFT_MST_MIN_ARCH

namespace raft {
namespace sparse::solver::detail {

using mst_ull = unsigned long long;
#if RAFT_MST_HAS_CAS128
using mst_u128 = unsigned __int128;
#endif

constexpr int mst_block_size = 512;

// order-preserving unsigned reinterpretation (negatives included)
RAFT_DEVICE_INLINE_FUNCTION uint32_t mst_order_key(float w)
{
  uint32_t k = __float_as_uint(w);
  return (k & 0x80000000u) ? ~k : (k | 0x80000000u);
}
RAFT_DEVICE_INLINE_FUNCTION uint32_t mst_order_key(int32_t w)
{
  return static_cast<uint32_t>(w) ^ 0x80000000u;
}
RAFT_DEVICE_INLINE_FUNCTION mst_ull mst_order_key(double w)
{
  mst_ull k = __double_as_longlong(w);
  return (k & 0x8000000000000000ull) ? ~k : (k | 0x8000000000000000ull);
}
RAFT_DEVICE_INLINE_FUNCTION mst_ull mst_order_key(int64_t w)
{
  return static_cast<mst_ull>(w) ^ 0x8000000000000000ull;
}

// Wide worklist entry: 16-byte-aligned {x, y, z, w} of long long.
// CUDA 13 deprecates longlong4 in favor of longlong4_16a
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
using mst_entry64 = longlong4_16a;
#else
using mst_entry64 = longlong4;
#endif
static_assert(sizeof(mst_entry64) == 32 && alignof(mst_entry64) == 16,
              "mst_entry64 layout differs between toolkit branches");

template <typename weight_t, typename edge_t>
struct mst_traits {
  static constexpr bool narrow = (sizeof(weight_t) == 4) && (sizeof(edge_t) == 4);
  using key_t                  = std::conditional_t<narrow, uint32_t, mst_ull>;
  using entry_t                = std::conditional_t<narrow, int4, mst_entry64>;
  using wl_size_t              = std::conditional_t<sizeof(edge_t) == 4, int, long long>;
};

// atomics over possibly-signed types; all values here are non-negative
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION T mst_atomic_cas(T* addr, T compare, T val)
{
  if constexpr (sizeof(T) == 4) {
    return static_cast<T>(atomicCAS(reinterpret_cast<unsigned int*>(addr),
                                    static_cast<unsigned int>(compare),
                                    static_cast<unsigned int>(val)));
  } else {
    return static_cast<T>(atomicCAS(
      reinterpret_cast<mst_ull*>(addr), static_cast<mst_ull>(compare), static_cast<mst_ull>(val)));
  }
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION T mst_atomic_add(T* addr, T val)
{
  if constexpr (sizeof(T) == 4) {
    return static_cast<T>(
      atomicAdd(reinterpret_cast<unsigned int*>(addr), static_cast<unsigned int>(val)));
  } else {
    return static_cast<T>(atomicAdd(reinterpret_cast<mst_ull*>(addr), static_cast<mst_ull>(val)));
  }
}

// Single-instruction word accesses: racing reads can never see a torn value
// (uniform-size races are defined, PTX ISA 8.7.2) and L1 is kept. Do NOT
// replace with atomics: they bypass L1, causing a large perf hit.
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION T mst_word_load(const T* addr)
{
  if constexpr (sizeof(T) == 4) {
    unsigned int r;
    asm volatile("ld.b32 %0, [%1];" : "=r"(r) : "l"(addr) : "memory");
    return static_cast<T>(r);
  } else {
    unsigned long long r;
    asm volatile("ld.b64 %0, [%1];" : "=l"(r) : "l"(addr) : "memory");
    return static_cast<T>(r);
  }
}

template <typename T>
RAFT_DEVICE_INLINE_FUNCTION void mst_word_store(T* addr, T val)
{
  if constexpr (sizeof(T) == 4) {
    asm volatile("st.b32 [%0], %1;" ::"l"(addr), "r"(static_cast<unsigned int>(val)) : "memory");
  } else {
    asm volatile("st.b64 [%0], %1;" ::"l"(addr), "l"(static_cast<unsigned long long>(val))
                 : "memory");
  }
}

// Find with path halving (without it, equal-weight tie chains go quadratic)
template <typename vertex_t>
RAFT_DEVICE_INLINE_FUNCTION vertex_t mst_uf_find(vertex_t curr, vertex_t* const __restrict__ parent)
{
  vertex_t next;
  while (curr != (next = mst_word_load(&parent[curr]))) {
    const vertex_t grand = mst_word_load(&parent[next]);
    if (grand != next) mst_word_store(&parent[curr], grand);
    curr = next;
  }
  return curr;
}

template <typename vertex_t>
RAFT_DEVICE_INLINE_FUNCTION void mst_uf_join(vertex_t arep,
                                             vertex_t brep,
                                             vertex_t* const __restrict__ parent)
{
  vertex_t mrep;
  do {
    mrep = max(arep, brep);
    arep = min(arep, brep);
  } while ((brep = mst_atomic_cas(&parent[mrep], mrep, arep)) != mrep);
}

RAFT_INLINE_FUNCTION unsigned int mst_sample_hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

RAFT_DEVICE_INLINE_FUNCTION long long mst_grid_idx()
{
  return threadIdx.x + static_cast<long long>(blockIdx.x) * mst_block_size;
}

template <typename vertex_t>
RAFT_KERNEL mst_init_parent_kernel(const vertex_t v,
                                   const vertex_t* const __restrict__ color,
                                   vertex_t* const __restrict__ parent)
{
  const long long i = mst_grid_idx();
  if (i < v) parent[i] = color[i];
}

template <typename vertex_t>
RAFT_KERNEL mst_flatten_colors_kernel(const vertex_t v,
                                      vertex_t* const __restrict__ parent,
                                      vertex_t* const __restrict__ color)
{
  const long long i = mst_grid_idx();
  if (i < v) color[i] = mst_uf_find(static_cast<vertex_t>(i), parent);
}

template <typename edge_t, typename weight_t>
RAFT_KERNEL mst_sample_keys_kernel(
  const int n_samples,
  const edge_t e,
  const weight_t* const __restrict__ weights,
  typename mst_traits<weight_t, edge_t>::key_t* const __restrict__ keys)
{
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < n_samples) keys[i] = mst_order_key(weights[mst_sample_hash(i) % e]);
}

// gather flagged CSR edges into COO; optionally emit both directions
template <typename vertex_t, typename edge_t, typename weight_t>
RAFT_KERNEL mst_extract_coo_kernel(const vertex_t v,
                                   const edge_t e,
                                   const edge_t* const __restrict__ offsets,
                                   const vertex_t* const __restrict__ indices,
                                   const weight_t* const __restrict__ weights,
                                   const bool* const __restrict__ in_mst,
                                   const bool symmetrize,
                                   vertex_t* const __restrict__ out_src,
                                   vertex_t* const __restrict__ out_dst,
                                   weight_t* const __restrict__ out_w,
                                   edge_t* const __restrict__ out_count)
{
  const long long j = mst_grid_idx();
  if (j < e && in_mst[j]) {
    // row of edge j by binary search over the offsets
    vertex_t lo = 0, hi = v;
    while (lo + 1 < hi) {
      const vertex_t mid = lo + (hi - lo) / 2;
      if (offsets[mid] <= j) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const edge_t k = mst_atomic_add(out_count, static_cast<edge_t>(symmetrize ? 2 : 1));
    out_src[k]     = lo;
    out_dst[k]     = indices[j];
    out_w[k]       = weights[j];
    if (symmetrize) {
      out_src[k + 1] = indices[j];
      out_dst[k + 1] = lo;
      out_w[k + 1]   = weights[j];
    }
  }
}

template <bool FIRST, typename vertex_t, typename edge_t, typename weight_t>
RAFT_KERNEL mst_init_worklist_kernel(
  typename mst_traits<weight_t, edge_t>::entry_t* const __restrict__ wl,
  typename mst_traits<weight_t, edge_t>::wl_size_t* const __restrict__ wl_size,
  const typename mst_traits<weight_t, edge_t>::wl_size_t wl_capacity,
  const vertex_t v,
  const edge_t e,
  const edge_t* const __restrict__ offsets,
  const vertex_t* const __restrict__ indices,
  const weight_t* const __restrict__ weights,
  vertex_t* const __restrict__ parent,
  const typename mst_traits<weight_t, edge_t>::key_t thr_key)
{
  const long long j = mst_grid_idx();
  if (j < e) {
    const vertex_t n = indices[j];
    // row of edge j by binary search over the offsets
    vertex_t lo = 0, hi = v;
    while (lo + 1 < hi) {
      const vertex_t mid = lo + (hi - lo) / 2;
      if (offsets[mid] <= j) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    const vertex_t r = lo;
    if (n > r) {
      const typename mst_traits<weight_t, edge_t>::key_t k = mst_order_key(weights[j]);
      if (FIRST ? (k <= thr_key) : (k > thr_key)) {
        const vertex_t arep = FIRST ? r : mst_uf_find(r, parent);
        const vertex_t brep = FIRST ? n : mst_uf_find(n, parent);
        if (FIRST || (arep != brep)) {
          using wl_size_t      = typename mst_traits<weight_t, edge_t>::wl_size_t;
          const wl_size_t slot = mst_atomic_add(wl_size, static_cast<wl_size_t>(1));
          // slot >= 0: counter wraparound defense for malformed inputs
          if (slot >= 0 && slot < wl_capacity) {
            if constexpr (mst_traits<weight_t, edge_t>::narrow) {
              wl[slot] = int4{static_cast<int>(arep),
                              static_cast<int>(brep),
                              static_cast<int>(k),
                              static_cast<int>(j)};
            } else {
              wl[slot] = mst_entry64{static_cast<long long>(arep),
                                     static_cast<long long>(brep),
                                     static_cast<long long>(k),
                                     j};
            }
          }
        }
      }
    }
  }
}

// ---- narrow path (4-byte weight_t + edge_t): packed 64-bit (key, index) ----
template <typename vertex_t>
RAFT_KERNEL mst_filter_min_kernel(const int4* const __restrict__ wl1,
                                  const int wl1_size,
                                  int4* const __restrict__ wl2,
                                  int* const __restrict__ wl2_size,
                                  vertex_t* const __restrict__ parent,
                                  volatile mst_ull* const __restrict__ minv,
                                  volatile mst_ull* const __restrict__ minv_prev)
{
  const int idx = threadIdx.x + blockIdx.x * mst_block_size;
  if (idx < wl1_size) {
    int4 el             = wl1[idx];
    const vertex_t arep = mst_uf_find(static_cast<vertex_t>(el.x), parent);
    const vertex_t brep = mst_uf_find(static_cast<vertex_t>(el.y), parent);
    if (arep != brep) {
      minv_prev[arep]             = ~0ull;  // ping-pong reset
      minv_prev[brep]             = ~0ull;
      el.x                        = arep;
      el.y                        = brep;
      wl2[atomicAdd(wl2_size, 1)] = el;
      const mst_ull val =
        ((static_cast<mst_ull>(static_cast<uint32_t>(el.z))) << 32) | static_cast<uint32_t>(el.w);
      if (minv[arep] > val) atomicMin(const_cast<mst_ull*>(&minv[arep]), val);
      if (minv[brep] > val) atomicMin(const_cast<mst_ull*>(&minv[brep]), val);
    }
  }
}

template <typename vertex_t>
RAFT_KERNEL mst_select_join_kernel(const int4* const __restrict__ wl,
                                   const int wl_size,
                                   vertex_t* const __restrict__ parent,
                                   const mst_ull* const __restrict__ minv,
                                   bool* const __restrict__ in_mst)
{
  const int idx = threadIdx.x + blockIdx.x * mst_block_size;
  if (idx < wl_size) {
    const int4 el = wl[idx];
    const mst_ull val =
      ((static_cast<mst_ull>(static_cast<uint32_t>(el.z))) << 32) | static_cast<uint32_t>(el.w);
    if ((val == minv[el.x]) || (val == minv[el.y])) {
      mst_uf_join(static_cast<vertex_t>(el.x), static_cast<vertex_t>(el.y), parent);
      in_mst[el.w] = true;
    }
  }
}

#if RAFT_MST_HAS_CAS128
// ---- wide path, sm_90+: single-pass packed 128-bit (key, edge index) -------
RAFT_DEVICE_INLINE_FUNCTION void mst_atomic_min_u128(mst_u128* const addr, const mst_u128 val)
{
  mst_u128 old = atomicCAS(addr, val, val);
  while (old > val) {
    const mst_u128 assumed = old;
    old                    = atomicCAS(addr, assumed, val);
    if (old == assumed) break;
  }
}

template <typename vertex_t, typename wl_size_t>
RAFT_KERNEL mst_filter_min_kernel(const mst_entry64* const __restrict__ wl1,
                                  const wl_size_t wl1_size,
                                  mst_entry64* const __restrict__ wl2,
                                  wl_size_t* const __restrict__ wl2_size,
                                  vertex_t* const __restrict__ parent,
                                  mst_u128* const __restrict__ minv,
                                  mst_u128* const __restrict__ minv_prev)
{
  const long long idx = mst_grid_idx();
  if (idx < wl1_size) {
    mst_entry64 el      = wl1[idx];
    const vertex_t arep = mst_uf_find(static_cast<vertex_t>(el.x), parent);
    const vertex_t brep = mst_uf_find(static_cast<vertex_t>(el.y), parent);
    if (arep != brep) {
      minv_prev[arep]                                          = ~static_cast<mst_u128>(0);
      minv_prev[brep]                                          = ~static_cast<mst_u128>(0);
      el.x                                                     = arep;
      el.y                                                     = brep;
      wl2[mst_atomic_add(wl2_size, static_cast<wl_size_t>(1))] = el;
      const mst_u128 val =
        ((static_cast<mst_u128>(static_cast<mst_ull>(el.z))) << 64) | static_cast<mst_ull>(el.w);
      const mst_ull key_a = reinterpret_cast<volatile mst_ull*>(&minv[arep])[1];
      if (key_a >= static_cast<mst_ull>(el.z)) mst_atomic_min_u128(&minv[arep], val);
      const mst_ull key_b = reinterpret_cast<volatile mst_ull*>(&minv[brep])[1];
      if (key_b >= static_cast<mst_ull>(el.z)) mst_atomic_min_u128(&minv[brep], val);
    }
  }
}

template <typename vertex_t, typename wl_size_t>
RAFT_KERNEL mst_select_join_kernel(const mst_entry64* const __restrict__ wl,
                                   const wl_size_t wl_size,
                                   vertex_t* const __restrict__ parent,
                                   const mst_u128* const __restrict__ minv,
                                   bool* const __restrict__ in_mst)
{
  const long long idx = mst_grid_idx();
  if (idx < wl_size) {
    const mst_entry64 el = wl[idx];
    const mst_u128 val =
      ((static_cast<mst_u128>(static_cast<mst_ull>(el.z))) << 64) | static_cast<mst_ull>(el.w);
    if ((val == minv[el.x]) || (val == minv[el.y])) {  // no concurrent writers
      mst_uf_join(static_cast<vertex_t>(el.x), static_cast<vertex_t>(el.y), parent);
      in_mst[el.w] = true;
    }
  }
}

#else
// ---- wide path, portable: two-pass min (weight key, then edge index) -------

template <typename vertex_t, typename wl_size_t>
RAFT_KERNEL mst_filter_min_kernel(const mst_entry64* const __restrict__ wl1,
                                  const wl_size_t wl1_size,
                                  mst_entry64* const __restrict__ wl2,
                                  wl_size_t* const __restrict__ wl2_size,
                                  vertex_t* const __restrict__ parent,
                                  volatile mst_ull* const __restrict__ minw,
                                  volatile mst_ull* const __restrict__ minw_prev,
                                  volatile mst_ull* const __restrict__ mine_prev)
{
  const long long idx = mst_grid_idx();
  if (idx < wl1_size) {
    mst_entry64 el      = wl1[idx];
    const vertex_t arep = mst_uf_find(static_cast<vertex_t>(el.x), parent);
    const vertex_t brep = mst_uf_find(static_cast<vertex_t>(el.y), parent);
    if (arep != brep) {
      minw_prev[arep]                                          = ~0ull;  // ping-pong resets
      minw_prev[brep]                                          = ~0ull;
      mine_prev[arep]                                          = ~0ull;
      mine_prev[brep]                                          = ~0ull;
      el.x                                                     = arep;
      el.y                                                     = brep;
      wl2[mst_atomic_add(wl2_size, static_cast<wl_size_t>(1))] = el;
      const mst_ull k                                          = static_cast<mst_ull>(el.z);
      if (minw[arep] > k) atomicMin(const_cast<mst_ull*>(&minw[arep]), k);
      if (minw[brep] > k) atomicMin(const_cast<mst_ull*>(&minw[brep]), k);
    }
  }
}

// pass 2: min edge index among key-tied edges
template <typename wl_size_t, typename entry_t = mst_entry64>
RAFT_KERNEL mst_min_index_kernel(const entry_t* const __restrict__ wl,
                                 const wl_size_t wl_size,
                                 const mst_ull* const __restrict__ minw,
                                 volatile mst_ull* const __restrict__ mine)
{
  const long long idx = mst_grid_idx();
  if (idx < wl_size) {
    const entry_t el = wl[idx];
    const mst_ull k  = static_cast<mst_ull>(el.z);
    const mst_ull id = static_cast<mst_ull>(el.w);
    if (k == minw[el.x] && mine[el.x] > id) atomicMin(const_cast<mst_ull*>(&mine[el.x]), id);
    if (k == minw[el.y] && mine[el.y] > id) atomicMin(const_cast<mst_ull*>(&mine[el.y]), id);
  }
}

template <typename vertex_t, typename wl_size_t>
RAFT_KERNEL mst_select_join_kernel(const mst_entry64* const __restrict__ wl,
                                   const wl_size_t wl_size,
                                   vertex_t* const __restrict__ parent,
                                   const mst_ull* const __restrict__ mine,
                                   bool* const __restrict__ in_mst)
{
  const long long idx = mst_grid_idx();
  if (idx < wl_size) {
    const mst_entry64 el = wl[idx];
    const mst_ull id     = static_cast<mst_ull>(el.w);
    if ((id == mine[el.x]) || (id == mine[el.y])) {  // edge ids globally unique
      mst_uf_join(static_cast<vertex_t>(el.x), static_cast<vertex_t>(el.y), parent);
      in_mst[el.w] = true;
    }
  }
}
#endif  // RAFT_MST_HAS_CAS128

}  // namespace sparse::solver::detail
}  // namespace raft
