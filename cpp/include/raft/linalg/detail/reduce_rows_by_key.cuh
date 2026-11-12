/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>
#include <raft/util/vectorized.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>

#define MAX_BLOCKS 65535u
namespace raft {
namespace linalg {
namespace detail {

//
// Small helper function to convert from int->char and char->int
// Transform ncols*nrows read of int in 2*nrows reads of int + ncols*rows reads of chars
//

template <typename IteratorT1, typename IteratorT2>
RAFT_KERNEL convert_array_kernel(IteratorT1 dst, IteratorT2 src, int n)
{
  for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
    dst[idx] = src[idx];
  }
}

//
// Small helper function to convert from int->char and char->int
// Transform ncols*nrows read of int in 2*nrows reads of int + ncols*rows reads of chars
//

template <typename IteratorT1, typename IteratorT2>
void convert_array(IteratorT1 dst, IteratorT2 src, int n, cudaStream_t st)
{
  dim3 grid, block;
  block.x = 256;

  grid.x = raft::ceildiv(n, (int)block.x);
  grid.x = std::min(grid.x, MAX_BLOCKS);

  raft::launch_kernel(st, grid, block, convert_array_kernel, dst, src, n);
}

/** Floating point including the extended types, which std::is_floating_point does not cover. */
template <typename T>
inline constexpr bool is_floating_point_v =
  std::is_floating_point_v<T> || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;

//
// Deterministic reduction by key.
//
// Every path gives the same bits on every run and on every GPU: no floating-point atomic has more
// than one writer, and every work split and summation order depends only on (nrows, ncols, nkeys).
//

/** Accumulates a * w + acc with one rounding. nvcc contracts `acc += a * w` differently per
 * architecture, which changes the bits between GPUs, so the fused form is explicit. */
template <typename T>
DI T rrbk_fma(T a, T w, T acc)
{
  if constexpr (std::is_same_v<T, float>) {
    return __fmaf_rn(a, w, acc);
  } else if constexpr (std::is_same_v<T, double>) {
    return __fma_rn(a, w, acc);
  } else {
    return a * w + acc;
  }
}

template <typename SumsT, typename WeightT, typename IdxT>
DI SumsT rrbk_weight(const WeightT* d_weights, IdxT row)
{
  return d_weights != nullptr ? static_cast<SumsT>(d_weights[row]) : SumsT(1);
}

// ---------------------------------------------------------------- privatized (few keys)

constexpr std::size_t kPrivSmemBytes = 48 * 1024;
constexpr int kPrivMaxThreads        = 512;
constexpr int kPrivMinThreads        = 96;
constexpr int kPrivMaxParts          = 256;

/**
 * Few keys: each block owns a fixed range of rows and one tile of columns. It holds n_copies copies
 * of the (nkeys x tile_cols) sums in shared memory; each thread owns one column of one copy and
 * adds its rows in row order, so no atomics are needed. The copies are added in copy order and
 * written to the block's slot of `partials`.
 */
template <int kRowsAhead,
          typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
RAFT_KERNEL __launch_bounds__(kPrivMaxThreads) rrbk_priv_kernel(DataIteratorT d_A,
                                                                IdxT lda,
                                                                const WeightT* d_weights,
                                                                KeysIteratorT d_keys,
                                                                IdxT nrows,
                                                                IdxT ncols,
                                                                IdxT nkeys,
                                                                int tile_cols,
                                                                IdxT rows_per_part,
                                                                SumsT* partials)
{
  extern __shared__ __align__(16) unsigned char rrbk_priv_smem[];
  SumsT* priv        = reinterpret_cast<SumsT*>(rrbk_priv_smem);
  const int n_copies = blockDim.x / tile_cols;
  const int c        = threadIdx.x % tile_cols;
  const int p        = threadIdx.x / tile_cols;
  const IdxT col     = static_cast<IdxT>(blockIdx.y) * tile_cols + c;
  const IdxT r0      = static_cast<IdxT>(blockIdx.x) * rows_per_part;
  const IdxT r1      = std::min<IdxT>(r0 + rows_per_part, nrows);
  const IdxT n_copy  = nkeys * tile_cols;
  for (IdxT i = threadIdx.x; i < n_copies * n_copy; i += blockDim.x) {
    priv[i] = SumsT(0);
  }
  __syncthreads();
  if (col < ncols) {
    SumsT* mine = priv + static_cast<IdxT>(p) * n_copy + c;
    for (IdxT r = r0 + p; r < r1; r += static_cast<IdxT>(kRowsAhead) * n_copies) {
      SumsT a[kRowsAhead], w[kRowsAhead];
      IdxT key[kRowsAhead];
#pragma unroll
      for (int u = 0; u < kRowsAhead; ++u) {
        const IdxT row   = r + static_cast<IdxT>(u) * n_copies;
        const bool valid = row < r1;
        a[u]             = valid ? static_cast<SumsT>(d_A[lda * row + col]) : SumsT(0);
        w[u]             = valid ? rrbk_weight<SumsT>(d_weights, row) : SumsT(0);
        key[u]           = valid ? static_cast<IdxT>(d_keys[row]) : IdxT(0);
      }
#pragma unroll
      for (int u = 0; u < kRowsAhead; ++u) {
        mine[key[u] * tile_cols] = rrbk_fma(a[u], w[u], mine[key[u] * tile_cols]);
      }
    }
  }
  __syncthreads();
  SumsT* out = partials + static_cast<IdxT>(blockIdx.x) * nkeys * ncols;
  for (IdxT i = threadIdx.x; i < n_copy; i += blockDim.x) {
    const IdxT cc = static_cast<IdxT>(blockIdx.y) * tile_cols + i % tile_cols;
    if (cc >= ncols) { continue; }
    SumsT t = SumsT(0);
    for (int q = 0; q < n_copies; ++q) {
      t += priv[static_cast<IdxT>(q) * n_copy + i];
    }
    out[(i / tile_cols) * ncols + cc] = t;
  }
}

/** Adds the n_parts partials of each output in a fixed order: thread row s adds partials s, s + 8,
 * ..., then the 8 rows are added in order. */
template <typename SumsT, typename IdxT>
RAFT_KERNEL rrbk_ordered_partials_kernel(const SumsT* partials,
                                         IdxT n_parts,
                                         IdxT n_out,
                                         SumsT* d_sums)
{
  constexpr int kSplit = 8;
  __shared__ SumsT part[kSplit][32];
  const IdxT i = static_cast<IdxT>(blockIdx.x) * 32 + threadIdx.x;
  SumsT t      = SumsT(0);
  if (i < n_out) {
    for (IdxT b = threadIdx.y; b < n_parts; b += kSplit) {
      t += partials[b * n_out + i];
    }
  }
  part[threadIdx.y][threadIdx.x] = t;
  __syncthreads();
  if (threadIdx.y == 0 && i < n_out) {
    SumsT s = d_sums[i];
#pragma unroll
    for (int q = 0; q < kSplit; ++q) {
      s += part[q][threadIdx.x];
    }
    d_sums[i] = s;
  }
}

struct rrbk_priv_plan {
  int tile_cols         = 0;
  int n_copies          = 0;
  int64_t n_parts       = 0;
  int64_t rows_per_part = 0;
  bool ok               = false;
};

template <typename SumsT, typename IdxT>
rrbk_priv_plan rrbk_make_priv_plan(IdxT nrows, IdxT ncols, IdxT nkeys)
{
  rrbk_priv_plan pl{};
  if (nrows <= 0 || ncols <= 0 || nkeys <= 0) { return pl; }
  const IdxT max_cols =
    std::min<IdxT>({ncols, IdxT{128}, static_cast<IdxT>(kPrivSmemBytes / (sizeof(SumsT) * nkeys))});
  if (max_cols < 1) { return pl; }
  // Equal-width column tiles: a ragged last tile would leave most of its block idle.
  pl.tile_cols = static_cast<int>(ceildiv<IdxT>(ncols, ceildiv<IdxT>(ncols, max_cols)));
  pl.n_copies  = static_cast<int>(std::min<std::size_t>(
    kPrivMaxThreads / pl.tile_cols, kPrivSmemBytes / (sizeof(SumsT) * nkeys * pl.tile_cols)));
  // Few threads per block starve the loads; the sorted paths are faster there.
  pl.ok =
    pl.n_copies >= 2 && pl.n_copies * pl.tile_cols >= std::min<IdxT>(kPrivMinThreads, ncols * 16);
  // Partial traffic (n_parts * nkeys * ncols) stays below ~1/16 of the input.
  pl.n_parts       = std::clamp<int64_t>(nrows / (16 * nkeys), 1, kPrivMaxParts);
  pl.rows_per_part = ceildiv<int64_t>(nrows, pl.n_parts);
  pl.n_parts       = ceildiv<int64_t>(nrows, pl.rows_per_part);
  return pl;
}

template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void rrbk_priv(DataIteratorT d_A,
               IdxT lda,
               KeysIteratorT d_keys,
               const WeightT* d_weights,
               IdxT nrows,
               IdxT ncols,
               IdxT nkeys,
               SumsT* d_sums,
               cudaStream_t st,
               bool dry_run)
{
  const auto pl    = rrbk_make_priv_plan<SumsT>(nrows, ncols, nkeys);
  const IdxT n_out = nkeys * ncols;
  rmm::device_uvector<SumsT> partials(static_cast<std::size_t>(pl.n_parts) * n_out, st);
  const dim3 grid(static_cast<uint32_t>(pl.n_parts),
                  static_cast<uint32_t>(ceildiv<IdxT>(ncols, pl.tile_cols)));
  const std::size_t smem = sizeof(SumsT) * nkeys * pl.tile_cols * pl.n_copies;
  raft::launch_kernel({st, smem, dry_run},
                      grid,
                      dim3(pl.n_copies * pl.tile_cols),
                      rrbk_priv_kernel<8, DataIteratorT, KeysIteratorT, WeightT, SumsT, IdxT>,
                      d_A,
                      lda,
                      d_weights,
                      d_keys,
                      nrows,
                      ncols,
                      nkeys,
                      pl.tile_cols,
                      static_cast<IdxT>(pl.rows_per_part),
                      partials.data());
  raft::launch_kernel({st, 0, dry_run},
                      dim3(static_cast<uint32_t>(ceildiv<IdxT>(n_out, 32))),
                      dim3(32, 8),
                      rrbk_ordered_partials_kernel<SumsT, IdxT>,
                      static_cast<const SumsT*>(partials.data()),
                      static_cast<IdxT>(pl.n_parts),
                      n_out,
                      d_sums);
}

// ---------------------------------------------------------------- exact integer bins (float)

constexpr int kBinsBlock       = 256;
constexpr int64_t kBinsMaxGrid = 8192;
/** Upper bound on the integer accumulator, in bytes, for the exact-bin path. */
constexpr int64_t kBinsMaxBytes = int64_t{16} << 20;

struct rrbk_bins_plan {
  int bw    = 0;  // bits per bin
  int off   = 0;  // shift of the bin grid in units of 2^-149
  int nbins = 0;  // int64 words per output
};

/**
 * Bin layout for nrows rows. A finite float x is m * 2^(l - off - 149) with an integer mantissa
 * m < 2^24 and l in [0, 254 + off). Bin b holds units of 2^(b * bw - off - 149). One product adds
 * less than 2^bw to each of at most two bins, so with bw <= 62 - ceil(log2(nrows)) no bin can
 * overflow an int64.
 */
template <typename IdxT>
rrbk_bins_plan rrbk_make_bins_plan(IdxT nrows)
{
  int log2n = 0;
  while ((IdxT{1} << log2n) < nrows) {
    ++log2n;
  }
  rrbk_bins_plan pl{};
  pl.bw = std::min(40, 62 - log2n);
  // Bin edge at leading exponent -9, so products of typical magnitude take one atomic.
  pl.off   = ((117 / pl.bw + 1) * pl.bw - 117) % pl.bw;
  pl.nbins = (253 + pl.off) / pl.bw + 2;
  return pl;
}

DI double rrbk_pow2(int k) { return __longlong_as_double(static_cast<long long>(k + 1023) << 52); }

/**
 * Adds every rounded product x = a * w as an exact integer: its mantissa shifted to its exponent
 * and split across at most two int64 bins. Integer addition is associative, so the order of the
 * atomics does not change the result.
 */
template <typename DataIteratorT, typename KeysIteratorT, typename WeightT, typename IdxT>
RAFT_KERNEL __launch_bounds__(kBinsBlock) rrbk_bins_add_kernel(DataIteratorT d_A,
                                                               IdxT lda,
                                                               const WeightT* d_weights,
                                                               KeysIteratorT d_keys,
                                                               IdxT nrows,
                                                               IdxT ncols,
                                                               int lc_log2,
                                                               rrbk_bins_plan pl,
                                                               unsigned long long* acc)
{
  const int lc      = 1 << lc_log2;
  const IdxT groups = static_cast<IdxT>(gridDim.x) * (blockDim.x >> lc_log2);
  const IdxT g0 =
    static_cast<IdxT>(blockIdx.x) * (blockDim.x >> lc_log2) + (threadIdx.x >> lc_log2);
  const int c0                     = threadIdx.x & (lc - 1);
  const unsigned long long lo_mask = (1ull << pl.bw) - 1;
  for (IdxT i = g0; i < nrows; i += groups) {
    const IdxT key           = static_cast<IdxT>(d_keys[i]);
    const float w            = rrbk_weight<float>(d_weights, i);
    unsigned long long* base = acc + key * pl.nbins * ncols;
    for (IdxT c = c0; c < ncols; c += lc) {
      const unsigned u = __float_as_uint(__fmul_rn(static_cast<float>(d_A[i * lda + c]), w));
      const unsigned e = (u >> 23) & 0xffu;
      const unsigned long long m = (u & 0x7fffffu) | (e != 0 ? 0x800000u : 0u);
      if (m == 0) { continue; }
      const int l                 = static_cast<int>(e != 0 ? e : 1u) - 1 + pl.off;
      const int b                 = l / pl.bw;
      const unsigned long long v  = m << (l - b * pl.bw);
      const unsigned long long lo = v & lo_mask;
      const unsigned long long hi = v >> pl.bw;
      const bool neg              = (u >> 31) != 0;
      if (lo != 0) { atomicAdd(base + b * ncols + c, neg ? 0ull - lo : lo); }
      if (hi != 0) { atomicAdd(base + (b + 1) * ncols + c, neg ? 0ull - hi : hi); }
    }
  }
}

/** Normalizes each output's bins with carries and converts the integer sum to float once. */
template <typename IdxT>
RAFT_KERNEL rrbk_bins_finish_kernel(
  const unsigned long long* acc, IdxT nkeys, IdxT ncols, rrbk_bins_plan pl, float* d_sums)
{
  constexpr int kMaxBins = 16;
  const IdxT i           = static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= nkeys * ncols) { return; }
  const IdxT key              = i / ncols;
  const unsigned long long* p = acc + key * pl.nbins * ncols + (i - key * ncols);
  long long carry             = 0;
  long long bins[kMaxBins];
  for (int b = 0; b < pl.nbins; ++b) {
    const long long t = static_cast<long long>(p[b * ncols]) + carry;
    carry             = t >> pl.bw;
    bins[b]           = t - (carry << pl.bw);
  }
  double s = static_cast<double>(carry) * rrbk_pow2(pl.nbins * pl.bw - pl.off - 149);
  for (int b = pl.nbins - 1; b >= 0; --b) {
    s += static_cast<double>(bins[b]) * rrbk_pow2(b * pl.bw - pl.off - 149);
  }
  d_sums[i] += static_cast<float>(s);
}

template <typename DataIteratorT, typename KeysIteratorT, typename WeightT, typename IdxT>
void rrbk_bins(DataIteratorT d_A,
               IdxT lda,
               KeysIteratorT d_keys,
               const WeightT* d_weights,
               IdxT nrows,
               IdxT ncols,
               IdxT nkeys,
               float* d_sums,
               cudaStream_t st,
               bool dry_run)
{
  const auto pl    = rrbk_make_bins_plan(nrows);
  const IdxT n_acc = nkeys * pl.nbins * ncols;
  rmm::device_uvector<unsigned long long> acc(n_acc, st);
  if (!dry_run) {
    RAFT_CUDA_TRY(cudaMemsetAsync(acc.data(), 0, sizeof(unsigned long long) * n_acc, st));
  }
  int lc_log2 = 0;
  while ((IdxT{1} << lc_log2) < std::min<IdxT>(ncols, 32)) {
    ++lc_log2;
  }
  const auto grid = static_cast<uint32_t>(
    std::min<IdxT>(ceildiv<IdxT>(nrows, kBinsBlock >> lc_log2), kBinsMaxGrid));
  raft::launch_kernel({st, 0, dry_run},
                      dim3(grid),
                      dim3(kBinsBlock),
                      rrbk_bins_add_kernel<DataIteratorT, KeysIteratorT, WeightT, IdxT>,
                      d_A,
                      lda,
                      d_weights,
                      d_keys,
                      nrows,
                      ncols,
                      lc_log2,
                      pl,
                      acc.data());
  raft::launch_kernel({st, 0, dry_run},
                      dim3(static_cast<uint32_t>(ceildiv<IdxT>(nkeys * ncols, 256))),
                      dim3(256),
                      rrbk_bins_finish_kernel<IdxT>,
                      static_cast<const unsigned long long*>(acc.data()),
                      nkeys,
                      ncols,
                      pl,
                      d_sums);
}

// ---------------------------------------------------------------- stable sort, per-key reduce

constexpr int64_t kSegChunkRows = 256;
constexpr int kSegAhead         = 4;
constexpr int kSegBlock         = 256;
/** The per-key path is used below this many rows per key. */
constexpr int64_t kSegMaxRowsPerKey = 56;
constexpr int64_t kSegMaxCols       = 128;
constexpr int64_t kSegMaxKeys       = 100000;

/** Number of bits needed to represent every key in [0, nkeys). */
template <typename IdxT>
int rrbk_key_bits(IdxT nkeys)
{
  int end_bit = 1;
  while (end_bit < 32 && (IdxT{1} << end_bit) < nkeys) {
    ++end_bit;
  }
  return end_bit;
}

template <typename KeysIteratorT, typename IdxT>
RAFT_KERNEL rrbk_sort_input_kernel(KeysIteratorT d_keys,
                                   IdxT nrows,
                                   uint32_t* keys,
                                   uint32_t* row_ids)
{
  const IdxT i = static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < nrows) {
    keys[i]    = static_cast<uint32_t>(d_keys[i]);
    row_ids[i] = static_cast<uint32_t>(i);
  }
}

/** Dense segment offsets of the sorted keys: offsets[k] is the first row with key >= k. */
template <typename IdxT>
RAFT_KERNEL rrbk_boundaries_kernel(const uint32_t* keys, IdxT nrows, IdxT nkeys, IdxT* offsets)
{
  const IdxT r = static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (r >= nrows) { return; }
  const IdxT key  = keys[r];
  const IdxT prev = r == 0 ? IdxT{-1} : static_cast<IdxT>(keys[r - 1]);
  for (IdxT k = prev + 1; k <= key; ++k) {
    offsets[k] = r;
  }
  if (r == nrows - 1) {
    for (IdxT k = key + 1; k <= nkeys; ++k) {
      offsets[k] = nrows;
    }
  }
}

/** Number of 256-row chunks of each key longer than one chunk (0 otherwise). */
template <typename IdxT>
RAFT_KERNEL rrbk_chunk_counts_kernel(const IdxT* offsets, IdxT nkeys, IdxT* counts)
{
  const IdxT k = static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k > nkeys) { return; }
  const IdxT len = k < nkeys ? offsets[k + 1] - offsets[k] : 0;
  counts[k]      = len > kSegChunkRows ? ceildiv<IdxT>(len, kSegChunkRows) : 0;
}

template <int V>
struct rrbk_vec;
template <>
struct rrbk_vec<1> {
  using type = float;
};
template <>
struct rrbk_vec<4> {
  using type = float4;
};

/**
 * Sums sorted rows [begin, end) for S-lane column vectors cv = lane + j * S, strictly in sorted row
 * order. Row group g of G takes rows begin + g, begin + g + G, ...
 */
template <int V, int NV, int S, typename WeightT, typename IdxT>
DI void rrbk_seg_sum(const float* A,
                     IdxT lda,
                     const WeightT* w,
                     const uint32_t* rows,
                     IdxT begin,
                     IdxT end,
                     int nvec,
                     int lane,
                     float (&acc)[NV][V],
                     int g = 0,
                     int G = 1)
{
  using T = typename rrbk_vec<V>::type;
#pragma unroll
  for (int j = 0; j < NV; ++j) {
#pragma unroll
    for (int i = 0; i < V; ++i) {
      acc[j][i] = 0.0f;
    }
  }
  auto add = [&](float (&dst)[V], T a, float wr) {
    const float* p = reinterpret_cast<const float*>(&a);
#pragma unroll
    for (int i = 0; i < V; ++i) {
      dst[i] = __fmaf_rn(p[i], wr, dst[i]);
    }
  };
  IdxT p = begin + g;
  for (; p + (kSegAhead - 1) * G < end; p += kSegAhead * G) {
    uint32_t r[kSegAhead];
    float wr[kSegAhead];
    T a[kSegAhead][NV];
#pragma unroll
    for (int u = 0; u < kSegAhead; ++u) {
      r[u] = rows[p + u * G];
    }
#pragma unroll
    for (int u = 0; u < kSegAhead; ++u) {
      wr[u]        = rrbk_weight<float>(w, static_cast<IdxT>(r[u]));
      const T* row = reinterpret_cast<const T*>(A + static_cast<IdxT>(r[u]) * lda);
#pragma unroll
      for (int j = 0; j < NV; ++j) {
        if (lane + j * S < nvec) { a[u][j] = row[lane + j * S]; }
      }
    }
#pragma unroll
    for (int u = 0; u < kSegAhead; ++u) {
#pragma unroll
      for (int j = 0; j < NV; ++j) {
        if (lane + j * S < nvec) { add(acc[j], a[u][j], wr[u]); }
      }
    }
  }
  for (; p < end; p += G) {
    const uint32_t r = rows[p];
    const float wr   = rrbk_weight<float>(w, static_cast<IdxT>(r));
    const T* row     = reinterpret_cast<const T*>(A + static_cast<IdxT>(r) * lda);
#pragma unroll
    for (int j = 0; j < NV; ++j) {
      if (lane + j * S < nvec) { add(acc[j], row[lane + j * S], wr); }
    }
  }
}

/** One warp per 256-row chunk of a long key; row groups are added by a fixed shuffle tree. */
template <int V, int NV, int S, typename WeightT, typename IdxT>
RAFT_KERNEL rrbk_seg_partials_kernel(const float* A,
                                     IdxT lda,
                                     const WeightT* w,
                                     const uint32_t* rows,
                                     const IdxT* offsets,
                                     const IdxT* chunk_offsets,
                                     IdxT ncols,
                                     IdxT nkeys,
                                     float* partials)
{
  constexpr int G = WarpSize / S;
  const IdxT part = (static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x) / WarpSize;
  const int lane  = threadIdx.x % S;
  const int g     = (threadIdx.x % WarpSize) / S;
  if (part >= chunk_offsets[nkeys]) { return; }
  IdxT lo = 0, hi = nkeys;
  while (lo < hi) {
    const IdxT mid = lo + (hi - lo) / 2;
    if (chunk_offsets[mid + 1] <= part) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const IdxT begin = offsets[lo] + (part - chunk_offsets[lo]) * kSegChunkRows;
  const IdxT end   = std::min<IdxT>(begin + kSegChunkRows, offsets[lo + 1]);
  const int nvec   = static_cast<int>(ncols / V);
  float acc[NV][V];
  rrbk_seg_sum<V, NV, S>(A, lda, w, rows, begin, end, nvec, lane, acc, g, G);
#pragma unroll
  for (int off = G / 2; off >= 1; off /= 2) {
#pragma unroll
    for (int j = 0; j < NV; ++j) {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        acc[j][i] = __fadd_rn(acc[j][i], __shfl_down_sync(0xffffffffu, acc[j][i], off * S));
      }
    }
  }
  if (g != 0) { return; }
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    const int cv = lane + j * S;
    if (cv < nvec) {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        partials[part * ncols + cv * V + i] = acc[j][i];
      }
    }
  }
}

/** S lanes per key: short keys are summed directly, long keys add their chunk partials in order. */
template <int V, int NV, int S, typename WeightT, typename IdxT>
RAFT_KERNEL rrbk_seg_finish_kernel(const float* A,
                                   IdxT lda,
                                   const WeightT* w,
                                   const uint32_t* rows,
                                   const IdxT* offsets,
                                   const IdxT* chunk_offsets,
                                   IdxT ncols,
                                   IdxT nkeys,
                                   const float* partials,
                                   float* sums)
{
  const IdxT key = (static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x) / S;
  const int lane = threadIdx.x % S;
  if (key >= nkeys) { return; }
  const IdxT begin = offsets[key];
  const IdxT end   = offsets[key + 1];
  if (begin == end) { return; }
  const int nvec = static_cast<int>(ncols / V);
  float acc[NV][V];
  if (end - begin <= kSegChunkRows) {
    rrbk_seg_sum<V, NV, S>(A, lda, w, rows, begin, end, nvec, lane, acc);
  } else {
#pragma unroll
    for (int j = 0; j < NV; ++j) {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        acc[j][i] = 0.0f;
      }
    }
    for (IdxT part = chunk_offsets[key]; part < chunk_offsets[key + 1]; ++part) {
#pragma unroll
      for (int j = 0; j < NV; ++j) {
        if (lane + j * S < nvec) {
#pragma unroll
          for (int i = 0; i < V; ++i) {
            acc[j][i] = __fadd_rn(acc[j][i], partials[part * ncols + (lane + j * S) * V + i]);
          }
        }
      }
    }
  }
#pragma unroll
  for (int j = 0; j < NV; ++j) {
    const int cv = lane + j * S;
    if (cv < nvec) {
#pragma unroll
      for (int i = 0; i < V; ++i) {
        const IdxT out = key * ncols + cv * V + i;
        sums[out]      = __fadd_rn(sums[out], acc[j][i]);
      }
    }
  }
}

template <int V, int NV, int S, typename WeightT, typename IdxT>
void rrbk_seg_launch(const float* A,
                     IdxT lda,
                     const WeightT* w,
                     const uint32_t* rows,
                     const IdxT* offsets,
                     const IdxT* chunk_offsets,
                     IdxT ncols,
                     IdxT nkeys,
                     IdxT max_parts,
                     float* partials,
                     float* sums,
                     cudaStream_t st,
                     bool dry_run)
{
  if (max_parts > 0) {
    raft::launch_kernel({st, 0, dry_run},
                        dim3(static_cast<uint32_t>(ceildiv<IdxT>(max_parts * WarpSize, kSegBlock))),
                        dim3(kSegBlock),
                        rrbk_seg_partials_kernel<V, NV, S, WeightT, IdxT>,
                        A,
                        lda,
                        w,
                        rows,
                        offsets,
                        chunk_offsets,
                        ncols,
                        nkeys,
                        partials);
  }
  raft::launch_kernel({st, 0, dry_run},
                      dim3(static_cast<uint32_t>(ceildiv<IdxT>(nkeys * S, kSegBlock))),
                      dim3(kSegBlock),
                      rrbk_seg_finish_kernel<V, NV, S, WeightT, IdxT>,
                      A,
                      lda,
                      w,
                      rows,
                      offsets,
                      chunk_offsets,
                      ncols,
                      nkeys,
                      static_cast<const float*>(partials),
                      sums);
}

/** S lanes per key: vectors per row rounded up to a power of two, at most a warp. */
template <int V, typename WeightT, typename IdxT>
void rrbk_seg_dispatch(const float* A,
                       IdxT lda,
                       const WeightT* w,
                       const uint32_t* rows,
                       const IdxT* offsets,
                       const IdxT* chunk_offsets,
                       IdxT ncols,
                       IdxT nkeys,
                       IdxT max_parts,
                       float* partials,
                       float* sums,
                       cudaStream_t st,
                       bool dry_run)
{
  const IdxT nvec = ncols / V;
  auto run        = [&](auto nv, auto s) {
    rrbk_seg_launch<V, decltype(nv)::value, decltype(s)::value>(A,
                                                                lda,
                                                                w,
                                                                rows,
                                                                offsets,
                                                                chunk_offsets,
                                                                ncols,
                                                                nkeys,
                                                                max_parts,
                                                                partials,
                                                                sums,
                                                                st,
                                                                dry_run);
  };
  using std::integral_constant;
  if (nvec <= 1) {
    run(integral_constant<int, 1>{}, integral_constant<int, 1>{});
  } else if (nvec <= 2) {
    run(integral_constant<int, 1>{}, integral_constant<int, 2>{});
  } else if (nvec <= 4) {
    run(integral_constant<int, 1>{}, integral_constant<int, 4>{});
  } else if (nvec <= 8) {
    run(integral_constant<int, 1>{}, integral_constant<int, 8>{});
  } else if (nvec <= 16) {
    run(integral_constant<int, 1>{}, integral_constant<int, 16>{});
  } else if (nvec <= 32) {
    run(integral_constant<int, 1>{}, integral_constant<int, 32>{});
  } else if (nvec <= 64) {
    run(integral_constant<int, 2>{}, integral_constant<int, 32>{});
  } else {
    run(integral_constant<int, 4>{}, integral_constant<int, 32>{});
  }
}

/**
 * Many keys, narrow rows: stable radix sort of (key, row), dense segment offsets from the key
 * boundaries, then each key's rows are summed in sorted row order. Keys longer than one chunk are
 * split into fixed 256-row chunks whose partials are added in chunk order.
 */
template <typename KeysIteratorT, typename WeightT, typename IdxT>
void rrbk_seg(const float* d_A,
              IdxT lda,
              KeysIteratorT d_keys,
              const WeightT* d_weights,
              IdxT nrows,
              IdxT ncols,
              IdxT nkeys,
              float* d_sums,
              cudaStream_t st,
              bool dry_run)
{
  rmm::device_uvector<uint32_t> keys(nrows, st), keys_alt(nrows, st);
  rmm::device_uvector<uint32_t> row_ids(nrows, st), row_ids_alt(nrows, st);
  rmm::device_uvector<IdxT> offsets(nkeys + 1, st), counts(nkeys + 1, st);
  rmm::device_uvector<IdxT> chunk_offsets(nkeys + 1, st);
  cub::DoubleBuffer<uint32_t> keys_db(keys.data(), keys_alt.data());
  cub::DoubleBuffer<uint32_t> row_ids_db(row_ids.data(), row_ids_alt.data());
  const int end_bit      = rrbk_key_bits(nkeys);
  std::size_t sort_bytes = 0, scan_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    nullptr, sort_bytes, keys_db, row_ids_db, static_cast<int>(nrows), 0, end_bit, st));
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, scan_bytes, counts.data(), chunk_offsets.data(), static_cast<int>(nkeys + 1), st));
  rmm::device_buffer tmp(std::max(sort_bytes, scan_bytes), st);
  // A key with more than one chunk has more than kSegChunkRows rows.
  const IdxT max_parts = nrows / kSegChunkRows + std::min<IdxT>(nkeys, nrows / (kSegChunkRows + 1));
  rmm::device_uvector<float> partials(static_cast<std::size_t>(max_parts) * ncols, st);

  raft::launch_kernel({st, 0, dry_run},
                      dim3(static_cast<uint32_t>(ceildiv<IdxT>(nrows, 256))),
                      dim3(256),
                      rrbk_sort_input_kernel<KeysIteratorT, IdxT>,
                      d_keys,
                      nrows,
                      keys_db.Current(),
                      row_ids_db.Current());
  if (!dry_run) {
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
      tmp.data(), sort_bytes, keys_db, row_ids_db, static_cast<int>(nrows), 0, end_bit, st));
  }
  raft::launch_kernel({st, 0, dry_run},
                      dim3(static_cast<uint32_t>(ceildiv<IdxT>(nrows, 256))),
                      dim3(256),
                      rrbk_boundaries_kernel<IdxT>,
                      static_cast<const uint32_t*>(keys_db.Current()),
                      nrows,
                      nkeys,
                      offsets.data());
  raft::launch_kernel({st, 0, dry_run},
                      dim3(static_cast<uint32_t>(ceildiv<IdxT>(nkeys + 1, 256))),
                      dim3(256),
                      rrbk_chunk_counts_kernel<IdxT>,
                      static_cast<const IdxT*>(offsets.data()),
                      nkeys,
                      counts.data());
  if (!dry_run) {
    RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(tmp.data(),
                                                scan_bytes,
                                                counts.data(),
                                                chunk_offsets.data(),
                                                static_cast<int>(nkeys + 1),
                                                st));
  }
  const bool vectorized = ncols % 4 == 0 && lda % 4 == 0 && raft::is_aligned(d_A, 16);
  if (vectorized) {
    rrbk_seg_dispatch<4>(d_A,
                         lda,
                         d_weights,
                         static_cast<const uint32_t*>(row_ids_db.Current()),
                         static_cast<const IdxT*>(offsets.data()),
                         static_cast<const IdxT*>(chunk_offsets.data()),
                         ncols,
                         nkeys,
                         max_parts,
                         partials.data(),
                         d_sums,
                         st,
                         dry_run);
  } else {
    rrbk_seg_dispatch<1>(d_A,
                         lda,
                         d_weights,
                         static_cast<const uint32_t*>(row_ids_db.Current()),
                         static_cast<const IdxT*>(offsets.data()),
                         static_cast<const IdxT*>(chunk_offsets.data()),
                         ncols,
                         nkeys,
                         max_parts,
                         partials.data(),
                         d_sums,
                         st,
                         dry_run);
  }
}

// ---------------------------------------------------------------- sorted chunks (general)

constexpr int64_t kCsrBatchRows  = 4000000;
constexpr int64_t kCsrChunk      = 1024;
constexpr int64_t kCsrMinChunk   = 256;
constexpr int64_t kCsrMinWorkers = 4096;
constexpr int kCsrRowsAhead      = 4;
constexpr uint32_t kCsrBlock     = 128;

/**
 * Segment offsets of the sorted keys, the number of chunks (items) of each key, and whether it
 * needs partial slots, packed as (items << 32 | slots) so a single exclusive scan yields both
 * offsets. Also resets the per-(key, tile) completion counters.
 */
template <typename IdxT>
RAFT_KERNEL rrbk_csr_offsets_kernel(const uint32_t* s_keys,
                                    IdxT n,
                                    IdxT nkeys,
                                    IdxT chunk,
                                    IdxT n_tiles,
                                    IdxT* off,
                                    uint64_t* cnt,
                                    unsigned* counters)
{
  const IdxT k = static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (k > nkeys) { return; }
  auto lower_bound = [&](IdxT key) {
    IdxT lo = 0, hi = n;
    while (lo < hi) {
      const IdxT mid = (lo + hi) / 2;
      if (static_cast<IdxT>(s_keys[mid]) < key) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    return lo;
  };
  const IdxT b = lower_bound(k);
  off[k]       = b;
  if (k == nkeys) {
    cnt[k] = 0;
    return;
  }
  for (IdxT t = 0; t < n_tiles; ++t) {
    counters[k * n_tiles + t] = 0;
  }
  const auto c = static_cast<uint64_t>(ceildiv<IdxT>(lower_bound(k + 1) - b, chunk));
  cnt[k]       = (c << 32) | (c > 1 ? c : 0);
}

/** (Vectorized) load of VecLen contiguous elements, zero when invalid. */
template <int VecLen, typename DataIteratorT, typename SumsT, typename IdxT>
DI void rrbk_load(DataIteratorT d_A, IdxT idx, bool valid, SumsT (&v)[VecLen])
{
  if constexpr (VecLen > 1) {
    TxN_t<SumsT, VecLen> x;
    x.fill(SumsT(0));
    if (valid) { x.load(static_cast<const SumsT*>(d_A), idx); }
#pragma unroll
    for (int c = 0; c < VecLen; ++c) {
      v[c] = x.val.data[c];
    }
  } else {
    v[0] = valid ? static_cast<SumsT>(d_A[idx]) : SumsT(0);
  }
}

/**
 * A group of g lanes (lc column lanes x g / lc row lanes) reduces one (item, column tile), where an
 * item is a fixed `chunk`-row slice of one key's sorted rows. Row lanes take rows at a fixed stride
 * and are combined by a fixed shuffle tree. Single-item keys add their result to the output
 * directly; multi-item keys write a partial, and the last item to finish (counted with an integer
 * atomic) adds all partials in item order.
 */
template <int VecLen,
          int kVecs,
          typename DataIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
RAFT_KERNEL rrbk_csr_reduce_kernel(DataIteratorT d_A,
                                   IdxT lda,
                                   const WeightT* d_weights,
                                   const uint32_t* d_row_ids,
                                   const IdxT* off,
                                   const uint64_t* cnt_off,
                                   IdxT nkeys,
                                   IdxT ncols,
                                   IdxT chunk,
                                   int lc,
                                   int g,
                                   IdxT n_tiles,
                                   SumsT* partials,
                                   unsigned* counters,
                                   SumsT* d_sums)
{
  const IdxT worker = (static_cast<IdxT>(blockIdx.x) * blockDim.x + threadIdx.x) / g;
  const IdxT item   = worker / n_tiles;
  const IdxT tile   = worker % n_tiles;
  auto item_off     = [&](IdxT k) { return static_cast<IdxT>(cnt_off[k] >> 32); };
  if (item >= item_off(nkeys)) { return; }
  const int lane_in_warp = laneId();
  const int lane         = lane_in_warp % g;
  const unsigned mask    = g == WarpSize ? 0xffffffffu : ((1u << g) - 1u) << (lane_in_warp - lane);
  IdxT lo = 0, hi = nkeys + 1;  // key = last k with item_off(k) <= item
  while (lo < hi) {
    const IdxT mid = (lo + hi) / 2;
    if (item_off(mid) <= item) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const IdxT key    = lo - 1;
  const IdxT j      = item - item_off(key);
  const IdxT nitems = item_off(key + 1) - item_off(key);
  const IdxT r0     = off[key] + j * chunk;
  const IdxT r1     = std::min<IdxT>(r0 + chunk, off[key + 1]);
  const int c       = lane % lc;
  const int r       = lane / lc;
  const int lr      = g / lc;
  const IdxT col0   = tile * (kVecs * VecLen * lc) + c * VecLen;

  SumsT acc[kVecs][VecLen] = {};
  for (IdxT i = r0 + r; i < r1; i += static_cast<IdxT>(lr) * kCsrRowsAhead) {
    SumsT a[kCsrRowsAhead][kVecs][VecLen];
    SumsT w[kCsrRowsAhead];
#pragma unroll
    for (int u = 0; u < kCsrRowsAhead; ++u) {
      const IdxT ii    = i + static_cast<IdxT>(u) * lr;
      const bool valid = ii < r1;
      const IdxT row   = valid ? static_cast<IdxT>(d_row_ids[ii]) : IdxT(0);
      w[u]             = valid ? rrbk_weight<SumsT>(d_weights, row) : SumsT(0);
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
        const IdxT col = col0 + v * VecLen * lc;
        rrbk_load<VecLen>(d_A, lda * row + col, valid && col < ncols, a[u][v]);
      }
    }
#pragma unroll
    for (int u = 0; u < kCsrRowsAhead; ++u) {
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
#pragma unroll
        for (int cc = 0; cc < VecLen; ++cc) {
          acc[v][cc] = rrbk_fma(a[u][v][cc], w[u], acc[v][cc]);
        }
      }
    }
  }
  auto row_lane_tree = [&]() {
    for (int o = g / 2; o >= lc; o /= 2) {
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
#pragma unroll
        for (int cc = 0; cc < VecLen; ++cc) {
          acc[v][cc] += __shfl_down_sync(mask, acc[v][cc], o, g);
        }
      }
    }
  };
  auto add_to_sums = [&]() {
    if (r != 0) { return; }
#pragma unroll
    for (int v = 0; v < kVecs; ++v) {
      const IdxT col = col0 + v * VecLen * lc;
      if (col < ncols) {
#pragma unroll
        for (int cc = 0; cc < VecLen; ++cc) {
          d_sums[key * ncols + col + cc] += acc[v][cc];
        }
      }
    }
  };
  row_lane_tree();
  if (nitems == 1) {
    add_to_sums();
    return;
  }
  const IdxT pbase = static_cast<IdxT>(cnt_off[key] & 0xffffffffu);
  if (r == 0) {
#pragma unroll
    for (int v = 0; v < kVecs; ++v) {
      const IdxT col = col0 + v * VecLen * lc;
      if (col < ncols) {
#pragma unroll
        for (int cc = 0; cc < VecLen; ++cc) {
          partials[(pbase + j) * ncols + col + cc] = acc[v][cc];
        }
      }
    }
  }
  __threadfence();
  __syncwarp(mask);
  unsigned last = 0;
  if (lane == 0) {
    last = atomicAdd(counters + key * n_tiles + tile, 1u) == static_cast<unsigned>(nitems - 1);
  }
  if (!__shfl_sync(mask, last, 0, g)) { return; }
  __threadfence();
  // Four interleaved accumulators keep four loads in flight; they are added in a fixed order.
  constexpr int kCombine              = 4;
  SumsT acc4[kCombine][kVecs][VecLen] = {};
  for (IdxT jj = r; jj < nitems; jj += static_cast<IdxT>(lr) * kCombine) {
#pragma unroll
    for (int q = 0; q < kCombine; ++q) {
      const IdxT jq = jj + static_cast<IdxT>(q) * lr;
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
        const IdxT col = col0 + v * VecLen * lc;
        if (jq < nitems && col < ncols) {
#pragma unroll
          for (int cc = 0; cc < VecLen; ++cc) {
            acc4[q][v][cc] += __ldcg(partials + (pbase + jq) * ncols + col + cc);
          }
        }
      }
    }
  }
#pragma unroll
  for (int v = 0; v < kVecs; ++v) {
#pragma unroll
    for (int cc = 0; cc < VecLen; ++cc) {
      acc[v][cc] = (acc4[0][v][cc] + acc4[1][v][cc]) + (acc4[2][v][cc] + acc4[3][v][cc]);
    }
  }
  row_lane_tree();
  add_to_sums();
  if (lane == 0) { counters[key * n_tiles + tile] = 0; }
}

template <int VecLen,
          int kLaneBytes,
          typename DataIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void rrbk_csr_launch(DataIteratorT d_A,
                     IdxT lda,
                     const WeightT* d_weights,
                     const uint32_t* d_row_ids,
                     const IdxT* off,
                     const uint64_t* cnt_off,
                     IdxT nrows,
                     IdxT nkeys,
                     IdxT ncols,
                     IdxT chunk,
                     SumsT* partials,
                     unsigned* counters,
                     SumsT* d_sums,
                     cudaStream_t st,
                     bool dry_run)
{
  constexpr int kVecs     = kLaneBytes / (VecLen * sizeof(SumsT));
  const IdxT lanes_needed = ceildiv<IdxT>(ncols, IdxT{kVecs * VecLen});
  int lc                  = 1;
  while (lc < WarpSize && lc < lanes_needed) {
    lc *= 2;
  }
  const IdxT n_tiles   = ceildiv<IdxT>(ncols, IdxT{kVecs * VecLen} * lc);
  const IdxT max_items = nkeys + ceildiv<IdxT>(nrows, chunk);
  // Row lanes per group: about kCsrRowsAhead rows per lane for the mean item length.
  const IdxT mean_rows = std::min<IdxT>(chunk, ceildiv<IdxT>(nrows, nkeys));
  int g                = lc;
  while (g < WarpSize && IdxT{g / lc} * 2 * kCsrRowsAhead <= mean_rows) {
    g *= 2;
  }
  const IdxT n_threads = max_items * n_tiles * g;
  raft::launch_kernel({st, 0, dry_run},
                      dim3(static_cast<uint32_t>(ceildiv<IdxT>(n_threads, IdxT{kCsrBlock}))),
                      dim3(kCsrBlock),
                      rrbk_csr_reduce_kernel<VecLen, kVecs, DataIteratorT, WeightT, SumsT, IdxT>,
                      d_A,
                      lda,
                      d_weights,
                      d_row_ids,
                      off,
                      cnt_off,
                      nkeys,
                      ncols,
                      chunk,
                      lc,
                      g,
                      n_tiles,
                      partials,
                      counters,
                      d_sums);
}

/**
 * General path: stable radix sort of (key, row) in batches of at most kCsrBatchRows rows, then each
 * key's sorted rows are reduced in fixed `chunk`-row items whose partials are added in item order.
 * The chunk size depends only on the shape.
 */
template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void rrbk_csr(DataIteratorT d_A,
              IdxT lda,
              KeysIteratorT d_keys,
              const WeightT* d_weights,
              IdxT nrows,
              IdxT ncols,
              IdxT nkeys,
              SumsT* d_sums,
              cudaStream_t st,
              bool dry_run)
{
  const IdxT batch_rows = std::min<IdxT>(nrows, kCsrBatchRows);
  IdxT chunk            = kCsrChunk;
  {
    // Shrink chunks until there are enough (item, tile) workers.
    const IdxT tiles = ceildiv<IdxT>(ncols * static_cast<IdxT>(sizeof(SumsT)), 32 * WarpSize);
    while (chunk > kCsrMinChunk &&
           (ceildiv<IdxT>(batch_rows, chunk) + std::min(nkeys, batch_rows)) * tiles <
             kCsrMinWorkers) {
      chunk /= 2;
    }
  }
  rmm::device_uvector<uint32_t> keys(batch_rows, st), keys_alt(batch_rows, st);
  rmm::device_uvector<uint32_t> row_ids(batch_rows, st), row_ids_alt(batch_rows, st);
  cub::DoubleBuffer<uint32_t> keys_db(keys.data(), keys_alt.data());
  cub::DoubleBuffer<uint32_t> row_ids_db(row_ids.data(), row_ids_alt.data());
  const int end_bit      = rrbk_key_bits(nkeys);
  std::size_t sort_bytes = 0, scan_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    nullptr, sort_bytes, keys_db, row_ids_db, static_cast<int>(batch_rows), 0, end_bit, st));
  rmm::device_uvector<IdxT> off(nkeys + 1, st);
  rmm::device_uvector<uint64_t> cnt(nkeys + 1, st), cnt_off(nkeys + 1, st);
  RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
    nullptr, scan_bytes, cnt.data(), cnt_off.data(), static_cast<int>(nkeys + 1), st));
  rmm::device_buffer tmp(std::max(sort_bytes, scan_bytes), st);

  constexpr int kVecLen = 16 / sizeof(SumsT);
  constexpr bool kIsPlainPointer =
    std::is_same_v<DataIteratorT, const SumsT*> || std::is_same_v<DataIteratorT, SumsT*>;
  bool vectorized = false;
  if constexpr (kIsPlainPointer) {
    vectorized = ncols % kVecLen == 0 && lda % kVecLen == 0 && raft::is_aligned(d_A, 16) &&
                 raft::is_aligned(d_sums, 16);
  }
  const bool narrow_rows = ncols * sizeof(SumsT) < 32 * WarpSize;
  // Each multi-item key has more than `chunk` rows, so it has fewer than 2 * len / chunk items.
  const IdxT max_parts = 2 * ceildiv<IdxT>(batch_rows, chunk);
  rmm::device_uvector<SumsT> partials(max_parts * ncols, st);
  const IdxT max_tiles = ceildiv<IdxT>(ncols, IdxT{kVecLen});
  rmm::device_uvector<unsigned> counters(nkeys * max_tiles, st);

  for (IdxT batch_start = 0; batch_start < nrows; batch_start += batch_rows) {
    const IdxT n_batch_rows = std::min<IdxT>(batch_rows, nrows - batch_start);
    keys_db                 = cub::DoubleBuffer<uint32_t>(keys.data(), keys_alt.data());
    row_ids_db              = cub::DoubleBuffer<uint32_t>(row_ids.data(), row_ids_alt.data());
    raft::launch_kernel({st, 0, dry_run},
                        dim3(static_cast<uint32_t>(ceildiv<IdxT>(n_batch_rows, 256))),
                        dim3(256),
                        rrbk_sort_input_kernel<KeysIteratorT, IdxT>,
                        d_keys + batch_start,
                        n_batch_rows,
                        keys_db.Current(),
                        row_ids_db.Current());
    if (!dry_run) {
      RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(tmp.data(),
                                                    sort_bytes,
                                                    keys_db,
                                                    row_ids_db,
                                                    static_cast<int>(n_batch_rows),
                                                    0,
                                                    end_bit,
                                                    st));
    }
    raft::launch_kernel({st, 0, dry_run},
                        dim3(static_cast<uint32_t>(ceildiv<IdxT>(nkeys + 1, 256))),
                        dim3(256),
                        rrbk_csr_offsets_kernel<IdxT>,
                        static_cast<const uint32_t*>(keys_db.Current()),
                        n_batch_rows,
                        nkeys,
                        chunk,
                        max_tiles,
                        off.data(),
                        cnt.data(),
                        counters.data());
    if (!dry_run) {
      RAFT_CUDA_TRY(cub::DeviceScan::ExclusiveSum(
        tmp.data(), scan_bytes, cnt.data(), cnt_off.data(), static_cast<int>(nkeys + 1), st));
    }
    const auto batch_A     = d_A + batch_start * lda;
    const WeightT* batch_w = d_weights != nullptr ? d_weights + batch_start : nullptr;
    auto launch            = [&](auto vec_len) {
      constexpr int VecLen = decltype(vec_len)::value;
      if (narrow_rows) {
        rrbk_csr_launch<VecLen, 16>(batch_A,
                                    lda,
                                    batch_w,
                                    row_ids_db.Current(),
                                    off.data(),
                                    cnt_off.data(),
                                    n_batch_rows,
                                    nkeys,
                                    ncols,
                                    chunk,
                                    partials.data(),
                                    counters.data(),
                                    d_sums,
                                    st,
                                    dry_run);
      } else {
        rrbk_csr_launch<VecLen, 32>(batch_A,
                                    lda,
                                    batch_w,
                                    row_ids_db.Current(),
                                    off.data(),
                                    cnt_off.data(),
                                    n_batch_rows,
                                    nkeys,
                                    ncols,
                                    chunk,
                                    partials.data(),
                                    counters.data(),
                                    d_sums,
                                    st,
                                    dry_run);
      }
    };
    if constexpr (kIsPlainPointer) {
      if (vectorized) {
        launch(std::integral_constant<int, kVecLen>{});
        continue;
      }
    }
    launch(std::integral_constant<int, 1>{});
  }
}

// ---------------------------------------------------------------- dispatch

/**
 * Picks the path from the shape alone, so a given shape takes the same path, with the same
 * summation order, on every GPU.
 */
template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void sum_rows_by_keyrowmajor(DataIteratorT d_A,
                             IdxT lda,
                             KeysIteratorT d_keys,
                             const WeightT* d_weights,
                             IdxT nrows,
                             IdxT ncols,
                             IdxT nkeys,
                             SumsT* d_sums,
                             cudaStream_t st,
                             bool dry_run)
{
  if (nrows <= 0 || ncols <= 0 || nkeys <= 0) { return; }
  RAFT_EXPECTS(static_cast<uint64_t>(nkeys) <= std::numeric_limits<uint32_t>::max(),
               "Too many keys");
  // IdxT may be unsigned or 32-bit; all paths index with int64_t.
  const int64_t m = nrows, n = ncols, k = nkeys, ld = lda;
  if (rrbk_make_priv_plan<SumsT>(m, n, k).ok) {
    rrbk_priv(d_A, ld, d_keys, d_weights, m, n, k, d_sums, st, dry_run);
    return;
  }
  if constexpr (std::is_same_v<SumsT, float>) {
    if (n <= 16 && k * n * rrbk_make_bins_plan(m).nbins * 8 <= kBinsMaxBytes) {
      rrbk_bins(d_A, ld, d_keys, d_weights, m, n, k, d_sums, st, dry_run);
      return;
    }
    constexpr bool kIsPlainPointer =
      std::is_same_v<DataIteratorT, const float*> || std::is_same_v<DataIteratorT, float*>;
    if constexpr (kIsPlainPointer) {
      if (n <= kSegMaxCols && k <= kSegMaxKeys && m <= std::numeric_limits<int>::max() &&
          m < kSegMaxRowsPerKey * k) {
        rrbk_seg(
          static_cast<const float*>(d_A), ld, d_keys, d_weights, m, n, k, d_sums, st, dry_run);
        return;
      }
    }
  }
  rrbk_csr(d_A, ld, d_keys, d_weights, m, n, k, d_sums, st, dry_run);
}

/**
 * @brief Computes the weighted reduction of matrix rows for each given key
 *
 * The result is deterministic: repeated calls with the same arguments return the same bits, also
 * on different GPUs.
 *
 * @tparam DataIteratorT Random-access iterator type, for reading input matrix
 *                       (may be a simple pointer type)
 * @tparam KeysIteratorT Random-access iterator type, for reading input keys
 *                       (may be a simple pointer type)
 * @tparam SumsT         Type of the output sums
 * @tparam IdxT          Index type
 *
 * @param[in]  d_A         Input data array (lda x nrows)
 * @param[in]  lda         Real row size for input data, d_A
 * @param[in]  d_keys      Keys for each row (1 x nrows)
 * @param[in]  d_weights   Weights for each observation in d_A (1 x nrows)
 * @param[out] d_keys_char UNUSED, kept for compatibility
 * @param[in]  nrows       Number of rows in d_A and d_keys
 * @param[in]  ncols       Number of data columns in d_A
 * @param[in]  nkeys       Number of unique keys in d_keys
 * @param[out] d_sums      Row sums by key (ncols x d_keys)
 * @param[in]  stream      CUDA stream
 * @param[in]  reset_sums  Whether to reset the output sums to zero before reducing
 * @param[in]  dry_run     Only allocate the scratch memory, without running any CUDA work
 */
template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void reduce_rows_by_key(const DataIteratorT d_A,
                        IdxT lda,
                        KeysIteratorT d_keys,
                        const WeightT* d_weights,
                        char* d_keys_char,
                        IdxT nrows,
                        IdxT ncols,
                        IdxT nkeys,
                        SumsT* d_sums,
                        cudaStream_t stream,
                        bool reset_sums,
                        bool dry_run = false)
{
  static_assert(
    !is_floating_point_v<WeightT> ||
      (is_floating_point_v<SumsT> && sizeof(WeightT) <= sizeof(SumsT)),
    "Floating weight type is broader than the sum type. Instead of silently casting down the "
    "weight precision, we throw an error here.");

  if (reset_sums && !dry_run) {
    RAFT_CUDA_TRY(cudaMemsetAsync(d_sums, 0, ncols * nkeys * sizeof(SumsT), stream));
  }

  (void)d_keys_char;
  sum_rows_by_keyrowmajor(
    d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, stream, dry_run);
}

/**
 * @brief Computes the reduction of matrix rows for each given key
 * @tparam DataIteratorT Random-access iterator type, for reading input matrix (may be a simple
 * pointer type)
 * @tparam KeysIteratorT Random-access iterator type, for reading input keys (may be a simple
 * pointer type)
 * @tparam SumsT         Type of the output sums
 * @tparam IdxT          Index type
 * @param[in]  d_A         Input data array (lda x nrows)
 * @param[in]  lda         Real row size for input data, d_A
 * @param[in]  d_keys      Keys for each row (1 x nrows)
 * @param      d_keys_char UNUSED, kept for compatibility
 * @param[in]  nrows       Number of rows in d_A and d_keys
 * @param[in]  ncols       Number of data columns in d_A
 * @param[in]  nkeys       Number of unique keys in d_keys
 * @param[out] d_sums      Row sums by key (ncols x d_keys)
 * @param[in]  stream      CUDA stream
 * @param[in]  reset_sums  Whether to reset the output sums to zero before reducing
 * @param[in]  dry_run     Only allocate the scratch memory, without running any CUDA work
 */
template <typename DataIteratorT, typename KeysIteratorT, typename SumsT, typename IdxT>
void reduce_rows_by_key(const DataIteratorT d_A,
                        IdxT lda,
                        KeysIteratorT d_keys,
                        char* d_keys_char,
                        IdxT nrows,
                        IdxT ncols,
                        IdxT nkeys,
                        SumsT* d_sums,
                        cudaStream_t stream,
                        bool reset_sums,
                        bool dry_run = false)
{
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;
  reduce_rows_by_key(d_A,
                     lda,
                     d_keys,
                     static_cast<DataType*>(nullptr),
                     d_keys_char,
                     nrows,
                     ncols,
                     nkeys,
                     d_sums,
                     stream,
                     reset_sums,
                     dry_run);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
