/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/detail/coalesced_reduction.cuh>
#include <raft/linalg/detail/map.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>
#include <raft/util/vectorized.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_ptx.cuh>

#include <algorithm>
#include <cstdint>
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

//
// Reduce by keys
// We need to sum each dimension by labels
// The labels are not adjacent
//

/** Floating point including the extended types, which std::is_floating_point does not cover. */
template <typename T>
inline constexpr bool is_floating_point_v =
  std::is_floating_point_v<T> || std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;

/** (Vectorized) load */
template <int VecLen, typename DataIteratorT, typename SumsT, typename IdxT>
DI void sum_rows_by_key_load(const DataIteratorT d_A, IdxT idx, bool valid, SumsT (&v)[VecLen])
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

/** (Vectorized) atomic add */
template <int VecLen, typename SumsT>
DI void sum_rows_by_key_atomic_add(SumsT* dst, const SumsT (&v)[VecLen])
{
  if constexpr (std::is_same_v<SumsT, float> && VecLen == 4) {
    raft::myAtomicAdd(reinterpret_cast<float4*>(dst), make_float4(v[0], v[1], v[2], v[3]));
  } else {
#pragma unroll
    for (int c = 0; c < VecLen; ++c) {
      raft::myAtomicAdd(dst + c, v[c]);
    }
  }
}

// Allocates a buffer in double precision in addition to d_sums when SumsT != double
// This is needed for atomic add in double precision.
template <typename SumsT>
struct sum_rows_by_key_accumulator {
  static constexpr bool kOwnBuffer = !std::is_same_v<SumsT, double>;

  sum_rows_by_key_accumulator(SumsT* d_sums, size_t n, bool dry_run, cudaStream_t st)
    : sums_(d_sums), buffer_(kOwnBuffer ? n : 0, st)
  {
    if (kOwnBuffer && !dry_run) {
      RAFT_CUDA_TRY(cudaMemsetAsync(buffer_.data(), 0, n * sizeof(double), st));
    }
  }

  double* data()
  {
    if constexpr (kOwnBuffer) {
      return buffer_.data();
    } else {
      return sums_;
    }
  }

  // d_sums += accumulated sums, rounded to from double to SumsT
  // no-op when SumsT is double
  void finish(cudaStream_t st)
  {
    if constexpr (kOwnBuffer) {
      map<false>(st,
                 sums_,
                 static_cast<int64_t>(buffer_.size()),
                 raft::add_op{},
                 static_cast<const SumsT*>(sums_),
                 static_cast<const double*>(buffer_.data()));
    }
  }

 private:
  SumsT* sums_;
  rmm::device_uvector<double> buffer_;
};

/** Adds the sums of a thread to dst with double atomics. */
template <int VecLen, typename SumsT>
DI void sum_rows_by_key_atomic_add_double(double* dst, const SumsT (&v)[VecLen])
{
#pragma unroll
  for (int c = 0; c < VecLen; ++c) {
    raft::myAtomicAdd(dst + c, static_cast<double>(v[c]));
  }
}

/**
 * One thread owns one element of a row, multiplies it by the row's weight and adds it to the sums
 * of the row's key. When vectorized, one threads owns a vector of contiguous elements.
 */
template <int VecLen,
          typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
RAFT_KERNEL sum_rows_by_key_kernel_rowmajor_elementwise(const DataIteratorT d_A,
                                                        IdxT lda,
                                                        const WeightT* d_weights,
                                                        KeysIteratorT d_keys,
                                                        IdxT nrows,
                                                        IdxT ncols,
                                                        SumsT* d_sums)
{
  const IdxT q            = threadIdx.x + blockDim.x * static_cast<IdxT>(blockIdx.x);
  const IdxT vecs_per_row = ncols / VecLen;
  const IdxT i            = q / vecs_per_row;
  if (i >= nrows) return;
  const IdxT j  = (q - i * vecs_per_row) * VecLen;
  const SumsT w = d_weights != nullptr ? static_cast<SumsT>(d_weights[i]) : SumsT(1);
  SumsT a[VecLen];
  sum_rows_by_key_load<VecLen>(d_A, lda * i + j, true, a);
#pragma unroll
  for (int c = 0; c < VecLen; ++c) {
    a[c] *= w;
  }
  sum_rows_by_key_atomic_add<VecLen>(d_sums + ncols * static_cast<IdxT>(d_keys[i]) + j, a);
}

template <int VecLen,
          typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void sum_rows_by_key_rowmajor_elementwise(const DataIteratorT d_A,
                                          IdxT lda,
                                          const KeysIteratorT d_keys,
                                          const WeightT* d_weights,
                                          IdxT nrows,
                                          IdxT ncols,
                                          SumsT* d_sums,
                                          uint32_t block_dim,
                                          cudaStream_t st)
{
  const IdxT n_items  = nrows * (ncols / VecLen);
  const auto grid_dim = static_cast<uint32_t>(ceildiv<IdxT>(n_items, static_cast<IdxT>(block_dim)));
  raft::launch_kernel(st,
                      grid_dim,
                      block_dim,
                      sum_rows_by_key_kernel_rowmajor_elementwise<VecLen,
                                                                  DataIteratorT,
                                                                  KeysIteratorT,
                                                                  WeightT,
                                                                  SumsT,
                                                                  IdxT>,
                      d_A,
                      lda,
                      d_weights,
                      d_keys,
                      nrows,
                      ncols,
                      d_sums);
}

constexpr int kFewKeysMaxThreads   = 512;        // Max threads per block
constexpr size_t kFewKeysSmemBytes = 48 * 1024;  // 48 KiB - Each GPU from Ampere+ should have it.

/**
 * Kernel specialized for few keys (high atomic contention) and narrow rows.
 *
 * - One copy has nkeys * ncols sums (same as the global sums). Each blocks holds multiple copies in
 * its shared memory.
 * - A block owns a range of whole rows.
 * - Each thread owns one column in one copy. It sums the rows in its column of its copy, using the
 * row's key. Therefore, each thread owns different addresses in shared memory, so no atomics are
 * needed.
 * - At the end, the copies are summed into the partial sums (i.e. d_partials[key][col][block]) in
 * global memory
 */
template <int kRowsAhead,
          typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
RAFT_KERNEL __launch_bounds__(kFewKeysMaxThreads)
  sum_rows_by_key_kernel_rowmajor_small_keys_narrow_rows(const DataIteratorT d_A,
                                                         IdxT lda,
                                                         const WeightT* d_weights,
                                                         KeysIteratorT d_keys,
                                                         IdxT nrows,
                                                         IdxT ncols,
                                                         IdxT nkeys,
                                                         IdxT rows_per_block,
                                                         SumsT* d_partials)
{
  // Shared memory which holds multiple copies. Chosen layout: smem[n_copies][nkeys][ncols]
  /// Layout choice:
  ///    - Each threads owns one column in one copy. So having ncols in the last dimension
  ///      makes neighboring threads access neighboring shared memory addresses - no bank conflicts.
  ///    - nkeys X ncols matches the global sum layout. Writing the sums to global memory is
  ///    coalesced.
  extern __shared__ __align__(16) unsigned char few_keys_smem[];
  SumsT* private_sums = reinterpret_cast<SumsT*>(few_keys_smem);
  const int n         = static_cast<int>(ncols);
  const int n_copies  = blockDim.x / n;
  /// c: column index owned by this thread
  const int c = threadIdx.x % n;
  /// p: copy index owned by this thread
  const int p = threadIdx.x / n;
  /// [r0, r1): range of rows owned by this block.
  const IdxT r0 = blockIdx.x * rows_per_block;
  const IdxT r1 = std::min<IdxT>(r0 + rows_per_block, nrows);

  // Zero-initialize private sums
  for (IdxT i = threadIdx.x; i < n_copies * nkeys * ncols; i += blockDim.x) {
    private_sums[i] = SumsT(0);
  }
  __syncthreads();

  // Load rows and add them to the sums of their keys.
  /// private_sums[copies][keys][cols] -> mine[keys][:]
  SumsT* mine = private_sums + static_cast<IdxT>(p) * nkeys * ncols + c;
  for (IdxT r = r0 + p; r < r1; r += static_cast<IdxT>(kRowsAhead) * n_copies) {
    SumsT a[kRowsAhead];
    SumsT w[kRowsAhead];
    IdxT key[kRowsAhead];
    /// Load all kRowsAhead rows beforehand to have enough bytes in flight
#pragma unroll
    for (int u = 0; u < kRowsAhead; ++u) {
      const IdxT row   = r + static_cast<IdxT>(u) * n_copies;
      const bool valid = row < r1;
      a[u]             = valid ? static_cast<SumsT>(d_A[lda * row + c]) : SumsT(0);
      w[u]             = !valid                 ? SumsT(0)
                         : d_weights != nullptr ? static_cast<SumsT>(d_weights[row])
                                                : SumsT(1);
      key[u]           = valid ? static_cast<IdxT>(d_keys[row]) : IdxT(0);
    }
    /// Add each row to the sum of its key
#pragma unroll
    for (int u = 0; u < kRowsAhead; ++u) {
      mine[key[u] * ncols] += a[u] * w[u];  // mine[key][0] - each key is ncols apart
    }
  }
  __syncthreads();

  // Sum the copies into the partial sums
  for (IdxT i = threadIdx.x; i < nkeys * ncols; i += blockDim.x) {
    SumsT t = SumsT(0);
    for (int q = 0; q < n_copies; ++q) {
      t += private_sums[static_cast<IdxT>(q) * nkeys * ncols + i];
    }
    d_partials[static_cast<size_t>(i) * gridDim.x + blockIdx.x] = t;
  }
}

/** Whether one private copy of the sums fits the few-keys kernel's shared memory. */
template <typename SumsT, typename IdxT>
bool sum_rows_by_key_few_keys_fits(IdxT nkeys, IdxT ncols)
{
  const size_t one_copy_bytes =
    sizeof(SumsT) * static_cast<size_t>(nkeys) * static_cast<size_t>(ncols);
  return one_copy_bytes <= kFewKeysSmemBytes;
}

template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void sum_rows_by_key_rowmajor_few_keys(const DataIteratorT d_A,
                                       IdxT lda,
                                       const KeysIteratorT d_keys,
                                       const WeightT* d_weights,
                                       IdxT nrows,
                                       IdxT ncols,
                                       IdxT nkeys,
                                       SumsT* d_sums,
                                       cudaStream_t st,
                                       bool dry_run)
{
  RAFT_EXPECTS(
    ncols > 0 && ncols <= kFewKeysMaxThreads && sum_rows_by_key_few_keys_fits<SumsT>(nkeys, ncols),
    "The sums do not fit in shared memory");
  if (nrows == 0) { return; }
  constexpr int kRowsAhead  = 8;
  const size_t copy_bytes   = std::max<size_t>(1, sizeof(SumsT) * nkeys * ncols);
  const int n_copies        = static_cast<int>(std::max<size_t>(
    1, std::min<size_t>(kFewKeysMaxThreads / ncols, kFewKeysSmemBytes / copy_bytes)));
  const auto block_dim      = static_cast<uint32_t>(n_copies * ncols);
  const int blocks_per_sm   = block_dim >= 256 ? 2 : 4;
  const IdxT rows_per_block = std::max<IdxT>(
    ceildiv<IdxT>(nrows, static_cast<IdxT>(blocks_per_sm) * raft::getMultiProcessorCount()),
    n_copies * kRowsAhead);
  const auto grid_dim = static_cast<uint32_t>(ceildiv<IdxT>(nrows, rows_per_block));
  rmm::device_uvector<SumsT> partials(size_t{grid_dim} * nkeys * ncols, st);
  if (!dry_run) {
    raft::launch_kernel({st, copy_bytes * n_copies},
                        grid_dim,
                        block_dim,
                        sum_rows_by_key_kernel_rowmajor_small_keys_narrow_rows<kRowsAhead,
                                                                               DataIteratorT,
                                                                               KeysIteratorT,
                                                                               WeightT,
                                                                               SumsT,
                                                                               IdxT>,
                        d_A,
                        lda,
                        d_weights,
                        d_keys,
                        nrows,
                        ncols,
                        nkeys,
                        rows_per_block,
                        partials.data());
  }
  // Reduce partial sums into the final sum.
  coalescedReduction<SumsT, SumsT, IdxT>(dry_run,
                                         d_sums,
                                         partials.data(),
                                         static_cast<IdxT>(grid_dim),
                                         nkeys * ncols,
                                         SumsT(0),
                                         st,
                                         /* inplace */ true);
}

/** Prepare keys and row ids for radix-sort. */
template <typename KeysIteratorT, typename IdxT>
RAFT_KERNEL sum_rows_by_key_sort_input_kernel(KeysIteratorT d_keys,
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

/**
 * Sum sorted-by-key rows in registers and flush to output when the key changes or the chunk ends.
 *
 * Work split:
 *                 tile 0    tile 1   ...
 *   chunk 0    | warp 0  | warp 1  | ...
 *   chunk 1    | warp n_tiles | ...
 *
 * To have enough bytes in flight, each warp loads kRowsAhead rows before adding any of them to the
 * sums. On each row, each lane owns kVecs vectors of VecLen contiguous columns.
 */
template <int VecLen,
          int kVecs,
          int kRowsAhead,
          typename DataIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
RAFT_KERNEL sum_rows_by_key_kernel_rowmajor_sorted(const DataIteratorT d_A,
                                                   IdxT lda,
                                                   const WeightT* d_weights,
                                                   const uint32_t* d_keys,
                                                   const uint32_t* d_row_ids,
                                                   IdxT nrows,
                                                   IdxT ncols,
                                                   IdxT chunk,
                                                   IdxT n_tiles,
                                                   double* d_sums)
{
  // Each warp owns: rows [r0, r1) and columns [col0, col0 + kTileCols)
  constexpr IdxT kTileCols = kVecs * VecLen * WarpSize;
  const int64_t warp_id = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / WarpSize;
  const IdxT r0         = static_cast<IdxT>(warp_id / n_tiles) * chunk;
  if (r0 >= nrows) return;  // spare warps of the last block
  const IdxT r1   = std::min<IdxT>(r0 + chunk, nrows);
  const IdxT col0 = static_cast<IdxT>(warp_id % n_tiles) * kTileCols + laneId() * VecLen;

  // Each lane tracks in registers the sums of each vector and the current key of the processed row.
  SumsT acc[kVecs][VecLen] = {};
  uint32_t cur             = d_keys[r0];

  // Flush to output
  auto flush = [&]() {
#pragma unroll
    for (int v = 0; v < kVecs; ++v) {
      const IdxT col = col0 + v * VecLen * WarpSize;
      if (col < ncols) {
        sum_rows_by_key_atomic_add_double<VecLen>(d_sums + ncols * static_cast<IdxT>(cur) + col,
                                                  acc[v]);
      }
#pragma unroll
      for (int c = 0; c < VecLen; ++c) {
        acc[v][c] = SumsT(0);  // restart at zero for the next run
      }
    }
  };

  for (IdxT r = r0; r < r1; r += kRowsAhead) {
    // Phase 1: issue all loads of the next kRowsAhead sorted rows (keys, weights, data) before
    // using any of them, so they are in flight together.
    SumsT a[kRowsAhead][kVecs][VecLen];
    uint32_t key[kRowsAhead];
    SumsT w[kRowsAhead];
#pragma unroll
    for (int u = 0; u < kRowsAhead; ++u) {
      const IdxT i = r + u;
      // Invalid rows: Apply zero weight and unchanged key
      const bool valid = i < r1;
      const IdxT row   = valid ? static_cast<IdxT>(d_row_ids[i]) : IdxT(0);
      key[u]           = valid ? d_keys[i] : cur;
      w[u]             = !valid                 ? SumsT(0)
                         : d_weights != nullptr ? static_cast<SumsT>(d_weights[row])
                                                : SumsT(1);
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
        const IdxT col = col0 + v * VecLen * WarpSize;
        sum_rows_by_key_load<VecLen>(d_A, lda * row + col, valid && col < ncols, a[u][v]);
      }
    }
    // Phase 2: do FMA and flush on key changes
#pragma unroll
    for (int u = 0; u < kRowsAhead; ++u) {
      if (key[u] != cur) {
        flush();
        cur = key[u];
      }
#pragma unroll
      for (int v = 0; v < kVecs; ++v) {
#pragma unroll
        for (int c = 0; c < VecLen; ++c) {
          acc[v][c] += a[u][v][c] * w[u];
        }
      }
    }
  }
  flush();  // necessary
}

template <int VecLen,
          int kLaneBytes,
          typename SumsT,
          typename DataIteratorT,
          typename WeightT,
          typename IdxT>
void sum_rows_by_key_rowmajor_sorted_launch(const DataIteratorT d_A,
                                            IdxT lda,
                                            const WeightT* d_weights,
                                            const uint32_t* d_keys,
                                            const uint32_t* d_row_ids,
                                            IdxT nrows,
                                            IdxT ncols,
                                            double* d_sums,
                                            cudaStream_t st)
{
  constexpr int kVecs        = kLaneBytes / (VecLen * sizeof(SumsT));
  constexpr int kRowsAhead   = kLaneBytes == 16 ? 4 : 8;
  constexpr IdxT kWarpsPerSM = 64;
  // n_tiles: how many tiles of kVecs*VecLen columns fit in a row. Each warp owns one tile.
  const IdxT n_tiles = ceildiv<IdxT>(ncols, IdxT{kVecs * VecLen * WarpSize});
  // n_chunks: how many chunks of rows to split the work into. Each warp owns one chunk.
  const IdxT n_chunks =
    std::max<IdxT>(1, IdxT{kWarpsPerSM} * raft::getMultiProcessorCount() / n_tiles);
  // chunk: how many rows each warp owns. The last chunk may be smaller.
  const IdxT chunk = ceildiv<IdxT>(nrows, n_chunks);
  // n_warps: how many warps to cover everything.
  const IdxT n_warps           = ceildiv<IdxT>(nrows, chunk) * n_tiles;
  constexpr uint32_t block_dim = 128;
  const auto grid_dim = static_cast<uint32_t>(ceildiv<IdxT>(n_warps, IdxT{block_dim / WarpSize}));
  raft::launch_kernel(st,
                      grid_dim,
                      block_dim,
                      sum_rows_by_key_kernel_rowmajor_sorted<VecLen,
                                                             kVecs,
                                                             kRowsAhead,
                                                             DataIteratorT,
                                                             WeightT,
                                                             SumsT,
                                                             IdxT>,
                      d_A,
                      lda,
                      d_weights,
                      d_keys,
                      d_row_ids,
                      nrows,
                      ncols,
                      chunk,
                      n_tiles,
                      d_sums);
}

/**
 * Radix-sort the row ids by key and then sum runs of equal keys in registers.
 * Batching defines upper limit for peak memory allocation.
 */
template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void sum_rows_by_key_rowmajor_sorted(const DataIteratorT d_A,
                                     IdxT lda,
                                     const KeysIteratorT d_keys,
                                     const WeightT* d_weights,
                                     IdxT nrows,
                                     IdxT ncols,
                                     IdxT nkeys,
                                     SumsT* d_sums,
                                     cudaStream_t st,
                                     bool dry_run)
{
  // Limit batch size to 4M rows
  const IdxT batch_rows = std::min<IdxT>(nrows, 4000000);
  rmm::device_uvector<uint32_t> keys(batch_rows, st), keys_alt(batch_rows, st);
  rmm::device_uvector<uint32_t> row_ids(batch_rows, st), row_ids_alt(batch_rows, st);
  // Double buffering avoids additional scratch space when radix sorts undergoes > 1 pass.
  // 1 pass corresponds to 8-bit (selected by the CUB library)
  cub::DoubleBuffer<uint32_t> keys_db(keys.data(), keys_alt.data());
  cub::DoubleBuffer<uint32_t> row_ids_db(row_ids.data(), row_ids_alt.data());
  // Only the bits that can be set in a key in [0, nkeys) take part in the sort.
  int end_bit = 1;
  while (end_bit < 32 && (IdxT{1} << end_bit) < nkeys) {
    ++end_bit;
  }
  size_t tmp_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    nullptr, tmp_bytes, keys_db, row_ids_db, static_cast<int>(batch_rows), 0, end_bit, st));
  rmm::device_buffer tmp(tmp_bytes, st);
  sum_rows_by_key_accumulator<SumsT> acc(d_sums, static_cast<size_t>(nkeys) * ncols, dry_run, st);
  if (dry_run) { return; }
  auto* d_acc = acc.data();

  // Pick 16-byte vector when the input is a plain pointer and rows and sums are aligned.
  constexpr int kVecLen = 16 / sizeof(SumsT);
  constexpr bool kIsPlainPointer =
    std::is_same_v<DataIteratorT, const SumsT*> || std::is_same_v<DataIteratorT, SumsT*>;
  bool vectorized = false;
  if constexpr (kIsPlainPointer) {
    vectorized = ncols % kVecLen == 0 && lda % kVecLen == 0 && raft::is_aligned(d_A, 16) &&
                 raft::is_aligned(d_sums, 16);
  }

  // A warp covers kLaneBytes * 32 bytes of a row: 32 bytes per lane (a 1 KiB tile), or 16 bytes
  // per lane when the row is narrower than that, so fewer lanes idle.
  const bool narrow_rows = ncols * sizeof(SumsT) < 32 * WarpSize;

  for (IdxT batch_start = 0; batch_start < nrows; batch_start += batch_rows) {
    // Rows [batch_start, batch_start + n_batch_rows); row ids are local to the batch.
    const IdxT n_batch_rows      = std::min<IdxT>(batch_rows, nrows - batch_start);
    keys_db                      = cub::DoubleBuffer<uint32_t>(keys.data(), keys_alt.data());
    row_ids_db                   = cub::DoubleBuffer<uint32_t>(row_ids.data(), row_ids_alt.data());
    constexpr uint32_t block_dim = 256;
    const auto grid_dim = static_cast<uint32_t>(ceildiv<IdxT>(n_batch_rows, IdxT{block_dim}));
    raft::launch_kernel(st,
                        grid_dim,
                        block_dim,
                        sum_rows_by_key_sort_input_kernel,
                        d_keys + batch_start,
                        n_batch_rows,
                        keys_db.Current(),
                        row_ids_db.Current());
    RAFT_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
      tmp.data(), tmp_bytes, keys_db, row_ids_db, static_cast<int>(n_batch_rows), 0, end_bit, st));
    const auto batch_A       = d_A + batch_start * lda;
    const WeightT* batch_w   = d_weights != nullptr ? d_weights + batch_start : nullptr;
    const uint32_t* s_keys   = keys_db.Current();
    const uint32_t* s_rowids = row_ids_db.Current();
    auto launch              = [&](auto vec_len) {
      constexpr int VecLen = decltype(vec_len)::value;
      if (narrow_rows) {
        sum_rows_by_key_rowmajor_sorted_launch<VecLen, 16, SumsT>(
          batch_A, lda, batch_w, s_keys, s_rowids, n_batch_rows, ncols, d_acc, st);
      } else {
        sum_rows_by_key_rowmajor_sorted_launch<VecLen, 32, SumsT>(
          batch_A, lda, batch_w, s_keys, s_rowids, n_batch_rows, ncols, d_acc, st);
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
  acc.finish(st);
}

enum class sum_rows_by_key_kernel { elementwise, few_keys, sorted };

struct sum_rows_by_key_plan {
  sum_rows_by_key_kernel kernel;
  int vec_len;
  uint32_t block_dim;
};

/**
 * Kernel selection based on the problem size
 */
template <typename DataIteratorT, typename SumsT, typename IdxT>
sum_rows_by_key_plan sum_rows_by_key_select_kernel(
  const DataIteratorT d_A, IdxT lda, IdxT nrows, IdxT ncols, IdxT nkeys, const SumsT* d_sums)
{
  constexpr double kSortedMinRowsPerKey = 5000;
  constexpr double kSortedMinRowBytes   = 512;
  constexpr double kSortedMinInputBytes = 128.0 * 1024 * 1024;
  // Few keys and narrow rows: per-block sums in shared memory.
  constexpr IdxT kFewKeysMaxCols        = 128;
  constexpr IdxT kFewKeysMaxKeys        = 100;
  constexpr size_t kFewKeysMaxCopyBytes = kFewKeysSmemBytes / 2;  // at least 2 copies

  int dev_id, cc_major;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, dev_id));

  // Select vectorized loads and vectorized atomic writes
  bool whole_vectors = false;
  /// plain float pointer
  if constexpr (std::is_same_v<SumsT, float> && (std::is_same_v<DataIteratorT, const float*> ||
                                                 std::is_same_v<DataIteratorT, float*>)) {
    /// 16-byte alignments
    whole_vectors =
      ncols % 4 == 0 && lda % 4 == 0 && raft::is_aligned(d_A, 16) && raft::is_aligned(d_sums, 16);
  }
  /// Hopper+ (sm_9x) supports float4 atomics
  const int vec_len = whole_vectors && cc_major >= 9 ? 4 : 1;

  // Select kernel
  if (ncols > 0 && ncols <= kFewKeysMaxCols && nkeys <= kFewKeysMaxKeys &&
      sizeof(SumsT) * static_cast<size_t>(nkeys) * static_cast<size_t>(ncols) <=
        kFewKeysMaxCopyBytes) {
    return {sum_rows_by_key_kernel::few_keys, 1, 0};
  }
  const double row_bytes = static_cast<double>(ncols) * sizeof(SumsT);
  const bool contended   = static_cast<double>(nrows) >= kSortedMinRowsPerKey * nkeys;
  const bool wide_large =
    row_bytes >= kSortedMinRowBytes && row_bytes * nrows >= kSortedMinInputBytes;
  if (static_cast<uint64_t>(nkeys) <= std::numeric_limits<uint32_t>::max() &&
      (contended || wide_large)) {
    return {sum_rows_by_key_kernel::sorted, vec_len, 0};
  }
  return {sum_rows_by_key_kernel::elementwise, vec_len, 256u};
}

template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void sum_rows_by_keyrowmajor(const DataIteratorT d_A,
                             IdxT lda,
                             const KeysIteratorT d_keys,
                             const WeightT* d_weights,
                             IdxT nrows,
                             IdxT ncols,
                             IdxT nkeys,
                             SumsT* d_sums,
                             cudaStream_t st,
                             bool dry_run)
{
  const auto plan = sum_rows_by_key_select_kernel(d_A, lda, nrows, ncols, nkeys, d_sums);
  if (dry_run && plan.kernel == sum_rows_by_key_kernel::elementwise) { return; }

  constexpr bool kCanVectorize =
    std::is_same_v<SumsT, float> &&
    (std::is_same_v<DataIteratorT, const float*> || std::is_same_v<DataIteratorT, float*>);

  switch (plan.kernel) {
    case sum_rows_by_key_kernel::sorted:
      sum_rows_by_key_rowmajor_sorted(
        d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st, dry_run);
      break;
    case sum_rows_by_key_kernel::few_keys:
      sum_rows_by_key_rowmajor_few_keys(
        d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st, dry_run);
      break;
    case sum_rows_by_key_kernel::elementwise:
      if constexpr (kCanVectorize) {
        if (plan.vec_len == 4) {
          sum_rows_by_key_rowmajor_elementwise<4>(
            d_A, lda, d_keys, d_weights, nrows, ncols, d_sums, plan.block_dim, st);
          break;
        }
      }
      sum_rows_by_key_rowmajor_elementwise<1>(
        d_A, lda, d_keys, d_weights, nrows, ncols, d_sums, plan.block_dim, st);
      break;
  }
}

/**
 * @brief Computes the weighted reduction of matrix rows for each given key
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
 * @param[in]  dry_run     dry run mode
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
  typedef typename std::iterator_traits<KeysIteratorT>::value_type KeyType;
  static_assert(
    !is_floating_point_v<WeightT> ||
      (is_floating_point_v<SumsT> && sizeof(WeightT) <= sizeof(SumsT)),
    "Floating weight type is broader than the sum type. Instead of silently casting down the "
    "weight precision, we throw an error here.");

  // Following kernel needs memset
  if (reset_sums && !dry_run) { cudaMemsetAsync(d_sums, 0, ncols * nkeys * sizeof(SumsT), stream); }

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
 * @param[in]  dry_run     dry run mode
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
