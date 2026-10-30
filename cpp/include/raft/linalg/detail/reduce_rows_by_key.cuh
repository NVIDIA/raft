/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/util_ptx.cuh>

#include <algorithm>
#include <limits>

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

//
// Reduce by keys - for keys <= 4
//
// Each block sums a tile of TileCols columns over a chunk of rows. Thread t reads column
// t % TileCols of rows t / TileCols, t / TileCols + RowLanes, ..., so the loads of a warp are
// coalesced on row-major data, and keeps one running sum per key in registers. The block then
// reduces them over its RowLanes row lanes in shared memory, which leaves one global atomic per
// key and column for each block.
//

#define SUM_ROWS_SMALL_K_DIMX         256
#define SUM_ROWS_BY_KEY_SMALL_K_MAX_K 4
template <int TileCols,
          typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
__launch_bounds__(SUM_ROWS_SMALL_K_DIMX)

  RAFT_KERNEL sum_rows_by_key_small_nkeys_kernel(const DataIteratorT d_A,
                                                 IdxT lda,
                                                 KeysIteratorT d_keys,
                                                 const WeightT* d_weights,
                                                 IdxT nrows,
                                                 IdxT ncols,
                                                 IdxT nkeys,
                                                 IdxT rows_per_block,
                                                 SumsT* d_sums)
{
  constexpr int RowLanes = SUM_ROWS_SMALL_K_DIMX / TileCols;
  const int tile_col     = threadIdx.x % TileCols;
  const int row_lane     = threadIdx.x / TileCols;
  const IdxT col         = static_cast<IdxT>(blockIdx.x) * TileCols + tile_col;
  const IdxT row_start   = static_cast<IdxT>(blockIdx.y) * rows_per_block;
  const IdxT row_end     = raft::min(nrows, row_start + rows_per_block);

  SumsT sums[SUM_ROWS_BY_KEY_SMALL_K_MAX_K] = {};
  if (col < ncols) {
#pragma unroll 4
    for (IdxT irow = row_start + row_lane; irow < row_end; irow += RowLanes) {
      const IdxT key = static_cast<IdxT>(d_keys[irow]);
      SumsT val      = d_A[irow * lda + col];
      if (d_weights != nullptr) val *= d_weights[irow];
#pragma unroll
      for (int k = 0; k < SUM_ROWS_BY_KEY_SMALL_K_MAX_K; k++) {
        sums[k] += (key == static_cast<IdxT>(k)) ? val : SumsT(0);
      }
    }
  }

  __shared__ SumsT block_sums[SUM_ROWS_BY_KEY_SMALL_K_MAX_K][RowLanes][TileCols];
#pragma unroll
  for (int k = 0; k < SUM_ROWS_BY_KEY_SMALL_K_MAX_K; k++) {
    block_sums[k][row_lane][tile_col] = sums[k];
  }
  __syncthreads();
  for (int stride = RowLanes / 2; stride > 0; stride /= 2) {
    if (row_lane < stride) {
#pragma unroll
      for (int k = 0; k < SUM_ROWS_BY_KEY_SMALL_K_MAX_K; k++) {
        block_sums[k][row_lane][tile_col] += block_sums[k][row_lane + stride][tile_col];
      }
    }
    __syncthreads();
  }

  if (row_lane == 0 && col < ncols) {
    for (IdxT k = 0; k < nkeys; k++) {
      raft::myAtomicAdd(&d_sums[k * ncols + col], block_sums[k][0][tile_col]);
    }
  }
}

template <int TileCols,
          typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void sum_rows_by_key_small_nkeys_tiled(const DataIteratorT d_A,
                                       IdxT lda,
                                       KeysIteratorT d_keys,
                                       const WeightT* d_weights,
                                       IdxT nrows,
                                       IdxT ncols,
                                       IdxT nkeys,
                                       SumsT* d_sums,
                                       cudaStream_t st)
{
  constexpr IdxT RowLanes = SUM_ROWS_SMALL_K_DIMX / TileCols;
  auto kernel             = sum_rows_by_key_small_nkeys_kernel<TileCols,
                                                               DataIteratorT,
                                                               KeysIteratorT,
                                                               WeightT,
                                                               SumsT,
                                                               IdxT>;
  int blocks_per_sm       = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, kernel, SUM_ROWS_SMALL_K_DIMX, 0));

  // Two waves of blocks: the column tiles first, then enough row chunks to fill the GPU.
  const IdxT n_tiles  = raft::ceildiv(ncols, static_cast<IdxT>(TileCols));
  const IdxT n_blocks = 2 * static_cast<IdxT>(blocks_per_sm) * raft::getMultiProcessorCount();
  IdxT n_chunks       = std::min(raft::ceildiv(n_blocks, n_tiles), raft::ceildiv(nrows, RowLanes));
  n_chunks            = std::clamp(n_chunks, IdxT(1), static_cast<IdxT>(MAX_BLOCKS));
  const IdxT rows_per_block = raft::ceildiv(raft::ceildiv(nrows, n_chunks), RowLanes) * RowLanes;
  n_chunks                  = raft::ceildiv(nrows, rows_per_block);

  dim3 grid(n_tiles, n_chunks);
  raft::launch_kernel(st,
                      grid,
                      SUM_ROWS_SMALL_K_DIMX,
                      kernel,
                      d_A,
                      lda,
                      d_keys,
                      d_weights,
                      nrows,
                      ncols,
                      nkeys,
                      rows_per_block,
                      d_sums);
}

template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void sum_rows_by_key_small_nkeys(const DataIteratorT d_A,
                                 IdxT lda,
                                 KeysIteratorT d_keys,
                                 const WeightT* d_weights,
                                 IdxT nrows,
                                 IdxT ncols,
                                 IdxT nkeys,
                                 SumsT* d_sums,
                                 cudaStream_t st)
{
  if (nrows == 0 || ncols == 0) return;
  // Narrow tiles for narrow rows, so that the lanes of a warp still read contiguous memory.
  if (ncols <= 1) {
    sum_rows_by_key_small_nkeys_tiled<1>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  } else if (ncols <= 2) {
    sum_rows_by_key_small_nkeys_tiled<2>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  } else if (ncols <= 4) {
    sum_rows_by_key_small_nkeys_tiled<4>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  } else if (ncols <= 8) {
    sum_rows_by_key_small_nkeys_tiled<8>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  } else if (ncols <= 16) {
    sum_rows_by_key_small_nkeys_tiled<16>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  } else if (ncols <= 32) {
    sum_rows_by_key_small_nkeys_tiled<32>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  } else if (ncols <= 64) {
    sum_rows_by_key_small_nkeys_tiled<64>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  } else if (ncols <= 128) {
    sum_rows_by_key_small_nkeys_tiled<128>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  } else {
    sum_rows_by_key_small_nkeys_tiled<256>(
      d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, st);
  }
}

//
// Reduce by keys - large number of keys
// Computing a "weighted histogram" with local histograms in smem
// Keeping it simple - not optimized
//

#define SUM_ROWS_BY_KEY_LARGE_K_MAX_K 1024

template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
RAFT_KERNEL sum_rows_by_key_large_nkeys_kernel_colmajor(const DataIteratorT d_A,
                                                        IdxT lda,
                                                        KeysIteratorT d_keys,
                                                        const WeightT* d_weights,
                                                        IdxT nrows,
                                                        IdxT ncols,
                                                        int key_offset,
                                                        IdxT nkeys,
                                                        SumsT* d_sums)
{
  typedef typename std::iterator_traits<KeysIteratorT>::value_type KeyType;
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;
  __shared__ SumsT local_sums[SUM_ROWS_BY_KEY_LARGE_K_MAX_K];

  for (IdxT local_key = threadIdx.x; local_key < nkeys; local_key += blockDim.x)
    local_sums[local_key] = 0.0;

  for (IdxT idim = blockIdx.y; idim < ncols; idim += gridDim.y) {
    __syncthreads();  // local_sums

    // At this point local_sums if full of zeros

    for (IdxT irow = blockIdx.x * blockDim.x + threadIdx.x; irow < nrows;
         irow += blockDim.x * gridDim.x) {
      // Branch div in this loop - not an issue with current code
      DataType val = d_A[idim * lda + irow];
      if (d_weights) val = val * d_weights[irow];

      IdxT local_key = d_keys[irow] - key_offset;

      // We could load next val here
      raft::myAtomicAdd(&local_sums[local_key], static_cast<SumsT>(val));
    }

    __syncthreads();  // local_sums

    for (IdxT local_key = threadIdx.x; local_key < nkeys; local_key += blockDim.x) {
      SumsT local_sum = local_sums[local_key];

      if (local_sum != 0.0) {
        KeyType global_key = key_offset + local_key;
        raft::myAtomicAdd(&d_sums[global_key * ncols + idim], local_sum);
        local_sums[local_key] = 0.0;
      }
    }
  }
}

template <typename DataIteratorT, typename KeysIteratorT, typename SumsT, typename IdxT>
void sum_rows_by_key_large_nkeys_colmajor(const DataIteratorT d_A,
                                          IdxT lda,
                                          KeysIteratorT d_keys,
                                          IdxT nrows,
                                          IdxT ncols,
                                          int key_offset,
                                          IdxT nkeys,
                                          SumsT* d_sums,
                                          cudaStream_t st)
{
  dim3 grid, block;
  block.x = SUM_ROWS_SMALL_K_DIMX;
  block.y = 1;  // Necessary

  grid.x = raft::ceildiv(nrows, (IdxT)block.x);
  grid.x = std::min(grid.x, 32u);
  grid.y = ncols;
  grid.y = std::min(grid.y, MAX_BLOCKS);
  raft::launch_kernel(st,
                      grid,
                      block,
                      sum_rows_by_key_large_nkeys_kernel_colmajor,
                      d_A,
                      lda,
                      d_keys,
                      nrows,
                      ncols,
                      key_offset,
                      nkeys,
                      d_sums);
}

template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
RAFT_KERNEL sum_rows_by_key_large_nkeys_kernel_rowmajor(const DataIteratorT d_A,
                                                        IdxT lda,
                                                        const WeightT* d_weights,
                                                        KeysIteratorT d_keys,
                                                        IdxT nrows,
                                                        IdxT ncols,
                                                        SumsT* d_sums)
{
  IdxT gid = threadIdx.x + (blockDim.x * static_cast<IdxT>(blockIdx.x));
  IdxT j   = gid % ncols;
  IdxT i   = gid / ncols;
  if (i >= nrows) return;
  IdxT l    = static_cast<IdxT>(d_keys[i]);
  SumsT val = d_A[j + lda * i];
  if (d_weights != nullptr) val *= d_weights[i];
  raft::myAtomicAdd(&d_sums[j + ncols * l], val);
}

template <typename DataIteratorT,
          typename KeysIteratorT,
          typename WeightT,
          typename SumsT,
          typename IdxT>
void sum_rows_by_key_large_nkeys_rowmajor(const DataIteratorT d_A,
                                          IdxT lda,
                                          const KeysIteratorT d_keys,
                                          const WeightT* d_weights,
                                          IdxT nrows,
                                          IdxT ncols,
                                          SumsT* d_sums,
                                          cudaStream_t st)
{
  uint32_t block_dim = 128;
  auto grid_dim      = static_cast<uint32_t>(ceildiv<IdxT>(nrows * ncols, (IdxT)block_dim));
  raft::launch_kernel(st,
                      grid_dim,
                      block_dim,
                      sum_rows_by_key_large_nkeys_kernel_rowmajor,
                      d_A,
                      lda,
                      d_weights,
                      d_keys,
                      nrows,
                      ncols,
                      d_sums);
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
 * @param[out] d_keys_char Unused, kept for API compatibility
 * @param[in]  nrows       Number of rows in d_A and d_keys
 * @param[in]  ncols       Number of data columns in d_A
 * @param[in]  nkeys       Number of unique keys in d_keys
 * @param[out] d_sums      Row sums by key (ncols x d_keys)
 * @param[in]  stream      CUDA stream
 * @param[in]  reset_sums  Whether to reset the output sums to zero before reducing
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
                        [[maybe_unused]] char* d_keys_char,
                        IdxT nrows,
                        IdxT ncols,
                        IdxT nkeys,
                        SumsT* d_sums,
                        cudaStream_t stream,
                        bool reset_sums)
{
  typedef typename std::iterator_traits<KeysIteratorT>::value_type KeyType;

  // Following kernel needs memset
  if (reset_sums) { cudaMemsetAsync(d_sums, 0, ncols * nkeys * sizeof(SumsT), stream); }

  if (nkeys <= SUM_ROWS_BY_KEY_SMALL_K_MAX_K) {
    sum_rows_by_key_small_nkeys(d_A, lda, d_keys, d_weights, nrows, ncols, nkeys, d_sums, stream);
  } else {
    sum_rows_by_key_large_nkeys_rowmajor(d_A, lda, d_keys, d_weights, nrows, ncols, d_sums, stream);
  }
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
 * @param      d_keys_char Unused, kept for API compatibility
 * @param[in]  nrows       Number of rows in d_A and d_keys
 * @param[in]  ncols       Number of data columns in d_A
 * @param[in]  nkeys       Number of unique keys in d_keys
 * @param[out] d_sums      Row sums by key (ncols x d_keys)
 * @param[in]  stream      CUDA stream
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
                        bool reset_sums)
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
                     reset_sums);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
