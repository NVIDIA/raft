/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/util_ptx.cuh>

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

template <typename T>
struct quad {
  T x, y, z, w;
};

//
// Functor for reduce by key, small k
//
template <typename T>
struct quadSum {
  __host__ __device__ __forceinline__ quad<T> operator()(const quad<T>& a, const quad<T>& b) const
  {
    // wasting a double4..
    quad<T> c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    c.w = a.w + b.w;

    return c;
  }
};

//
// Reduce by keys
// We need to sum each dimension by labels
// The labels are not adjacent
//

//
// Reduce by keys - for keys <= 4
//

#define SUM_ROWS_SMALL_K_DIMX         256
#define SUM_ROWS_BY_KEY_SMALL_K_MAX_K 4
template <typename DataIteratorT, typename WeightT, typename SumsT, typename IdxT>
__launch_bounds__(SUM_ROWS_SMALL_K_DIMX, 4)

  RAFT_KERNEL sum_rows_by_key_small_nkeys_kernel(const DataIteratorT d_A,
                                                 IdxT lda,
                                                 const char* d_keys,
                                                 const WeightT* d_weights,
                                                 IdxT nrows,
                                                 IdxT ncols,
                                                 IdxT nkeys,
                                                 SumsT* d_sums)
{
  typedef typename std::iterator_traits<DataIteratorT>::value_type DataType;
  typedef cub::BlockReduce<quad<SumsT>, SUM_ROWS_SMALL_K_DIMX> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (IdxT idim = static_cast<IdxT>(blockIdx.y); idim < ncols; idim += gridDim.y) {
    if (idim != static_cast<IdxT>(blockIdx.y)) __syncthreads();  // we're reusing temp_storage

    // threadIdx.x stores partial sum for current dim and key=threadIdx.x in this reg
    quad<SumsT> thread_sums;
    thread_sums.x = 0.0;
    thread_sums.y = 0.0;
    thread_sums.z = 0.0;
    thread_sums.w = 0.0;

    // May use vectorized load - not necessary for doubles
    for (IdxT block_offset_irow = blockIdx.x * blockDim.x;
         block_offset_irow < nrows;  // we will syncthreads() inside the loop, no CTA divergence
         block_offset_irow += blockDim.x * gridDim.x) {
      IdxT irow    = block_offset_irow + threadIdx.x;
      DataType val = (irow < nrows) ? d_A[irow * lda + idim] : 0.0;
      if (d_weights && irow < nrows) { val = val * d_weights[irow]; }
      // we are not reusing the keys - after profiling
      // d_keys is mainly loaded from L2, and this kernel is DRAM BW bounded
      // (experimentation gave a 10% speed up - not worth the many code lines added)
      IdxT row_key = (irow < nrows) ? d_keys[irow] : std::numeric_limits<IdxT>::max();

      thread_sums.x += (row_key == 0) ? static_cast<SumsT>(val) : 0.0;
      thread_sums.y += (row_key == 1) ? static_cast<SumsT>(val) : 0.0;
      thread_sums.z += (row_key == 2) ? static_cast<SumsT>(val) : 0.0;
      thread_sums.w += (row_key == 3) ? static_cast<SumsT>(val) : 0.0;
    }

    // End of column
    // Saving local sums back to global mem

    // Strided access

    // Reducing by key
    thread_sums = BlockReduce(temp_storage).Reduce(thread_sums, quadSum<SumsT>());

    if (threadIdx.x < 32) {
      // We only need 4
      thread_sums = cub::ShuffleIndex<32>(thread_sums, 0, 0xffffffff);
      if (static_cast<IdxT>(threadIdx.x) < nkeys) {
        if (threadIdx.x == 0) raft::myAtomicAdd(&d_sums[threadIdx.x * ncols + idim], thread_sums.x);
        if (threadIdx.x == 1) raft::myAtomicAdd(&d_sums[threadIdx.x * ncols + idim], thread_sums.y);
        if (threadIdx.x == 2) raft::myAtomicAdd(&d_sums[threadIdx.x * ncols + idim], thread_sums.z);
        if (threadIdx.x == 3) raft::myAtomicAdd(&d_sums[threadIdx.x * ncols + idim], thread_sums.w);
      }
    }
  }
}

template <typename DataIteratorT, typename WeightT, typename SumsT, typename IdxT>
void sum_rows_by_key_small_nkeys(const DataIteratorT d_A,
                                 IdxT lda,
                                 const char* d_keys,
                                 const WeightT* d_weights,
                                 IdxT nrows,
                                 IdxT ncols,
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
                      sum_rows_by_key_small_nkeys_kernel,
                      d_A,
                      lda,
                      d_keys,
                      d_weights,
                      nrows,
                      ncols,
                      nkeys,
                      d_sums);
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
 * @param[out] d_keys_char Scratch memory for conversion of keys to char
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
                        char* d_keys_char,
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

  if (d_keys_char != nullptr && nkeys <= SUM_ROWS_BY_KEY_SMALL_K_MAX_K) {
    // sum_rows_by_key_small_k is BW bounded. d_keys is loaded ncols time - avoiding wasting BW
    // with doubles we have ~20% speed up - with floats we can hope something around 2x
    // Converting d_keys to char
    convert_array(d_keys_char, d_keys, nrows, stream);
    sum_rows_by_key_small_nkeys(
      d_A, lda, d_keys_char, d_weights, nrows, ncols, nkeys, d_sums, stream);
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
 * @param      d_keys_char Scratch memory for conversion of keys to char
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
