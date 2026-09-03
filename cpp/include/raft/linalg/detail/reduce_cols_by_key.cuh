/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

#include <stdlib.h>

#include <limits>

namespace raft {
namespace linalg {
namespace detail {

///@todo: support col-major
///@todo: specialize this to support shared-mem based atomics

template <typename T, typename KeyIteratorT, typename IdxType>
RAFT_KERNEL reduce_cols_by_key_direct_kernel(
  const T* data, const KeyIteratorT keys, T* out, IdxType nrows, IdxType ncols, IdxType nkeys)
{
  typedef typename std::iterator_traits<KeyIteratorT>::value_type KeyType;

  IdxType idx = static_cast<IdxType>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= (nrows * ncols)) return;
  ///@todo: yikes! use fast-int-div
  IdxType colId = idx % ncols;
  IdxType rowId = idx / ncols;
  KeyType key   = keys[colId];
  raft::myAtomicAdd(out + rowId * nkeys + key, data[idx]);
}

template <typename T, typename KeyIteratorT, typename IdxType>
RAFT_KERNEL reduce_cols_by_key_cached_kernel(
  const T* data, const KeyIteratorT keys, T* out, IdxType nrows, IdxType ncols, IdxType nkeys)
{
  typedef typename std::iterator_traits<KeyIteratorT>::value_type KeyType;
  extern __shared__ char smem[];
  T* out_cache = reinterpret_cast<T*>(smem);

  // Initialize the shared memory accumulators to 0.
  for (IdxType idx = threadIdx.x; idx < nrows * nkeys; idx += blockDim.x) {
    out_cache[idx] = T{0};
  }
  __syncthreads();

  // Accumulate in shared memory
  for (IdxType idx = static_cast<IdxType>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < nrows * ncols;
       idx += blockDim.x * static_cast<IdxType>(gridDim.x)) {
    IdxType colId = idx % ncols;
    IdxType rowId = idx / ncols;
    KeyType key   = keys[colId];
    raft::myAtomicAdd(out_cache + rowId * nkeys + key, data[idx]);
  }

  // Add the shared-memory accumulators to the global results.
  __syncthreads();
  for (IdxType idx = threadIdx.x; idx < nrows * nkeys; idx += blockDim.x) {
    T val = out_cache[idx];
    if (val != T{0}) { raft::myAtomicAdd(out + idx, val); }
  }
}

/**
 * @brief Computes the sum-reduction of matrix columns for each given key
 * @tparam T the input data type (as well as the output reduced matrix)
 * @tparam KeyType data type of the keys
 * @tparam IdxType indexing arithmetic type
 * @param data the input data (dim = nrows x ncols). This is assumed to be in
 * row-major layout
 * @param keys keys array (len = ncols). It is assumed that each key in this
 * array is between [0, nkeys). In case this is not true, the caller is expected
 * to have called make_monotonic primitive to prepare such a contiguous and
 * monotonically increasing keys array.
 * @param out the output reduced matrix along columns (dim = nrows x nkeys).
 * This will be assumed to be in row-major layout
 * @param nrows number of rows in the input data
 * @param ncols number of columns in the input data
 * @param nkeys number of unique keys in the keys array
 * @param stream cuda stream to launch the kernel onto
 * @param reset_sums Whether to reset the output sums to zero before reducing
 */
template <typename T, typename KeyIteratorT, typename IdxType = int>
void reduce_cols_by_key(const T* data,
                        const KeyIteratorT keys,
                        T* out,
                        IdxType nrows,
                        IdxType ncols,
                        IdxType nkeys,
                        cudaStream_t stream,
                        bool reset_sums)
{
  typedef typename std::iterator_traits<KeyIteratorT>::value_type KeyType;

  RAFT_EXPECTS(static_cast<size_t>(nrows) * static_cast<size_t>(ncols) <=
                 static_cast<size_t>(std::numeric_limits<IdxType>::max()),
               "Index type too small to represent indices in the input array.");
  RAFT_EXPECTS(static_cast<size_t>(nrows) * static_cast<size_t>(nkeys) <=
                 static_cast<size_t>(std::numeric_limits<IdxType>::max()),
               "Index type too small to represent indices in the output array.");

  // Memset the output to zero to use atomics-based reduction.
  if (reset_sums) { RAFT_CUDA_TRY(cudaMemsetAsync(out, 0, sizeof(T) * nrows * nkeys, stream)); }

  // The cached version is used when the cache fits in shared memory and the number of input
  // elements is above a threshold (the cached version is slightly slower for small input arrays,
  // and orders of magnitude faster for large input arrays).
  size_t cache_size = static_cast<size_t>(nrows * nkeys) * sizeof(T);
  if (cache_size <= 49152ull && nrows * ncols >= IdxType{8192}) {
    constexpr int TPB = 256;
    int n_sm          = raft::getMultiProcessorCount();
    int target_nblks  = 4 * n_sm;
    int max_nblks     = raft::ceildiv<IdxType>(nrows * ncols, TPB);
    int nblks         = std::min(target_nblks, max_nblks);
    raft::launch_kernel({stream, cache_size},
                        nblks,
                        TPB,
                        reduce_cols_by_key_cached_kernel,
                        data,
                        keys,
                        out,
                        nrows,
                        ncols,
                        nkeys);
  } else {
    constexpr int TPB = 256;
    int nblks         = raft::ceildiv<IdxType>(nrows * ncols, TPB);
    raft::launch_kernel(
      stream, nblks, TPB, reduce_cols_by_key_direct_kernel, data, keys, out, nrows, ncols, nkeys);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
