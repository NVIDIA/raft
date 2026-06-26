/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __REDUCE_COLS_BY_KEY
#define __REDUCE_COLS_BY_KEY

#pragma once

#include "detail/reduce_cols_by_key.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

namespace raft {
namespace linalg {

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
 * @param reset_sums  Whether to reset the output sums to zero before reducing
 */
template <typename T, typename KeyIteratorT, typename IdxType = int>
void reduce_cols_by_key(const T* data,
                        const KeyIteratorT keys,
                        T* out,
                        IdxType nrows,
                        IdxType ncols,
                        IdxType nkeys,
                        cudaStream_t stream,
                        bool reset_sums = true)
{
  detail::reduce_cols_by_key(data, keys, out, nrows, ncols, nkeys, stream, reset_sums);
}

/**
 * @defgroup reduce_cols_by_key Reduce Across Columns by Key
 * @{
 */

/**
 * @brief Computes the sum-reduction of matrix columns for each given key
 * TODO: Support generic reduction lambdas https://github.com/rapidsai/raft/issues/860
 * @tparam ElementType the input data type (as well as the output reduced matrix)
 * @tparam KeyType data type of the keys
 * @tparam IndexType indexing arithmetic type
 * @param[in] handle raft::resources
 * @param[in] data the input data (dim = nrows x ncols). This is assumed to be in
 * row-major layout of type raft::device_matrix_view
 * @param[in] keys keys raft::device_vector_view (len = ncols). It is assumed that each key in this
 * array is between [0, nkeys). In case this is not true, the caller is expected
 * to have called make_monotonic primitive to prepare such a contiguous and
 * monotonically increasing keys array.
 * @param[out] out the output reduced raft::device_matrix_view along columns (dim = nrows x nkeys).
 * This will be assumed to be in row-major layout
 * @param[in] nkeys Number of unique keys in the keys array. By default, inferred from the number of
 * columns of out
 * @param[in] reset_sums  Whether to reset the output sums to zero before reducing
 */
template <typename ElementType, typename KeyType = ElementType, typename IndexType = std::uint32_t>
void reduce_cols_by_key(
  raft::resources const& handle,
  raft::device_matrix_view<const ElementType, IndexType, raft::row_major> data,
  raft::device_vector_view<const KeyType, IndexType> keys,
  raft::device_matrix_view<ElementType, IndexType, raft::row_major> out,
  IndexType nkeys = 0,
  bool reset_sums = true)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  if (nkeys > 0) {
    RAFT_EXPECTS(out.extent(1) == nkeys, "Output doesn't have nkeys columns");
  } else {
    nkeys = out.extent(1);
  }
  RAFT_EXPECTS(out.extent(0) == data.extent(0),
               "Output doesn't have the same number of rows as input");
  RAFT_EXPECTS(keys.extent(0) == data.extent(1), "Keys is not of size ncols");

  reduce_cols_by_key(data.data_handle(),
                     keys.data_handle(),
                     out.data_handle(),
                     data.extent(0),
                     data.extent(1),
                     nkeys,
                     resource::get_cuda_stream(handle),
                     reset_sums);
}

/** @} */  // end of group reduce_cols_by_key

};  // end namespace linalg
};  // end namespace raft

#endif
