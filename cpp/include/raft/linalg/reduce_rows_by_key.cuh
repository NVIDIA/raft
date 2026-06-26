/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __REDUCE_ROWS_BY_KEY
#define __REDUCE_ROWS_BY_KEY

#pragma once

#include "detail/reduce_rows_by_key.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

namespace raft {
namespace linalg {

/**
 Small helper function to convert from int->char and char->int
 Transform ncols*nrows read of int in 2*nrows reads of int + ncols*rows reads of chars
**/
template <typename IteratorT1, typename IteratorT2>
void convert_array(IteratorT1 dst, IteratorT2 src, int n, cudaStream_t st)
{
  detail::convert_array(dst, src, n, st);
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
                        const KeysIteratorT d_keys,
                        const WeightT* d_weights,
                        char* d_keys_char,
                        IdxT nrows,
                        IdxT ncols,
                        IdxT nkeys,
                        SumsT* d_sums,
                        cudaStream_t stream,
                        bool reset_sums = true)
{
  detail::reduce_rows_by_key(
    d_A, lda, d_keys, d_weights, d_keys_char, nrows, ncols, nkeys, d_sums, stream, reset_sums);
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
 * @param[in]  reset_sums  Whether to reset the output sums to zero before reducing
 */
template <typename DataIteratorT, typename KeysIteratorT, typename SumsT, typename IdxT>
void reduce_rows_by_key(const DataIteratorT d_A,
                        IdxT lda,
                        const KeysIteratorT d_keys,
                        char* d_keys_char,
                        IdxT nrows,
                        IdxT ncols,
                        IdxT nkeys,
                        SumsT* d_sums,
                        cudaStream_t stream,
                        bool reset_sums = true)
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

/**
 * @defgroup reduce_rows_by_key Reduce Across Rows by Key
 * @{
 */

/**
 * @brief Computes the weighted sum-reduction of matrix rows for each given key
 * TODO: Support generic reduction lambdas https://github.com/rapidsai/raft/issues/860
 * @tparam ElementType data-type of input and output
 * @tparam KeyType data-type of keys
 * @tparam WeightType data-type of weights
 * @tparam IndexType index type
 * @param[in]  handle      raft::resources
 * @param[in]  d_A         Input raft::device_mdspan (ncols * nrows)
 * @param[in]  d_keys      Keys for each row raft::device_vector_view (1 x nrows)
 * @param[out] d_sums      Row sums by key raft::device_matrix_view (ncols x d_keys)
 * @param[in]  n_unique_keys       Number of unique keys in d_keys
 * @param[out] d_keys_char Scratch memory for conversion of keys to char, raft::device_vector_view
 * @param[in]  d_weights   Weights for each observation in d_A raft::device_vector_view optional (1
 * x nrows)
 * @param[in]  reset_sums  Whether to reset the output sums to zero before reducing
 */
template <typename ElementType, typename KeyType, typename WeightType, typename IndexType>
void reduce_rows_by_key(
  raft::resources const& handle,
  raft::device_matrix_view<const ElementType, IndexType, raft::row_major> d_A,
  raft::device_vector_view<const KeyType, IndexType> d_keys,
  raft::device_matrix_view<ElementType, IndexType, raft::row_major> d_sums,
  IndexType n_unique_keys,
  raft::device_vector_view<char, IndexType> d_keys_char,
  std::optional<raft::device_vector_view<const WeightType, IndexType>> d_weights = std::nullopt,
  bool reset_sums                                                                = true)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(d_A.extent(0) == d_A.extent(0) && d_sums.extent(1) == n_unique_keys,
               "Output is not of size ncols * n_unique_keys");
  RAFT_EXPECTS(d_keys.extent(0) == d_A.extent(1), "Keys is not of size nrows");

  if (d_weights) {
    RAFT_EXPECTS(d_weights.value().extent(0) == d_A.extent(1), "Weights is not of size nrows");

    reduce_rows_by_key(d_A.data_handle(),
                       d_A.extent(0),
                       d_keys.data_handle(),
                       d_weights.value().data_handle(),
                       d_keys_char.data_handle(),
                       d_A.extent(1),
                       d_A.extent(0),
                       n_unique_keys,
                       d_sums.data_handle(),
                       resource::get_cuda_stream(handle),
                       reset_sums);
  } else {
    reduce_rows_by_key(d_A.data_handle(),
                       d_A.extent(0),
                       d_keys.data_handle(),
                       d_keys_char.data_handle(),
                       d_A.extent(1),
                       d_A.extent(0),
                       n_unique_keys,
                       d_sums.data_handle(),
                       resource::get_cuda_stream(handle),
                       reset_sums);
  }
}

/** @} */  // end of group reduce_rows_by_key

};  // end namespace linalg
};  // end namespace raft

#endif
