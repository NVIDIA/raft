/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/copy.cuh>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/map_reduce.cuh>
#include <raft/stats/histogram.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/reduce.h>

namespace raft {
namespace sparse::matrix::detail {

/**
 * @brief Get unique counts
 * @param handle[in] raft resource handle
 * @param rows[in] Input COO array rows, of size nnz.
 * @param columns[in] Input COO columns, of size nnz.
 * @param values[in] Input COO values array, of size nnz.
 * @param nnz[in] Size of the COO input arrays.
 * @param keys_out[out] Output array with one entry for each key. (same size as counts_out)
 * @param counts_out[out] Output array with cumulative sum for each key. (same size as keys_out)
 */
template <typename IndexType, typename ValueType, typename IdxT>
void get_uniques_counts(raft::resources const& handle,
                        IndexType* rows,
                        IndexType* columns,
                        ValueType* values,
                        IndexType nnz,
                        raft::device_vector_view<IndexType, int64_t> keys_out,
                        raft::device_vector_view<ValueType, int64_t> counts_out)
{
  if (resource::get_dry_run_flag(handle)) {
    // Upper bound for thrust::reduce_by_key internal workspace
    rmm::device_uvector<char> reduce_ws(static_cast<size_t>(nnz) * sizeof(IndexType) + 4096,
                                        resource::get_cuda_stream(handle));
    return;
  }
  thrust::reduce_by_key(raft::resource::get_thrust_policy(handle),
                        rows,
                        rows + nnz,
                        values,
                        keys_out.data_handle(),
                        counts_out.data_handle());
}

/**
 * @brief This function counts the number of occurrences per feature(column) and records the total
 * number of features
 * @param handle[in] raft resource handle
 * @param columns[in] Input COO columns, of size nnz.
 * @param values[in] Input COO values array, of size nnz.
 * @param num_cols[in] total number of columns in the matrix.
 * @param nnz[in] Size of the COO input arrays.
 * @param idFeatCount[out] Output array holding the occurrences per feature for matrix, size of
 * num_cols.
 * @param fullFeatCount[out] Output value corresponding to total number of features in matrix.
 */
template <typename ValueType = float, typename IndexType = int>
void fit_tfidf(raft::resources const& handle,
               IndexType* columns,
               ValueType* values,
               IndexType num_cols,
               IndexType nnz,
               raft::device_vector_view<IndexType, int64_t> idFeatCount,
               int& fullFeatCount)
{
  auto batchIdLen = raft::make_host_scalar<ValueType>(handle, 0);
  auto values_mat = raft::make_device_scalar<ValueType>(handle, 0);

  raft::stats::histogram(
    handle,
    raft::stats::HistTypeAuto,
    raft::make_device_matrix_view<const IndexType, IndexType, raft::col_major>(columns, nnz, 1),
    raft::make_device_matrix_view<int, IndexType, raft::col_major>(
      idFeatCount.data_handle(), num_cols, 1),
    raft::stats::IdentityBinner<IndexType, IndexType>());

  raft::linalg::map_reduce(handle,
                           raft::make_device_vector_view<const ValueType, IndexType>(values, nnz),
                           values_mat.view(),
                           ValueType{0},
                           raft::identity_op(),
                           raft::add_op());

  raft::copy(handle, batchIdLen.view(), raft::make_const_mdspan(values_mat.view()));
  if (resource::get_dry_run_flag(handle)) { return; }
  fullFeatCount += (int)batchIdLen(0);
}

/**
 * @brief This function counts the number of occurrences per feature(column) and records the total
 * number of features and calculates the per row feature occurrences.
 * @param handle[in] raft resource handle
 * @param rows[in] Input COO rows, of size nnz.
 * @param columns[in] Input COO columns, of size nnz.
 * @param values[in] Input COO values array, of size nnz.
 * @param num_rows[in] total number of rows in the matrix.
 * @param num_cols[in] total number of columns in the matrix.
 * @param nnz[in] Size of the COO input arrays.
 * @param idFeatCount[out] Output array holding the occurrences per feature for matrix, size of
 * num_cols.
 * @param fullFeatCount[out] Output value corresponding to total number of features in matrix.
 * @param rowFeatCnts[out] Output array holding the feature occurrences per row for matrix, size of
 * num_rows.
 */
template <typename ValueType = float, typename IndexType = int>
void fit_bm25(raft::resources const& handle,
              IndexType* rows,
              IndexType* columns,
              ValueType* values,
              IndexType num_rows,
              IndexType num_cols,
              IndexType nnz,
              raft::device_vector_view<IndexType, int64_t> idFeatCount,
              int& fullFeatCount,
              raft::device_vector_view<IndexType, int64_t> rowFeatCnts)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<IndexType> temp_unique_rows(0, stream);
  int uniq_cnt   = raft::label::getUniquelabels(handle, temp_unique_rows, rows, nnz);
  auto row_keys  = raft::make_device_vector<IndexType>(handle, uniq_cnt);
  auto row_cnts  = raft::make_device_vector<ValueType>(handle, uniq_cnt);
  auto dummy_vec = raft::make_device_vector<IndexType>(handle, uniq_cnt);

  get_uniques_counts<IndexType, ValueType, int64_t>(
    handle, rows, columns, values, nnz, row_keys.view(), row_cnts.view());

  raft::linalg::map(
    handle,
    dummy_vec.view(),
    [=] __device__(const IndexType& key, const ValueType& val) {
      rowFeatCnts.data_handle()[key] = val;
      return 0;
    },
    raft::make_const_mdspan(row_keys.view()),
    raft::make_const_mdspan(row_cnts.view()));

  fit_tfidf(handle, columns, values, num_cols, nnz, idFeatCount, fullFeatCount);
}

/**
 * @brief This function calculate the BM25 value for the COO records of the matrix.
 * @param handle[in] raft resource handle
 * @param rows[in] Input COO rows, of size nnz.
 * @param columns[in] Input COO columns, of size nnz.
 * @param values[in] Input COO values array, of size nnz.
 * @param num_rows[in] total number of rows in the matrix.
 * @param idFeatCount[in] array holding the occurrences per feature for matrix, size of num_cols.
 * @param fullFeatCount[in] value corresponding to total number of features in matrix.
 * @param rowFeatCnts[in] array holding the feature occurrences per row for matrix, size of
 * num_rows.
 * @param k_param[in] bm25 okapi optimization parameter k1.
 * @param b_param[in] bm25 okapi optimization parameter b.
 * @param results[out] Output array with values of the bm25 transform for each coo entry.
 */
template <typename ValueType = float, typename IndexType = int>
void transform_bm25(raft::resources const& handle,
                    IndexType* rows,
                    IndexType* columns,
                    ValueType* values,
                    IndexType num_rows,
                    raft::device_vector_view<IndexType, int64_t> featIdCount,
                    IndexType fullIdLen,
                    raft::device_vector_view<IndexType, int64_t> rowFeatCnts,
                    float k_param,
                    float b_param,
                    raft::device_vector_view<ValueType, int64_t> results)
{
  float avgIdLen = (ValueType)fullIdLen / num_rows;
  raft::linalg::map_offset(handle, results, [=] __device__(IndexType idx) {
    ValueType tf     = raft::log<ValueType>(values[idx]);
    ValueType idf_in = static_cast<ValueType>(num_rows) / featIdCount.data_handle()[columns[idx]];
    ValueType idf    = raft::log<ValueType>(idf_in + 1);
    ValueType bm =
      ((k_param + 1) * tf) /
      (k_param * ((1.0f - b_param) + b_param * (rowFeatCnts.data_handle()[rows[idx]] / avgIdLen)) +
       tf);
    return idf * bm;
  });
}

/**
 * @brief This function calculate the tf-idf value for the COO records of the matrix.
 * @param handle[in] raft resource handle
 * @param rows[in] Input COO rows, of size nnz.
 * @param columns[in] Input COO columns, of size nnz.
 * @param values[in] Input COO values array, of size nnz.
 * @param num_rows[in] total number of rows in the matrix.
 * @param idFeatCount[in] array holding the occurrences per feature for matrix, size of num_cols.
 * @param fullFeatCount[in] value corresponding to total number of features in matrix.
 * @param results[out] Output array with values of the bm25 transform for each coo entry.
 */
template <typename ValueType = float, typename IndexType = int>
void transform_tfidf(raft::resources const& handle,
                     IndexType* columns,
                     ValueType* values,
                     IndexType num_rows,
                     raft::device_vector_view<IndexType, int64_t> featIdCount,
                     IndexType fullIdLen,
                     raft::device_vector_view<ValueType, int64_t> results)
{
  raft::linalg::map_offset(handle, results, [=] __device__(IndexType idx) {
    ValueType tf     = raft::log<ValueType>(values[idx]);
    ValueType idf_in = static_cast<ValueType>(num_rows) / featIdCount[columns[idx]];
    ValueType idf    = raft::log<ValueType>(idf_in + 1);
    return tf * idf;
  });
}

}  // namespace sparse::matrix::detail
}  // namespace raft
