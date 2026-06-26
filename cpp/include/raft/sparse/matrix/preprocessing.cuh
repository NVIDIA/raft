/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/matrix/detail/preprocessing.cuh>

namespace raft {
namespace sparse::matrix {
/**
 * @brief This function calculate the tf-idf values for each entry in the COO sparse
 * matrix
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] coo_in
 *   Raft container housing a coordinate format matrix representation.
 * @param[out] results
 *    vector that will contain the resulting encoded values, size of nnz.
 */
template <typename ValueType = float, typename IndexType = int>
void encode_tfidf(raft::resources const& handle,
                  raft::device_coo_matrix<ValueType,
                                          IndexType,
                                          IndexType,
                                          IndexType,
                                          raft::device_container_policy,
                                          raft::PRESERVING>& coo_in,
                  raft::device_vector_view<ValueType, int64_t> results)
{
  auto num_cols = coo_in.structure_view().get_n_cols();
  auto num_rows = coo_in.structure_view().get_n_rows();
  auto columns  = coo_in.structure_view().get_cols();
  auto values   = coo_in.view().get_elements();
  auto nnz      = coo_in.structure_view().get_nnz();

  int fullFeatCount = 0;
  auto featIdCount  = raft::make_device_vector<IndexType, int64_t>(handle, num_cols);
  detail::fit_tfidf<ValueType, IndexType>(
    handle, columns.data(), values.data(), num_cols, nnz, featIdCount.view(), fullFeatCount);
  detail::transform_tfidf<ValueType, IndexType>(
    handle, columns.data(), values.data(), num_rows, featIdCount.view(), fullFeatCount, results);
}

/**
 * @brief This function calculate the tf-idf values for each entry in the CSR sparse
 * matrix
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] csr_in
 *   Raft container housing a compressed sparse row matrix representation.
 * @param[out] results
 *    vector that will contain the resulting encoded values, size of nnz.
 */
template <typename ValueType = float, typename IndexType = int>
void encode_tfidf(raft::resources const& handle,
                  raft::device_csr_matrix<ValueType,
                                          IndexType,
                                          IndexType,
                                          IndexType,
                                          raft::device_container_policy,
                                          raft::PRESERVING>& csr_in,
                  raft::device_vector_view<ValueType, int64_t> results)
{
  auto num_cols     = csr_in.structure_view().get_n_cols();
  auto num_rows     = csr_in.structure_view().get_n_rows();
  auto columns      = csr_in.structure_view().get_indices();
  auto values       = csr_in.view().get_elements();
  auto nnz          = csr_in.structure_view().get_nnz();
  int fullFeatCount = 0;
  auto featIdCount  = raft::make_device_vector<IndexType, int64_t>(handle, num_cols);
  detail::fit_tfidf<ValueType, IndexType>(
    handle, columns.data(), values.data(), num_cols, nnz, featIdCount.view(), fullFeatCount);
  detail::transform_tfidf<ValueType, IndexType>(
    handle, columns.data(), values.data(), num_rows, featIdCount.view(), fullFeatCount, results);
}

/**
 * @brief This function calculate the bm25 values for each entry in the CSR sparse
 * matrix
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] csr_in
 *   Raft container housing a compressed sparse row matrix representation.
 * @param[out] results
 *    vector that will contain the resulting encoded values, size of nnz.
 * @param[in] k_param
 *    bm25 okapi optimization parameter k1
 * @param[in] b_param
 *    bm25 okapi optimization parameter b
 */
template <typename ValueType = float, typename IndexType = int>
void encode_bm25(raft::resources const& handle,
                 raft::device_csr_matrix<ValueType,
                                         IndexType,
                                         IndexType,
                                         IndexType,
                                         raft::device_container_policy,
                                         raft::PRESERVING>& csr_in,
                 raft::device_vector_view<ValueType, int64_t> results,
                 float k_param = 1.6f,
                 float b_param = 0.75f)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // auto coo_in = detail::create_coo_from_csr<ValueType, IndexType>(handle, csr_in);
  auto num_cols = csr_in.structure_view().get_n_cols();
  auto num_rows = csr_in.structure_view().get_n_rows();
  auto columns  = csr_in.structure_view().get_indices();
  auto values   = csr_in.view().get_elements();
  auto nnz      = csr_in.structure_view().get_nnz();
  auto indptr   = csr_in.structure_view().get_indptr();

  auto rows         = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  int fullFeatCount = 0;
  auto featIdCount  = raft::make_device_vector<IndexType, int64_t>(handle, num_cols);
  auto rowFeatCnts  = raft::make_device_vector<IndexType, int64_t>(handle, num_rows);

  if (!resource::get_dry_run_flag(handle)) {
    raft::sparse::convert::csr_to_coo(
      indptr.data(), (int)indptr.size(), rows.data_handle(), (int)nnz, stream);
  }
  detail::fit_bm25<ValueType, IndexType>(handle,
                                         rows.data_handle(),
                                         columns.data(),
                                         values.data(),
                                         num_rows,
                                         num_cols,
                                         nnz,
                                         featIdCount.view(),
                                         fullFeatCount,
                                         rowFeatCnts.view());
  detail::transform_bm25<ValueType, IndexType>(handle,
                                               rows.data_handle(),
                                               columns.data(),
                                               values.data(),
                                               num_rows,
                                               featIdCount.view(),
                                               fullFeatCount,
                                               rowFeatCnts.view(),
                                               k_param,
                                               b_param,
                                               results);
}

/**
 * @brief This function calculate the bm25 values for each entry in the COO sparse
 * matrix
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] coo_in
 *   Raft container housing a coordinate format matrix representation.
 * @param[out] results
 *    vector that will contain the resulting encoded values, size of nnz.
 * @param[in] k_param
 *    bm25 okapi optimization parameter k1
 * @param[in] b_param
 *    bm25 okapi optimization parameter b
 */
template <typename ValueType = float, typename IndexType = int>
void encode_bm25(raft::resources const& handle,
                 raft::device_coo_matrix<ValueType,
                                         IndexType,
                                         IndexType,
                                         IndexType,
                                         raft::device_container_policy,
                                         raft::PRESERVING>& coo_in,
                 raft::device_vector_view<ValueType, int64_t> results,
                 float k_param = 1.6f,
                 float b_param = 0.75f)
{
  auto num_cols           = coo_in.structure_view().get_n_cols();
  auto nnz                = coo_in.structure_view().get_nnz();
  auto num_rows           = coo_in.structure_view().get_n_rows();
  auto rows               = coo_in.structure_view().get_rows();
  auto columns            = coo_in.structure_view().get_cols();
  auto values             = coo_in.view().get_elements();
  IndexType fullFeatCount = 0;
  auto featIdCount        = raft::make_device_vector<IndexType, int64_t>(handle, num_cols);
  auto rowFeatCnts        = raft::make_device_vector<IndexType, int64_t>(handle, num_rows);
  detail::fit_bm25<ValueType, IndexType>(handle,
                                         rows.data(),
                                         columns.data(),
                                         values.data(),
                                         num_rows,
                                         num_cols,
                                         nnz,
                                         featIdCount.view(),
                                         fullFeatCount,
                                         rowFeatCnts.view());
  detail::transform_bm25<ValueType, IndexType>(handle,
                                               rows.data(),
                                               columns.data(),
                                               values.data(),
                                               num_rows,
                                               featIdCount.view(),
                                               fullFeatCount,
                                               rowFeatCnts.view(),
                                               k_param,
                                               b_param,
                                               results);
}

}  // namespace sparse::matrix
}  // namespace raft
