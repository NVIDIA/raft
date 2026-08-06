/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/sparse/solver/lanczos_svds.cuh>

#include <raft_runtime/solver/lanczos_svds.hpp>

#include <optional>

#define FUNC_DEF(ValueType)                                                                       \
  void sparse_lanczos_svd(                                                                        \
    const raft::resources& handle,                                                                \
    const raft::sparse::solver::sparse_lanczos_svd_config<ValueType>& config,                     \
    raft::device_vector_view<int, uint32_t, raft::row_major> indptr,                              \
    raft::device_vector_view<int, uint32_t, raft::row_major> indices,                             \
    raft::device_vector_view<ValueType, uint32_t, raft::row_major> data,                          \
    int n_rows,                                                                                   \
    int n_cols,                                                                                   \
    int nnz,                                                                                      \
    raft::device_vector_view<ValueType, uint32_t> singular_values,                                \
    std::optional<raft::device_matrix_view<ValueType, uint32_t, raft::col_major>> U,              \
    std::optional<raft::device_matrix_view<ValueType, uint32_t, raft::col_major>> Vt)             \
  {                                                                                               \
    auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(              \
      indptr.data_handle(), indices.data_handle(), n_rows, n_cols, nnz);                          \
    auto csr_matrix = raft::make_device_csr_matrix_view<const ValueType, int, int, int>(          \
      data.data_handle(), csr_structure);                                                         \
    raft::sparse::solver::sparse_lanczos_svd(handle, config, csr_matrix, singular_values, U, Vt); \
  }
