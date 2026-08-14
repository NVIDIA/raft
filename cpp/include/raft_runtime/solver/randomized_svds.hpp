/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/solver_types.hpp>

#include <cstdint>
#include <optional>

namespace raft::runtime::solver {

/**
 * @defgroup sparse_randomized_svd_runtime Sparse Randomized SVD Runtime API
 * @{
 */

#define FUNC_DECL(ValueType)                                                         \
  RAFT_EXPORT void sparse_randomized_svd(                                            \
    const raft::resources& handle,                                                   \
    const raft::sparse::solver::sparse_svd_config<ValueType>& config,                \
    raft::device_vector_view<int, uint32_t, raft::row_major> indptr,                 \
    raft::device_vector_view<int, uint32_t, raft::row_major> indices,                \
    raft::device_vector_view<ValueType, uint32_t, raft::row_major> data,             \
    int n_rows,                                                                      \
    int n_cols,                                                                      \
    int nnz,                                                                         \
    raft::device_vector_view<ValueType, uint32_t> singular_values,                   \
    std::optional<raft::device_matrix_view<ValueType, uint32_t, raft::col_major>> U, \
    std::optional<raft::device_matrix_view<ValueType, uint32_t, raft::col_major>> Vt)

FUNC_DECL(float);
FUNC_DECL(double);

#undef FUNC_DECL

/** @} */

}  // namespace raft::runtime::solver
