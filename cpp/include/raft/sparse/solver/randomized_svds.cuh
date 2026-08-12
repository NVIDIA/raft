/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/detail/csr_linear_operator.cuh>
#include <raft/sparse/solver/detail/randomized_svds.cuh>
#include <raft/sparse/solver/detail/svds_optional.hpp>

#include <optional>

namespace raft::sparse::solver {

/**
 * @defgroup sparse_randomized_svd Sparse Randomized SVD
 * @{
 */

/**
 * @brief Compute truncated SVD using randomized algorithm with a generic linear operator.
 *
 * Implements randomized SVD (Halko et al. 2009) with CholeskyQR2 orthogonalization
 * (Tomás et al. 2024) for efficient GPU execution on sparse matrices.
 *
 * The operator interface allows implicit linear operators (e.g. mean-centered sparse
 * matrices for PCA) without materializing the dense matrix. `OperatorT` must expose
 * the minimal interface used by the randomized SVD inner loop:
 *   - `int rows() const` / `int cols() const`
 *   - `void apply(handle, X, Y) const`           computes `Y = A @ X`
 *   - `void apply_transpose(handle, X, Z) const` computes `Z = A^T @ X`
 *
 * where `X`, `Y`, `Z` are `raft::device_matrix_view<..., uint32_t, raft::col_major>`.
 * This is intentionally a narrow, SVD-specific operator: only the two matrix products
 * above are required, and no other operations (addition, scaling, inverse, eigensolves,
 * etc.) are assumed or supported; it is not a general-purpose linear operator like
 * `scipy.sparse.linalg.LinearOperator`.
 *
 * @tparam ValueTypeT Data type (float or double)
 * @tparam OperatorT Linear operator type satisfying the interface above
 *
 * @param[in] handle raft resources handle
 * @param[in] config SVD configuration parameters
 * @param[in] op linear operator representing the matrix to decompose
 * @param[out] singular_values output singular values of shape (n_components,) in descending order
 * @param[out] U optional output left singular vectors of shape (m, n_components), col-major.
 *             Pass `std::nullopt` to skip computing U.
 * @param[out] Vt optional output right singular vectors of shape (n_components, n), col-major.
 *             Pass `std::nullopt` to skip computing Vt.
 *
 * @note References:
 *   [1] Halko, Martinsson, Tropp (2009) "Finding structure with randomness"
 *       https://arxiv.org/abs/0909.4061
 *   [2] Tomás, Quintana-Ortí, Anzt (2024) "Fast Truncated SVD of Sparse and Dense Matrices
 *       on Graphics Processors" https://arxiv.org/abs/2403.06218
 */
template <typename ValueTypeT, typename OperatorT>
void sparse_randomized_svd(
  raft::resources const& handle,
  sparse_svd_config<ValueTypeT> const& config,
  OperatorT const& op,
  raft::device_vector_view<ValueTypeT, uint32_t> singular_values,
  detail::nondeduced_optional_matrix_view_t<
    raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> U = std::nullopt,
  detail::nondeduced_optional_matrix_view_t<
    raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> Vt = std::nullopt)
{
  detail::sparse_randomized_svd(handle, config, op, singular_values, U, Vt);
}

/**
 * @brief Compute truncated SVD of a sparse CSR matrix using randomized algorithm.
 *
 * Convenience overload that accepts a CSR matrix view directly.
 *
 * @tparam ValueTypeT Data type (float or double)
 * @tparam NNZTypeT Type for number of non-zeros
 *
 * @param[in] handle raft resources handle
 * @param[in] config SVD configuration parameters
 * @param[in] A input sparse CSR matrix of shape (m, n)
 * @param[out] singular_values output singular values of shape (n_components,) in descending order
 * @param[out] U optional output left singular vectors of shape (m, n_components), col-major.
 *             Pass `std::nullopt` to skip computing U.
 * @param[out] Vt optional output right singular vectors of shape (n_components, n), col-major.
 *             Pass `std::nullopt` to skip computing Vt.
 */
template <typename ValueTypeT, typename NNZTypeT>
void sparse_randomized_svd(
  raft::resources const& handle,
  sparse_svd_config<ValueTypeT> const& config,
  raft::device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT> A,
  raft::device_vector_view<ValueTypeT, uint32_t> singular_values,
  detail::nondeduced_optional_matrix_view_t<
    raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> U = std::nullopt,
  detail::nondeduced_optional_matrix_view_t<
    raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> Vt = std::nullopt)
{
  detail::csr_linear_operator<ValueTypeT, NNZTypeT> op(A);
  detail::sparse_randomized_svd(handle, config, op, singular_values, U, Vt);
}

/** @} */

}  // namespace raft::sparse::solver
