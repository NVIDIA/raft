/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/detail/csr_linear_operator.cuh>
#include <raft/sparse/solver/detail/lanczos_svds.cuh>
#include <raft/sparse/solver/detail/svds_optional.hpp>

#include <optional>

namespace raft::sparse::solver {

/**
 * @defgroup sparse_lanczos_svd Sparse Lanczos SVD
 * @{
 */

/**
 * @brief Compute truncated SVD using Lanczos bidiagonalization with a generic linear operator.
 *
 * This solver computes the largest singular triplets using implicitly restarted Lanczos
 * bidiagonalization, full reorthogonalization, and final `A @ V` post-refinement of
 * returned left singular vectors. It is intended as the higher-accuracy sparse SVD path.
 * The operator interface matches sparse_randomized_svd and supports implicit operators
 * such as centered sparse matrices.
 *
 * By default, the solver uses two-pass classical Gram-Schmidt (CGS2), which is efficient
 * on GPUs. The configuration can request two-pass modified Gram-Schmidt (MGS2) as an
 * alternate path for difficult spectra. If the requested components do not converge
 * within `config.max_iterations`, this function raises an exception instead of returning
 * partially converged vectors.
 *
 * `OperatorT` must expose:
 *   - `int rows() const` / `int cols() const`
 *   - `void apply(handle, X, Y) const` computes `Y = A @ X`
 *   - `void apply_transpose(handle, X, Z) const` computes `Z = A^T @ X`
 *
 * @tparam ValueTypeT Data type (float or double)
 * @tparam OperatorT Linear operator type satisfying the interface above
 *
 * @param[in] handle raft resources handle
 * @param[in] config SVD configuration parameters
 * @param[in] op linear operator representing the matrix to decompose
 * @param[out] singular_values output singular values of shape (n_components,) in descending order
 * @param[out] U optional output left singular vectors of shape (m, n_components), col-major.
 *             Pass `std::nullopt` to skip storing U.
 * @param[out] Vt optional output right singular vectors of shape (n_components, n), col-major.
 *             Pass `std::nullopt` to skip storing Vt.
 */
template <typename ValueTypeT, typename OperatorT>
void sparse_lanczos_svd(
  raft::resources const& handle,
  sparse_lanczos_svd_config<ValueTypeT> const& config,
  OperatorT const& op,
  raft::device_vector_view<ValueTypeT, uint32_t> singular_values,
  detail::nondeduced_optional_matrix_view_t<
    raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> U = std::nullopt,
  detail::nondeduced_optional_matrix_view_t<
    raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> Vt = std::nullopt)
{
  detail::sparse_lanczos_svd(handle, config, op, singular_values, U, Vt);
}

/**
 * @brief Compute truncated SVD of a sparse CSR matrix using Lanczos bidiagonalization.
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
 *             Pass `std::nullopt` to skip storing U.
 * @param[out] Vt optional output right singular vectors of shape (n_components, n), col-major.
 *             Pass `std::nullopt` to skip storing Vt.
 */
template <typename ValueTypeT, typename NNZTypeT>
void sparse_lanczos_svd(
  raft::resources const& handle,
  sparse_lanczos_svd_config<ValueTypeT> const& config,
  raft::device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT> A,
  raft::device_vector_view<ValueTypeT, uint32_t> singular_values,
  detail::nondeduced_optional_matrix_view_t<
    raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> U = std::nullopt,
  detail::nondeduced_optional_matrix_view_t<
    raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> Vt = std::nullopt)
{
  detail::csr_linear_operator<ValueTypeT, NNZTypeT> op(A);
  detail::sparse_lanczos_svd(handle, config, op, singular_values, U, Vt);
}

/** @} */

}  // namespace raft::sparse::solver
