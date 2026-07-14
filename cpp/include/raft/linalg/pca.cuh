/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/pca.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>

namespace raft {
namespace linalg {

/**
 * @defgroup pca PCA operations
 * @{
 */

/**
 * @brief perform fit operation for PCA. Generates eigenvectors, explained vars, singular vals, etc.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam LayoutPolicy layout of the input/components matrices (raft::row_major or
 *                      raft::col_major)
 * @param[in] handle: raft::resources
 * @param[in] prms PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input the data is fitted to PCA. Size n_rows x n_cols. Modified
 * temporarily during computation.
 * @param[out] components the principal components of the input data. Size n_components x n_cols.
 * @param[out] explained_var explained variances (eigenvalues) of the principal components. Size
 * n_components.
 * @param[out] explained_var_ratio the ratio of the explained variance and total variance. Size
 * n_components.
 * @param[out] singular_vals singular values of the data. Size n_components.
 * @param[out] mu mean of all the features (all the columns in the data). Size n_cols.
 * @param[out] noise_vars variance of the noise. Scalar.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t, typename LayoutPolicy>
void pca_fit(raft::resources const& handle,
             const paramsPCA& prms,
             raft::device_matrix_view<math_t, idx_t, LayoutPolicy> input,
             raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
             raft::device_vector_view<math_t, idx_t> explained_var,
             raft::device_vector_view<math_t, idx_t> explained_var_ratio,
             raft::device_vector_view<math_t, idx_t> singular_vals,
             raft::device_vector_view<math_t, idx_t> mu,
             raft::device_scalar_view<math_t, idx_t> noise_vars,
             bool flip_signs_based_on_U = false)
{
  detail::pca_fit(handle,
                  prms,
                  input,
                  components,
                  explained_var,
                  explained_var_ratio,
                  singular_vals,
                  mu,
                  noise_vars,
                  flip_signs_based_on_U);
}

/**
 * @brief perform fit and transform operations for PCA. Generates transformed data,
 * eigenvectors, explained vars, singular vals, etc.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam LayoutPolicy layout of the input/output matrices (raft::row_major or raft::col_major)
 * @param[in] handle raft::resources
 * @param[in] prms PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input the data is fitted to PCA. Size n_rows x n_cols. Modified
 * temporarily during computation.
 * @param[out] trans_input the transformed data. Size n_rows x n_components.
 * @param[out] components the principal components of the input data. Size n_components x n_cols.
 * @param[out] explained_var explained variances (eigenvalues) of the principal components. Size
 * n_components.
 * @param[out] explained_var_ratio the ratio of the explained variance and total variance. Size
 * n_components.
 * @param[out] singular_vals singular values of the data. Size n_components.
 * @param[out] mu mean of all the features (all the columns in the data). Size n_cols.
 * @param[out] noise_vars variance of the noise. Scalar.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t, typename LayoutPolicy>
void pca_fit_transform(raft::resources const& handle,
                       const paramsPCA& prms,
                       raft::device_matrix_view<math_t, idx_t, LayoutPolicy> input,
                       raft::device_matrix_view<math_t, idx_t, LayoutPolicy> trans_input,
                       raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                       raft::device_vector_view<math_t, idx_t> explained_var,
                       raft::device_vector_view<math_t, idx_t> explained_var_ratio,
                       raft::device_vector_view<math_t, idx_t> singular_vals,
                       raft::device_vector_view<math_t, idx_t> mu,
                       raft::device_scalar_view<math_t, idx_t> noise_vars,
                       bool flip_signs_based_on_U = false)
{
  detail::pca_fit_transform(handle,
                            prms,
                            input,
                            trans_input,
                            components,
                            explained_var,
                            explained_var_ratio,
                            singular_vals,
                            mu,
                            noise_vars,
                            flip_signs_based_on_U);
}

/**
 * @brief performs inverse transform operation for PCA. Transforms the transformed data back to
 * original data.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam LayoutPolicy layout of the input/output matrices (raft::row_major or raft::col_major)
 * @param[in] handle raft::resources
 * @param[in] prms PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[in] trans_input the transformed data. Size n_rows x n_components.
 * @param[in] components the principal components of the input data. Size n_components x n_cols.
 * @param[in] singular_vals singular values of the data. Size n_components.
 * @param[in] mu mean of features (every column). Size n_cols.
 * @param[out] output the reconstructed data. Size n_rows x n_cols.
 */
template <typename math_t, typename idx_t, typename LayoutPolicy>
void pca_inverse_transform(raft::resources const& handle,
                           const paramsPCA& prms,
                           raft::device_matrix_view<math_t, idx_t, LayoutPolicy> trans_input,
                           raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                           raft::device_vector_view<math_t, idx_t> singular_vals,
                           raft::device_vector_view<math_t, idx_t> mu,
                           raft::device_matrix_view<math_t, idx_t, LayoutPolicy> output)
{
  detail::pca_inverse_transform(handle, prms, trans_input, components, singular_vals, mu, output);
}

/**
 * @brief performs transform operation for PCA. Transforms the data to eigenspace.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam LayoutPolicy layout of the input/output matrices (raft::row_major or raft::col_major)
 * @param[in] handle raft::resources
 * @param[in] prms PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input the data to be transformed. Size n_rows x n_cols. Modified
 * temporarily during computation (mean-centered then restored).
 * @param[in] components principal components of the input data. Size n_components x n_cols.
 * @param[in] singular_vals singular values of the data. Size n_components.
 * @param[in] mu mean value of the input data. Size n_cols.
 * @param[out] trans_input the transformed data. Size n_rows x n_components.
 */
template <typename math_t, typename idx_t, typename LayoutPolicy>
void pca_transform(raft::resources const& handle,
                   const paramsPCA& prms,
                   raft::device_matrix_view<math_t, idx_t, LayoutPolicy> input,
                   raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                   raft::device_vector_view<math_t, idx_t> singular_vals,
                   raft::device_vector_view<math_t, idx_t> mu,
                   raft::device_matrix_view<math_t, idx_t, LayoutPolicy> trans_input)
{
  detail::pca_transform(handle, prms, input, components, singular_vals, mu, trans_input);
}

/**
 * @brief Compute truncated components, explained variances, explained variance ratios,
 * and noise variance from a covariance matrix.
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @param[in] handle raft::resources
 * @param[in] prms tSVD parameters (controls algorithm, tolerance, iterations)
 * @param[inout] in covariance matrix [n_cols x n_cols] (col-major). Overwritten.
 * @param[out] components truncated eigenvectors [n_components x n_cols] (col-major)
 * @param[out] explained_var explained variances [n_components]
 * @param[out] explained_var_ratio explained variance ratios [n_components]
 * @param[out] noise_vars noise variance scalar
 * @param[in] n_rows number of rows in the original data (needed for noise variance computation)
 */
template <typename math_t, typename idx_t>
void trunc_comp_exp_vars(raft::resources const& handle,
                         const paramsTSVD& prms,
                         raft::device_matrix_view<math_t, idx_t, raft::col_major> in,
                         raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                         raft::device_vector_view<math_t, idx_t> explained_var,
                         raft::device_vector_view<math_t, idx_t> explained_var_ratio,
                         raft::device_scalar_view<math_t, idx_t> noise_vars,
                         std::size_t n_rows)
{
  detail::trunc_comp_exp_vars(
    handle, prms, in, components, explained_var, explained_var_ratio, noise_vars, n_rows);
}

/** @} */  // end group pca

};  // end namespace linalg
};  // end namespace raft
