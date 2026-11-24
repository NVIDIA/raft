/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/detail/pca_tsvd.cuh>
#include <raft/linalg/rsvd.cuh>

namespace raft {
namespace linalg::detail {

template <typename math_t, typename idx_t>
void cal_comp_exp_vars_svd(raft::resources const& handle,
                           const paramsTSVD& prms,
                           raft::device_matrix_view<math_t, idx_t, raft::col_major> in,
                           raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                           raft::device_vector_view<math_t, idx_t> singular_vals,
                           raft::device_vector_view<math_t, idx_t> explained_vars,
                           raft::device_vector_view<math_t, idx_t> explained_var_ratio)
{
  auto stream          = resource::get_cuda_stream(handle).get();
  auto cusolver_handle = raft::resource::get_cusolver_dn_handle(handle);
  auto cublas_handle   = raft::resource::get_cublas_handle(handle);

  auto n_rows       = in.extent(0);
  auto n_cols       = in.extent(1);
  auto n_components = components.extent(0);

  auto diff    = n_cols - n_components;
  math_t ratio = math_t(diff) / math_t(n_cols);
  ASSERT(ratio >= math_t(0.2),
         "Number of components should be less than at least 80 percent of the "
         "number of features");

  std::size_t p = static_cast<std::size_t>(math_t(0.1) * math_t(n_cols));
  ASSERT(p >= 5, "RSVD should be used where the number of columns are at least 50");

  auto total_random_vecs = static_cast<std::size_t>(n_components) + p;
  ASSERT(total_random_vecs < static_cast<std::size_t>(n_cols),
         "RSVD should be used where the number of columns are at least 50");

  rmm::device_uvector<math_t> components_temp(static_cast<std::size_t>(n_cols * n_components),
                                              stream);
  math_t* left_eigvec = nullptr;
  raft::linalg::rsvdFixedRank(handle,
                              in.data_handle(),
                              n_rows,
                              n_cols,
                              singular_vals.data_handle(),
                              left_eigvec,
                              components_temp.data(),
                              n_components,
                              p,
                              true,
                              false,
                              true,
                              false,
                              (math_t)prms.tol,
                              prms.n_iterations,
                              stream);

  raft::linalg::transpose(
    handle, components_temp.data(), components.data_handle(), n_cols, n_components, stream);

  raft::matrix::weighted_power(handle,
                               raft::make_device_matrix_view<const math_t, idx_t, raft::row_major>(
                                 singular_vals.data_handle(), idx_t(1), n_components),
                               raft::make_device_matrix_view<math_t, idx_t, raft::row_major>(
                                 explained_vars.data_handle(), idx_t(1), n_components),
                               math_t(1));
  raft::matrix::ratio(
    handle, explained_vars.data_handle(), explained_var_ratio.data_handle(), n_components, stream);
}

/**
 * @brief perform fit operation for the tsvd.
 * @param[in] handle: raft::resources
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] input: the data is fitted to tSVD. Size n_rows x n_cols (col-major).
 * @param[out] components: the principal components. Size n_components x n_cols (col-major).
 * @param[out] singular_vals: singular values of the data. Size n_components.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t>
void tsvd_fit(raft::resources const& handle,
              const paramsTSVD& prms,
              raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
              raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
              raft::device_vector_view<math_t, idx_t> singular_vals,
              bool flip_signs_based_on_U = false)
{
  auto stream        = resource::get_cuda_stream(handle).get();
  auto cublas_handle = raft::resource::get_cublas_handle(handle);

  auto n_rows = input.extent(0);
  auto n_cols = input.extent(1);

  auto n_components = components.extent(0);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(n_components > 0, "Parameter n_components: number of components cannot be less than one");
  ASSERT(n_components <= n_cols, "n_components cannot exceed n_cols");

  auto len = static_cast<std::size_t>(n_cols * n_cols);
  rmm::device_uvector<math_t> input_cross_mult(len, stream);

  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  raft::linalg::gemm(handle,
                     input.data_handle(),
                     n_rows,
                     n_cols,
                     input.data_handle(),
                     input_cross_mult.data(),
                     n_cols,
                     n_cols,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     stream);

  rmm::device_uvector<math_t> components_all(len, stream);
  rmm::device_uvector<math_t> explained_var_all(static_cast<std::size_t>(n_cols), stream);

  detail::cal_eig(handle,
                  prms,
                  raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                    input_cross_mult.data(), n_cols, n_cols),
                  raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                    components_all.data(), n_cols, n_cols),
                  raft::make_device_vector_view<math_t, idx_t>(explained_var_all.data(), n_cols));

  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
      components_all.data(), n_cols, n_cols),
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      components.data_handle(), n_components, n_cols));

  math_t scalar = math_t(1);
  raft::matrix::weighted_sqrt(handle,
                              raft::make_device_matrix_view<const math_t, idx_t, raft::row_major>(
                                explained_var_all.data(), idx_t(1), n_components),
                              raft::make_device_matrix_view<math_t, idx_t, raft::row_major>(
                                singular_vals.data_handle(), idx_t(1), n_components),
                              raft::make_host_scalar_view(&scalar));

  detail::sign_flip_components(handle,
                               input,
                               raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                                 components.data_handle(), n_components, n_cols),
                               false,
                               flip_signs_based_on_U);
}

/**
 * @brief performs fit and transform operations for the tsvd.
 * @param[in] handle: raft::resources
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] input: the data is fitted to tSVD. Size n_rows x n_cols (col-major).
 * @param[out] trans_input: the transformed data. Size n_rows x n_components (col-major).
 * @param[out] components: the principal components. Size n_components x n_cols (col-major).
 * @param[out] explained_var: explained variances. Size n_components.
 * @param[out] explained_var_ratio: ratio of explained variance to total. Size n_components.
 * @param[out] singular_vals: singular values of the data. Size n_components.
 * @param[in] flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t>
void tsvd_fit_transform(raft::resources const& handle,
                        const paramsTSVD& prms,
                        raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
                        raft::device_matrix_view<math_t, idx_t, raft::col_major> trans_input,
                        raft::device_matrix_view<math_t, idx_t, raft::col_major> components,
                        raft::device_vector_view<math_t, idx_t> explained_var,
                        raft::device_vector_view<math_t, idx_t> explained_var_ratio,
                        raft::device_vector_view<math_t, idx_t> singular_vals,
                        bool flip_signs_based_on_U = false)
{
  auto stream = resource::get_cuda_stream(handle).get();

  auto n_cols       = input.extent(1);
  auto n_components = components.extent(0);

  detail::tsvd_fit(handle, prms, input, components, singular_vals, flip_signs_based_on_U);
  detail::tsvd_transform(handle, prms, input, components, trans_input);

  auto mu_trans = raft::make_device_vector<math_t, idx_t>(handle, n_components);
  raft::stats::mean(handle, raft::make_const_mdspan(trans_input), mu_trans.view());
  raft::stats::vars(handle,
                    raft::make_const_mdspan(trans_input),
                    raft::make_const_mdspan(mu_trans.view()),
                    explained_var,
                    false);

  auto mu   = raft::make_device_vector<math_t, idx_t>(handle, n_cols);
  auto vars = raft::make_device_vector<math_t, idx_t>(handle, n_cols);

  raft::stats::mean(handle, raft::make_const_mdspan(input), mu.view());
  raft::stats::vars(
    handle, raft::make_const_mdspan(input), raft::make_const_mdspan(mu.view()), vars.view(), false);

  rmm::device_scalar<math_t> total_vars(stream);
  raft::stats::sum(handle,
                   raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
                     vars.data_handle(), n_cols, idx_t(1)),
                   raft::make_device_vector_view<math_t, idx_t>(total_vars.data(), idx_t(1)));

  if (resource::get_dry_run_flag(handle)) { return; }

  math_t total_vars_h;
  raft::update_host(&total_vars_h, total_vars.data(), 1, stream);
  raft::resource::sync_stream(handle, stream);
  math_t scalar = math_t(1) / total_vars_h;

  raft::linalg::scalarMultiply(
    explained_var_ratio.data_handle(), explained_var.data_handle(), scalar, n_components, stream);
}

};  // end namespace linalg::detail
};  // end namespace raft
