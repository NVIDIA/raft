/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/pca_types.hpp>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/rsvd.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/power.cuh>
#include <raft/matrix/ratio.cuh>
#include <raft/matrix/reverse.cuh>
#include <raft/matrix/sqrt.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

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
  auto stream          = resource::get_cuda_stream(handle);
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

template <typename math_t, typename idx_t, typename LayoutPolicy>
void cal_eig(raft::resources const& handle,
             const paramsTSVD& prms,
             raft::device_matrix_view<math_t, idx_t, LayoutPolicy> in,
             raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
             raft::device_vector_view<math_t, idx_t> explained_var)
{
  static_assert(
    std::is_same_v<LayoutPolicy, raft::row_major> || std::is_same_v<LayoutPolicy, raft::col_major>,
    "cal_eig: layout must be raft::row_major or raft::col_major");

  constexpr bool is_row_major = std::is_same_v<LayoutPolicy, raft::row_major>;

  auto stream          = resource::get_cuda_stream(handle);
  auto cusolver_handle = raft::resource::get_cusolver_dn_handle(handle);

  auto n_cols = in.extent(0);

  if (prms.algorithm == solver::COV_EIG_JACOBI) {
    raft::linalg::eigJacobi(handle,
                            in.data_handle(),
                            n_cols,
                            n_cols,
                            components.data_handle(),
                            explained_var.data_handle(),
                            stream,
                            (math_t)prms.tol,
                            prms.n_iterations);
  } else {
    raft::linalg::eigDC(handle,
                        in.data_handle(),
                        n_cols,
                        n_cols,
                        components.data_handle(),
                        explained_var.data_handle(),
                        stream);
  }
  raft::resources handle_stream_zero;
  raft::resource::set_cuda_stream(handle_stream_zero, stream);

  raft::matrix::col_reverse(handle_stream_zero,
                            raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                              components.data_handle(), n_cols, n_cols));
  if constexpr (!is_row_major) {
    raft::linalg::transpose(components.data_handle(), n_cols, stream);
  }

  raft::matrix::row_reverse(handle_stream_zero,
                            raft::make_device_matrix_view<math_t, idx_t, raft::row_major>(
                              explained_var.data_handle(), n_cols, idx_t(1)));
}

/**
 * @brief sign flip for PCA and tSVD. Stabilizes the sign of the eigenvectors.
 * @tparam math_t element type
 * @tparam idx_t index type
 * @tparam LayoutPolicy layout of the input and components matrices
 * @param handle: raft::resources
 * @param input: input data [n_samples x n_features]
 * @param components: components matrix [n_components x n_features]
 * @param center whether to mean-center input before computing signs
 * @param flip_signs_based_on_U whether to determine signs by U (true) or V.T (false)
 */
template <typename math_t, typename idx_t, typename LayoutPolicy>
void sign_flip_components(raft::resources const& handle,
                          raft::device_matrix_view<math_t, idx_t, LayoutPolicy> input,
                          raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                          bool center,
                          bool flip_signs_based_on_U = false)
{
  static_assert(
    std::is_same_v<LayoutPolicy, raft::row_major> || std::is_same_v<LayoutPolicy, raft::col_major>,
    "sign_flip_components: layout must be raft::row_major or raft::col_major");
  constexpr bool is_row_major = std::is_same_v<LayoutPolicy, raft::row_major>;

  auto stream       = resource::get_cuda_stream(handle);
  auto n_samples    = input.extent(0);
  auto n_features   = input.extent(1);
  auto n_components = components.extent(0);

  rmm::device_uvector<math_t> max_vals(static_cast<std::size_t>(n_components), stream);
  auto components_view = raft::make_device_matrix_view<math_t, idx_t, LayoutPolicy>(
    components.data_handle(), n_components, n_features);
  auto max_vals_view = raft::make_device_vector_view<math_t, idx_t>(max_vals.data(), n_components);

  if (flip_signs_based_on_U) {
    if (center) {
      rmm::device_uvector<math_t> col_means(static_cast<std::size_t>(n_features), stream);
      raft::stats::mean<is_row_major>(
        col_means.data(), input.data_handle(), n_features, n_samples, stream);
      raft::stats::meanCenter<is_row_major, true>(
        input.data_handle(), input.data_handle(), col_means.data(), n_features, n_samples, stream);
    }
    rmm::device_uvector<math_t> US(static_cast<std::size_t>(n_samples * n_components), stream);
    raft::linalg::gemm(handle,
                       input,
                       raft::make_device_matrix_view<
                         math_t,
                         idx_t,
                         std::conditional_t<is_row_major, raft::col_major, raft::row_major>>(
                         components.data_handle(), n_features, n_components),
                       raft::make_device_matrix_view<math_t, idx_t, LayoutPolicy>(
                         US.data(), n_samples, n_components));

    raft::linalg::reduce<is_row_major, false>(
      max_vals.data(),
      US.data(),
      n_components,
      n_samples,
      math_t(0),
      stream,
      false,
      raft::identity_op(),
      [] __device__(math_t a, math_t b) {
        math_t abs_a = a >= math_t(0) ? a : -a;
        math_t abs_b = b >= math_t(0) ? b : -b;
        return abs_a >= abs_b ? a : b;
      },
      raft::identity_op());
  } else {
    raft::linalg::reduce<is_row_major, true>(
      max_vals.data(),
      components.data_handle(),
      n_features,
      n_components,
      math_t(0),
      stream,
      false,
      raft::identity_op(),
      [] __device__(math_t a, math_t b) {
        math_t abs_a = a >= math_t(0) ? a : -a;
        math_t abs_b = b >= math_t(0) ? b : -b;
        return abs_a >= abs_b ? a : b;
      },
      raft::identity_op());
  }

  raft::linalg::map_offset(
    handle,
    components_view,
    [components_view, max_vals_view, n_components, n_features] __device__(auto idx) {
      idx_t row;
      idx_t column;
      if constexpr (is_row_major) {
        row    = idx / n_features;
        column = idx % n_features;
      } else {
        row    = idx % n_components;
        column = idx / n_components;
      }
      return (max_vals_view(row) < math_t(0)) ? (-components_view(row, column))
                                              : components_view(row, column);
    });
}

/**
 * @brief sign flip for PCA and tSVD. Stabilizes the sign of column major eigenvectors.
 * @param handle: raft::resources
 * @param input: input matrix [n_rows x n_cols] (col-major). Modified in place.
 * @param components: components matrix [n_rows x n_cols_comp] (col-major). Modified in place.
 */
template <typename math_t, typename idx_t>
void sign_flip(raft::resources const& handle,
               raft::device_matrix_view<math_t, idx_t, raft::col_major> input,
               raft::device_matrix_view<math_t, idx_t, raft::col_major> components)
{
  auto stream      = resource::get_cuda_stream(handle);
  auto n_rows      = input.extent(0);
  auto n_cols      = input.extent(1);
  auto n_cols_comp = components.extent(1);

  auto* input_ptr      = input.data_handle();
  auto* components_ptr = components.data_handle();
  auto counting        = thrust::make_counting_iterator(0);
  auto m               = n_rows;

  thrust::for_each(
    rmm::exec_policy(stream), counting, counting + n_cols, [=] __device__(idx_t idx) {
      auto d_i = idx * m;
      auto end = d_i + m;

      math_t max      = 0.0;
      idx_t max_index = 0;
      for (auto i = d_i; i < end; i++) {
        math_t val = input_ptr[i];
        if (val < 0.0) { val = -val; }
        if (val > max) {
          max       = val;
          max_index = i;
        }
      }

      if (input_ptr[max_index] < 0.0) {
        for (auto i = d_i; i < end; i++) {
          input_ptr[i] = -input_ptr[i];
        }

        auto len = n_cols * n_cols_comp;
        for (auto i = idx; i < len; i = i + n_cols) {
          components_ptr[i] = -components_ptr[i];
        }
      }
    });
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
  auto stream        = resource::get_cuda_stream(handle);
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
 * @brief performs transform operation for the tsvd. Transforms the data to eigenspace.
 * @param[in] handle raft::resources
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] input: the data to transform. Size n_rows x n_cols.
 * @param[in] components: principal components. Size n_components x n_cols.
 * @param[out] trans_input: transformed output. Size n_rows x n_components.
 */
template <typename math_t, typename idx_t, typename LayoutPolicy = raft::col_major>
void tsvd_transform(raft::resources const& handle,
                    const paramsTSVD& prms,
                    raft::device_matrix_view<math_t, idx_t, LayoutPolicy> input,
                    raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                    raft::device_matrix_view<math_t, idx_t, LayoutPolicy> trans_input)
{
  static_assert(
    std::is_same_v<LayoutPolicy, raft::row_major> || std::is_same_v<LayoutPolicy, raft::col_major>,
    "tsvd_transform: layout must be raft::row_major or raft::col_major");
  constexpr bool is_row_major = std::is_same_v<LayoutPolicy, raft::row_major>;

  auto n_rows       = input.extent(0);
  auto n_cols       = input.extent(1);
  auto n_components = components.extent(0);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(n_components > 0, "Parameter n_components: number of components cannot be less than one");

  raft::linalg::gemm(handle,
                     input,
                     raft::make_device_matrix_view<
                       math_t,
                       idx_t,
                       std::conditional_t<is_row_major, raft::col_major, raft::row_major>>(
                       components.data_handle(), n_cols, n_components),
                     trans_input);
}

/**
 * @brief performs inverse transform operation for the tsvd.
 * @param[in] handle raft::resources
 * @param[in] prms: data structure that includes all the parameters from input size to algorithm.
 * @param[in] trans_input: the transformed data. Size n_rows x n_components.
 * @param[in] components: principal components. Size n_components x n_cols.
 * @param[out] output: reconstructed output. Size n_rows x n_cols.
 */
template <typename math_t, typename idx_t, typename LayoutPolicy = raft::col_major>
void tsvd_inverse_transform(raft::resources const& handle,
                            const paramsTSVD& prms,
                            raft::device_matrix_view<math_t, idx_t, LayoutPolicy> trans_input,
                            raft::device_matrix_view<math_t, idx_t, LayoutPolicy> components,
                            raft::device_matrix_view<math_t, idx_t, LayoutPolicy> output)
{
  static_assert(
    std::is_same_v<LayoutPolicy, raft::row_major> || std::is_same_v<LayoutPolicy, raft::col_major>,
    "tsvd_inverse_transform: layout must be raft::row_major or raft::col_major");

  auto n_rows       = output.extent(0);
  auto n_cols       = output.extent(1);
  auto n_components = components.extent(0);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than one");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(n_components > 0, "Parameter n_components: number of components cannot be less than one");

  raft::linalg::gemm(handle, trans_input, components, output);
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
  auto stream = resource::get_cuda_stream(handle);

  auto n_rows       = input.extent(0);
  auto n_cols       = input.extent(1);
  auto n_components = components.extent(0);

  detail::tsvd_fit(handle, prms, input, components, singular_vals, flip_signs_based_on_U);
  detail::tsvd_transform(handle, prms, input, components, trans_input);

  rmm::device_uvector<math_t> mu_trans(static_cast<std::size_t>(n_components), stream);
  raft::stats::mean<false>(
    mu_trans.data(), trans_input.data_handle(), n_components, n_rows, false, stream);
  raft::stats::vars<false>(explained_var.data_handle(),
                           trans_input.data_handle(),
                           mu_trans.data(),
                           n_components,
                           n_rows,
                           false,
                           stream);

  rmm::device_uvector<math_t> mu(static_cast<std::size_t>(n_cols), stream);
  rmm::device_uvector<math_t> vars(static_cast<std::size_t>(n_cols), stream);

  raft::stats::mean<false>(mu.data(), input.data_handle(), n_cols, n_rows, false, stream);
  raft::stats::vars<false>(
    vars.data(), input.data_handle(), mu.data(), n_cols, n_rows, false, stream);

  rmm::device_scalar<math_t> total_vars(stream);
  raft::stats::sum<false>(
    total_vars.data(), vars.data(), std::size_t(1), static_cast<std::size_t>(n_cols), stream);

  math_t total_vars_h;
  raft::update_host(&total_vars_h, total_vars.data(), 1, stream);
  raft::resource::sync_stream(handle, stream);
  math_t scalar = math_t(1) / total_vars_h;

  raft::linalg::scalarMultiply(
    explained_var_ratio.data_handle(), explained_var.data_handle(), scalar, n_components, stream);
}

};  // end namespace linalg::detail
};  // end namespace raft
