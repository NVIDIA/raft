/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/pca_types.hpp>
#include <raft/linalg/reduce.cuh>
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

  auto stream          = resource::get_cuda_stream(handle).get();
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
  raft::matrix::col_reverse(handle,
                            raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                              components.data_handle(), n_cols, n_cols));
  if constexpr (!is_row_major) {
    // The in-place square transpose has no overload taking raft::resources to carry the flag.
    if (!resource::get_dry_run_flag(handle)) {
      raft::linalg::transpose(components.data_handle(), n_cols, stream);
    }
  }

  raft::matrix::row_reverse(handle,
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

  auto stream       = resource::get_cuda_stream(handle).get();
  auto n_samples    = input.extent(0);
  auto n_features   = input.extent(1);
  auto n_components = components.extent(0);

  rmm::device_uvector<math_t> max_vals(static_cast<std::size_t>(n_components), stream);
  auto components_view = raft::make_device_matrix_view<math_t, idx_t, LayoutPolicy>(
    components.data_handle(), n_components, n_features);
  auto max_vals_view = raft::make_device_vector_view<math_t, idx_t>(max_vals.data(), n_components);

  if (flip_signs_based_on_U) {
    if (center) {
      auto col_means = raft::make_device_vector<math_t, idx_t>(handle, n_features);
      raft::stats::mean(handle, raft::make_const_mdspan(input), col_means.view());
      raft::stats::mean_center<raft::Apply::ALONG_ROWS>(
        handle, raft::make_const_mdspan(input), raft::make_const_mdspan(col_means.view()), input);
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

    raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(
      handle,
      raft::make_device_matrix_view<const math_t, idx_t, LayoutPolicy>(
        US.data(), n_samples, n_components),
      max_vals_view,
      math_t(0),
      false,
      raft::identity_op(),
      [] __device__(math_t a, math_t b) {
        math_t abs_a = a >= math_t(0) ? a : -a;
        math_t abs_b = b >= math_t(0) ? b : -b;
        return abs_a >= abs_b ? a : b;
      },
      raft::identity_op());
  } else {
    raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
      handle,
      raft::make_const_mdspan(components_view),
      max_vals_view,
      math_t(0),
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
  auto stream      = resource::get_cuda_stream(handle).get();
  auto n_rows      = input.extent(0);
  auto n_cols      = input.extent(1);
  auto n_cols_comp = components.extent(1);

  auto* input_ptr      = input.data_handle();
  auto* components_ptr = components.data_handle();
  auto counting        = thrust::make_counting_iterator(0);
  auto m               = n_rows;

  if (resource::get_dry_run_flag(handle)) { return; }

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

}  // namespace linalg::detail
}  // namespace raft
