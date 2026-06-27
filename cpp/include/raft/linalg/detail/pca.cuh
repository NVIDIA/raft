/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/pca_types.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/tsvd.cuh>
#include <raft/matrix/copy.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/ratio.cuh>
#include <raft/matrix/sqrt.cuh>
#include <raft/stats/cov.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

namespace raft {
namespace linalg::detail {

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
  auto stream = resource::get_cuda_stream(handle);

  auto n_cols       = in.extent(0);
  auto n_components = components.extent(0);

  auto len = static_cast<std::size_t>(n_cols * n_cols);
  rmm::device_uvector<math_t> components_all(len, stream);
  rmm::device_uvector<math_t> explained_var_all(static_cast<std::size_t>(n_cols), stream);
  rmm::device_uvector<math_t> explained_var_ratio_all(static_cast<std::size_t>(n_cols), stream);

  detail::cal_eig<math_t, idx_t>(
    handle,
    prms,
    in,
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      components_all.data(), n_cols, n_cols),
    raft::make_device_vector_view<math_t, idx_t>(explained_var_all.data(), n_cols));
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
      components_all.data(), n_cols, n_cols),
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      components.data_handle(), n_components, n_cols));
  raft::matrix::ratio(handle,
                      raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
                        explained_var_all.data(), n_cols, idx_t(1)),
                      raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
                        explained_var_ratio_all.data(), n_cols, idx_t(1)));
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
      explained_var_all.data(), n_cols, idx_t(1)),
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      explained_var.data_handle(), n_components, idx_t(1)));
  raft::matrix::trunc_zero_origin(
    handle,
    raft::make_device_matrix_view<const math_t, idx_t, raft::col_major>(
      explained_var_ratio_all.data(), n_cols, idx_t(1)),
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      explained_var_ratio.data_handle(), n_components, idx_t(1)));

  if (static_cast<std::size_t>(n_components) < static_cast<std::size_t>(n_cols) &&
      static_cast<std::size_t>(n_components) < n_rows) {
    raft::stats::mean<true>(noise_vars.data_handle(),
                            explained_var_all.data() + static_cast<std::size_t>(n_components),
                            std::size_t{1},
                            static_cast<std::size_t>(n_cols - n_components),
                            false,
                            stream);
  } else {
    raft::matrix::fill(
      handle,
      raft::make_device_vector_view<math_t, idx_t>(noise_vars.data_handle(), idx_t(1)),
      math_t{0});
  }
}

/**
 * @brief perform fit operation for PCA.
 *
 * Supports both row-major and col-major input layouts via the LayoutPolicy template
 * parameter. The output `components` matrix has the same layout as the input.
 *
 * @tparam math_t element type
 * @tparam idx_t index type
 * @tparam LayoutPolicy layout of the input matrix (raft::row_major or raft::col_major)
 * @param[in] handle: raft::resources
 * @param[in] prms: PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input: the data is fitted to PCA. Size n_rows x n_cols.
 * @param[out] components: the principal components. Size n_components x n_cols.
 * @param[out] explained_var: explained variances. Size n_components.
 * @param[out] explained_var_ratio: ratio of explained to total variance. Size n_components.
 * @param[out] singular_vals: singular values. Size n_components.
 * @param[out] mu: mean of all features. Size n_cols.
 * @param[out] noise_vars: noise variance scalar.
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
  static_assert(
    std::is_same_v<LayoutPolicy, raft::row_major> || std::is_same_v<LayoutPolicy, raft::col_major>,
    "pca_fit: input layout must be raft::row_major or raft::col_major");
  constexpr bool input_row_major = std::is_same_v<LayoutPolicy, raft::row_major>;

  auto stream        = resource::get_cuda_stream(handle);
  auto cublas_handle = raft::resource::get_cublas_handle(handle);

  auto n_rows = input.extent(0);
  auto n_cols = input.extent(1);

  auto n_components = components.extent(0);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(n_rows > 1, "Parameter n_rows: number of rows cannot be less than two");
  ASSERT(n_components > 0, "Parameter n_components: number of components cannot be less than one");
  ASSERT(n_components <= n_cols, "n_components cannot exceed n_cols");

  raft::stats::mean<input_row_major>(
    mu.data_handle(), input.data_handle(), n_cols, n_rows, false, stream);

  auto len = static_cast<std::size_t>(n_cols * n_cols);
  rmm::device_uvector<math_t> cov(len, stream);

  raft::stats::cov<input_row_major>(
    handle, cov.data(), input.data_handle(), mu.data_handle(), n_cols, n_rows, true, true, stream);

  // The eigendecomposition of the (symmetric) covariance matrix naturally produces a
  // col-major components buffer. For row-major output we accumulate into a temporary
  // and physically transpose at the end.
  std::optional<raft::device_matrix<math_t, idx_t, raft::col_major>> components_col_storage;
   if constexpr (input_row_major) {
     components_col_storage = raft::make_device_matrix_view<...)(...);
   }
  math_t* components_col_data =
    input_row_major ? components_col_storage->data_handle() : components.data_handle();
  auto components_col_view = raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
    components_col_data, n_components, n_cols);

  detail::trunc_comp_exp_vars(
    handle,
    prms,
    raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(cov.data(), n_cols, n_cols),
    components_col_view,
    explained_var,
    explained_var_ratio,
    noise_vars,
    static_cast<std::size_t>(n_rows));

  math_t scalar = (n_rows - 1);
  raft::matrix::weighted_sqrt(handle,
                              raft::make_device_matrix_view<const math_t, idx_t, raft::row_major>(
                                explained_var.data_handle(), idx_t(1), n_components),
                              raft::make_device_matrix_view<math_t, idx_t, raft::row_major>(
                                singular_vals.data_handle(), idx_t(1), n_components),
                              raft::make_host_scalar_view(&scalar),
                              true);

  raft::stats::meanAdd<input_row_major, true>(
    input.data_handle(), input.data_handle(), mu.data_handle(), n_cols, n_rows, stream);

  detail::sign_flip_components(handle, input, components_col_view, true, flip_signs_based_on_U);

  if constexpr (input_row_major) {
    // Transpose the internal col-major (n_components x n_cols) components into the user's
    // row-major (n_components x n_cols) buffer. The same memory laid out as col-major
    // (n_cols x n_components) is exactly the row-major (n_components x n_cols) we want.
    auto components_as_col_view = raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      components.data_handle(), n_cols, n_components);
    raft::linalg::transpose(handle, components_col_view, components_as_col_view);
  }
}

/**
 * @brief performs transform operation for PCA. Transforms the data to eigenspace.
 *
 * Supports both row-major and col-major layouts via the LayoutPolicy template parameter.
 * `input`, `components`, and `trans_input` must all share the same layout.
 *
 * @tparam math_t element type
 * @tparam idx_t index type
 * @tparam LayoutPolicy layout (raft::row_major or raft::col_major)
 * @param[in] handle: raft::resources
 * @param[in] prms: PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input: the data to transform. Size n_rows x n_cols.
 * @param[in] components: principal components. Size n_components x n_cols.
 * @param[in] singular_vals: singular values. Size n_components.
 * @param[in] mu: mean of features. Size n_cols.
 * @param[out] trans_input: the transformed data. Size n_rows x n_components.
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
  static_assert(
    std::is_same_v<LayoutPolicy, raft::row_major> || std::is_same_v<LayoutPolicy, raft::col_major>,
    "pca_transform: layout must be raft::row_major or raft::col_major");
  constexpr bool input_row_major = std::is_same_v<LayoutPolicy, raft::row_major>;

  auto stream = resource::get_cuda_stream(handle);

  auto n_rows       = input.extent(0);
  auto n_cols       = input.extent(1);
  auto n_components = components.extent(0);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(n_components > 0, "Parameter n_components: number of components cannot be less than one");

  auto components_len = static_cast<std::size_t>(n_cols * n_components);
  rmm::device_uvector<math_t> components_copy{components_len, stream};
  raft::copy(components_copy.data(), components.data_handle(), components_len, stream);

  auto components_copy_view = raft::make_device_matrix_view<math_t, idx_t, LayoutPolicy>(
    components_copy.data(), n_components, n_cols);

  if (prms.whiten) {
    math_t scalar = math_t(sqrt(n_rows - 1));
    raft::linalg::scalarMultiply(
      components_copy.data(), components_copy.data(), scalar, components_len, stream);
    // Divide each row of (n_components x n_cols) components by the corresponding singular
    // value. Apply::ALONG_COLUMNS broadcasts a vector of size n_rows-of-matrix
    // (= n_components) over each column, which is the same operation in both layouts.
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_COLUMNS>(
      handle,
      components_copy_view,
      raft::make_device_vector_view<const math_t, idx_t>(singular_vals.data_handle(),
                                                         n_components));
  }

  raft::stats::meanCenter<input_row_major, true>(
    input.data_handle(), input.data_handle(), mu.data_handle(), n_cols, n_rows, stream);

  // trans_input = input @ components_copy^T, in the user's layout.
  // Reinterpreting the components_copy buffer with the opposite layout swaps the logical
  // dimensions, giving us the (n_cols x n_components) transposed view we need for gemm.
  using transposed_layout = std::conditional_t<input_row_major, raft::col_major, raft::row_major>;
  auto components_copy_transposed = raft::make_device_matrix_view<math_t, idx_t, transposed_layout>(
    components_copy.data(), n_cols, n_components);
  raft::linalg::gemm(handle, input, components_copy_transposed, trans_input);

  raft::stats::meanAdd<input_row_major, true>(
    input.data_handle(), input.data_handle(), mu.data_handle(), n_cols, n_rows, stream);
}

/**
 * @brief performs inverse transform operation for PCA.
 *
 * Supports both row-major and col-major layouts via the LayoutPolicy template parameter.
 * `trans_input`, `components`, and `output` must all share the same layout.
 *
 * @tparam math_t element type
 * @tparam idx_t index type
 * @tparam LayoutPolicy layout (raft::row_major or raft::col_major)
 * @param[in] handle: raft::resources
 * @param[in] prms: PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[in] trans_input: the transformed data. Size n_rows x n_components.
 * @param[in] components: principal components. Size n_components x n_cols.
 * @param[in] singular_vals: singular values. Size n_components.
 * @param[in] mu: mean of features. Size n_cols.
 * @param[out] output: the reconstructed data. Size n_rows x n_cols.
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
  static_assert(
    std::is_same_v<LayoutPolicy, raft::row_major> || std::is_same_v<LayoutPolicy, raft::col_major>,
    "pca_inverse_transform: layout must be raft::row_major or raft::col_major");
  constexpr bool input_row_major = std::is_same_v<LayoutPolicy, raft::row_major>;

  auto stream = resource::get_cuda_stream(handle);

  auto n_rows       = output.extent(0);
  auto n_cols       = output.extent(1);
  auto n_components = components.extent(0);

  ASSERT(n_cols > 1, "Parameter n_cols: number of columns cannot be less than two");
  ASSERT(n_rows > 0, "Parameter n_rows: number of rows cannot be less than one");
  ASSERT(n_components > 0, "Parameter n_components: number of components cannot be less than one");

  auto components_len = static_cast<std::size_t>(n_cols * n_components);
  rmm::device_uvector<math_t> components_copy{components_len, stream};
  raft::copy(components_copy.data(), components.data_handle(), components_len, stream);

  auto components_copy_view = raft::make_device_matrix_view<math_t, idx_t, LayoutPolicy>(
    components_copy.data(), n_components, n_cols);

  if (prms.whiten) {
    math_t sqrt_n_samples = sqrt(n_rows - 1);
    math_t scalar         = n_rows - 1 > 0 ? math_t(1 / sqrt_n_samples) : 0;
    raft::linalg::scalarMultiply(
      components_copy.data(), components_copy.data(), scalar, components_len, stream);
    raft::linalg::binary_mult_skip_zero<raft::Apply::ALONG_COLUMNS>(
      handle,
      components_copy_view,
      raft::make_device_vector_view<const math_t, idx_t>(singular_vals.data_handle(),
                                                         n_components));
  }

  // output = trans_input @ components_copy. All three matrices share the user's layout,
  // so the mdspan gemm picks the correct cuBLAS transposes automatically.
  raft::linalg::gemm(handle, trans_input, components_copy_view, output);

  raft::stats::meanAdd<input_row_major, true>(
    output.data_handle(), output.data_handle(), mu.data_handle(), n_cols, n_rows, stream);
}

/**
 * @brief perform fit and transform operations for PCA.
 *
 * Supports both row-major and col-major layouts via the LayoutPolicy template parameter.
 *
 * @tparam math_t element type
 * @tparam idx_t index type
 * @tparam LayoutPolicy layout (raft::row_major or raft::col_major)
 * @param[in] handle: raft::resources
 * @param[in] prms: PCA parameters (n_components, algorithm, whiten, etc.)
 * @param[inout] input: the data is fitted to PCA. Size n_rows x n_cols.
 * @param[out] trans_input: the transformed data. Size n_rows x n_components.
 * @param[out] components: the principal components. Size n_components x n_cols.
 * @param[out] explained_var: explained variances. Size n_components.
 * @param[out] explained_var_ratio: ratio of explained to total variance. Size n_components.
 * @param[out] singular_vals: singular values. Size n_components.
 * @param[out] mu: mean of all features. Size n_cols.
 * @param[out] noise_vars: noise variance scalar.
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
  detail::pca_transform(handle, prms, input, components, singular_vals, mu, trans_input);
}

};  // end namespace linalg::detail
};  // end namespace raft
