/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __GEMV_H
#define __GEMV_H

#pragma once

#include "detail/gemv.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief the wrapper of cublas gemv function
 *  It computes the following equation: y = alpha .* op(A) * x + beta .* y
 *
 * @tparam math_t the element type
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @param [in] handle raft handle
 * @param [in] trans_a cublas transpose op for A
 * @param [in] m number of rows of A
 * @param [in] n number of columns of A
 * @param [in] alpha host or device scalar
 * @param [in] A column-major matrix of size [m, n]
 * @param [in] lda leading dimension of A
 * @param [in] x vector of length n if trans_a else m
 * @param [in] incx stride between consecutive elements of x
 * @param [in] beta host or device scalar
 * @param [inout] y vector of length m if trans_a else n
 * @param [in] incy stride between consecutive elements of y
 * @param [in] stream
 */
template <typename math_t, bool DevicePointerMode = false>
void gemv(raft::resources const& handle,
          const bool trans_a,
          const int m,
          const int n,
          const math_t* alpha,
          const math_t* A,
          const int lda,
          const math_t* x,
          const int incx,
          const math_t* beta,
          math_t* y,
          const int incy,
          cudaStream_t stream)
{
  detail::gemv<math_t, DevicePointerMode>(
    handle, trans_a, m, n, alpha, A, lda, x, incx, beta, y, incy, stream);
}

template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows,
          const int n_cols,
          const math_t* x,
          const int incx,
          math_t* y,
          const int incy,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          cudaStream_t stream)
{
  detail::gemv(handle, A, n_rows, n_cols, x, incx, y, incy, trans_a, alpha, beta, stream);
}

/**
 * y = alpha * op(A) * x + beta * y
 *
 * where
 *
 * @param handle raft handle
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 * @param n_rows_a number of rows in A
 * @param n_cols_a number of cols in A
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 * @param trans_a whether to take transpose of a
 * @param alpha is a scalar scale of Ax.
 * @param beta is a scalar scale of y.
 * @param stream stream on which this function is run
 */
template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          cudaStream_t stream)
{
  detail::gemv(handle, A, n_rows_a, n_cols_a, x, y, trans_a, alpha, beta, stream);
}

/**
 * y = op(A) * x
 *
 * where
 *
 * @param handle raft handle
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 * @param n_rows_a number of rows in A
 * @param n_cols_a number of cols in A
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 * @param trans_a whether to take transpose of a
 * @param stream stream on which this function is run
 */
template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          cudaStream_t stream)
{
  detail::gemv(handle, A, n_rows_a, n_cols_a, x, y, trans_a, stream);
}

/**
 * y = alpha * op(A) * x + beta * y
 *
 * where
 * @param handle raft handle
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 * @param n_rows_a number of rows in A
 * @param n_cols_a number of cols in A
 * @param lda is the leading dimension of A (number of rows); lda must be not smaller than n_rows_a.
 *     set it when you need to use only the first n_rows_a rows of the matrix A, which has
 *     (perhaps, due to padding) lda rows.
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 * @param trans_a whether to take transpose of a
 * @param alpha is a scalar scale of Ax.
 * @param beta is a scalar scale of y.
 * @param stream stream on which this function is run
 */
template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const int lda,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          cudaStream_t stream)
{
  detail::gemv(handle, A, n_rows_a, n_cols_a, lda, x, y, trans_a, alpha, beta, stream);
}

/**
 * y = op(A) * x
 *
 * where
 * @param handle raft handle
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 * @param n_rows_a number of rows in A
 * @param n_cols_a number of cols in A
 * @param lda is the leading dimension of A (number of rows); lda must be not smaller than n_rows_a.
 *     set it when you need to use only the first n_rows_a rows of the matrix A, which has
 *     (perhaps, due to padding) lda rows.
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 * @param trans_a whether to take transpose of a
 * @param stream stream on which this function is run
 *
 */
template <typename math_t>
void gemv(raft::resources const& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const int lda,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          cudaStream_t stream)
{
  detail::gemv(handle, A, n_rows_a, n_cols_a, lda, x, y, trans_a, stream);
}

/**
 * @defgroup gemv Matrix-Vector Multiplication
 * @{
 */

/**
 * @brief GEMV function designed for raft::col_major layout for A
 * It computes y  = alpha * op(A) * x + beta * y, where length of y is number
 * of rows in A while length of x is number of columns in A
 * If layout for A is provided as raft::row_major, then a transpose of A
 * is used in the computation, where length of y is number of columns in A
 * while length of x is number of rows in A
 * If alpha is not provided, it is assumed to be 1.0
 * If beta is not provided, it is assumed to be 0.0
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of index
 * @tparam LayoutPolicyX layout of X
 * @tparam LayoutPolicyY layout of Y
 * @tparam LayoutPolicyZ layout of Z
 * @param[in] handle raft handle
 * @param[in] A input raft::device_matrix_view of size (M, N)
 * @param[in] x input raft::device_matrix_view of size (N, 1) if A is raft::col_major, else (M, 1)
 * @param[out] y output raft::device_matrix_view of size (M, 1) if A is raft::col_major, else (N, 1)
 * @param[in] alpha optional raft::host_scalar_view or raft::device_scalar_view, default 1.0
 * @param[in] beta optional raft::host_scalar_view or raft::device_scalar_view, default 0.0
 */
template <typename ValueType,
          typename IndexType,
          typename LayoutPolicy,
          typename ScalarIdxType  = std::uint32_t,
          typename ScalarViewType = raft::host_scalar_view<ValueType, ScalarIdxType>,
          typename                = std::enable_if_t<std::disjunction_v<
                           std::is_same<ScalarViewType, raft::host_scalar_view<ValueType, ScalarIdxType>>,
                           std::is_same<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>>>>
void gemv(raft::resources const& handle,
          raft::device_matrix_view<const ValueType, IndexType, LayoutPolicy> A,
          raft::device_vector_view<const ValueType, IndexType> x,
          raft::device_vector_view<ValueType, IndexType> y,
          std::optional<ScalarViewType> alpha = std::nullopt,
          std::optional<ScalarViewType> beta  = std::nullopt)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(A), "A is not contiguous");

  constexpr auto is_A_col_major =
    std::is_same_v<typename decltype(A)::layout_type, raft::col_major>;

  if (is_A_col_major) {
    RAFT_EXPECTS(x.extent(0) == A.extent(1),
                 "Number of columns of A and length of x should be equal");
    RAFT_EXPECTS(y.extent(0) == A.extent(0), "Number of rows of A and length of y should be equal");
  } else {
    RAFT_EXPECTS(x.extent(0) == A.extent(0), "Number of rows of A and length of x should be equal");
    RAFT_EXPECTS(y.extent(0) == A.extent(1),
                 "Number of columns of A and length of y should be equal");
  }

  constexpr auto device_mode =
    std::is_same_v<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>;

  ValueType alpha_value = 1;
  ValueType beta_value  = 0;

  auto alpha_device = raft::make_device_scalar(handle, alpha_value);
  auto beta_device  = raft::make_device_scalar(handle, beta_value);

  auto alpha_host = raft::make_host_scalar(handle, alpha_value);
  auto beta_host  = raft::make_host_scalar(handle, beta_value);

  if constexpr (device_mode) {
    if (!alpha) { alpha = alpha_device.view(); }
    if (!beta) { beta = beta_device.view(); }
  } else {
    if (!alpha) { alpha = alpha_host.view(); }
    if (!beta) { beta = beta_host.view(); }
  }

  gemv<ValueType, device_mode>(handle,
                               !is_A_col_major,
                               A.extent(0),
                               A.extent(1),
                               alpha.value().data_handle(),
                               A.data_handle(),
                               A.extent(0),
                               x.data_handle(),
                               1,
                               beta.value().data_handle(),
                               y.data_handle(),
                               1,
                               resource::get_cuda_stream(handle).get());
}
/** @} */  // end of gemv

};  // namespace linalg
};  // namespace raft
#endif
