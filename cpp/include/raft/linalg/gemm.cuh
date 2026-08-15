/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __GEMM_H
#define __GEMM_H

#pragma once

#include "detail/cublaslt_wrappers.hpp"
#include "detail/gemm.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief the wrapper of cublas gemm function
 *  It computes the following equation: C = alpha .* opA(A) * opB(B) + beta .* C
 *
 * @tparam A_t the element type of A
 * @tparam B_t the element type of B
 * @tparam C_t the element type of C
 * @tparam S_t the element type of alpha and beta
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @param [in] handle raft handle
 * @param [in] trans_a cublas transpose op for A
 * @param [in] trans_b cublas transpose op for B
 * @param [in] m number of rows of C
 * @param [in] n number of columns of C
 * @param [in] k number of rows of opB(B) / number of columns of opA(A)
 * @param [in] alpha host or device scalar
 * @param [in] A such a matrix that the shape of column-major opA(A) is [m, k]
 * @param [in] lda leading dimension of A
 * @param [in] B such a matrix that the shape of column-major opA(B) is [k, n]
 * @param [in] ldb leading dimension of B
 * @param [in] beta host or device scalar
 * @param [inout] C column-major matrix of size [m, n]
 * @param [in] ldc leading dimension of C
 * @param [in] stream
 */
template <typename A_t, typename B_t, typename C_t, typename S_t, bool DevicePointerMode = false>
void gemm(raft::resources const& handle,
          const bool trans_a,
          const bool trans_b,
          const int m,
          const int n,
          const int k,
          const S_t* alpha,
          const A_t* A,
          const int lda,
          const B_t* B,
          const int ldb,
          const S_t* beta,
          C_t* C,
          const int ldc,
          cudaStream_t stream)
{
  return detail::legacy_gemm(
    handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

/**
 * @brief the wrapper of cublas gemm function
 *  It computes the following equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam A_t the element type of A
 * @tparam B_t the element type of B
 * @tparam C_t the element type of C
 * @tparam S_t the element type of alpha and beta
 * @param handle raft handle
 * @param a input matrix
 * @param n_rows_a number of rows of A
 * @param n_cols_a number of columns of A
 * @param b input matrix
 * @param c output matrix
 * @param n_rows_c number of rows of C
 * @param n_cols_c number of columns of C
 * @param trans_a cublas transpose op for A
 * @param trans_b cublas transpose op for B
 * @param alpha scalar
 * @param beta scalar
 * @param stream cuda stream
 */
template <typename A_t, typename B_t, typename C_t, typename S_t>
void gemm(raft::resources const& handle,
          const A_t* a,
          int n_rows_a,
          int n_cols_a,
          const B_t* b,
          C_t* c,
          int n_rows_c,
          int n_cols_c,
          cublasOperation_t trans_a,
          cublasOperation_t trans_b,
          S_t alpha,
          S_t beta,
          cudaStream_t stream)
{
  detail::legacy_gemm(
    handle, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, alpha, beta, stream);
}

/**
 * @brief the wrapper of cublas gemm function
 *  It computes the following equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam A_t the element type of A
 * @tparam B_t the element type of B
 * @tparam C_t the element type of C
 * @param handle raft handle
 * @param a input matrix
 * @param n_rows_a number of rows of A
 * @param n_cols_a number of columns of A
 * @param b input matrix
 * @param c output matrix
 * @param n_rows_c number of rows of C
 * @param n_cols_c number of columns of C
 * @param trans_a cublas transpose op for A
 * @param trans_b cublas transpose op for B
 * @param stream cuda stream
 */
template <typename A_t, typename B_t, typename C_t>
void gemm(raft::resources const& handle,
          const A_t* a,
          int n_rows_a,
          int n_cols_a,
          const B_t* b,
          C_t* c,
          int n_rows_c,
          int n_cols_c,
          cublasOperation_t trans_a,
          cublasOperation_t trans_b,
          cudaStream_t stream)
{
  detail::legacy_gemm(
    handle, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, stream);
}

/**
 * @brief A wrapper for CUBLS GEMM function designed for handling all possible
 * combinations of operand layouts.
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * @tparam z_T the element type of z
 * @tparam x_T the element type of x
 * @tparam y_T the element type of y
 * @tparam s_T the element type of alpha and beta, equal to z_T by default
 * @param handle raft handle
 * @param z output matrix of size M rows x N columns
 * @param x input matrix of size M rows x K columns
 * @param y input matrix of size K rows x N columns
 * @param _M number of rows of X and Z
 * @param _N number of columns of Y and columns of Z
 * @param _K number of columns of X and rows of Y
 * @param isZColMajor Storage layout of Z. true = col major, false = row major
 * @param isXColMajor Storage layout of X. true = col major, false = row major
 * @param isYColMajor Storage layout of Y. true = col major, false = row major
 * @param stream cuda stream
 * @param alpha scalar
 * @param beta scalar
 */
template <typename z_T, typename x_T, typename y_T, typename s_T = z_T>
void gemm(raft::resources const& handle,
          z_T* z,
          x_T* x,
          y_T* y,
          int _M,
          int _N,
          int _K,
          bool isZColMajor,
          bool isXColMajor,
          bool isYColMajor,
          cudaStream_t stream,
          s_T alpha = s_T(1.0),
          s_T beta  = s_T(0.0))
{
  return detail::legacy_gemm<x_T, y_T, z_T, s_T, false>(
    handle, z, x, y, _M, _N, _K, isZColMajor, isXColMajor, isYColMajor, stream, &alpha, &beta);
}

/**
 * @defgroup gemm Matrix-Matrix Multiplication
 * @{
 */

/**
 * @brief GEMM function designed for handling all possible
 * combinations of operand layouts (raft::row_major or raft::col_major)
 * with scalars alpha and beta on the host or device
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * If alpha is not provided, it is assumed to be 1.0
 * If beta is not provided, it is assumed to be 0.0
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of index
 * @tparam LayoutPolicyX layout of X
 * @tparam LayoutPolicyY layout of Y
 * @tparam LayoutPolicyZ layout of Z
 * @param[in] res raft handle
 * @param[in] x input raft::device_matrix_view of size M rows x K columns
 * @param[in] y input raft::device_matrix_view of size K rows x N columns
 * @param[out] z output raft::device_matrix_view of size M rows x N columns
 * @param[in] alpha optional raft::host_scalar_view or raft::device_scalar_view, default 1.0
 * @param[in] beta optional raft::host_scalar_view or raft::device_scalar_view, default 0.0
 */
template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyX,
          typename LayoutPolicyY,
          typename LayoutPolicyZ,
          typename ScalarIdxType  = std::uint32_t,
          typename ScalarViewType = raft::host_scalar_view<ValueType, ScalarIdxType>,
          typename                = std::enable_if_t<std::disjunction_v<
                           std::is_same<ScalarViewType, raft::host_scalar_view<ValueType, ScalarIdxType>>,
                           std::is_same<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>>>>
void gemm(raft::resources const& res,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyX> x,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyY> y,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyZ> z,
          std::optional<ScalarViewType> alpha = std::nullopt,
          std::optional<ScalarViewType> beta  = std::nullopt)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(x), "X is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(y), "Y is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(z), "Z is not contiguous");

  RAFT_EXPECTS(x.extent(0) == z.extent(0), "Number of rows of X and Z should be equal");
  RAFT_EXPECTS(y.extent(1) == z.extent(1), "Number of columns of Y and Z should be equal");
  RAFT_EXPECTS(x.extent(1) == y.extent(0), "Number of columns of X and rows of Y should be equal");

  constexpr auto kXColMajor = std::is_same_v<typename decltype(x)::layout_type, raft::col_major>;
  constexpr auto kYColMajor = std::is_same_v<typename decltype(y)::layout_type, raft::col_major>;
  constexpr auto kZColMajor = std::is_same_v<typename decltype(z)::layout_type, raft::col_major>;

  // NB: the function type constraints only ever allow two view types, so using std::is_same_v is
  // fine
  constexpr auto kDeviceMode =
    std::is_same_v<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>;

  // NB: we rely on the implementation of detail::matmul to set defaults
  ValueType* alpha_ptr = nullptr;
  ValueType* beta_ptr  = nullptr;
  if (alpha.has_value()) { alpha_ptr = alpha.value().data_handle(); }
  if (beta.has_value()) { beta_ptr = beta.value().data_handle(); }

  if constexpr (kZColMajor) {
    return detail::matmul<kDeviceMode, ValueType, ValueType, ValueType, ValueType>(
      res,
      !kXColMajor,
      !kYColMajor,
      static_cast<uint64_t>(z.extent(0)),
      static_cast<uint64_t>(z.extent(1)),
      static_cast<uint64_t>(x.extent(1)),
      alpha_ptr,
      x.data_handle(),
      static_cast<uint64_t>(x.extent(kXColMajor ? 0 : 1)),
      y.data_handle(),
      static_cast<uint64_t>(y.extent(kYColMajor ? 0 : 1)),
      beta_ptr,
      z.data_handle(),
      static_cast<uint64_t>(z.extent(0)));
  } else {
    return detail::matmul<kDeviceMode, ValueType, ValueType, ValueType, ValueType>(
      res,
      kYColMajor,
      kXColMajor,
      static_cast<uint64_t>(z.extent(1)),
      static_cast<uint64_t>(z.extent(0)),
      static_cast<uint64_t>(x.extent(1)),
      alpha_ptr,
      y.data_handle(),
      static_cast<uint64_t>(y.extent(kYColMajor ? 0 : 1)),
      x.data_handle(),
      static_cast<uint64_t>(x.extent(kXColMajor ? 0 : 1)),
      beta_ptr,
      z.data_handle(),
      static_cast<uint64_t>(z.extent(1)));
  }
}

/**
 * @brief Batched matrix multiplication with an explicit cublasLt compute type.
 *
 * Computes `Z_i = alpha * X_i * Y_i + beta * Z_i`. The first mdspan extent is the batch
 * dimension. `layout_stride` views encode matrix and batch strides in their mappings; contiguous
 * batches use `layout_right`, with the matrix size as the batch stride. Use `layout_stride` for
 * column-major batches because rank-3 `layout_left` interleaves the batch dimension.
 *
 * @tparam ValueType element type of the input and output matrices
 * @tparam IndexType index type
 * @tparam LayoutPolicyX layout policy of X
 * @tparam LayoutPolicyY layout policy of Y
 * @tparam LayoutPolicyZ layout policy of Z
 * @tparam ScalarIdxType index type of alpha and beta
 * @tparam ScalarViewType scalar view type of alpha and beta
 * @param[in] res RAFT resources
 * @param[in] x input matrices X, with shape `[batch_count, m, k]`
 * @param[in] y input matrices Y, with shape `[batch_count, k, n]`
 * @param[inout] z input/output matrices Z, with shape `[batch_count, m, n]`
 * @param[in] alpha scalar multiplier for X * Y; defaults to one when empty
 * @param[in] beta scalar multiplier for Z; defaults to zero when empty
 * @param[in] compute_type_override cublasLt compute type
 */
template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyX,
          typename LayoutPolicyY,
          typename LayoutPolicyZ,
          typename ScalarIdxType  = std::uint32_t,
          typename ScalarViewType = raft::host_scalar_view<ValueType, ScalarIdxType>,
          typename                = std::enable_if_t<std::disjunction_v<
                           std::is_same<ScalarViewType, raft::host_scalar_view<ValueType, ScalarIdxType>>,
                           std::is_same<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>>>>
void gemm_batched(raft::resources const& res,
                  raft::device_mdspan<ValueType, raft::extent_3d<IndexType>, LayoutPolicyX> x,
                  raft::device_mdspan<ValueType, raft::extent_3d<IndexType>, LayoutPolicyY> y,
                  raft::device_mdspan<ValueType, raft::extent_3d<IndexType>, LayoutPolicyZ> z,
                  std::optional<ScalarViewType> alpha,
                  std::optional<ScalarViewType> beta,
                  cublasComputeType_t compute_type_override)
{
  static_assert(!std::is_same_v<LayoutPolicyX, raft::layout_left> &&
                  !std::is_same_v<LayoutPolicyY, raft::layout_left> &&
                  !std::is_same_v<LayoutPolicyZ, raft::layout_left>,
                "Use layout_stride for column-major batched matrices");

  RAFT_EXPECTS(x.extent(0) == y.extent(0) && x.extent(0) == z.extent(0),
               "All operands must have the same batch count");
  RAFT_EXPECTS(x.extent(1) == z.extent(1), "X and Z must have the same number of rows");
  RAFT_EXPECTS(y.extent(2) == z.extent(2), "Y and Z must have the same number of columns");
  RAFT_EXPECTS(x.extent(2) == y.extent(1),
               "The number of columns in X must equal the number of rows in Y");

  const auto is_row_major = [](auto const& view) {
    return view.stride(2) == 1 && view.stride(1) >= view.extent(2);
  };
  const auto is_col_major = [](auto const& view) {
    return view.stride(1) == 1 && view.stride(2) >= view.extent(1);
  };
  const auto batch_stride = [](auto const& view) -> int64_t {
    using layout_type = typename std::decay_t<decltype(view)>::layout_type;
    if constexpr (std::is_same_v<layout_type, raft::layout_stride>) {
      return static_cast<int64_t>(view.stride(0));
    } else {
      return static_cast<int64_t>(view.extent(1)) * static_cast<int64_t>(view.extent(2));
    }
  };
  RAFT_EXPECTS(is_row_major(x) || is_col_major(x),
               "Each matrix in X must be row-major or column-major");
  RAFT_EXPECTS(is_row_major(y) || is_col_major(y),
               "Each matrix in Y must be row-major or column-major");
  RAFT_EXPECTS(is_row_major(z) || is_col_major(z),
               "Each matrix in Z must be row-major or column-major");

  constexpr auto kDeviceMode =
    std::is_same_v<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>;
  ValueType* alpha_ptr = nullptr;
  ValueType* beta_ptr  = nullptr;
  if (alpha.has_value()) { alpha_ptr = alpha->data_handle(); }
  if (beta.has_value()) { beta_ptr = beta->data_handle(); }

  const bool x_col_major = is_col_major(x);
  const bool y_col_major = is_col_major(y);
  const bool z_col_major = is_col_major(z);
  const auto x_ld        = x_col_major ? x.stride(2) : x.stride(1);
  const auto y_ld        = y_col_major ? y.stride(2) : y.stride(1);
  const auto z_ld        = z_col_major ? z.stride(2) : z.stride(1);
  const auto batch_count = static_cast<int32_t>(z.extent(0));

  if (z_col_major) {
    return detail::matmul_strided_batched<kDeviceMode>(res,
                                                       !x_col_major,
                                                       !y_col_major,
                                                       z.extent(1),
                                                       z.extent(2),
                                                       x.extent(2),
                                                       alpha_ptr,
                                                       x.data_handle(),
                                                       x_ld,
                                                       batch_stride(x),
                                                       y.data_handle(),
                                                       y_ld,
                                                       batch_stride(y),
                                                       beta_ptr,
                                                       z.data_handle(),
                                                       z_ld,
                                                       batch_stride(z),
                                                       batch_count,
                                                       compute_type_override);
  }

  return detail::matmul_strided_batched<kDeviceMode>(res,
                                                     y_col_major,
                                                     x_col_major,
                                                     z.extent(2),
                                                     z.extent(1),
                                                     x.extent(2),
                                                     alpha_ptr,
                                                     y.data_handle(),
                                                     y_ld,
                                                     batch_stride(y),
                                                     x.data_handle(),
                                                     x_ld,
                                                     batch_stride(x),
                                                     beta_ptr,
                                                     z.data_handle(),
                                                     z_ld,
                                                     batch_stride(z),
                                                     batch_count,
                                                     compute_type_override);
}

/**
 * @brief Batched matrix multiplication using the default cublasLt compute type.
 *
 * Computes `Z_i = alpha * X_i * Y_i + beta * Z_i`. The first mdspan extent is the batch
 * dimension. `layout_stride` views encode matrix and batch strides in their mappings; contiguous
 * batches use `layout_right`, with the matrix size as the batch stride. Use `layout_stride` for
 * column-major batches because rank-3 `layout_left` interleaves the batch dimension.
 *
 * @tparam ValueType element type of the input and output matrices
 * @tparam IndexType index type
 * @tparam LayoutPolicyX layout policy of X
 * @tparam LayoutPolicyY layout policy of Y
 * @tparam LayoutPolicyZ layout policy of Z
 * @tparam ScalarIdxType index type of alpha and beta
 * @tparam ScalarViewType scalar view type of alpha and beta
 * @param[in] res RAFT resources
 * @param[in] x input matrices X, with shape `[batch_count, m, k]`
 * @param[in] y input matrices Y, with shape `[batch_count, k, n]`
 * @param[inout] z input/output matrices Z, with shape `[batch_count, m, n]`
 * @param[in] alpha scalar multiplier for X * Y; defaults to one when empty
 * @param[in] beta scalar multiplier for Z; defaults to zero when empty
 */
template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyX,
          typename LayoutPolicyY,
          typename LayoutPolicyZ,
          typename ScalarIdxType  = std::uint32_t,
          typename ScalarViewType = raft::host_scalar_view<ValueType, ScalarIdxType>,
          typename                = std::enable_if_t<std::disjunction_v<
                           std::is_same<ScalarViewType, raft::host_scalar_view<ValueType, ScalarIdxType>>,
                           std::is_same<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>>>>
void gemm_batched(raft::resources const& res,
                  raft::device_mdspan<ValueType, raft::extent_3d<IndexType>, LayoutPolicyX> x,
                  raft::device_mdspan<ValueType, raft::extent_3d<IndexType>, LayoutPolicyY> y,
                  raft::device_mdspan<ValueType, raft::extent_3d<IndexType>, LayoutPolicyZ> z,
                  std::optional<ScalarViewType> alpha = std::nullopt,
                  std::optional<ScalarViewType> beta  = std::nullopt)
{
  return gemm_batched(res,
                      x,
                      y,
                      z,
                      alpha,
                      beta,
                      detail::get_matmul_type<ValueType, ValueType, ValueType, ValueType>());
}

/** @} */  // end of gemm

}  // namespace linalg
}  // namespace raft

#endif
