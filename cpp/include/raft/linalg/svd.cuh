/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __SVD_H
#define __SVD_H

#pragma once

#include "detail/svd.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <optional>

namespace raft {
namespace linalg {

/**
 * @brief singular value decomposition (SVD) on the column major float type
 * input matrix using QR method
 * @param handle: raft handle
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular values of input matrix
 * @param right_sing_vecs: right singular values of input matrix
 * @param trans_right: transpose right vectors or not
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @param stream cuda stream
 */
template <typename T>
void svdQR(raft::resources const& handle,
           T* in,
           int n_rows,
           int n_cols,
           T* sing_vals,
           T* left_sing_vecs,
           T* right_sing_vecs,
           bool trans_right,
           bool gen_left_vec,
           bool gen_right_vec,
           cudaStream_t stream)
{
  detail::svdQR(handle,
                in,
                n_rows,
                n_cols,
                sing_vals,
                left_sing_vecs,
                right_sing_vecs,
                trans_right,
                gen_left_vec,
                gen_right_vec,
                stream);
}

template <typename math_t, typename idx_t>
void svdEig(raft::resources const& handle,
            math_t* in,
            idx_t n_rows,
            idx_t n_cols,
            math_t* S,
            math_t* U,
            math_t* V,
            bool gen_left_vec,
            cudaStream_t stream)
{
  detail::svdEig(handle, in, n_rows, n_cols, S, U, V, gen_left_vec, stream);
}

/**
 * @brief on the column major input matrix using Jacobi method
 * @param handle: raft handle
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular vectors of input matrix
 * @param right_sing_vecs: right singular vectors of input matrix
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the
 * error is below tol
 * @param max_sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 * @param stream cuda stream
 */
template <typename math_t>
void svdJacobi(raft::resources const& handle,
               math_t* in,
               int n_rows,
               int n_cols,
               math_t* sing_vals,
               math_t* left_sing_vecs,
               math_t* right_sing_vecs,
               bool gen_left_vec,
               bool gen_right_vec,
               math_t tol,
               int max_sweeps,
               cudaStream_t stream)
{
  detail::svdJacobi(handle,
                    in,
                    n_rows,
                    n_cols,
                    sing_vals,
                    left_sing_vecs,
                    right_sing_vecs,
                    gen_left_vec,
                    gen_right_vec,
                    tol,
                    max_sweeps,
                    stream);
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param handle: raft handle
 * @param U: left singular vectors of size n_rows x k
 * @param S: square matrix with singular values on its diagonal, k x k
 * @param V: right singular vectors of size n_cols x k
 * @param out: reconstructed matrix to be returned
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values
 * @param stream cuda stream
 */
template <typename math_t>
void svdReconstruction(raft::resources const& handle,
                       math_t* U,
                       math_t* S,
                       math_t* V,
                       math_t* out,
                       int n_rows,
                       int n_cols,
                       int k,
                       cudaStream_t stream)
{
  detail::svdReconstruction(handle, U, S, V, out, n_rows, n_cols, k, stream);
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param handle: raft handle
 * @param A_d: input matrix
 * @param U: left singular vectors of size n_rows x k
 * @param S_vec: singular values as a vector
 * @param V: right singular vectors of size n_cols x k
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values to be computed, 1.0 for normal SVD
 * @param tol: tolerance for the evaluation
 * @param stream cuda stream
 */
template <typename math_t>
bool evaluateSVDByL2Norm(raft::resources const& handle,
                         math_t* A_d,
                         math_t* U,
                         math_t* S_vec,
                         math_t* V,
                         int n_rows,
                         int n_cols,
                         int k,
                         math_t tol,
                         cudaStream_t stream)
{
  return detail::evaluateSVDByL2Norm(handle, A_d, U, S_vec, V, n_rows, n_cols, k, tol, stream);
}

/**
 * @defgroup svd Singular Value Decomposition
 * @{
 */

/**
 * @brief singular value decomposition (SVD) on a column major
 * matrix using QR decomposition
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @param[in] handle raft::resources
 * @param[in] in input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] sing_vals singular values raft::device_vector_view of shape (K)
 * @param[out] U std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, n)
 * @param[out] V std::optional right singular values of raft::device_matrix_view with
 * layout raft::col_major and dimensions (n, n)
 */
template <typename ValueType, typename IndexType>
void svd_qr(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
  raft::device_vector_view<ValueType, IndexType> sing_vals,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  ValueType* left_sing_vecs_ptr  = nullptr;
  ValueType* right_sing_vecs_ptr = nullptr;

  if (U) {
    RAFT_EXPECTS(in.extent(0) == U.value().extent(0) && in.extent(1) == U.value().extent(1),
                 "U should have dimensions m * n");
    left_sing_vecs_ptr = U.value().data_handle();
  }
  if (V) {
    RAFT_EXPECTS(in.extent(1) == V.value().extent(0) && in.extent(1) == V.value().extent(1),
                 "V should have dimensions n * n");
    right_sing_vecs_ptr = V.value().data_handle();
  }
  svdQR(handle,
        const_cast<ValueType*>(in.data_handle()),
        in.extent(0),
        in.extent(1),
        sing_vals.data_handle(),
        left_sing_vecs_ptr,
        right_sing_vecs_ptr,
        false,
        U.has_value(),
        V.has_value(),
        resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `svd_qr` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `svd_qr`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void svd_qr(raft::resources const& handle,
            raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
            raft::device_vector_view<ValueType, IndexType> sing_vals,
            UType&& U_in = std::nullopt,
            VType&& V_in = std::nullopt)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);

  svd_qr(handle, in, sing_vals, U, V);
}

/**
 * @brief singular value decomposition (SVD) on a column major
 * matrix using QR decomposition. Right singular vector matrix is transposed before returning
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @param[in] handle raft::resources
 * @param[in] in input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] sing_vals singular values raft::device_vector_view of shape (K)
 * @param[out] U std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, n)
 * @param[out] V std::optional right singular values of raft::device_matrix_view with
 * layout raft::col_major and dimensions (n, n)
 */
template <typename ValueType, typename IndexType>
void svd_qr_transpose_right_vec(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
  raft::device_vector_view<ValueType, IndexType> sing_vals,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V = std::nullopt)
{
  ValueType* left_sing_vecs_ptr  = nullptr;
  ValueType* right_sing_vecs_ptr = nullptr;

  if (U) {
    RAFT_EXPECTS(in.extent(0) == U.value().extent(0) && in.extent(1) == U.value().extent(1),
                 "U should have dimensions m * n");
    left_sing_vecs_ptr = U.value().data_handle();
  }
  if (V) {
    RAFT_EXPECTS(in.extent(1) == V.value().extent(0) && in.extent(1) == V.value().extent(1),
                 "V should have dimensions n * n");
    right_sing_vecs_ptr = V.value().data_handle();
  }
  svdQR(handle,
        const_cast<ValueType*>(in.data_handle()),
        in.extent(0),
        in.extent(1),
        sing_vals.data_handle(),
        left_sing_vecs_ptr,
        right_sing_vecs_ptr,
        true,
        U.has_value(),
        V.has_value(),
        resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `svd_qr_transpose_right_vec` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `svd_qr_transpose_right_vec`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void svd_qr_transpose_right_vec(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
  raft::device_vector_view<ValueType, IndexType> sing_vals,
  UType&& U_in = std::nullopt,
  VType&& V_in = std::nullopt)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);

  svd_qr_transpose_right_vec(handle, in, sing_vals, U, V);
}

/**
 * @brief singular value decomposition (SVD) on a column major
 * matrix using Eigen decomposition. A square symmetric covariance matrix is constructed for the SVD
 * @param[in] handle raft::resources
 * @param[in] in input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S singular values raft::device_vector_view of shape (K)
 * @param[out] V right singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (n, n)
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, n)
 */
template <typename ValueType, typename IndexType>
void svd_eig(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
  raft::device_vector_view<ValueType, IndexType> S,
  raft::device_matrix_view<ValueType, IndexType, raft::col_major> V,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt)
{
  ValueType* left_sing_vecs_ptr = nullptr;
  if (U) {
    RAFT_EXPECTS(in.extent(0) == U.value().extent(0) && in.extent(1) == U.value().extent(1),
                 "U should have dimensions m * n");
    left_sing_vecs_ptr = U.value().data_handle();
  }
  RAFT_EXPECTS(in.extent(1) == V.extent(0) && in.extent(1) == V.extent(1),
               "V should have dimensions n * n");
  svdEig(handle,
         const_cast<ValueType*>(in.data_handle()),
         in.extent(0),
         in.extent(1),
         S.data_handle(),
         left_sing_vecs_ptr,
         V.data_handle(),
         U.has_value(),
         resource::get_cuda_stream(handle).get());
}

template <typename ValueType, typename IndexType, typename UType>
void svd_eig(raft::resources const& handle,
             raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
             raft::device_vector_view<ValueType, IndexType> S,
             raft::device_matrix_view<ValueType, IndexType, raft::col_major> V,
             UType&& U = std::nullopt)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  svd_eig(handle, in, S, V, U_optional);
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param[in] handle raft::resources
 * @param[in] U left singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, k)
 * @param[in] S square matrix with singular values on its diagonal of shape (k, k)
 * @param[in] V right singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (k, n)
 * @param[out] out output raft::device_matrix_view with layout raft::col_major of shape (m, n)
 */
template <typename ValueType, typename IndexType>
void svd_reconstruction(raft::resources const& handle,
                        raft::device_matrix_view<const ValueType, IndexType, raft::col_major> U,
                        raft::device_matrix_view<const ValueType, IndexType, raft::col_major> S,
                        raft::device_matrix_view<const ValueType, IndexType, raft::col_major> V,
                        raft::device_matrix_view<ValueType, IndexType, raft::col_major> out)
{
  RAFT_EXPECTS(S.extent(0) == S.extent(1), "S should be a square matrix");
  RAFT_EXPECTS(S.extent(0) == U.extent(1),
               "Number of rows of S should be equal to number of columns in U");
  RAFT_EXPECTS(S.extent(1) == V.extent(0),
               "Number of columns of S should be equal to number of rows in V");
  RAFT_EXPECTS(out.extent(0) == U.extent(0) && out.extent(1) == V.extent(1),
               "Number of rows should be equal in out and U and number of columns should be equal "
               "in out and V");

  svdReconstruction(handle,
                    const_cast<ValueType*>(U.data_handle()),
                    const_cast<ValueType*>(S.data_handle()),
                    const_cast<ValueType*>(V.data_handle()),
                    out.data_handle(),
                    out.extent(0),
                    out.extent(1),
                    S.extent(0),
                    resource::get_cuda_stream(handle).get());
}

/** @} */  // end of group svd

};  // end namespace linalg
};  // end namespace raft

#endif
