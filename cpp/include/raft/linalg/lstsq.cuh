/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __LSTSQ_H
#define __LSTSQ_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/lstsq.cuh>
namespace raft {
namespace linalg {

/** Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S Vt` using default cuSOLVER routine.
 *
 * @param[in] handle raft handle
 * @param[inout] A input feature matrix.
 *            Warning: the content of this matrix is modified by the cuSOLVER routines.
 * @param[in] n_rows number of rows in A
 * @param[in] n_cols number of columns in A
 * @param[inout] b input target vector.
 *            Warning: the content of this vector is modified by the cuSOLVER routines.
 * @param[out] w output coefficient vector
 * @param[in] stream cuda stream for ordering operations
 */
template <typename math_t>
void lstsqSvdQR(raft::resources const& handle,
                math_t* A,
                const int n_rows,
                const int n_cols,
                const math_t* b,
                math_t* w,
                cudaStream_t stream)
{
  detail::lstsqSvdQR(handle, A, n_rows, n_cols, b, w, stream);
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S V^T` using Jacobi iterations (cuSOLVER).
 *
 * @param[in] handle raft handle
 * @param[inout] A input feature matrix.
 *            Warning: the content of this matrix is modified by the cuSOLVER routines.
 * @param[in] n_rows number of rows in A
 * @param[in] n_cols number of columns in A
 * @param[inout] b input target vector.
 *            Warning: the content of this vector is modified by the cuSOLVER routines.
 * @param[out] w output coefficient vector
 * @param[in] stream cuda stream for ordering operations
 */
template <typename math_t>
void lstsqSvdJacobi(raft::resources const& handle,
                    math_t* A,
                    const int n_rows,
                    const int n_cols,
                    const math_t* b,
                    math_t* w,
                    cudaStream_t stream)
{
  detail::lstsqSvdJacobi(handle, A, n_rows, n_cols, b, w, stream);
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  via eigenvalue decomposition of `A^T * A` (covariance matrix for dataset A).
 *  (`w = (A^T A)^-1  A^T b`)
 */
template <typename math_t>
void lstsqEig(raft::resources const& handle,
              const math_t* A,
              const int n_rows,
              const int n_cols,
              const math_t* b,
              math_t* w,
              cudaStream_t stream)
{
  detail::lstsqEig(handle, A, n_rows, n_cols, b, w, stream);
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  via QR decomposition of `A = QR`.
 *  (triangular system of equations `Rw = Q^T b`)
 *
 * @param[in] handle raft handle
 * @param[inout] A input feature matrix.
 *            Warning: the content of this matrix is modified by the cuSOLVER routines.
 * @param[in] n_rows number of rows in A
 * @param[in] n_cols number of columns in A
 * @param[inout] b input target vector.
 *            Warning: the content of this vector is modified by the cuSOLVER routines.
 * @param[out] w output coefficient vector
 * @param[in] stream cuda stream for ordering operations
 */
template <typename math_t>
void lstsqQR(raft::resources const& handle,
             math_t* A,
             const int n_rows,
             const int n_cols,
             math_t* b,
             math_t* w,
             cudaStream_t stream)
{
  detail::lstsqQR(handle, A, n_rows, n_cols, b, w, stream);
}

/**
 * @defgroup lstsq Least Squares Methods
 * @{
 */

/**
 * @brief Solves the linear ordinary least squares problem `Aw = b`
 * Via SVD decomposition of `A = U S Vt`.
 *
 * @tparam ValueType the data-type of input/output
 * @param[in] handle raft::resources
 * @param[inout] A input raft::device_matrix_view
 *            Warning: the content of this matrix is modified.
 * @param[inout] b input target raft::device_vector_view
 *            Warning: the content of this vector is modified.
 * @param[out] w output coefficient raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void lstsq_svd_qr(raft::resources const& handle,
                  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> A,
                  raft::device_vector_view<const ValueType, IndexType> b,
                  raft::device_vector_view<ValueType, IndexType> w)
{
  RAFT_EXPECTS(A.extent(1) == w.size(), "Size mismatch between A and w");
  RAFT_EXPECTS(A.extent(0) == b.size(), "Size mismatch between A and b");

  lstsqSvdQR(handle,
             const_cast<ValueType*>(A.data_handle()),
             A.extent(0),
             A.extent(1),
             const_cast<ValueType*>(b.data_handle()),
             w.data_handle(),
             resource::get_cuda_stream(handle));
}

/**
 * @brief Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S V^T` using Jacobi iterations.
 *
 * @tparam ValueType the data-type of input/output
 * @param[in] handle raft::resources
 * @param[inout] A input raft::device_matrix_view
 *            Warning: the content of this matrix is modified.
 * @param[inout] b input target raft::device_vector_view
 *            Warning: the content of this vector is modified.
 * @param[out] w output coefficient raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void lstsq_svd_jacobi(raft::resources const& handle,
                      raft::device_matrix_view<const ValueType, IndexType, raft::col_major> A,
                      raft::device_vector_view<const ValueType, IndexType> b,
                      raft::device_vector_view<ValueType, IndexType> w)
{
  RAFT_EXPECTS(A.extent(1) == w.size(), "Size mismatch between A and w");
  RAFT_EXPECTS(A.extent(0) == b.size(), "Size mismatch between A and b");

  lstsqSvdJacobi(handle,
                 const_cast<ValueType*>(A.data_handle()),
                 A.extent(0),
                 A.extent(1),
                 const_cast<ValueType*>(b.data_handle()),
                 w.data_handle(),
                 resource::get_cuda_stream(handle));
}

/**
 * @brief Solves the linear ordinary least squares problem `Aw = b`
 *  via eigenvalue decomposition of `A^T * A` (covariance matrix for dataset A).
 *  (`w = (A^T A)^-1  A^T b`)
 *
 * @tparam ValueType the data-type of input/output
 * @param[in] handle raft::resources
 * @param[in] A input raft::device_matrix_view
 * @param[in] b input target raft::device_vector_view
 * @param[out] w output coefficient raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void lstsq_eig(raft::resources const& handle,
               raft::device_matrix_view<const ValueType, IndexType, raft::col_major> A,
               raft::device_vector_view<const ValueType, IndexType> b,
               raft::device_vector_view<ValueType, IndexType> w)
{
  RAFT_EXPECTS(A.extent(1) == w.size(), "Size mismatch between A and w");
  RAFT_EXPECTS(A.extent(0) == b.size(), "Size mismatch between A and b");

  lstsqEig(handle,
           const_cast<ValueType*>(A.data_handle()),
           A.extent(0),
           A.extent(1),
           const_cast<ValueType*>(b.data_handle()),
           w.data_handle(),
           resource::get_cuda_stream(handle));
}

/**
 * @brief Solves the linear ordinary least squares problem `Aw = b`
 *  via QR decomposition of `A = QR`.
 *  (triangular system of equations `Rw = Q^T b`)
 *
 * @tparam ValueType the data-type of input/output
 * @param[in] handle raft::resources
 * @param[inout] A input raft::device_matrix_view
 *            Warning: the content of this matrix is modified.
 * @param[inout] b input target raft::device_vector_view
 *            Warning: the content of this vector is modified.
 * @param[out] w output coefficient raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void lstsq_qr(raft::resources const& handle,
              raft::device_matrix_view<const ValueType, IndexType, raft::col_major> A,
              raft::device_vector_view<const ValueType, IndexType> b,
              raft::device_vector_view<ValueType, IndexType> w)
{
  RAFT_EXPECTS(A.extent(1) == w.size(), "Size mismatch between A and w");
  RAFT_EXPECTS(A.extent(0) == b.size(), "Size mismatch between A and b");

  lstsqQR(handle,
          const_cast<ValueType*>(A.data_handle()),
          A.extent(0),
          A.extent(1),
          const_cast<ValueType*>(b.data_handle()),
          w.data_handle(),
          resource::get_cuda_stream(handle));
}

/** @} */  // end of lstsq

};  // namespace linalg
};  // namespace raft

#endif
