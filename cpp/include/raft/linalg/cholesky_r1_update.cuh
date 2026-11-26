/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __CHOLESKY_R1_UPDATE_H
#define __CHOLESKY_R1_UPDATE_H

#pragma once

#include "detail/cholesky_r1_update.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cublas_handle.hpp>

namespace raft {
namespace linalg {

/**
 * @brief Rank 1 update of Cholesky decomposition.
 * NOTE: The new mdspan-based API will not be provided for this function.
 *
 * This method is useful if an algorithm iteratively builds up matrix A, and
 * the Cholesky decomposition of A is required at each step.
 *
 * On entry, L is the Cholesky decomposition of matrix A, where both A and L
 * have size n-1 x n-1. We are interested in the Cholesky decomposition of a new
 * matrix A', which we get by adding a row and column to A. In Python notation:
 * - A'[0:n-1, 0:n-1] = A;
 * - A'[:,n-1] = A[n-1,:] = A_new
 *
 * On entry, the new column A_new, is stored as the n-th column of L if uplo ==
 * CUBLAS_FILL_MODE_UPPER, else A_new is stored as the n-th row of L.
 *
 * On exit L contains the Cholesky decomposition of A'. In practice the elements
 * of A_new are overwritten with new row/column of the L matrix.
 *
 * The uplo parameter is used to select the matrix layout.
 * If (uplo != CUBLAS_FILL_MODE_UPPER) then the input arg L stores the
 * lower triangular matrix L, so that A = L * L.T. Otherwise the input arg L
 * stores an upper triangular matrix U: A = U.T * U.
 *
 * On exit L will be updated to store the Cholesky decomposition of A'.
 *
 * If the matrix is not positive definite, or very ill conditioned then the new
 * diagonal element of L would be NaN. In such a case an exception is thrown.
 * The eps argument can be used to override this behavior: if eps >= 0 then
 * the diagonal element is replaced by eps in case the diagonal is NaN or
 * smaller than eps. Note: for an iterative solver it is probably better to
 * stop early in case of error, rather than relying on the eps parameter.
 *
 * Examples:
 *
 * - Lower triangular factorization:
 * @code{.cpp}
 * // Initialize arrays
 * int ld_L = n_rows;
 * rmm::device_uvector<math_t> L(ld_L * n_rows, stream);
 * raft::linalg::choleskyRank1Update(handle, L, n_rows, ld_L, nullptr,
 *                                       &n_bytes, CUBLAS_FILL_MODE_LOWER,
 *                                       stream);
 * rmm::device_uvector<char> workspace(n_bytes, stream);
 *
 * for (n=1; n<=n_rows; n++) {
 *   // Calculate a new row/column of matrix A into A_new
 *   // ...
 *   // Copy new row to L[n-1,:]
 *   RAFT_CUBLAS_TRY(cublasCopy(resource::get_cublas_handle(handle), n - 1, A_new, 1,
 *                           L + n - 1, ld_L, stream));
 *   // Update Cholesky factorization
 *   raft::linalg::choleskyRank1Update(
 *       handle, L, n, ld_L, workspace, &n_bytes, CUBLAS_FILL_MODE_LOWER,
 *       stream);
 * }
 * Now L stores the Cholesky decomposition of A: A = L * L.T
 * @endcode
 *
 * - Upper triangular factorization:
 * @code{.cpp}
 * // Initialize arrays
 * int ld_U = n_rows;
 * rmm::device_uvector<math_t> U(ld_U * n_rows, stream);
 * raft::linalg::choleskyRank1Update(handle, U, n_rows, ld_U, nullptr,
 *                                       &n_bytes, CUBLAS_FILL_MODE_UPPER,
 *                                       stream);
 * rmm::device_uvector<char> workspace(n_bytes, stream);
 *
 * for (n=1; n<=n_rows; n++) {
 *   // Calculate a new row/column of matrix A into array A_new
 *   // ...
 *   // Copy new row to U[:,n-1] (column major layout)
 *   raft::copy(U + ld_U * (n-1), A_new, n-1, stream);
 *   //
 *   // Update Cholesky factorization
 *   raft::linalg::choleskyRank1Update(
 *       handle, U, n, ld_U, workspace, &n_bytes, CUBLAS_FILL_MODE_UPPER,
 *       stream);
 * }
 * // Now U stores the Cholesky decomposition of A: A = U.T * U
 * @endcode
 *
 * @param handle RAFT handle (used to retrieve cuBLAS handles).
 * @param L device array for to store the triangular matrix L, and the new
 *     column of A in column major format, size [n*n]
 * @param n number of elements in the new row.
 * @param ld stride of columns in L
 * @param workspace device pointer to workspace shall be nullptr ar an array
 *    of size [n_bytes].
 * @param n_bytes size of workspace is returned here if workspace==nullptr.
 * @param stream CUDA stream
 * @param uplo indicates whether L is stored as an upper or lower triangular
 *    matrix (CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER)
 * @param eps numerical parameter that can act as a regularizer for ill
 *    conditioned systems. Negative values mean no regularizaton.
 */
template <typename math_t>
void choleskyRank1Update(raft::resources const& handle,
                         math_t* L,
                         int n,
                         int ld,
                         void* workspace,
                         int* n_bytes,
                         cublasFillMode_t uplo,
                         cudaStream_t stream,
                         math_t eps = -1)
{
  detail::choleskyRank1Update(handle, L, n, ld, workspace, n_bytes, uplo, stream, eps);
}

};  // namespace linalg
};  // namespace raft

#endif
