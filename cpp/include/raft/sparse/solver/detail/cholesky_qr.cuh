/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/copy.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace raft::sparse::solver::detail {

/**
 * @brief Single pass of CholeskyQR: orthogonalize Q in-place via Cholesky factorization
 *        of the Gram matrix W = Q^T @ Q.
 *
 * @return true on success, false if Cholesky factorization failed (matrix not SPD / rank deficient)
 */
template <typename ValueTypeT>
bool cholesky_qr_pass(raft::resources const& handle,
                      ValueTypeT* Q,
                      int m,
                      int k,
                      ValueTypeT* W,
                      ValueTypeT* workspace,
                      int workspace_size,
                      int* dev_info)
{
  auto stream     = raft::resource::get_cuda_stream(handle);
  auto cublas_h   = raft::resource::get_cublas_handle(handle);
  auto cusolver_h = raft::resource::get_cusolver_dn_handle(handle);

  const ValueTypeT one  = 1;
  const ValueTypeT zero = 0;

  // W = Q^T @ Q  (k x k)
  // Q is col-major (m x k), so: W = Q^T * Q via gemm(TRANS, NOTRANS, k, k, m)
  raft::linalg::gemm(handle,
                     true,   // trans_a
                     false,  // trans_b
                     k,
                     k,
                     m,
                     &one,
                     Q,
                     m,
                     Q,
                     m,
                     &zero,
                     W,
                     k,
                     stream);

  if (raft::resource::get_dry_run_flag(handle)) {
    // Return false to force fallback to QR in dry run mode to track its memory usage.
    // There are no more allocations to track below this point - safe to return early.
    return false;
  }

  // L = cholesky(W, LOWER)  — W is overwritten with L in lower triangle
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf(
    cusolver_h, CUBLAS_FILL_MODE_LOWER, k, W, k, workspace, workspace_size, dev_info, stream));

  // Check if Cholesky succeeded
  int h_dev_info = 0;
  raft::update_host(&h_dev_info, dev_info, 1, stream);
  raft::resource::sync_stream(handle);
  if (h_dev_info != 0) { return false; }

  // Q = Q @ L^{-T}
  // This is equivalent to solving X * L^T = Q for X, i.e. trsm with RIGHT, LOWER, TRANS
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublastrsm(cublas_h,
                                                   CUBLAS_SIDE_RIGHT,
                                                   CUBLAS_FILL_MODE_LOWER,
                                                   CUBLAS_OP_T,
                                                   CUBLAS_DIAG_NON_UNIT,
                                                   m,
                                                   k,
                                                   &one,
                                                   W,
                                                   k,
                                                   Q,
                                                   m,
                                                   stream));

  return true;
}

/**
 * @brief CholeskyQR2 orthogonalization: two passes of CholeskyQR for numerical stability.
 *
 * This is the GPU-optimized orthogonalization from Tomás, Quintana-Ortí, Anzt (2024),
 * "Fast Truncated SVD of Sparse and Dense Matrices on Graphics Processors".
 * It uses GEMM + Cholesky + TRSM operations which are highly efficient on GPU,
 * providing ~3x speedup over standard Householder QR.
 *
 * If Cholesky factorization fails (input is rank-deficient), falls back to standard QR.
 *
 * @param handle raft resources handle
 * @param Q matrix to orthogonalize of shape (m, k), col-major, modified in-place
 * @return true if CholeskyQR2 succeeded, false if fell back to QR
 */
template <typename ValueTypeT>
bool cholesky_qr2(raft::resources const& handle,
                  raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> Q)
{
  int m = Q.extent(0);
  int k = Q.extent(1);

  auto stream     = raft::resource::get_cuda_stream(handle);
  auto cusolver_h = raft::resource::get_cusolver_dn_handle(handle);

  // Allocate workspace for Gram matrix and Cholesky
  rmm::device_uvector<ValueTypeT> W(k * k, stream);
  rmm::device_scalar<int> dev_info(stream);

  // Query workspace size for potrf
  int potrf_workspace_size = 0;
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf_bufferSize(
    cusolver_h, CUBLAS_FILL_MODE_LOWER, k, W.data(), k, &potrf_workspace_size));
  rmm::device_uvector<ValueTypeT> potrf_workspace(potrf_workspace_size, stream);

  // First pass
  if (!cholesky_qr_pass(handle,
                        Q.data_handle(),
                        m,
                        k,
                        W.data(),
                        potrf_workspace.data(),
                        potrf_workspace_size,
                        dev_info.data())) {
    // Fallback to standard QR (qrGetQ handles src==dst via internal copy)
    raft::linalg::qrGetQ(handle, Q.data_handle(), Q.data_handle(), m, k, stream);
    return false;
  }

  // Second pass for improved numerical stability
  if (!cholesky_qr_pass(handle,
                        Q.data_handle(),
                        m,
                        k,
                        W.data(),
                        potrf_workspace.data(),
                        potrf_workspace_size,
                        dev_info.data())) {
    // Fallback to standard QR (qrGetQ handles src==dst via internal copy)
    raft::linalg::qrGetQ(handle, Q.data_handle(), Q.data_handle(), m, k, stream);
    return false;
  }

  return true;
}

}  // namespace raft::sparse::solver::detail
