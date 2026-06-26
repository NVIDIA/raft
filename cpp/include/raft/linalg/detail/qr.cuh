/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cublas_wrappers.hpp"
#include "cusolver_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/triangular.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace raft {
namespace linalg {
namespace detail {

/**
 * @brief Calculate the QR decomposition and get matrix Q in place of the input.
 *
 * Subject to the algorithm constraint `n_rows >= n_cols`.
 *
 * @param handle
 * @param[inout] Q device pointer to input matrix and the output matrix Q,
 *                 both column-major and of size [n_rows, n_cols].
 * @param n_rows
 * @param n_cols
 * @param stream
 */
template <typename math_t>
void qrGetQ_inplace(
  raft::resources const& handle, math_t* Q, int n_rows, int n_cols, cudaStream_t stream)
{
  RAFT_EXPECTS(n_rows >= n_cols, "QR decomposition expects n_rows >= n_cols.");
  cusolverDnHandle_t cusolver = resource::get_cusolver_dn_handle(handle);
  auto is_dry_run             = resource::get_dry_run_flag(handle);

  rmm::device_uvector<math_t> tau(n_cols, stream);
  if (!is_dry_run) {
    RAFT_CUDA_TRY(cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * n_cols, stream));
  }

  rmm::device_scalar<int> dev_info(stream);
  int ws_size_Dngeqrf;
  int ws_size_Dnorgqr;

  RAFT_CUSOLVER_TRY(
    cusolverDngeqrf_bufferSize(cusolver, n_rows, n_cols, Q, n_rows, &ws_size_Dngeqrf));
  RAFT_CUSOLVER_TRY(cusolverDnorgqr_bufferSize(
    cusolver, n_rows, n_cols, n_cols, Q, n_rows, tau.data(), &ws_size_Dnorgqr));

  rmm::device_uvector<math_t> workspace(std::max(ws_size_Dngeqrf, ws_size_Dnorgqr), stream);

  if (is_dry_run) { return; }

  RAFT_CUSOLVER_TRY(cusolverDngeqrf(cusolver,
                                    n_rows,
                                    n_cols,
                                    Q,
                                    n_rows,
                                    tau.data(),
                                    workspace.data(),
                                    ws_size_Dngeqrf,
                                    dev_info.data(),
                                    stream));

  RAFT_CUSOLVER_TRY(cusolverDnorgqr(cusolver,
                                    n_rows,
                                    n_cols,
                                    n_cols,
                                    Q,
                                    n_rows,
                                    tau.data(),
                                    workspace.data(),
                                    ws_size_Dnorgqr,
                                    dev_info.data(),
                                    stream));
}

template <typename math_t>
void qrGetQ(raft::resources const& handle,
            const math_t* M,
            math_t* Q,
            int n_rows,
            int n_cols,
            cudaStream_t stream)
{
  if (!resource::get_dry_run_flag(handle)) { raft::copy(Q, M, n_rows * n_cols, stream); }
  qrGetQ_inplace(handle, Q, n_rows, n_cols, stream);
}

template <typename math_t>
void qrGetQR(raft::resources const& handle,
             math_t* M,
             math_t* Q,
             math_t* R,
             int n_rows,
             int n_cols,
             cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

  int m = n_rows, n = n_cols;
  int R_full_nrows = m, R_full_ncols = n;
  int Q_nrows = m, Q_ncols = n;
  int Lwork_Dngeqrf, Lwork_Dnorgqr;
  rmm::device_uvector<math_t> R_full(m * n, stream);
  rmm::device_uvector<math_t> tau(std::min(m, n), stream);
  rmm::device_scalar<int> devInfo(stream);

  RAFT_CUSOLVER_TRY(cusolverDngeqrf_bufferSize(
    cusolverH, R_full_nrows, R_full_ncols, R_full.data(), R_full_nrows, &Lwork_Dngeqrf));
  RAFT_CUSOLVER_TRY(cusolverDnorgqr_bufferSize(cusolverH,
                                               Q_nrows,
                                               Q_ncols,
                                               std::min(Q_ncols, Q_nrows),
                                               Q,
                                               Q_nrows,
                                               tau.data(),
                                               &Lwork_Dnorgqr));

  rmm::device_uvector<math_t> workspace(std::max(Lwork_Dngeqrf, Lwork_Dnorgqr), stream);

  if (resource::get_dry_run_flag(handle)) { return; }

  RAFT_CUDA_TRY(cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * std::min(m, n), stream));
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(R_full.data(), M, sizeof(math_t) * m * n, cudaMemcpyDeviceToDevice, stream));

  RAFT_CUSOLVER_TRY(cusolverDngeqrf(cusolverH,
                                    R_full_nrows,
                                    R_full_ncols,
                                    R_full.data(),
                                    R_full_nrows,
                                    tau.data(),
                                    workspace.data(),
                                    Lwork_Dngeqrf,
                                    devInfo.data(),
                                    stream));

  raft::matrix::upper_triangular<math_t, int>(
    handle,
    make_device_matrix_view<const math_t, int, col_major>(R_full.data(), m, n),
    make_device_matrix_view<math_t, int, col_major>(R, std::min(m, n), std::min(m, n)));

  RAFT_CUDA_TRY(
    cudaMemcpyAsync(Q, R_full.data(), sizeof(math_t) * m * n, cudaMemcpyDeviceToDevice, stream));

  RAFT_CUSOLVER_TRY(cusolverDnorgqr(cusolverH,
                                    Q_nrows,
                                    Q_ncols,
                                    std::min(Q_ncols, Q_nrows),
                                    Q,
                                    Q_nrows,
                                    tau.data(),
                                    workspace.data(),
                                    Lwork_Dnorgqr,
                                    devInfo.data(),
                                    stream));
}

};  // namespace detail
};  // namespace linalg
};  // namespace raft
