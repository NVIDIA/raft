/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cublas_wrappers.hpp"
#include "cusolver_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/norm.cuh>
#include <raft/matrix/reverse.cuh>
#include <raft/matrix/sqrt.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace linalg {
namespace detail {

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
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "raft::linalg::svdQR(%d, %d)", n_rows, n_cols);
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);
  cublasHandle_t cublasH       = resource::get_cublas_handle(handle);

  const int m = n_rows;
  const int n = n_cols;

  rmm::device_scalar<int> devInfo(stream);
  T* d_rwork = nullptr;

  int lwork = 0;
  RAFT_CUSOLVER_TRY(cusolverDngesvd_bufferSize<T>(cusolverH, n_rows, n_cols, &lwork));
  rmm::device_uvector<T> d_work(lwork, stream);

  if (resource::get_dry_run_flag(handle)) { return; }

  char jobu  = 'S';
  char jobvt = 'A';

  if (!gen_left_vec) { jobu = 'N'; }

  if (!gen_right_vec) { jobvt = 'N'; }

  RAFT_CUSOLVER_TRY(cusolverDngesvd(cusolverH,
                                    jobu,
                                    jobvt,
                                    m,
                                    n,
                                    in,
                                    m,
                                    sing_vals,
                                    left_sing_vecs,
                                    m,
                                    right_sing_vecs,
                                    n,
                                    d_work.data(),
                                    lwork,
                                    d_rwork,
                                    devInfo.data(),
                                    stream));

  // Transpose the right singular vector back
  if (trans_right && right_sing_vecs != nullptr)
    raft::linalg::transpose(right_sing_vecs, n_cols, stream);

  RAFT_CUDA_TRY(cudaGetLastError());

  int dev_info;
  raft::update_host(&dev_info, devInfo.data(), 1, stream);
  resource::sync_stream(handle, stream);
  ASSERT(dev_info == 0,
         "svd.cuh: svd couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");
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
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "raft::linalg::svdEig(%d, %d)", n_rows, n_cols);
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);
  cublasHandle_t cublasH       = resource::get_cublas_handle(handle);

  auto len = n_cols * n_cols;
  rmm::device_uvector<math_t> in_cross_mult(len, stream);

  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  raft::linalg::gemm(handle,
                     in,
                     n_rows,
                     n_cols,
                     in,
                     in_cross_mult.data(),
                     n_cols,
                     n_cols,
                     CUBLAS_OP_T,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     stream);

  raft::linalg::eigDC(handle, in_cross_mult.data(), n_cols, n_cols, V, S, stream);

  raft::matrix::col_reverse(handle,
                            make_device_matrix_view<math_t, idx_t, col_major>(V, n_cols, n_cols));
  raft::matrix::row_reverse(handle,
                            make_device_matrix_view<math_t, idx_t, col_major>(S, n_cols, idx_t(1)));

  raft::matrix::weighted_sqrt(
    handle,
    make_device_matrix_view<math_t, idx_t, col_major>(S, n_cols, idx_t(1)),
    make_host_scalar_view<math_t>(&alpha),
    true);

  if (gen_left_vec) {
    raft::linalg::gemm(handle,
                       in,
                       n_rows,
                       n_cols,
                       V,
                       U,
                       n_rows,
                       n_cols,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       alpha,
                       beta,
                       stream);
    raft::linalg::binary_div_skip_zero<raft::Apply::ALONG_ROWS>(
      handle,
      make_device_matrix_view<math_t, idx_t, col_major>(U, n_rows, n_cols),
      make_device_vector_view<const math_t, idx_t>(S, n_cols));
  }
}

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
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "raft::linalg::svdJacobi(%d, %d)", n_rows, n_cols);
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

  gesvdjInfo_t gesvdj_params = NULL;

  RAFT_CUSOLVER_TRY(cusolverDnCreateGesvdjInfo(&gesvdj_params));
  RAFT_CUSOLVER_TRY(cusolverDnXgesvdjSetTolerance(gesvdj_params, tol));
  RAFT_CUSOLVER_TRY(cusolverDnXgesvdjSetMaxSweeps(gesvdj_params, max_sweeps));

  int m = n_rows;
  int n = n_cols;

  rmm::device_scalar<int> devInfo(stream);

  int lwork = 0;
  int econ  = 1;

  RAFT_CUSOLVER_TRY(cusolverDngesvdj_bufferSize(cusolverH,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                econ,
                                                m,
                                                n,
                                                in,
                                                m,
                                                sing_vals,
                                                left_sing_vecs,
                                                m,
                                                right_sing_vecs,
                                                n,
                                                &lwork,
                                                gesvdj_params));

  rmm::device_uvector<math_t> d_work(lwork, stream);

  if (resource::get_dry_run_flag(handle)) {
    RAFT_CUSOLVER_TRY(cusolverDnDestroyGesvdjInfo(gesvdj_params));
    return;
  }

  RAFT_CUSOLVER_TRY(cusolverDngesvdj(cusolverH,
                                     CUSOLVER_EIG_MODE_VECTOR,
                                     econ,
                                     m,
                                     n,
                                     in,
                                     m,
                                     sing_vals,
                                     left_sing_vecs,
                                     m,
                                     right_sing_vecs,
                                     n,
                                     d_work.data(),
                                     lwork,
                                     devInfo.data(),
                                     gesvdj_params,
                                     stream));

  RAFT_CUSOLVER_TRY(cusolverDnDestroyGesvdjInfo(gesvdj_params));
}

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
  const math_t alpha = 1.0, beta = 0.0;
  rmm::device_uvector<math_t> SVT(k * n_cols, stream);

  raft::linalg::gemm(
    handle, S, k, k, V, SVT.data(), k, n_cols, CUBLAS_OP_N, CUBLAS_OP_T, alpha, beta, stream);
  raft::linalg::gemm(handle,
                     U,
                     n_rows,
                     k,
                     SVT.data(),
                     out,
                     n_rows,
                     n_cols,
                     CUBLAS_OP_N,
                     CUBLAS_OP_N,
                     alpha,
                     beta,
                     stream);
}

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
  int m = n_rows, n = n_cols;
  bool is_dry_run = resource::get_dry_run_flag(handle);

  // form product matrix
  rmm::device_uvector<math_t> P_d(m * n, stream);
  rmm::device_uvector<math_t> S_mat(k * k, stream);

  if (!is_dry_run) {
    RAFT_CUDA_TRY(cudaMemsetAsync(P_d.data(), 0, sizeof(math_t) * m * n, stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(S_mat.data(), 0, sizeof(math_t) * k * k, stream));
  }

  // These RAFT functions have their own dry-run guards at the leaf level
  raft::matrix::set_diagonal(handle,
                             make_device_vector_view<const math_t>(S_vec, k),
                             make_device_matrix_view<math_t>(S_mat.data(), k, k));
  svdReconstruction(handle, U, S_mat.data(), V, P_d.data(), m, n, k, stream);

  // get norms of each
  math_t normA = raft::matrix::l2_norm(handle, make_device_matrix_view<const math_t>(A_d, m, n));
  math_t normU = raft::matrix::l2_norm(handle, make_device_matrix_view<const math_t>(U, m, k));
  math_t normS =
    raft::matrix::l2_norm(handle, make_device_matrix_view<const math_t>(S_mat.data(), k, k));
  math_t normV = raft::matrix::l2_norm(handle, make_device_matrix_view<const math_t>(V, n, k));
  math_t normP =
    raft::matrix::l2_norm(handle, make_device_matrix_view<const math_t>(P_d.data(), m, n));

  // calculate percent error
  const math_t alpha = 1.0, beta = -1.0;
  rmm::device_uvector<math_t> A_minus_P(m * n, stream);

  if (is_dry_run) { return false; }

  RAFT_CUDA_TRY(cudaMemsetAsync(A_minus_P.data(), 0, sizeof(math_t) * m * n, stream));

  cublasHandle_t cublasH = resource::get_cublas_handle(handle);
  RAFT_CUBLAS_TRY(cublasgeam(cublasH,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             m,
                             n,
                             &alpha,
                             A_d,
                             m,
                             &beta,
                             P_d.data(),
                             m,
                             A_minus_P.data(),
                             m,
                             stream));

  math_t norm_A_minus_P =
    raft::matrix::l2_norm(handle, make_device_matrix_view<const math_t>(A_minus_P.data(), m, n));
  math_t percent_error = 100.0 * norm_A_minus_P / normA;
  return (percent_error / 100.0 < tol);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
