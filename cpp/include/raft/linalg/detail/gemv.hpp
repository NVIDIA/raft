/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cublas_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

#include <cublas_v2.h>

namespace raft {
namespace linalg {
namespace detail {

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
  if (resource::get_dry_run_flag(handle)) { return; }
  cublasHandle_t cublas_h = resource::get_cublas_handle(handle);
  detail::cublas_device_pointer_mode<DevicePointerMode> pmode(cublas_h);
  RAFT_CUBLAS_TRY(detail::cublasgemv(cublas_h,
                                     trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                                     m,
                                     n,
                                     alpha,
                                     A,
                                     lda,
                                     x,
                                     incx,
                                     beta,
                                     y,
                                     incy,
                                     stream));
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
  gemv(handle, trans_a, n_rows, n_cols, &alpha, A, n_rows, x, incx, &beta, y, incy, stream);
}

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
  gemv(handle, A, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, stream);
}

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
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);

  gemv(handle, A, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, stream);
}

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
  if (resource::get_dry_run_flag(handle)) { return; }
  cublasHandle_t cublas_h = resource::get_cublas_handle(handle);
  cublasOperation_t op_a  = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  RAFT_CUBLAS_TRY(
    cublasgemv(cublas_h, op_a, n_rows_a, n_cols_a, &alpha, A, lda, x, 1, &beta, y, 1, stream));
}

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
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  gemv(handle, A, n_rows_a, n_cols_a, lda, x, y, trans_a, alpha, beta, stream);
}

};  // namespace detail
};  // namespace linalg
};  // namespace raft
