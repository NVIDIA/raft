/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cublaslt_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resources.hpp>

namespace raft {
namespace linalg::detail {

template <typename A_T, typename B_T, typename C_T, typename S_T, bool DevicePointerMode = false>
void legacy_gemm(raft::resources const& res,
                 const bool trans_a,
                 const bool trans_b,
                 const int m,
                 const int n,
                 const int k,
                 const S_T* alpha,
                 const A_T* A,
                 const int lda,
                 const B_T* B,
                 const int ldb,
                 const S_T* beta,
                 C_T* C,
                 const int ldc,
                 cudaStream_t stream,
                 const C_T* bias = nullptr)
{
  return legacy_matmul<DevicePointerMode, S_T, A_T, B_T, C_T>(res,
                                                              trans_a,
                                                              trans_b,
                                                              static_cast<uint64_t>(m),
                                                              static_cast<uint64_t>(n),
                                                              static_cast<uint64_t>(k),
                                                              alpha,
                                                              A,
                                                              static_cast<uint64_t>(lda),
                                                              B,
                                                              static_cast<uint64_t>(ldb),
                                                              beta,
                                                              C,
                                                              static_cast<uint64_t>(ldc),
                                                              stream,
                                                              bias);
}

template <typename A_T, typename B_T, typename C_T, typename S_T>
void legacy_gemm(raft::resources const& res,
                 const A_T* a,
                 int n_rows_a,
                 int n_cols_a,
                 const B_T* b,
                 C_T* c,
                 int n_rows_c,
                 int n_cols_c,
                 cublasOperation_t trans_a,
                 cublasOperation_t trans_b,
                 S_T alpha,
                 S_T beta,
                 cudaStream_t stream)
{
  int m  = n_rows_c;
  int n  = n_cols_c;
  auto k = trans_a == CUBLAS_OP_T ? n_rows_a : n_cols_a;
  return legacy_matmul<false, S_T, A_T, B_T, C_T>(
    res,
    trans_a == CUBLAS_OP_T,
    trans_b == CUBLAS_OP_T,
    static_cast<uint64_t>(n_rows_c),
    static_cast<uint64_t>(n_cols_c),
    static_cast<uint64_t>(k),
    &alpha,
    a,
    static_cast<uint64_t>(trans_a == CUBLAS_OP_T ? k : m),
    b,
    static_cast<uint64_t>(trans_b == CUBLAS_OP_T ? n : k),
    &beta,
    c,
    static_cast<uint64_t>(m),
    stream);
}

template <typename A_T, typename B_T, typename C_T>
void legacy_gemm(raft::resources const& res,
                 const A_T* a,
                 int n_rows_a,
                 int n_cols_a,
                 const B_T* b,
                 C_T* c,
                 int n_rows_c,
                 int n_cols_c,
                 cublasOperation_t trans_a,
                 cublasOperation_t trans_b,
                 cudaStream_t stream)
{
  return legacy_gemm(
    res, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, C_T{1}, C_T{0}, stream);
}

template <typename x_T, typename y_T, typename z_T, typename s_T, bool DevicePointerMode = false>
void legacy_gemm(raft::resources const& res,
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
                 const s_T* alpha,
                 const s_T* beta)
{
  if (isZColMajor) {
    return legacy_matmul<DevicePointerMode, s_T, x_T, y_T, z_T>(
      res,
      !isXColMajor,
      !isYColMajor,
      static_cast<uint64_t>(_M),
      static_cast<uint64_t>(_N),
      static_cast<uint64_t>(_K),
      alpha,
      x,
      static_cast<uint64_t>(isXColMajor ? _M : _K),
      y,
      static_cast<uint64_t>(isYColMajor ? _K : _N),
      beta,
      z,
      static_cast<uint64_t>(_M),
      stream);
  } else {
    return legacy_gemm<x_T, y_T, z_T, s_T, DevicePointerMode>(
      res, z, y, x, _N, _M, _K, true, !isYColMajor, !isXColMajor, stream, alpha, beta);
  }
}

}  // namespace linalg::detail
}  // namespace raft
