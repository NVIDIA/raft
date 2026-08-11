/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cublaslt_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

namespace raft {
namespace linalg::detail {

/** Description of one operand of a batched gemm in cublas (column-major) terms. */
struct batched_gemm_operand {
  /** Whether every matrix of the batch is column-major (row-major otherwise). */
  bool col_major;
  /** Leading dimension of every matrix of the batch. */
  uint64_t ld;
  /** Offset in elements between consecutive matrices of the batch. */
  int64_t batch_stride;
};

/**
 * Interpret a 3D mdspan as a batch of matrices: the first (slowest-varying) dimension indexes the
 * batch, the two remaining dimensions form a row- or column-major matrix. The batch stride is
 * unconstrained: a stride of zero broadcasts a single matrix over the whole batch.
 */
template <typename ValueType, typename IndexType, typename LayoutPolicy>
auto describe_batched_gemm_operand(
  raft::device_mdspan<ValueType, raft::extent_3d<IndexType>, LayoutPolicy> x, const char* name)
  -> batched_gemm_operand
{
  const auto rows       = static_cast<uint64_t>(x.extent(1));
  const auto cols       = static_cast<uint64_t>(x.extent(2));
  const auto row_stride = static_cast<uint64_t>(x.stride(1));
  const auto col_stride = static_cast<uint64_t>(x.stride(2));

  batched_gemm_operand r{};
  if (col_stride == 1 && row_stride >= cols) {
    r.col_major = false;
    r.ld        = row_stride;
  } else if (row_stride == 1 && col_stride >= rows) {
    r.col_major = true;
    r.ld        = col_stride;
  } else {
    RAFT_FAIL(
      "%s is not a batch of row- or column-major matrices: with extents [batch, %zu, %zu] the "
      "matrix strides are [%zu, %zu], one of which must be 1",
      name,
      static_cast<size_t>(rows),
      static_cast<size_t>(cols),
      static_cast<size_t>(row_stride),
      static_cast<size_t>(col_stride));
  }
  r.batch_stride = static_cast<int64_t>(x.stride(0));
  return r;
}

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
                 cudaStream_t stream)
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
                                                              stream);
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
