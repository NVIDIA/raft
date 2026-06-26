/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/util/cache_util.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda_runtime.h>
#include <thrust/for_each.h>

#include <cusolverDn.h>

#include <algorithm>
#include <cstddef>

namespace raft {
namespace matrix {
namespace detail {

template <typename m_t, typename idx_array_t = int, typename idx_t = size_t>
void copyRows(const m_t* in,
              idx_t n_rows,
              idx_t n_cols,
              m_t* out,
              const idx_array_t* indices,
              idx_t n_rows_indices,
              cudaStream_t stream,
              bool rowMajor = false)
{
  if (rowMajor) {
    const idx_t TPB = 256;
    cache::get_vecs<<<raft::ceildiv(n_rows_indices * n_cols, TPB), TPB, 0, stream>>>(
      in, n_cols, indices, n_rows_indices, out);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    return;
  }

  idx_t size    = n_rows_indices * n_cols;
  auto counting = cuda::make_counting_iterator<idx_t>(0);

  thrust::for_each(rmm::exec_policy(stream), counting, counting + size, [=] __device__(idx_t idx) {
    idx_t row = idx % n_rows_indices;
    idx_t col = idx / n_rows_indices;

    out[col * n_rows_indices + row] = in[col * n_rows + indices[row]];
  });
}

template <typename m_t, typename idx_t = int>
void truncZeroOrigin(
  const m_t* in, idx_t in_n_rows, m_t* out, idx_t out_n_rows, idx_t out_n_cols, cudaStream_t stream)
{
  auto m         = out_n_rows;
  auto k         = in_n_rows;
  idx_t size     = out_n_rows * out_n_cols;
  auto d_q       = in;
  auto d_q_trunc = out;
  auto counting  = cuda::make_counting_iterator<idx_t>(0);

  thrust::for_each(rmm::exec_policy(stream), counting, counting + size, [=] __device__(idx_t idx) {
    idx_t row                = idx % m;
    idx_t col                = idx / m;
    d_q_trunc[col * m + row] = d_q[col * k + row];
  });
}

template <typename m_t, typename idx_t = int>
void colReverse(m_t* inout, idx_t n_rows, idx_t n_cols, cudaStream_t stream)
{
  auto n            = n_cols;
  auto m            = n_rows;
  idx_t size        = n_rows * n_cols;
  auto d_q          = inout;
  auto d_q_reversed = inout;
  auto counting     = cuda::make_counting_iterator<idx_t>(0);

  thrust::for_each(
    rmm::exec_policy(stream), counting, counting + (size / 2), [=] __device__(idx_t idx) {
      idx_t dest_row             = idx % m;
      idx_t dest_col             = idx / m;
      idx_t src_row              = dest_row;
      idx_t src_col              = (n - dest_col) - 1;
      m_t temp                   = (m_t)d_q_reversed[idx];
      d_q_reversed[idx]          = d_q[src_col * m + src_row];
      d_q[src_col * m + src_row] = temp;
    });
}

template <typename m_t, typename idx_t = int>
void rowReverse(m_t* inout, idx_t n_rows, idx_t n_cols, cudaStream_t stream)
{
  auto m            = n_rows;
  idx_t size        = n_rows * n_cols;
  auto d_q          = inout;
  auto d_q_reversed = inout;
  auto counting     = cuda::make_counting_iterator<idx_t>(0);

  thrust::for_each(
    rmm::exec_policy(stream), counting, counting + (size / 2), [=] __device__(idx_t idx) {
      idx_t dest_row = idx % (m / 2);
      idx_t dest_col = idx / (m / 2);
      idx_t src_row  = (m - dest_row) - 1;
      idx_t src_col  = dest_col;

      m_t temp                              = (m_t)d_q_reversed[dest_col * m + dest_row];
      d_q_reversed[dest_col * m + dest_row] = d_q[src_col * m + src_row];
      d_q[src_col * m + src_row]            = temp;
    });
}

template <typename m_t, typename idx_t = int>
void print(const m_t* in,
           idx_t n_rows,
           idx_t n_cols,
           char h_separator    = ' ',
           char v_separator    = '\n',
           cudaStream_t stream = rmm::cuda_stream_default)
{
  std::vector<m_t> h_matrix = std::vector<m_t>(n_cols * n_rows);
  raft::update_host(h_matrix.data(), in, n_cols * n_rows, stream);

  for (idx_t i = 0; i < n_rows; i++) {
    for (idx_t j = 0; j < n_cols; j++) {
      printf("%1.4f%c", h_matrix[j * n_rows + i], j < n_cols - 1 ? h_separator : v_separator);
    }
  }
}

template <typename m_t, typename idx_t = int>
void printHost(const m_t* in, idx_t n_rows, idx_t n_cols)
{
  for (idx_t i = 0; i < n_rows; i++) {
    for (idx_t j = 0; j < n_cols; j++) {
      printf("%1.4f ", in[j * n_rows + i]);
    }
    printf("\n");
  }
}

/**
 * @brief Kernel for copying a slice of a big matrix to a small matrix with a
 * size matches that slice
 * @param src_d: input matrix
 * @param m: number of rows of input matrix
 * @param n: number of columns of input matrix
 * @param dst_d: output matrix
 * @param x1, y1: coordinate of the top-left point of the wanted area (0-based)
 * @param x2, y2: coordinate of the bottom-right point of the wanted area
 * (1-based)
 */
template <typename m_t, typename idx_t = int>
RAFT_KERNEL slice(const m_t* src_d, idx_t lda, m_t* dst_d, idx_t x1, idx_t y1, idx_t x2, idx_t y2)
{
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx_t dm = x2 - x1, dn = y2 - y1;
  if (idx < dm * dn) {
    idx_t i = idx % dm, j = idx / dm;
    idx_t is = i + x1, js = j + y1;
    dst_d[idx] = src_d[is + js * lda];
  }
}

template <typename m_t, typename idx_t = int>
void sliceMatrix(const m_t* in,
                 idx_t n_rows,
                 idx_t n_cols,
                 m_t* out,
                 idx_t x1,
                 idx_t y1,
                 idx_t x2,
                 idx_t y2,
                 bool row_major,
                 cudaStream_t stream)
{
  auto lda = row_major ? n_cols : n_rows;
  dim3 block(64);
  dim3 grid(((x2 - x1) * (y2 - y1) + block.x - 1) / block.x);
  if (row_major)
    slice<<<grid, block, 0, stream>>>(in, lda, out, y1, x1, y2, x2);
  else
    slice<<<grid, block, 0, stream>>>(in, lda, out, x1, y1, x2, y2);
}

/**
 * @brief Kernel for copying the upper triangular part of a matrix to another
 * @param src: input matrix with a size of mxn
 * @param dst: output matrix with a size of kxk
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param k: min(n_rows, n_cols)
 */
template <typename m_t, typename idx_t = int>
RAFT_KERNEL getUpperTriangular(const m_t* src, m_t* dst, idx_t n_rows, idx_t n_cols, idx_t k)
{
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx_t m = n_rows, n = n_cols;
  if (idx < m * n) {
    idx_t i = idx % m, j = idx / m;
    if (i < k && j < k && j >= i) { dst[i + j * k] = src[idx]; }
  }
}

template <typename m_t, typename idx_t = int>
void copyUpperTriangular(const m_t* src, m_t* dst, idx_t n_rows, idx_t n_cols, cudaStream_t stream)
{
  idx_t m = n_rows, n = n_cols;
  idx_t k = std::min(m, n);
  dim3 block(64);
  dim3 grid((m * n + block.x - 1) / block.x);
  getUpperTriangular<<<grid, block, 0, stream>>>(src, dst, m, n, k);
}

/**
 * @brief Copy a vector to the diagonal of a matrix
 * @param vec: vector of length k = min(n_rows, n_cols)
 * @param matrix: matrix of size n_rows x n_cols (leading dimension = lda)
 * @param lda: leading dimension of the matrix
 * @param k: dimensionality
 */
template <typename m_t, typename idx_t = int>
RAFT_KERNEL copyVectorToMatrixDiagonal(const m_t* vec, m_t* matrix, idx_t lda, idx_t k)
{
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < k) { matrix[idx + idx * lda] = vec[idx]; }
}

/**
 * @brief Copy matrix diagonal to vector
 * @param vec: vector of length k = min(n_rows, n_cols)
 * @param matrix: matrix of size n_rows x n_cols (leading dimension = lda)
 * @param lda: leading dimension of the matrix
 * @param k: dimensionality
 */
template <typename m_t, typename idx_t = int>
RAFT_KERNEL copyVectorFromMatrixDiagonal(m_t* vec, const m_t* matrix, idx_t lda, idx_t k)
{
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < k) { vec[idx] = matrix[idx + idx * lda]; }
}

template <typename m_t, typename idx_t = int>
void initializeDiagonalMatrix(
  const m_t* vec, m_t* matrix, idx_t n_rows, idx_t n_cols, bool row_major, cudaStream_t stream)
{
  idx_t k   = std::min(n_rows, n_cols);
  idx_t lda = row_major ? n_cols : n_rows;
  dim3 block(64);
  dim3 grid((k + block.x - 1) / block.x);
  copyVectorToMatrixDiagonal<<<grid, block, 0, stream>>>(vec, matrix, lda, k);
}

template <typename m_t, typename idx_t = int>
void getDiagonalMatrix(
  m_t* vec, const m_t* matrix, idx_t n_rows, idx_t n_cols, bool row_major, cudaStream_t stream)
{
  idx_t k   = std::min(n_rows, n_cols);
  idx_t lda = row_major ? n_cols : n_rows;
  dim3 block(64);
  dim3 grid((k + block.x - 1) / block.x);
  copyVectorFromMatrixDiagonal<<<grid, block, 0, stream>>>(vec, matrix, lda, k);
}

/**
 * @brief Calculate the inverse of the diagonal of a square matrix
 * element-wise and in place
 * @param in: square input matrix with size len x len
 * @param len: size of one side of the matrix
 */
template <typename m_t, typename idx_t = int>
RAFT_KERNEL matrixDiagonalInverse(m_t* in, idx_t len)
{
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < len) { in[idx + idx * len] = 1.0 / in[idx + idx * len]; }
}

template <typename m_t, typename idx_t = int>
void getDiagonalInverseMatrix(m_t* in, idx_t len, cudaStream_t stream)
{
  dim3 block(64);
  dim3 grid((len + block.x - 1) / block.x);
  matrixDiagonalInverse<m_t><<<grid, block, 0, stream>>>(in, len);
}

template <typename m_t, typename idx_t = int>
m_t getL2Norm(raft::resources const& handle, const m_t* in, idx_t size, cudaStream_t stream)
{
  if (resource::get_dry_run_flag(handle)) { return m_t{0}; }
  cublasHandle_t cublasH = resource::get_cublas_handle(handle);
  m_t normval            = 0;
  RAFT_EXPECTS(
    std::is_integral_v<idx_t> && (std::size_t)size <= (std::size_t)std::numeric_limits<int>::max(),
    "Index type not supported");
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasnrm2(cublasH, static_cast<int>(size), in, 1, &normval, stream));
  return normval;
}

}  // end namespace detail
}  // end namespace matrix
}  // namespace raft
