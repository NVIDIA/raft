/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cublas_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/for_each.h>

#include <cmath>

namespace raft {
namespace linalg {
namespace detail {

template <typename IndexType, int TILE_DIM, int BLOCK_ROWS>
RAFT_KERNEL transpose_half_kernel(IndexType n_rows,
                                  IndexType n_cols,
                                  const half* __restrict__ in,
                                  half* __restrict__ out,
                                  const IndexType stride_in,
                                  const IndexType stride_out)
{
  __shared__ half tile[TILE_DIM][TILE_DIM + 1];

  for (int block_offset_y = 0; block_offset_y < n_rows; block_offset_y += gridDim.y * TILE_DIM) {
    for (int block_offset_x = 0; block_offset_x < n_cols; block_offset_x += gridDim.x * TILE_DIM) {
      auto x = block_offset_x + blockIdx.x * TILE_DIM + threadIdx.x;
      auto y = block_offset_y + blockIdx.y * TILE_DIM + threadIdx.y;

      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n_cols && (y + j) < n_rows) {
          tile[threadIdx.y + j][threadIdx.x] = __ldg(&in[(y + j) * stride_in + x]);
        }
      }
      __syncthreads();

      x = block_offset_y + blockIdx.y * TILE_DIM + threadIdx.x;
      y = block_offset_x + blockIdx.x * TILE_DIM + threadIdx.y;

      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n_rows && (y + j) < n_cols) {
          out[(y + j) * stride_out + x] = tile[threadIdx.x][threadIdx.y + j];
        }
      }
      __syncthreads();
    }
  }
}

/**
 * @brief Transposes a matrix stored in row-major order.
 *
 * This function transposes a matrix of half-precision floating-point numbers (`half`).
 * Both the input (`in`) and output (`out`) matrices are assumed to be stored in row-major order.
 *
 * @tparam IndexType The type used for indexing the matrix dimensions (e.g., int).
 * @param handle The RAFT resource handle which contains resources.
 * @param n_rows The number of rows in the input matrix.
 * @param n_cols The number of columns in the input matrix.
 * @param in Pointer to the input matrix in row-major order.
 * @param out Pointer to the output matrix in row-major order, where the transposed matrix will be
 * stored.
 * @param stride_in The stride (number of elements between consecutive rows) for the input matrix.
 *                  Default is 1, which means the input matrix is contiguous in memory.
 * @param stride_out The stride (number of elements between consecutive rows) for the output matrix.
 *                   Default is 1, which means the output matrix is contiguous in memory.
 */

template <typename IndexType>
void transpose_half(raft::resources const& handle,
                    IndexType n_rows,
                    IndexType n_cols,
                    const half* in,
                    half* out,
                    const IndexType stride_in  = 1,
                    const IndexType stride_out = 1)
{
  if (n_cols == 0 || n_rows == 0) return;
  if (resource::get_dry_run_flag(handle)) { return; }
  auto stream = resource::get_cuda_stream(handle);

  int dev_id, sm_count;

  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  constexpr int tpb         = 256;
  constexpr int block_dim_x = 128 / sizeof(half);
  constexpr int block_dim_y = tpb / block_dim_x;

  dim3 blocks(block_dim_x, block_dim_y);

  int max_active_blocks = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, transpose_half_kernel<IndexType, block_dim_x, block_dim_y>, tpb, 0));
  int num_blocks = max_active_blocks * sm_count;

  int grid_x = (n_cols + block_dim_x - 1) / block_dim_x;
  int grid_y = (n_rows + block_dim_x - 1) / block_dim_x;

  float ratio = static_cast<float>(grid_y) / static_cast<float>(grid_x);
  int adjusted_grid_y =
    std::max(std::min(grid_y, static_cast<int>(std::sqrt(num_blocks * ratio))), 1);
  int adjusted_grid_x = std::max(std::min(grid_x, num_blocks / adjusted_grid_y), 1);

  dim3 grids(adjusted_grid_x, adjusted_grid_y);

  if (stride_in > 1 || stride_out > 1) {
    transpose_half_kernel<IndexType, block_dim_x, block_dim_y>
      <<<grids, blocks, 0, stream>>>(n_rows, n_cols, in, out, stride_in, stride_out);
  } else {
    transpose_half_kernel<IndexType, block_dim_x, block_dim_y>
      <<<grids, blocks, 0, stream>>>(n_rows, n_cols, in, out, n_cols, n_rows);
  }

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename math_t>
void transpose(raft::resources const& handle,
               math_t* in,
               math_t* out,
               int n_rows,
               int n_cols,
               cudaStream_t stream)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  int out_n_rows = n_cols;
  int out_n_cols = n_rows;

  if constexpr (std::is_same_v<math_t, half>) {
    transpose_half(handle, n_cols, n_rows, in, out);
  } else {
    cublasHandle_t cublas_h = resource::get_cublas_handle(handle);
    RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, stream));
    const math_t alpha = 1.0;
    const math_t beta  = 0.0;
    RAFT_CUBLAS_TRY(cublasgeam(cublas_h,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               out_n_rows,
                               out_n_cols,
                               &alpha,
                               in,
                               n_rows,
                               &beta,
                               out,
                               out_n_rows,
                               out,
                               out_n_rows,
                               stream));
  }
}

template <typename math_t>
void transpose(math_t* inout, int n, cudaStream_t stream)
{
  auto m        = n;
  auto size     = n * n;
  auto d_inout  = inout;
  auto counting = cuda::make_counting_iterator<int>(0);

  thrust::for_each(rmm::exec_policy(stream), counting, counting + size, [=] __device__(int idx) {
    int s_row = idx % m;
    int s_col = idx / m;
    int d_row = s_col;
    int d_col = s_row;
    if (s_row < s_col) {
      auto temp                  = d_inout[d_col * m + d_row];
      d_inout[d_col * m + d_row] = d_inout[s_col * m + s_row];
      d_inout[s_col * m + s_row] = temp;
    }
  });
}

template <typename T, typename IndexType, typename LayoutPolicy, typename AccessorPolicy>
void transpose_row_major_impl(
  raft::resources const& handle,
  raft::mdspan<T, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> in,
  raft::mdspan<T, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  auto out_n_rows   = in.extent(1);
  auto out_n_cols   = in.extent(0);
  T constexpr kOne  = 1;
  T constexpr kZero = 0;

  CUBLAS_TRY(cublasgeam(resource::get_cublas_handle(handle),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        out_n_cols,
                        out_n_rows,
                        &kOne,
                        in.data_handle(),
                        in.stride(0),
                        &kZero,
                        static_cast<T*>(nullptr),
                        out.stride(0),
                        out.data_handle(),
                        out.stride(0),
                        resource::get_cuda_stream(handle)));
}

template <typename IndexType, typename LayoutPolicy, typename AccessorPolicy>
void transpose_row_major_impl(
  raft::resources const& handle,
  raft::mdspan<half, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> in,
  raft::mdspan<half, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> out)
{
  transpose_half<IndexType>(handle,
                            in.extent(0),
                            in.extent(1),
                            in.data_handle(),
                            out.data_handle(),
                            in.stride(0),
                            out.stride(0));
}

template <typename T, typename IndexType, typename LayoutPolicy, typename AccessorPolicy>
void transpose_col_major_impl(
  raft::resources const& handle,
  raft::mdspan<T, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> in,
  raft::mdspan<T, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  auto out_n_rows   = in.extent(1);
  auto out_n_cols   = in.extent(0);
  T constexpr kOne  = 1;
  T constexpr kZero = 0;

  CUBLAS_TRY(cublasgeam(resource::get_cublas_handle(handle),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        out_n_rows,
                        out_n_cols,
                        &kOne,
                        in.data_handle(),
                        in.stride(1),
                        &kZero,
                        static_cast<T*>(nullptr),
                        out.stride(1),
                        out.data_handle(),
                        out.stride(1),
                        resource::get_cuda_stream(handle)));
}

template <typename IndexType, typename LayoutPolicy, typename AccessorPolicy>
void transpose_col_major_impl(
  raft::resources const& handle,
  raft::mdspan<half, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> in,
  raft::mdspan<half, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> out)
{
  transpose_half<IndexType>(handle,
                            in.extent(1),
                            in.extent(0),
                            in.data_handle(),
                            out.data_handle(),
                            in.stride(1),
                            out.stride(1));
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
