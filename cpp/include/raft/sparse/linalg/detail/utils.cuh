/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/math.hpp>
#include <raft/core/resource/dry_run_flag.hpp>

#include <cub/warp/warp_reduce.cuh>
#include <cuda_fp16.h>
#include <cuda_pipeline.h>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

/**
 * Computes the maximum number of columns that can be stored
 * in shared memory in dense form with the given block size
 * and precision.
 * @return the maximum number of columns that can be stored in smem
 */
template <typename value_idx, typename value_t, int tpb = 1024>
inline int max_cols_per_block()
{
  // max cols = (total smem available - cub reduction smem)
  return (raft::getSharedMemPerBlock() - ((tpb / raft::warp_size()) * sizeof(value_t))) /
         sizeof(value_t);
}

template <typename value_idx, typename value_t, typename dot_t = value_t>
RAFT_KERNEL faster_dot_on_csr_kernel(dot_t* __restrict__ dot,
                                     const value_idx* __restrict__ indptr,
                                     const value_idx* __restrict__ cols,
                                     const value_t* __restrict__ A,
                                     const value_t* __restrict__ B,
                                     const value_idx nnz,
                                     const value_idx n_rows,
                                     const value_idx dim)
{
  auto vec_id  = threadIdx.x;
  auto lane_id = threadIdx.x & 0x1f;

  extern __shared__ char smem[];
  value_t* s_A      = (value_t*)smem;
  value_idx cur_row = -1;

  for (int row = blockIdx.x; row < n_rows; row += gridDim.x) {
    for (int dot_id = blockIdx.y + indptr[row]; dot_id < indptr[row + 1]; dot_id += gridDim.y) {
      if (dot_id >= nnz) { return; }
      const value_idx col               = cols[dot_id] * dim;
      const value_t* __restrict__ B_col = B + col;

      if (threadIdx.x == 0) { dot[dot_id] = 0.0; }
      __syncthreads();

      if (cur_row != row) {
        for (value_idx k = vec_id; k < dim; k += blockDim.x) {
          s_A[k] = A[row * dim + k];
        }
        cur_row = row;
      }

      dot_t l_dot_ = 0.0;
      for (value_idx k = vec_id; k < dim; k += blockDim.x) {
        asm("prefetch.global.L2 [%0];" ::"l"(B_col + k + blockDim.x));
        if constexpr ((std::is_same_v<dot_t, float> && std::is_same_v<value_t, half>)) {
          l_dot_ += __half2float(s_A[k]) * __half2float(__ldcg(B_col + k));
        } else {
          l_dot_ += s_A[k] * __ldcg(B_col + k);
        }
      }

      typedef cub::WarpReduce<dot_t> WarpReduce;
      __shared__ typename WarpReduce::TempStorage temp_storage;
      dot_t warp_sum = WarpReduce(temp_storage).Sum(l_dot_);

      if (lane_id == 0) { atomicAdd_block(dot + dot_id, warp_sum); }
    }
  }
}

template <typename value_idx, typename value_t, typename dot_t = value_t>
void faster_dot_on_csr(raft::resources const& handle,
                       dot_t* dot,
                       const value_idx nnz,
                       const value_idx* indptr,
                       const value_idx* cols,
                       const value_t* A,
                       const value_t* B,
                       const value_idx n_rows,
                       const value_idx dim)
{
  if (nnz == 0 || n_rows == 0) return;
  if (resource::get_dry_run_flag(handle)) { return; }  // No allocations below

  auto stream = resource::get_cuda_stream(handle);

  constexpr value_idx MAX_ROW_PER_ITER = 500;
  int dev_id, sm_count, blocks_per_sm;

  const int smem_size = dim * sizeof(value_t);
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  if (dim < 128) {
    constexpr int tpb = 64;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t, dot_t>, tpb, smem_size);
    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y =
      (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<value_idx, value_t, dot_t>
      <<<blocks, tpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);

  } else if (dim < 256) {
    constexpr int tpb = 128;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t, dot_t>, tpb, smem_size);
    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y =
      (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<value_idx, value_t, dot_t>
      <<<blocks, tpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  } else if (dim < 512) {
    constexpr int tpb = 256;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t, dot_t>, tpb, smem_size);
    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y =
      (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<value_idx, value_t, dot_t>
      <<<blocks, tpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  } else {
    constexpr int tpb = 512;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, faster_dot_on_csr_kernel<value_idx, value_t, dot_t>, tpb, smem_size);
    auto block_x = std::min(n_rows, MAX_ROW_PER_ITER);
    auto block_y =
      (std::min(value_idx(blocks_per_sm * sm_count * 16), nnz) + block_x - 1) / block_x;
    dim3 blocks(block_x, block_y, 1);

    faster_dot_on_csr_kernel<value_idx, value_t, dot_t>
      <<<blocks, tpb, smem_size, stream>>>(dot, indptr, cols, A, B, nnz, n_rows, dim);
  }

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace linalg
}  // namespace sparse
}  // namespace raft
