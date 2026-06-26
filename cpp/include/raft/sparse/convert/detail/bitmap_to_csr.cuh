/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/mdspan_util.cuh>  // detail::popc
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/convert/detail/adj_to_csr.cuh>
#include <raft/util/device_loads_stores.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <thrust/fill.h>

#include <limits>

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

// Threads per block in bitmap_to_csr.
static const constexpr int bitmap_to_csr_tpb = 256;

template <typename bitmap_t, typename index_t, typename nnz_t>
RAFT_KERNEL __launch_bounds__(bitmap_to_csr_tpb) calc_nnz_by_rows_kernel(const bitmap_t* bitmap,
                                                                         index_t num_rows,
                                                                         index_t num_cols,
                                                                         index_t bitmap_num,
                                                                         nnz_t* sub_col_nnz,
                                                                         index_t bits_per_sub_col)
{
  using mutable_bitmap_t = typename std::remove_const_t<bitmap_t>;
  using BlockReduce      = cub::BlockReduce<index_t, bitmap_to_csr_tpb>;

  __shared__ typename BlockReduce::TempStorage reduce_storage;

  constexpr index_t BITS_PER_BITMAP = sizeof(bitmap_t) * 8;

  const auto tid = threadIdx.x;
  const auto row = blockIdx.x;

  const auto num_sub_cols = gridDim.y;
  const auto sub_col      = blockIdx.y;

  size_t s_bit = size_t(row) * num_cols + sub_col * bits_per_sub_col;
  size_t e_bit = min(s_bit + bits_per_sub_col, size_t(num_cols) * (row + 1));

  nnz_t l_sum = 0;
  nnz_t g_sum = 0;

  index_t s_offset  = s_bit % BITS_PER_BITMAP;
  size_t bitmap_idx = s_bit / BITS_PER_BITMAP;

  if (tid == 0 && s_offset != 0) {
    mutable_bitmap_t l_bitmap = bitmap[bitmap_idx];

    l_bitmap >>= s_offset;

    size_t remaining_bits = min(size_t(BITS_PER_BITMAP - s_offset), e_bit - s_bit);

    if (remaining_bits < BITS_PER_BITMAP) {
      l_bitmap &= ((mutable_bitmap_t(1) << remaining_bits) - 1);
    }
    l_sum += static_cast<nnz_t>(raft::detail::popc(l_bitmap));
  }
  if (s_offset != 0) { s_bit += (BITS_PER_BITMAP - s_offset); }

  for (size_t bit_idx = s_bit; bit_idx < e_bit; bit_idx += BITS_PER_BITMAP * blockDim.x) {
    mutable_bitmap_t l_bitmap = 0;
    bitmap_idx                = bit_idx / BITS_PER_BITMAP + tid;

    index_t remaining_bits = min(BITS_PER_BITMAP, index_t(e_bit - bitmap_idx * BITS_PER_BITMAP));

    if (bitmap_idx * BITS_PER_BITMAP < e_bit) { l_bitmap = bitmap[bitmap_idx]; }

    if (remaining_bits < BITS_PER_BITMAP) {
      l_bitmap &= ((mutable_bitmap_t(1) << remaining_bits) - 1);
    }
    l_sum += static_cast<nnz_t>(raft::detail::popc(l_bitmap));
  }
  g_sum = BlockReduce(reduce_storage).Reduce(l_sum, cuda::std::plus{});
  stg(g_sum, sub_col_nnz + sub_col + row * num_sub_cols, tid == 0);
}

template <typename bitmap_t, typename index_t, typename nnz_t>
void calc_nnz_by_rows(raft::resources const& handle,
                      const bitmap_t* bitmap,
                      index_t num_rows,
                      index_t num_cols,
                      nnz_t* sub_col_nnz,
                      size_t& sub_nnz_size,
                      index_t& bits_per_sub_col)
{
  if (sub_nnz_size == 0) {
    bits_per_sub_col = bitmap_to_csr_tpb * sizeof(index_t) * 8 * 8;
    auto grid_dim_y  = (num_cols + bits_per_sub_col - 1) / bits_per_sub_col;
    sub_nnz_size     = num_rows * ((num_cols + bits_per_sub_col - 1) / bits_per_sub_col);
    return;
  }
  if (resource::get_dry_run_flag(handle)) { return; }

  auto stream        = resource::get_cuda_stream(handle);
  const size_t total = num_rows * num_cols;
  const size_t bitmap_num =
    (total + index_t(sizeof(bitmap_t) * 8) - 1) / index_t(sizeof(bitmap_t) * 8);

  auto block_x = num_rows;
  auto block_y = sub_nnz_size / num_rows;
  dim3 grid(block_x, block_y, 1);

  auto block = bitmap_to_csr_tpb;

  calc_nnz_by_rows_kernel<bitmap_t, index_t, nnz_t><<<grid, block, 0, stream>>>(
    bitmap, num_rows, num_cols, bitmap_num, sub_col_nnz, bits_per_sub_col);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename bitmap_t, typename index_t, typename nnz_t, bool check_nnz>
RAFT_KERNEL __launch_bounds__(bitmap_to_csr_tpb)
  fill_indices_by_rows_kernel(const bitmap_t* bitmap,
                              index_t* indptr,
                              size_t num_rows,
                              size_t num_cols,
                              nnz_t nnz,
                              index_t* indices,
                              nnz_t* sub_col_nnz,
                              index_t bits_per_sub_col)
{
  constexpr bitmap_t ONE            = bitmap_t(1u);
  constexpr index_t BITS_PER_BITMAP = sizeof(bitmap_t) * 8;

  using mutable_bitmap_t = typename std::remove_const_t<bitmap_t>;
  using BlockScan        = cub::BlockScan<int, bitmap_to_csr_tpb>;

  __shared__ typename BlockScan::TempStorage scan_storage;

  const auto tid = threadIdx.x;
  const auto row = blockIdx.x;

  const auto num_sub_cols = gridDim.y;
  const auto sub_col      = blockIdx.y;

  // Ensure the HBM allocated for CSR values is sufficient to handle all non-zero bitmap bits.
  // An assert will trigger if the allocated HBM is insufficient when `NDEBUG` isn't defined.
  // Note: Assertion is active only if `NDEBUG` is undefined.
  if constexpr (check_nnz) {
    if (tid == 0) { assert(sub_col_nnz[num_rows * num_sub_cols] <= nnz); }
  }

  size_t s_bit = size_t(row) * num_cols + sub_col * bits_per_sub_col;
  size_t e_bit = min(s_bit + bits_per_sub_col, size_t(num_cols) * (row + 1));

  size_t l_sum = 0;
  __shared__ size_t g_sum;

  index_t s_offset  = s_bit % BITS_PER_BITMAP;
  size_t bitmap_idx = s_bit / BITS_PER_BITMAP;

  if (tid == 0 && row == 0 && sub_col == 0) { indptr[0] = 0; }
  if (tid == 0 && sub_col == 0) { indptr[row + 1] = sub_col_nnz[(row + 1) * num_sub_cols]; }

  size_t g_nnz                   = sub_col_nnz[sub_col + row * num_sub_cols];
  index_t* sub_cols_indices_addr = indices + g_nnz;

  bool guard[BITS_PER_BITMAP];

  index_t g_bits = sub_col * bits_per_sub_col + tid * BITS_PER_BITMAP;

  if (tid == 0 && s_offset != 0) {
    mutable_bitmap_t l_bitmap = bitmap[bitmap_idx];
    l_bitmap >>= s_offset;

    size_t remaining_bits = min(size_t(BITS_PER_BITMAP - s_offset), e_bit - s_bit);
    if (remaining_bits < BITS_PER_BITMAP) {
      l_bitmap &= ((mutable_bitmap_t(1) << remaining_bits) - 1);
    }

#pragma unroll
    for (int i = 0; i < BITS_PER_BITMAP; i++) {
      guard[i] = l_bitmap & (ONE << i);
    }
#pragma unroll
    for (int i = 0; i < BITS_PER_BITMAP; i++) {
      stg(index_t(i + g_bits), sub_cols_indices_addr + l_sum, guard[i]);
      l_sum += guard[i];
    }
  }

  if (tid == 0) { g_sum = l_sum; }
  __syncthreads();

  if (s_offset != 0) {
    s_bit += (BITS_PER_BITMAP - s_offset);
    g_bits += (BITS_PER_BITMAP - s_offset);
  }

  for (size_t bit_idx = s_bit; bit_idx < e_bit; bit_idx += BITS_PER_BITMAP * blockDim.x) {
    mutable_bitmap_t l_bitmap = 0;
    bitmap_idx                = bit_idx / BITS_PER_BITMAP + tid;

    if (bitmap_idx * BITS_PER_BITMAP < e_bit) { l_bitmap = bitmap[bitmap_idx]; }

    index_t remaining_bits = min(BITS_PER_BITMAP, index_t(e_bit - bitmap_idx * BITS_PER_BITMAP));
    if (remaining_bits < BITS_PER_BITMAP) {
      l_bitmap &= ((mutable_bitmap_t(1) << remaining_bits) - 1);
    }

    int l_bits    = raft::detail::popc(l_bitmap);
    int l_sum_32b = 0;
    BlockScan(scan_storage).InclusiveSum(l_bits, l_sum_32b);
    l_sum = l_sum_32b + g_sum - l_bits;
    __syncthreads();

#pragma unroll
    for (int i = 0; i < BITS_PER_BITMAP; i++) {
      guard[i] = l_bitmap & (ONE << i);
    }
#pragma unroll
    for (int i = 0; i < BITS_PER_BITMAP; i++) {
      stg(index_t(i + g_bits), sub_cols_indices_addr + l_sum, guard[i]);
      l_sum += guard[i];
    }

    if (threadIdx.x == (bitmap_to_csr_tpb - 1)) { g_sum += (l_sum_32b); }
    g_bits += BITS_PER_BITMAP * blockDim.x;
  }
}

template <typename bitmap_t, typename index_t, typename nnz_t, bool check_nnz = false>
void fill_indices_by_rows(raft::resources const& handle,
                          const bitmap_t* bitmap,
                          index_t* indptr,
                          index_t num_rows,
                          index_t num_cols,
                          nnz_t nnz,
                          index_t* indices,
                          nnz_t* sub_col_nnz,
                          index_t bits_per_sub_col,
                          size_t sub_nnz_size)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  auto stream  = resource::get_cuda_stream(handle);
  auto block_x = num_rows;
  auto block_y = sub_nnz_size / num_rows;
  dim3 grid(block_x, block_y, 1);

  auto block = bitmap_to_csr_tpb;

  fill_indices_by_rows_kernel<bitmap_t, index_t, nnz_t, check_nnz><<<grid, block, 0, stream>>>(
    bitmap, indptr, num_rows, num_cols, nnz, indices, sub_col_nnz, bits_per_sub_col);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename bitmap_t,
          typename index_t,
          typename csr_matrix_t,
          typename = std::enable_if_t<raft::is_device_csr_matrix_v<csr_matrix_t>>>
void bitmap_to_csr(raft::resources const& handle,
                   raft::core::bitmap_view<bitmap_t, index_t> bitmap,
                   csr_matrix_t& csr)
{
  using nnz_t   = typename csr_matrix_t::nnz_type;
  auto csr_view = csr.structure_view();

  RAFT_EXPECTS(bitmap.get_n_rows() == csr_view.get_n_rows(),
               "Number of rows in bitmap must be equal to "
               "number of rows in csr");

  RAFT_EXPECTS(bitmap.get_n_cols() == csr_view.get_n_cols(),
               "Number of columns in bitmap must be equal to "
               "number of columns in csr");

  if (csr_view.get_n_rows() == 0 || csr_view.get_n_cols() == 0) { return; }

  auto thrust_policy = resource::get_thrust_policy(handle);
  auto stream        = resource::get_cuda_stream(handle);

  index_t* indptr  = csr_view.get_indptr().data();
  index_t* indices = csr_view.get_indices().data();

  if (!resource::get_dry_run_flag(handle)) {
    RAFT_CUDA_TRY(
      cudaMemsetAsync(indptr, 0, (csr_view.get_n_rows() + 1) * sizeof(index_t), stream));
  }

  size_t sub_nnz_size      = 0;
  index_t bits_per_sub_col = 0;

  // Get buffer size and number of bits per each sub-columns
  calc_nnz_by_rows(handle,
                   bitmap.data(),
                   csr_view.get_n_rows(),
                   csr_view.get_n_cols(),
                   static_cast<nnz_t*>(nullptr),
                   sub_nnz_size,
                   bits_per_sub_col);

  rmm::device_async_resource_ref device_memory = resource::get_workspace_resource_ref(handle);
  rmm::device_uvector<nnz_t> sub_nnz(sub_nnz_size + 1, stream, device_memory);

  size_t scan_ws_bytes = 0;
  cub::DeviceScan::ExclusiveSum(
    nullptr, scan_ws_bytes, sub_nnz.data(), sub_nnz.data(), sub_nnz_size + 1, stream);
  rmm::device_uvector<char> scan_ws(scan_ws_bytes, stream);

  if (resource::get_dry_run_flag(handle)) {
    if constexpr (is_device_csr_sparsity_owning_v<csr_matrix_t>) {
      auto safe_nnz = std::min(
        static_cast<uint64_t>(csr_view.get_n_rows()) * static_cast<uint64_t>(csr_view.get_n_cols()),
        static_cast<uint64_t>(std::numeric_limits<nnz_t>::max()));
      csr.initialize_sparsity(static_cast<nnz_t>(safe_nnz));
    }
    return;
  }

  calc_nnz_by_rows(handle,
                   bitmap.data(),
                   csr_view.get_n_rows(),
                   csr_view.get_n_cols(),
                   sub_nnz.data(),
                   sub_nnz_size,
                   bits_per_sub_col);

  cub::DeviceScan::ExclusiveSum(
    scan_ws.data(), scan_ws_bytes, sub_nnz.data(), sub_nnz.data(), sub_nnz_size + 1, stream);

  if constexpr (is_device_csr_sparsity_owning_v<csr_matrix_t>) {
    nnz_t nnz = 0;
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      &nnz, sub_nnz.data() + sub_nnz_size, sizeof(nnz_t), cudaMemcpyDeviceToHost, stream));
    resource::sync_stream(handle);
    csr.initialize_sparsity(nnz);
    if (nnz == 0) return;
  }

  constexpr bool check_nnz = is_device_csr_sparsity_preserving_v<csr_matrix_t>;
  fill_indices_by_rows<bitmap_t, index_t, nnz_t, check_nnz>(handle,
                                                            bitmap.data(),
                                                            indptr,
                                                            csr_view.get_n_rows(),
                                                            csr_view.get_n_cols(),
                                                            csr_view.get_nnz(),
                                                            indices,
                                                            sub_nnz.data(),
                                                            bits_per_sub_col,
                                                            sub_nnz_size);

  thrust::fill_n(thrust_policy,
                 csr.get_elements().data(),
                 csr_view.get_nnz(),
                 typename csr_matrix_t::element_type(1));
}

};  // end NAMESPACE detail
};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // namespace raft
