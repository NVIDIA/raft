/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>
#include <raft/util/vectorized.cuh>

#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

// Threads per block in adj_to_csr_kernel.
static const constexpr int adj_to_csr_tpb = 512;

/**
 * @brief Convert dense adjacency matrix into unsorted CSR format.
 *
 * The adj_to_csr kernel converts a boolean adjacency matrix into CSR
 * format. High performance comes at the cost of non-deterministic output: the
 * column indices are not guaranteed to be stored in order.
 *
 * The kernel has been optimized to handle matrices that are non-square, for
 * instance subsets of a full adjacency matrix. In practice, these matrices can
 * be very wide and not very tall. In principle, each row is assigned to one
 * block. If there are more SMs than rows, multiple blocks operate on a single
 * row. To enable cooperation between these blocks, each row is provided a
 * counter where the current output index can be cooperatively (atomically)
 * incremented. As a result, the order of the output indices is not guaranteed.
 *
 * @param[in]     adj          A num_rows x num_cols boolean matrix in contiguous row-major
 *                             format.
 * @param[in]     row_ind      An array of length num_rows that indicates at which index
 *                             a row starts in out_col_ind. Equivalently, it is the
 *                             exclusive scan of the number of non-zeros in each row of
 *                             adj.
 * @param[in]     num_rows     Number of rows of adj.
 * @param[in]     num_cols     Number of columns of adj.
 * @param[in,out] row_counters A temporary zero-initialized array of length num_rows.
 * @param[out]    out_col_ind  An array containing the column indices of the
 *                             non-zero values in `adj`. Size should be at least
 *                             the number of non-zeros in `adj`.
 */
template <typename index_t>
RAFT_KERNEL __launch_bounds__(adj_to_csr_tpb)
  adj_to_csr_kernel(const bool* adj,         // row-major adjacency matrix
                    const index_t* row_ind,  // precomputed row indices
                    index_t num_rows,        // # rows of adj
                    index_t num_cols,        // # cols of adj
                    index_t* row_counters,   // pre-allocated (zeroed) atomic counters
                    index_t* out_col_ind     // output column indices
  )
{
  const int chunk_size = 16;
  typedef raft::TxN_t<bool, chunk_size> chunk_bool;

  for (index_t i = blockIdx.y; i < num_rows; i += gridDim.y) {
    // Load row information
    index_t row_base   = row_ind[i];
    index_t* row_count = row_counters + i;
    const bool* row    = adj + i * num_cols;

    // Peeling: process the first j0 elements that are not aligned to a chunk_size-byte
    // boundary.
    index_t j0 = (chunk_size - (((uintptr_t)(const void*)row) % chunk_size)) % chunk_size;
    j0         = min(j0, num_cols);
    if (threadIdx.x < j0 && blockIdx.x == 0) {
      if (row[threadIdx.x]) { out_col_ind[row_base + atomicIncWarp(row_count)] = threadIdx.x; }
    }

    // Process the rest of the row in chunk_size byte chunks starting at j0.
    // This is a grid-stride loop.
    index_t j = j0 + chunk_size * (blockIdx.x * blockDim.x + threadIdx.x);
    for (; j + chunk_size - 1 < num_cols; j += chunk_size * (blockDim.x * gridDim.x)) {
      chunk_bool chunk;
      chunk.load(row, j);
      for (int k = 0; k < chunk_size; ++k) {
        if (chunk.val.data[k]) { out_col_ind[row_base + atomicIncWarp(row_count)] = j + k; }
      }
    }

    // Remainder: process the last j1 bools in the row individually.
    index_t j1 = (num_cols - j0) % chunk_size;
    if (threadIdx.x < j1 && blockIdx.x == 0) {
      int j = num_cols - j1 + threadIdx.x;
      if (row[j]) { out_col_ind[row_base + atomicIncWarp(row_count)] = j; }
    }
  }
}

/**
 * @brief Converts a boolean adjacency matrix into unsorted CSR format.
 *
 * The conversion supports non-square matrices.
 *
 * @tparam     index_t     Indexing arithmetic type
 *
 * @param[in]  handle      RAFT handle
 * @param[in]  adj         A num_rows x num_cols boolean matrix in contiguous row-major
 *                         format.
 * @param[in]  row_ind     An array of length num_rows that indicates at which index
 *                         a row starts in out_col_ind. Equivalently, it is the
 *                         exclusive scan of the number of non-zeros in each row of
 *                         adj.
 * @param[in]  num_rows    Number of rows of adj.
 * @param[in]  num_cols    Number of columns of adj.
 * @param      tmp         A pre-allocated array of size num_rows.
 * @param[out] out_col_ind An array containing the column indices of the
 *                         non-zero values in adj. Size should be at least the
 *                         number of non-zeros in adj.
 */
template <typename index_t = int>
void adj_to_csr(raft::resources const& handle,
                const bool* adj,         // row-major adjacency matrix
                const index_t* row_ind,  // precomputed row indices
                index_t num_rows,        // # rows of adj
                index_t num_cols,        // # cols of adj
                index_t* tmp,            // pre-allocated atomic counters
                index_t* out_col_ind     // output column indices
)
{
  if (resource::get_dry_run_flag(handle)) { return; }  // No allocations below

  auto stream = resource::get_cuda_stream(handle);

  // Check inputs and return early if possible.
  if (num_rows == 0 || num_cols == 0) { return; }
  RAFT_EXPECTS(tmp != nullptr, "adj_to_csr: tmp workspace may not be null.");

  // Zero-fill a temporary vector that is be used by the kernel to keep track of
  // the number of entries added to a row.
  RAFT_CUDA_TRY(cudaMemsetAsync(tmp, 0, num_rows * sizeof(index_t), stream));

  // Split the grid in the row direction (since each row can be processed
  // independently). If the maximum number of active blocks (num_sms *
  // occupancy) exceeds the number of rows, assign multiple blocks to a single
  // row.
  int dev_id, sm_count, blocks_per_sm;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, adj_to_csr_kernel<index_t>, adj_to_csr_tpb, 0);

  index_t max_active_blocks = sm_count * blocks_per_sm;
  index_t blocks_per_row    = raft::ceildiv(max_active_blocks, num_rows);
  index_t grid_rows         = raft::ceildiv(max_active_blocks, blocks_per_row);
  dim3 block(adj_to_csr_tpb, 1);
  dim3 grid(blocks_per_row, grid_rows);

  adj_to_csr_kernel<index_t>
    <<<grid, block, 0, stream>>>(adj, row_ind, num_rows, num_cols, tmp, out_col_ind);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

};  // end NAMESPACE detail
};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // namespace raft
