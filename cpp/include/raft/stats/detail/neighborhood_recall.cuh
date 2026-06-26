/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/math.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>

#include <cstddef>
#include <optional>

namespace raft {
namespace stats::detail {

template <typename IndicesValueType,
          typename DistanceValueType,
          typename IndexType,
          typename ScalarType>
RAFT_KERNEL neighborhood_recall(
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> indices,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> ref_indices,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    distances,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    ref_distances,
  raft::device_scalar_view<ScalarType> recall_score,
  DistanceValueType const eps)
{
  auto constexpr kThreadsPerBlock = 32;
  IndexType const row_idx         = blockIdx.x;
  auto const lane_idx             = threadIdx.x % kThreadsPerBlock;

  // Each warp stores a recall score computed across the columns per row
  IndexType thread_recall_score = 0;
  for (IndexType col_idx = lane_idx; col_idx < indices.extent(1); col_idx += kThreadsPerBlock) {
    for (IndexType ref_col_idx = 0; ref_col_idx < ref_indices.extent(1); ref_col_idx++) {
      if (indices(row_idx, col_idx) == ref_indices(row_idx, ref_col_idx)) {
        thread_recall_score += 1;
        break;
      } else if (distances.has_value()) {
        auto dist               = distances.value()(row_idx, col_idx);
        auto ref_dist           = ref_distances.value()(row_idx, ref_col_idx);
        DistanceValueType diff  = raft::abs(dist - ref_dist);
        DistanceValueType m     = std::max(raft::abs(dist), raft::abs(ref_dist));
        DistanceValueType ratio = diff > eps ? diff / m : diff;

        if (ratio <= eps) {
          thread_recall_score += 1;
          break;
        }
      }
    }
  }

  // Reduce across a warp for row score
  typedef cub::BlockReduce<IndexType, kThreadsPerBlock> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  ScalarType row_recall_score = BlockReduce(temp_storage).Sum(thread_recall_score);

  // Reduce across all rows for global score
  if (lane_idx == 0) {
    cuda::atomic_ref<ScalarType, cuda::thread_scope_device> device_recall_score{
      *recall_score.data_handle()};
    std::size_t const total_count = indices.extent(0) * indices.extent(1);
    device_recall_score.fetch_add(row_recall_score / total_count);
  }
}

template <typename IndicesValueType,
          typename DistanceValueType,
          typename IndexType,
          typename ScalarType>
void neighborhood_recall(
  raft::resources const& res,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> indices,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> ref_indices,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    distances,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    ref_distances,
  raft::device_scalar_view<ScalarType> recall_score,
  DistanceValueType const eps)
{
  if (resource::get_dry_run_flag(res)) { return; }
  // One warp per row, launch a warp-width block per-row kernel
  auto constexpr kThreadsPerBlock = 32;
  auto const num_blocks           = indices.extent(0);

  neighborhood_recall<<<num_blocks, kThreadsPerBlock, 0, raft::resource::get_cuda_stream(res)>>>(
    indices, ref_indices, distances, ref_distances, recall_score, eps);
}

}  // namespace stats::detail
}  // namespace raft
