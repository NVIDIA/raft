/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __RAND_INDEX_H
#define __RAND_INDEX_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/detail/rand_index.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate RandIndex
 * <a href="https://en.wikipedia.org/wiki/Rand_index">more info on rand index</a>
 * @param firstClusterArray: the array of classes of type T
 * @param secondClusterArray: the array of classes of type T
 * @param size: the size of the data points of type uint64_t
 * @param stream: the cudaStream object
 */
template <typename T>
double rand_index(T* firstClusterArray, T* secondClusterArray, uint64_t size, cudaStream_t stream)
{
  return detail::compute_rand_index(false, firstClusterArray, secondClusterArray, size, stream);
}

/**
 * @defgroup stats_rand_index Rand Index
 * @{
 */

/**
 * @brief Function to calculate RandIndex
 * <a href="https://en.wikipedia.org/wiki/Rand_index">more info on rand index</a>
 * @tparam value_t the data type
 * @tparam idx_t index type
 * @param[in] handle the raft handle
 * @param[in] first_cluster_array: the array of classes of type value_t
 * @param[in] second_cluster_array: the array of classes of type value_t
 * @return: The RandIndex value.
 */
template <typename value_t, typename idx_t>
double rand_index(raft::resources const& handle,
                  raft::device_vector_view<const value_t, idx_t> first_cluster_array,
                  raft::device_vector_view<const value_t, idx_t> second_cluster_array)
{
  RAFT_EXPECTS(first_cluster_array.extent(0) == second_cluster_array.extent(0),
               "Size mismatch between first_cluster_array and second_cluster_array");
  RAFT_EXPECTS(first_cluster_array.is_exhaustive(), "first_cluster_array must be contiguous");
  RAFT_EXPECTS(second_cluster_array.is_exhaustive(), "second_cluster_array must be contiguous");
  return detail::compute_rand_index(resource::get_dry_run_flag(handle),
                                    first_cluster_array.data_handle(),
                                    second_cluster_array.data_handle(),
                                    second_cluster_array.extent(0),
                                    resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_rand_index

};  // end namespace stats
};  // namespace raft
#endif
