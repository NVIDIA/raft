/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __MUTUAL_INFO_SCORE_H
#define __MUTUAL_INFO_SCORE_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/stats/detail/mutual_info_score.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate the mutual information between two clusters
 * <a href="https://en.wikipedia.org/wiki/Mutual_information">more info on mutual information</a>
 * @param firstClusterArray: the array of classes of type T
 * @param secondClusterArray: the array of classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 */
template <typename T>
double mutual_info_score(const T* firstClusterArray,
                         const T* secondClusterArray,
                         int size,
                         T lowerLabelRange,
                         T upperLabelRange,
                         cudaStream_t stream)
{
  return detail::mutual_info_score(
    false, firstClusterArray, secondClusterArray, size, lowerLabelRange, upperLabelRange, stream);
}

/**
 * @defgroup stats_mutual_info Mutual Information
 * @{
 */

/**
 * @brief Function to calculate the mutual information between two clusters
 * <a href="https://en.wikipedia.org/wiki/Mutual_information">more info on mutual information</a>
 * @tparam value_t the data type
 * @tparam idx_t index type
 * @param[in] handle the raft handle
 * @param[in] first_cluster_array: the array of classes of type value_t
 * @param[in] second_cluster_array: the array of classes of type value_t
 * @param[in] lower_label_range: the lower bound of the range of labels
 * @param[in] upper_label_range: the upper bound of the range of labels
 * @return the mutual information score
 */
template <typename value_t, typename idx_t>
double mutual_info_score(raft::resources const& handle,
                         raft::device_vector_view<const value_t, idx_t> first_cluster_array,
                         raft::device_vector_view<const value_t, idx_t> second_cluster_array,
                         value_t lower_label_range,
                         value_t upper_label_range)
{
  RAFT_EXPECTS(first_cluster_array.extent(0) == second_cluster_array.extent(0),
               "Size mismatch between first_cluster_array and second_cluster_array");
  RAFT_EXPECTS(first_cluster_array.is_exhaustive(), "first_cluster_array must be contiguous");
  RAFT_EXPECTS(second_cluster_array.is_exhaustive(), "second_cluster_array must be contiguous");
  return detail::mutual_info_score(resource::get_dry_run_flag(handle),
                                   first_cluster_array.data_handle(),
                                   second_cluster_array.data_handle(),
                                   first_cluster_array.extent(0),
                                   lower_label_range,
                                   upper_label_range,
                                   resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_mutual_info

};  // end namespace stats
};  // namespace raft
#endif
