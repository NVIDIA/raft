/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __COMPLETENESS_SCORE_H
#define __COMPLETENESS_SCORE_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/stats/detail/homogeneity_score.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate the completeness score between two clusters
 *
 * @param truthClusterArray: the array of truth classes of type T
 * @param predClusterArray: the array of predicted classes of type T
 * @param size: the size of the data points of type int
 * @param lower_label_range: the lower bound of the range of labels
 * @param upper_label_range: the upper bound of the range of labels
 * @param stream: the cudaStream object
 */
template <typename T>
double completeness_score(const T* truthClusterArray,
                          const T* predClusterArray,
                          int size,
                          T lower_label_range,
                          T upper_label_range,
                          cudaStream_t stream)
{
  return detail::homogeneity_score(
    false, predClusterArray, truthClusterArray, size, lower_label_range, upper_label_range, stream);
}

/**
 * @defgroup stats_completeness Completeness Score
 * @{
 */

/**
 * @brief Function to calculate the completeness score between two clusters
 *
 * @tparam value_t the data type
 * @tparam idx_t Index type of matrix extent.
 * @param[in] handle: the raft handle.
 * @param[in] truth_cluster_array: the array of truth classes of type value_t
 * @param[in] pred_cluster_array: the array of predicted classes of type value_t
 * @param[in] lower_label_range: the lower bound of the range of labels
 * @param[in] upper_label_range: the upper bound of the range of labels
 * @return the cluster completeness score
 */
template <typename value_t, typename idx_t>
double completeness_score(raft::resources const& handle,
                          raft::device_vector_view<const value_t, idx_t> truth_cluster_array,
                          raft::device_vector_view<const value_t, idx_t> pred_cluster_array,
                          value_t lower_label_range,
                          value_t upper_label_range)
{
  RAFT_EXPECTS(truth_cluster_array.size() == pred_cluster_array.size(), "Size mismatch");
  RAFT_EXPECTS(truth_cluster_array.is_exhaustive(), "truth_cluster_array must be contiguous");
  RAFT_EXPECTS(pred_cluster_array.is_exhaustive(), "pred_cluster_array must be contiguous");
  return detail::homogeneity_score(resource::get_dry_run_flag(handle),
                                   pred_cluster_array.data_handle(),
                                   truth_cluster_array.data_handle(),
                                   truth_cluster_array.extent(0),
                                   lower_label_range,
                                   upper_label_range,
                                   resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_completeness

};  // end namespace stats
};  // namespace raft
#endif
