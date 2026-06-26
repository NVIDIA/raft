/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __V_MEASURE_H
#define __V_MEASURE_H

#pragma once
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/detail/v_measure.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate the v-measure between two clusters
 *
 * @param truthClusterArray: the array of truth classes of type T
 * @param predClusterArray: the array of predicted classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 * @param beta: v_measure parameter
 */
template <typename T>
double v_measure(const T* truthClusterArray,
                 const T* predClusterArray,
                 int size,
                 T lowerLabelRange,
                 T upperLabelRange,
                 cudaStream_t stream,
                 double beta = 1.0)
{
  return detail::v_measure(false,
                           truthClusterArray,
                           predClusterArray,
                           size,
                           lowerLabelRange,
                           upperLabelRange,
                           stream,
                           beta);
}

/**
 * @defgroup stats_vmeasure V-Measure
 * @{
 */

/**
 * @brief Function to calculate the v-measure between two clusters
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @param[in] handle the raft handle
 * @param[in] truth_cluster_array: the array of truth classes of type T
 * @param[in] pred_cluster_array: the array of predicted classes of type T
 * @param[in] lower_label_range: the lower bound of the range of labels
 * @param[in] upper_label_range: the upper bound of the range of labels
 * @param[in] beta: v_measure parameter
 * @return the v-measure between the two clusters
 */
template <typename value_t, typename idx_t>
double v_measure(raft::resources const& handle,
                 raft::device_vector_view<const value_t, idx_t> truth_cluster_array,
                 raft::device_vector_view<const value_t, idx_t> pred_cluster_array,
                 value_t lower_label_range,
                 value_t upper_label_range,
                 double beta = 1.0)
{
  RAFT_EXPECTS(truth_cluster_array.extent(0) == pred_cluster_array.extent(0),
               "Size mismatch between truth_cluster_array and pred_cluster_array");
  RAFT_EXPECTS(truth_cluster_array.is_exhaustive(), "truth_cluster_array must be contiguous");
  RAFT_EXPECTS(pred_cluster_array.is_exhaustive(), "pred_cluster_array must be contiguous");

  return detail::v_measure(resource::get_dry_run_flag(handle),
                           truth_cluster_array.data_handle(),
                           pred_cluster_array.data_handle(),
                           truth_cluster_array.extent(0),
                           lower_label_range,
                           upper_label_range,
                           resource::get_cuda_stream(handle),
                           beta);
}

/** @} */  // end group stats_vmeasure

};  // end namespace stats
};  // namespace raft
#endif
