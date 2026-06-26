/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __REGRESSION_METRICS_H
#define __REGRESSION_METRICS_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/detail/scores.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute regression metrics mean absolute error, mean squared error, median absolute error
 * @tparam T: data type for predictions (e.g., float or double for regression).
 * @param[in] predictions: array of predictions (GPU pointer).
 * @param[in] ref_predictions: array of reference (ground-truth) predictions (GPU pointer).
 * @param[in] n: number of elements in each of predictions, ref_predictions. Should be > 0.
 * @param[in] stream: cuda stream.
 * @param[out] mean_abs_error: Mean Absolute Error. Sum over n of (|predictions[i] -
 * ref_predictions[i]|) / n.
 * @param[out] mean_squared_error: Mean Squared Error. Sum over n of ((predictions[i] -
 * ref_predictions[i])^2) / n.
 * @param[out] median_abs_error: Median Absolute Error. Median of |predictions[i] -
 * ref_predictions[i]| for i in [0, n).
 */
template <typename T>
void regression_metrics(const T* predictions,
                        const T* ref_predictions,
                        int n,
                        cudaStream_t stream,
                        double& mean_abs_error,
                        double& mean_squared_error,
                        double& median_abs_error)
{
  detail::regression_metrics(false,
                             predictions,
                             ref_predictions,
                             n,
                             stream,
                             mean_abs_error,
                             mean_squared_error,
                             median_abs_error);
}

/**
 * @defgroup stats_regression_metrics Regression Metrics
 * @{
 */

/**
 * @brief Compute regression metrics mean absolute error, mean squared error, median absolute error
 * @tparam value_t the data type for predictions (e.g., float or double for regression).
 * @tparam idx_t index type
 * @param[in]  handle the raft handle
 * @param[in]  predictions: array of predictions.
 * @param[in]  ref_predictions: array of reference (ground-truth) predictions.
 * @param[out] mean_abs_error: Mean Absolute Error. Sum over n of (|predictions[i] -
 * ref_predictions[i]|) / n.
 * @param[out] mean_squared_error: Mean Squared Error. Sum over n of ((predictions[i] -
 * ref_predictions[i])^2) / n.
 * @param[out] median_abs_error: Median Absolute Error. Median of |predictions[i] -
 * ref_predictions[i]| for i in [0, n).
 */
template <typename value_t, typename idx_t>
void regression_metrics(raft::resources const& handle,
                        raft::device_vector_view<const value_t, idx_t> predictions,
                        raft::device_vector_view<const value_t, idx_t> ref_predictions,
                        raft::host_scalar_view<double> mean_abs_error,
                        raft::host_scalar_view<double> mean_squared_error,
                        raft::host_scalar_view<double> median_abs_error)
{
  RAFT_EXPECTS(predictions.extent(0) == ref_predictions.extent(0),
               "Size mismatch between predictions and ref_predictions");
  RAFT_EXPECTS(predictions.is_exhaustive(), "predictions must be contiguous");
  RAFT_EXPECTS(ref_predictions.is_exhaustive(), "ref_predictions must be contiguous");
  RAFT_EXPECTS(mean_abs_error.data_handle() != nullptr, "mean_abs_error view must not be empty");
  RAFT_EXPECTS(mean_squared_error.data_handle() != nullptr,
               "mean_squared_error view must not be empty");
  RAFT_EXPECTS(median_abs_error.data_handle() != nullptr,
               "median_abs_error view must not be empty");
  detail::regression_metrics(resource::get_dry_run_flag(handle),
                             predictions.data_handle(),
                             ref_predictions.data_handle(),
                             predictions.extent(0),
                             resource::get_cuda_stream(handle),
                             *mean_abs_error.data_handle(),
                             *mean_squared_error.data_handle(),
                             *median_abs_error.data_handle());
}

/** @} */  // end group stats_regression_metrics

}  // namespace stats
}  // namespace raft

#endif
