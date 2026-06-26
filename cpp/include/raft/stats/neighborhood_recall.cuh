/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/neighborhood_recall.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

#include <optional>

namespace raft {
namespace stats {

/**
 * @defgroup stats_neighborhood_recall Neighborhood Recall Score
 * @{
 */

/**
 * @brief Calculate Neighborhood Recall score on the device for indices, distances computed by any
 * Nearest Neighbors Algorithm against reference indices, distances. Recall score is calculated by
 * comparing the total number of matching indices and dividing that value by the total size of the
 * indices matrix of dimensions (D, k). If distance matrices are provided, then non-matching indices
 * could be considered a match if abs(dist, ref_dist) < eps.
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources res;
 * // assume D rows and N column dataset
 * auto k = 64;
 * auto indices = raft::make_device_matrix<int>(res, D, k);
 * auto distances = raft::make_device_matrix<float>(res, D, k);
 * // run ANN algorithm of choice
 *
 * auto ref_indices = raft::make_device_matrix<int>(res, D, k);
 * auto ref_distances = raft::make_device_matrix<float>(res, D, k);
 * // run brute-force KNN for reference
 *
 * auto scalar = 0.0f;
 * auto recall_score = raft::make_device_scalar(res, scalar);
 *
 * raft::stats::neighborhood_recall(res,
                                    raft::make_const_mdspan(indices.view()),
                                    raft::make_const_mdspan(ref_indices.view()),
                                    recall_score.view(),
                                    raft::make_const_mdspan(distances.view()),
                                    raft::make_const_mdspan(ref_distances.view()));
 * @endcode
 *
 * @tparam IndicesValueType data-type of the indices
 * @tparam IndexType data-type to index all matrices
 * @tparam ScalarType data-type to store recall score
 * @tparam DistanceValueType data-type of the distances
 * @param res raft::resources object to manage resources
 * @param[in] indices raft::device_matrix_view indices of neighbors
 * @param[in] ref_indices raft::device_matrix_view reference indices of neighbors
 * @param[out] recall_score raft::device_scalar_view output recall score
 * @param[in] distances (optional) raft::device_matrix_view distances of neighbors
 * @param[in] ref_distances (optional) raft::device_matrix_view reference distances of neighbors
 * @param[in] eps (optional, default = 0.001) value within which distances are considered matching
 */
template <typename IndicesValueType,
          typename IndexType,
          typename ScalarType,
          typename DistanceValueType = float>
void neighborhood_recall(
  raft::resources const& res,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> indices,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> ref_indices,
  raft::device_scalar_view<ScalarType> recall_score,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    distances = std::nullopt,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    ref_distances                                                    = std::nullopt,
  std::optional<raft::host_scalar_view<const DistanceValueType>> eps = std::nullopt)
{
  RAFT_EXPECTS(indices.extent(0) == ref_indices.extent(0),
               "The number of rows in indices and reference indices should be equal");
  RAFT_EXPECTS(indices.extent(1) == ref_indices.extent(1),
               "The number of columns in indices and reference indices should be equal");

  if (distances.has_value() or ref_distances.has_value()) {
    RAFT_EXPECTS(distances.has_value() and ref_distances.has_value(),
                 "Both distances and reference distances should have values");

    RAFT_EXPECTS(distances.value().extent(0) == ref_distances.value().extent(0),
                 "The number of rows in distances and reference distances should be equal");
    RAFT_EXPECTS(distances.value().extent(1) == ref_distances.value().extent(1),
                 "The number of columns in indices and reference indices should be equal");

    RAFT_EXPECTS(indices.extent(0) == distances.value().extent(0),
                 "The number of rows in indices and distances should be equal");
    RAFT_EXPECTS(indices.extent(1) == distances.value().extent(1),
                 "The number of columns in indices and distances should be equal");
  }

  DistanceValueType eps_val = 0.001;
  if (eps.has_value()) { eps_val = *eps.value().data_handle(); }

  detail::neighborhood_recall(
    res, indices, ref_indices, distances, ref_distances, recall_score, eps_val);
}

/**
 * @brief Calculate Neighborhood Recall score on the host for indices, distances computed by any
 * Nearest Neighbors Algorithm against reference indices, distances. Recall score is calculated by
 * comparing the total number of matching indices and dividing that value by the total size of the
 * indices matrix of dimensions (D, k). If distance matrices are provided, then non-matching indices
 * could be considered a match if abs(dist, ref_dist) < eps.
 *
 * Usage example:
 * @code{.cpp}
 * raft::device_resources res;
 * // assume D rows and N column dataset
 * auto k = 64;
 * auto indices = raft::make_device_matrix<int>(res, D, k);
 * auto distances = raft::make_device_matrix<float>(res, D, k);
 * // run ANN algorithm of choice
 *
 * auto ref_indices = raft::make_device_matrix<int>(res, D, k);
 * auto ref_distances = raft::make_device_matrix<float>(res, D, k);
 * // run brute-force KNN for reference
 *
 * auto scalar = 0.0f;
 * auto recall_score = raft::make_host_scalar(res, scalar);
 *
 * raft::stats::neighborhood_recall(res,
                                    raft::make_const_mdspan(indices.view()),
                                    raft::make_const_mdspan(ref_indices.view()),
                                    recall_score.view(),
                                    raft::make_const_mdspan(distances.view()),
                                    raft::make_const_mdspan(ref_distances.view()));
 * @endcode
 *
 * @tparam IndicesValueType data-type of the indices
 * @tparam IndexType data-type to index all matrices
 * @tparam ScalarType data-type to store recall score
 * @tparam DistanceValueType data-type of the distances
 * @param res raft::resources object to manage resources
 * @param[in] indices raft::device_matrix_view indices of neighbors
 * @param[in] ref_indices raft::device_matrix_view reference indices of neighbors
 * @param[out] recall_score raft::host_scalar_view output recall score
 * @param[in] distances (optional) raft::device_matrix_view distances of neighbors
 * @param[in] ref_distances (optional) raft::device_matrix_view reference distances of neighbors
 * @param[in] eps (optional, default = 0.001) value within which distances are considered matching
 */
template <typename IndicesValueType,
          typename IndexType,
          typename ScalarType,
          typename DistanceValueType = float>
void neighborhood_recall(
  raft::resources const& res,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> indices,
  raft::device_matrix_view<const IndicesValueType, IndexType, raft::row_major> ref_indices,
  raft::host_scalar_view<ScalarType> recall_score,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    distances = std::nullopt,
  std::optional<raft::device_matrix_view<const DistanceValueType, IndexType, raft::row_major>>
    ref_distances                                                    = std::nullopt,
  std::optional<raft::host_scalar_view<const DistanceValueType>> eps = std::nullopt)
{
  auto recall_score_d = raft::make_device_scalar(res, *recall_score.data_handle());
  neighborhood_recall(
    res, indices, ref_indices, recall_score_d.view(), distances, ref_distances, eps);
  if (resource::get_dry_run_flag(res)) { return; }
  raft::update_host(recall_score.data_handle(),
                    recall_score_d.data_handle(),
                    1,
                    raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
}

/** @} */  // end group stats_recall

}  // namespace stats
}  // namespace raft
