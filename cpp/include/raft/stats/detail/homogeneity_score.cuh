/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file homogeneity_score.cuh
 *
 * @brief A clustering result satisfies homogeneity if all of its clusters
 * contain only data points which are members of a single class.
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/stats/entropy.cuh>
#include <raft/stats/mutual_info_score.cuh>

namespace raft {
namespace stats {
namespace detail {
/**
 * @brief Function to calculate the homogeneity score between two clusters
 * <a href="https://en.wikipedia.org/wiki/Homogeneity_(statistics)">more info on mutual
 * information</a>
 * @param dry_run: whether to run in dry-run mode
 * @param truthClusterArray: the array of truth classes of type T
 * @param predClusterArray: the array of predicted classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 */
template <typename T>
double homogeneity_score(bool dry_run,
                         const T* truthClusterArray,
                         const T* predClusterArray,
                         int size,
                         T lowerLabelRange,
                         T upperLabelRange,
                         cudaStream_t stream)
{
  if (size == 0) return 1.0;

  double computedMI, computedEntropy;

  computedMI = mutual_info_score(
    dry_run, truthClusterArray, predClusterArray, size, lowerLabelRange, upperLabelRange, stream);
  computedEntropy =
    entropy(dry_run, truthClusterArray, size, lowerLabelRange, upperLabelRange, stream);

  double homogeneity;

  if (computedEntropy) {
    homogeneity = computedMI / computedEntropy;
  } else
    homogeneity = 1.0;

  return homogeneity;
}

};  // end namespace detail
};  // end namespace stats
};  // namespace raft
