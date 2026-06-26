/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/**
 * @file entropy.cuh
 * @brief Calculates the entropy for a labeling in nats.(ie, uses natural logarithm for the
 * calculations)
 */

#pragma once
#include <raft/core/detail/macros.hpp>
#include <raft/linalg/divide.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_histogram.cuh>

#include <math.h>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Lambda to calculate the entropy of a sample given its probability value
 *
 * @param p: the input to the functional mapping
 * @param q: dummy param
 */
struct entropyOp {
  HDI double operator()(double p, double q)
  {
    if (p)
      return -1 * (p) * (log(p));
    else
      return 0.0;
  }
};

/**
 * @brief function to calculate the bincounts of number of samples in every label
 *
 * @tparam LabelT: type of the labels
 * @param dry_run: whether to run in dry-run mode
 * @param labels: the pointer to the array containing labels for every data sample
 * @param binCountArray: pointer to the 1D array that contains the count of samples per cluster.
 *                       Can be nullptr when workspace is nullptr (for size query).
 * @param nRows: number of data samples
 * @param lowerLabelRange
 * @param upperLabelRange
 * @param workspace: device buffer containing workspace memory. Pass nullptr to query workspace
 * size.
 * @param workspace_size: [in/out] When workspace is nullptr, this is set to the required workspace
 * size. When workspace is not nullptr, this must be the size of the workspace.
 * @param stream: the cuda stream where to launch this kernel
 */
template <typename LabelT>
void countLabels(const LabelT* labels,
                 double* binCountArray,
                 int nRows,
                 LabelT lowerLabelRange,
                 LabelT upperLabelRange,
                 void* workspace,
                 size_t& workspace_size,
                 cudaStream_t stream)
{
  int num_levels     = upperLabelRange - lowerLabelRange + 2;
  LabelT lower_level = lowerLabelRange;
  LabelT upper_level = upperLabelRange + 1;

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace,
                                                    workspace_size,
                                                    labels,
                                                    binCountArray,
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    nRows,
                                                    stream));
}

/**
 * @brief Function to calculate entropy
 * <a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">more info on entropy</a>
 *
 * @param dry_run: whether to run in dry-run mode
 * @param clusterArray: the array of classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 * @return the entropy score
 */
template <typename T>
double entropy(bool dry_run,
               const T* clusterArray,
               const int size,
               const T lowerLabelRange,
               const T upperLabelRange,
               cudaStream_t stream)
{
  if (!size) return 1.0;

  T numUniqueClasses = upperLabelRange - lowerLabelRange + 1;

  // declaring, allocating and initializing memory for bincount array and entropy values
  rmm::device_uvector<double> prob(numUniqueClasses, stream);
  if (!dry_run) {
    RAFT_CUDA_TRY(cudaMemsetAsync(prob.data(), 0, numUniqueClasses * sizeof(double), stream));
  }
  rmm::device_scalar<double> d_entropy(stream);
  if (!dry_run) { RAFT_CUDA_TRY(cudaMemsetAsync(d_entropy.data(), 0, sizeof(double), stream)); }

  // Query workspace size for countLabels (can run in dry-run)
  size_t countLabels_ws_size = 0;
  countLabels(clusterArray,
              nullptr,
              size,
              lowerLabelRange,
              upperLabelRange,
              nullptr,
              countLabels_ws_size,
              stream);
  // workspace allocation
  rmm::device_uvector<char> workspace(countLabels_ws_size, stream);

  if (dry_run) { return 0.0; }

  // calculating the bincounts and populating the prob array
  countLabels(clusterArray,
              prob.data(),
              size,
              lowerLabelRange,
              upperLabelRange,
              workspace.data(),
              countLabels_ws_size,
              stream);

  // scalar dividing by size
  raft::linalg::divideScalar<double>(
    prob.data(), prob.data(), (double)size, numUniqueClasses, stream);

  // calculating the aggregate entropy
  raft::linalg::mapThenSumReduce<double, entropyOp>(
    d_entropy.data(), numUniqueClasses, entropyOp(), stream, prob.data(), prob.data());

  // updating in the host memory
  double h_entropy;
  raft::update_host(&h_entropy, d_entropy.data(), 1, stream);

  raft::interruptible::synchronize(stream);

  return h_entropy;
}

};  // end namespace detail
};  // end namespace stats
};  // namespace raft
