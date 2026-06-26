/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>

#include <cub/device/device_histogram.cuh>

#include <math.h>

#include <algorithm>
#include <iostream>
#include <numeric>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief kernel that calculates the average intra-cluster distance for every sample data point and
 * updates the cluster distance to max value
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param sampleToClusterSumOfDistances: the pointer to the 2D array that contains the sum of
 * distances from every sample to every cluster (nRows x nLabels)
 * @param binCountArray: pointer to the 1D array that contains the count of samples per cluster (1 x
 * nLabels)
 * @param d_aArray: the pointer to the array of average intra-cluster distances for every sample in
 * device memory (1 x nRows)
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nRows: number of data samples
 * @param nLabels: number of Labels
 * @param MAX_VAL: DataT specific upper limit
 */
template <typename DataT, typename LabelT>
RAFT_KERNEL populateAKernel(DataT* sampleToClusterSumOfDistances,
                            DataT* binCountArray,
                            DataT* d_aArray,
                            const LabelT* labels,
                            int nRows,
                            int nLabels,
                            const DataT MAX_VAL)
{
  // getting the current index
  int sampleIndex = threadIdx.x + blockIdx.x * blockDim.x;

  if (sampleIndex >= nRows) return;

  // sampleDistanceVector is an array that stores that particular row of the distanceMatrix
  DataT* sampleToClusterSumOfDistancesVector =
    &sampleToClusterSumOfDistances[sampleIndex * nLabels];

  LabelT sampleCluster = labels[sampleIndex];

  int sampleClusterIndex = (int)sampleCluster;

  if (binCountArray[sampleClusterIndex] - 1 <= 0) {
    d_aArray[sampleIndex] = -1;
    return;

  }

  else {
    d_aArray[sampleIndex] = (sampleToClusterSumOfDistancesVector[sampleClusterIndex]) /
                            (binCountArray[sampleClusterIndex] - 1);

    // modifying the sampleDistanceVector to give sample average distance
    sampleToClusterSumOfDistancesVector[sampleClusterIndex] = MAX_VAL;
  }
}

/**
 * @brief function to calculate the bincounts of number of samples in every label
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param binCountArray: pointer to the 1D array that contains the count of samples per cluster (1 x
 * nLabels). Can be nullptr when workspace is nullptr (for size query).
 * @param nRows: number of data samples
 * @param nUniqueLabels: number of Labels
 * @param workspace: device buffer containing workspace memory. Pass nullptr to query workspace
 * size.
 * @param workspace_size: [in/out] When workspace is nullptr, this is set to the required workspace
 * size. When workspace is not nullptr, this must be the size of the workspace.
 * @param stream: the cuda stream where to launch this kernel
 */
template <typename DataT, typename LabelT>
void countLabels(const LabelT* labels,
                 DataT* binCountArray,
                 int nRows,
                 int nUniqueLabels,
                 void* workspace,
                 size_t& workspace_size,
                 cudaStream_t stream)
{
  int num_levels     = nUniqueLabels + 1;
  LabelT lower_level = 0;
  LabelT upper_level = nUniqueLabels;

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
 * @brief structure that defines the division Lambda for elementwise op
 */
template <typename DataT>
struct DivOp {
  HDI DataT operator()(DataT a, int b, int c)
  {
    if (b == 0)
      return ULLONG_MAX;
    else
      return a / b;
  }
};

/**
 * @brief structure that defines the elementwise operation to calculate silhouette score using
 * params 'a' and 'b'
 */
template <typename DataT>
struct SilOp {
  HDI DataT operator()(DataT a, DataT b)
  {
    if (a == 0 && b == 0 || a == b)
      return 0;
    else if (a == -1)
      return 0;
    else if (a > b)
      return (b - a) / a;
    else
      return (b - a) / b;
  }
};

/**
 * @brief main function that returns the average silhouette score for a given set of data and its
 * clusterings
 * @tparam DataT: type of the data samples
 * @tparam LabelT: type of the labels
 * @param X_in: pointer to the input Data samples array (nRows x nCols)
 * @param nRows: number of data samples
 * @param nCols: number of features
 * @param labels: the pointer to the array containing labels for every data sample (1 x nRows)
 * @param nLabels: number of Labels
 * @param silhouette_scorePerSample: pointer to the array that is optionally taken in as input and
 * is populated with the silhouette score for every sample (1 x nRows)
 * @param stream: the cuda stream where to launch this kernel
 * @param metric: the numerical value that maps to the type of distance metric to be used in the
 * calculations
 */
template <typename DataT, typename LabelT>
DataT silhouette_score(
  raft::resources const& handle,
  const DataT* X_in,
  int nRows,
  int nCols,
  const LabelT* labels,
  int nLabels,
  DataT* silhouette_scorePerSample,
  cudaStream_t stream,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Unexpanded)
{
  bool is_dry_run = resource::get_dry_run_flag(handle);
  ASSERT(nLabels >= 2 && nLabels <= (nRows - 1),
         "silhouette Score not defined for the given number of labels!");

  // compute the distance matrix
  rmm::device_uvector<DataT> distanceMatrix(nRows * nRows, stream);

  // Query workspace size for countLabels (can run in dry-run)
  size_t countLabels_ws_size = 0;
  countLabels<DataT, LabelT>(labels, nullptr, nRows, nLabels, nullptr, countLabels_ws_size, stream);
  rmm::device_uvector<char> workspace(countLabels_ws_size, stream);

  // deciding on the array of silhouette scores for each dataPoint
  rmm::device_uvector<DataT> silhouette_scoreSamples(
    silhouette_scorePerSample == nullptr ? nRows : 0, stream);
  DataT* perSampleSilScore = nullptr;
  if (silhouette_scorePerSample == nullptr) {
    perSampleSilScore = silhouette_scoreSamples.data();
  } else {
    perSampleSilScore = silhouette_scorePerSample;
  }

  // getting the sample count per cluster
  rmm::device_uvector<DataT> binCountArray(nLabels, stream);

  // calculating the sample-cluster-distance-sum-array
  rmm::device_uvector<DataT> sampleToClusterSumOfDistances(nRows * nLabels, stream);

  // creating the a array and b array
  rmm::device_uvector<DataT> d_aArray(nRows, stream);
  rmm::device_uvector<DataT> d_bArray(nRows, stream);

  // elementwise dividing by bincounts
  rmm::device_uvector<DataT> averageDistanceBetweenSampleAndCluster(nRows * nLabels, stream);

  // calculating the sum of all the silhouette score
  rmm::device_scalar<DataT> d_avgSilhouetteScore(stream);

  if (is_dry_run) { return DataT{0}; }

  raft::distance::pairwise_distance(
    handle, X_in, X_in, distanceMatrix.data(), nRows, nRows, nCols, metric);

  RAFT_CUDA_TRY(cudaMemsetAsync(perSampleSilScore, 0, nRows * sizeof(DataT), stream));

  RAFT_CUDA_TRY(cudaMemsetAsync(binCountArray.data(), 0, nLabels * sizeof(DataT), stream));
  size_t workspace_size = workspace.size();
  countLabels<DataT, LabelT>(
    labels, binCountArray.data(), nRows, nLabels, workspace.data(), workspace_size, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(
    sampleToClusterSumOfDistances.data(), 0, nRows * nLabels * sizeof(DataT), stream));
  raft::linalg::reduce_cols_by_key(distanceMatrix.data(),
                                   labels,
                                   sampleToClusterSumOfDistances.data(),
                                   nRows,
                                   nRows,
                                   nLabels,
                                   stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(d_aArray.data(), 0, nRows * sizeof(DataT), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(d_bArray.data(), 0, nRows * sizeof(DataT), stream));

  // kernel that populates the d_aArray
  dim3 numThreadsPerBlock(32, 1, 1);
  dim3 numBlocks(raft::ceildiv<int>(nRows, numThreadsPerBlock.x), 1, 1);

  populateAKernel<<<numBlocks, numThreadsPerBlock, 0, stream>>>(
    sampleToClusterSumOfDistances.data(),
    binCountArray.data(),
    d_aArray.data(),
    labels,
    nRows,
    nLabels,
    std::numeric_limits<DataT>::max());

  RAFT_CUDA_TRY(cudaMemsetAsync(
    averageDistanceBetweenSampleAndCluster.data(), 0, nRows * nLabels * sizeof(DataT), stream));

  raft::linalg::matrix_vector_op<raft::Apply::ALONG_ROWS>(
    handle,
    raft::make_device_matrix_view<const DataT, int, raft::row_major>(
      sampleToClusterSumOfDistances.data(), nRows, nLabels),
    raft::make_device_vector_view<const DataT, int>(binCountArray.data(), nLabels),
    raft::make_device_vector_view<const DataT, int>(binCountArray.data(), nLabels),
    raft::make_device_matrix_view<DataT, int, raft::row_major>(
      averageDistanceBetweenSampleAndCluster.data(), nRows, nLabels),
    DivOp<DataT>());

  // calculating row-wise minimum
  raft::linalg::reduce<DataT, DataT, int, raft::identity_op, raft::min_op>(
    d_bArray.data(),
    averageDistanceBetweenSampleAndCluster.data(),
    nLabels,
    nRows,
    std::numeric_limits<DataT>::max(),
    true,
    true,
    stream,
    false,
    raft::identity_op{},
    raft::min_op{});

  // calculating the silhouette score per sample using the d_aArray and d_bArray
  raft::linalg::binaryOp<DataT, SilOp<DataT>>(
    perSampleSilScore, d_aArray.data(), d_bArray.data(), nRows, SilOp<DataT>(), stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(d_avgSilhouetteScore.data(), 0, sizeof(DataT), stream));

  raft::linalg::mapThenSumReduce<double, raft::identity_op>(d_avgSilhouetteScore.data(),
                                                            nRows,
                                                            raft::identity_op(),
                                                            stream,
                                                            perSampleSilScore,
                                                            perSampleSilScore);

  DataT avgSilhouetteScore = d_avgSilhouetteScore.value(stream);

  resource::sync_stream(handle, stream);

  avgSilhouetteScore /= nRows;

  return avgSilhouetteScore;
}

};  // namespace detail
};  // namespace stats
};  // namespace raft
