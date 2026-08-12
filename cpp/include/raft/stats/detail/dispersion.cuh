/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/interruptible.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>

#include <memory>

namespace raft {
namespace stats {
namespace detail {

///@todo: ColsPerBlk has been tested only for 32!
template <typename DataT, typename IdxT, int TPB, int ColsPerBlk = 32>
RAFT_KERNEL weightedMeanKernel(DataT* mu, const DataT* data, const IdxT* counts, IdxT D, IdxT N)
{
  constexpr int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxT thisColId                  = threadIdx.x % ColsPerBlk;
  IdxT thisRowId                  = threadIdx.x / ColsPerBlk;
  IdxT colId                      = thisColId + ((IdxT)blockIdx.y * ColsPerBlk);
  IdxT rowId                      = thisRowId + ((IdxT)blockIdx.x * RowsPerBlkPerIter);
  DataT thread_data               = DataT(0);
  const IdxT stride               = RowsPerBlkPerIter * gridDim.x;
  __shared__ DataT smu[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk) smu[threadIdx.x] = DataT(0);
  for (IdxT i = rowId; i < N; i += stride) {
    thread_data += (colId < D) ? data[i * D + colId] * (DataT)counts[i] : DataT(0);
  }
  __syncthreads();
  raft::myAtomicAdd(smu + thisColId, thread_data);
  __syncthreads();
  if (threadIdx.x < ColsPerBlk && colId < D) raft::myAtomicAdd(mu + colId, smu[thisColId]);
}

template <typename DataT, typename IdxT, int TPB>
RAFT_KERNEL dispersionKernel(DataT* result,
                             const DataT* clusters,
                             const IdxT* clusterSizes,
                             const DataT* mu,
                             IdxT dim,
                             IdxT nClusters)
{
  IdxT tid    = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT len    = dim * nClusters;
  IdxT stride = blockDim.x * gridDim.x;
  DataT sum   = DataT(0);
  for (; tid < len; tid += stride) {
    IdxT col   = tid % dim;
    IdxT row   = tid / dim;
    DataT diff = clusters[tid] - mu[col];
    sum += diff * diff * DataT(clusterSizes[row]);
  }
  typedef cub::BlockReduce<DataT, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __syncthreads();
  auto acc = BlockReduce(temp_storage).Sum(sum);
  __syncthreads();
  if (threadIdx.x == 0) raft::myAtomicAdd(result, acc);
}

/**
 * @brief Compute cluster dispersion metric. This is very useful for
 * automatically finding the 'k' (in kmeans) that improves this metric.
 * @tparam DataT data type
 * @tparam IdxT index type
 * @tparam TPB threads block for kernels launched
 * @param centroids the cluster centroids. This is assumed to be row-major
 *   and of dimension (nClusters x dim)
 * @param clusterSizes number of points in the dataset which belong to each
 *   cluster. This is of length nClusters
 * @param globalCentroid compute the global weighted centroid of all cluster
 *   centroids. This is of length dim. Pass a nullptr if this is not needed
 * @param nClusters number of clusters
 * @param nPoints number of points in the dataset
 * @param dim dataset dimensionality
 * @param stream cuda stream
 * @return the cluster dispersion value
 */
template <typename DataT, typename IdxT = int, int TPB = 256>
DataT dispersion(const DataT* centroids,
                 const IdxT* clusterSizes,
                 DataT* globalCentroid,
                 IdxT nClusters,
                 IdxT nPoints,
                 IdxT dim,
                 cudaStream_t stream)
{
  static const int RowsPerThread = 4;
  static const int ColsPerBlk    = 32;
  static const int RowsPerBlk    = (TPB / ColsPerBlk) * RowsPerThread;
  dim3 grid(raft::ceildiv(nPoints, (IdxT)RowsPerBlk), raft::ceildiv(dim, (IdxT)ColsPerBlk));
  rmm::device_uvector<DataT> mean(0, stream);
  rmm::device_uvector<DataT> result(1, stream);
  DataT* mu = globalCentroid;
  if (globalCentroid == nullptr) {
    mean.resize(dim, stream);
    mu = mean.data();
  }
  RAFT_CUDA_TRY(cudaMemsetAsync(mu, 0, sizeof(DataT) * dim, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(result.data(), 0, sizeof(DataT), stream));
  raft::launch_kernel(stream,
                      grid,
                      TPB,
                      weightedMeanKernel<DataT, IdxT, TPB, ColsPerBlk>,
                      mu,
                      centroids,
                      clusterSizes,
                      dim,
                      nClusters);
  DataT ratio = DataT(1) / DataT(nPoints);
  raft::linalg::scalarMultiply(mu, mu, ratio, dim, stream);
  // finally, compute the dispersion
  constexpr int ItemsPerThread = 4;
  int nblks                    = raft::ceildiv<int>(dim * nClusters, TPB * ItemsPerThread);
  raft::launch_kernel(stream,
                      nblks,
                      TPB,
                      dispersionKernel<DataT, IdxT, TPB>,
                      result.data(),
                      centroids,
                      clusterSizes,
                      mu,
                      dim,
                      nClusters);
  DataT h_result;
  raft::update_host(&h_result, result.data(), 1, stream);
  raft::interruptible::synchronize(stream);
  return sqrt(h_result);
}

}  // end namespace detail
}  // end namespace stats
}  // namespace raft
