/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "permute.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_device.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <vector>

namespace raft {
namespace random {

namespace detail {

// generate the labels first and shuffle them instead of shuffling the dataset
template <typename IdxT>
void generate_labels(IdxT* labels,
                     IdxT n_rows,
                     IdxT n_clusters,
                     bool shuffle,
                     raft::random::RngState& r,
                     cudaStream_t stream)
{
  raft::handle_t handle(stream);
  IdxT a, b;
  raft::random::affine_transform_params(r, n_clusters, a, b);
  auto op = [=] __device__(IdxT idx) {
    if (shuffle) { idx = static_cast<IdxT>((a * int64_t(idx)) + b); }
    idx %= n_clusters;
    return idx;
  };
  auto labels_view = raft::make_device_vector_view<IdxT, IdxT>(labels, n_rows);
  linalg::map_offset(handle, labels_view, op);
}

template <typename DataT, typename IdxT>
DI void get_mu_sigma(DataT& mu,
                     DataT& sigma,
                     IdxT idx,
                     const IdxT* labels,
                     bool row_major,
                     const DataT* centers,
                     const DataT* cluster_std,
                     DataT cluster_std_scalar,
                     IdxT n_rows,
                     IdxT n_cols,
                     IdxT n_clusters)
{
  IdxT cid, fid;
  if (row_major) {
    cid = idx / n_cols;
    fid = idx % n_cols;
  } else {
    cid = idx % n_rows;
    fid = idx / n_rows;
  }
  IdxT center_id;
  if (cid < n_rows) {
    center_id = labels[cid];
  } else {
    center_id = 0;
  }

  if (fid >= n_cols) { fid = 0; }

  if (row_major) {
    center_id = center_id * n_cols + fid;
  } else {
    center_id += fid * n_clusters;
  }
  sigma = cluster_std == nullptr ? cluster_std_scalar : cluster_std[cid];
  mu    = centers[center_id];
}

template <typename DataT, typename IdxT, typename GenType>
RAFT_KERNEL generate_data_kernel(raft::random::DeviceState<GenType> rng_state,
                                 DataT* out,
                                 const IdxT* labels,
                                 IdxT n_rows,
                                 IdxT n_cols,
                                 IdxT n_clusters,
                                 bool row_major,
                                 const DataT* centers,
                                 const DataT* cluster_std,
                                 const DataT cluster_std_scalar)
{
  uint64_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  GenType gen(rng_state, tid);
  const IdxT stride = gridDim.x * blockDim.x;
  IdxT len          = n_rows * n_cols;
  for (IdxT idx = tid; idx < len; idx += stride) {
    DataT val1, val2;
    do {
      gen.next(val1);
    } while (val1 == DataT(0.0));
    gen.next(val2);
    DataT mu1, sigma1, mu2, sigma2;
    get_mu_sigma(mu1,
                 sigma1,
                 idx,
                 labels,
                 row_major,
                 centers,
                 cluster_std,
                 cluster_std_scalar,
                 n_rows,
                 n_cols,
                 n_clusters);
    get_mu_sigma(mu2,
                 sigma2,
                 idx + stride,
                 labels,
                 row_major,
                 centers,
                 cluster_std,
                 cluster_std_scalar,
                 n_rows,
                 n_cols,
                 n_clusters);
    raft::random::box_muller_transform<DataT>(val1, val2, sigma1, mu1, sigma2, mu2);

    if (idx < len) out[idx] = val1;
    idx += stride;
    if (idx < len) out[idx] = val2;
  }
}

template <typename DataT, typename IdxT>
void generate_data(DataT* out,
                   const IdxT* labels,
                   IdxT n_rows,
                   IdxT n_cols,
                   IdxT n_clusters,
                   cudaStream_t stream,
                   bool row_major,
                   const DataT* centers,
                   const DataT* cluster_std,
                   const DataT cluster_std_scalar,
                   raft::random::RngState& rng_state)
{
  constexpr IdxT block_size = 128;
  int64_t items             = static_cast<int64_t>(n_rows) * n_cols;
  // Choose a grid size so that each thread can write two output values.
  int64_t nBlocks = ceildiv<int64_t>(items, 2 * block_size);
  // parentheses needed here for kernel, otherwise macro interprets the arguments
  // of triple chevron notation as macro arguments
  RAFT_CALL_RNG_FUNC(rng_state,
                     (generate_data_kernel<<<nBlocks, 128, 0, stream>>>),
                     out,
                     labels,
                     n_rows,
                     n_cols,
                     n_clusters,
                     row_major,
                     centers,
                     cluster_std,
                     cluster_std_scalar);
}

/**
 * @brief GPU-equivalent of sklearn.datasets.make_blobs
 *
 * @tparam DataT output data type
 * @tparam IdxT  indexing arithmetic type
 *
 * @param[out] out                generated data [on device]
 *                                [dim = n_rows x n_cols]
 * @param[out] labels             labels for the generated data [on device]
 *                                [len = n_rows]
 * @param[in]  n_rows             number of rows in the generated data
 * @param[in]  n_cols             number of columns in the generated data
 * @param[in]  n_clusters         number of clusters (or classes) to generate
 * @param[in]  stream             cuda stream to schedule the work on
 * @param[in]  row_major          whether input `centers` and output `out`
 *                                buffers are to be stored in row or column
 *                                major layout
 * @param[in]  centers            centers of each of the cluster, pass a nullptr
 *                                if you need this also to be generated randomly
 *                                [on device] [dim = n_clusters x n_cols]
 * @param[in]  cluster_std        standard deviation of each cluster center,
 *                                pass a nullptr if this is to be read from the
 *                                `cluster_std_scalar`. [on device]
 *                                [len = n_clusters]
 * @param[in]  cluster_std_scalar if 'cluster_std' is nullptr, then use this as
 *                                the std-dev across all dimensions.
 * @param[in]  shuffle            shuffle the generated dataset and labels
 * @param[in]  center_box_min     min value of box from which to pick cluster
 *                                centers. Useful only if 'centers' is nullptr
 * @param[in]  center_box_max     max value of box from which to pick cluster
 *                                centers. Useful only if 'centers' is nullptr
 * @param[in]  seed               seed for the RNG
 * @param[in]  type               RNG type
 */
template <typename DataT, typename IdxT>
void make_blobs_caller(bool dry_run,
                       DataT* out,
                       IdxT* labels,
                       IdxT n_rows,
                       IdxT n_cols,
                       IdxT n_clusters,
                       cudaStream_t stream,
                       bool row_major,
                       const DataT* centers,
                       const DataT* cluster_std,
                       const DataT cluster_std_scalar,
                       bool shuffle,
                       DataT center_box_min,
                       DataT center_box_max,
                       uint64_t seed,
                       raft::random::GeneratorType type)
{
  raft::random::RngState r(seed, type);
  // use the right centers buffer for data generation
  rmm::device_uvector<DataT> rand_centers(centers == nullptr ? n_clusters * n_cols : 0, stream);
  const DataT* _centers;
  if (centers == nullptr) {
    _centers = rand_centers.data();
  } else {
    _centers = centers;
  }
  if (dry_run) { return; }
  if (centers == nullptr) {
    detail::uniform(
      r, rand_centers.data(), n_clusters * n_cols, center_box_min, center_box_max, stream);
  }
  generate_labels(labels, n_rows, n_clusters, shuffle, r, stream);
  generate_data(out,
                labels,
                n_rows,
                n_cols,
                n_clusters,
                stream,
                row_major,
                _centers,
                cluster_std,
                cluster_std_scalar,
                r);
}

}  // end namespace detail
}  // end namespace random
}  // namespace raft
