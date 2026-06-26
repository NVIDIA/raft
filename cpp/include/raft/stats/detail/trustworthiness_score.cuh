/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/distance/distance.cuh>
#include <raft/matrix/col_wise_sort.cuh>
#include <raft/spatial/knn/knn.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#define N_THREADS 512

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Build the lookup table
 * @param[out] lookup_table: Lookup table giving nearest neighbor order
 *                of pairwise distance calculations given sample index
 * @param[in] X_ind: Sorted indexes of pairwise distance calculations of X
 * @param n: Number of samples
 * @param work: Number of elements to consider
 */
RAFT_KERNEL build_lookup_table(int* lookup_table, const int* X_ind, int n, int work)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= work) return;

  int sample_idx = i / n;
  int nn_idx     = i % n;

  int idx                              = X_ind[i];
  lookup_table[(sample_idx * n) + idx] = nn_idx;
}

/**
 * @brief Compute a the rank of trustworthiness score
 * @param[out] rank: Resulting rank
 * @param[out] lookup_table: Lookup table giving nearest neighbor order
 *                of pairwise distance calculations given sample index
 * @param[in] emb_ind: Indexes of KNN on embeddings
 * @param n: Number of samples
 * @param n_neighbors: Number of neighbors considered by trustworthiness score
 * @param work: Batch to consider (to do it at once use n * n_neighbors)
 */
template <typename knn_index_t>
RAFT_KERNEL compute_rank(double* rank,
                         const int* lookup_table,
                         const knn_index_t* emb_ind,
                         int n,
                         int n_neighbors,
                         int work)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= work) return;

  int sample_idx = i / n_neighbors;

  knn_index_t emb_nn_ind = emb_ind[i];

  int r   = lookup_table[(sample_idx * n) + emb_nn_ind];
  int tmp = r - n_neighbors + 1;
  if (tmp > 0) raft::myAtomicAdd<double>(rank, tmp);
}

/**
 * @brief Compute a kNN and returns the indices of the nearest neighbors
 * @param h Raft handle
 * @param[in] input Input matrix containing the dataset
 * @param n Number of samples
 * @param d Number of features
 * @param n_neighbors number of neighbors
 * @param[out] indices KNN indexes
 * @param[out] distances KNN distances
 */
template <raft::distance::DistanceType distance_type, typename math_t>
void run_knn(const raft::resources& h,
             math_t* input,
             int n,
             int d,
             int n_neighbors,
             int64_t* indices,
             math_t* distances)
{
  std::vector<math_t*> ptrs(1);
  std::vector<int> sizes(1);
  ptrs[0]  = input;
  sizes[0] = n;

  raft::spatial::knn::brute_force_knn<int64_t, float, int>(h,
                                                           ptrs,
                                                           sizes,
                                                           d,
                                                           input,
                                                           n,
                                                           indices,
                                                           distances,
                                                           n_neighbors,
                                                           true,
                                                           true,
                                                           nullptr,
                                                           distance_type);
}

/**
 * @brief Compute the trustworthiness score
 * @param h Raft handle
 * @param X[in]: Data in original dimension
 * @param X_embedded[in]: Data in target dimension (embedding)
 * @param n: Number of samples
 * @param m: Number of features in high/original dimension
 * @param d: Number of features in low/embedded dimension
 * @param n_neighbors Number of neighbors considered by trustworthiness score
 * @param batchSize Batch size
 * @return Trustworthiness score
 */
template <typename math_t, raft::distance::DistanceType distance_type>
double trustworthiness_score(const raft::resources& h,
                             const math_t* X,
                             math_t* X_embedded,
                             int n,
                             int m,
                             int d,
                             int n_neighbors,
                             int batchSize = 512)
{
  cudaStream_t stream = resource::get_cuda_stream(h);

  const int KNN_ALLOC = n * (n_neighbors + 1);
  rmm::device_uvector<int64_t> emb_ind(KNN_ALLOC, stream);
  rmm::device_uvector<math_t> emb_dist(KNN_ALLOC, stream);

  run_knn<distance_type>(h, X_embedded, n, d, n_neighbors + 1, emb_ind.data(), emb_dist.data());

  const int PAIRWISE_ALLOC = batchSize * n;
  rmm::device_uvector<int> X_ind(PAIRWISE_ALLOC, stream);
  rmm::device_uvector<math_t> X_dist(PAIRWISE_ALLOC, stream);
  rmm::device_uvector<int> lookup_table(PAIRWISE_ALLOC, stream);

  double t = 0.0;
  rmm::device_scalar<double> t_dbuf(stream);

  int toDo = n;
  while (toDo > 0) {
    int curBatchSize = min(toDo, batchSize);

    // Takes at most batchSize vectors at a time
    raft::distance::pairwise_distance(
      h, &X[(n - toDo) * m], X, X_dist.data(), curBatchSize, n, m, distance_type);

    // Use dry-run compliant handle-based overload that manages workspace internally
    auto dist_view = raft::make_device_matrix_view<const math_t, int, raft::row_major>(
      X_dist.data(), curBatchSize, n);
    auto ind_view =
      raft::make_device_matrix_view<int, int, raft::row_major>(X_ind.data(), curBatchSize, n);
    raft::matrix::sort_cols_per_row(h, dist_view, ind_view, std::nullopt);

    // The workspace won't grow anymore
    if (resource::get_dry_run_flag(h)) { return 0.0; }

    int work     = curBatchSize * n;
    int n_blocks = raft::ceildiv(work, N_THREADS);
    build_lookup_table<<<n_blocks, N_THREADS, 0, stream>>>(
      lookup_table.data(), X_ind.data(), n, work);

    RAFT_CUDA_TRY(cudaMemsetAsync(t_dbuf.data(), 0, sizeof(double), stream));

    work     = curBatchSize * (n_neighbors + 1);
    n_blocks = raft::ceildiv(work, N_THREADS);
    compute_rank<<<n_blocks, N_THREADS, 0, stream>>>(
      t_dbuf.data(),
      lookup_table.data(),
      &emb_ind.data()[(n - toDo) * (n_neighbors + 1)],
      n,
      n_neighbors + 1,
      work);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    t += t_dbuf.value(stream);

    toDo -= curBatchSize;
  }

  t = 1.0 - ((2.0 / ((n * n_neighbors) * ((2.0 * n) - (3.0 * n_neighbors) - 1.0))) * t);

  return t;
}

}  // namespace detail
}  // namespace stats
}  // namespace raft
