/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/spectral/detail/spectral_util.cuh>
#include <raft/spectral/matrix_wrappers.hpp>

#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <math.h>
#include <stdio.h>

#include <tuple>

namespace raft {
namespace spectral {
namespace detail {

// =========================================================
// Analysis of graph partition
// =========================================================

/// Compute cost function for partition
/** This function determines the edges cut by a partition and a cost
 *  function:
 *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
 *  Graph is assumed to be weighted and undirected.
 *
 *  @param G Weighted graph in CSR format
 *  @param nClusters Number of partitions.
 *  @param clusters (Input, device memory, n entries) Partition
 *    assignments.
 *  @param edgeCut On exit, weight of edges cut by partition.
 *  @param cost On exit, partition cost function.
 *  @return error flag.
 */
template <typename vertex_t, typename weight_t, typename nnz_t>
void analyzePartition(raft::resources const& handle,
                      spectral::matrix::sparse_matrix_t<vertex_t, weight_t, nnz_t> const& csr_m,
                      vertex_t nClusters,
                      const vertex_t* __restrict__ clusters,
                      weight_t& edgeCut,
                      weight_t& cost)
{
  RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");

  vertex_t i;
  vertex_t n = csr_m.nrows_;

  weight_t partEdgesCut, clustersize;

  // Device memory - allocate before dry-run check to track allocations
  spectral::matrix::vector_t<weight_t> part_i(handle, n);
  spectral::matrix::vector_t<weight_t> Lx(handle, n);

  // Initialize Laplacian - allocate before dry-run check to track allocations
  /// sparse_matrix_t<vertex_t, weight_t> A{handle, graph};
  spectral::matrix::laplacian_matrix_t<vertex_t, weight_t, nnz_t> L{handle, csr_m};

  if (resource::get_dry_run_flag(handle)) { return; }

  auto stream   = resource::get_cuda_stream(handle);
  auto cublas_h = resource::get_cublas_handle(handle);

  // Initialize cuBLAS
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

  // Initialize output
  cost    = 0;
  edgeCut = 0;

  // Iterate through partitions
  for (i = 0; i < nClusters; ++i) {
    // Construct indicator vector for ith partition
    if (!construct_indicator(handle, i, n, clustersize, partEdgesCut, clusters, part_i, Lx, L)) {
      RAFT_LOG_WARN("empty partition");
      continue;
    }

    // Record results
    cost += partEdgesCut / clustersize;
    edgeCut += partEdgesCut / 2;
  }
}

}  // namespace detail
}  // namespace spectral
}  // namespace raft
