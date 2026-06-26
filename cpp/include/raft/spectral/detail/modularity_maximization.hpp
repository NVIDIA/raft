/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/logger_macros.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/normalize.cuh>
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

//===================================================
// Analysis of graph partition
// =========================================================

/// Compute modularity
/** This function determines the modularity based on a graph and cluster assignments
 *  @param G Weighted graph in CSR format
 *  @param nClusters Number of clusters.
 *  @param clusters (Input, device memory, n entries) Cluster assignments.
 *  @param modularity On exit, modularity
 */
template <typename vertex_t, typename weight_t, typename nnz_t>
void analyzeModularity(
  raft::resources const& handle,
  raft::spectral::matrix::sparse_matrix_t<vertex_t, weight_t, nnz_t> const& csr_m,
  vertex_t nClusters,
  vertex_t const* __restrict__ clusters,
  weight_t& modularity)
{
  bool is_dry_run = resource::get_dry_run_flag(handle);
  RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");

  vertex_t i;
  vertex_t n = csr_m.nrows_;
  weight_t partModularity, clustersize;

  auto cublas_h = resource::get_cublas_handle(handle);
  auto stream   = resource::get_cuda_stream(handle);

  // Device memory
  raft::spectral::matrix::vector_t<weight_t> part_i(handle, n);
  raft::spectral::matrix::vector_t<weight_t> Bx(handle, n);

  // Initialize cuBLAS
  if (!is_dry_run) {
    RAFT_CUBLAS_TRY(
      linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));
  }

  // Initialize Modularity
  raft::spectral::matrix::modularity_matrix_t<vertex_t, weight_t, nnz_t> B{handle, csr_m};

  if (is_dry_run) {
    // Early stopping because construct_indicator doesn't allocate anything.
    return;
  }

  // Initialize output
  modularity = 0;

  // Iterate through partitions
  for (i = 0; i < nClusters; ++i) {
    if (!construct_indicator(handle, i, n, clustersize, partModularity, clusters, part_i, Bx, B)) {
      RAFT_LOG_WARN("empty partition");
      continue;
    }

    // Record results
    modularity += partModularity;
  }

  modularity = modularity / B.diagonal_.nrm1();
}

}  // namespace detail
}  // namespace spectral
}  // namespace raft
