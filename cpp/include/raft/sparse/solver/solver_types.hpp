/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>

namespace raft::sparse::solver {

/**
 * @addtogroup sparse_randomized_svd
 * @{
 */

/**
 * @brief Configuration parameters for the sparse randomized SVD solver
 *
 * @tparam ValueTypeT Data type for values (float or double)
 */
template <typename ValueTypeT>
struct sparse_svd_config {
  /** @brief Number of singular values/vectors to compute. Must be set by the user. */
  int n_components = 0;

  /** @brief Number of extra random vectors for better approximation.
   *  Total subspace dimension is n_components + n_oversamples. */
  int n_oversamples = 10;

  /** @brief Number of power iteration passes. More iterations improve accuracy
   *  for matrices with slowly decaying singular values. */
  int n_power_iters = 2;

  /** @brief Random seed for reproducibility */
  std::optional<uint64_t> seed = std::nullopt;
};

/** @} */

/**
 * @addtogroup sparse_lanczos_svd
 * @{
 */

/**
 * @brief Configuration parameters for the sparse Lanczos SVD solver
 *
 * @tparam ValueTypeT Data type for values (float or double)
 */
template <typename ValueTypeT>
struct sparse_lanczos_svd_config {
  /** @brief Number of singular values/vectors to compute. Must be set by the user.
   *  @note Must satisfy 0 < n_components < min(m, n), where (m, n) is the matrix shape. */
  int n_components = 0;

  /**
   * @brief Number of Lanczos vectors per restart.
   *
   * If zero, a matrix-shape dependent default is selected. Larger values can improve
   * convergence margin and orthogonality for clustered spectra, but increase sparse
   * matrix-vector work and memory use.
   *
   * @note The value is clamped to [n_components + 10, min(m, n) - 1]. The extra
   *       subspace slack beyond n_components is required for reliable convergence,
   *       in particular to resolve repeated or tightly clustered singular values.
   */
  int ncv = 0;

  /** @brief Convergence tolerance for Lanczos Ritz residual estimates. */
  ValueTypeT tolerance = ValueTypeT(1e-4);

  /** @brief Maximum number of restart iterations before reporting non-convergence. */
  int max_iterations = 100;

  /** @brief Random seed for reproducibility. */
  std::optional<uint64_t> seed = std::nullopt;

  /**
   * @brief Use launch-heavy MGS2 instead of the default GPU-efficient CGS2 reorthogonalization.
   *
   * MGS2 is kept as an alternate path for difficult spectra; CGS2 is the default used in
   * normal GPU workloads.
   */
  bool use_mgs2_orthogonalization = false;
};

/**
 * @brief Optional per-call diagnostics for the sparse Lanczos SVD solver.
 *
 * Purely host-side counters, filled in as the restart loop runs. Pass a pointer to one of
 * these to `sparse_lanczos_svd` to observe the restart path; pass `nullptr` (the default)
 * to skip collection entirely. Collecting these adds no device work and no extra
 * synchronization.
 */
struct sparse_lanczos_svd_stats {
  /**
   * @brief Number of restarts performed.
   *
   * A restart is a sweep after which a new starting vector was built and another
   * bidiagonalization followed, i.e. it excludes the final sweep that completes
   * `n_components` and exits the loop.
   */
  int n_restarts = 0;

  /**
   * @brief Largest number of Ritz pairs locked in a single sweep that was followed by a
   *        restart.
   *
   * This is the `num_found` (`d`) of the restart path. Values >= 2 mean the restart window
   * `V_full[:, n_locked : n_locked + active_ncv]` extended past the bidiagonalization write
   * frontier, which is the multi-vector locking case. Sweeps that lock the final components
   * and exit are deliberately not counted, because no restart vector is built from them.
   */
  int max_locked_per_restart = 0;

  /** @brief Number of bidiagonalization sweeps executed, including the final one. */
  int total_iterations = 0;

  /**
   * @brief Number of times a candidate Lanczos vector fell below the breakdown threshold
   *        and was replaced by a random vector.
   */
  int breakdown_events = 0;
};

/** @} */

}  // namespace raft::sparse::solver
