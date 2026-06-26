/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
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
   * @note When nonzero, the value is clamped to [n_components, min(m, n) - 1].
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

/** @} */

}  // namespace raft::sparse::solver
