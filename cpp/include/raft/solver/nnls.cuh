/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/solver/detail/nnls.cuh>

#include <cstdint>
#include <optional>

namespace raft::solver {

/** Parameters of raft::solver::nnls. */
struct nnls_params {
  /** Optimality (KKT) tolerance on the projected gradient A^T (b - A x).  The solver also
   *  allows for the rounding error of that gradient, which grows with the magnitude of the
   *  data; for large-scale inputs, and for float32 in general, it can exceed `tol`.
   *  Normalize A and B if an absolute tolerance matters. */
  double tol = 1e-6;
  /** Cap on the number of iterations per problem; <= 0 selects a default that scales with n. */
  int max_iter = 0;
};

/**
 * @defgroup nnls Batched non-negative least squares
 * @{
 */

/**
 * @brief Solve a batch of non-negative least-squares problems that share one matrix.
 *
 * For every problem p, solves `min_{x >= 0} ||A x - b_p||_2`, where the columns b_p of B
 * are the right-hand sides.  Optionally, each problem can be restricted to a subset of the
 * columns of A.
 *
 * Problems whose min(m, n) is too large for the device raise a `raft::logic_error` before
 * any work is done.
 *
 * The outputs `X` and `rnorm` must not overlap each other or any input (`A`, `B`, `masks`).
 *
 * @tparam T float or double
 * @param[in]  res     RAFT resources
 * @param[in]  params  tolerance and iteration cap
 * @param[in]  A       (m x n) column-major matrix
 * @param[in]  B       (m x P) column-major right-hand sides, one problem per column
 * @param[out] X       (n x P) column-major solutions
 * @param[in]  masks   optional (n x P) column-major support: problem p may only use the
 *                     columns j of A with masks(j, p) != 0 (the others are 0 in X)
 * @param[out] rnorm   optional length-P residual norms: `rnorm(p) = ||b_p - A x_p||_2`
 */
template <typename T>
void nnls(raft::resources const& res,
          nnls_params const& params,
          raft::device_matrix_view<const T, std::int64_t, raft::col_major> A,
          raft::device_matrix_view<const T, std::int64_t, raft::col_major> B,
          raft::device_matrix_view<T, std::int64_t, raft::col_major> X,
          std::optional<raft::device_matrix_view<const std::uint8_t, std::int64_t, raft::col_major>>
            masks                                                        = std::nullopt,
          std::optional<raft::device_vector_view<T, std::int64_t>> rnorm = std::nullopt)
{
  detail::nnls(res, params.tol, params.max_iter, A, B, X, masks, rnorm);
}

/** @} */

}  // namespace raft::solver
