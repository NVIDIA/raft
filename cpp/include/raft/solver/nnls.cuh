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
  /** Optimality tolerance on the projected gradient w = A^T (b - A x).  In both precisions
   *  a column enters the active set only if w_j exceeds `tol` plus an allowance for the
   *  rounding error of w_j, about sqrt(np + 1) * eps * sum_k |G_jk x_k| (np: active-set size,
   *  G = A^T A).  That allowance scales with the magnitude of the data: for large-scale
   *  inputs, or for float32 in general, it can exceed `tol`, which then is not the
   *  effective stopping threshold.  Normalize A and B if an absolute `tol` matters. */
  double tol = 1e-6;
  /** Cap on the outer (column-activation) iterations per problem; <= 0 selects 3 * n + 1. */
  int max_iter = 0;
};

/**
 * @defgroup nnls Batched non-negative least squares
 * @{
 */

/**
 * @brief Solve a batch of non-negative least-squares problems that share one matrix.
 *
 * For every problem p, solves `min_{x >= 0} ||A x - b_p||_2` with the Lawson-Hanson
 * active-set method, one problem per CUDA block.  The Gram matrix A^T A and the
 * projections A^T B are formed once with GEMM and shared by the whole batch; the active
 * set is kept linearly independent, so it never exceeds min(m, n) columns.
 *
 * The shared memory of a block is O(min(m, n)); a hard `raft::logic_error` is raised
 * before any work is done when min(m, n) exceeds what the device allows (about 8e3 in
 * float32 and 5e3 in float64 on sm_90).  A numerically dependent column ends the solve of
 * its problem with the active set found so far.  Exact ties between columns may resolve
 * differently on different block sizes; the objective is the same.
 *
 * Dry-run compliant: allocations are sized from the arguments only.
 *
 * The outputs `X` and `fitted` must not overlap each other or any input (`A`, `B`,
 * `masks`): `X` is written while `A` is still needed to form `fitted`.
 *
 * @tparam T float or double
 * @param[in]  res     RAFT resources
 * @param[in]  params  tolerance and iteration cap
 * @param[in]  A       (m x n) column-major matrix
 * @param[in]  B       (m x P) column-major right-hand sides, one problem per column
 * @param[out] X       (n x P) column-major solutions
 * @param[in]  masks   optional (n x P) column-major support: problem p may only use the
 *                     columns j of A with masks(j, p) != 0 (the others are 0 in X)
 * @param[out] fitted  optional (m x P) column-major A X
 */
template <typename T>
void nnls(
  raft::resources const& res,
  nnls_params const& params,
  raft::device_matrix_view<const T, std::int64_t, raft::col_major> A,
  raft::device_matrix_view<const T, std::int64_t, raft::col_major> B,
  raft::device_matrix_view<T, std::int64_t, raft::col_major> X,
  std::optional<raft::device_matrix_view<const std::uint8_t, std::int64_t, raft::col_major>> masks =
    std::nullopt,
  std::optional<raft::device_matrix_view<T, std::int64_t, raft::col_major>> fitted = std::nullopt)
{
  detail::nnls(res, params.tol, params.max_iter, A, B, X, masks, fitted);
}

/** @} */

}  // namespace raft::solver
