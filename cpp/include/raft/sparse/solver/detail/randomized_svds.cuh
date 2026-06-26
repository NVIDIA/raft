/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/solver/detail/cholesky_qr.cuh>
#include <raft/sparse/solver/detail/svds_sign_correction.cuh>
#include <raft/sparse/solver/svds_config.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <optional>
#include <random>

namespace raft::sparse::solver::detail {

/**
 * @brief Randomized SVD for sparse matrices using block power iteration with CholeskyQR2.
 *
 * Implements randomized SVD (Halko et al. 2009) with GPU-optimized CholeskyQR2
 * orthogonalization (Tomás et al. 2024).
 *
 * The operator interface allows implicit operators (e.g. mean-centered sparse matrices)
 * without materializing the dense matrix.
 *
 * @tparam ValueTypeT Data type (float or double)
 * @tparam OperatorT Linear operator type providing apply() and apply_transpose()
 *
 * @param[in] handle raft resources handle
 * @param[in] config SVD configuration (n_components, n_oversamples, n_power_iters, seed)
 * @param[in] op linear operator representing the matrix to decompose
 * @param[out] singular_values output singular values of shape (k,) in descending order
 * @param[out] U optional output left singular vectors of shape (m, k), col-major
 * @param[out] Vt optional output right singular vectors of shape (k, n), col-major
 */
template <typename ValueTypeT, typename OperatorT>
void sparse_randomized_svd(
  raft::resources const& handle,
  sparse_svd_config<ValueTypeT> const& config,
  OperatorT const& op,
  raft::device_vector_view<ValueTypeT, uint32_t> singular_values,
  std::optional<raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> U,
  std::optional<raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> Vt)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "raft::sparse::solver::sparse_randomized_svd(%d, %d, %d)",
    op.rows(),
    op.cols(),
    config.n_components);

  int m = op.rows();
  int n = op.cols();
  int k = config.n_components;
  int p = config.n_oversamples;

  RAFT_EXPECTS(k > 0, "n_components must be positive");
  RAFT_EXPECTS(k < std::min(m, n), "n_components must be less than min(m, n)");
  RAFT_EXPECTS(p >= 0, "n_oversamples must be non-negative");
  RAFT_EXPECTS(config.n_power_iters >= 0, "n_power_iters must be non-negative");
  RAFT_EXPECTS(singular_values.extent(0) == static_cast<uint32_t>(k),
               "singular_values must have size n_components");
  if (U) {
    RAFT_EXPECTS(
      U->extent(0) == static_cast<uint32_t>(m) && U->extent(1) == static_cast<uint32_t>(k),
      "U must have shape (m, n_components)");
  }
  if (Vt) {
    RAFT_EXPECTS(
      Vt->extent(0) == static_cast<uint32_t>(k) && Vt->extent(1) == static_cast<uint32_t>(n),
      "Vt must have shape (n_components, n)");
  }

  auto stream = raft::resource::get_cuda_stream(handle);

  int min_dim = std::min(m, n);
  if (k + p > min_dim) {
    RAFT_LOG_WARN(
      "n_components (%d) + n_oversamples (%d) = %d exceeds min(n_rows, n_cols) = %d. "
      "Clamping to %d. This may affect approximation quality.",
      k,
      p,
      k + p,
      min_dim,
      min_dim);
  }
  int block_size = std::min(k + p, min_dim);
  RAFT_EXPECTS(block_size >= k,
               "block_size (n_components + n_oversamples) must be >= n_components");

  // Initialize RNG
  uint64_t seed = config.seed.value_or(std::random_device{}());
  raft::random::RngState rng_state(seed);

  // Step 1-3: Y = A @ Omega, orthogonalize
  auto Y = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
    handle, static_cast<uint32_t>(m), static_cast<uint32_t>(block_size));
  {
    auto Omega = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
      handle, static_cast<uint32_t>(n), static_cast<uint32_t>(block_size));
    raft::random::normal(handle,
                         rng_state,
                         Omega.data_handle(),
                         static_cast<std::size_t>(n) * block_size,
                         ValueTypeT(0),
                         ValueTypeT(1));
    op.apply(handle,
             raft::make_device_matrix_view<const ValueTypeT, uint32_t, raft::col_major>(
               Omega.data_handle(), n, block_size),
             Y.view());
  }  // Omega freed here
  if (!cholesky_qr2(handle, Y.view())) {
    RAFT_LOG_WARN("CholeskyQR2 fell back to standard QR during initial orthogonalization");
  }

  // Step 4: Power iterations
  auto Z = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
    handle, static_cast<uint32_t>(n), static_cast<uint32_t>(block_size));

  for (int iter = 0; iter < config.n_power_iters; ++iter) {
    // Z = A^T @ Q  -> (n, block_size)
    op.apply_transpose(handle,
                       raft::make_device_matrix_view<const ValueTypeT, uint32_t, raft::col_major>(
                         Y.data_handle(), m, block_size),
                       Z.view());
    if (!cholesky_qr2(handle, Z.view())) {
      RAFT_LOG_WARN(
        "CholeskyQR2 fell back to standard QR during power iteration %d (transpose step)", iter);
    }

    // Y = A @ Z  -> (m, block_size)
    op.apply(handle,
             raft::make_device_matrix_view<const ValueTypeT, uint32_t, raft::col_major>(
               Z.data_handle(), n, block_size),
             Y.view());
    if (!cholesky_qr2(handle, Y.view())) {
      RAFT_LOG_WARN("CholeskyQR2 fell back to standard QR during power iteration %d (forward step)",
                    iter);
    }
  }

  // Q = Y after power iterations (already orthogonal)
  // Q is (m, block_size)

  // Step 5: Bt = A^T @ Q  -> (n, block_size)
  op.apply_transpose(handle,
                     raft::make_device_matrix_view<const ValueTypeT, uint32_t, raft::col_major>(
                       Y.data_handle(), m, block_size),
                     Z.view());

  // Step 6-7: SVD of B = Bt^T where Bt = Z (n x block_size, tall matrix)
  // We compute SVD(Bt) directly to avoid cuSOLVER gesvd issues with wide matrices.
  // SVD(Bt) = U_bt * S * Vt_bt → SVD(B) has U_b = V_bt and Vt_b = U_bt^T
  auto S_full = raft::make_device_vector<ValueTypeT, uint32_t>(handle, block_size);

  // Bt = B^T is (n x block_size), we already have Bt = Z from step 5
  // Z is (n, block_size) = A^T @ Q = B^T. We can reuse Z directly!
  // SVD(Bt) with Bt being (n x block_size): jobu='S' gives U_bt (n x block_size),
  // jobvt='A' gives Vt_bt (block_size x block_size). Both are needed when either
  // U or Vt is requested by the caller; we only allocate the ones we will read.
  std::optional<raft::device_matrix<ValueTypeT, uint32_t, raft::col_major>> U_bt;
  std::optional<raft::device_matrix<ValueTypeT, uint32_t, raft::col_major>> Vt_bt;
  if (Vt) {
    U_bt.emplace(raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
      handle, static_cast<uint32_t>(n), static_cast<uint32_t>(block_size)));
  }
  if (U) {
    Vt_bt.emplace(raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
      handle, static_cast<uint32_t>(block_size), static_cast<uint32_t>(block_size)));
  }

  // Z is consumed by svdQR (modifies input in-place) — this is fine since Z is not used after.
  // Only generate the singular-vector side we will actually consume: U_bt feeds Vt
  // (step 9), Vt_bt (after trans_right -> V_bt) feeds the GEMM that builds U (step 8).
  raft::linalg::svdQR(handle,
                      Z.data_handle(),
                      n,
                      block_size,
                      S_full.data_handle(),
                      U_bt ? U_bt->data_handle() : nullptr,
                      Vt_bt ? Vt_bt->data_handle() : nullptr,
                      true,                   // transpose right vectors: Vt_bt -> V_bt
                      static_cast<bool>(Vt),  // gen_left_vec: U_bt needed iff Vt requested
                      static_cast<bool>(U),   // gen_right_vec: V_bt needed iff U requested
                      stream);
  // After svdQR with trans_right=true:
  //   U_bt is (n, block_size) — left singular vectors of Bt
  //   Vt_bt is now V_bt (block_size x block_size) — right singular vectors of Bt (transposed)
  //   S_full has block_size singular values
  //
  // For B = Bt^T: U_b = V_bt = Vt_bt (after transpose), Vt_b = U_bt^T
  // So: U_b[:, :k] = Vt_bt[:, :k] and Vt_b[:k, :] = U_bt[:, :k]^T

  // Step 8: U = Q @ U_b[:, :k] = Q @ V_bt[:, :k]
  // Q is Y (m, block_size), V_bt is (block_size, block_size)
  // U = Y @ V_bt[:, :k] → (m, block_size) * (block_size, k) → (m, k)
  if (U) {
    const ValueTypeT one  = 1;
    const ValueTypeT zero = 0;
    raft::linalg::gemm(handle,
                       Y.data_handle(),
                       m,
                       block_size,
                       Vt_bt->data_handle(),  // This is V_bt after trans_right=true
                       U->data_handle(),
                       m,
                       k,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       one,
                       zero,
                       stream);
  }

  // Step 9: Truncate S, optionally compute Vt
  raft::copy(handle,
             singular_values,
             raft::make_device_vector_view<const ValueTypeT, uint32_t>(S_full.data_handle(),
                                                                       static_cast<uint32_t>(k)));

  // Vt[:k, :] = U_bt[:, :k]^T
  // U_bt is col-major (n, block_size); transpose its first k columns to (k, n) col-major.
  if (Vt) { raft::linalg::transpose(handle, U_bt->data_handle(), Vt->data_handle(), n, k, stream); }

  // Step 10: Sign correction (no-op when neither U nor Vt is requested)
  svd_sign_correction<ValueTypeT>(handle, U, Vt);
}

}  // namespace raft::sparse::solver::detail
