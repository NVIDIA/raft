/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/solver/detail/svds_sign_correction.cuh>
#include <raft/sparse/solver/solver_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace raft::sparse::solver::detail {

template <typename ValueTypeT>
RAFT_KERNEL negate_scalar_kernel(ValueTypeT* x)
{
  *x = -*x;
}

template <typename ValueTypeT>
ValueTypeT vector_norm(raft::resources const& handle, ValueTypeT const* x, int n)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::vector_norm");
  auto cublas_handle = resource::get_cublas_handle(handle);
  auto stream        = resource::get_cuda_stream(handle);
  ValueTypeT result{};
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasnrm2(cublas_handle, n, x, 1, &result, stream));
  resource::sync_stream(handle, stream);
  return result;
}

template <typename ValueTypeT>
void scale_vector(raft::resources const& handle, ValueTypeT* x, int n, ValueTypeT alpha)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::scale_vector");
  auto cublas_handle = resource::get_cublas_handle(handle);
  auto stream        = resource::get_cuda_stream(handle);
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasscal(cublas_handle, n, &alpha, x, 1, stream));
}

template <typename ValueTypeT>
void axpy(
  raft::resources const& handle, int n, ValueTypeT alpha, ValueTypeT const* x, ValueTypeT* y)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::axpy");
  auto cublas_handle = resource::get_cublas_handle(handle);
  auto stream        = resource::get_cuda_stream(handle);
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasaxpy(cublas_handle, n, &alpha, x, 1, y, 1, stream));
}

template <typename ValueTypeT>
void cgs2_orthogonalize(raft::resources const& handle,
                        ValueTypeT* target,
                        ValueTypeT const* basis,
                        int n_rows,
                        int n_valid,
                        ValueTypeT* coeffs)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::cgs2_orthogonalize");
  if (n_valid <= 0) { return; }

  auto cublas_handle = resource::get_cublas_handle(handle);
  auto stream        = resource::get_cuda_stream(handle);

  ValueTypeT one     = ValueTypeT(1);
  ValueTypeT zero    = ValueTypeT(0);
  ValueTypeT neg_one = ValueTypeT(-1);
  for (int pass = 0; pass < 2; ++pass) {
    {
      common::nvtx::range<common::nvtx::domain::raft> project_scope("lanczos_svds::cgs2_project");
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_handle,
                                                       CUBLAS_OP_T,
                                                       n_rows,
                                                       n_valid,
                                                       &one,
                                                       basis,
                                                       n_rows,
                                                       target,
                                                       1,
                                                       &zero,
                                                       coeffs,
                                                       1,
                                                       stream));
    }
    {
      common::nvtx::range<common::nvtx::domain::raft> subtract_scope("lanczos_svds::cgs2_subtract");
      RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_handle,
                                                       CUBLAS_OP_N,
                                                       n_rows,
                                                       n_valid,
                                                       &neg_one,
                                                       basis,
                                                       n_rows,
                                                       coeffs,
                                                       1,
                                                       &one,
                                                       target,
                                                       1,
                                                       stream));
    }
  }
}

template <typename ValueTypeT>
void mgs2_orthogonalize(raft::resources const& handle,
                        ValueTypeT* target,
                        ValueTypeT const* basis,
                        int n_rows,
                        int n_valid,
                        ValueTypeT* coeffs)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::mgs2_orthogonalize");
  if (n_valid <= 0) { return; }

  auto cublas_handle = resource::get_cublas_handle(handle);
  auto stream        = resource::get_cuda_stream(handle);

  RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE));
  for (int pass = 0; pass < 2; ++pass) {
    for (int j = 0; j < n_valid; ++j) {
      auto const* q_j = basis + static_cast<std::size_t>(j) * n_rows;
      auto* coeff     = coeffs + j;
      {
        common::nvtx::range<common::nvtx::domain::raft> dot_scope("lanczos_svds::mgs2_dot");
        RAFT_CUBLAS_TRY(
          raft::linalg::detail::cublasdot(cublas_handle, n_rows, q_j, 1, target, 1, coeff, stream));
      }
      {
        common::nvtx::range<common::nvtx::domain::raft> negate_scope("lanczos_svds::mgs2_negate");
        negate_scalar_kernel<<<1, 1, 0, stream>>>(coeff);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
      {
        common::nvtx::range<common::nvtx::domain::raft> subtract_scope(
          "lanczos_svds::mgs2_subtract");
        RAFT_CUBLAS_TRY(raft::linalg::detail::cublasaxpy(
          cublas_handle, n_rows, coeff, q_j, 1, target, 1, stream));
      }
    }
  }
  RAFT_CUBLAS_TRY(cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST));
}

template <typename ValueTypeT>
void orthogonalize(raft::resources const& handle,
                   ValueTypeT* target,
                   ValueTypeT const* basis,
                   int n_rows,
                   int n_valid,
                   ValueTypeT* coeffs,
                   bool use_mgs2)
{
  if (use_mgs2) {
    mgs2_orthogonalize(handle, target, basis, n_rows, n_valid, coeffs);
  } else {
    cgs2_orthogonalize(handle, target, basis, n_rows, n_valid, coeffs);
  }
}

template <typename ValueTypeT>
ValueTypeT normalize_or_randomize(raft::resources const& handle,
                                  raft::random::RngState& rng_state,
                                  ValueTypeT* target,
                                  int n_rows,
                                  ValueTypeT const* basis,
                                  int n_valid,
                                  ValueTypeT* coeffs,
                                  ValueTypeT eps,
                                  bool use_mgs2,
                                  bool* used_random_vector)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::normalize_or_randomize");
  auto stream         = resource::get_cuda_stream(handle);
  auto nrm            = vector_norm(handle, target, n_rows);
  *used_random_vector = false;

  if (nrm < eps) {
    {
      common::nvtx::range<common::nvtx::domain::raft> random_scope(
        "lanczos_svds::random_restart_vector");
      raft::random::normal(
        handle, rng_state, target, static_cast<std::size_t>(n_rows), ValueTypeT(0), ValueTypeT(1));
    }
    orthogonalize(handle, target, basis, n_rows, n_valid, coeffs, use_mgs2);
    nrm                 = vector_norm(handle, target, n_rows);
    *used_random_vector = true;
  }

  RAFT_EXPECTS(nrm >= eps, "Unable to generate a non-zero Lanczos vector");
  auto inv_nrm = ValueTypeT(1) / nrm;
  scale_vector(handle, target, n_rows, inv_nrm);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  resource::sync_stream(handle, stream);
  return nrm;
}

template <typename ValueTypeT>
void build_bidiagonal_matrix(raft::resources const& handle,
                             std::vector<ValueTypeT> const& alphas,
                             std::vector<ValueTypeT> const& betas,
                             raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> B)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::build_bidiagonal_matrix");
  auto stream = resource::get_cuda_stream(handle);
  int ncv     = static_cast<int>(alphas.size());
  std::vector<ValueTypeT> h_B(static_cast<std::size_t>(ncv) * ncv, ValueTypeT(0));

  for (int i = 0; i < ncv; ++i) {
    h_B[static_cast<std::size_t>(i) * ncv + i] = alphas[i];
    if (i + 1 < ncv) { h_B[static_cast<std::size_t>(i + 1) * ncv + i] = betas[i + 1]; }
  }

  raft::update_device(B.data_handle(), h_B.data(), static_cast<std::size_t>(ncv) * ncv, stream);
}

template <typename ValueTypeT, typename OperatorT>
void lanczos_bidiagonalize(raft::resources const& handle,
                           OperatorT const& op,
                           int ncv,
                           ValueTypeT* v_start,
                           raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> U_full,
                           raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> V_full,
                           int n_locked,
                           raft::random::RngState& rng_state,
                           ValueTypeT* ortho_coeffs,
                           bool use_mgs2,
                           std::vector<ValueTypeT>& alphas,
                           std::vector<ValueTypeT>& betas)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::bidiagonalize");
  int m          = op.rows();
  int n          = op.cols();
  ValueTypeT eps = ValueTypeT(1e-9);
  auto stream    = resource::get_cuda_stream(handle);

  alphas.assign(ncv, ValueTypeT(0));
  betas.assign(ncv + 1, ValueTypeT(0));

  auto* v0 = V_full.data_handle() + static_cast<std::size_t>(n_locked) * n;
  raft::copy(v0, v_start, n, stream);
  orthogonalize(handle, v0, V_full.data_handle(), n, n_locked, ortho_coeffs, use_mgs2);

  bool used_random = false;
  normalize_or_randomize(handle,
                         rng_state,
                         v0,
                         n,
                         V_full.data_handle(),
                         n_locked,
                         ortho_coeffs,
                         eps,
                         use_mgs2,
                         &used_random);

  for (int i = 0; i < ncv; ++i) {
    int idx_u = n_locked + i;
    int idx_v = n_locked + i;

    auto* u = U_full.data_handle() + static_cast<std::size_t>(idx_u) * m;
    auto* v = V_full.data_handle() + static_cast<std::size_t>(idx_v) * n;

    {
      common::nvtx::range<common::nvtx::domain::raft> apply_scope("lanczos_svds::apply_A");
      op.apply(handle,
               raft::make_device_matrix_view<ValueTypeT const, uint32_t, raft::col_major>(v, n, 1),
               raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(u, m, 1));
    }

    if (i > 0) {
      auto* u_prev = U_full.data_handle() + static_cast<std::size_t>(idx_u - 1) * m;
      axpy(handle, m, -betas[i], u_prev, u);
    }

    orthogonalize(handle, u, U_full.data_handle(), m, idx_u, ortho_coeffs, use_mgs2);
    used_random     = false;
    auto alpha_norm = normalize_or_randomize(handle,
                                             rng_state,
                                             u,
                                             m,
                                             U_full.data_handle(),
                                             idx_u,
                                             ortho_coeffs,
                                             eps,
                                             use_mgs2,
                                             &used_random);
    alphas[i]       = used_random ? ValueTypeT(0) : alpha_norm;

    auto* v_next = V_full.data_handle() + static_cast<std::size_t>(idx_v + 1) * n;
    {
      common::nvtx::range<common::nvtx::domain::raft> apply_t_scope("lanczos_svds::apply_AT");
      op.apply_transpose(
        handle,
        raft::make_device_matrix_view<ValueTypeT const, uint32_t, raft::col_major>(u, m, 1),
        raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(v_next, n, 1));
    }

    axpy(handle, n, -alphas[i], v, v_next);
    orthogonalize(handle, v_next, V_full.data_handle(), n, idx_v + 1, ortho_coeffs, use_mgs2);

    used_random   = false;
    auto beta_nrm = normalize_or_randomize(handle,
                                           rng_state,
                                           v_next,
                                           n,
                                           V_full.data_handle(),
                                           idx_v + 1,
                                           ortho_coeffs,
                                           eps,
                                           use_mgs2,
                                           &used_random);
    betas[i + 1]  = used_random ? ValueTypeT(0) : beta_nrm;
  }
}

template <typename ValueTypeT>
void compute_ritz_vectors(raft::resources const& handle,
                          int m,
                          int n,
                          int ncv,
                          int n_locked,
                          std::vector<int> const& indices,
                          std::vector<ValueTypeT> const& singular_values,
                          std::vector<ValueTypeT> const& P,
                          std::vector<ValueTypeT> const& Qt,
                          raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> U_full,
                          raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> V_full,
                          std::vector<ValueTypeT>& locked_singular_values)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::compute_ritz_vectors");
  auto stream   = resource::get_cuda_stream(handle);
  int num_found = static_cast<int>(indices.size());
  if (num_found == 0) { return; }

  std::vector<ValueTypeT> h_P_good(static_cast<std::size_t>(ncv) * num_found);
  std::vector<ValueTypeT> h_Qt_good_t(static_cast<std::size_t>(ncv) * num_found);
  for (int j = 0; j < num_found; ++j) {
    int src_col = indices[j];
    for (int row = 0; row < ncv; ++row) {
      h_P_good[static_cast<std::size_t>(j) * ncv + row] =
        P[static_cast<std::size_t>(src_col) * ncv + row];
      h_Qt_good_t[static_cast<std::size_t>(j) * ncv + row] =
        Qt[static_cast<std::size_t>(row) * ncv + src_col];
    }
  }

  auto P_good =
    raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, ncv, num_found);
  auto Qt_good_t =
    raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, ncv, num_found);
  {
    common::nvtx::range<common::nvtx::domain::raft> copy_scope("lanczos_svds::ritz_copy_basis");
    raft::update_device(P_good.data_handle(), h_P_good.data(), h_P_good.size(), stream);
    raft::update_device(Qt_good_t.data_handle(), h_Qt_good_t.data(), h_Qt_good_t.size(), stream);
  }

  auto U_ritz =
    raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, m, num_found);
  auto V_ritz =
    raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, n, num_found);

  ValueTypeT one  = 1;
  ValueTypeT zero = 0;
  auto* U_segment = U_full.data_handle() + static_cast<std::size_t>(n_locked) * m;
  auto* V_segment = V_full.data_handle() + static_cast<std::size_t>(n_locked) * n;

  {
    common::nvtx::range<common::nvtx::domain::raft> gemm_scope("lanczos_svds::ritz_gemm_U");
    raft::linalg::gemm(handle,
                       U_segment,
                       m,
                       ncv,
                       P_good.data_handle(),
                       U_ritz.data_handle(),
                       m,
                       num_found,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       one,
                       zero,
                       stream);
  }

  {
    common::nvtx::range<common::nvtx::domain::raft> gemm_scope("lanczos_svds::ritz_gemm_V");
    raft::linalg::gemm(handle,
                       V_segment,
                       n,
                       ncv,
                       Qt_good_t.data_handle(),
                       V_ritz.data_handle(),
                       n,
                       num_found,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       one,
                       zero,
                       stream);
  }

  {
    common::nvtx::range<common::nvtx::domain::raft> copy_scope("lanczos_svds::ritz_copy_output");
    raft::copy(U_full.data_handle() + static_cast<std::size_t>(n_locked) * m,
               U_ritz.data_handle(),
               static_cast<std::size_t>(m) * num_found,
               stream);
    raft::copy(V_full.data_handle() + static_cast<std::size_t>(n_locked) * n,
               V_ritz.data_handle(),
               static_cast<std::size_t>(n) * num_found,
               stream);
  }

  for (int j = 0; j < num_found; ++j) {
    locked_singular_values[n_locked + j] = singular_values[indices[j]];
  }
}

template <typename ValueTypeT>
void compute_restart_vector(raft::resources const& handle,
                            int n,
                            int ncv,
                            int n_locked,
                            std::vector<ValueTypeT> const& coeffs,
                            raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> V_full,
                            ValueTypeT* v_start)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::compute_restart_vector");
  auto stream = resource::get_cuda_stream(handle);
  auto d_coeffs =
    raft::make_device_vector<ValueTypeT, uint32_t>(handle, static_cast<uint32_t>(ncv));
  raft::update_device(d_coeffs.data_handle(), coeffs.data(), ncv, stream);

  auto cublas_handle   = resource::get_cublas_handle(handle);
  ValueTypeT const one = ValueTypeT(1);
  ValueTypeT zero      = ValueTypeT(0);
  auto* V_segment      = V_full.data_handle() + static_cast<std::size_t>(n_locked) * n;
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(cublas_handle,
                                                   CUBLAS_OP_N,
                                                   n,
                                                   ncv,
                                                   &one,
                                                   V_segment,
                                                   n,
                                                   d_coeffs.data_handle(),
                                                   1,
                                                   &zero,
                                                   v_start,
                                                   1,
                                                   stream));

  auto nrm = vector_norm(handle, v_start, n);
  if (nrm > ValueTypeT(0)) { scale_vector(handle, v_start, n, ValueTypeT(1) / nrm); }
}

template <typename ValueTypeT>
void sort_singular_triplets(raft::resources const& handle,
                            ValueTypeT* U_cols,
                            int m,
                            ValueTypeT* V_cols,
                            int n,
                            std::vector<ValueTypeT>& singular_values)
{
  common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::sort_singular_triplets");
  int k = static_cast<int>(singular_values.size());
  std::vector<int> order(k);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return singular_values[a] > singular_values[b];
  });

  bool already_sorted = true;
  for (int i = 0; i < k; ++i) {
    already_sorted = already_sorted && order[i] == i;
  }
  if (already_sorted) { return; }

  auto stream   = resource::get_cuda_stream(handle);
  auto U_sorted = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, m, k);
  auto V_sorted = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, n, k);
  std::vector<ValueTypeT> sorted_s(k);

  for (int dst = 0; dst < k; ++dst) {
    int src = order[dst];
    raft::copy(U_sorted.data_handle() + static_cast<std::size_t>(dst) * m,
               U_cols + static_cast<std::size_t>(src) * m,
               m,
               stream);
    raft::copy(V_sorted.data_handle() + static_cast<std::size_t>(dst) * n,
               V_cols + static_cast<std::size_t>(src) * n,
               n,
               stream);
    sorted_s[dst] = singular_values[src];
  }

  raft::copy(U_cols, U_sorted.data_handle(), static_cast<std::size_t>(m) * k, stream);
  raft::copy(V_cols, V_sorted.data_handle(), static_cast<std::size_t>(n) * k, stream);
  singular_values.swap(sorted_s);
}

/**
 * @brief Lanczos bidiagonalization SVD for sparse matrices and linear operators.
 */
template <typename ValueTypeT, typename OperatorT>
void sparse_lanczos_svd(
  raft::resources const& handle,
  sparse_lanczos_svd_config<ValueTypeT> const& config,
  OperatorT const& op,
  raft::device_vector_view<ValueTypeT, uint32_t> singular_values,
  std::optional<raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> U,
  std::optional<raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> Vt)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "raft::sparse::solver::sparse_lanczos_svd(%d, %d, %d)",
    op.rows(),
    op.cols(),
    config.n_components);

  int m       = op.rows();
  int n       = op.cols();
  int k       = config.n_components;
  int min_dim = std::min(m, n);

  RAFT_EXPECTS(k > 0, "n_components must be positive");
  RAFT_EXPECTS(k < min_dim, "n_components must be less than min(m, n)");
  RAFT_EXPECTS(config.max_iterations > 0, "max_iterations must be positive");
  RAFT_EXPECTS(config.tolerance >= ValueTypeT(0), "tolerance must be non-negative");
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

  int ncv = config.ncv;
  if (ncv <= 0) {
    if (m > 100000) {
      ncv = (k < 75) ? (22 * k + 9) / 10 : static_cast<int>(3.9 * k);
    } else {
      ncv = std::max(3 * k, 50);
    }
  }
  ncv = std::max(ncv, k);
  ncv = std::min(ncv, min_dim - 1);
  RAFT_EXPECTS(ncv >= k, "ncv must be at least n_components after clamping");

  auto stream   = resource::get_cuda_stream(handle);
  uint64_t seed = config.seed.value_or(std::random_device{}());
  raft::random::RngState rng_state(seed);

  int total_capacity = k + ncv + 2;
  auto U_full        = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
    handle, static_cast<uint32_t>(m), static_cast<uint32_t>(total_capacity));
  auto V_full = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
    handle, static_cast<uint32_t>(n), static_cast<uint32_t>(total_capacity));
  auto ortho_coeffs =
    raft::make_device_vector<ValueTypeT, uint32_t>(handle, static_cast<uint32_t>(total_capacity));
  auto v_start = raft::make_device_vector<ValueTypeT, uint32_t>(handle, static_cast<uint32_t>(n));

  {
    common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::initial_random_vector");
    raft::random::normal(handle,
                         rng_state,
                         v_start.data_handle(),
                         static_cast<std::size_t>(n),
                         ValueTypeT(0),
                         ValueTypeT(1));
  }

  std::vector<ValueTypeT> locked_singular_values(k, ValueTypeT(0));
  int n_locked   = 0;
  int total_iter = 0;

  while (n_locked < k && total_iter < config.max_iterations) {
    common::nvtx::range<common::nvtx::domain::raft> restart_scope(
      "lanczos_svds::restart_iteration");
    int active_ncv = std::min(ncv, min_dim - n_locked - 1);
    RAFT_EXPECTS(active_ncv > 0, "No remaining subspace available for Lanczos restart");

    std::vector<ValueTypeT> alphas;
    std::vector<ValueTypeT> betas;
    lanczos_bidiagonalize(handle,
                          op,
                          active_ncv,
                          v_start.data_handle(),
                          U_full.view(),
                          V_full.view(),
                          n_locked,
                          rng_state,
                          ortho_coeffs.data_handle(),
                          config.use_mgs2_orthogonalization,
                          alphas,
                          betas);

    auto B = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
      handle, static_cast<uint32_t>(active_ncv), static_cast<uint32_t>(active_ncv));
    build_bidiagonal_matrix(handle, alphas, betas, B.view());

    auto P = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
      handle, static_cast<uint32_t>(active_ncv), static_cast<uint32_t>(active_ncv));
    auto Qt = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
      handle, static_cast<uint32_t>(active_ncv), static_cast<uint32_t>(active_ncv));
    auto S_full =
      raft::make_device_vector<ValueTypeT, uint32_t>(handle, static_cast<uint32_t>(active_ncv));

    {
      common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::bidiagonal_svd");
      raft::linalg::svdQR(handle,
                          B.data_handle(),
                          active_ncv,
                          active_ncv,
                          S_full.data_handle(),
                          P.data_handle(),
                          Qt.data_handle(),
                          false,
                          true,
                          true,
                          stream);
    }

    std::vector<ValueTypeT> h_s(active_ncv);
    std::vector<ValueTypeT> h_P(static_cast<std::size_t>(active_ncv) * active_ncv);
    std::vector<ValueTypeT> h_Qt(static_cast<std::size_t>(active_ncv) * active_ncv);
    {
      common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::copy_small_svd_host");
      raft::update_host(h_s.data(), S_full.data_handle(), active_ncv, stream);
      raft::update_host(h_P.data(), P.data_handle(), h_P.size(), stream);
      raft::update_host(h_Qt.data(), Qt.data_handle(), h_Qt.size(), stream);
      resource::sync_stream(handle, stream);
    }

    int remaining        = k - n_locked;
    ValueTypeT max_s     = h_s.empty() || h_s[0] <= ValueTypeT(0) ? ValueTypeT(1) : h_s[0];
    int check_components = std::min(remaining + 2, active_ncv);

    std::vector<int> selected;
    {
      common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::select_converged");
      for (int i = 0; i < check_components; ++i) {
        auto residual = betas[active_ncv] *
                        std::abs(h_P[static_cast<std::size_t>(i) * active_ncv + active_ncv - 1]);
        if (residual < config.tolerance * max_s) { selected.push_back(i); }
      }
    }
    if (static_cast<int>(selected.size()) > remaining) { selected.resize(remaining); }

    if (!selected.empty()) {
      compute_ritz_vectors(handle,
                           m,
                           n,
                           active_ncv,
                           n_locked,
                           selected,
                           h_s,
                           h_P,
                           h_Qt,
                           U_full.view(),
                           V_full.view(),
                           locked_singular_values);
      n_locked += static_cast<int>(selected.size());
      if (n_locked >= k) { break; }
    }

    std::vector<ValueTypeT> restart_coeffs(active_ncv, ValueTypeT(0));
    {
      common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::build_restart_coeffs");
      if (!selected.empty()) {
        int best_unconverged = 0;
        for (; best_unconverged < active_ncv; ++best_unconverged) {
          if (std::find(selected.begin(), selected.end(), best_unconverged) == selected.end()) {
            break;
          }
        }
        if (best_unconverged == active_ncv) { best_unconverged = 0; }
        for (int col = 0; col < active_ncv; ++col) {
          restart_coeffs[col] = h_Qt[static_cast<std::size_t>(col) * active_ncv + best_unconverged];
        }
      } else {
        int k_mix = std::min(k - n_locked, active_ncv);
        for (int col = 0; col < active_ncv; ++col) {
          ValueTypeT sum = 0;
          for (int row = 0; row < k_mix; ++row) {
            sum += h_Qt[static_cast<std::size_t>(col) * active_ncv + row];
          }
          restart_coeffs[col] = sum;
        }
      }
    }

    compute_restart_vector(
      handle, n, active_ncv, n_locked, restart_coeffs, V_full.view(), v_start.data_handle());
    ++total_iter;
  }

  RAFT_EXPECTS(n_locked >= k,
               "sparse_lanczos_svd failed to converge all requested components within "
               "max_iterations");

  auto U_refined = raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(
    handle, static_cast<uint32_t>(m), static_cast<uint32_t>(k));
  {
    common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::post_refine_A_V");
    op.apply(handle,
             raft::make_device_matrix_view<ValueTypeT const, uint32_t, raft::col_major>(
               V_full.data_handle(), n, k),
             U_refined.view());
  }

  {
    common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::post_refine_normalize");
    for (int j = 0; j < k; ++j) {
      auto* u_col               = U_refined.data_handle() + static_cast<std::size_t>(j) * m;
      locked_singular_values[j] = vector_norm(handle, u_col, m);
      if (locked_singular_values[j] > ValueTypeT(0)) {
        scale_vector(handle, u_col, m, ValueTypeT(1) / locked_singular_values[j]);
      }
    }
  }

  sort_singular_triplets(
    handle, U_refined.data_handle(), m, V_full.data_handle(), n, locked_singular_values);

  {
    common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::copy_outputs");
    raft::update_device(singular_values.data_handle(), locked_singular_values.data(), k, stream);
    if (U) {
      raft::copy(
        U->data_handle(), U_refined.data_handle(), static_cast<std::size_t>(m) * k, stream);
    }
  }

  if (Vt) {
    common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::transpose_vt");
    raft::linalg::transpose(handle, V_full.data_handle(), Vt->data_handle(), n, k, stream);
  }

  {
    common::nvtx::range<common::nvtx::domain::raft> scope("lanczos_svds::sign_correction");
    svd_sign_correction<ValueTypeT>(handle, U, Vt);
  }
}

}  // namespace raft::sparse::solver::detail
