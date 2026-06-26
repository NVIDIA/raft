/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../test_utils.cuh"

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/lanczos_svds.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace raft::sparse::solver {

template <typename ValueType>
void check_orthonormal_columns(std::vector<ValueType> const& Q, int n_rows, int n_cols, double tol)
{
  for (int i = 0; i < n_cols; ++i) {
    for (int j = 0; j < n_cols; ++j) {
      double dot = 0;
      for (int row = 0; row < n_rows; ++row) {
        dot += static_cast<double>(Q[static_cast<std::size_t>(i) * n_rows + row]) *
               static_cast<double>(Q[static_cast<std::size_t>(j) * n_rows + row]);
      }
      double expected = (i == j) ? 1.0 : 0.0;
      ASSERT_NEAR(dot, expected, tol);
    }
  }
}

template <typename ValueType>
void check_orthonormal_vt_rows(std::vector<ValueType> const& Vt, int k, int n, double tol)
{
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      double dot = 0;
      for (int col = 0; col < n; ++col) {
        dot += static_cast<double>(Vt[static_cast<std::size_t>(col) * k + i]) *
               static_cast<double>(Vt[static_cast<std::size_t>(col) * k + j]);
      }
      double expected = (i == j) ? 1.0 : 0.0;
      ASSERT_NEAR(dot, expected, tol);
    }
  }
}

template <typename ValueType>
class LanczosSvdsTest : public ::testing::Test {
 public:
  LanczosSvdsTest()
    : stream(resource::get_cuda_stream(handle)),
      m(12),
      n(8),
      k(3),
      nnz(8),
      d_indptr(raft::make_device_vector<int, uint32_t>(handle, m + 1)),
      d_indices(raft::make_device_vector<int, uint32_t>(handle, nnz)),
      d_values(raft::make_device_vector<ValueType, uint32_t>(handle, nnz))
  {
  }

 protected:
  void SetUp() override
  {
    h_indptr = {0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8};
    h_indices.resize(nnz);
    h_values.resize(nnz);
    for (int i = 0; i < nnz; ++i) {
      h_indices[i] = i;
      h_values[i]  = ValueType(8 - i);
    }

    raft::update_device(d_indptr.data_handle(), h_indptr.data(), m + 1, stream);
    raft::update_device(d_indices.data_handle(), h_indices.data(), nnz, stream);
    raft::update_device(d_values.data_handle(), h_values.data(), nnz, stream);
  }

  void Run(bool use_mgs2 = false)
  {
    auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
      d_indptr.data_handle(), d_indices.data_handle(), m, n, nnz);
    auto csr_matrix = raft::make_device_csr_matrix_view<const ValueType, int, int, int>(
      d_values.data_handle(), csr_structure);

    sparse_lanczos_svd_config<ValueType> config;
    config.n_components               = k;
    config.ncv                        = 6;
    config.tolerance                  = ValueType(1e-6);
    config.max_iterations             = 20;
    config.seed                       = 42;
    config.use_mgs2_orthogonalization = use_mgs2;

    auto S  = raft::make_device_vector<ValueType, uint32_t>(handle, k);
    auto U  = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, k);
    auto Vt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, n);

    sparse_lanczos_svd(handle, config, csr_matrix, S.view(), U.view(), Vt.view());

    std::vector<ValueType> h_S(k);
    std::vector<ValueType> h_U(static_cast<std::size_t>(m) * k);
    std::vector<ValueType> h_Vt(static_cast<std::size_t>(k) * n);
    raft::update_host(h_S.data(), S.data_handle(), k, stream);
    raft::update_host(h_U.data(), U.data_handle(), h_U.size(), stream);
    raft::update_host(h_Vt.data(), Vt.data_handle(), h_Vt.size(), stream);
    resource::sync_stream(handle, stream);

    ValueType tol       = std::is_same_v<ValueType, float> ? ValueType(2e-3) : ValueType(1e-8);
    double residual_tol = std::is_same_v<ValueType, float> ? 2e-2 : 1e-5;
    for (int i = 0; i < k; ++i) {
      ASSERT_NEAR(h_S[i], ValueType(8 - i), tol);
    }

    check_orthonormal_columns(h_U, m, k, tol);
    check_orthonormal_vt_rows(h_Vt, k, n, tol);

    for (int comp = 0; comp < k; ++comp) {
      double left_residual_sq  = 0;
      double right_residual_sq = 0;
      for (int row = 0; row < m; ++row) {
        ValueType av =
          row < n ? h_values[row] * h_Vt[static_cast<std::size_t>(row) * k + comp] : ValueType(0);
        ValueType su = h_S[comp] * h_U[static_cast<std::size_t>(comp) * m + row];
        double diff  = static_cast<double>(av - su);
        left_residual_sq += diff * diff;
      }
      for (int col = 0; col < n; ++col) {
        ValueType atu = h_values[col] * h_U[static_cast<std::size_t>(comp) * m + col];
        ValueType sv  = h_S[comp] * h_Vt[static_cast<std::size_t>(col) * k + comp];
        double diff   = static_cast<double>(atu - sv);
        right_residual_sq += diff * diff;
      }
      ASSERT_LT(std::sqrt(left_residual_sq), residual_tol);
      ASSERT_LT(std::sqrt(right_residual_sq), residual_tol);
    }
  }

  raft::resources handle;
  cudaStream_t stream;
  int m, n, k, nnz;
  std::vector<int> h_indptr;
  std::vector<int> h_indices;
  std::vector<ValueType> h_values;
  raft::device_vector<int, uint32_t> d_indptr;
  raft::device_vector<int, uint32_t> d_indices;
  raft::device_vector<ValueType, uint32_t> d_values;
};

using LanczosSvdsTestF = LanczosSvdsTest<float>;
TEST_F(LanczosSvdsTestF, DiagonalSpectrum) { Run(); }
TEST_F(LanczosSvdsTestF, DiagonalSpectrumMgs2) { Run(true); }

using LanczosSvdsTestD = LanczosSvdsTest<double>;
TEST_F(LanczosSvdsTestD, DiagonalSpectrum) { Run(); }

template <typename ValueType>
class LanczosClusteredSpectrumTest : public ::testing::Test {
 public:
  LanczosClusteredSpectrumTest()
    : stream(resource::get_cuda_stream(handle)),
      m(48),
      n(32),
      k(8),
      nnz(32),
      d_indptr(raft::make_device_vector<int, uint32_t>(handle, m + 1)),
      d_indices(raft::make_device_vector<int, uint32_t>(handle, nnz)),
      d_values(raft::make_device_vector<ValueType, uint32_t>(handle, nnz))
  {
  }

 protected:
  void SetUp() override
  {
    h_indptr.resize(m + 1);
    h_indices.resize(nnz);
    h_values.resize(nnz);
    for (int i = 0; i <= m; ++i) {
      h_indptr[i] = std::min(i, nnz);
    }
    for (int i = 0; i < nnz; ++i) {
      h_indices[i] = i;
    }

    std::vector<double> singular_values = {10.0, 9.999, 9.998, 9.5, 9.499, 9.0, 8.999, 8.5};
    for (int i = 0; i < nnz; ++i) {
      h_values[i] = i < static_cast<int>(singular_values.size())
                      ? static_cast<ValueType>(singular_values[i])
                      : static_cast<ValueType>(1.0 / (i + 1));
    }

    raft::update_device(d_indptr.data_handle(), h_indptr.data(), m + 1, stream);
    raft::update_device(d_indices.data_handle(), h_indices.data(), nnz, stream);
    raft::update_device(d_values.data_handle(), h_values.data(), nnz, stream);
  }

  void Run(bool use_mgs2 = false)
  {
    auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
      d_indptr.data_handle(), d_indices.data_handle(), m, n, nnz);
    auto csr_matrix = raft::make_device_csr_matrix_view<const ValueType, int, int, int>(
      d_values.data_handle(), csr_structure);

    sparse_lanczos_svd_config<ValueType> config;
    config.n_components   = k;
    config.ncv            = 18;
    config.tolerance      = std::is_same_v<ValueType, float> ? ValueType(1e-5) : ValueType(1e-10);
    config.max_iterations = 80;
    config.seed           = 123;
    config.use_mgs2_orthogonalization = use_mgs2;

    auto S  = raft::make_device_vector<ValueType, uint32_t>(handle, k);
    auto U  = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, k);
    auto Vt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, n);

    sparse_lanczos_svd(handle, config, csr_matrix, S.view(), U.view(), Vt.view());

    std::vector<ValueType> h_S(k);
    std::vector<ValueType> h_U(static_cast<std::size_t>(m) * k);
    std::vector<ValueType> h_Vt(static_cast<std::size_t>(k) * n);
    raft::update_host(h_S.data(), S.data_handle(), k, stream);
    raft::update_host(h_U.data(), U.data_handle(), h_U.size(), stream);
    raft::update_host(h_Vt.data(), Vt.data_handle(), h_Vt.size(), stream);
    resource::sync_stream(handle, stream);

    double sv_tol     = std::is_same_v<ValueType, float> ? 1e-3 : 1e-8;
    double orthog_tol = std::is_same_v<ValueType, float> ? 2e-4 : 1e-10;
    for (int i = 0; i < k; ++i) {
      ASSERT_NEAR(static_cast<double>(h_S[i]), static_cast<double>(h_values[i]), sv_tol);
    }
    check_orthonormal_columns(h_U, m, k, orthog_tol);
    check_orthonormal_vt_rows(h_Vt, k, n, orthog_tol);
  }

  raft::resources handle;
  cudaStream_t stream;
  int m, n, k, nnz;
  std::vector<int> h_indptr;
  std::vector<int> h_indices;
  std::vector<ValueType> h_values;
  raft::device_vector<int, uint32_t> d_indptr;
  raft::device_vector<int, uint32_t> d_indices;
  raft::device_vector<ValueType, uint32_t> d_values;
};

using LanczosClusteredSpectrumTestF = LanczosClusteredSpectrumTest<float>;
TEST_F(LanczosClusteredSpectrumTestF, Orthonormality) { Run(); }
TEST_F(LanczosClusteredSpectrumTestF, OrthonormalityMgs2) { Run(true); }

using LanczosClusteredSpectrumTestD = LanczosClusteredSpectrumTest<double>;
TEST_F(LanczosClusteredSpectrumTestD, Orthonormality) { Run(); }

template <typename ValueType>
void run_diagonal_hardening_case(int m,
                                 int n,
                                 std::vector<double> const& diagonal,
                                 int k,
                                 int ncv,
                                 int max_iterations,
                                 ValueType tolerance,
                                 double singular_value_tol,
                                 double orthog_tol,
                                 double residual_tol,
                                 bool use_mgs2 = false)
{
  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);

  std::vector<int> h_indptr(m + 1);
  std::vector<int> h_indices;
  std::vector<ValueType> h_values;
  int min_dim = std::min(m, n);
  for (int row = 0; row < m; ++row) {
    h_indptr[row] = static_cast<int>(h_values.size());
    if (row < min_dim && row < static_cast<int>(diagonal.size()) && diagonal[row] != 0.0) {
      h_indices.push_back(row);
      h_values.push_back(static_cast<ValueType>(diagonal[row]));
    }
  }
  h_indptr[m] = static_cast<int>(h_values.size());

  auto nnz       = static_cast<int>(h_values.size());
  auto d_indptr  = raft::make_device_vector<int, uint32_t>(handle, m + 1);
  auto d_indices = raft::make_device_vector<int, uint32_t>(handle, nnz);
  auto d_values  = raft::make_device_vector<ValueType, uint32_t>(handle, nnz);
  raft::update_device(d_indptr.data_handle(), h_indptr.data(), m + 1, stream);
  raft::update_device(d_indices.data_handle(), h_indices.data(), nnz, stream);
  raft::update_device(d_values.data_handle(), h_values.data(), nnz, stream);

  auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
    d_indptr.data_handle(), d_indices.data_handle(), m, n, nnz);
  auto csr_matrix = raft::make_device_csr_matrix_view<const ValueType, int, int, int>(
    d_values.data_handle(), csr_structure);

  sparse_lanczos_svd_config<ValueType> config;
  config.n_components               = k;
  config.ncv                        = ncv;
  config.tolerance                  = tolerance;
  config.max_iterations             = max_iterations;
  config.seed                       = 2026;
  config.use_mgs2_orthogonalization = use_mgs2;

  auto S  = raft::make_device_vector<ValueType, uint32_t>(handle, k);
  auto U  = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, k);
  auto Vt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, n);

  sparse_lanczos_svd(handle, config, csr_matrix, S.view(), U.view(), Vt.view());

  std::vector<ValueType> h_S(k);
  std::vector<ValueType> h_U(static_cast<std::size_t>(m) * k);
  std::vector<ValueType> h_Vt(static_cast<std::size_t>(k) * n);
  raft::update_host(h_S.data(), S.data_handle(), k, stream);
  raft::update_host(h_U.data(), U.data_handle(), h_U.size(), stream);
  raft::update_host(h_Vt.data(), Vt.data_handle(), h_Vt.size(), stream);
  resource::sync_stream(handle, stream);

  for (int i = 0; i < k; ++i) {
    ASSERT_NEAR(static_cast<double>(h_S[i]), diagonal[i], singular_value_tol);
  }
  check_orthonormal_columns(h_U, m, k, orthog_tol);
  check_orthonormal_vt_rows(h_Vt, k, n, orthog_tol);

  auto diag_at = [&](int i) {
    return (i < min_dim && i < static_cast<int>(diagonal.size()))
             ? static_cast<ValueType>(diagonal[i])
             : ValueType(0);
  };

  for (int comp = 0; comp < k; ++comp) {
    double left_residual_sq  = 0;
    double right_residual_sq = 0;
    for (int row = 0; row < m; ++row) {
      ValueType av =
        row < n ? diag_at(row) * h_Vt[static_cast<std::size_t>(row) * k + comp] : ValueType(0);
      ValueType su = h_S[comp] * h_U[static_cast<std::size_t>(comp) * m + row];
      double diff  = static_cast<double>(av - su);
      left_residual_sq += diff * diff;
    }
    for (int col = 0; col < n; ++col) {
      ValueType atu =
        col < m ? diag_at(col) * h_U[static_cast<std::size_t>(comp) * m + col] : ValueType(0);
      ValueType sv = h_S[comp] * h_Vt[static_cast<std::size_t>(col) * k + comp];
      double diff  = static_cast<double>(atu - sv);
      right_residual_sq += diff * diff;
    }
    ASSERT_LT(std::sqrt(left_residual_sq), residual_tol);
    ASSERT_LT(std::sqrt(right_residual_sq), residual_tol);
  }
}

TEST(LanczosHardeningF, RepeatedSpectrum)
{
  run_diagonal_hardening_case<float>(64,
                                     40,
                                     {10.0, 10.0, 10.0, 9.0, 9.0, 8.0, 7.0, 6.0, 1.0, 0.5},
                                     6,
                                     18,
                                     120,
                                     1e-5f,
                                     2e-3,
                                     2e-3,
                                     2e-2);
}

TEST(LanczosHardeningF, RankDeficientTall)
{
  run_diagonal_hardening_case<float>(
    96, 28, {12.0, 8.0, 6.0, 3.0, 1.0, 0.0, 0.0, 0.0}, 4, 12, 80, 1e-5f, 2e-3, 2e-3, 2e-2);
}

TEST(LanczosHardeningF, WideMatrix)
{
  run_diagonal_hardening_case<float>(
    24, 96, {12.0, 11.5, 8.0, 4.0, 2.0, 1.0, 0.5}, 4, 12, 80, 1e-5f, 2e-3, 2e-3, 2e-2);
}

TEST(LanczosHardeningD, RankDeficientTall)
{
  run_diagonal_hardening_case<double>(
    96, 28, {12.0, 8.0, 6.0, 3.0, 1.0, 0.0, 0.0, 0.0}, 4, 12, 80, 1e-10, 1e-8, 1e-10, 1e-5);
}

TEST(LanczosHardeningF, ReportsNonConvergence)
{
  EXPECT_THROW(run_diagonal_hardening_case<float>(
                 64, 40, {10.0, 9.0, 8.0, 7.0, 6.0, 5.0}, 4, 6, 1, 0.0f, 2e-3, 2e-3, 2e-2),
               raft::logic_error);
}

template <typename ValueType>
void check_sparse_residuals(std::vector<int> const& indptr,
                            std::vector<int> const& indices,
                            std::vector<ValueType> const& values,
                            std::vector<ValueType> const& S,
                            std::vector<ValueType> const& U,
                            std::vector<ValueType> const& Vt,
                            int m,
                            int n,
                            int k,
                            double residual_tol)
{
  for (int comp = 0; comp < k; ++comp) {
    std::vector<double> av(m, 0.0);
    std::vector<double> atu(n, 0.0);
    for (int row = 0; row < m; ++row) {
      for (int nz = indptr[row]; nz < indptr[row + 1]; ++nz) {
        int col      = indices[nz];
        double value = static_cast<double>(values[nz]);
        av[row] += value * static_cast<double>(Vt[static_cast<std::size_t>(col) * k + comp]);
        atu[col] += value * static_cast<double>(U[static_cast<std::size_t>(comp) * m + row]);
      }
    }

    double left_residual_sq  = 0.0;
    double right_residual_sq = 0.0;
    for (int row = 0; row < m; ++row) {
      double su = static_cast<double>(S[comp]) *
                  static_cast<double>(U[static_cast<std::size_t>(comp) * m + row]);
      double diff = av[row] - su;
      left_residual_sq += diff * diff;
    }
    for (int col = 0; col < n; ++col) {
      double sv = static_cast<double>(S[comp]) *
                  static_cast<double>(Vt[static_cast<std::size_t>(col) * k + comp]);
      double diff = atu[col] - sv;
      right_residual_sq += diff * diff;
    }
    ASSERT_LT(std::sqrt(left_residual_sq), residual_tol);
    ASSERT_LT(std::sqrt(right_residual_sq), residual_tol);
  }
}

template <typename ValueType>
void run_rotated_block_hardening_case(bool use_mgs2 = false)
{
  constexpr int m                = 64;
  constexpr int n                = 48;
  constexpr int k                = 8;
  std::vector<double> expected_s = {10.0, 9.9995, 9.999, 9.5, 9.4995, 9.0, 8.5, 8.0};
  std::vector<double> all_s      = expected_s;
  all_s.push_back(0.75);
  all_s.push_back(0.25);

  std::vector<int> h_indptr(m + 1);
  std::vector<int> h_indices;
  std::vector<ValueType> h_values;

  auto add_block = [&](int row0, int col0, double s0, double s1, double theta, double phi) {
    double cu  = std::cos(theta);
    double su  = std::sin(theta);
    double cv  = std::cos(phi);
    double sv  = std::sin(phi);
    double a00 = cu * s0 * cv + su * s1 * sv;
    double a01 = cu * s0 * sv - su * s1 * cv;
    double a10 = su * s0 * cv - cu * s1 * sv;
    double a11 = su * s0 * sv + cu * s1 * cv;
    h_indices.push_back(col0);
    h_values.push_back(static_cast<ValueType>(a00));
    h_indices.push_back(col0 + 1);
    h_values.push_back(static_cast<ValueType>(a01));
    h_indptr[row0 + 1] = static_cast<int>(h_values.size());
    h_indices.push_back(col0);
    h_values.push_back(static_cast<ValueType>(a10));
    h_indices.push_back(col0 + 1);
    h_values.push_back(static_cast<ValueType>(a11));
    h_indptr[row0 + 2] = static_cast<int>(h_values.size());
  };

  for (int block = 0; block < static_cast<int>(all_s.size() / 2); ++block) {
    int row0       = 2 * block;
    int col0       = 2 * block;
    h_indptr[row0] = static_cast<int>(h_values.size());
    add_block(
      row0, col0, all_s[2 * block], all_s[2 * block + 1], 0.17 + 0.11 * block, 0.31 + 0.07 * block);
  }
  for (int row = static_cast<int>(all_s.size()); row <= m; ++row) {
    h_indptr[row] = static_cast<int>(h_values.size());
  }

  raft::resources handle;
  auto stream    = resource::get_cuda_stream(handle);
  auto nnz       = static_cast<int>(h_values.size());
  auto d_indptr  = raft::make_device_vector<int, uint32_t>(handle, m + 1);
  auto d_indices = raft::make_device_vector<int, uint32_t>(handle, nnz);
  auto d_values  = raft::make_device_vector<ValueType, uint32_t>(handle, nnz);
  raft::update_device(d_indptr.data_handle(), h_indptr.data(), m + 1, stream);
  raft::update_device(d_indices.data_handle(), h_indices.data(), nnz, stream);
  raft::update_device(d_values.data_handle(), h_values.data(), nnz, stream);

  auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
    d_indptr.data_handle(), d_indices.data_handle(), m, n, nnz);
  auto csr_matrix = raft::make_device_csr_matrix_view<const ValueType, int, int, int>(
    d_values.data_handle(), csr_structure);

  sparse_lanczos_svd_config<ValueType> config;
  config.n_components   = k;
  config.ncv            = 20;
  config.tolerance      = std::is_same_v<ValueType, float> ? ValueType(1e-5) : ValueType(1e-10);
  config.max_iterations = 120;
  config.seed           = 2027;
  config.use_mgs2_orthogonalization = use_mgs2;

  auto S  = raft::make_device_vector<ValueType, uint32_t>(handle, k);
  auto U  = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, m, k);
  auto Vt = raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(handle, k, n);
  sparse_lanczos_svd(handle, config, csr_matrix, S.view(), U.view(), Vt.view());

  std::vector<ValueType> h_S(k);
  std::vector<ValueType> h_U(static_cast<std::size_t>(m) * k);
  std::vector<ValueType> h_Vt(static_cast<std::size_t>(k) * n);
  raft::update_host(h_S.data(), S.data_handle(), k, stream);
  raft::update_host(h_U.data(), U.data_handle(), h_U.size(), stream);
  raft::update_host(h_Vt.data(), Vt.data_handle(), h_Vt.size(), stream);
  resource::sync_stream(handle, stream);

  double sv_tol       = std::is_same_v<ValueType, float> ? 2e-3 : 1e-8;
  double orthog_tol   = std::is_same_v<ValueType, float> ? 2e-3 : 1e-10;
  double residual_tol = std::is_same_v<ValueType, float> ? 2e-2 : 1e-5;
  for (int i = 0; i < k; ++i) {
    ASSERT_NEAR(static_cast<double>(h_S[i]), expected_s[i], sv_tol);
  }
  check_orthonormal_columns(h_U, m, k, orthog_tol);
  check_orthonormal_vt_rows(h_Vt, k, n, orthog_tol);
  check_sparse_residuals(h_indptr, h_indices, h_values, h_S, h_U, h_Vt, m, n, k, residual_tol);
}

TEST(LanczosHardeningF, RotatedClusteredSpectrum) { run_rotated_block_hardening_case<float>(); }

TEST(LanczosHardeningF, RotatedClusteredSpectrumMgs2)
{
  run_rotated_block_hardening_case<float>(true);
}

TEST(LanczosHardeningD, RotatedClusteredSpectrum) { run_rotated_block_hardening_case<double>(); }

}  // namespace raft::sparse::solver
