/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/dry_run_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/rsvd.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>

namespace raft {
namespace linalg {

template <typename T>
struct RsvdInputs {
  T tolerance;
  int n_row;
  int n_col;
  float redundancy;
  T PC_perc;
  T UpS_perc;
  int k;
  int p;
  bool use_bbt;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RsvdInputs<T>& dims)
{
  return os;
}

template <typename T>
class RsvdTest : public ::testing::TestWithParam<RsvdInputs<T>> {
 protected:
  RsvdTest()
    : A(0, stream),
      U(0, stream),
      S(0, stream),
      V(0, stream),
      left_eig_vectors_ref(0, stream),
      right_eig_vectors_ref(0, stream),
      sing_vals_ref(0, stream)
  {
  }

  void SetUp() override
  {
    raft::resources handle;
    stream = resource::get_cuda_stream(handle);

    params = ::testing::TestWithParam<RsvdInputs<T>>::GetParam();
    // rSVD seems to be very sensitive to the random number sequence as well!
    raft::random::RngState r(params.seed, raft::random::GenPC);
    int m = params.n_row, n = params.n_col;
    T eig_svd_tol  = 1.e-7;
    int max_sweeps = 100;

    T mu = 0.0, sigma = 1.0;
    A.resize(m * n, stream);
    if (params.tolerance > 1) {  // Sanity check
      ASSERT(m == 3, "This test only supports mxn=3x2!");
      ASSERT(m * n == 6, "This test only supports mxn=3x2!");
      T data_h[] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
      raft::update_device(A.data(), data_h, m * n, stream);

      T left_eig_vectors_ref_h[]  = {-0.308219, -0.906133, -0.289695};
      T right_eig_vectors_ref_h[] = {-0.638636, -0.769509};
      T sing_vals_ref_h[]         = {7.065283};

      left_eig_vectors_ref.resize(m, stream);
      right_eig_vectors_ref.resize(n, stream);
      sing_vals_ref.resize(1, stream);

      raft::update_device(left_eig_vectors_ref.data(), left_eig_vectors_ref_h, m * 1, stream);
      raft::update_device(right_eig_vectors_ref.data(), right_eig_vectors_ref_h, n * 1, stream);
      raft::update_device(sing_vals_ref.data(), sing_vals_ref_h, 1, stream);

    } else {                                 // Other normal tests
      int n_informative   = int(0.25f * n);  // Informative cols
      int len_informative = m * n_informative;

      int n_redundant   = n - n_informative;  // Redundant cols
      int len_redundant = m * n_redundant;

      normal(handle, r, A.data(), len_informative, mu, sigma);
      RAFT_CUDA_TRY(cudaMemcpyAsync(A.data() + len_informative,
                                    A.data(),
                                    len_redundant * sizeof(T),
                                    cudaMemcpyDeviceToDevice,
                                    stream));
    }
    std::vector<T> A_backup_cpu(m *
                                n);  // Backup A matrix as svdJacobi will destroy the content of A
    raft::update_host(A_backup_cpu.data(), A.data(), m * n, stream);

    if (params.k == 0) {
      params.k = std::max((int)(std::min(m, n) * params.PC_perc), 1);
      params.p = std::max((int)(std::min(m, n) * params.UpS_perc), 1);
    }

    U.resize(m * params.k, stream);
    S.resize(params.k, stream);
    V.resize(n * params.k, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(U.data(), 0, U.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(S.data(), 0, S.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(V.data(), 0, V.size() * sizeof(T), stream));

    auto A_view = raft::make_device_matrix_view<const T, int, raft::col_major>(A.data(), m, n);
    std::optional<raft::device_matrix_view<T, int, raft::col_major>> U_view =
      raft::make_device_matrix_view<T, int, raft::col_major>(U.data(), m, params.k);
    std::optional<raft::device_matrix_view<T, int, raft::col_major>> V_view =
      raft::make_device_matrix_view<T, int, raft::col_major>(V.data(), params.k, n);
    auto S_vec_view = raft::make_device_vector_view(S.data(), params.k);

    // RSVD tests
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.k == 0) {  // Test with PC and upsampling ratio
          if (params.use_bbt) {
            rsvd_perc_symmetric(
              h, A_view, S_vec_view, params.PC_perc, params.UpS_perc, U_view, V_view);
          } else {
            rsvd_perc(h, A_view, S_vec_view, params.PC_perc, params.UpS_perc, U_view, V_view);
          }
        } else {  // Test with directly given fixed rank
          if (params.use_bbt) {
            rsvd_fixed_rank_symmetric_jacobi(
              h, A_view, S_vec_view, params.p, eig_svd_tol, max_sweeps, U_view, V_view);
          } else {
            rsvd_fixed_rank_jacobi(
              h, A_view, S_vec_view, params.p, eig_svd_tol, max_sweeps, U_view, V_view);
          }
        }
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      sizeof(int));
    raft::update_device(A.data(), A_backup_cpu.data(), m * n, stream);
  }

 protected:
  cudaStream_t stream = 0;
  RsvdInputs<T> params;
  rmm::device_uvector<T> A, U, S, V, left_eig_vectors_ref, right_eig_vectors_ref, sing_vals_ref;
};

const std::vector<RsvdInputs<float>> inputs_fx = {
  // Test with ratios
  {0.20f, 256, 256, 0.25f, 0.2f, 0.05f, 0, 0, true, 4321ULL},   // Square + BBT
  {0.20f, 2048, 256, 0.25f, 0.2f, 0.05f, 0, 0, true, 4321ULL},  // Tall + BBT

  {0.20f, 256, 256, 0.25f, 0.2f, 0.05f, 0, 0, false, 4321ULL},   // Square + non-BBT
  {0.20f, 2048, 256, 0.25f, 0.2f, 0.05f, 0, 0, false, 4321ULL},  // Tall + non-BBT

  {0.20f, 2048, 2048, 0.25f, 0.2f, 0.05f, 0, 0, true, 4321ULL},   // Square + BBT
  {0.60f, 16384, 2048, 0.25f, 0.2f, 0.05f, 0, 0, true, 4321ULL},  // Tall + BBT

  {0.20f, 2048, 2048, 0.25f, 0.2f, 0.05f, 0, 0, false, 4321ULL},  // Square + non-BBT
  {0.60f, 16384, 2048, 0.25f, 0.2f, 0.05f, 0, 0, false, 4321ULL}  // Tall + non-BBT

  ,                                                              // Test with fixed ranks
  {0.10f, 256, 256, 0.25f, 0.0f, 0.0f, 100, 5, true, 4321ULL},   // Square + BBT
  {0.12f, 2048, 256, 0.25f, 0.0f, 0.0f, 100, 5, true, 4321ULL},  // Tall + BBT

  {0.10f, 256, 256, 0.25f, 0.0f, 0.0f, 100, 5, false, 4321ULL},   // Square + non-BBT
  {0.12f, 2048, 256, 0.25f, 0.0f, 0.0f, 100, 5, false, 4321ULL},  // Tall + non-BBT

  {0.60f, 2048, 2048, 0.25f, 0.0f, 0.0f, 100, 5, true, 4321ULL},   // Square + BBT
  {1.00f, 16384, 2048, 0.25f, 0.0f, 0.0f, 100, 5, true, 4321ULL},  // Tall + BBT

  {0.60f, 2048, 2048, 0.25f, 0.0f, 0.0f, 100, 5, false, 4321ULL},  // Square + non-BBT
  {1.00f, 16384, 2048, 0.25f, 0.0f, 0.0f, 100, 5, false, 4321ULL}  // Tall + non-BBT
};

const std::vector<RsvdInputs<double>> inputs_dx = {
  // Test with ratios
  {0.20, 256, 256, 0.25f, 0.2, 0.05, 0, 0, true, 4321ULL},     // Square + BBT
  {0.20, 2048, 256, 0.25f, 0.2, 0.05, 0, 0, true, 4321ULL},    // Tall + BBT
  {0.20, 256, 256, 0.25f, 0.2, 0.05, 0, 0, false, 4321ULL},    // Square + non-BBT
  {0.20, 2048, 256, 0.25f, 0.2, 0.05, 0, 0, false, 4321ULL},   // Tall + non-BBT
  {0.20, 2048, 2048, 0.25f, 0.2, 0.05, 0, 0, true, 4321ULL},   // Square + BBT
  {0.60, 16384, 2048, 0.25f, 0.2, 0.05, 0, 0, true, 4321ULL},  // Tall + BBT
  {0.20, 2048, 2048, 0.25f, 0.2, 0.05, 0, 0, false, 4321ULL},  // Square + non-BBT
  {0.60, 16384, 2048, 0.25f, 0.2, 0.05, 0, 0, false, 4321ULL}  // Tall + non-BBT

  ,                                                             // Test with fixed ranks
  {0.10, 256, 256, 0.25f, 0.0, 0.0, 100, 5, true, 4321ULL},     // Square + BBT
  {0.12, 2048, 256, 0.25f, 0.0, 0.0, 100, 5, true, 4321ULL},    // Tall + BBT
  {0.10, 256, 256, 0.25f, 0.0, 0.0, 100, 5, false, 4321ULL},    // Square + non-BBT
  {0.12, 2048, 256, 0.25f, 0.0, 0.0, 100, 5, false, 4321ULL},   // Tall + non-BBT
  {0.60, 2048, 2048, 0.25f, 0.0, 0.0, 100, 5, true, 4321ULL},   // Square + BBT
  {1.00, 16384, 2048, 0.25f, 0.0, 0.0, 100, 5, true, 4321ULL},  // Tall + BBT
  {0.60, 2048, 2048, 0.25f, 0.0, 0.0, 100, 5, false, 4321ULL},  // Square + non-BBT
  {1.00, 16384, 2048, 0.25f, 0.0, 0.0, 100, 5, false, 4321ULL}  // Tall + non-BBT
};

const std::vector<RsvdInputs<float>> sanity_inputs_fx = {
  {100000000000000000.0f, 3, 2, 0.25f, 0.2f, 0.05f, 0, 0, true, 4321ULL},
  {100000000000000000.0f, 3, 2, 0.25f, 0.0f, 0.0f, 1, 1, true, 4321ULL},
  {100000000000000000.0f, 3, 2, 0.25f, 0.2f, 0.05f, 0, 0, false, 4321ULL},
  {100000000000000000.0f, 3, 2, 0.25f, 0.0f, 0.0f, 1, 1, false, 4321ULL}};

const std::vector<RsvdInputs<double>> sanity_inputs_dx = {
  {100000000000000000.0, 3, 2, 0.25f, 0.2, 0.05, 0, 0, true, 4321ULL},
  {100000000000000000.0, 3, 2, 0.25f, 0.0, 0.0, 1, 1, true, 4321ULL},
  {100000000000000000.0, 3, 2, 0.25f, 0.2, 0.05, 0, 0, false, 4321ULL},
  {100000000000000000.0, 3, 2, 0.25f, 0.0, 0.0, 1, 1, false, 4321ULL}};

typedef RsvdTest<float> RsvdSanityCheckValF;
TEST_P(RsvdSanityCheckValF, Result)
{
  ASSERT_TRUE(devArrMatch(
    sing_vals_ref.data(), S.data(), params.k, raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef RsvdTest<double> RsvdSanityCheckValD;
TEST_P(RsvdSanityCheckValD, Result)
{
  ASSERT_TRUE(devArrMatch(
    sing_vals_ref.data(), S.data(), params.k, raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef RsvdTest<float> RsvdSanityCheckLeftVecF;
TEST_P(RsvdSanityCheckLeftVecF, Result)
{
  ASSERT_TRUE(devArrMatch(left_eig_vectors_ref.data(),
                          U.data(),
                          params.n_row * params.k,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef RsvdTest<double> RsvdSanityCheckLeftVecD;
TEST_P(RsvdSanityCheckLeftVecD, Result)
{
  ASSERT_TRUE(devArrMatch(left_eig_vectors_ref.data(),
                          U.data(),
                          params.n_row * params.k,
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef RsvdTest<float> RsvdSanityCheckRightVecF;
TEST_P(RsvdSanityCheckRightVecF, Result)
{
  ASSERT_TRUE(devArrMatch(right_eig_vectors_ref.data(),
                          V.data(),
                          params.n_col * params.k,
                          raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef RsvdTest<double> RsvdSanityCheckRightVecD;
TEST_P(RsvdSanityCheckRightVecD, Result)
{
  ASSERT_TRUE(devArrMatch(right_eig_vectors_ref.data(),
                          V.data(),
                          params.n_col * params.k,
                          raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef RsvdTest<float> RsvdTestSquareMatrixNormF;
TEST_P(RsvdTestSquareMatrixNormF, Result)
{
  raft::resources handle;

  ASSERT_TRUE(raft::linalg::evaluateSVDByL2Norm(handle,
                                                A.data(),
                                                U.data(),
                                                S.data(),
                                                V.data(),
                                                params.n_row,
                                                params.n_col,
                                                params.k,
                                                4 * params.tolerance,
                                                resource::get_cuda_stream(handle)));
}

typedef RsvdTest<double> RsvdTestSquareMatrixNormD;
TEST_P(RsvdTestSquareMatrixNormD, Result)
{
  raft::resources handle;

  ASSERT_TRUE(raft::linalg::evaluateSVDByL2Norm(handle,
                                                A.data(),
                                                U.data(),
                                                S.data(),
                                                V.data(),
                                                params.n_row,
                                                params.n_col,
                                                params.k,
                                                4 * params.tolerance,
                                                resource::get_cuda_stream(handle)));
}

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckValF, ::testing::ValuesIn(sanity_inputs_fx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckValD, ::testing::ValuesIn(sanity_inputs_dx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckLeftVecF, ::testing::ValuesIn(sanity_inputs_fx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckLeftVecD, ::testing::ValuesIn(sanity_inputs_dx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckRightVecF, ::testing::ValuesIn(sanity_inputs_fx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdSanityCheckRightVecD, ::testing::ValuesIn(sanity_inputs_dx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdTestSquareMatrixNormF, ::testing::ValuesIn(inputs_fx));

INSTANTIATE_TEST_CASE_P(RsvdTests, RsvdTestSquareMatrixNormD, ::testing::ValuesIn(inputs_dx));

// ===================================================================
// Dry-run tests for RSVD public API functions
// ===================================================================

TEST(RsvdDryRun, FixedRankQRWithBothVectors)
{
  raft::resources res;

  constexpr int n_rows = 256;
  constexpr int n_cols = 128;
  constexpr int k      = 50;
  constexpr int p      = 10;

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto U     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, k);
  auto V     = raft::make_device_matrix<float, int, raft::col_major>(res, k, n_cols);

  // Run rsvd_fixed_rank in dry-run mode (QR, no BBT, no Jacobi, both U and V)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_fixed_rank(
      handle, raft::make_const_mdspan(M.view()), S_vec.view(), p, U.view(), V.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, FixedRankSymmetricWithBothVectors)
{
  raft::resources res;

  constexpr int n_rows = 256;
  constexpr int n_cols = 128;
  constexpr int k      = 50;
  constexpr int p      = 10;

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto U     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, k);
  auto V     = raft::make_device_matrix<float, int, raft::col_major>(res, k, n_cols);

  // Run rsvd_fixed_rank_symmetric in dry-run mode (QR, with BBT, no Jacobi, both U and V)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_fixed_rank_symmetric(
      handle, raft::make_const_mdspan(M.view()), S_vec.view(), p, U.view(), V.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, FixedRankJacobiWithBothVectors)
{
  raft::resources res;

  constexpr int n_rows     = 256;
  constexpr int n_cols     = 128;
  constexpr int k          = 50;
  constexpr int p          = 10;
  constexpr float tol      = 1e-7f;
  constexpr int max_sweeps = 100;

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto U     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, k);
  auto V     = raft::make_device_matrix<float, int, raft::col_major>(res, k, n_cols);

  // Run rsvd_fixed_rank_jacobi in dry-run mode (QR, no BBT, with Jacobi, both U and V)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_fixed_rank_jacobi(handle,
                                         raft::make_const_mdspan(M.view()),
                                         S_vec.view(),
                                         p,
                                         tol,
                                         max_sweeps,
                                         U.view(),
                                         V.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, FixedRankSymmetricJacobiWithBothVectors)
{
  raft::resources res;

  constexpr int n_rows     = 256;
  constexpr int n_cols     = 128;
  constexpr int k          = 50;
  constexpr int p          = 10;
  constexpr float tol      = 1e-7f;
  constexpr int max_sweeps = 100;

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto U     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, k);
  auto V     = raft::make_device_matrix<float, int, raft::col_major>(res, k, n_cols);

  // Run rsvd_fixed_rank_symmetric_jacobi in dry-run mode (QR, with BBT, with Jacobi, both U and V)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_fixed_rank_symmetric_jacobi(handle,
                                                   raft::make_const_mdspan(M.view()),
                                                   S_vec.view(),
                                                   p,
                                                   tol,
                                                   max_sweeps,
                                                   U.view(),
                                                   V.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, FixedRankWithOnlyU)
{
  raft::resources res;

  constexpr int n_rows = 256;
  constexpr int n_cols = 128;
  constexpr int k      = 50;
  constexpr int p      = 10;

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto U     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, k);

  // Run rsvd_fixed_rank in dry-run mode (only U, no V)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_fixed_rank(
      handle, raft::make_const_mdspan(M.view()), S_vec.view(), p, U.view(), std::nullopt);
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, FixedRankWithOnlyV)
{
  raft::resources res;

  constexpr int n_rows = 256;
  constexpr int n_cols = 128;
  constexpr int k      = 50;
  constexpr int p      = 10;

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto V     = raft::make_device_matrix<float, int, raft::col_major>(res, k, n_cols);

  // Run rsvd_fixed_rank in dry-run mode (only V, no U)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_fixed_rank(
      handle, raft::make_const_mdspan(M.view()), S_vec.view(), p, std::nullopt, V.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, FixedRankWithNoVectors)
{
  raft::resources res;

  constexpr int n_rows = 256;
  constexpr int n_cols = 128;
  constexpr int k      = 50;
  constexpr int p      = 10;

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);

  // Run rsvd_fixed_rank in dry-run mode (no U, no V - only singular values)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_fixed_rank(
      handle, raft::make_const_mdspan(M.view()), S_vec.view(), p, std::nullopt, std::nullopt);
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, PercWithBothVectors)
{
  raft::resources res;

  constexpr int n_rows     = 256;
  constexpr int n_cols     = 128;
  constexpr float PC_perc  = 0.2f;
  constexpr float UpS_perc = 0.05f;
  constexpr int k          = static_cast<int>(std::min(n_rows, n_cols) * PC_perc);

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto U     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, k);
  auto V     = raft::make_device_matrix<float, int, raft::col_major>(res, k, n_cols);

  // Run rsvd_perc in dry-run mode (percentage-based, QR, no BBT, no Jacobi, both U and V)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_perc(handle,
                            raft::make_const_mdspan(M.view()),
                            S_vec.view(),
                            PC_perc,
                            UpS_perc,
                            U.view(),
                            V.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, PercSymmetricJacobiWithBothVectors)
{
  raft::resources res;

  constexpr int n_rows     = 256;
  constexpr int n_cols     = 128;
  constexpr float PC_perc  = 0.2f;
  constexpr float UpS_perc = 0.05f;
  constexpr float tol      = 1e-7f;
  constexpr int max_sweeps = 100;
  constexpr int k          = static_cast<int>(std::min(n_rows, n_cols) * PC_perc);

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto U     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, k);
  auto V     = raft::make_device_matrix<float, int, raft::col_major>(res, k, n_cols);

  // Run rsvd_perc_symmetric_jacobi in dry-run mode (percentage-based, QR, with BBT, with Jacobi,
  // both U and V)
  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_perc_symmetric_jacobi(handle,
                                             raft::make_const_mdspan(M.view()),
                                             S_vec.view(),
                                             PC_perc,
                                             UpS_perc,
                                             tol,
                                             max_sweeps,
                                             U.view(),
                                             V.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

TEST(RsvdDryRun, TallMatrix)
{
  raft::resources res;

  constexpr int n_rows = 512;
  constexpr int n_cols = 128;
  constexpr int k      = 50;
  constexpr int p      = 10;

  // Pre-allocate input/output buffers (outside dry-run)
  auto M     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, n_cols);
  auto S_vec = raft::make_device_vector<float, int>(res, k);
  auto U     = raft::make_device_matrix<float, int, raft::col_major>(res, n_rows, k);
  auto V     = raft::make_device_matrix<float, int, raft::col_major>(res, k, n_cols);

  // Run rsvd_fixed_rank_jacobi in dry-run mode on a tall matrix
  constexpr float tol      = 1e-7f;
  constexpr int max_sweeps = 100;
  auto stats               = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::linalg::rsvd_fixed_rank_jacobi(handle,
                                         raft::make_const_mdspan(M.view()),
                                         S_vec.view(),
                                         p,
                                         tol,
                                         max_sweeps,
                                         U.view(),
                                         V.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device memory allocation";
}

}  // end namespace linalg
}  // end namespace raft
