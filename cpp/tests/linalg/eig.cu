/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T>
struct EigInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  unsigned long long int seed;
  int n;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const EigInputs<T>& dims)
{
  return os;
}

template <typename T>
class EigTest : public ::testing::TestWithParam<EigInputs<T>> {
 public:
  EigTest()
    : params(::testing::TestWithParam<EigInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      cov_matrix(params.len, stream),
      eig_vectors(params.len, stream),
      eig_vectors_jacobi(params.len, stream),
      eig_vectors_ref(params.len, stream),
      eig_vals(params.n_col, stream),
      eig_vals_jacobi(params.n_col, stream),
      eig_vals_ref(params.n_col, stream),
      cov_matrix_large(params.n * params.n, stream),
      eig_vectors_large(params.n * params.n, stream),
      eig_vectors_jacobi_large(params.n * params.n, stream),
      eig_vals_large(params.n, stream),
      eig_vals_jacobi_large(params.n, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int len = params.len;

    T cov_matrix_h[] = {
      1.0, 0.9, 0.81, 0.729, 0.9, 1.0, 0.9, 0.81, 0.81, 0.9, 1.0, 0.9, 0.729, 0.81, 0.9, 1.0};
    ASSERT(len == 16, "This test only works with 4x4 matrices!");
    raft::update_device(cov_matrix.data(), cov_matrix_h, len, stream);

    T eig_vectors_ref_h[] = {0.2790,
                             -0.6498,
                             0.6498,
                             -0.2789,
                             -0.5123,
                             0.4874,
                             0.4874,
                             -0.5123,
                             0.6498,
                             0.2789,
                             -0.2789,
                             -0.6498,
                             0.4874,
                             0.5123,
                             0.5123,
                             0.4874};
    T eig_vals_ref_h[]    = {0.0614, 0.1024, 0.3096, 3.5266};

    raft::update_device(eig_vectors_ref.data(), eig_vectors_ref_h, len, stream);
    raft::update_device(eig_vals_ref.data(), eig_vals_ref_h, params.n_col, stream);

    auto cov_matrix_view = raft::make_device_matrix_view<const T, std::uint32_t, raft::col_major>(
      cov_matrix.data(), params.n_row, params.n_col);
    auto eig_vectors_view = raft::make_device_matrix_view<T, std::uint32_t, raft::col_major>(
      eig_vectors.data(), params.n_row, params.n_col);
    auto eig_vals_view =
      raft::make_device_vector_view<T, std::uint32_t>(eig_vals.data(), params.n_row);

    auto eig_vectors_jacobi_view = raft::make_device_matrix_view<T, std::uint32_t, raft::col_major>(
      eig_vectors_jacobi.data(), params.n_row, params.n_col);
    auto eig_vals_jacobi_view =
      raft::make_device_vector_view<T, std::uint32_t>(eig_vals_jacobi.data(), params.n_row);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        eig_dc(h, cov_matrix_view, eig_vectors_view, eig_vals_view);
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      sizeof(int));

    T tol      = 1.e-7;
    int sweeps = 15;
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        eig_jacobi(h, cov_matrix_view, eig_vectors_jacobi_view, eig_vals_jacobi_view, tol, sweeps);
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      sizeof(int));

    // test code for comparing two methods
    len = params.n * params.n;

    uniform(handle, r, cov_matrix_large.data(), len, T(-1.0), T(1.0));

    auto cov_matrix_large_view =
      raft::make_device_matrix_view<const T, std::uint32_t, raft::col_major>(
        cov_matrix_large.data(), params.n, params.n);
    auto eig_vectors_large_view = raft::make_device_matrix_view<T, std::uint32_t, raft::col_major>(
      eig_vectors_large.data(), params.n, params.n);
    auto eig_vals_large_view =
      raft::make_device_vector_view<T, std::uint32_t>(eig_vals_large.data(), params.n);

    auto eig_vectors_jacobi_large_view =
      raft::make_device_matrix_view<T, std::uint32_t, raft::col_major>(
        eig_vectors_jacobi_large.data(), params.n, params.n);
    auto eig_vals_jacobi_large_view =
      raft::make_device_vector_view<T, std::uint32_t>(eig_vals_jacobi_large.data(), params.n);

    eig_dc(handle, cov_matrix_large_view, eig_vectors_large_view, eig_vals_large_view);
    eig_jacobi(handle,
               cov_matrix_large_view,
               eig_vectors_jacobi_large_view,
               eig_vals_jacobi_large_view,
               tol,
               sweeps);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  EigInputs<T> params;

  rmm::device_uvector<T> cov_matrix, eig_vectors, eig_vectors_jacobi, eig_vectors_ref, eig_vals,
    eig_vals_jacobi, eig_vals_ref;

  rmm::device_uvector<T> cov_matrix_large, eig_vectors_large, eig_vectors_jacobi_large,
    eig_vals_large, eig_vals_jacobi_large;
};

TEST(Raft, EigStream)
{
  // Separate test to check eig_dc stream workaround for CUDA 12+
  raft::random::RngState r(1234ULL);
  raft::resources handle;
  auto n_rows = 5000;
  auto cov_matrix_stream =
    raft::make_device_matrix<float, std::uint32_t, raft::col_major>(handle, n_rows, n_rows);
  auto eig_vectors_stream =
    raft::make_device_matrix<float, std::uint32_t, raft::col_major>(handle, n_rows, n_rows);
  auto eig_vals_stream = raft::make_device_vector<float, std::uint32_t>(handle, n_rows);
  uniform(handle, r, cov_matrix_stream.data_handle(), n_rows * n_rows, float(-1.0), float(1.0));

  raft::execute_with_dry_run_check(
    handle,
    [&](raft::resources const& h) {
      raft::linalg::eig_dc(h,
                           raft::make_const_mdspan(cov_matrix_stream.view()),
                           eig_vectors_stream.view(),
                           eig_vals_stream.view());
    },
    raft::alloc_behavior::ARGUMENT_DRIVEN,
    sizeof(int));
  raft::resource::sync_stream(handle, raft::resource::get_cuda_stream(handle));
}

const std::vector<EigInputs<float>> inputsf2 = {{0.001f, 4 * 4, 4, 4, 1234ULL, 256}};

const std::vector<EigInputs<double>> inputsd2 = {{0.001, 4 * 4, 4, 4, 1234ULL, 256}};

typedef EigTest<float> EigTestValF;
TEST_P(EigTestValF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vals_ref.data(),
                                eig_vals.data(),
                                params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance),
                                stream));
}

typedef EigTest<double> EigTestValD;
TEST_P(EigTestValD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vals_ref.data(),
                                eig_vals.data(),
                                params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance),
                                stream));
}

typedef EigTest<float> EigTestVecF;
TEST_P(EigTestVecF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vectors_ref.data(),
                                eig_vectors.data(),
                                params.len,
                                raft::CompareApproxAbs<float>(params.tolerance),
                                stream));
}

typedef EigTest<double> EigTestVecD;
TEST_P(EigTestVecD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vectors_ref.data(),
                                eig_vectors.data(),
                                params.len,
                                raft::CompareApproxAbs<double>(params.tolerance),
                                stream));
}

typedef EigTest<float> EigTestValJacobiF;
TEST_P(EigTestValJacobiF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vals_ref.data(),
                                eig_vals_jacobi.data(),
                                params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance),
                                stream));
}

typedef EigTest<double> EigTestValJacobiD;
TEST_P(EigTestValJacobiD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vals_ref.data(),
                                eig_vals_jacobi.data(),
                                params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance),
                                stream));
}

typedef EigTest<float> EigTestVecJacobiF;
TEST_P(EigTestVecJacobiF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vectors_ref.data(),
                                eig_vectors_jacobi.data(),
                                params.len,
                                raft::CompareApproxAbs<float>(params.tolerance),
                                stream));
}

typedef EigTest<double> EigTestVecJacobiD;
TEST_P(EigTestVecJacobiD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vectors_ref.data(),
                                eig_vectors_jacobi.data(),
                                params.len,
                                raft::CompareApproxAbs<double>(params.tolerance),
                                stream));
}

typedef EigTest<float> EigTestVecCompareF;
TEST_P(EigTestVecCompareF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vectors_large.data(),
                                eig_vectors_jacobi_large.data(),
                                (params.n * params.n),
                                raft::CompareApproxAbs<float>(params.tolerance),
                                stream));
}

typedef EigTest<double> EigTestVecCompareD;
TEST_P(EigTestVecCompareD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vectors_large.data(),
                                eig_vectors_jacobi_large.data(),
                                (params.n * params.n),
                                raft::CompareApproxAbs<double>(params.tolerance),
                                stream));
}

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestValJacobiF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestValJacobiD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestVecJacobiF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestVecJacobiD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestVecCompareF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(EigTests, EigTestVecCompareD, ::testing::ValuesIn(inputsd2));

}  // namespace linalg
}  // namespace raft
