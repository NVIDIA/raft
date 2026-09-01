/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T>
struct EigSelInputs {
  T tolerance;
  int len;
  int n;
  int n_eigen_vals;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const EigSelInputs<T>& dims)
{
  return os;
}

template <typename T>
class EigSelTest : public ::testing::TestWithParam<EigSelInputs<T>> {
 public:
  EigSelTest()
    : params(::testing::TestWithParam<EigSelInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      cov_matrix(params.len, stream),
      eig_vectors(params.n_eigen_vals * params.n, stream),
      eig_vectors_ref(params.n_eigen_vals * params.n, stream),
      eig_vals(params.n, stream),
      eig_vals_ref(params.n, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len = params.len;

    ///@todo: Generate a random symmetric matrix
    T cov_matrix_h[] = {
      1.0, 0.9, 0.81, 0.729, 0.9, 1.0, 0.9, 0.81, 0.81, 0.9, 1.0, 0.9, 0.729, 0.81, 0.9, 1.0};
    ASSERT(len == 16, "This test only works with 4x4 matrices!");
    raft::update_device(cov_matrix.data(), cov_matrix_h, len, stream);

    T eig_vectors_ref_h[] = {-0.5123,
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
    T eig_vals_ref_h[]    = {0.1024, 0.3096, 3.5266, 0.0};

    raft::update_device(
      eig_vectors_ref.data(), eig_vectors_ref_h, params.n_eigen_vals * params.n, stream);
    raft::update_device(eig_vals_ref.data(), eig_vals_ref_h, params.n_eigen_vals, stream);

    auto cov_matrix_view = raft::make_device_matrix_view<const T, std::uint32_t, raft::col_major>(
      cov_matrix.data(), params.n, params.n);
    auto eig_vectors_view = raft::make_device_matrix_view<T, std::uint32_t, raft::col_major>(
      eig_vectors.data(), params.n_eigen_vals, params.n);
    auto eig_vals_view =
      raft::make_device_vector_view<T, std::uint32_t>(eig_vals.data(), params.n_eigen_vals);

    raft::linalg::eig_dc_selective(handle,
                                   cov_matrix_view,
                                   eig_vectors_view,
                                   eig_vals_view,
                                   static_cast<std::size_t>(params.n_eigen_vals),
                                   EigVecMemUsage::OVERWRITE_INPUT);
    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  EigSelInputs<T> params;
  rmm::device_uvector<T> cov_matrix;
  rmm::device_uvector<T> eig_vectors;
  rmm::device_uvector<T> eig_vectors_ref;
  rmm::device_uvector<T> eig_vals;
  rmm::device_uvector<T> eig_vals_ref;
};

const std::vector<EigSelInputs<float>> inputsf2 = {{0.001f, 4 * 4, 4, 3}};

const std::vector<EigSelInputs<double>> inputsd2 = {{0.001, 4 * 4, 4, 3}};

typedef EigSelTest<float> EigSelTestValF;
TEST_P(EigSelTestValF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vals_ref.data(),
                                eig_vals.data(),
                                params.n_eigen_vals,
                                raft::CompareApproxAbs<float>(params.tolerance),
                                stream));
}

typedef EigSelTest<double> EigSelTestValD;
TEST_P(EigSelTestValD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vals_ref.data(),
                                eig_vals.data(),
                                params.n_eigen_vals,
                                raft::CompareApproxAbs<double>(params.tolerance),
                                stream));
}

typedef EigSelTest<float> EigSelTestVecF;
TEST_P(EigSelTestVecF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vectors_ref.data(),
                                eig_vectors.data(),
                                params.n_eigen_vals * params.n,
                                raft::CompareApproxAbs<float>(params.tolerance),
                                stream));
}

typedef EigSelTest<double> EigSelTestVecD;
TEST_P(EigSelTestVecD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(eig_vectors_ref.data(),
                                eig_vectors.data(),
                                params.n_eigen_vals * params.n,
                                raft::CompareApproxAbs<double>(params.tolerance),
                                stream));
}

INSTANTIATE_TEST_SUITE_P(EigSelTest, EigSelTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(EigSelTest, EigSelTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_SUITE_P(EigSelTest, EigSelTestVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(EigSelTest, EigSelTestVecD, ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
