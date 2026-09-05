/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/accuracy.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <optional>
#include <vector>

namespace raft {
namespace stats {

template <typename T>
struct AccuracyInputs {
  T tolerance;
  int nrows;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const AccuracyInputs<T>& dims)
{
  return os;
}

template <typename T>
class AccuracyTest : public ::testing::TestWithParam<AccuracyInputs<T>> {
 protected:
  AccuracyTest() : stream(resource::get_cuda_stream(handle).get()) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<AccuracyInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    rmm::device_uvector<int> predictions(params.nrows, stream);
    rmm::device_uvector<int> ref_predictions(params.nrows, stream);
    uniformInt(handle, r, predictions.data(), params.nrows, 0, 10);
    uniformInt(handle, r, ref_predictions.data(), params.nrows, 0, 10);

    actualVal =
      accuracy(handle,
               raft::make_device_vector_view<const int>(predictions.data(), params.nrows),
               raft::make_device_vector_view<const int>(ref_predictions.data(), params.nrows));
    expectedVal = T(0);
    std::vector<int> h_predictions(params.nrows, 0);
    std::vector<int> h_ref_predictions(params.nrows, 0);
    raft::update_host(h_predictions.data(), predictions.data(), params.nrows, stream);
    raft::update_host(h_ref_predictions.data(), ref_predictions.data(), params.nrows, stream);

    unsigned long long correctly_predicted = 0ULL;
    for (int i = 0; i < params.nrows; ++i) {
      correctly_predicted += (h_predictions[i] - h_ref_predictions[i]) == 0;
    }
    expectedVal = correctly_predicted * 1.0f / params.nrows;
    raft::interruptible::synchronize(stream);
  }

 protected:
  AccuracyInputs<T> params;
  raft::resources handle;
  cudaStream_t stream = 0;
  T expectedVal, actualVal;
};

const std::vector<AccuracyInputs<float>> inputsf = {
  {0.001f, 30, 1234ULL}, {0.001f, 100, 1234ULL}, {0.001f, 1000, 1234ULL}};
typedef AccuracyTest<float> AccuracyTestF;
TEST_P(AccuracyTestF, Result)
{
  auto eq = raft::CompareApprox<float>(params.tolerance);
  ASSERT_TRUE(match(expectedVal, actualVal, eq));
}
INSTANTIATE_TEST_CASE_P(AccuracyTests, AccuracyTestF, ::testing::ValuesIn(inputsf));

const std::vector<AccuracyInputs<double>> inputsd = {
  {0.001, 30, 1234ULL}, {0.001, 100, 1234ULL}, {0.001, 1000, 1234ULL}};
typedef AccuracyTest<double> AccuracyTestD;
TEST_P(AccuracyTestD, Result)
{
  auto eq = raft::CompareApprox<double>(params.tolerance);
  ASSERT_TRUE(match(expectedVal, actualVal, eq));
}
INSTANTIATE_TEST_CASE_P(AccuracyTests, AccuracyTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
