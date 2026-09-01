/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/r2_score.cuh>
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
struct R2_scoreInputs {
  T tolerance;
  int nrows;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const R2_scoreInputs<T>& dims)
{
  return os;
}

template <typename T>
class R2_scoreTest : public ::testing::TestWithParam<R2_scoreInputs<T>> {
 protected:
  R2_scoreTest() : stream(resource::get_cuda_stream(handle).get()) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<R2_scoreInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    rmm::device_uvector<T> y(params.nrows, stream);
    rmm::device_uvector<T> y_hat(params.nrows, stream);
    uniform(handle, r, y.data(), params.nrows, (T)-1.0, (T)1.0);
    uniform(handle, r, y_hat.data(), params.nrows, (T)-1.0, (T)1.0);

    actualVal   = r2_score(handle,
                         raft::make_device_vector_view<const T>(y.data(), params.nrows),
                         raft::make_device_vector_view<const T>(y_hat.data(), params.nrows));
    expectedVal = T(0);
    std::vector<T> h_y(params.nrows, 0);
    std::vector<T> h_y_hat(params.nrows, 0);
    raft::update_host(h_y.data(), y.data(), params.nrows, stream);
    raft::update_host(h_y_hat.data(), y_hat.data(), params.nrows, stream);
    T mean = T(0);
    for (int i = 0; i < params.nrows; ++i) {
      mean += h_y[i];
    }
    mean /= params.nrows;

    std::vector<T> sse_arr(params.nrows, 0);
    std::vector<T> ssto_arr(params.nrows, 0);
    T sse  = T(0);
    T ssto = T(0);
    for (int i = 0; i < params.nrows; ++i) {
      sse += (h_y[i] - h_y_hat[i]) * (h_y[i] - h_y_hat[i]);
      ssto += (h_y[i] - mean) * (h_y[i] - mean);
    }
    expectedVal = 1.0 - sse / ssto;
    raft::interruptible::synchronize(stream);
  }

 protected:
  R2_scoreInputs<T> params;
  raft::resources handle;
  cudaStream_t stream = 0;
  T expectedVal, actualVal;
};

const std::vector<R2_scoreInputs<float>> inputsf = {
  {0.001f, 30, 1234ULL}, {0.001f, 100, 1234ULL}, {0.001f, 1000, 1234ULL}};
typedef R2_scoreTest<float> R2_scoreTestF;
TEST_P(R2_scoreTestF, Result)
{
  auto eq = raft::CompareApprox<float>(params.tolerance);
  ASSERT_TRUE(match(expectedVal, actualVal, eq));
}
INSTANTIATE_TEST_CASE_P(R2_scoreTests, R2_scoreTestF, ::testing::ValuesIn(inputsf));

const std::vector<R2_scoreInputs<double>> inputsd = {
  {0.001, 30, 1234ULL}, {0.001, 100, 1234ULL}, {0.001, 1000, 1234ULL}};
typedef R2_scoreTest<double> R2_scoreTestD;
TEST_P(R2_scoreTestD, Result)
{
  auto eq = raft::CompareApprox<double>(params.tolerance);
  ASSERT_TRUE(match(expectedVal, actualVal, eq));
}
INSTANTIATE_TEST_CASE_P(R2_scoreTests, R2_scoreTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
