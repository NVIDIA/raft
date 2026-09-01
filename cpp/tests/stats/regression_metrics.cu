/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/regression_metrics.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <optional>
#include <vector>

namespace raft {
namespace stats {

template <typename T>
struct RegressionInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RegressionInputs<T>& dims)
{
  return os;
}

template <typename T>
void naive_reg_metrics(std::vector<T>& predictions,
                       std::vector<T>& ref_predictions,
                       double& mean_abs_error,
                       double& mean_squared_error,
                       double& median_abs_error)
{
  auto len        = predictions.size();
  double abs_diff = 0;
  double sq_diff  = 0;
  std::vector<double> abs_errors(len);
  for (std::size_t i = 0; i < len; ++i) {
    auto diff = predictions[i] - ref_predictions[i];
    abs_diff += abs(diff);
    sq_diff += diff * diff;
    abs_errors[i] = abs(diff);
  }
  mean_abs_error     = abs_diff / len;
  mean_squared_error = sq_diff / len;

  std::sort(abs_errors.begin(), abs_errors.end());
  auto middle = len / 2;
  if (len % 2 == 1) {
    median_abs_error = abs_errors[middle];
  } else {
    median_abs_error = (abs_errors[middle] + abs_errors[middle - 1]) / 2;
  }
}

template <typename T>
class RegressionTest : public ::testing::TestWithParam<RegressionInputs<T>> {
 protected:
  RegressionTest() : stream(resource::get_cuda_stream(handle).get()) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<RegressionInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    rmm::device_uvector<T> predictions(params.len, stream);
    rmm::device_uvector<T> ref_predictions(params.len, stream);
    uniform(handle, r, predictions.data(), params.len, T(-10.0), T(10.0));
    uniform(handle, r, ref_predictions.data(), params.len, T(-10.0), T(10.0));

    regression_metrics(handle,
                       raft::make_device_vector_view<const T>(predictions.data(), params.len),
                       raft::make_device_vector_view<const T>(ref_predictions.data(), params.len),
                       raft::make_host_scalar_view(&mean_abs_error),
                       raft::make_host_scalar_view(&mean_squared_error),
                       raft::make_host_scalar_view(&median_abs_error));
    std::vector<T> h_predictions(params.len, 0);
    std::vector<T> h_ref_predictions(params.len, 0);
    raft::update_host(h_predictions.data(), predictions.data(), params.len, stream);
    raft::update_host(h_ref_predictions.data(), ref_predictions.data(), params.len, stream);

    naive_reg_metrics(h_predictions,
                      h_ref_predictions,
                      ref_mean_abs_error,
                      ref_mean_squared_error,
                      ref_median_abs_error);
    raft::interruptible::synchronize(stream);
  }

 protected:
  raft::resources handle;
  RegressionInputs<T> params;
  cudaStream_t stream           = 0;
  double mean_abs_error         = 0;
  double mean_squared_error     = 0;
  double median_abs_error       = 0;
  double ref_mean_abs_error     = 0;
  double ref_mean_squared_error = 0;
  double ref_median_abs_error   = 0;
};

const std::vector<RegressionInputs<float>> inputsf = {
  {0.001f, 30, 1234ULL}, {0.001f, 100, 1234ULL}, {0.001f, 4000, 1234ULL}};
typedef RegressionTest<float> RegressionTestF;
TEST_P(RegressionTestF, Result)
{
  auto eq = raft::CompareApprox<float>(params.tolerance);
  ASSERT_TRUE(match(ref_mean_abs_error, mean_abs_error, eq));
  ASSERT_TRUE(match(ref_mean_squared_error, mean_squared_error, eq));
  ASSERT_TRUE(match(ref_median_abs_error, median_abs_error, eq));
}
INSTANTIATE_TEST_CASE_P(RegressionTests, RegressionTestF, ::testing::ValuesIn(inputsf));

const std::vector<RegressionInputs<double>> inputsd = {
  {0.001, 30, 1234ULL}, {0.001, 100, 1234ULL}, {0.001, 4000, 1234ULL}};
typedef RegressionTest<double> RegressionTestD;
TEST_P(RegressionTestD, Result)
{
  auto eq = raft::CompareApprox<double>(params.tolerance);
  ASSERT_TRUE(match(ref_mean_abs_error, mean_abs_error, eq));
  ASSERT_TRUE(match(ref_mean_squared_error, mean_squared_error, eq));
  ASSERT_TRUE(match(ref_median_abs_error, median_abs_error, eq));
}
INSTANTIATE_TEST_CASE_P(RegressionTests, RegressionTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
