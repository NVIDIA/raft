/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/dispersion.cuh>
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
struct DispersionInputs {
  T tolerance;
  int dim, clusters;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const DispersionInputs<T>& dims)
{
  return os;
}

template <typename T>
class DispersionTest : public ::testing::TestWithParam<DispersionInputs<T>> {
 protected:
  DispersionTest()
    : stream(resource::get_cuda_stream(handle).get()), exp_mean(0, stream), act_mean(0, stream)
  {
  }

  void SetUp() override
  {
    params = ::testing::TestWithParam<DispersionInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    int len = params.clusters * params.dim;
    rmm::device_uvector<T> data(len, stream);
    rmm::device_uvector<int> counts(params.clusters, stream);
    exp_mean.resize(params.dim, stream);
    act_mean.resize(params.dim, stream);
    uniform(handle, r, data.data(), len, (T)-1.0, (T)1.0);
    uniformInt(handle, r, counts.data(), params.clusters, 1, 100);
    std::vector<int> h_counts(params.clusters, 0);
    raft::update_host(&(h_counts[0]), counts.data(), params.clusters, stream);
    npoints = 0;
    for (const auto& val : h_counts) {
      npoints += val;
    }
    actualVal = cluster_dispersion(
      handle,
      raft::make_device_matrix_view<const T, int>(data.data(), params.clusters, params.dim),
      raft::make_device_vector_view<const int, int>(counts.data(), params.clusters),
      std::make_optional(raft::make_device_vector_view<T, int>(act_mean.data(), params.dim)),
      npoints);
    expectedVal = T(0);
    std::vector<T> h_data(len, T(0));
    raft::update_host(&(h_data[0]), data.data(), len, stream);
    std::vector<T> mean(params.dim, T(0));
    for (int i = 0; i < params.clusters; ++i) {
      for (int j = 0; j < params.dim; ++j) {
        mean[j] += h_data[i * params.dim + j] * T(h_counts[i]);
      }
    }
    for (int i = 0; i < params.dim; ++i) {
      mean[i] /= T(npoints);
    }
    raft::update_device(exp_mean.data(), &(mean[0]), params.dim, stream);
    for (int i = 0; i < params.clusters; ++i) {
      for (int j = 0; j < params.dim; ++j) {
        auto diff = h_data[i * params.dim + j] - mean[j];
        expectedVal += diff * diff * T(h_counts[i]);
      }
    }
    expectedVal = sqrt(expectedVal);
    raft::interruptible::synchronize(stream);
  }

 protected:
  DispersionInputs<T> params;
  raft::resources handle;
  rmm::device_uvector<T> exp_mean, act_mean;
  cudaStream_t stream = 0;
  int npoints;
  T expectedVal, actualVal;
};

const std::vector<DispersionInputs<float>> inputsf = {
  {0.001f, 10, 1000, 1234ULL}, {0.001f, 100, 100, 1234ULL}, {0.001f, 1000, 1000, 1234ULL}};
typedef DispersionTest<float> DispersionTestF;
TEST_P(DispersionTestF, Result)
{
  auto eq = raft::CompareApprox<float>(params.tolerance);
  ASSERT_TRUE(devArrMatch(exp_mean.data(), act_mean.data(), params.dim, eq));
  ASSERT_TRUE(match(expectedVal, actualVal, eq));
}
INSTANTIATE_TEST_CASE_P(DispersionTests, DispersionTestF, ::testing::ValuesIn(inputsf));

const std::vector<DispersionInputs<double>> inputsd = {
  {0.001, 10, 1000, 1234ULL}, {0.001, 100, 100, 1234ULL}, {0.001, 1000, 1000, 1234ULL}};
typedef DispersionTest<double> DispersionTestD;
TEST_P(DispersionTestD, Result)
{
  auto eq = raft::CompareApprox<double>(params.tolerance);
  ASSERT_TRUE(devArrMatch(exp_mean.data(), act_mean.data(), params.dim, eq));
  ASSERT_TRUE(match(expectedVal, actualVal, eq));
}
INSTANTIATE_TEST_CASE_P(DispersionTests, DispersionTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
