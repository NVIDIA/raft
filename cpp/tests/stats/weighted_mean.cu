/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/weighted_mean.cuh>
#include <raft/util/cuda_utils.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <gtest/gtest.h>

#include <cstdint>

namespace raft {
namespace stats {

template <typename T>
struct WeightedMeanInputs {
  T tolerance;
  int M, N;
  unsigned long long int seed;
  bool along_rows;  // Used only for the weightedMean test function
  bool row_major;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const WeightedMeanInputs<T>& I)
{
  return os << "{ " << I.tolerance << ", " << I.M << ", " << I.N << ", " << I.seed << ", "
            << I.along_rows << "}" << std::endl;
}

///// weighted row-wise mean test and support functions
template <typename T>
void naiveRowWeightedMean(T* R, T* D, T* W, int M, int N, bool rowMajor)
{
  int istr = rowMajor ? 1 : M;
  int jstr = rowMajor ? N : 1;

  // sum the weights
  T WS = 0;
  for (int i = 0; i < N; i++)
    WS += W[i];

  for (int j = 0; j < M; j++) {
    R[j] = (T)0;
    for (int i = 0; i < N; i++) {
      // R[j] += (W[i]*D[i*istr + j*jstr] - R[j])/(T)(i+1);
      R[j] += (W[i] * D[i * istr + j * jstr]) / WS;
    }
  }
}

template <typename T>
class RowWeightedMeanTest : public ::testing::TestWithParam<WeightedMeanInputs<T>> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<WeightedMeanInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    int rows = params.M, cols = params.N, len = rows * cols;
    auto stream = resource::get_cuda_stream(handle);
    // device-side data
    din.resize(len);
    dweights.resize(cols);
    dexp.resize(rows);
    dact.resize(rows);

    // create random matrix and weights
    uniform(handle, r, din.data().get(), len, T(-1.0), T(1.0));
    uniform(handle, r, dweights.data().get(), cols, T(-1.0), T(1.0));

    // host-side data
    thrust::host_vector<T> hin      = din;
    thrust::host_vector<T> hweights = dweights;
    thrust::host_vector<T> hexp(rows);

    // compute naive result & copy to GPU
    naiveRowWeightedMean(hexp.data(), hin.data(), hweights.data(), rows, cols, params.row_major);
    dexp        = hexp;
    auto output = raft::make_device_vector_view<T, std::uint32_t>(dact.data().get(), rows);
    auto weights =
      raft::make_device_vector_view<const T, std::uint32_t>(dweights.data().get(), cols);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.row_major) {
          auto input = raft::make_device_matrix_view<const T, std::uint32_t, raft::row_major>(
            din.data().get(), rows, cols);
          row_weighted_mean(h, input, weights, output);
        } else {
          auto input = raft::make_device_matrix_view<const T, std::uint32_t, raft::col_major>(
            din.data().get(), rows, cols);
          row_weighted_mean(h, input, weights, output);
        }
      },
      raft::alloc_behavior::NO_ALLOCATIONS);

    // adjust tolerance to account for round-off accumulation
    params.tolerance *= params.N;
  }

 protected:
  raft::resources handle;
  WeightedMeanInputs<T> params;
  thrust::host_vector<T> hin, hweights;
  thrust::device_vector<T> din, dweights, dexp, dact;
};

///// weighted column-wise mean test and support functions
template <typename T>
void naiveColWeightedMean(T* R, T* D, T* W, int M, int N, bool rowMajor)
{
  int istr = rowMajor ? 1 : M;
  int jstr = rowMajor ? N : 1;

  // sum the weights
  T WS = 0;
  for (int j = 0; j < M; j++)
    WS += W[j];

  for (int i = 0; i < N; i++) {
    R[i] = (T)0;
    for (int j = 0; j < M; j++) {
      // R[i] += (W[j]*D[i*istr + j*jstr] - R[i])/(T)(j+1);
      R[i] += (W[j] * D[i * istr + j * jstr]) / WS;
    }
  }
}

template <typename T>
class ColWeightedMeanTest : public ::testing::TestWithParam<WeightedMeanInputs<T>> {
  void SetUp() override
  {
    params = ::testing::TestWithParam<WeightedMeanInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    int rows = params.M, cols = params.N, len = rows * cols;

    auto stream = resource::get_cuda_stream(handle);
    // device-side data
    din.resize(len);
    dweights.resize(rows);
    dexp.resize(cols);
    dact.resize(cols);

    // create random matrix and weights
    uniform(handle, r, din.data().get(), len, T(-1.0), T(1.0));
    uniform(handle, r, dweights.data().get(), rows, T(-1.0), T(1.0));

    // host-side data
    thrust::host_vector<T> hin      = din;
    thrust::host_vector<T> hweights = dweights;
    thrust::host_vector<T> hexp(cols);

    // compute naive result & copy to GPU
    naiveColWeightedMean(hexp.data(), hin.data(), hweights.data(), rows, cols, params.row_major);
    dexp = hexp;

    auto output = raft::make_device_vector_view<T, std::uint32_t>(dact.data().get(), cols);
    auto weights =
      raft::make_device_vector_view<const T, std::uint32_t>(dweights.data().get(), rows);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.row_major) {
          auto input = raft::make_device_matrix_view<const T, std::uint32_t, raft::row_major>(
            din.data().get(), rows, cols);
          col_weighted_mean(h, input, weights, output);
        } else {
          auto input = raft::make_device_matrix_view<const T, std::uint32_t, raft::col_major>(
            din.data().get(), rows, cols);
          col_weighted_mean(h, input, weights, output);
        }
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    // adjust tolerance to account for round-off accumulation
    params.tolerance *= params.M;
  }

 protected:
  raft::resources handle;
  WeightedMeanInputs<T> params;
  thrust::host_vector<T> hin, hweights;
  thrust::device_vector<T> din, dweights, dexp, dact;
};

template <typename T>
class WeightedMeanTest : public ::testing::TestWithParam<WeightedMeanInputs<T>> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<WeightedMeanInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    auto stream = resource::get_cuda_stream(handle);
    int rows = params.M, cols = params.N, len = rows * cols;
    auto weight_size = params.along_rows ? cols : rows;
    auto mean_size   = params.along_rows ? rows : cols;
    // device-side data
    din.resize(len);
    dweights.resize(weight_size);
    dexp.resize(mean_size);
    dact.resize(mean_size);

    // create random matrix and weights
    uniform(handle, r, din.data().get(), len, T(-1.0), T(1.0));
    uniform(handle, r, dweights.data().get(), weight_size, T(-1.0), T(1.0));

    // host-side data
    thrust::host_vector<T> hin      = din;
    thrust::host_vector<T> hweights = dweights;
    thrust::host_vector<T> hexp(mean_size);

    // compute naive result & copy to GPU
    if (params.along_rows)
      naiveRowWeightedMean(hexp.data(), hin.data(), hweights.data(), rows, cols, params.row_major);
    else
      naiveColWeightedMean(hexp.data(), hin.data(), hweights.data(), rows, cols, params.row_major);
    dexp = hexp;

    auto output = raft::make_device_vector_view<T, std::uint32_t>(dact.data().get(), mean_size);
    auto weights =
      raft::make_device_vector_view<const T, std::uint32_t>(dweights.data().get(), weight_size);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.row_major) {
          auto input = raft::make_device_matrix_view<const T, std::uint32_t, raft::row_major>(
            din.data().get(), rows, cols);
          if (params.along_rows) {
            weighted_mean<Apply::ALONG_ROWS>(h, input, weights, output);
          } else {
            weighted_mean<Apply::ALONG_COLUMNS>(h, input, weights, output);
          }
        } else {
          auto input = raft::make_device_matrix_view<const T, std::uint32_t, raft::col_major>(
            din.data().get(), rows, cols);
          if (params.along_rows) {
            weighted_mean<Apply::ALONG_ROWS>(h, input, weights, output);
          } else {
            weighted_mean<Apply::ALONG_COLUMNS>(h, input, weights, output);
          }
        }
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    // adjust tolerance to account for round-off accumulation
    params.tolerance *= params.N;
  }

 protected:
  raft::resources handle;
  WeightedMeanInputs<T> params;
  thrust::host_vector<T> hin, hweights;
  thrust::device_vector<T> din, dweights, dexp, dact;
};

////// Parameter sets and test instantiation
static const float tolF  = 128 * std::numeric_limits<float>::epsilon();
static const double tolD = 256 * std::numeric_limits<double>::epsilon();

const std::vector<WeightedMeanInputs<float>> inputsf = {{tolF, 4, 4, 1234, true, true},
                                                        {tolF, 32, 32, 1234, true, false},
                                                        {tolF, 32, 64, 1234, false, false},
                                                        {tolF, 32, 256, 1234, true, true},
                                                        {tolF, 32, 256, 1234, false, false},
                                                        {tolF, 1024, 32, 1234, true, false},
                                                        {tolF, 1024, 64, 1234, true, true},
                                                        {tolF, 1024, 128, 1234, true, false},
                                                        {tolF, 1024, 256, 1234, true, true},
                                                        {tolF, 1024, 32, 1234, false, false},
                                                        {tolF, 1024, 64, 1234, false, true},
                                                        {tolF, 1024, 128, 1234, false, false},
                                                        {tolF, 1024, 256, 1234, false, true}};

const std::vector<WeightedMeanInputs<double>> inputsd = {{tolD, 4, 4, 1234, true, true},
                                                         {tolD, 32, 32, 1234, true, false},
                                                         {tolD, 32, 64, 1234, false, false},
                                                         {tolD, 32, 256, 1234, true, true},
                                                         {tolD, 32, 256, 1234, false, false},
                                                         {tolD, 1024, 32, 1234, true, false},
                                                         {tolD, 1024, 64, 1234, true, true},
                                                         {tolD, 1024, 128, 1234, true, false},
                                                         {tolD, 1024, 256, 1234, true, true},
                                                         {tolD, 1024, 32, 1234, false, false},
                                                         {tolD, 1024, 64, 1234, false, true},
                                                         {tolD, 1024, 128, 1234, false, false},
                                                         {tolD, 1024, 256, 1234, false, true}};

using RowWeightedMeanTestF = RowWeightedMeanTest<float>;
TEST_P(RowWeightedMeanTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    dexp.data().get(), dact.data().get(), params.M, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(RowWeightedMeanTest, RowWeightedMeanTestF, ::testing::ValuesIn(inputsf));

using RowWeightedMeanTestD = RowWeightedMeanTest<double>;
TEST_P(RowWeightedMeanTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    dexp.data().get(), dact.data().get(), params.M, raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(RowWeightedMeanTest, RowWeightedMeanTestD, ::testing::ValuesIn(inputsd));

using ColWeightedMeanTestF = ColWeightedMeanTest<float>;
TEST_P(ColWeightedMeanTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    dexp.data().get(), dact.data().get(), params.N, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ColWeightedMeanTest, ColWeightedMeanTestF, ::testing::ValuesIn(inputsf));

using ColWeightedMeanTestD = ColWeightedMeanTest<double>;
TEST_P(ColWeightedMeanTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    dexp.data().get(), dact.data().get(), params.N, raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ColWeightedMeanTest, ColWeightedMeanTestD, ::testing::ValuesIn(inputsd));

using WeightedMeanTestF = WeightedMeanTest<float>;
TEST_P(WeightedMeanTestF, Result)
{
  auto mean_size = params.along_rows ? params.M : params.N;
  ASSERT_TRUE(devArrMatch(
    dexp.data().get(), dact.data().get(), mean_size, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(WeightedMeanTest, WeightedMeanTestF, ::testing::ValuesIn(inputsf));

using WeightedMeanTestD = WeightedMeanTest<double>;
TEST_P(WeightedMeanTestD, Result)
{
  auto mean_size = params.along_rows ? params.M : params.N;
  ASSERT_TRUE(devArrMatch(dexp.data().get(),
                          dact.data().get(),
                          mean_size,
                          raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(WeightedMeanTest, WeightedMeanTestD, ::testing::ValuesIn(inputsd));

};  // end namespace stats
};  // end namespace raft
