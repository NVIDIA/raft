/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda_fp16.h>

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace raft {
namespace stats {

template <typename T>
float toFloat(T value)
{
  return static_cast<float>(value);
}

template <>
inline float toFloat<half>(half value)
{
  return __half2float(value);
}

struct float_to_half_op {
  __device__ half operator()(float x) const { return __float2half(x); }
};

template <typename InputT, typename OutputT = InputT>
struct MeanInputs {
  OutputT tolerance;
  InputT mean;
  int rows, cols;
  bool rowMajor;
  unsigned long long int seed;
  InputT stddev = (InputT)1.0;
};

template <typename InputT, typename OutputT>
::std::ostream& operator<<(::std::ostream& os, const MeanInputs<InputT, OutputT>& dims)
{
  return os << "{ tol=" << toFloat(dims.tolerance) << ", mean=" << toFloat(dims.mean)
            << ", rows=" << dims.rows << ", cols=" << dims.cols << ", rowMajor=" << dims.rowMajor
            << ", stddev=" << toFloat(dims.stddev) << "}" << std::endl;
}

template <typename InputT, typename OutputT = InputT>
class MeanTest : public ::testing::TestWithParam<MeanInputs<InputT, OutputT>> {
 public:
  MeanTest()
    : params(::testing::TestWithParam<MeanInputs<InputT, OutputT>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rows(params.rows),
      cols(params.cols),
      data(raft::make_device_matrix<InputT, int>(handle, rows, cols)),
      mean_act(raft::make_device_vector<OutputT, int>(handle, cols))
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int len = rows * cols;
    if constexpr (std::is_same_v<InputT, half>) {
      rmm::device_uvector<float> data_float(len, stream);
      normal(handle, r, data_float.data(), len, toFloat(params.mean), toFloat(params.stddev));
      raft::linalg::unaryOp(data.data_handle(), data_float.data(), len, float_to_half_op{}, stream);
    } else if constexpr (std::is_integral_v<InputT>) {
      normalInt(handle, r, data.data_handle(), len, params.mean, params.stddev);
    } else {
      normal(handle, r, data.data_handle(), len, params.mean, params.stddev);
    }
    meanSGtest();
  }

  void meanSGtest()
  {
    int rows = params.rows, cols = params.cols;
    if (params.rowMajor) {
      using layout = raft::row_major;
      mean(handle,
           raft::make_device_matrix_view<const InputT, int, layout>(data.data_handle(), rows, cols),
           raft::make_device_vector_view<OutputT, int>(mean_act.data_handle(), cols));
    } else {
      using layout = raft::col_major;
      mean(handle,
           raft::make_device_matrix_view<const InputT, int, layout>(data.data_handle(), rows, cols),
           raft::make_device_vector_view<OutputT, int>(mean_act.data_handle(), cols));
    }
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MeanInputs<InputT, OutputT> params;
  int rows, cols;
  raft::device_matrix<InputT, int> data;
  raft::device_vector<OutputT, int> mean_act;
};

// Note: For 1024 samples, 256 experiments, a mean of 1.0 with stddev=1.0, the
// measured mean (of a normal distribution) will fall outside of an epsilon of
// 0.15 only 4/10000 times. (epsilon of 0.1 will fail 30/100 times)
const std::vector<MeanInputs<float>> inputsf = {
  {0.15f, -1.f, 1024, 32, false, 1234ULL},
  {0.15f, -1.f, 1024, 64, false, 1234ULL},
  {0.15f, -1.f, 1024, 128, false, 1234ULL},
  {0.15f, -1.f, 1024, 256, false, 1234ULL},
  {0.15f, -1.f, 1024, 32, true, 1234ULL},
  {0.15f, -1.f, 1024, 64, true, 1234ULL},
  {0.15f, -1.f, 1024, 128, true, 1234ULL},
  {0.15f, -1.f, 1024, 256, true, 1234ULL},
  {0.15f, -1.f, 1030, 1, false, 1234ULL},
  {2.0f, -1.f, 31, 120, false, 1234ULL},
  {2.0f, -1.f, 1, 130, false, 1234ULL},
  {0.15f, -1.f, 1030, 1, true, 1234ULL},
  {2.0f, -1.f, 31, 120, true, 1234ULL},
  {2.0f, -1.f, 1, 130, true, 1234ULL},
  {2.0f, -1.f, 1, 1, false, 1234ULL},
  {2.0f, -1.f, 1, 1, true, 1234ULL},
  {2.0f, -1.f, 7, 23, false, 1234ULL},
  {2.0f, -1.f, 7, 23, true, 1234ULL},
  {2.0f, -1.f, 17, 5, false, 1234ULL},
  {2.0f, -1.f, 17, 5, true, 1234ULL},
  {0.0001f, 0.1f, 1 << 27, 2, false, 1234ULL, 0.0001f},
  {0.0001f, 0.1f, 1 << 27, 2, true, 1234ULL, 0.0001f}};

const std::vector<MeanInputs<double>> inputsd = {{0.15, -1.0, 1024, 32, false, 1234ULL},
                                                 {0.15, -1.0, 1024, 64, false, 1234ULL},
                                                 {0.15, -1.0, 1024, 128, false, 1234ULL},
                                                 {0.15, -1.0, 1024, 256, false, 1234ULL},
                                                 {0.15, -1.0, 1024, 32, true, 1234ULL},
                                                 {0.15, -1.0, 1024, 64, true, 1234ULL},
                                                 {0.15, -1.0, 1024, 128, true, 1234ULL},
                                                 {0.15, -1.0, 1024, 256, true, 1234ULL},
                                                 {0.15, -1.0, 1030, 1, false, 1234ULL},
                                                 {2.0, -1.0, 31, 120, false, 1234ULL},
                                                 {2.0, -1.0, 1, 130, false, 1234ULL},
                                                 {0.15, -1.0, 1030, 1, true, 1234ULL},
                                                 {2.0, -1.0, 31, 120, true, 1234ULL},
                                                 {2.0, -1.0, 1, 130, true, 1234ULL},
                                                 {2.0, -1.0, 1, 1, false, 1234ULL},
                                                 {2.0, -1.0, 1, 1, true, 1234ULL},
                                                 {2.0, -1.0, 7, 23, false, 1234ULL},
                                                 {2.0, -1.0, 7, 23, true, 1234ULL},
                                                 {2.0, -1.0, 17, 5, false, 1234ULL},
                                                 {2.0, -1.0, 17, 5, true, 1234ULL},
                                                 {1e-8, 1e-1, 1 << 27, 2, false, 1234ULL, 0.0001},
                                                 {1e-8, 1e-1, 1 << 27, 2, true, 1234ULL, 0.0001}};

const std::vector<MeanInputs<half, float>> inputshf = {
  {0.15f, -1.f, 1024, 32, false, 1234ULL},
  {0.15f, -1.f, 1024, 64, false, 1234ULL},
  {0.15f, -1.f, 1024, 128, false, 1234ULL},
  {0.15f, -1.f, 1024, 256, false, 1234ULL},
  {0.15f, -1.f, 1024, 32, true, 1234ULL},
  {0.15f, -1.f, 1024, 64, true, 1234ULL},
  {0.0001f, 0.1f, 1 << 27, 2, false, 1234ULL, 0.0001f}};

const std::vector<MeanInputs<int8_t, half>> inputsi8h = {{0.95f, -5, 8096, 32, false, 1234ULL, 1},
                                                         {0.5f, 1, 8096, 10, false, 1234ULL, 10},
                                                         {0.15f, 0, 60000, 128, false, 1234ULL, 6},
                                                         {0.5f, -1, 8096, 256, false, 1234ULL, 2},
                                                         {1.0f, 8, 2000, 32, true, 1234ULL, 1},
                                                         {0.50f, -1, 20000, 64, true, 1234ULL, 5},
                                                         {1.0f, 6, 10024, 2, false, 1234ULL, 10}};

typedef MeanTest<float> MeanTestF;
TEST_P(MeanTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    params.mean, mean_act.data_handle(), params.cols, CompareApprox<float>(params.tolerance)));
}

typedef MeanTest<double> MeanTestD;
TEST_P(MeanTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    params.mean, mean_act.data_handle(), params.cols, CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(MeanTests, MeanTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MeanTests, MeanTestD, ::testing::ValuesIn(inputsd));

typedef MeanTest<half, float> MeanTestHF;
TEST_P(MeanTestHF, Result)
{
  ASSERT_TRUE(devArrMatch(toFloat(params.mean),
                          mean_act.data_handle(),
                          params.cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef MeanTest<int8_t, half> MeanTestI8H;
TEST_P(MeanTestI8H, Result)
{
  std::vector<half> mean_act_h(params.cols);
  raft::update_host(mean_act_h.data(), mean_act.data_handle(), params.cols, stream);
  raft::resource::sync_stream(handle);

  auto expected  = toFloat(params.mean);
  auto tolerance = toFloat(params.tolerance);
  for (int i = 0; i < params.cols; ++i) {
    ASSERT_NEAR(toFloat(mean_act_h[i]), expected, tolerance) << " @col=" << i;
  }
}

INSTANTIATE_TEST_SUITE_P(MeanTests, MeanTestHF, ::testing::ValuesIn(inputshf));

INSTANTIATE_TEST_SUITE_P(MeanTests, MeanTestI8H, ::testing::ValuesIn(inputsi8h));
}  // end namespace stats
}  // end namespace raft
