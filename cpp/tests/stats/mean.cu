/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

namespace raft {
namespace stats {

template <typename T>
struct MeanInputs {
  T tolerance, mean;
  int rows, cols;
  bool rowMajor;
  unsigned long long int seed;
  T stddev = (T)1.0;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MeanInputs<T>& dims)
{
  return os << "{ " << dims.tolerance << ", " << dims.rows << ", " << dims.cols << ", "
            << ", " << dims.rowMajor << ", " << dims.stddev << "}" << std::endl;
}

template <typename T>
class MeanTest : public ::testing::TestWithParam<MeanInputs<T>> {
 public:
  MeanTest()
    : params(::testing::TestWithParam<MeanInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rows(params.rows),
      cols(params.cols),
      data(rows * cols, stream),
      mean_act(cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int len = rows * cols;
    normal(handle, r, data.data(), len, params.mean, params.stddev);
    meanSGtest(data.data(), stream);
  }

  void meanSGtest(T* data, cudaStream_t stream)
  {
    int rows = params.rows, cols = params.cols;
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.rowMajor) {
          using layout = raft::row_major;
          mean(h,
               raft::make_device_matrix_view<const T, int, layout>(data, rows, cols),
               raft::make_device_vector_view<T, int>(mean_act.data(), cols));
        } else {
          using layout = raft::col_major;
          mean(h,
               raft::make_device_matrix_view<const T, int, layout>(data, rows, cols),
               raft::make_device_vector_view<T, int>(mean_act.data(), cols));
        }
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MeanInputs<T> params;
  int rows, cols;
  rmm::device_uvector<T> data, mean_act;
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

typedef MeanTest<float> MeanTestF;
TEST_P(MeanTestF, Result)
{
  ASSERT_TRUE(
    devArrMatch(params.mean, mean_act.data(), params.cols, CompareApprox<float>(params.tolerance)));
}

typedef MeanTest<double> MeanTestD;
TEST_P(MeanTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    params.mean, mean_act.data(), params.cols, CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(MeanTests, MeanTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MeanTests, MeanTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
