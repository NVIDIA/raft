/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/cov.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace stats {

template <typename T>
struct CovInputs {
  T tolerance, mean, var;
  int rows, cols;
  bool sample, rowMajor, stable;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const CovInputs<T>& dims)
{
  return os << "{ " << dims.tolerance << ", " << dims.rows << ", " << dims.cols << ", "
            << dims.sample << ", " << dims.rowMajor << "}" << std::endl;
}

template <typename T>
class CovTest : public ::testing::TestWithParam<CovInputs<T>> {
 protected:
  CovTest()
    : data(0, stream),
      mean_act(0, stream),
      cov_act(0, stream),
      cov_cm(0, stream),
      cov_cm_ref(0, stream)
  {
  }

  void SetUp() override
  {
    raft::resources handle;
    cudaStream_t stream = resource::get_cuda_stream(handle);

    params = ::testing::TestWithParam<CovInputs<T>>::GetParam();
    params.tolerance *= 2;
    raft::random::RngState r(params.seed);
    int rows = params.rows, cols = params.cols;
    auto len = rows * cols;
    T var    = params.var;
    data.resize(len, stream);
    mean_act.resize(cols, stream);
    cov_act.resize(cols * cols, stream);

    normal(handle, r, data.data(), len, params.mean, var);
    if (params.rowMajor) {
      using layout = raft::row_major;
      raft::stats::mean<true>(mean_act.data(), data.data(), cols, rows, stream);
      raft::execute_with_dry_run_check(
        handle,
        [&](raft::resources const& h) {
          cov(h,
              raft::make_device_matrix_view<T, std::uint32_t, layout>(data.data(), rows, cols),
              raft::make_device_vector_view<const T, std::uint32_t>(mean_act.data(), cols),
              raft::make_device_matrix_view<T, std::uint32_t, layout>(cov_act.data(), cols, cols),
              params.sample,
              params.stable);
        },
        raft::alloc_behavior::NO_ALLOCATIONS);
    } else {
      using layout = raft::col_major;
      raft::stats::mean<false>(mean_act.data(), data.data(), cols, rows, stream);
      raft::execute_with_dry_run_check(
        handle,
        [&](raft::resources const& h) {
          cov(h,
              raft::make_device_matrix_view<T, std::uint32_t, layout>(data.data(), rows, cols),
              raft::make_device_vector_view<const T, std::uint32_t>(mean_act.data(), cols),
              raft::make_device_matrix_view<T, std::uint32_t, layout>(cov_act.data(), cols, cols),
              params.sample,
              params.stable);
        },
        raft::alloc_behavior::NO_ALLOCATIONS);
    }

    T data_h[6]       = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
    T cov_cm_ref_h[4] = {4.3333, -2.8333, -2.8333, 2.333};

    cov_cm.resize(4, stream);
    cov_cm_ref.resize(4, stream);
    rmm::device_uvector<T> data_cm(6, stream);
    rmm::device_uvector<T> mean_cm(2, stream);

    raft::update_device(data_cm.data(), data_h, 6, stream);
    raft::update_device(cov_cm_ref.data(), cov_cm_ref_h, 4, stream);

    raft::stats::mean<false>(mean_cm.data(), data_cm.data(), 2, 3, stream);
    cov<false>(handle, cov_cm.data(), data_cm.data(), mean_cm.data(), 2, 3, true, true, stream);
  }

 protected:
  cublasHandle_t handle;
  cudaStream_t stream = 0;
  CovInputs<T> params;
  rmm::device_uvector<T> data, mean_act, cov_act, cov_cm, cov_cm_ref;
};

///@todo: add stable=false after it has been implemented
const std::vector<CovInputs<float>> inputsf = {
  {0.03f, 1.f, 2.f, 32 * 1024, 32, true, false, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 64, true, false, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 128, true, false, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 256, true, false, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 32, false, false, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 64, false, false, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 128, false, false, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 256, false, false, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 32, true, true, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 64, true, true, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 128, true, true, true, 1234ULL},
  {0.03f, 1.f, 2.f, 32 * 1024, 256, true, true, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 32, false, true, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 64, false, true, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 128, false, true, true, 1234ULL},
  {0.03f, -1.f, 2.f, 32 * 1024, 256, false, true, true, 1234ULL}};

const std::vector<CovInputs<double>> inputsd = {
  {0.03, 1.0, 2.0, 32 * 1024, 32, true, false, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 64, true, false, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 128, true, false, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 256, true, false, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 32, false, false, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 64, false, false, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 128, false, false, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 256, false, false, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 32, true, true, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 64, true, true, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 128, true, true, true, 1234ULL},
  {0.03, 1.0, 2.0, 32 * 1024, 256, true, true, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 32, false, true, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 64, false, true, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 128, false, true, true, 1234ULL},
  {0.03, -1.0, 2.0, 32 * 1024, 256, false, true, true, 1234ULL}};

typedef CovTest<float> CovTestF;
TEST_P(CovTestF, Result)
{
  ASSERT_TRUE(raft::diagonalMatch(params.var * params.var,
                                  cov_act.data(),
                                  params.cols,
                                  params.cols,
                                  raft::CompareApprox<float>(params.tolerance)));
}

typedef CovTest<double> CovTestD;
TEST_P(CovTestD, Result)
{
  ASSERT_TRUE(raft::diagonalMatch(params.var * params.var,
                                  cov_act.data(),
                                  params.cols,
                                  params.cols,
                                  raft::CompareApprox<double>(params.tolerance)));
}

typedef CovTest<float> CovTestSmallF;
TEST_P(CovTestSmallF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    cov_cm_ref.data(), cov_cm.data(), 2, 2, raft::CompareApprox<float>(params.tolerance)));
}

typedef CovTest<double> CovTestSmallD;
TEST_P(CovTestSmallD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    cov_cm_ref.data(), cov_cm.data(), 2, 2, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(CovTests, CovTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CovTests, CovTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_CASE_P(CovTests, CovTestSmallF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(CovTests, CovTestSmallD, ::testing::ValuesIn(inputsd));

}  // namespace stats
}  // namespace raft
