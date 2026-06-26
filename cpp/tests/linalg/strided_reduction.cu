/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "reduce.cuh"

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/strided_reduction.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T>
struct stridedReductionInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
void stridedReductionLaunch(
  const raft::resources& handle, T* dots, const T* data, int cols, int rows, bool inplace)
{
  auto dots_view = raft::make_device_vector_view(dots, cols);
  auto data_view = raft::make_device_matrix_view(data, rows, cols);
  strided_reduction(handle, data_view, dots_view, (T)0, inplace, raft::sq_op{});
}

template <typename T>
class stridedReductionTest : public ::testing::TestWithParam<stridedReductionInputs<T>> {
 public:
  stridedReductionTest()
    : params(::testing::TestWithParam<stridedReductionInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      dots_exp(params.cols, stream),  // expected dot products (from test)
      dots_act(params.cols, stream)   // actual dot products (from prim)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    uniform(handle, r, data.data(), len, T(-1.0), T(1.0));  // initialize matrix to random

    // Perform reduction with default inplace = false first and inplace = true next

    naiveStridedReduction(dots_exp.data(),
                          data.data(),
                          cols,
                          rows,
                          stream,
                          T(0),
                          false,
                          raft::sq_op{},
                          raft::add_op{},
                          raft::identity_op{});
    naiveStridedReduction(dots_exp.data(),
                          data.data(),
                          cols,
                          rows,
                          stream,
                          T(0),
                          true,
                          raft::sq_op{},
                          raft::add_op{},
                          raft::identity_op{});
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        stridedReductionLaunch(h, dots_act.data(), data.data(), cols, rows, false);
        stridedReductionLaunch(h, dots_act.data(), data.data(), cols, rows, true);
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  stridedReductionInputs<T> params;
  rmm::device_uvector<T> data, dots_exp, dots_act;
};

const std::vector<stridedReductionInputs<float>> inputsf = {{0.00001f, 1024, 32, 1234ULL},
                                                            {0.00001f, 1024, 64, 1234ULL},
                                                            {0.00001f, 1024, 128, 1234ULL},
                                                            {0.00001f, 1024, 256, 1234ULL}};

const std::vector<stridedReductionInputs<double>> inputsd = {{0.000000001, 1024, 32, 1234ULL},
                                                             {0.000000001, 1024, 64, 1234ULL},
                                                             {0.000000001, 1024, 128, 1234ULL},
                                                             {0.000000001, 1024, 256, 1234ULL}};

typedef stridedReductionTest<float> stridedReductionTestF;
TEST_P(stridedReductionTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    dots_exp.data(), dots_act.data(), params.cols, raft::CompareApprox<float>(params.tolerance)));
}

typedef stridedReductionTest<double> stridedReductionTestD;
TEST_P(stridedReductionTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    dots_exp.data(), dots_act.data(), params.cols, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(stridedReductionTests, stridedReductionTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(stridedReductionTests, stridedReductionTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
