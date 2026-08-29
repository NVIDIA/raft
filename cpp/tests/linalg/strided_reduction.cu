/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "reduce.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/strided_reduction.cuh>
#include <raft/matrix/init.cuh>
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

/*
 * The reduced dimension is mapped onto the y dimension of the grid, which CUDA limits to 65535
 * blocks. That is far fewer blocks than the number of rows a caller may reduce along, so the grid
 * must be bounded and the kernels must stride over the remaining rows.
 */
TEST(stridedReductionTest, LargeReducedDimension)
{
  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);
  // More rows than the largest grid the kernels are launched with can cover one row per thread.
  constexpr int64_t kRows = 8'400'000;
  constexpr int64_t kCols = 2;

  auto data = raft::make_device_matrix<float, int64_t>(handle, kRows, kCols);
  raft::matrix::fill(handle, data.view(), 1.0f);
  auto data_view = raft::make_const_mdspan(data.view());

  // Summing into the input type takes the compensated-summation path.
  auto sums_same_type = raft::make_device_vector<float, int64_t>(handle, kCols);
  strided_reduction(handle, data_view, sums_same_type.view(), 0.0f);

  // Summing into a wider type takes the generic path, which handles an arbitrary reduce operation.
  auto sums_wider_type = raft::make_device_vector<double, int64_t>(handle, kCols);
  strided_reduction(handle, data_view, sums_wider_type.view(), 0.0, false, raft::cast_op<double>{});

  resource::sync_stream(handle, stream);

  ASSERT_TRUE(devArrMatch(static_cast<float>(kRows),
                          sums_same_type.data_handle(),
                          kCols,
                          raft::CompareApprox<float>(1e-6f),
                          stream));
  ASSERT_TRUE(devArrMatch(static_cast<double>(kRows),
                          sums_wider_type.data_handle(),
                          kCols,
                          raft::CompareApprox<double>(1e-12),
                          stream));
}

}  // end namespace linalg
}  // end namespace raft
