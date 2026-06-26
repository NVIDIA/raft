/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "reduce.cuh"

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T>
struct coalescedReductionInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const coalescedReductionInputs<T>& dims)
{
  return os << "{ " << dims.tolerance << ", " << dims.rows << ", " << dims.cols << ", "
            << dims.seed;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void coalescedReductionLaunch(
  const raft::resources& handle, T* dots, const T* data, int cols, int rows, bool inplace = false)
{
  auto dots_view = raft::make_device_vector_view(dots, rows);
  auto data_view = raft::make_device_matrix_view(data, rows, cols);
  coalesced_reduction(handle, data_view, dots_view, (T)0, inplace, raft::sq_op{});
}

template <typename T>
class coalescedReductionTest : public ::testing::TestWithParam<coalescedReductionInputs<T>> {
 public:
  coalescedReductionTest()
    : params(::testing::TestWithParam<coalescedReductionInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      dots_exp(params.rows * params.cols, stream),
      dots_act(params.rows * params.cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    uniform(handle, r, data.data(), len, T(-1.0), T(1.0));

    // Perform reduction with default inplace = false first and inplace = true next

    naiveCoalescedReduction(dots_exp.data(),
                            data.data(),
                            cols,
                            rows,
                            stream,
                            T(0),
                            false,
                            raft::sq_op{},
                            raft::add_op{},
                            raft::identity_op{});
    naiveCoalescedReduction(dots_exp.data(),
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
        coalescedReductionLaunch(h, dots_act.data(), data.data(), cols, rows);
        coalescedReductionLaunch(h, dots_act.data(), data.data(), cols, rows, true);
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN);

    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  coalescedReductionInputs<T> params;
  rmm::device_uvector<T> data;
  rmm::device_uvector<T> dots_exp;
  rmm::device_uvector<T> dots_act;
};

// Note: it's important to have a variety of rows/columns combinations to test all possible code
// paths: thin (few cols or many rows), medium, thick (many cols, very few rows).

const std::vector<coalescedReductionInputs<float>> inputsf = {{0.000002f, 50, 2, 1234ULL},
                                                              {0.000002f, 50, 3, 1234ULL},
                                                              {0.000002f, 50, 7, 1234ULL},
                                                              {0.000002f, 50, 9, 1234ULL},
                                                              {0.000002f, 50, 20, 1234ULL},
                                                              {0.000002f, 50, 55, 1234ULL},
                                                              {0.000002f, 50, 100, 1234ULL},
                                                              {0.000002f, 50, 270, 1234ULL},
                                                              {0.000002f, 10000, 3, 1234ULL},
                                                              {0.000002f, 10000, 9, 1234ULL},
                                                              {0.000002f, 10000, 20, 1234ULL},
                                                              {0.000002f, 10000, 55, 1234ULL},
                                                              {0.000002f, 10000, 100, 1234ULL},
                                                              {0.000002f, 10000, 270, 1234ULL},
                                                              {0.0001f, 10, 25000, 1234ULL},
                                                              {0.0001f, 2, 200000, 1234ULL}};

const std::vector<coalescedReductionInputs<double>> inputsd = {{0.000000001, 50, 2, 1234ULL},
                                                               {0.000000001, 50, 3, 1234ULL},
                                                               {0.000000001, 50, 7, 1234ULL},
                                                               {0.000000001, 50, 9, 1234ULL},
                                                               {0.000000001, 50, 20, 1234ULL},
                                                               {0.000000001, 50, 55, 1234ULL},
                                                               {0.000000001, 50, 100, 1234ULL},
                                                               {0.000000001, 50, 270, 1234ULL},
                                                               {0.000000001, 10000, 3, 1234ULL},
                                                               {0.000000001, 10000, 9, 1234ULL},
                                                               {0.000000001, 10000, 20, 1234ULL},
                                                               {0.000000001, 10000, 55, 1234ULL},
                                                               {0.000000001, 10000, 100, 1234ULL},
                                                               {0.000000001, 10000, 270, 1234ULL},
                                                               {0.0000001, 10, 25000, 1234ULL},
                                                               {0.0000001, 2, 200000, 1234ULL}};

typedef coalescedReductionTest<float> coalescedReductionTestF;
TEST_P(coalescedReductionTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(dots_exp.data(),
                                dots_act.data(),
                                params.rows,
                                raft::CompareApprox<float>(params.tolerance),
                                stream));
}

typedef coalescedReductionTest<double> coalescedReductionTestD;
TEST_P(coalescedReductionTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(dots_exp.data(),
                                dots_act.data(),
                                params.rows,
                                raft::CompareApprox<double>(params.tolerance),
                                stream));
}

INSTANTIATE_TEST_CASE_P(coalescedReductionTests,
                        coalescedReductionTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(coalescedReductionTests,
                        coalescedReductionTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
