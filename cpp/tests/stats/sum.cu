/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace raft {
namespace stats {

template <typename T>
struct SumInputs {
  T tolerance;
  int rows, cols;
  bool rowMajor;
  T value = T(1);
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SumInputs<T>& dims)
{
  return os << "{ " << dims.tolerance << ", " << dims.rows << ", " << dims.cols << ", "
            << dims.rowMajor << ", " << dims.value << "}" << std::endl;
}

template <typename T>
class SumTest : public ::testing::TestWithParam<SumInputs<T>> {
 public:
  SumTest()
    : params(::testing::TestWithParam<SumInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      rows(params.rows),
      cols(params.cols),
      data(rows * cols, stream),
      sum_act(cols, stream)
  {
  }

 protected:
  void runTest(bool checkErrorCompensation = false)
  {
    int len = rows * cols;

    double large_factor = 1e7;

    if constexpr (std::is_same_v<double, T>) large_factor = 1e12;

    std::vector<T> data_h(len);
    for (int i = 0; i < len; i++) {
      data_h[i] = double(params.value);
      int row   = params.rowMajor ? i / cols : i % rows;

      // every 3 elements (in a column) contain 2 large dummy elements
      // (one of them negative) and one element with 3x compensating
      // for the 2 missing elements
      if (checkErrorCompensation && row % 3 == 2) {
        data_h[i] = double(params.value) * large_factor;
        // compensate with opposite error 3 rows up
        int idx2     = params.rowMajor ? (i - cols) : (i - 1);
        data_h[idx2] = -1 * double(params.value) * large_factor;
        // compensate 2 missing values
        int idx3     = params.rowMajor ? (i - 2 * cols) : (i - 2);
        data_h[idx3] = 3.0 * double(params.value);
      }
    }

    raft::update_device(data.data(), data_h.data(), len, stream);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.rowMajor) {
          using layout = raft::row_major;
          sum(h,
              raft::make_device_matrix_view<const T, int, layout>(data.data(), rows, cols),
              raft::make_device_vector_view(sum_act.data(), cols));
        } else {
          using layout = raft::col_major;
          sum(h,
              raft::make_device_matrix_view<const T, int, layout>(data.data(), rows, cols),
              raft::make_device_vector_view(sum_act.data(), cols));
        }
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN);
    resource::sync_stream(handle, stream);

    double expected = double(params.rows) * params.value;

    double tolerance = checkErrorCompensation ? 100 * params.tolerance : params.tolerance;

    ASSERT_TRUE(raft::devArrMatch(
      T(expected), sum_act.data(), params.cols, raft::CompareApprox<T>(tolerance)));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SumInputs<T> params;
  int rows, cols;
  rmm::device_uvector<T> data, sum_act;
};

template <typename T, typename Layout>
void check_sum_nonfinite(int rows)
{
  constexpr int cols = 7;
  const T inf        = std::numeric_limits<T>::infinity();
  const T max        = std::numeric_limits<T>::max();
  std::vector<T> input(rows * cols, T{1});
  auto at = [&](int row, int col) -> T& {
    return input[std::is_same_v<Layout, raft::row_major> ? row * cols + col : col * rows + row];
  };
  at(0, 0)        = inf;
  at(0, 1)        = -inf;
  at(0, 2)        = inf;
  at(rows - 1, 2) = -inf;
  at(0, 3)        = std::numeric_limits<T>::quiet_NaN();
  at(0, 4) = at(1, 4) = max;
  at(0, 5) = at(1, 5) = -max;

  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);
  rmm::device_uvector<T> data(input.size(), stream);
  rmm::device_uvector<T> output(cols, stream);
  raft::update_device(data.data(), input.data(), input.size(), stream);
  sum(handle,
      raft::make_device_matrix_view<const T, int, Layout>(data.data(), rows, cols),
      raft::make_device_vector_view(output.data(), cols));
  std::array<T, cols> actual;
  raft::update_host(actual.data(), output.data(), actual.size(), stream);
  raft::resource::sync_stream(handle);
  EXPECT_EQ(actual[0], inf);
  EXPECT_EQ(actual[1], -inf);
  EXPECT_TRUE(std::isnan(actual[2]));
  EXPECT_TRUE(std::isnan(actual[3]));
  EXPECT_EQ(actual[4], inf);
  EXPECT_EQ(actual[5], -inf);
  EXPECT_EQ(actual[6], T(rows));
}

template <typename T>
class NonfiniteSumTest : public ::testing::Test {};

using SumTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NonfiniteSumTest, SumTypes);

TYPED_TEST(NonfiniteSumTest, ColumnSums)
{
  // The contiguous layout selects different kernels for short, medium, and long columns.
  for (int rows : {32, 1024, 131072}) {
    SCOPED_TRACE(rows);
    check_sum_nonfinite<TypeParam, raft::row_major>(rows);
    check_sum_nonfinite<TypeParam, raft::col_major>(rows);
  }
}

const std::vector<SumInputs<float>> inputsf = {
  {0.0001f, 4, 5, true, 1},          {0.0001f, 1024, 32, true, 1},
  {0.0001f, 1024, 256, true, 1},     {0.0001f, 100000000, 1, true, 0.001},
  {0.0001f, 1 << 27, 2, true, 0.1},  {0.0001f, 1, 30, true, 0.001},
  {0.0001f, 1, 1, true, 0.001},      {0.0001f, 17, 5, true, 0.001},
  {0.0001f, 7, 23, true, 0.001},     {0.0001f, 3, 97, true, 0.001},
  {0.0001f, 4, 5, false, 1},         {0.0001f, 1024, 32, false, 1},
  {0.0001f, 1024, 256, false, 1},    {0.0001f, 100000000, 1, false, 0.001},
  {0.0001f, 1 << 27, 2, false, 0.1}, {0.0001f, 1, 30, false, 0.001},
  {0.0001f, 1, 1, false, 0.001},     {0.0001f, 17, 5, false, 0.001},
  {0.0001f, 7, 23, false, 0.001},    {0.0001f, 3, 97, false, 0.001}};

const std::vector<SumInputs<double>> inputsd = {
  {0.000001, 1024, 32, true, 1},    {0.000001, 1024, 256, true, 1},
  {0.000001, 1024, 256, true, 1},   {0.000001, 100000000, 1, true, 0.001},
  {1e-9, 1 << 27, 2, true, 0.1},    {0.000001, 1, 30, true, 0.0001},
  {0.000001, 1, 1, true, 0.0001},   {0.000001, 17, 5, true, 0.0001},
  {0.000001, 7, 23, true, 0.0001},  {0.000001, 3, 97, true, 0.0001},
  {0.000001, 1024, 32, false, 1},   {0.000001, 1024, 256, false, 1},
  {0.000001, 1024, 256, false, 1},  {0.000001, 100000000, 1, false, 0.001},
  {1e-9, 1 << 27, 2, false, 0.1},   {0.000001, 1, 30, false, 0.0001},
  {0.000001, 1, 1, false, 0.0001},  {0.000001, 17, 5, false, 0.0001},
  {0.000001, 7, 23, false, 0.0001}, {0.000001, 3, 97, false, 0.0001}};

typedef SumTest<float> SumTestF;
typedef SumTest<double> SumTestD;

TEST_P(SumTestF, Result) { runTest(); }
TEST_P(SumTestD, Result) { runTest(); }

TEST_P(SumTestF, Accuracy) { runTest(true); }
TEST_P(SumTestD, Accuracy) { runTest(true); }

INSTANTIATE_TEST_CASE_P(SumTests, SumTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(SumTests, SumTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
