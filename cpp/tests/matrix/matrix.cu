/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

template <typename T>
struct MatrixInputs {
  T tolerance;
  int n_row;
  int n_col;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MatrixInputs<T>& dims)
{
  return os;
}

template <typename T>
class MatrixTest : public ::testing::TestWithParam<MatrixInputs<T>> {
 public:
  MatrixTest()
    : params(::testing::TestWithParam<MatrixInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      in1(params.n_row * params.n_col, stream),
      in2(params.n_row * params.n_col, stream),
      in1_revr(params.n_row * params.n_col, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int len = params.n_row * params.n_col;
    uniform(handle, r, in1.data(), len, T(-1.0), T(1.0));

    auto in1_view = raft::make_device_matrix_view<const T, int, col_major>(
      in1.data(), params.n_row, params.n_col);
    auto in2_view =
      raft::make_device_matrix_view<T, int, col_major>(in2.data(), params.n_row, params.n_col);

    copy<T, int>(handle, in1_view, in2_view);
    // copy(in1, in1_revr, params.n_row, params.n_col);
    // colReverse(in1_revr, params.n_row, params.n_col);

    rmm::device_uvector<T> outTrunc(6, stream);

    auto out_trunc_view = raft::make_device_matrix_view<T, int, col_major>(outTrunc.data(), 3, 2);
    trunc_zero_origin<T, int>(handle, in1_view, out_trunc_view);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MatrixInputs<T> params;
  rmm::device_uvector<T> in1, in2, in1_revr;
};

const std::vector<MatrixInputs<float>> inputsf2 = {{0.000001f, 4, 4, 1234ULL}};

const std::vector<MatrixInputs<double>> inputsd2 = {{0.00000001, 4, 4, 1234ULL}};

typedef MatrixTest<float> MatrixTestF;
TEST_P(MatrixTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(in1.data(),
                                in2.data(),
                                params.n_row * params.n_col,
                                raft::CompareApprox<float>(params.tolerance),
                                stream));
}

typedef MatrixTest<double> MatrixTestD;
TEST_P(MatrixTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(in1.data(),
                                in2.data(),
                                params.n_row * params.n_col,
                                raft::CompareApprox<double>(params.tolerance),
                                stream));
}

INSTANTIATE_TEST_SUITE_P(MatrixTests, MatrixTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(MatrixTests, MatrixTestD, ::testing::ValuesIn(inputsd2));

template <typename T>
class MatrixCopyRowsTest : public ::testing::Test {
  using math_t      = typename std::tuple_element<0, T>::type;
  using idx_t       = typename std::tuple_element<1, T>::type;
  using idx_array_t = typename std::tuple_element<2, T>::type;

 protected:
  MatrixCopyRowsTest()
    : stream(resource::get_cuda_stream(handle).get()),
      input(n_cols * n_rows, resource::get_cuda_stream(handle).get()),
      indices(n_selected, resource::get_cuda_stream(handle)),
      output(n_cols * n_selected, resource::get_cuda_stream(handle))
  {
    raft::update_device(indices.data(), indices_host, n_selected, stream);
    // Init input array
    cuda::counting_iterator<idx_t> first(0);
    thrust::device_ptr<math_t> ptr(input.data());
    thrust::copy(resource::get_thrust_policy(handle), first, first + n_cols * n_rows, ptr);
  }

  void testCopyRows()
  {
    auto input_view = raft::make_device_matrix_view<const math_t, idx_array_t, col_major>(
      input.data(), n_rows, n_cols);
    auto output_view = raft::make_device_matrix_view<math_t, idx_array_t, col_major>(
      output.data(), n_selected, n_cols);

    auto indices_view =
      raft::make_device_vector_view<const idx_array_t, idx_array_t>(indices.data(), n_selected);

    raft::matrix::copy_rows(handle, input_view, output_view, indices_view);

    EXPECT_TRUE(raft::devArrMatchHost(
      output_exp_colmajor, output.data(), n_selected * n_cols, raft::Compare<math_t>(), stream));

    auto input_row_view = raft::make_device_matrix_view<const math_t, idx_array_t, row_major>(
      input.data(), n_rows, n_cols);
    auto output_row_view = raft::make_device_matrix_view<math_t, idx_array_t, row_major>(
      output.data(), n_selected, n_cols);

    raft::matrix::copy_rows(handle, input_row_view, output_row_view, indices_view);
    EXPECT_TRUE(raft::devArrMatchHost(
      output_exp_rowmajor, output.data(), n_selected * n_cols, raft::Compare<math_t>(), stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  int n_rows     = 10;
  int n_cols     = 3;
  int n_selected = 5;

  idx_array_t indices_host[5]    = {0, 3, 4, 7, 9};
  math_t output_exp_colmajor[15] = {0, 3, 4, 7, 9, 10, 13, 14, 17, 19, 20, 23, 24, 27, 29};
  math_t output_exp_rowmajor[15] = {0, 1, 2, 9, 10, 11, 12, 13, 14, 21, 22, 23, 27, 28, 29};
  rmm::device_uvector<math_t> input;
  rmm::device_uvector<math_t> output;
  rmm::device_uvector<idx_array_t> indices;
};

using TypeTuple = ::testing::Types<std::tuple<float, int, int>,
                                   std::tuple<float, int64_t, int>,
                                   std::tuple<double, int, int>,
                                   std::tuple<double, int64_t, int>>;

TYPED_TEST_CASE(MatrixCopyRowsTest, TypeTuple);
TYPED_TEST(MatrixCopyRowsTest, CopyRows) { this->testCopyRows(); }
}  // namespace matrix
}  // namespace raft
