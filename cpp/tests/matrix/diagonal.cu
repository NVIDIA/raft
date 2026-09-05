/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

template <typename T>
struct DiagonalInputs {
  int n_rows;
  int n_cols;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const DiagonalInputs<T>& dims)
{
  return os;
}

template <typename T>
class DiagonalTest : public ::testing::TestWithParam<DiagonalInputs<T>> {
 public:
  DiagonalTest()
    : params(::testing::TestWithParam<DiagonalInputs<T>>::GetParam()),
      input(raft::make_device_matrix<T, std::uint32_t>(handle, params.n_rows, params.n_cols)),
      diag_expected(raft::make_device_vector<T, std::uint32_t>(handle, diag_size)),
      diag_actual(raft::make_device_vector<T, std::uint32_t>(handle, diag_size)),
      diag_size(std::min(params.n_rows, params.n_cols))
  {
    T mat_fill_scalar  = 1.0;
    T diag_fill_scalar = 5.0;

    auto input_view = raft::make_device_matrix_view<const T, std::uint32_t>(
      input.data_handle(), input.extent(0), input.extent(1));
    auto diag_expected_view =
      raft::make_device_vector_view<const T, std::uint32_t>(diag_expected.data_handle(), diag_size);

    raft::matrix::fill(
      handle, input_view, input.view(), raft::make_host_scalar_view<T>(&mat_fill_scalar));
    raft::matrix::fill(handle,
                       diag_expected_view,
                       diag_expected.view(),
                       raft::make_host_scalar_view<T>(&diag_fill_scalar));

    resource::sync_stream(handle);

    raft::matrix::set_diagonal(handle, diag_expected_view, input.view());

    resource::sync_stream(handle);

    raft::matrix::get_diagonal(handle, input_view, diag_actual.view());

    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  DiagonalInputs<T> params;

  int diag_size;

  raft::device_matrix<T, std::uint32_t> input;
  raft::device_vector<T, std::uint32_t> diag_expected;
  raft::device_vector<T, std::uint32_t> diag_actual;
};

const std::vector<DiagonalInputs<float>> inputsf = {{4, 4}, {4, 10}, {10, 4}};

const std::vector<DiagonalInputs<double>> inputsd = {{4, 4}, {4, 10}, {10, 4}};

typedef DiagonalTest<float> DiagonalTestF;
TEST_P(DiagonalTestF, Result)
{
  ASSERT_TRUE(devArrMatch(diag_expected.data_handle(),
                          diag_actual.data_handle(),
                          diag_size,
                          Compare<float>(),
                          resource::get_cuda_stream(handle).get()));
}

typedef DiagonalTest<double> DiagonalTestD;
TEST_P(DiagonalTestD, Result)
{
  ASSERT_TRUE(devArrMatch(diag_expected.data_handle(),
                          diag_actual.data_handle(),
                          diag_size,
                          Compare<double>(),
                          resource::get_cuda_stream(handle).get()));
}

INSTANTIATE_TEST_SUITE_P(DiagonalTest, DiagonalTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(DiagonalTest, DiagonalTestD, ::testing::ValuesIn(inputsd));

}  // namespace matrix
}  // namespace raft
