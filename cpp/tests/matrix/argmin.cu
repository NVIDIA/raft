/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/argmin.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace raft {
namespace matrix {

template <typename T, typename IdxT>
struct ArgMinInputs {
  std::vector<T> input_matrix;
  std::vector<IdxT> output_matrix;
  std::size_t n_rows;
  std::size_t n_cols;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const ArgMinInputs<T, IdxT>& dims)
{
  return os;
}

template <typename T, typename IdxT>
class ArgMinTest : public ::testing::TestWithParam<ArgMinInputs<T, IdxT>> {
 public:
  ArgMinTest()
    : params(::testing::TestWithParam<ArgMinInputs<T, IdxT>>::GetParam()),
      input(raft::make_device_matrix<T, std::uint32_t, row_major>(
        handle, params.n_rows, params.n_cols)),
      output(raft::make_device_vector<IdxT, std::uint32_t>(handle, params.n_rows)),
      expected(raft::make_device_vector<IdxT, std::uint32_t>(handle, params.n_rows))
  {
    raft::update_device(input.data_handle(),
                        params.input_matrix.data(),
                        params.input_matrix.size(),
                        resource::get_cuda_stream(handle));
    raft::update_device(expected.data_handle(),
                        params.output_matrix.data(),
                        params.output_matrix.size(),
                        resource::get_cuda_stream(handle).get());

    auto input_const_view = raft::make_device_matrix_view<const T, std::uint32_t, row_major>(
      input.data_handle(), input.extent(0), input.extent(1));

    raft::matrix::argmin(handle, input_const_view, output.view());

    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  ArgMinInputs<T, IdxT> params;

  raft::device_matrix<T, std::uint32_t, row_major> input;
  raft::device_vector<IdxT, std::uint32_t> output;
  raft::device_vector<IdxT, std::uint32_t> expected;
};

const std::vector<ArgMinInputs<float, int>> inputsf = {
  {{0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f}, {0, 3, 3}, 3, 4}};

const std::vector<ArgMinInputs<double, int>> inputsd = {
  {{0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.5, 0.0}, {0, 3, 3}, 3, 4}};

typedef ArgMinTest<float, int> ArgMinTestF;
TEST_P(ArgMinTestF, Result)
{
  ASSERT_TRUE(devArrMatch(expected.data_handle(),
                          output.data_handle(),
                          params.n_rows,
                          Compare<int>(),
                          resource::get_cuda_stream(handle).get()));
}

typedef ArgMinTest<double, int> ArgMinTestD;
TEST_P(ArgMinTestD, Result)
{
  ASSERT_TRUE(devArrMatch(expected.data_handle(),
                          output.data_handle(),
                          params.n_rows,
                          Compare<int>(),
                          resource::get_cuda_stream(handle).get()));
}

INSTANTIATE_TEST_SUITE_P(ArgMinTest, ArgMinTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(ArgMinTest, ArgMinTestD, ::testing::ValuesIn(inputsd));

}  // namespace matrix
}  // namespace raft
