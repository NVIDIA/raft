/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/diagonal.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft::matrix {

template <typename T>
struct InitInputs {
  int n_row;
  int n_col;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const InitInputs<T>& dims)
{
  return os;
}

template <typename T>
class InitTest : public ::testing::TestWithParam<InitInputs<T>> {
 public:
  InitTest()
    : params(::testing::TestWithParam<InitInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get())
  {
  }

 protected:
  void test_eye()
  {
    ASSERT_TRUE(params.n_row == 4 && params.n_col == 5);
    auto eyemat_col =
      raft::make_device_matrix<T, int, raft::col_major>(handle, params.n_row, params.n_col);
    raft::matrix::eye(handle, eyemat_col.view());
    std::vector<T> eye_exp{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
    std::vector<T> eye_act(params.n_col * params.n_row);
    raft::copy(eye_act.data(), eyemat_col.data_handle(), eye_act.size(), stream);
    resource::sync_stream(handle, stream);
    ASSERT_TRUE(hostVecMatch(eye_exp, eye_act, raft::Compare<T>()));

    auto eyemat_row =
      raft::make_device_matrix<T, int, raft::row_major>(handle, params.n_row, params.n_col);
    raft::matrix::eye(handle, eyemat_row.view());
    eye_exp = std::vector<T>{1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0};
    raft::copy(eye_act.data(), eyemat_row.data_handle(), eye_act.size(), stream);
    resource::sync_stream(handle, stream);
    ASSERT_TRUE(hostVecMatch(eye_exp, eye_act, raft::Compare<T>()));
  }

  void SetUp() override { test_eye(); }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  InitInputs<T> params;
};

const std::vector<InitInputs<float>> inputsf1 = {{4, 5}};

const std::vector<InitInputs<double>> inputsd1 = {{4, 5}};

typedef InitTest<float> InitTestF;
TEST_P(InitTestF, Result) {}

typedef InitTest<double> InitTestD;
TEST_P(InitTestD, Result) {}

INSTANTIATE_TEST_SUITE_P(InitTests, InitTestF, ::testing::ValuesIn(inputsf1));
INSTANTIATE_TEST_SUITE_P(InitTests, InitTestD, ::testing::ValuesIn(inputsd1));

}  // namespace raft::matrix
