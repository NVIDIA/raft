/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/reverse.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

template <typename T>
struct ReverseInputs {
  bool row_major, row_reverse;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const ReverseInputs<T>& I)
{
  os << "{ " << I.row_major << ", " << I.row_reverse << ", " << I.rows << ", " << I.cols << ", "
     << I.seed << '}' << std::endl;
  return os;
}

// col-reverse reference test
template <typename Type>
void naive_col_reverse(std::vector<Type>& data, int rows, int cols, bool row_major)
{
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols / 2; ++j) {
      auto index_in   = row_major ? i * cols + j : i + j * rows;
      auto index_out  = row_major ? i * cols + (cols - j - 1) : i + (cols - j - 1) * rows;
      auto tmp        = data[index_in];
      data[index_in]  = data[index_out];
      data[index_out] = tmp;
    }
  }
}

// row-reverse reference test
template <typename Type>
void naive_row_reverse(std::vector<Type>& data, int rows, int cols, bool row_major)
{
  for (int i = 0; i < rows / 2; ++i) {
    for (int j = 0; j < cols; ++j) {
      auto index_in   = row_major ? i * cols + j : i + j * rows;
      auto index_out  = row_major ? (rows - i - 1) * cols + j : (rows - i - 1) + j * rows;
      auto tmp        = data[index_in];
      data[index_in]  = data[index_out];
      data[index_out] = tmp;
    }
  }
}

template <typename T>
class ReverseTest : public ::testing::TestWithParam<ReverseInputs<T>> {
 public:
  ReverseTest()
    : params(::testing::TestWithParam<ReverseInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      data(params.rows * params.cols, stream)
  {
  }

  void SetUp() override
  {
    std::random_device rd;
    std::default_random_engine dre(rd());
    raft::random::RngState r(params.seed);
    int rows = params.rows, cols = params.cols, len = rows * cols;

    act_result.resize(len);
    exp_result.resize(len);

    uniform(handle, r, data.data(), len, T(-10.0), T(10.0));
    raft::update_host(exp_result.data(), data.data(), len, stream);

    auto input_col_major =
      raft::make_device_matrix_view<T, int, raft::col_major>(data.data(), rows, cols);
    auto input_row_major =
      raft::make_device_matrix_view<T, int, raft::row_major>(data.data(), rows, cols);
    if (params.row_major) {
      if (params.row_reverse) {
        row_reverse(handle, input_row_major);
        naive_row_reverse(exp_result, rows, cols, params.row_major);
      } else {
        col_reverse(handle, input_row_major);
        naive_col_reverse(exp_result, rows, cols, params.row_major);
      }
    } else {
      if (params.row_reverse) {
        row_reverse(handle, input_col_major);
        naive_row_reverse(exp_result, rows, cols, params.row_major);
      } else {
        col_reverse(handle, input_col_major);
        naive_col_reverse(exp_result, rows, cols, params.row_major);
      }
    }

    raft::update_host(act_result.data(), data.data(), len, stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  ReverseInputs<T> params;
  rmm::device_uvector<T> data;
  std::vector<T> exp_result, act_result;
};

///// Row- and column-wise tests
const std::vector<ReverseInputs<float>> inputsf = {{true, true, 4, 4, 1234ULL},
                                                   {true, true, 2, 12, 1234ULL},
                                                   {true, false, 2, 12, 1234ULL},
                                                   {true, false, 2, 64, 1234ULL},
                                                   {true, true, 64, 512, 1234ULL},
                                                   {true, false, 64, 1024, 1234ULL},
                                                   {true, true, 128, 1024, 1234ULL},
                                                   {true, false, 256, 1024, 1234ULL},
                                                   {false, true, 512, 512, 1234ULL},
                                                   {false, false, 1024, 32, 1234ULL},
                                                   {false, true, 1024, 64, 1234ULL},
                                                   {false, false, 1024, 128, 1234ULL},
                                                   {false, true, 1024, 256, 1234ULL}};

const std::vector<ReverseInputs<double>> inputsd = {{true, true, 4, 4, 1234ULL},
                                                    {true, true, 2, 12, 1234ULL},
                                                    {true, false, 2, 12, 1234ULL},
                                                    {true, false, 2, 64, 1234ULL},
                                                    {true, true, 64, 512, 1234ULL},
                                                    {true, false, 64, 1024, 1234ULL},
                                                    {true, true, 128, 1024, 1234ULL},
                                                    {true, false, 256, 1024, 1234ULL},
                                                    {false, true, 512, 512, 1234ULL},
                                                    {false, false, 1024, 32, 1234ULL},
                                                    {false, true, 1024, 64, 1234ULL},
                                                    {false, false, 1024, 128, 1234ULL},
                                                    {false, true, 1024, 256, 1234ULL}};

typedef ReverseTest<float> ReverseTestF;
TEST_P(ReverseTestF, Result)
{
  ASSERT_TRUE(hostVecMatch(exp_result, act_result, raft::Compare<float>()));
}

typedef ReverseTest<double> ReverseTestD;
TEST_P(ReverseTestD, Result)
{
  ASSERT_TRUE(hostVecMatch(exp_result, act_result, raft::Compare<double>()));
}

INSTANTIATE_TEST_CASE_P(ReverseTests, ReverseTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(ReverseTests, ReverseTestD, ::testing::ValuesIn(inputsd));

}  // end namespace matrix
}  // end namespace raft
