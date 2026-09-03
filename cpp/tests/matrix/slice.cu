/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/slice.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

template <typename T>
struct SliceInputs {
  int rows, cols;
  unsigned long long int seed;
  bool rowMajor;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SliceInputs<T>& I)
{
  os << "{ " << I.rows << ", " << I.cols << ", " << I.seed << ", " << I.rowMajor << '}'
     << std::endl;
  return os;
}

// Col-major slice reference test
template <typename Type>
void naiveSlice(
  const Type* in, Type* out, int in_lda, int x1, int y1, int x2, int y2, bool row_major)
{
  int out_lda = row_major ? y2 - y1 : x2 - x1;
  for (int j = y1; j < y2; ++j) {
    for (int i = x1; i < x2; ++i) {
      if (row_major)
        out[(i - x1) * out_lda + (j - y1)] = in[j + i * in_lda];
      else
        out[(i - x1) + (j - y1) * out_lda] = in[i + j * in_lda];
    }
  }
}

template <typename T>
class SliceTest : public ::testing::TestWithParam<SliceInputs<T>> {
 public:
  SliceTest()
    : params(::testing::TestWithParam<SliceInputs<T>>::GetParam()),
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
    auto lda = params.rowMajor ? cols : rows;
    uniform(handle, r, data.data(), len, T(-10.0), T(10.0));

    std::uniform_int_distribution<int> rowGenerator(0, (rows / 2) - 1);
    auto row1 = rowGenerator(dre);
    auto row2 = rowGenerator(dre) + rows / 2;

    std::uniform_int_distribution<int> colGenerator(0, (cols / 2) - 1);
    auto col1 = colGenerator(dre);
    auto col2 = colGenerator(dre) + cols / 2;

    rmm::device_uvector<T> d_act_result((row2 - row1) * (col2 - col1), stream);
    act_result.resize((row2 - row1) * (col2 - col1));
    exp_result.resize((row2 - row1) * (col2 - col1));

    std::vector<T> h_data(rows * cols);
    raft::update_host(h_data.data(), data.data(), rows * cols, stream);
    naiveSlice(h_data.data(), exp_result.data(), lda, row1, col1, row2, col2, params.rowMajor);
    auto input_F =
      raft::make_device_matrix_view<const T, int, raft::col_major>(data.data(), rows, cols);
    auto output_F = raft::make_device_matrix_view<T, int, raft::col_major>(
      d_act_result.data(), row2 - row1, col2 - col1);
    auto input_C =
      raft::make_device_matrix_view<const T, int, raft::row_major>(data.data(), rows, cols);
    auto output_C = raft::make_device_matrix_view<T, int, raft::row_major>(
      d_act_result.data(), row2 - row1, col2 - col1);
    if (params.rowMajor)
      slice(handle, input_C, output_C, slice_coordinates(row1, col1, row2, col2));
    else
      slice(handle, input_F, output_F, slice_coordinates(row1, col1, row2, col2));

    raft::update_host(act_result.data(), d_act_result.data(), d_act_result.size(), stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SliceInputs<T> params;
  rmm::device_uvector<T> data;
  std::vector<T> exp_result, act_result;
};

///// Row- and column-wise tests
const std::vector<SliceInputs<float>> inputsf = {{32, 1024, 1234ULL, true},
                                                 {64, 1024, 1234ULL, false},
                                                 {128, 1024, 1234ULL, true},
                                                 {256, 1024, 1234ULL, false},
                                                 {512, 512, 1234ULL, true},
                                                 {1024, 32, 1234ULL, false},
                                                 {1024, 64, 1234ULL, true},
                                                 {1024, 128, 1234ULL, false},
                                                 {1024, 256, 1234ULL, true}};

const std::vector<SliceInputs<double>> inputsd = {
  {32, 1024, 1234ULL, true},
  {64, 1024, 1234ULL, false},
  {128, 1024, 1234ULL, true},
  {256, 1024, 1234ULL, false},
  {512, 512, 1234ULL, true},
  {1024, 32, 1234ULL, false},
  {1024, 64, 1234ULL, true},
  {1024, 128, 1234ULL, false},
  {1024, 256, 1234ULL, true},
};

typedef SliceTest<float> SliceTestF;
TEST_P(SliceTestF, Result)
{
  ASSERT_TRUE(hostVecMatch(exp_result, act_result, raft::Compare<float>()));
}

typedef SliceTest<double> SliceTestD;
TEST_P(SliceTestD, Result)
{
  ASSERT_TRUE(hostVecMatch(exp_result, act_result, raft::Compare<double>()));
}

INSTANTIATE_TEST_CASE_P(SliceTests, SliceTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(SliceTests, SliceTestD, ::testing::ValuesIn(inputsd));

}  // end namespace matrix
}  // end namespace raft
