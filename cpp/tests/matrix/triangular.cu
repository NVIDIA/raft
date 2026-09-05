/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/init.cuh>
#include <raft/matrix/triangular.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

template <typename T>
struct TriangularInputs {
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const TriangularInputs<T>& I)
{
  os << "{ " << I.rows << ", " << I.cols << ", " << I.seed << '}' << std::endl;
  return os;
}

// triangular reference test
template <typename Type>
void naive_triangular(std::vector<Type>& in, std::vector<Type>& out, int rows, int cols)
{
  auto k = std::min(rows, cols);
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j <= i; ++j) {
      auto index     = i * rows + j;
      out[i * k + j] = in[index];
    }
  }
}

template <typename T>
class TriangularTest : public ::testing::TestWithParam<TriangularInputs<T>> {
 public:
  TriangularTest()
    : params(::testing::TestWithParam<TriangularInputs<T>>::GetParam()),
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
    auto k = std::min(rows, cols);

    rmm::device_uvector<T> d_act_result(len, stream);
    std::vector<T> h_data(len);
    act_result.resize(k * k);
    exp_result.resize(k * k);

    uniform(handle, r, data.data(), len, T(-10.0), T(10.0));
    raft::update_host(h_data.data(), data.data(), len, stream);
    raft::matrix::fill(
      handle,
      raft::make_device_matrix_view<T, int, raft::col_major>(d_act_result.data(), k, k),
      T(0));

    upper_triangular(
      handle,
      raft::make_device_matrix_view<const T, int, raft::col_major>(data.data(), rows, cols),
      raft::make_device_matrix_view<T, int, raft::col_major>(d_act_result.data(), k, k));
    naive_triangular(h_data, exp_result, rows, cols);

    raft::update_host(act_result.data(), d_act_result.data(), k * k, stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  TriangularInputs<T> params;
  rmm::device_uvector<T> data;
  std::vector<T> exp_result, act_result;
};

///// Row- and column-wise tests
const std::vector<TriangularInputs<float>> inputsf = {{4, 4, 1234ULL},
                                                      {2, 64, 1234ULL},
                                                      {64, 512, 1234ULL},
                                                      {64, 1024, 1234ULL},
                                                      {256, 1024, 1234ULL},
                                                      {512, 512, 1234ULL},
                                                      {1024, 32, 1234ULL},
                                                      {1024, 128, 1234ULL},
                                                      {1024, 256, 1234ULL}};

const std::vector<TriangularInputs<double>> inputsd = {{4, 4, 1234ULL},
                                                       {2, 64, 1234ULL},
                                                       {64, 512, 1234ULL},
                                                       {64, 1024, 1234ULL},
                                                       {256, 1024, 1234ULL},
                                                       {512, 512, 1234ULL},
                                                       {1024, 32, 1234ULL},
                                                       {1024, 128, 1234ULL},
                                                       {1024, 256, 1234ULL}};

typedef TriangularTest<float> TriangularTestF;
TEST_P(TriangularTestF, Result)
{
  ASSERT_TRUE(hostVecMatch(exp_result, act_result, raft::Compare<float>()));
}

typedef TriangularTest<double> TriangularTestD;
TEST_P(TriangularTestD, Result)
{
  ASSERT_TRUE(hostVecMatch(exp_result, act_result, raft::Compare<double>()));
}

INSTANTIATE_TEST_CASE_P(TriangularTests, TriangularTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(TriangularTests, TriangularTestD, ::testing::ValuesIn(inputsd));

}  // end namespace matrix
}  // end namespace raft
