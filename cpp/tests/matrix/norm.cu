/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

template <typename T>
struct NormInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const NormInputs<T>& I)
{
  os << "{ " << I.tolerance << ", " << I.rows << ", " << I.cols << ", " << I.seed << '}'
     << std::endl;
  return os;
}

template <typename Type>
Type naiveNorm(const Type* data, int D, int N)
{
  Type out_scalar = 0;
  for (int i = 0; i < N * D; ++i) {
    out_scalar += data[i] * data[i];
  }
  out_scalar = std::sqrt(out_scalar);
  return out_scalar;
}

template <typename T>
class NormTest : public ::testing::TestWithParam<NormInputs<T>> {
 public:
  NormTest()
    : params(::testing::TestWithParam<NormInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      data(params.rows * params.cols, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int rows = params.rows, cols = params.cols, len = rows * cols;
    uniform(handle, r, data.data(), len, T(-10.0), T(10.0));
    std::vector<T> h_data(rows * cols);
    raft::update_host(h_data.data(), data.data(), rows * cols, stream);
    out_scalar_exp = naiveNorm(h_data.data(), cols, rows);
    auto input = raft::make_device_matrix_view<const T, int>(data.data(), params.rows, params.cols);
    out_scalar_act = l2_norm(handle, input);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  NormInputs<T> params;
  rmm::device_uvector<T> data;
  T out_scalar_exp = 0;
  T out_scalar_act = 0;
};

///// Row- and column-wise tests
const std::vector<NormInputs<float>> inputsf = {{0.00001f, 32, 1024, 1234ULL},
                                                {0.00001f, 64, 1024, 1234ULL},
                                                {0.00001f, 128, 1024, 1234ULL},
                                                {0.00001f, 256, 1024, 1234ULL},
                                                {0.00001f, 512, 512, 1234ULL},
                                                {0.00001f, 1024, 32, 1234ULL},
                                                {0.00001f, 1024, 64, 1234ULL},
                                                {0.00001f, 1024, 128, 1234ULL},
                                                {0.00001f, 1024, 256, 1234ULL}};

const std::vector<NormInputs<double>> inputsd = {
  {0.00000001, 32, 1024, 1234ULL},
  {0.00000001, 64, 1024, 1234ULL},
  {0.00000001, 128, 1024, 1234ULL},
  {0.00000001, 256, 1024, 1234ULL},
  {0.00000001, 512, 512, 1234ULL},
  {0.00000001, 1024, 32, 1234ULL},
  {0.00000001, 1024, 64, 1234ULL},
  {0.00000001, 1024, 128, 1234ULL},
  {0.00000001, 1024, 256, 1234ULL},
};

typedef NormTest<float> NormTestF;
TEST_P(NormTestF, Result)
{
  ASSERT_NEAR(out_scalar_exp, out_scalar_act, params.tolerance * params.rows * params.cols);
}

typedef NormTest<double> NormTestD;
TEST_P(NormTestD, Result)
{
  ASSERT_NEAR(out_scalar_exp, out_scalar_act, params.tolerance * params.rows * params.cols);
}

INSTANTIATE_TEST_CASE_P(NormTests, NormTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(NormTests, NormTestD, ::testing::ValuesIn(inputsd));

}  // end namespace matrix
}  // end namespace raft
