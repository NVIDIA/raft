/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

//// Testing unary ops

template <typename Type>
RAFT_KERNEL naiveScaleKernel(Type* out, const Type* in, Type scalar, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = scalar * in[idx]; }
}

template <typename Type>
void naiveScale(Type* out, const Type* in, Type scalar, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(stream, nblks, TPB, naiveScaleKernel<Type>, out, in, scalar, len);
}

template <typename T>
struct ScalarMultiplyInputs {
  T tolerance;
  int len;
  T scalar;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const ScalarMultiplyInputs<T>& dims)
{
  return os;
}

template <typename T>
class ScalarMultiplyTest : public ::testing::TestWithParam<ScalarMultiplyInputs<T>> {
 public:
  ScalarMultiplyTest()
    : params(::testing::TestWithParam<ScalarMultiplyInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      in(len, stream),
      out_ref(len, stream),
      out(len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int len  = params.len;
    T scalar = params.scalar;
    uniform(handle, r, in, len, T(-1.0), T(1.0));
    naiveScale(out_ref, in, scalar, len, stream);
    scalarMultiply(out, in, scalar, len, stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  ScalarMultiplyInputs<T> params;
  rmm::device_uvector<T> in, out_ref, out;
};

const std::vector<ScalarMultiplyInputs<float>> inputsf1 = {{0.000001f, 1024 * 1024, 2.f, 1234ULL}};

const std::vector<ScalarMultiplyInputs<double>> inputsd1 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};

typedef ScalarMultiplyTest<float> ScalarMultiplyTestF;
TEST_P(ScalarMultiplyTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<float>(params.tolerance), stream));
}

typedef ScalarMultiplyTest<double> ScalarMultiplyTestD;
TEST_P(ScalarMultiplyTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance), stream));
}

INSTANTIATE_TEST_SUITE_P(ScalarMultiplyTests, ScalarMultiplyTestF, ::testing::ValuesIn(inputsf1));

INSTANTIATE_TEST_SUITE_P(ScalarMultiplyTests, ScalarMultiplyTestD, ::testing::ValuesIn(inputsd1));

//// Testing binary ops

template <typename Type>
RAFT_KERNEL naiveAddKernel(Type* out, const Type* in1, const Type* in2, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = in1[idx] + in2[idx]; }
}

template <typename Type>
void naiveAdd(Type* out, const Type* in1, const Type* in2, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(stream, nblks, TPB, naiveAddKernel<Type>, out, in1, in2, len);
}

template <typename T>
struct EltwiseAddInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const EltwiseAddInputs<T>& dims)
{
  return os;
}

template <typename T>
class EltwiseAddTest : public ::testing::TestWithParam<EltwiseAddInputs<T>> {
 public:
  EltwiseAddTest()
    : params(::testing::TestWithParam<EltwiseAddInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      in1(params.len, stream),
      in2(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<EltwiseAddInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    int len = params.len;
    uniform(handle, r, in1, len, T(-1.0), T(1.0));
    uniform(handle, r, in2, len, T(-1.0), T(1.0));
    naiveAdd(out_ref, in1, in2, len, stream);
    eltwiseAdd(out, in1, in2, len, stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  EltwiseAddInputs<T> params;
  rmm::device_uvector<T> in1, in2, out_ref, out;
};

const std::vector<EltwiseAddInputs<float>> inputsf2 = {{0.000001f, 1024 * 1024, 1234ULL}};

const std::vector<EltwiseAddInputs<double>> inputsd2 = {{0.00000001, 1024 * 1024, 1234ULL}};

typedef EltwiseAddTest<float> EltwiseAddTestF;
TEST_P(EltwiseAddTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<float>(params.tolerance), stream));
}

typedef EltwiseAddTest<double> EltwiseAddTestD;
TEST_P(EltwiseAddTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance), stream));
}

INSTANTIATE_TEST_SUITE_P(EltwiseAddTests, EltwiseAddTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(EltwiseAddTests, EltwiseAddTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
