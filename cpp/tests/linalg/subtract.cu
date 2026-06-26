/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/subtract.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename Type>
RAFT_KERNEL naiveSubtractElemKernel(Type* out, const Type* in1, const Type* in2, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = in1[idx] - in2[idx]; }
}

template <typename Type>
void naiveSubtractElem(Type* out, const Type* in1, const Type* in2, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  naiveSubtractElemKernel<Type><<<nblks, TPB, 0, stream>>>(out, in1, in2, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Type>
RAFT_KERNEL naiveSubtractScalarKernel(Type* out, const Type* in1, const Type in2, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = in1[idx] - in2; }
}

template <typename Type>
void naiveSubtractScalar(Type* out, const Type* in1, const Type in2, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  naiveSubtractScalarKernel<Type><<<nblks, TPB, 0, stream>>>(out, in1, in2, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
struct SubtractInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SubtractInputs<T>& dims)
{
  return os;
}

template <typename T>
class SubtractTest : public ::testing::TestWithParam<SubtractInputs<T>> {
 public:
  SubtractTest()
    : params(::testing::TestWithParam<SubtractInputs<T>>::GetParam()),
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
    raft::random::RngState r(params.seed);
    int len = params.len;
    uniform(handle, r, in1.data(), len, T(-1.0), T(1.0));
    uniform(handle, r, in2.data(), len, T(-1.0), T(1.0));

    naiveSubtractElem(out_ref.data(), in1.data(), in2.data(), len, stream);
    naiveSubtractScalar(out_ref.data(), out_ref.data(), T(1), len, stream);

    auto out_view       = raft::make_device_vector_view(out.data(), len);
    auto in1_view       = raft::make_device_vector_view(in1.data(), len);
    auto const_out_view = raft::make_device_vector_view<const T>(out.data(), len);
    auto const_in1_view = raft::make_device_vector_view<const T>(in1.data(), len);
    auto const_in2_view = raft::make_device_vector_view<const T>(in2.data(), len);
    const auto scalar   = static_cast<T>(1);
    auto scalar_view    = raft::make_host_scalar_view(&scalar);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        subtract(h, const_in1_view, const_in2_view, out_view);
        subtract_scalar(h, const_out_view, out_view, scalar_view);
        subtract(h, const_in1_view, const_in2_view, in1_view);
        subtract_scalar(h, const_in1_view, in1_view, scalar_view);
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SubtractInputs<T> params;
  rmm::device_uvector<T> in1, in2, out_ref, out;
};

const std::vector<SubtractInputs<float>> inputsf2 = {{0.000001f, 1024 * 1024, 1234ULL}};

const std::vector<SubtractInputs<double>> inputsd2 = {{0.00000001, 1024 * 1024, 1234ULL}};

typedef SubtractTest<float> SubtractTestF;
TEST_P(SubtractTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), in1.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}

typedef SubtractTest<double> SubtractTestD;
TEST_P(SubtractTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), in1.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(SubtractTests, SubtractTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(SubtractTests, SubtractTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
