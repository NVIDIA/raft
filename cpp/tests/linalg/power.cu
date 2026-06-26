/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/power.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename Type>
RAFT_KERNEL naivePowerElemKernel(Type* out, const Type* in1, const Type* in2, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = raft::pow(in1[idx], in2[idx]); }
}

template <typename Type>
void naivePowerElem(Type* out, const Type* in1, const Type* in2, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  naivePowerElemKernel<Type><<<nblks, TPB, 0, stream>>>(out, in1, in2, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename Type>
RAFT_KERNEL naivePowerScalarKernel(Type* out, const Type* in1, const Type in2, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = raft::pow(in1[idx], in2); }
}

template <typename Type>
void naivePowerScalar(Type* out, const Type* in1, const Type in2, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  naivePowerScalarKernel<Type><<<nblks, TPB, 0, stream>>>(out, in1, in2, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
struct PowerInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const PowerInputs<T>& dims)
{
  return os;
}

template <typename T>
class PowerTest : public ::testing::TestWithParam<PowerInputs<T>> {
 protected:
  PowerTest()
    : in1(0, resource::get_cuda_stream(handle)),
      in2(0, resource::get_cuda_stream(handle)),
      out_ref(0, resource::get_cuda_stream(handle)),
      out(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    params = ::testing::TestWithParam<PowerInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    int len = params.len;

    cudaStream_t stream = resource::get_cuda_stream(handle);

    in1.resize(len, stream);
    in2.resize(len, stream);
    out_ref.resize(len, stream);
    out.resize(len, stream);
    uniform(handle, r, in1.data(), len, T(1.0), T(2.0));
    uniform(handle, r, in2.data(), len, T(1.0), T(2.0));

    naivePowerElem(out_ref.data(), in1.data(), in2.data(), len, stream);
    naivePowerScalar(out_ref.data(), out_ref.data(), T(2), len, stream);

    auto out_view       = raft::make_device_vector_view(out.data(), len);
    auto in1_view       = raft::make_device_vector_view(in1.data(), len);
    auto const_out_view = raft::make_device_vector_view<const T>(out.data(), len);
    auto const_in1_view = raft::make_device_vector_view<const T>(in1.data(), len);
    auto const_in2_view = raft::make_device_vector_view<const T>(in2.data(), len);
    const auto scalar   = static_cast<T>(2);
    auto scalar_view    = raft::make_host_scalar_view(&scalar);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        power(h, const_in1_view, const_in2_view, out_view);
        power_scalar(h, const_out_view, out_view, scalar_view);
        power(h, const_in1_view, const_in2_view, in1_view);
        power_scalar(h, const_in1_view, in1_view, scalar_view);
      },
      raft::alloc_behavior::NO_ALLOCATIONS);

    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  PowerInputs<T> params;
  rmm::device_uvector<T> in1, in2, out_ref, out;
  int device_count = 0;
};

const std::vector<PowerInputs<float>> inputsf2 = {{0.000001f, 1024 * 1024, 1234ULL}};

const std::vector<PowerInputs<double>> inputsd2 = {{0.00000001, 1024 * 1024, 1234ULL}};

typedef PowerTest<float> PowerTestF;
TEST_P(PowerTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), in1.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}

typedef PowerTest<double> PowerTestD;
TEST_P(PowerTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), in1.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(PowerTests, PowerTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PowerTests, PowerTestD, ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
