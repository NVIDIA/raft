/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/mean_squared_error.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

// reference MSE calculation
template <typename T>
RAFT_KERNEL naiveMeanSquaredError(const int n, const T* a, const T* b, T weight, T* out)
{
  T err = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    T diff = a[i] - b[i];
    err += weight * diff * diff / n;
  }
  atomicAdd(out, err);
}

template <typename T, typename IndexType = std::uint32_t>
struct MeanSquaredErrorInputs {
  T tolerance;
  IndexType len;
  T weight;
  unsigned long long int seed;
};

template <typename T>
class MeanSquaredErrorTest : public ::testing::TestWithParam<MeanSquaredErrorInputs<T>> {
 protected:
  MeanSquaredErrorInputs<T> params;

  raft::resources handle;
  rmm::device_scalar<T> output;
  rmm::device_scalar<T> refoutput;

 public:
  MeanSquaredErrorTest()
    : testing::TestWithParam<MeanSquaredErrorInputs<T>>(),
      output(0, resource::get_cuda_stream(handle)),
      refoutput(0, resource::get_cuda_stream(handle))
  {
    resource::sync_stream(handle);
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<MeanSquaredErrorInputs<T>>::GetParam();

    cudaStream_t stream = resource::get_cuda_stream(handle);

    raft::random::RngState r(params.seed);

    rmm::device_uvector<T> a(params.len, stream);
    rmm::device_uvector<T> b(params.len, stream);
    uniform(handle, r, a.data(), params.len, T(-1.0), T(1.0));
    uniform(handle, r, b.data(), params.len, T(-1.0), T(1.0));
    resource::sync_stream(handle);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        mean_squared_error<T, std::uint32_t, T>(
          h,
          make_device_vector_view<const T>(a.data(), params.len),
          make_device_vector_view<const T>(b.data(), params.len),
          make_device_scalar_view<T>(output.data()),
          params.weight);
      },
      raft::alloc_behavior::NO_ALLOCATIONS);

    naiveMeanSquaredError<<<256, 256, 0, stream>>>(
      params.len, a.data(), b.data(), params.weight, refoutput.data());
    resource::sync_stream(handle);
  }

  void TearDown() override {}
};

const std::vector<MeanSquaredErrorInputs<float>> inputsf = {
  {0.0001f, 1024 * 1024, 1.0, 1234ULL},
  {0.0001f, 4 * 1024 * 1024, 8.0, 1234ULL},
  {0.0001f, 16 * 1024 * 1024, 24.0, 1234ULL},
  {0.0001f, 98689, 1.0, 1234ULL},
};

const std::vector<MeanSquaredErrorInputs<double>> inputsd = {
  {0.0001f, 1024 * 1024, 1.0, 1234ULL},
  {0.0001f, 4 * 1024 * 1024, 8.0, 1234ULL},
  {0.0001f, 16 * 1024 * 1024, 24.0, 1234ULL},
  {0.0001f, 98689, 1.0, 1234ULL},
};

typedef MeanSquaredErrorTest<float> MeanSquaredErrorTestF;
TEST_P(MeanSquaredErrorTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refoutput.data(), output.data(), 1, raft::CompareApprox<float>(params.tolerance)));
}

typedef MeanSquaredErrorTest<double> MeanSquaredErrorTestD;
TEST_P(MeanSquaredErrorTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refoutput.data(), output.data(), 1, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(MeanSquaredErrorTests,
                         MeanSquaredErrorTestF,
                         ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MeanSquaredErrorTests,
                         MeanSquaredErrorTestD,
                         ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
