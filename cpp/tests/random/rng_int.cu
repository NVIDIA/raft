/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <cub/block/block_reduce.cuh>

#include <gtest/gtest.h>

namespace raft {
namespace random {

using namespace raft::random;

enum RandomType { RNG_Uniform };

template <typename T, int TPB>
RAFT_KERNEL meanKernel(float* out, const T* data, int len)
{
  typedef cub::BlockReduce<float, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int tid   = threadIdx.x + blockIdx.x * blockDim.x;
  float val = tid < len ? data[tid] : T(0);
  float x   = BlockReduce(temp_storage).Sum(val);
  __syncthreads();
  float xx = BlockReduce(temp_storage).Sum(val * val);
  __syncthreads();
  if (threadIdx.x == 0) {
    raft::myAtomicAdd(out, x);
    raft::myAtomicAdd(out + 1, xx);
  }
}

template <typename T>
struct RngInputs {
  float tolerance;
  int len;
  // start, end: for uniform
  // mean, sigma: for normal/lognormal
  // mean, beta: for gumbel
  // mean, scale: for logistic and laplace
  // lambda: for exponential
  // sigma: for rayleigh
  T start, end;
  RandomType type;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RngInputs<T>& dims)
{
  return os;
}

template <typename T>
class RngTest : public ::testing::TestWithParam<RngInputs<T>> {
 public:
  RngTest()
    : params(::testing::TestWithParam<RngInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(0, stream),
      stats(2, stream)
  {
    data.resize(params.len, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(stats.data(), 0, 2 * sizeof(float), stream));
  }

 protected:
  void SetUp() override
  {
    RngState r(params.seed, params.gtype);

    switch (params.type) {
      case RNG_Uniform:
        uniformInt(handle, r, data.data(), params.len, params.start, params.end);
        break;
    };
    static const int threads = 128;
    raft::launch_kernel(stream,
                        raft::ceildiv(params.len, threads),
                        threads,
                        meanKernel<T, threads>,
                        stats.data(),
                        data.data(),
                        params.len);
    update_host<float>(h_stats, stats.data(), 2, stream);
    resource::sync_stream(handle, stream);
    h_stats[0] /= params.len;
    h_stats[1] = (h_stats[1] / params.len) - (h_stats[0] * h_stats[0]);
    resource::sync_stream(handle, stream);
  }

  void getExpectedMeanVar(float meanvar[2])
  {
    switch (params.type) {
      case RNG_Uniform:
        meanvar[0] = (params.start + params.end) * 0.5f;
        meanvar[1] = params.end - params.start;
        meanvar[1] = meanvar[1] * meanvar[1] / 12.f;
        break;
    };
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RngInputs<T> params;
  rmm::device_uvector<T> data;
  rmm::device_uvector<float> stats;
  float h_stats[2];  // mean, var
};

template <typename T>
class RngMdspanTest : public ::testing::TestWithParam<RngInputs<T>> {
 public:
  RngMdspanTest()
    : params(::testing::TestWithParam<RngInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(0, stream),
      stats(2, stream)
  {
    data.resize(params.len, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(stats.data(), 0, 2 * sizeof(float), stream));
  }

 protected:
  void SetUp() override
  {
    RngState r(params.seed, params.gtype);
    raft::device_vector_view<T> data_view(data.data(), data.size());

    switch (params.type) {
      case RNG_Uniform: uniformInt(handle, r, data_view, params.start, params.end); break;
    };
    static const int threads = 128;
    raft::launch_kernel(stream,
                        raft::ceildiv(params.len, threads),
                        threads,
                        meanKernel<T, threads>,
                        stats.data(),
                        data.data(),
                        params.len);
    update_host<float>(h_stats, stats.data(), 2, stream);
    resource::sync_stream(handle, stream);
    h_stats[0] /= params.len;
    h_stats[1] = (h_stats[1] / params.len) - (h_stats[0] * h_stats[0]);
    resource::sync_stream(handle, stream);
  }

  void getExpectedMeanVar(float meanvar[2])
  {
    switch (params.type) {
      case RNG_Uniform:
        meanvar[0] = (params.start + params.end) * 0.5f;
        meanvar[1] = params.end - params.start;
        meanvar[1] = meanvar[1] * meanvar[1] / 12.f;
        break;
    };
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RngInputs<T> params;
  rmm::device_uvector<T> data;
  rmm::device_uvector<float> stats;
  float h_stats[2];  // mean, var
};

const std::vector<RngInputs<uint32_t>> inputs_u32 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL}};

using RngTestU32 = RngTest<uint32_t>;
TEST_P(RngTestU32, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestU32, ::testing::ValuesIn(inputs_u32));

using RngMdspanTestU32 = RngMdspanTest<uint32_t>;
TEST_P(RngMdspanTestU32, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngMdspanTests, RngMdspanTestU32, ::testing::ValuesIn(inputs_u32));

const std::vector<RngInputs<uint64_t>> inputs_u64 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL}};

using RngTestU64 = RngTest<uint64_t>;
TEST_P(RngTestU64, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestU64, ::testing::ValuesIn(inputs_u64));

using RngMdspanTestU64 = RngMdspanTest<uint64_t>;
TEST_P(RngMdspanTestU64, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngMdspanTests, RngMdspanTestU64, ::testing::ValuesIn(inputs_u64));

const std::vector<RngInputs<int32_t>> inputs_s32 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL}};

using RngTestS32 = RngTest<int32_t>;
TEST_P(RngTestS32, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestS32, ::testing::ValuesIn(inputs_s32));

using RngMdspanTestS32 = RngMdspanTest<int32_t>;
TEST_P(RngMdspanTestS32, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngMdspanTests, RngMdspanTestS32, ::testing::ValuesIn(inputs_s32));

const std::vector<RngInputs<int64_t>> inputs_s64 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL}};

using RngTestS64 = RngTest<int64_t>;
TEST_P(RngTestS64, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestS64, ::testing::ValuesIn(inputs_s64));

using RngMdspanTestS64 = RngMdspanTest<int64_t>;
TEST_P(RngMdspanTestS64, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngMdspanTests, RngMdspanTestS64, ::testing::ValuesIn(inputs_s64));

}  // namespace random
}  // namespace raft
