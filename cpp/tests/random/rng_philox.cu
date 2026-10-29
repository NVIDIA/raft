/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <curand_kernel.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace raft::random {
namespace {

constexpr int seed_count        = 4;
constexpr int subsequence_count = 6;
constexpr int offset_count      = 13;
constexpr int words_per_case    = 128;

struct PhiloxCases {
  uint64_t seeds[seed_count];
  uint64_t subsequences[subsequence_count];
  uint64_t offsets[offset_count];
};

constexpr PhiloxCases philox_cases{
  {0, 1, 0x123456789abcdef0ULL, std::numeric_limits<uint64_t>::max()},
  {0,
   1,
   0xffffffffULL,
   0x100000000ULL,
   0xfedcba9876543210ULL,
   std::numeric_limits<uint64_t>::max()},
  {0,
   1,
   2,
   3,
   4,
   5,
   0xffffffffULL,
   0x100000000ULL,
   0x3fffffffcULL,
   0x3ffffffffULL,
   0x400000000ULL,
   0xfffffffffffffffcULL,
   std::numeric_limits<uint64_t>::max()}};

__device__ uint32_t count_mismatches(PhiloxGenerator& actual, curandStatePhilox4_32_10_t& expected)
{
  uint32_t mismatches = 0;
  for (int i = 0; i < words_per_case; ++i) {
    mismatches += actual.next_u32() != curand(&expected);
  }
  return mismatches;
}

RAFT_KERNEL compare_philox(PhiloxCases inputs, uint32_t* mismatches)
{
  auto const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= seed_count * subsequence_count * offset_count) { return; }
  auto const seed        = inputs.seeds[index / (subsequence_count * offset_count)];
  auto const subsequence = inputs.subsequences[(index / offset_count) % subsequence_count];
  auto const offset      = inputs.offsets[index % offset_count];
  PhiloxGenerator actual(seed, subsequence, offset);
  curandStatePhilox4_32_10_t expected;
  curand_init(seed, subsequence, offset, &expected);
  mismatches[index] = count_mismatches(actual, expected);
}

RAFT_KERNEL compare_philox_device_state(detail::DeviceState<PhiloxGenerator> state,
                                        PhiloxCases inputs,
                                        uint32_t* mismatches)
{
  auto const index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= subsequence_count) { return; }
  auto const subsequence = inputs.subsequences[index];
  PhiloxGenerator actual(state, subsequence);
  curandStatePhilox4_32_10_t expected;
  curand_init(state.seed, state.base_subsequence + subsequence, 0, &expected);
  mismatches[index] = count_mismatches(actual, expected);
}

}  // namespace

TEST(PhiloxGenerator, MatchesCurandAcrossCounterBoundaries)
{
  raft::resources handle;
  auto stream              = resource::get_cuda_stream(handle);
  constexpr int case_count = seed_count * subsequence_count * offset_count;
  rmm::device_uvector<uint32_t> mismatches(case_count, stream);
  // Include all four output lanes, carries from the low counter word, and maximum uint64 inputs.
  compare_philox<<<raft::ceildiv(case_count, 128), 128, 0, stream.get()>>>(philox_cases,
                                                                           mismatches.data());
  RAFT_CUDA_TRY(cudaGetLastError());
  std::vector<uint32_t> host_mismatches(case_count);
  RAFT_CUDA_TRY(cudaMemcpyAsync(host_mismatches.data(),
                                mismatches.data(),
                                sizeof(uint32_t) * case_count,
                                cudaMemcpyDeviceToHost,
                                stream.get()));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  for (int i = 0; i < case_count; ++i) {
    EXPECT_EQ(host_mismatches[i], 0u)
      << "seed=" << philox_cases.seeds[i / (subsequence_count * offset_count)]
      << " subsequence=" << philox_cases.subsequences[(i / offset_count) % subsequence_count]
      << " offset=" << philox_cases.offsets[i % offset_count];
  }
}

TEST(PhiloxGenerator, MatchesCurandWithBaseSubsequence)
{
  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);
  rmm::device_uvector<uint32_t> mismatches(subsequence_count, stream);
  std::vector<uint32_t> host_mismatches(subsequence_count);
  for (auto seed : philox_cases.seeds) {
    for (auto base : philox_cases.subsequences) {
      RngState state(seed, GeneratorType::GenPhilox);
      state.base_subsequence = base;
      // Exercise the constructor used by distribution kernels, including unsigned sequence wrap.
      compare_philox_device_state<<<1, 32, 0, stream.get()>>>(
        detail::DeviceState<PhiloxGenerator>(state), philox_cases, mismatches.data());
      RAFT_CUDA_TRY(cudaGetLastError());
      RAFT_CUDA_TRY(cudaMemcpyAsync(host_mismatches.data(),
                                    mismatches.data(),
                                    sizeof(uint32_t) * subsequence_count,
                                    cudaMemcpyDeviceToHost,
                                    stream.get()));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream.get()));
      for (int i = 0; i < subsequence_count; ++i) {
        EXPECT_EQ(host_mismatches[i], 0u)
          << "seed=" << seed << " base=" << base << " subsequence=" << philox_cases.subsequences[i];
      }
    }
  }
}

}  // namespace raft::random
