/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace util {

struct MulInputs {
  uint64_t expected_high;
  uint64_t expected_low;
  uint64_t operand_1;
  uint64_t operand_2;
};

RAFT_KERNEL mul64_test_kernel(uint64_t* result_high,
                              uint64_t* result_low,
                              uint64_t* swapped_result_high,
                              uint64_t* swapped_result_low,
                              const uint64_t op1,
                              const uint64_t op2)
{
  using raft::wmul_64bit;
  wmul_64bit(*result_high, *result_low, op1, op2);
  wmul_64bit(*swapped_result_high, *swapped_result_low, op2, op1);
}

class Multiplication64bit : public testing::TestWithParam<MulInputs> {
 protected:
  Multiplication64bit()
    : stream(resource::get_cuda_stream(handle)),
      d_result_high(stream),
      d_result_low(stream),
      d_swapped_result_high(stream),
      d_swapped_result_low(stream)
  {
  }

 protected:
  void SetUp() override
  {
    using raft::wmul_64bit;
    params = testing::TestWithParam<MulInputs>::GetParam();
    wmul_64bit(result_high, result_low, params.operand_1, params.operand_2);
    wmul_64bit(swapped_result_high, swapped_result_low, params.operand_2, params.operand_1);

    raft::launch_kernel(stream,
                        1,
                        1,
                        mul64_test_kernel,
                        d_result_high.data(),
                        d_result_low.data(),
                        d_swapped_result_high.data(),
                        d_swapped_result_low.data(),
                        params.operand_1,
                        params.operand_2);
  }

  raft::resources handle;
  cudaStream_t stream;

  rmm::device_scalar<uint64_t> d_result_high;
  rmm::device_scalar<uint64_t> d_result_low;
  rmm::device_scalar<uint64_t> d_swapped_result_high;
  rmm::device_scalar<uint64_t> d_swapped_result_low;

  MulInputs params;

  uint64_t result_high;
  uint64_t result_low;
  uint64_t swapped_result_high;
  uint64_t swapped_result_low;
};

const std::vector<MulInputs> inputs = {
  {0ULL, 0ULL, 0ULL, 0ULL},
  {0ULL, 0ULL, UINT64_MAX, 0ULL},
  {0ULL, UINT64_MAX, UINT64_MAX, 1ULL},
  {UINT64_MAX - 1, 1ULL, UINT64_MAX, UINT64_MAX},
  {0x10759F98370FEC6EULL, 0xD5349806F735F69CULL, 0x1D6F160410C23D03ULL, 0x8F27C29767468634ULL},
  {0xAF72C5B915A5ABDEULL >> 1, 0xAF72C5B915A5ABDEULL << 63, 0xAF72C5B915A5ABDEULL, 1ULL << 63},
  {0xCA82AAEB81C01931ULL >> (64 - 23),
   0xCA82AAEB81C01931ULL << 23,
   0xCA82AAEB81C01931ULL,
   1ULL << 23}};

TEST_P(Multiplication64bit, Result)
{
  ASSERT_EQ(params.expected_high, d_result_high.value(stream));
  ASSERT_EQ(params.expected_low, d_result_low.value(stream));
  ASSERT_EQ(params.expected_high, d_swapped_result_high.value(stream));
  ASSERT_EQ(params.expected_low, d_swapped_result_low.value(stream));

  ASSERT_EQ(params.expected_high, result_high);
  ASSERT_EQ(params.expected_low, result_low);
  ASSERT_EQ(params.expected_high, swapped_result_high);
  ASSERT_EQ(params.expected_low, swapped_result_low);
}

INSTANTIATE_TEST_CASE_P(Mul64bit, Multiplication64bit, testing::ValuesIn(inputs));

};  // end of namespace util
};  // end of namespace raft
