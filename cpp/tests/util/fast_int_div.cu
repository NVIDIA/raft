/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/util/fast_int_div.cuh>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace raft::util {

constexpr auto kInt32Max = std::numeric_limits<int32_t>::max();
constexpr auto kInt64Max = std::numeric_limits<int64_t>::max();

TEST(FastIntDivTest, UnsupportedNegativeDenumerator)
{
  ASSERT_THROW(FastIntDiv{-42}, raft::exception);
  ASSERT_THROW(FastIntDiv{0}, raft::exception);
}

TEST(FastIntDivTest, CompareWithNativeDivisionInt32)
{
  std::vector<int32_t> magnitudes{0, 1, 2, 3, 7, 13, 255, 12345, (1 << 20), kInt32Max};
  std::vector<int32_t> divisors{1, 2, 4, 7, 16, 31, 63, 128, 1000, (1 << 15), kInt32Max};

  for (int32_t d : divisors) {
    FastIntDiv fid(d);
    for (int32_t mag : magnitudes) {
      for (int32_t n : {mag, -mag}) {
        ASSERT_EQ(n / fid, n / d) << "operator/ mismatch for numerator=" << n << " divisor=" << d;
        ASSERT_EQ(n % fid, n % d) << "operator% mismatch for numerator=" << n << " divisor=" << d;
      }
    }
  }
}

TEST(FastIntDivTest, CompareWithNativeDivisionInt64)
{
  std::vector<int64_t> magnitudes{
    0, 1013, 10007, int64_t(kInt32Max) + 3, (int64_t(1) << 33), (int64_t(3) << 40), kInt64Max};
  std::vector<int64_t> divisors{
    1, 129, 772, 1000, (int64_t(1) << 31), int64_t(kInt32Max) + 1, int64_t(3) << 40, kInt64Max};

  for (int64_t d : divisors) {
    FastIntDiv fid(d);
    for (int64_t mag : magnitudes) {
      for (int64_t n : {mag, -mag}) {
        ASSERT_EQ(n / fid, n / d) << "operator/ mismatch for numerator=" << n << " divisor=" << d;
        ASSERT_EQ(n % fid, n % d) << "operator% mismatch for numerator=" << n << " divisor=" << d;
      }
    }
  }
}

}  // namespace raft::util
