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

constexpr auto kInt64Min = std::numeric_limits<int64_t>::min();
constexpr auto kInt64Max = std::numeric_limits<int64_t>::max();

TEST(FastIntDivTest, UnsupportedDivisors)
{
  ASSERT_THROW(FastIntDiv{-42}, raft::exception);
  ASSERT_THROW(FastIntDiv{0}, raft::exception);
}

TEST(FastIntDivTest, Int32Division)
{
  std::vector<int32_t> numerators{kInt32Min,
                                  -(1 << 20),
                                  -12345,
                                  -255,
                                  -13,
                                  -7,
                                  -3,
                                  -2,
                                  0,
                                  1,
                                  2,
                                  3,
                                  7,
                                  13,
                                  255,
                                  12345,
                                  (1 << 20),
                                  kInt32Max};
  std::vector<int32_t> divisors{1, 2, 4, 7, 16, 31, 63, 128, 1000, (1 << 15), kInt32Max};

  for (auto d : divisors) {
    FastIntDiv fid(d);
    for (auto n : numerators) {
      ASSERT_EQ(n / fid, n / d) << "operator/ mismatch for numerator=" << n << " divisor=" << d;
      ASSERT_EQ(n % fid, n % d) << "operator% mismatch for numerator=" << n << " divisor=" << d;
    }
  }
}

TEST(FastIntDivTest, Int64Divisors)
{
  std::vector<int64_t> numerators{kInt64Min,
                                  int64_t(kInt32Min) - 12345678,
                                  int64_t(kInt32Min),
                                  -10007,
                                  -1013,
                                  0,
                                  1013,
                                  10007,
                                  int64_t(kInt32Max) + 1,
                                  int64_t(kInt32Max) + 2,
                                  (int64_t(1) << 32),
                                  (int64_t(1) << 33),
                                  (int64_t(3) << 40),
                                  kInt64Max};

  std::vector<int64_t> divisors{
    1, 31, 129, 772, 1000, (int64_t(1) << 31), int64_t(kInt32Max) + 5, int64_t(3) << 40, kInt64Max};

  for (auto d : divisors) {
    FastIntDiv fid(d);
    for (auto n : numerators) {
      ASSERT_EQ(n / fid, n / d) << "operator/ mismatch for numerator=" << n << " divisor=" << d;
      ASSERT_EQ(n % fid, n % d) << "operator% mismatch for numerator=" << n << " divisor=" << d;
    }
  }
}

TEST(FastIntDivTest, CrossIntTypesDivision)
{
  // i32 numerators, i64 divisors
  {
    std::vector<int32_t> numerators{kInt32Min, -123, 0, 321, kInt32Max};
    std::vector<int64_t> divisors{1, 4321, kInt64Max};

    for (auto d : divisors) {
      FastIntDiv fid(d);
      for (auto n : numerators) {
        ASSERT_EQ(n / fid, n / d) << "operator/ mismatch for numerator=" << n << " divisor=" << d;
        ASSERT_EQ(n % fid, n % d) << "operator% mismatch for numerator=" << n << " divisor=" << d;
      }
    }
  }

  // i64 numerators, i32 divisors
  {
    std::vector<int64_t> numerators{kInt64Min, -1234, 0, 4321, kInt64Max};
    std::vector<int32_t> divisors{1, 1234, kInt32Max};

    for (auto d : divisors) {
      FastIntDiv fid(d);
      for (auto n : numerators) {
        ASSERT_EQ(n / fid, n / d) << "operator/ mismatch for numerator=" << n << " divisor=" << d;
        ASSERT_EQ(n % fid, n % d) << "operator% mismatch for numerator=" << n << " divisor=" << d;
      }
    }
  }
}

}  // namespace raft::util
