/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/util/fast_int_div.cuh>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

namespace raft::util {

constexpr int64_t kInt32Max = std::numeric_limits<int32_t>::max();

template <typename IntT>
class FastIntDivTest : public ::testing::Test {
 protected:
  void CompareWithNativeDivision()
  {
    std::vector<IntT> magnitudes{0, 1, 2, 3, 7, 13, 255, 12345, (1 << 20), kInt32Max};
    std::vector<IntT> divisors{1, 2, 4, 7, 16, 31, 63, 128, 1000, (1 << 15), kInt32Max};
    
    for (IntT d : divisors) {
      FastIntDiv fid(d);
      for (IntT mag : magnitudes) {
        for (IntT n : {mag, -mag}) {
          ASSERT_EQ(n / fid, n / d) << "operator/ mismatch for numerator=" << n << " divisor=" << d;
          ASSERT_EQ(n % fid, n % d) << "operator% mismatch for numerator=" << n << " divisor=" << d;
        }
      }
    }
  }
};

using FastIntDivTypes = ::testing::Types<int32_t, int64_t>;
TYPED_TEST_CASE(FastIntDivTest, FastIntDivTypes);

TYPED_TEST(FastIntDivTest, CompareWithNativeDivision) { this->CompareWithNativeDivision(); }

TEST(FastIntDiv, Int64NumeratorPastInt32Boundary)
{
  for (int64_t d : {129, 772, 1000}) {
    FastIntDiv fid(d);
    for (int64_t n : {1LL << 31, 2147704000LL, 3LL << 30}) {  // in [2^31, 2^32)
      ASSERT_EQ(n / fid, n / d) << "operator/ mismatch for numerator=" << n << " divisor=" << d;
      ASSERT_EQ(n % fid, n % d) << "operator% mismatch for numerator=" << n << " divisor=" << d;
    }
  }
}

}  // namespace raft::util
