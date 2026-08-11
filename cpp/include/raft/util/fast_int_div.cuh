/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>

#include <stdint.h>

#include <limits>
#include <type_traits>

namespace raft {
namespace util {

constexpr auto kInt32Min = std::numeric_limits<int32_t>::min();
constexpr auto kInt32Max = std::numeric_limits<int32_t>::max();

/**
 * @brief Perform fast integer division and modulo using a known divisor
 * From Hacker's Delight, Second Edition, Chapter 10
 *
 * **Usage**
 *
 * Construct the divisor once and call `/` and `%` operators repeatedly with
 * different numerators.

 * @code{.cpp}
 * raft::util::FastIntDiv<int32_t> div(stride);
 * int32_t quotient  = flat_index / div;
 * int32_t remainder = flat_index % div;
 * @endcode
 *
 * @note It will auto-fallback to the plain division when the divisor or the numerator
 * is beyond the 32-bit signed integer range.
 *
 * @todo Extend support for signed divisors
 */
template <typename IntT>
struct FastIntDiv {
  static_assert(std::is_same_v<IntT, int32_t> || std::is_same_v<IntT, int64_t>,
                "FastIntDiv: IntT must be int32_t or int64_t");
  using UIntT = std::make_unsigned_t<IntT>;

  /**
   * @defgroup HostMethods Ctor's that are accessible only from host
   * @{
   * @brief Host-only ctor's
   * @param _d the divisor
   */
  FastIntDiv(IntT _d) : d(_d) { computeScalars(); }
  FastIntDiv& operator=(IntT _d)
  {
    d = _d;
    computeScalars();
    return *this;
  }
  /** @} */

  /**
   * @defgroup DeviceMethods Ctor's which even the device-side can access
   * @{
   * @brief host and device ctor's
   * @param other source object to be copied from
   */
  HDI FastIntDiv(const FastIntDiv& other)
    : d(other.d), m(other.m), p(other.p), fallback(other.fallback)
  {
  }
  HDI FastIntDiv& operator=(const FastIntDiv& other)
  {
    d        = other.d;
    m        = other.m;
    p        = other.p;
    fallback = other.fallback;
    return *this;
  }
  /** @} */

  /** divisor */
  IntT d;
  /** the term 'm' as found in the reference chapter */
  UIntT m;
  /** the term 'p' as found in the reference chapter */
  int p;
  /** Flag for falling back to canonical division on unsupported divisor's ranges */
  bool fallback = false;

 private:
  void computeScalars()
  {
    if (d == 1) {
      m = 0;
      p = 1;
      return;
    } else if (d < 0) {
      ASSERT(false, "FastIntDiv: division by negative numbers not supported!");
    } else if (d == 0) {
      ASSERT(false, "FastIntDiv: got division by zero!");
    } else if (int64_t(d) > kInt32Max) {
      fallback = true;
      return;
    }
    int64_t nc = ((1LL << 31) / d) * d - 1;
    p          = 31;
    int64_t twoP, rhs;
    do {
      ++p;
      twoP = 1LL << p;
      rhs  = nc * (d - twoP % d);
    } while (twoP <= rhs);
    m = (twoP + d - twoP % d) / d;
  }
};  // struct FastIntDiv

/**
 * @brief Division overload, so that FastIntDiv can be transparently switched
 *
 * @note Not meant to be called directly, but via `n / div` where `div` is a
 *       `FastIntDiv` instance
 *
 * @param n numerator
 * @param divisor the precomputed divisor
 * @return the quotient
 */
template <typename NumIntT, typename DivIntT>
HDI std::common_type_t<NumIntT, DivIntT> operator/(NumIntT n, const FastIntDiv<DivIntT>& divisor)
{
  using CommonIntT = std::common_type_t<NumIntT, DivIntT>;
  if (divisor.d == 1) return CommonIntT(n);
  if (divisor.fallback || n < kInt32Min || n > kInt32Max) {
    return CommonIntT(n) / CommonIntT(divisor.d);
  }
  CommonIntT ret = (int64_t(divisor.m) * int64_t(n)) >> divisor.p;
  return ret + CommonIntT(n < 0);
}

/**
 * @brief Modulo overload enabling transparent use of `FastIntDiv` with `%`.
 *
 * @note Not meant to be called directly, but via `n % div` where `div` is a
 *       `FastIntDiv` instance
 *
 * @param n numerator
 * @param divisor the precomputed divisor
 * @return the remainder
 */
template <typename NumIntT, typename DivIntT>
HDI std::common_type_t<NumIntT, DivIntT> operator%(NumIntT n, const FastIntDiv<DivIntT>& divisor)
{
  using CommonIntT     = std::common_type_t<NumIntT, DivIntT>;
  CommonIntT quotient  = n / divisor;
  CommonIntT remainder = CommonIntT(n) - quotient * CommonIntT(divisor.d);
  return remainder;
  // return n % divisor.d;
}

};  // namespace util
}  // namespace raft
