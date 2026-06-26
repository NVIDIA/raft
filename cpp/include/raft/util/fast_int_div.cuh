/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>

#include <stdint.h>

#include <type_traits>

namespace raft {
namespace util {

/**
 * @brief Perform fast integer division and modulo using a known divisor
 * From Hacker's Delight, Second Edition, Chapter 10
 *
 * @note 32b signed integer is supported.
 * @note 64b signed integers is supported for an input data up to 2^31
 * because gpu-non-native int128 is avoided for performance.
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
  HDI FastIntDiv(const FastIntDiv& other) : d(other.d), m(other.m), p(other.p) {}
  HDI FastIntDiv& operator=(const FastIntDiv& other)
  {
    d = other.d;
    m = other.m;
    p = other.p;
    return *this;
  }
  /** @} */

  /** divisor */
  IntT d;
  /** the term 'm' as found in the reference chapter */
  UIntT m;
  /** the term 'p' as found in the reference chapter */
  int p;

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
 *        to even on device
 * @param n numerator
 * @param divisor the denominator
 * @return the quotient
 */
template <typename IntT>
HDI IntT operator/(IntT n, const FastIntDiv<IntT>& divisor)
{
  if (divisor.d == 1) return n;
  IntT ret = (int64_t(divisor.m) * int64_t(n)) >> divisor.p;
  if (n < 0) ++ret;
  return ret;
}

/**
 * @brief Modulo overload, so that FastIntDiv can be transparently switched
 *        to even on device
 * @param n numerator
 * @param divisor the denominator
 * @return the remainder
 */
template <typename IntT>
HDI IntT operator%(IntT n, const FastIntDiv<IntT>& divisor)
{
  IntT quotient  = n / divisor;
  IntT remainder = n - quotient * divisor.d;
  return remainder;
}

};  // namespace util
}  // namespace raft
