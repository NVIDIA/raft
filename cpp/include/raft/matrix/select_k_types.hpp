/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/detail/macros.hpp>

#include <type_traits>

namespace raft {
namespace matrix {

/**
 * @defgroup select_k Batched-select k smallest or largest key/values
 * @{
 */

/**
 * @brief Algorithm used to select the k largest neighbors
 *
 * Details about how the the select-k algorithms in RAFT work can be found in the
 * paper "Parallel Top-K Algorithms on GPU: A Comprehensive Study and New Methods"
 * https://doi.org/10.1145/3581784.3607062. The kRadix* variants below correspond
 * to the 'Air Top-k' algorithm described in the paper, and the kWarp* variants
 * correspond to the 'GridSelect' algorithm.
 */
enum class SelectAlgo : uint8_t {
  /** Automatically pick the select-k algorithm based off the input dimensions and k value */
  kAuto = 0,
  /** Radix Select using 8 bits per pass */
  kRadix8bits = 1,
  /** Radix Select using 11 bits per pass, fusing the last filter step */
  kRadix11bits = 2,
  /** Radix Select using 11 bits per pass, without fusing the last filter step */
  kRadix11bitsExtraPass = 3,
  /**
   * Automatically switches between the kWarpImmediate and kWarpFiltered algorithms
   * based off of input size
   */
  kWarpAuto = 4,
  /**
   * This version of warp_sort adds every input element into the intermediate sorting
   * buffer, and thus does the sorting step every `Capacity` input elements.
   *
   * This implementation is preferred for very small len values.
   */
  kWarpImmediate = 5,
  /**
   * This version of warp_sort compares each input element against the current
   * estimate of k-th value before adding it to the intermediate sorting buffer.
   * This makes the algorithm do less sorting steps for long input sequences
   * at the cost of extra checks on each step.
   *
   * This implementation is preferred for large len values.
   */
  kWarpFiltered = 6,
  /**
   * This version of warp_sort compares each input element against the current
   * estimate of k-th value before adding it to the intermediate sorting buffer.
   * In contrast to `warp_sort_filtered`, it keeps one distributed buffer for
   * all threads in a warp (independently of the subwarp size), which makes its flushing less often.
   */
  kWarpDistributed = 7,
  /**
   * The same as `warp_sort_distributed`, but keeps the temporary value and index buffers
   * in the given external pointers (normally, a shared memory pointer should be passed in).
   */
  kWarpDistributedShm = 8,
  /**
   * The same as `kWarpDistributedShm`, but orders an equal-key tie by the payload index instead of
   * by lane arrival, so a repeated selection over the same input returns the same elements.
   */
  kWarpDistributedShmStable = 9,
};

inline auto operator<<(std::ostream& os, const SelectAlgo& algo) -> std::ostream&
{
  auto underlying_value = static_cast<std::underlying_type<SelectAlgo>::type>(algo);

  switch (algo) {
    case SelectAlgo::kWarpDistributedShmStable:
      return os << "kWarpDistributedShmStable=" << underlying_value;
    case SelectAlgo::kAuto: return os << "kAuto=" << underlying_value;
    case SelectAlgo::kRadix8bits: return os << "kRadix8bits=" << underlying_value;
    case SelectAlgo::kRadix11bits: return os << "kRadix11bits=" << underlying_value;
    case SelectAlgo::kRadix11bitsExtraPass:
      return os << "kRadix11bitsExtraPass=" << underlying_value;
    case SelectAlgo::kWarpAuto: return os << "kWarpAuto=" << underlying_value;
    case SelectAlgo::kWarpImmediate: return os << "kWarpImmediate=" << underlying_value;
    case SelectAlgo::kWarpFiltered: return os << "kWarpFiltered=" << underlying_value;
    case SelectAlgo::kWarpDistributed: return os << "kWarpDistributed=" << underlying_value;
    case SelectAlgo::kWarpDistributedShm: return os << "kWarpDistributedShm=" << underlying_value;
    default: throw std::invalid_argument("invalid value for SelectAlgo");
  }
}

/** @} */  // end of group select_k

}  // namespace matrix
}  // namespace raft
