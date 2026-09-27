/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace util {

namespace {

template <typename T>
_RAFT_DEVICE _RAFT_FORCEINLINE void swap(T& x, T& y)
{
  T t = x;
  x   = y;
  y   = t;
}

template <typename T>
_RAFT_DEVICE _RAFT_FORCEINLINE void conditional_assign(bool cond, T& ptr, T x)
{
  if (cond) { ptr = x; }
}

/** First element of a payload pack. k-selection always passes the index first. */
template <typename P0, typename... Rest>
_RAFT_DEVICE _RAFT_FORCEINLINE P0 first_payload(P0 head, Rest...)
{
  return head;
}

/** Whether `a` must move after `b`, breaking an equal-key tie on the index. */
template <typename KeyT, typename IdxT>
_RAFT_DEVICE _RAFT_FORCEINLINE bool swap_needed(bool ascending,
                                                KeyT a,
                                                KeyT b,
                                                IdxT ia,
                                                IdxT ib)
{
  if (a != b) { return ascending ? (a > b) : (a < b); }
  // Must flip with `ascending`, or paired lanes disagree on a tie and duplicate an element.
  return ascending ? (ia > ib) : (ia < ib);
}

}  // namespace

/**
 * Warp-wide bitonic merge and sort.
 * The data is strided among `warp_width` threads,
 * e.g. calling `bitonic<4>(ascending=true).sort(arr)` takes a unique 4-element array as input of
 * each thread in a warp and sorts them, such that for a fixed i, arr[i] are sorted within the
 * threads in a warp, and for any i < j, arr[j] in any thread is not smaller than arr[i] in any
 * other thread.
 * When `warp_width < WarpSize`, the data is sorted within all subwarps of the warp independently.
 *
 * As an example, assuming `Size = 4`, `warp_width = 16`, and `WarpSize = 32`, sorting a permutation
 * of numbers 0-63 in each subwarp yield the following result:
 * `
 *  arr_i \ laneId()
 *       0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15    16  17  18 ...
 *      subwarp_1                                                         subwarp_2
 *   0   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15     0   1   2 ...
 *   1  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31    16  17  18 ...
 *   2  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47    32  33  34 ...
 *   3  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63    48  49  50 ...
 * `
 *
 * Here is a small usage example of device code, which sorts the arrays of length 8 (= 4 * 2)
 * grouped in pairs of threads in ascending order:
 * @code{.cpp}
 *   // Fill an array of four ints in each thread of a warp.
 *   int i = laneId();
 *   int arr[4] = {i+1, i+5, i, i+7};
 *   // Sort the arrays in groups of two threads.
 *   bitonic<4>(ascending=true, warp_width=2).sort(arr);
 *   // As a result,
 *   //  for every even thread (`i == 2j`):    arr == {2j,   2j+1, 2j+5, 2j+7}
 *   //  for every odd  thread (`i == 2j+1`):  arr == {2j+1, 2j+2, 2j+6, 2j+8}
 * @endcode
 *
 * @tparam Size
 *   number of elements processed in each thread;
 *   i.e. the total data size is `Size * warp_width`.
 *   Must be power-of-two.
 *
 */
template <int Size = 1>
class bitonic {
  static_assert(isPo2(Size));

 public:
  /**
   * Initialize bitonic sort config.
   *
   * @param ascending
   *   the resulting order (true: ascending, false: descending).
   * @param warp_width
   *   the number of threads participating in the warp-level primitives;
   *   the total size of the sorted data is `Size * warp_width`.
   *   Must be power-of-two, not larger than the WarpSize.
   */
  _RAFT_DEVICE _RAFT_FORCEINLINE explicit bitonic(bool ascending, int warp_width = WarpSize)
    : ascending_(ascending), warp_width_(warp_width)
  {
  }

  bitonic(bitonic const&)                    = delete;
  bitonic(bitonic&&)                         = delete;
  auto operator=(bitonic const&) -> bitonic& = delete;
  auto operator=(bitonic&&) -> bitonic&      = delete;

  /**
   * You can think of this function in two ways:
   *
   *   1) Sort any bitonic sequence.
   *   2) Merge two halves of the input data assuming they're already sorted, and their order is
   *      opposite (i.e. either ascending+descending or descending+ascending).
   *
   * The input pointers are unique per-thread.
   * See the class description for the description of the data layout.
   *
   * @param keys
   *   is a device pointer to a contiguous array of keys, unique per thread; must be at least `Size`
   *   elements long.
   * @param payloads
   *   are zero or more associated arrays of the same size as keys, which are sorted together with
   *   the keys; must be at least `Size` elements long.
   */
  template <typename KeyT, typename... PayloadTs>
  _RAFT_DEVICE _RAFT_FORCEINLINE void merge(KeyT* __restrict__ keys,
                                            PayloadTs* __restrict__... payloads) const
  {
    return bitonic<Size>::merge_impl(ascending_, warp_width_, keys, payloads...);
  }

  /**
   * Sort the data.
   * The input pointers are unique per-thread.
   * See the class description for the description of the data layout.
   *
   * @param keys
   *   is a device pointer to a contiguous array of keys, unique per thread; must be at least `Size`
   *   elements long.
   * @param payloads
   *   are zero or more associated arrays of the same size as keys, which are sorted together with
   *   the keys; must be at least `Size` elements long.
   */
  template <typename KeyT, typename... PayloadTs>
  _RAFT_DEVICE _RAFT_FORCEINLINE void sort(KeyT* __restrict__ keys,
                                           PayloadTs* __restrict__... payloads) const
  {
    return bitonic<Size>::sort_impl(ascending_, warp_width_, keys, payloads...);
  }

  /**
   * @brief `merge` variant for the case of one element per thread
   *        (pass input by a reference instead of a pointer).
   *
   * @param key
   * @param payload
   */
  template <typename KeyT, typename... PayloadTs, int S = Size>
  _RAFT_DEVICE _RAFT_FORCEINLINE auto merge(KeyT& __restrict__ key,
                                            PayloadTs& __restrict__... payload) const
    -> std::enable_if_t<S == 1, void>  // SFINAE to enable this for Size == 1 only
  {
    static_assert(S == Size);
    return merge(&key, &payload...);
  }

  /**
   * @brief `sort` variant for the case of one element per thread
   *        (pass input by a reference instead of a pointer).
   *
   * @param key
   * @param payload
   */
  template <typename KeyT, typename... PayloadTs, int S = Size>
  _RAFT_DEVICE _RAFT_FORCEINLINE auto sort(KeyT& __restrict__ key,
                                           PayloadTs& __restrict__... payload) const
    -> std::enable_if_t<S == 1, void>  // SFINAE to enable this for Size == 1 only
  {
    static_assert(S == Size);
    return sort(&key, &payload...);
  }

 private:
  const int warp_width_;
  const bool ascending_;

  template <int AnotherSize>
  friend class bitonic;

  template <typename KeyT, typename... PayloadTs>
  static _RAFT_DEVICE _RAFT_FORCEINLINE void merge_impl(bool ascending,
                                                        int warp_width,
                                                        KeyT* __restrict__ keys,
                                                        PayloadTs* __restrict__... payloads)
  {
#pragma unroll
    for (int size = Size; size > 1; size >>= 1) {
      const int stride = size >> 1;
#pragma unroll
      for (int offset = 0; offset < Size; offset += size) {
#pragma unroll
        for (int i = offset + stride - 1; i >= offset; i--) {
          const int other_i = i + stride;
          KeyT& key         = keys[i];
          KeyT& other       = keys[other_i];
          bool do_swap;
          if constexpr (sizeof...(PayloadTs) > 0) {
            do_swap = swap_needed(ascending,
                                  key,
                                  other,
                                  first_payload(payloads[i]...),
                                  first_payload(payloads[other_i]...));
          } else {
            do_swap = ascending ? key > other : key < other;
          }
          if (do_swap) {
            swap(key, other);
            (swap(payloads[i], payloads[other_i]), ...);
          }
        }
      }
    }
    const int lane = laneId();
#pragma unroll
    for (int i = 0; i < Size; i++) {
      KeyT& key = keys[i];
      for (int stride = (warp_width >> 1); stride > 0; stride >>= 1) {
        const bool is_second = lane & stride;
        const KeyT other     = shfl_xor(key, stride, warp_width);
        const bool asc       = (ascending != is_second);
        bool do_assign;
        if constexpr (sizeof...(PayloadTs) > 0) {
          // The partner's index must cross the lane boundary to break the tie.
          const auto my_id    = first_payload(payloads[i]...);
          const auto other_id = shfl_xor(my_id, stride, warp_width);
          do_assign           = swap_needed(asc, key, other, my_id, other_id);
        } else {
          do_assign = asc ? key > other : key < other;
        }

        conditional_assign(do_assign, key, other);
        // NB: don't put shfl_xor in a conditional; it must be called by all threads in a warp.
        (conditional_assign(do_assign, payloads[i], shfl_xor(payloads[i], stride, warp_width)),
         ...);
      }
    }
  }

  template <typename KeyT, typename... PayloadTs>
  static _RAFT_DEVICE _RAFT_FORCEINLINE void sort_impl(bool ascending,
                                                       int warp_width,
                                                       KeyT* __restrict__ keys,
                                                       PayloadTs* __restrict__... payloads)
  {
    if constexpr (Size == 1) {
      const int lane = laneId();
      for (int width = 2; width < warp_width; width <<= 1) {
        bitonic<1>::merge_impl(lane & width, width, keys, payloads...);
      }
    } else {
      constexpr int kSize2 = Size / 2;
      bitonic<kSize2>::sort_impl(false, warp_width, keys, payloads...);
      bitonic<kSize2>::sort_impl(true, warp_width, keys + kSize2, (payloads + kSize2)...);
    }
    bitonic<Size>::merge_impl(ascending, warp_width, keys, payloads...);
  }
};

}  // namespace util
}  // namespace raft
