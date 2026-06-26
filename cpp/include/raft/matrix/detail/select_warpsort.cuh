/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/custom_resource.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/matrix/detail/select_k_layout.cuh>
#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cache.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/util_type.cuh>  // cub::Traits

#include <algorithm>
#include <functional>
#include <type_traits>

/*
  Three APIs of different scopes are provided:
    1. host function: select_k()
    2. block-wide API: class block_sort
    3. warp-wide API: several implementations of warp_sort_*


  1. select_k()
    (see the docstring)

  2. class block_sort
    It can be regarded as a fixed size priority queue for a thread block,
    although the API is not typical.
    one of the classes `warp_sort_*` can be used to instantiate block_sort.

    It uses dynamic shared memory as an intermediate buffer.
    So the required shared memory size should be calculated using
    calc_smem_size_for_block_wide() and passed as the 3rd kernel launch parameter.

    To add elements to the queue, use add(T val, IdxT idx) with unique values per-thread.
    Use WarpSortClass<...>::kDummy constant for the threads outside of input bounds.

    After adding is finished, function done() should be called. And finally, store() is used to get
    the top-k result.

    Example:
      RAFT_KERNEL kernel() {
        block_sort<warp_sort_immediate, ...> queue(...);

        for (IdxT i = threadIdx.x; i < len, i += blockDim.x) {
          queue.add(in[i], in_idx[i]);
        }

        queue.done();
        queue.store(out, out_idx);
     }

     int smem_size = calc_smem_size_for_block_wide<T>(...);
     kernel<<<grid_dim, block_dim, smem_size>>>();


  3. class warp_sort_*
    These two classes can be regarded as fixed size priority queue for a warp.
    Usage is similar to class block_sort. No shared memory is needed.

    The host function (select_k) uses a heuristic to choose between these two classes for
    sorting, warp_sort_immediate being chosen when the number of inputs per warp is somewhat small
    (see the usage of LaunchThreshold<warp_sort_immediate>::len_factor_for_choosing).

    Example:
      RAFT_KERNEL kernel() {
        warp_sort_immediate<...> queue(...);
        int warp_id = threadIdx.x / WarpSize;
        int lane_id = threadIdx.x % WarpSize;

        for (IdxT i = lane_id; i < len, i += WarpSize) {
          queue.add(in[i], idx[i]);
        }

        queue.done();
        // each warp outputs to a different offset
        queue.store(out + warp_id * k, out_idx + warp_id * k);
      }
 */

namespace raft {
namespace matrix::detail::select::warpsort {

static constexpr int kMaxCapacity = 256;

namespace {

/** Whether 'left` should indeed be on the left w.r.t. `right`. */
template <bool Ascending, typename T>
_RAFT_DEVICE _RAFT_FORCEINLINE auto is_ordered(T left, T right) -> bool
{
  if constexpr (Ascending) { return left < right; }
  if constexpr (!Ascending) { return left > right; }
}

}  // namespace

/**
 * A fixed-size warp-level priority queue.
 * By feeding the data through this queue, you get the `k <= Capacity`
 * smallest/greatest values in the data.
 *
 * @tparam Capacity
 *   maximum number of elements in the queue.
 * @tparam Ascending
 *   which comparison to use: `true` means `<`, collect the smallest elements,
 *   `false` means `>`, collect the greatest elements.
 * @tparam T
 *   the type of keys (what is being compared)
 * @tparam IdxT
 *   the type of payload (normally, indices of elements), i.e.
 *   the content sorted alongside the keys.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort {
  static_assert(is_a_power_of_two(Capacity));
  static_assert(std::is_default_constructible_v<IdxT>);

 public:
  /**
   *  The `empty` value for the chosen binary operation,
   *  i.e. `Ascending ? upper_bound<T>() : lower_bound<T>()`.
   */
  static constexpr T kDummy = Ascending ? upper_bound<T>() : lower_bound<T>();
  /** Width of the subwarp. */
  static constexpr int kWarpWidth = std::min<int>(Capacity, WarpSize);
  /** The number of elements to select. */
  const int k;

  /** Extra memory required per-block for keeping the state (shared or global). */
  constexpr static auto mem_required(uint32_t block_size) -> size_t { return 0; }

  /**
   * Construct the warp_sort empty queue.
   *
   * @param k
   *   number of elements to select.
   */
  _RAFT_DEVICE warp_sort(int k) : k(k)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_arr_[i] = kDummy;
      idx_arr_[i] = IdxT{};
    }
  }

  /**
   * Load k values from the pointers at the given position, and merge them in the storage.
   *
   * When it actually loads the values, it always performs some collective warp operations in the
   * end, thus enforcing warp sync. This means, it's safe to call `store` with the same arguments
   * after `load_sorted` without extra sync. Note, however, that this is not necessarily true for
   * the reverse order, because the access patterns of `store` and `load_sorted` are different.
   *
   * @param[in] in
   *    a device pointer to a contiguous array, unique per-subwarp
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[in] in_idx
   *    a device pointer to a contiguous array, unique per-subwarp
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[in] do_merge
   *    must be the same for all threads within a subwarp of size `kWarpWidth`.
   *    It serves as a conditional; when `false` the function does nothing.
   *    We need it to ensure threads within a full warp don't diverge calling `bitonic::merge()`.
   */
  _RAFT_DEVICE void load_sorted(const T* in, const IdxT* in_idx, bool do_merge = true)
  {
    if (do_merge) {
      int idx = Pow2<kWarpWidth>::mod(laneId()) ^ Pow2<kWarpWidth>::Mask;
#pragma unroll
      for (int i = kMaxArrLen - 1; i >= 0; --i, idx += kWarpWidth) {
        if (idx < k) {
          T t = in[idx];
          if (is_ordered<Ascending>(t, val_arr_[i])) {
            val_arr_[i] = t;
            idx_arr_[i] = in_idx[idx];
          }
        }
      }
    }
    if (kWarpWidth < WarpSize || do_merge) {
      util::bitonic<kMaxArrLen>(Ascending, kWarpWidth).merge(val_arr_, idx_arr_);
    }
  }

  /**
   *  Save the content by the pointer location.
   *
   * @param[out] out
   *   device pointer to a contiguous array, unique per-subwarp of size `kWarpWidth`
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[out] out_idx
   *   device pointer to a contiguous array, unique per-subwarp of size `kWarpWidth`
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param valF (optional) postprocess values (T -> OutT)
   * @param idxF (optional) postprocess indices (IdxT -> OutIdxT)
   */
  template <typename OutT,
            typename OutIdxT,
            typename ValF = identity_op,
            typename IdxF = identity_op>
  _RAFT_DEVICE void store(OutT* out,
                          OutIdxT* out_idx,
                          ValF valF = raft::identity_op{},
                          IdxF idxF = raft::identity_op{}) const
  {
    int idx = Pow2<kWarpWidth>::mod(laneId());
#pragma unroll kMaxArrLen
    for (int i = 0; i < kMaxArrLen && idx < k; i++, idx += kWarpWidth) {
      out[idx]     = valF(val_arr_[i]);
      out_idx[idx] = idxF(idx_arr_[i]);
    }
  }

 protected:
  static constexpr int kMaxArrLen = Capacity / kWarpWidth;

  T val_arr_[kMaxArrLen];
  IdxT idx_arr_[kMaxArrLen];

  /**
   * Merge another array (sorted in the opposite direction) in the queue.
   * Thanks to the other array being sorted in the opposite direction,
   * it's enough to call bitonic.merge once to maintain the valid state
   * of the queue.
   *
   * @tparam PerThreadSizeIn
   *   the size of the other array per-thread (compared to `kMaxArrLen`).
   *
   * @param keys_in
   *   the values to be merged in. Pointers are unique per-thread. The values
   *   must already be sorted in the opposite direction.
   *   The layout of `keys_in` must be the same as the layout of `val_arr_`.
   * @param ids_in
   *   the associated indices of the elements in the same format as `keys_in`.
   */
  template <int PerThreadSizeIn>
  _RAFT_DEVICE _RAFT_FORCEINLINE void merge_in(const T* __restrict__ keys_in,
                                               const IdxT* __restrict__ ids_in)
  {
#pragma unroll
    for (int i = std::min(kMaxArrLen, PerThreadSizeIn); i > 0; i--) {
      T& key  = val_arr_[kMaxArrLen - i];
      T other = keys_in[PerThreadSizeIn - i];
      if (is_ordered<Ascending>(other, key)) {
        key                      = other;
        idx_arr_[kMaxArrLen - i] = ids_in[PerThreadSizeIn - i];
      }
    }
    util::bitonic<kMaxArrLen>(Ascending, kWarpWidth).merge(val_arr_, idx_arr_);
  }
};

/**
 * This version of warp_sort compares each input element against the current
 * estimate of k-th value before adding it to the intermediate sorting buffer.
 * This makes the algorithm do less sorting steps for long input sequences
 * at the cost of extra checks on each step.
 *
 * This implementation is preferred for large len values.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_filtered : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;
  using warp_sort<Capacity, Ascending, T, IdxT>::mem_required;

  explicit _RAFT_DEVICE warp_sort_filtered(int k, T limit = kDummy)
    : warp_sort<Capacity, Ascending, T, IdxT>(k), buf_len_(0), k_th_(limit)
  {
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = kDummy;
      idx_buf_[i] = IdxT{};
    }
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE static auto init_blockwide(int k,
                                                            uint8_t* = nullptr,
                                                            T limit  = kDummy)
  {
    return warp_sort_filtered<Capacity, Ascending, T, IdxT>{k, limit};
  }

  _RAFT_DEVICE void add(T val, IdxT idx)
  {
    // comparing for k_th should reduce the total amount of updates:
    // `false` means the input value is surely not in the top-k values.
    bool do_add = is_ordered<Ascending>(val, k_th_);
    // merge the buf if it's full and we cannot add an element anymore.
    if (any(buf_len_ + do_add > kMaxBufLen)) {
      // still, add an element before merging if possible for this thread
      if (do_add && buf_len_ < kMaxBufLen) {
        add_to_buf_(val, idx);
        do_add = false;
      }
      merge_buf_();
    }
    // add an element if necessary and haven't already.
    if (do_add) { add_to_buf_(val, idx); }
  }

  _RAFT_DEVICE void done()
  {
    if (any(buf_len_ != 0)) { merge_buf_(); }
  }

 private:
  _RAFT_DEVICE _RAFT_FORCEINLINE void set_k_th_()
  {
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    k_th_ = shfl(val_arr_[kMaxArrLen - 1], k - 1, kWarpWidth);
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE void merge_buf_()
  {
    util::bitonic<kMaxBufLen>(!Ascending, kWarpWidth).sort(val_buf_, idx_buf_);
    this->template merge_in<kMaxBufLen>(val_buf_, idx_buf_);
    buf_len_ = 0;
    set_k_th_();  // contains warp sync
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = kDummy;
    }
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE void add_to_buf_(T val, IdxT idx)
  {
    // NB: the loop is used here to ensure the constant indexing,
    //     to not force the buffers spill into the local memory.
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      if (i == buf_len_) {
        val_buf_[i] = val;
        idx_buf_[i] = idx;
      }
    }
    buf_len_++;
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  static constexpr int kMaxBufLen = (Capacity <= 64) ? 2 : 4;

  T val_buf_[kMaxBufLen];
  IdxT idx_buf_[kMaxBufLen];
  int buf_len_;

  T k_th_;
};

/**
 * This version of warp_sort compares each input element against the current
 * estimate of k-th value before adding it to the intermediate sorting buffer.
 * In contrast to `warp_sort_filtered`, it keeps one distributed buffer for
 * all threads in a warp (independently of the subwarp size), which makes its flushing less often.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_distributed : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;
  using warp_sort<Capacity, Ascending, T, IdxT>::mem_required;

  explicit _RAFT_DEVICE warp_sort_distributed(int k, T limit = kDummy)
    : warp_sort<Capacity, Ascending, T, IdxT>(k),
      buf_val_(kDummy),
      buf_idx_(IdxT{}),
      buf_len_(0),
      k_th_(limit)
  {
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE static auto init_blockwide(int k,
                                                            uint8_t* = nullptr,
                                                            T limit  = kDummy)
  {
    return warp_sort_distributed<Capacity, Ascending, T, IdxT>{k, limit};
  }

  _RAFT_DEVICE void add(T val, IdxT idx)
  {
    // mask tells which lanes in the warp have valid items to be added
    uint32_t mask = ballot(is_ordered<Ascending>(val, k_th_));
    if (mask == 0) { return; }
    // how many elements to be added
    uint32_t n_valid = __popc(mask);
    // index of the source lane containing the value to put into the current lane.
    uint32_t src_ix = 0;
    // remove a few smallest set bits from the mask.
    for (uint32_t i = std::min(n_valid, Pow2<WarpSize>::mod(uint32_t(laneId()) - buf_len_)); i > 0;
         i--) {
      src_ix = __ffs(mask) - 1;
      mask ^= (0x1u << src_ix);
    }
    // now the least significant bit of the mask corresponds to the lane id we want to get.
    // for not-added (invalid) indices, the mask is zeroed by now.
    src_ix = __ffs(mask) - 1;
    // rearrange the inputs to be ready to put them into the tmp buffer
    val = shfl(val, src_ix);
    idx = shfl(idx, src_ix);
    // for non-valid lanes, src_ix should be uint(-1)
    if (mask == 0) { val = kDummy; }
    // save the values into the free slots of the warp tmp buffer
    if (laneId() >= buf_len_) {
      buf_val_ = val;
      buf_idx_ = idx;
    }
    buf_len_ += n_valid;
    if (buf_len_ < WarpSize) { return; }
    // merge the warp tmp buffer into the queue
    merge_buf_();
    buf_len_ -= WarpSize;
    // save the inputs that couldn't fit before the merge
    if (laneId() < buf_len_) {
      buf_val_ = val;
      buf_idx_ = idx;
    }
  }

  _RAFT_DEVICE void done()
  {
    if (buf_len_ != 0) {
      merge_buf_();
      buf_len_ = 0;
    }
  }

 private:
  _RAFT_DEVICE _RAFT_FORCEINLINE void set_k_th_()
  {
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    k_th_ = shfl(val_arr_[kMaxArrLen - 1], k - 1, kWarpWidth);
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE void merge_buf_()
  {
    util::bitonic<1>(!Ascending, kWarpWidth).sort(buf_val_, buf_idx_);
    this->template merge_in<1>(&buf_val_, &buf_idx_);
    set_k_th_();  // contains warp sync
    buf_val_ = kDummy;
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  T buf_val_;
  IdxT buf_idx_;
  uint32_t buf_len_;  // 0 <= buf_len_ <= WarpSize

  T k_th_;
};

/**
 * The same as `warp_sort_distributed`, but keeps the temporary value and index buffers
 * in the given external pointers (normally, a shared memory pointer should be passed in).
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_distributed_ext : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;

  constexpr static auto mem_required(uint32_t block_size) -> size_t
  {
    return (sizeof(T) + sizeof(IdxT)) * block_size;
  }

  _RAFT_DEVICE warp_sort_distributed_ext(int k, T* val_buf, IdxT* idx_buf, T limit = kDummy)
    : warp_sort<Capacity, Ascending, T, IdxT>(k),
      val_buf_(val_buf),
      idx_buf_(idx_buf),
      buf_len_(0),
      k_th_(limit)
  {
    val_buf_[laneId()] = kDummy;
  }

  _RAFT_DEVICE static auto init_blockwide(int k, uint8_t* shmem, T limit = kDummy)
  {
    T* val_buf    = nullptr;
    IdxT* idx_buf = nullptr;
    if constexpr (alignof(T) >= alignof(IdxT)) {
      val_buf = reinterpret_cast<T*>(shmem);
      idx_buf = reinterpret_cast<IdxT*>(val_buf + blockDim.x);
    } else {
      idx_buf = reinterpret_cast<IdxT*>(shmem);
      val_buf = reinterpret_cast<T*>(idx_buf + blockDim.x);
    }
    auto warp_offset = Pow2<WarpSize>::roundDown(threadIdx.x);
    val_buf += warp_offset;
    idx_buf += warp_offset;
    return warp_sort_distributed_ext<Capacity, Ascending, T, IdxT>{k, val_buf, idx_buf, limit};
  }

  _RAFT_DEVICE void add(T val, IdxT idx)
  {
    bool do_add = is_ordered<Ascending>(val, k_th_);
    // mask tells which lanes in the warp have valid items to be added
    uint32_t mask = ballot(do_add);
    if (mask == 0) { return; }
    // where to put the element in the tmp buffer
    int dst_ix = buf_len_ + __popc(mask & ((1u << laneId()) - 1u));
    // put all elements, which fit into the current tmp buffer
    if (do_add && dst_ix < WarpSize) {
      val_buf_[dst_ix] = val;
      idx_buf_[dst_ix] = idx;
      do_add           = false;
    }
    // Total number of elements to be added
    buf_len_ += __popc(mask);
    // If the buffer is still not full, we can return
    if (buf_len_ < WarpSize) { return; }
    // Otherwise, merge the warp tmp buffer into the queue
    merge_buf_();  // implies warp sync
    buf_len_ -= WarpSize;
    // save the inputs that couldn't fit before the merge
    if (do_add) {
      dst_ix -= WarpSize;
      val_buf_[dst_ix] = val;
      idx_buf_[dst_ix] = idx;
    }
  }

  _RAFT_DEVICE void done()
  {
    if (buf_len_ != 0) {
      merge_buf_();
      buf_len_ = 0;
    }
    __syncthreads();
  }

 private:
  _RAFT_DEVICE _RAFT_FORCEINLINE void set_k_th_()
  {
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    k_th_ = shfl(val_arr_[kMaxArrLen - 1], k - 1, kWarpWidth);
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE void merge_buf_()
  {
    __syncwarp();  // make sure the threads are aware of the data written by others
    T buf_val          = val_buf_[laneId()];
    IdxT buf_idx       = idx_buf_[laneId()];
    val_buf_[laneId()] = kDummy;
    util::bitonic<1>(!Ascending, kWarpWidth).sort(buf_val, buf_idx);
    this->template merge_in<1>(&buf_val, &buf_idx);
    set_k_th_();  // contains warp sync
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  T* val_buf_;
  IdxT* idx_buf_;
  uint32_t buf_len_;  // 0 <= buf_len_ < WarpSize

  T k_th_;
};

/**
 * This version of warp_sort adds every input element into the intermediate sorting
 * buffer, and thus does the sorting step every `Capacity` input elements.
 *
 * This implementation is preferred for very small len values.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_immediate : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;
  using warp_sort<Capacity, Ascending, T, IdxT>::mem_required;

  explicit _RAFT_DEVICE warp_sort_immediate(int k)
    : warp_sort<Capacity, Ascending, T, IdxT>(k), buf_len_(0)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_buf_[i] = kDummy;
      idx_buf_[i] = IdxT{};
    }
  }

  _RAFT_DEVICE _RAFT_FORCEINLINE static auto init_blockwide(int k, uint8_t* = nullptr)
  {
    return warp_sort_immediate<Capacity, Ascending, T, IdxT>{k};
  }

  _RAFT_DEVICE void add(T val, IdxT idx)
  {
    // NB: the loop is used here to ensure the constant indexing,
    //     to not force the buffers spill into the local memory.
#pragma unroll
    for (int i = 0; i < kMaxArrLen; ++i) {
      if (i == buf_len_) {
        val_buf_[i] = val;
        idx_buf_[i] = idx;
      }
    }

    ++buf_len_;
    if (buf_len_ == kMaxArrLen) {
      util::bitonic<kMaxArrLen>(!Ascending, kWarpWidth).sort(val_buf_, idx_buf_);
      this->template merge_in<kMaxArrLen>(val_buf_, idx_buf_);
#pragma unroll
      for (int i = 0; i < kMaxArrLen; i++) {
        val_buf_[i] = kDummy;
      }
      buf_len_ = 0;
    }
  }

  _RAFT_DEVICE void done()
  {
    if (buf_len_ != 0) {
      util::bitonic<kMaxArrLen>(!Ascending, kWarpWidth).sort(val_buf_, idx_buf_);
      this->template merge_in<kMaxArrLen>(val_buf_, idx_buf_);
    }
  }

 private:
  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  T val_buf_[kMaxArrLen];
  IdxT idx_buf_[kMaxArrLen];
  int buf_len_;
};

template <typename T, typename IdxT>
auto calc_smem_size_for_block_wide(int num_of_warp, int k) -> int
{
  return Pow2<256>::roundUp(ceildiv(num_of_warp, 2) * sizeof(T) * k) +
         ceildiv(num_of_warp, 2) * sizeof(IdxT) * k;
}

template <template <int, bool, typename, typename> class WarpSortWarpWide,
          int Capacity,
          bool Ascending,
          typename T,
          typename IdxT>
class block_sort {
 public:
  using queue_t = WarpSortWarpWide<Capacity, Ascending, T, IdxT>;

  template <typename... Args>
  _RAFT_DEVICE block_sort(int k, Args... args) : queue_(queue_t::init_blockwide(k, args...))
  {
  }

  _RAFT_DEVICE void add(T val, IdxT idx) { queue_.add(val, idx); }

  /**
   * At the point of calling this function, the warp-level queues consumed all input
   * independently. The remaining work to be done is to merge them together.
   *
   * Here we tree-merge the results using the shared memory and block sync.
   */
  _RAFT_DEVICE void done(uint8_t* smem_buf)
  {
    queue_.done();

    int nwarps    = subwarp_align::div(blockDim.x);
    auto val_smem = reinterpret_cast<T*>(smem_buf);
    auto idx_smem = reinterpret_cast<IdxT*>(
      smem_buf + Pow2<256>::roundUp(ceildiv(nwarps, 2) * sizeof(T) * queue_.k));

    const int warp_id = subwarp_align::div(threadIdx.x);
    // NB: there is no need for the second __synchthreads between .load_sorted and .store:
    //     we shift the pointers every iteration, such that individual warps either access the same
    //     locations or do not overlap with any of the other warps. The access patterns within warps
    //     are different for the two functions, but .load_sorted implies warp sync at the end, so
    //     there is no need for __syncwarp either.
    for (int shift_mask = ~0, split = (nwarps + 1) >> 1; nwarps > 1;
         nwarps = split, split = (nwarps + 1) >> 1) {
      if (warp_id < nwarps && warp_id >= split) {
        int dst_warp_shift = (warp_id - (split & shift_mask)) * queue_.k;
        queue_.store(val_smem + dst_warp_shift, idx_smem + dst_warp_shift);
      }
      __syncthreads();

      shift_mask = ~shift_mask;  // invert the mask
      {
        int src_warp_shift = (warp_id + (split & shift_mask)) * queue_.k;
        // The last argument serves as a condition for loading
        //  -- to make sure threads within a full warp do not diverge on `bitonic::merge()`
        queue_.load_sorted(
          val_smem + src_warp_shift, idx_smem + src_warp_shift, warp_id < nwarps - split);
      }
    }
  }

  /** Save the content by the pointer location. */
  template <typename OutT,
            typename OutIdxT,
            typename ValF = identity_op,
            typename IdxF = identity_op>
  _RAFT_DEVICE void store(OutT* out,
                          OutIdxT* out_idx,
                          ValF valF = raft::identity_op{},
                          IdxF idxF = raft::identity_op{}) const
  {
    if (threadIdx.x < subwarp_align::Value) { queue_.store(out, out_idx, valF, idxF); }
  }

 private:
  using subwarp_align = Pow2<queue_t::kWarpWidth>;
  queue_t queue_;
};

/**
 * Uses the `WarpSortClass` to sort chunks of data within one block with no interblock
 * communication. It can be arranged so, that multiple blocks process one row of input; in this
 * case, they output multiple results of length k each. Then, a second pass is needed to merge
 * those into one final output.
 */
template <template <int, bool, typename, typename> class WarpSortClass,
          int Capacity,
          bool Ascending,
          typename T,
          typename IdxT,
          typename RowLayout>
__launch_bounds__(256) RAFT_KERNEL block_kernel(const T* in,
                                                const IdxT* in_idx,
                                                const IdxT* in_indptr,
                                                size_t offset,
                                                IdxT len,
                                                int k,
                                                T* out,
                                                IdxT* out_idx)
{
  // * per-block output
  {
    const int block_id = blockIdx.x + gridDim.x * blockIdx.y;
    out += block_id * k;
    out_idx += block_id * k;
  }
  // * per-block input
  {
    const size_t batch_id        = blockIdx.y;
    const auto [l_len, l_offset] = RowLayout::compute(len, offset, batch_id, in_indptr);
    in += l_offset;
    if (in_idx != nullptr) { in_idx += l_offset; }
    len = l_len;
  }
  // * trim k if the input is too short
  k = std::min<int>(k, len);

  // * set up the queue
  extern __shared__ __align__(256) uint8_t smem_buf_bytes[];
  // Map the value type T to an appropriate uint type using cub traits.
  // This gives us two advantages:
  //   1. More stable behavior for IEEE 754 floating-point values w.r.t. infinity and NaN
  //      (the behavior matches the select_radix implementation)
  //   2. Fewer expensive template instantiations
  using bits_t       = typename cub::Traits<T>::UnsignedBits;
  using bq_t         = block_sort<WarpSortClass, Capacity, Ascending, bits_t, IdxT>;
  uint8_t* warp_smem = bq_t::queue_t::mem_required(blockDim.x) > 0 ? smem_buf_bytes : nullptr;
  bq_t queue(k, warp_smem);

  // * main loop
  const IdxT stride         = gridDim.x * blockDim.x;
  const IdxT per_thread_lim = len + laneId();
  for (IdxT i = threadIdx.x + blockIdx.x * blockDim.x; i < per_thread_lim; i += stride) {
    // Twiddle the input value to ensure proper comparison of floating-point and signed int values
    queue.add(i < len ? cub::Traits<T>::TwiddleIn(__ldcs(reinterpret_cast<const bits_t*>(in) + i))
                      : WarpSortClass<Capacity, Ascending, bits_t, IdxT>::kDummy,
              (i < len && in_idx != nullptr) ? __ldcs(in_idx + i) : i);
  }

  // * write out the result
  queue.done(smem_buf_bytes);
  queue.store(reinterpret_cast<bits_t*>(out),
              out_idx,
              cub::Traits<T>::TwiddleOut /* Restore the FP / signed int representation.*/);
}

struct launch_params {
  int block_size    = 0;
  int min_grid_size = 0;
};

template <template <int, bool, typename, typename> class WarpSortClass,
          typename T,
          typename IdxT,
          typename RowLayout,
          int Capacity = kMaxCapacity>
struct launch_setup {
  /**
   * @brief Calculate the best block size and minimum grid size for the given `k`.
   *
   * @param[in] k
   *   The select-top-k parameter
   * @param[out] block_size
   *   Returned block size
   * @param[out] min_grid_size
   *   Returned minimum grid size needed to achieve the best potential occupancy
   * @param[in] block_size_limit
   *   Forcefully limit the block size (optional)
   */
  static auto calc_optimal_params(int k, int block_size_limit) -> launch_params
  {
    const int capacity = bound_by_power_of_two(k);
    if constexpr (Capacity > 1) {
      if (capacity < Capacity) {
        return launch_setup<WarpSortClass, T, IdxT, RowLayout, Capacity / 2>::calc_optimal_params(
          capacity, block_size_limit);
      }
    }
    ASSERT(capacity <= Capacity, "Requested k is too big (%d)", k);
    auto calc_smem = [k](int block_size) {
      int num_of_warp = block_size / std::min<int>(WarpSize, Capacity);
      return calc_smem_size_for_block_wide<T, IdxT>(num_of_warp, k);
    };
    launch_params ps;
    RAFT_CUDA_TRY(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      &ps.min_grid_size,
      &ps.block_size,
      block_kernel<WarpSortClass, Capacity, true, T, IdxT, RowLayout>,
      calc_smem,
      block_size_limit));
    return ps;
  }

  static void kernel(int k,
                     bool select_min,
                     size_t batch_size,
                     size_t len,
                     int num_blocks,
                     int block_dim,
                     int smem_size,
                     const T* in_key,
                     const IdxT* in_idx,
                     const IdxT* in_indptr,
                     T* out_key,
                     IdxT* out_idx,
                     rmm::cuda_stream_view stream)
  {
    const int capacity = bound_by_power_of_two(k);
    if constexpr (Capacity > 1) {
      if (capacity < Capacity) {
        return launch_setup<WarpSortClass, T, IdxT, RowLayout, Capacity / 2>::kernel(k,
                                                                                     select_min,
                                                                                     batch_size,
                                                                                     len,
                                                                                     num_blocks,
                                                                                     block_dim,
                                                                                     smem_size,
                                                                                     in_key,
                                                                                     in_idx,
                                                                                     in_indptr,
                                                                                     out_key,
                                                                                     out_idx,
                                                                                     stream);
      }
    }
    ASSERT(capacity <= Capacity, "Requested k is too big (%d)", k);

    // This is less than cuda's max block dim along Y axis (65535), but it's a
    // power-of-two, which ensures the alignment of batches in memory.
    constexpr size_t kMaxGridDimY = 32768;
    size_t g_offset               = 0;
    for (size_t offset = 0; offset < batch_size; offset += kMaxGridDimY) {
      size_t batch_chunk = std::min<size_t>(kMaxGridDimY, batch_size - offset);
      dim3 gs(num_blocks, batch_chunk, 1);
      if (select_min) {
        block_kernel<WarpSortClass, Capacity, true, T, IdxT, RowLayout>
          <<<gs, block_dim, smem_size, stream>>>(
            in_key, in_idx, in_indptr, g_offset, IdxT(len), k, out_key, out_idx);
      } else {
        block_kernel<WarpSortClass, Capacity, false, T, IdxT, RowLayout>
          <<<gs, block_dim, smem_size, stream>>>(
            in_key, in_idx, in_indptr, g_offset, IdxT(len), k, out_key, out_idx);
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      out_key += batch_chunk * num_blocks * k;
      out_idx += batch_chunk * num_blocks * k;

      if constexpr (!RowLayout::is_uniform) { in_indptr += batch_chunk; }
      g_offset += batch_chunk;
    }
  }
};

template <template <int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
struct warpsort_params_cache {
  static constexpr size_t kDefaultSize = 100;
  cache::lru<uint64_t, std::hash<uint64_t>, std::equal_to<>, launch_params> value{kDefaultSize};
};

template <template <int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
static auto calc_optimal_params(raft::resources const& res, int k, int block_size_limit = 0)
  -> launch_params
{
  uint64_t key = (static_cast<uint64_t>(k) << 32) | static_cast<uint64_t>(block_size_limit);
  auto& cache =
    resource::get_custom_resource<warpsort_params_cache<WarpSortClass, T, IdxT>>(res)->value;
  launch_params val;
  if (!cache.get(key, &val)) {
    val = launch_setup<WarpSortClass, T, IdxT, select::dense_layout<IdxT>>::calc_optimal_params(
      k, block_size_limit);
    cache.set(key, val);
  }
  return val;
}

template <template <int, bool, typename, typename> class WarpSortClass>
struct LaunchThreshold {};

template <>
struct LaunchThreshold<warp_sort_filtered> {
  static constexpr int len_factor_for_multi_block  = 2;
  static constexpr int len_factor_for_single_block = 32;
};

template <>
struct LaunchThreshold<warp_sort_distributed> {
  static constexpr int len_factor_for_multi_block  = 2;
  static constexpr int len_factor_for_single_block = 32;
};

template <>
struct LaunchThreshold<warp_sort_distributed_ext> {
  static constexpr int len_factor_for_multi_block  = 2;
  static constexpr int len_factor_for_single_block = 32;
};

template <>
struct LaunchThreshold<warp_sort_immediate> {
  static constexpr int len_factor_for_choosing     = 4;
  static constexpr int len_factor_for_multi_block  = 2;
  static constexpr int len_factor_for_single_block = 4;
};

template <template <int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
void calc_launch_parameter(raft::resources const& res,
                           size_t batch_size,
                           size_t len,
                           int k,
                           int* p_num_of_block,
                           int* p_num_of_warp)
{
  const int capacity               = bound_by_power_of_two(k);
  const int capacity_per_full_warp = std::max(capacity, WarpSize);
  auto lps                         = calc_optimal_params<WarpSortClass, T, IdxT>(res, k);
  int block_size                   = lps.block_size;
  int min_grid_size                = lps.min_grid_size;
  block_size                       = Pow2<WarpSize>::roundDown(block_size);

  int num_of_warp;
  int num_of_block;
  if (batch_size < size_t(min_grid_size)) {  // may use multiple blocks
    num_of_warp       = block_size / WarpSize;
    num_of_block      = min_grid_size / int(batch_size);
    int len_per_block = int(ceildiv<size_t>(len, num_of_block));
    int len_per_warp  = ceildiv(len_per_block, num_of_warp);

    len_per_warp  = Pow2<WarpSize>::roundUp(len_per_warp);
    len_per_block = len_per_warp * num_of_warp;
    num_of_block  = int(ceildiv<size_t>(len, len_per_block));

    constexpr int kLenFactor = LaunchThreshold<WarpSortClass>::len_factor_for_multi_block;
    if (len_per_warp < capacity_per_full_warp * kLenFactor) {
      len_per_warp  = capacity_per_full_warp * kLenFactor;
      len_per_block = num_of_warp * len_per_warp;
      if (size_t(len_per_block) > len) { len_per_block = len; }
      num_of_block = int(ceildiv<size_t>(len, len_per_block));
      num_of_warp  = ceildiv(len_per_block, len_per_warp);
    }
  } else {  // use only single block
    num_of_block = 1;

    auto adjust_block_size = [len, capacity_per_full_warp](int bs) {
      int warps_per_block = bs / WarpSize;
      int len_per_warp    = int(ceildiv<size_t>(len, warps_per_block));
      len_per_warp        = Pow2<WarpSize>::roundUp(len_per_warp);
      warps_per_block     = int(ceildiv<size_t>(len, len_per_warp));

      constexpr int kLenFactor = LaunchThreshold<WarpSortClass>::len_factor_for_single_block;
      if (len_per_warp < capacity_per_full_warp * kLenFactor) {
        len_per_warp    = capacity_per_full_warp * kLenFactor;
        warps_per_block = int(ceildiv<size_t>(len, len_per_warp));
      }

      return warps_per_block * WarpSize;
    };

    // gradually reduce the block size while the batch size allows and the len is not big enough
    // to occupy a single block well.
    block_size = adjust_block_size(block_size);
    do {
      num_of_warp        = block_size / WarpSize;
      auto another       = calc_optimal_params<WarpSortClass, T, IdxT>(res, k, block_size);
      another.block_size = adjust_block_size(another.block_size);
      if (batch_size >= size_t(another.min_grid_size)  // still have enough work
          && another.block_size < block_size           // protect against an infinite loop
          && another.min_grid_size * another.block_size >
               min_grid_size * block_size  // improve occupancy
      ) {
        block_size    = another.block_size;
        min_grid_size = another.min_grid_size;
      } else {
        break;
      }
    } while (block_size > WarpSize);
    num_of_warp = std::max(1, num_of_warp);
  }

  *p_num_of_block = num_of_block;
  *p_num_of_warp  = num_of_warp * capacity_per_full_warp / capacity;
}

template <template <int, bool, typename, typename> class WarpSortClass,
          typename T,
          typename IdxT,
          typename RowLayout>
void select_k_(bool dry_run,
               int num_of_block,
               int num_of_warp,
               const T* in,
               const IdxT* in_idx,
               const IdxT* in_indptr,
               size_t batch_size,
               size_t len,
               int k,
               T* out,
               IdxT* out_idx,
               bool select_min,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> tmp_val(num_of_block * k * batch_size, stream, mr);
  rmm::device_uvector<IdxT> tmp_idx(num_of_block * k * batch_size, stream, mr);
  if (dry_run) { return; }

  int capacity   = bound_by_power_of_two(k);
  int warp_width = std::min(capacity, WarpSize);

  T* result_val    = (num_of_block == 1) ? out : tmp_val.data();
  IdxT* result_idx = (num_of_block == 1) ? out_idx : tmp_idx.data();
  int block_dim    = num_of_warp * warp_width;
  int smem_size    = calc_smem_size_for_block_wide<T, IdxT>(num_of_warp, k);

  smem_size = std::max<int>(smem_size, WarpSortClass<1, true, T, IdxT>::mem_required(block_dim));

  launch_setup<WarpSortClass, T, IdxT, RowLayout>::kernel(k,
                                                          select_min,
                                                          batch_size,
                                                          len,
                                                          num_of_block,
                                                          block_dim,
                                                          smem_size,
                                                          in,
                                                          in_idx,
                                                          in_indptr,
                                                          result_val,
                                                          result_idx,
                                                          stream);

  if (num_of_block > 1) {
    // a second pass to merge the results if necessary
    launch_setup<WarpSortClass, T, IdxT, select::dense_layout<IdxT>>::kernel(k,
                                                                             select_min,
                                                                             batch_size,
                                                                             k * num_of_block,
                                                                             1,
                                                                             block_dim,
                                                                             smem_size,
                                                                             tmp_val.data(),
                                                                             tmp_idx.data(),
                                                                             nullptr,
                                                                             out,
                                                                             out_idx,
                                                                             stream);
  }
}

template <typename T,
          typename IdxT,
          template <int, bool, typename, typename> class WarpSortClass,
          typename RowLayout>
void select_k_impl(raft::resources const& res,
                   const T* in,
                   const IdxT* in_idx,
                   size_t batch_size,
                   size_t len,
                   int k,
                   T* out,
                   IdxT* out_idx,
                   bool select_min,
                   const IdxT* in_indptr = nullptr)
{
  int num_of_block = 0;
  int num_of_warp  = 0;
  calc_launch_parameter<WarpSortClass, T, IdxT>(
    res, batch_size, len, k, &num_of_block, &num_of_warp);

  select_k_<WarpSortClass, T, IdxT, RowLayout>(resource::get_dry_run_flag(res),
                                               num_of_block,
                                               num_of_warp,
                                               in,
                                               in_idx,
                                               in_indptr,
                                               batch_size,
                                               len,
                                               k,
                                               out,
                                               out_idx,
                                               select_min,
                                               resource::get_cuda_stream(res),
                                               resource::get_workspace_resource_ref(res));
}

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_keys` as a row-major matrix with len columns and
 * batch_size rows, then this function selects k smallest/largest values in each row and fills
 * in the row-major matrix `out` of size (batch_size, k).
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 *
 * @param[in] res container of reusable resources
 * @param[in] in
 *   contiguous device array of inputs of size (len * batch_size);
 *   these are compared and selected.
 * @param[in] in_idx
 *   contiguous device array of inputs of size (len * batch_size);
 *   typically, these are indices of the corresponding in_keys.
 * @param batch_size
 *   number of input rows, i.e. the batch size.
 * @param len
 *   length of a single input array (row); also sometimes referred as n_cols.
 *   Invariant: len >= k.
 * @param k
 *   the number of outputs to select in each input row.
 * @param[out] out
 *   contiguous device array of outputs of size (k * batch_size);
 *   the k smallest/largest values from each row of the `in_keys`.
 * @param[out] out_idx
 *   contiguous device array of outputs of size (k * batch_size);
 *   the payload selected together with `out`.
 * @param select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param[in] in_indptr
 *   CSR indptr array (used only when RowLayout is `csr_layout`); nullptr for dense.
 */
template <typename T, typename IdxT, typename RowLayout>
void select_k(raft::resources const& res,
              const T* in,
              const IdxT* in_idx,
              size_t batch_size,
              size_t len,
              int k,
              T* out,
              IdxT* out_idx,
              bool select_min,
              const IdxT* in_indptr = nullptr)
{
  if (resource::get_dry_run_flag(res)) { return; }
  ASSERT(k <= kMaxCapacity, "Current max k is %d (requested %d)", kMaxCapacity, k);
  ASSERT(len <= size_t(std::numeric_limits<IdxT>::max()),
         "The `len` (%zu) does not fit the indexing type",
         len);

  int capacity     = bound_by_power_of_two(k);
  int num_of_block = 0;
  int num_of_warp  = 0;
  calc_launch_parameter<warp_sort_immediate, T, IdxT>(
    res, batch_size, len, k, &num_of_block, &num_of_warp);
  int len_per_thread = len / (num_of_block * num_of_warp * std::min(capacity, WarpSize));

  if (len_per_thread <= LaunchThreshold<warp_sort_immediate>::len_factor_for_choosing) {
    select_k_<warp_sort_immediate, T, IdxT, RowLayout>(resource::get_dry_run_flag(res),
                                                       num_of_block,
                                                       num_of_warp,
                                                       in,
                                                       in_idx,
                                                       in_indptr,
                                                       batch_size,
                                                       len,
                                                       k,
                                                       out,
                                                       out_idx,
                                                       select_min,
                                                       resource::get_cuda_stream(res),
                                                       resource::get_workspace_resource_ref(res));
  } else {
    calc_launch_parameter<warp_sort_filtered, T, IdxT>(
      res, batch_size, len, k, &num_of_block, &num_of_warp);
    select_k_<warp_sort_filtered, T, IdxT, RowLayout>(resource::get_dry_run_flag(res),
                                                      num_of_block,
                                                      num_of_warp,
                                                      in,
                                                      in_idx,
                                                      in_indptr,
                                                      batch_size,
                                                      len,
                                                      k,
                                                      out,
                                                      out_idx,
                                                      select_min,
                                                      resource::get_cuda_stream(res),
                                                      resource::get_workspace_resource_ref(res));
  }
}

}  // namespace matrix::detail::select::warpsort
}  // namespace raft
