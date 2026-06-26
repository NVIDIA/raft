/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/select_k_layout.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>

namespace raft {
namespace matrix::detail::select::radix {
namespace impl {

constexpr int VECTORIZED_READ_SIZE = 16;

template <int BitsPerPass>
_RAFT_HOST_DEVICE constexpr int calc_num_buckets()
{
  return 1 << BitsPerPass;
}

template <typename T, int BitsPerPass>
_RAFT_HOST_DEVICE constexpr int calc_num_passes()
{
  return ceildiv<int>(sizeof(T) * 8, BitsPerPass);
}

/**
 * Bit 0 is the least significant (rightmost);
 * this implementation processes input from the most to the least significant bit.
 * This way, we can skip some passes in the end at the cost of having an unsorted output.
 *
 * NB: Use pass=-1 for calc_mask().
 */
template <typename T, int BitsPerPass>
_RAFT_DEVICE constexpr int calc_start_bit(int pass)
{
  int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
  if (start_bit < 0) { start_bit = 0; }
  return start_bit;
}

template <typename T, int BitsPerPass>
_RAFT_DEVICE constexpr unsigned calc_mask(int pass)
{
  static_assert(BitsPerPass <= 31);
  int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
  return (1 << num_bits) - 1;
}

/**
 * Use CUB to twiddle bits - so that we can correctly compare bits of floating-point values as well
 * as of integers.
 */
template <typename T>
_RAFT_DEVICE typename cub::Traits<T>::UnsignedBits twiddle_in(T key, bool select_min)
{
  auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
  bits      = cub::Traits<T>::TwiddleIn(bits);
  if (!select_min) { bits = ~bits; }
  return bits;
}

template <typename T>
_RAFT_DEVICE T twiddle_out(typename cub::Traits<T>::UnsignedBits bits, bool select_min)
{
  if (!select_min) { bits = ~bits; }
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T&>(bits);
}

template <typename T, int BitsPerPass>
_RAFT_DEVICE int calc_bucket(T x, int start_bit, unsigned mask, bool select_min)
{
  static_assert(BitsPerPass <= sizeof(int) * 8 - 1,
                "BitsPerPass is too large that the result type could not be int");
  return (twiddle_in(x, select_min) >> start_bit) & mask;
}

// Strangely, RATIO_T has a strong impact on register usage and occupancy for sm80, e.g.
// using RATIO_T=unsigned for radix_kernel decreases occupancy (with CUDA 12).
// In the meanwhile, RATIO_T has no impact for sm90.
template <typename T, typename IdxT, typename RATIO_T = float>
_RAFT_HOST_DEVICE IdxT calc_buf_len(IdxT len)
{
  // When writing is skipped, only read `in`(type T).
  // When writing is not skipped, read `in_buf`(T) and `in_idx_buf`(IdxT), and write `out_buf`(T)
  // and `out_idx_buf`(IdxT).
  // The ratio between these cases determines whether to skip writing and hence the buffer size.
  constexpr RATIO_T ratio = 2 + sizeof(IdxT) * 2 / sizeof(T);
  // Even such estimation is too conservative, so further decrease buf_len by 1/8
  IdxT buf_len = len / (ratio * 8);

  // one-block kernel splits one large buffer into smaller ones, so round buf size to 256 bytes to
  // avoid alignment issues
  static_assert(is_a_power_of_two(sizeof(T)));
  static_assert(is_a_power_of_two(sizeof(IdxT)));
  constexpr IdxT aligned = 256 / std::min(sizeof(T), sizeof(IdxT));
  buf_len                = Pow2<aligned>::roundDown(buf_len);
  return buf_len;
}

/**
 * Map a Func over the input data, using vectorized load instructions if possible.
 *
 * NB: in future, we should move this to cpp/include/raft/linalg/detail/unary_op.cuh, which
 *     currently does not support the second lambda argument (index of an element)
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param thread_rank rank of the calling thread among all participating threads
 * @param num_threads number of the threads that participate in processing
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename IdxT, typename Func>
_RAFT_DEVICE void vectorized_process(
  size_t thread_rank, size_t num_threads, const T* in, IdxT len, Func f)
{
  if constexpr (sizeof(T) >= VECTORIZED_READ_SIZE || VECTORIZED_READ_SIZE % sizeof(T) != 0) {
    for (IdxT i = thread_rank; i < len; i += num_threads) {
      f(in[i], i);
    }
  } else {
    using wide_t      = TxN_t<T, VECTORIZED_READ_SIZE / sizeof(T)>;
    using align_bytes = Pow2<(size_t)VECTORIZED_READ_SIZE>;
    using align_elems = Pow2<wide_t::Ratio>;
    wide_t wide;

    // how many elements to skip in order to do aligned vectorized load
    const IdxT skip_cnt_left = std::min<IdxT>((IdxT)(align_bytes::roundUp(in) - in), len);

    // The main loop: process all aligned data
    for (IdxT i = thread_rank * wide_t::Ratio + skip_cnt_left; i + wide_t::Ratio <= len;
         i += num_threads * wide_t::Ratio) {
      wide.load(in, i);
#pragma unroll
      for (int j = 0; j < wide_t::Ratio; ++j) {
        f(wide.val.data[j], i + j);
      }
    }

    static_assert(WarpSize >= wide_t::Ratio);
    // Processes the skipped elements on the left
    if (thread_rank < skip_cnt_left) { f(in[thread_rank], thread_rank); }
    // Processes the skipped elements on the right
    const IdxT skip_cnt_right = align_elems::mod(len - skip_cnt_left);
    const IdxT remain_i       = len - skip_cnt_right + thread_rank;
    if (remain_i < len) { f(in[remain_i], remain_i); }
  }
}

template <typename T, typename IdxT>
struct alignas(128) Counter {
  // We are processing the values in multiple passes, from most significant to least significant. In
  // each pass, we keep the length of input (`len`) and the `k` of current pass, and update them at
  // the end of the pass.
  IdxT k;
  IdxT len;

  //  `previous_len` is the length of input in previous pass. Note that `previous_len` rather
  //  than `len` is used for the filtering step because filtering is indeed for previous pass (see
  //  comments before `radix_kernel`).
  IdxT previous_len;

  // We determine the bits of the k_th value inside the mask processed by the pass. The
  // already known bits are stored in `kth_value_bits`. It's used to discriminate a element is a
  // result (written to `out`), a candidate for next pass (written to `out_buf`), or not useful
  // (discarded). The bits that are not yet processed do not matter for this purpose.
  typename cub::Traits<T>::UnsignedBits kth_value_bits;

  // Record how many elements have passed filtering. It's used to determine the position in the
  // `out_buf` where an element should be written.
  alignas(128) IdxT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This counter is used to
  // determine if the current block is the last running block. If so, this block will execute scan()
  // and choose_bucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements less (if
  // select_min==true) than the k-th value are written from front to back.
  alignas(128) IdxT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements equal to the k-th
  // value are written from back to front. We need to keep count of them separately because the
  // number of elements that <= the k-th value might exceed k.
  alignas(128) IdxT out_back_cnt;
};

/**
 * Fused filtering of the current pass and building histogram for the next pass
 * (see steps 4 & 1 in `radix_kernel` description).
 *
 * This function is more complicated than the one-block counterpart because this function handles
 * the case of early stopping. When early stopping is triggered, it's desirable to do the final
 * filtering in this function rather than in last_filter(), because this function is run by multiple
 * blocks while last_filter is run by a single block.
 */
template <typename T, typename IdxT, int BitsPerPass>
_RAFT_DEVICE void filter_and_histogram(const T* in_buf,
                                       const IdxT* in_idx_buf,
                                       T* out_buf,
                                       IdxT* out_idx_buf,
                                       T* out,
                                       IdxT* out_idx,
                                       IdxT previous_len,
                                       Counter<T, IdxT>* counter,
                                       IdxT* histogram,
                                       bool select_min,
                                       int pass,
                                       bool early_stop)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ IdxT histogram_smem[num_buckets];
  for (IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram_smem[i] = 0;
  }
  __syncthreads();

  const int start_bit = calc_start_bit<T, BitsPerPass>(pass);
  const unsigned mask = calc_mask<T, BitsPerPass>(pass);

  if (pass == 0) {
    // Passed to vectorized_process, this function executes in all blocks in parallel,
    // i.e. the work is split along the input (both, in batches and chunks of a single row).
    // Later, the histograms are merged using atomicAdd.
    auto f = [select_min, start_bit, mask](T value, IdxT) {
      int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
      atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
    };
    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                       static_cast<size_t>(blockDim.x) * gridDim.x,
                       in_buf,
                       previous_len,
                       f);
  } else {
    IdxT* p_filter_cnt           = &counter->filter_cnt;
    IdxT* p_out_cnt              = &counter->out_cnt;
    const auto kth_value_bits    = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    // See the remark above on the distributed execution of `f` using vectorized_process.
    auto f = [in_idx_buf,
              out_buf,
              out_idx_buf,
              out,
              out_idx,
              select_min,
              start_bit,
              mask,
              previous_start_bit,
              kth_value_bits,
              p_filter_cnt,
              p_out_cnt,
              early_stop](T value, IdxT i) {
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        if (early_stop) {
          IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
          out[pos]     = value;
          out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
        } else {
          if (out_buf) {
            IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
            out_buf[pos]     = value;
            out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
          }

          int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
          atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
        }
      }
      // the condition `(out_buf || early_stop)` is a little tricky:
      // If we skip writing to `out_buf` (when `out_buf` is nullptr), we should skip writing to
      // `out` too. So we won't write the same value to `out` multiple times in different passes.
      // And if we keep skipping the writing, values will be written in `last_filter_kernel()` at
      // last. But when `early_stop` is true, we need to write to `out` since it's the last chance.
      else if ((out_buf || early_stop) && previous_bits < kth_value_bits) {
        IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    };
    vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                       static_cast<size_t>(blockDim.x) * gridDim.x,
                       in_buf,
                       previous_len,
                       f);
  }
  if (early_stop) { return; }
  __syncthreads();

  // merge histograms produced by individual blocks
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    if (histogram_smem[i] != 0) { atomicAdd(histogram + i, histogram_smem[i]); }
  }
}

/**
 * Replace histogram with its own prefix sum
 * (step 2 in `radix_kernel` description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
_RAFT_DEVICE void scan(volatile IdxT* histogram)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  if constexpr (num_buckets >= BlockSize) {
    static_assert(num_buckets % BlockSize == 0);
    constexpr int items_per_thread = num_buckets / BlockSize;
    typedef cub::BlockLoad<IdxT, BlockSize, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
    typedef cub::BlockStore<IdxT, BlockSize, items_per_thread, cub::BLOCK_STORE_TRANSPOSE>
      BlockStore;
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;

    __shared__ union {
      typename BlockLoad::TempStorage load;
      typename BlockScan::TempStorage scan;
      typename BlockStore::TempStorage store;
    } temp_storage;
    IdxT thread_data[items_per_thread];

    BlockLoad(temp_storage.load).Load(histogram, thread_data);
    __syncthreads();

    BlockScan(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    BlockStore(temp_storage.store).Store(histogram, thread_data);
  } else {
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    IdxT thread_data = 0;
    if (threadIdx.x < num_buckets) { thread_data = histogram[threadIdx.x]; }

    BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    if (threadIdx.x < num_buckets) { histogram[threadIdx.x] = thread_data; }
  }
}

/**
 * Calculate in which bucket the k-th value will fall
 *  (steps 3 in `radix_kernel` description)
 */
template <typename T, typename IdxT, int BitsPerPass>
_RAFT_DEVICE void choose_bucket(Counter<T, IdxT>* counter,
                                const IdxT* histogram,
                                const IdxT k,
                                const int pass)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    IdxT prev = (i == 0) ? 0 : histogram[i - 1];
    IdxT cur  = histogram[i];

    // one and only one thread will satisfy this condition, so counter is written by only one thread
    if (prev < k && cur >= k) {
      counter->k   = k - prev;    // how many values still are there to find
      counter->len = cur - prev;  // number of values in next pass
      typename cub::Traits<T>::UnsignedBits bucket = i;
      int start_bit                                = calc_start_bit<T, BitsPerPass>(pass);
      counter->kth_value_bits |= bucket << start_bit;
    }
  }
}

// For one-block version, last_filter() could be called when pass < num_passes - 1.
// So `pass` could not be constexpr
template <typename T, typename IdxT, int BitsPerPass>
_RAFT_DEVICE void last_filter(const T* in_buf,
                              const IdxT* in_idx_buf,
                              T* out,
                              IdxT* out_idx,
                              IdxT current_len,
                              IdxT k,
                              Counter<T, IdxT>* counter,
                              const bool select_min,
                              const int pass)
{
  const auto kth_value_bits = counter->kth_value_bits;
  const int start_bit       = calc_start_bit<T, BitsPerPass>(pass);

  // changed in choose_bucket(); need to reload
  const IdxT num_of_kth_needed = counter->k;
  IdxT* p_out_cnt              = &counter->out_cnt;
  IdxT* p_out_back_cnt         = &counter->out_back_cnt;
  for (IdxT i = threadIdx.x; i < current_len; i += blockDim.x) {
    const T value   = in_buf[i];
    const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
    if (bits < kth_value_bits) {
      IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
      out[pos] = value;
      // For one-block version, `in_idx_buf` could be nullptr at pass 0.
      // For non one-block version, if writing has been skipped, `in_idx_buf` could be nullptr if
      // `in_buf` is `in`
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
      if (back_pos < num_of_kth_needed) {
        IdxT pos     = k - 1 - back_pos;
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

template <typename T, typename IdxT>
_RAFT_DEVICE void set_buf_pointers(const T* in,
                                   const IdxT* in_idx,
                                   uintptr_t bufs,
                                   IdxT buf_len,
                                   int pass,
                                   const T*& in_buf,
                                   const IdxT*& in_idx_buf,
                                   T*& out_buf,
                                   IdxT*& out_idx_buf)
{
  // bufs consists of 4 pieces in order: buf1, buf2, idx_buf1, idx_buf2
  if (pass == 0) {
    in_buf      = in;
    in_idx_buf  = nullptr;
    out_buf     = nullptr;
    out_idx_buf = nullptr;
  } else if (pass == 1) {
    in_buf      = in;
    in_idx_buf  = in_idx;
    out_buf     = reinterpret_cast<T*>(bufs);
    out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
  } else if (pass % 2 == 0) {
    in_buf      = reinterpret_cast<T*>(bufs);
    in_idx_buf  = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
    out_buf     = const_cast<T*>(in_buf + buf_len);
    out_idx_buf = const_cast<IdxT*>(in_idx_buf + buf_len);
  } else {
    out_buf     = reinterpret_cast<T*>(bufs);
    out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
    in_buf      = out_buf + buf_len;
    in_idx_buf  = out_idx_buf + buf_len;
  }
}

template <typename T, typename IdxT>
_RAFT_DEVICE void set_buf_pointers(const T* in,
                                   const IdxT* in_idx,
                                   uintptr_t bufs,
                                   IdxT buf_len,
                                   const int pass,
                                   const T*& out_buf,
                                   const IdxT*& out_idx_buf)
{
  // bufs consists of 4 pieces in order: buf1, buf2, idx_buf1, idx_buf2
  if (pass == 0) {
    out_buf     = nullptr;
    out_idx_buf = nullptr;
  } else if (pass == 1) {
    out_buf     = reinterpret_cast<T*>(bufs);
    out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
  } else if (pass % 2 == 0) {
    out_buf = const_cast<T*>(reinterpret_cast<T*>(bufs) + buf_len);
    out_idx_buf =
      const_cast<IdxT*>(reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len) + buf_len);
  } else {
    out_buf     = reinterpret_cast<T*>(bufs);
    out_idx_buf = reinterpret_cast<IdxT*>(bufs + sizeof(T) * 2 * buf_len);
  }
}

template <typename T, typename IdxT, int BitsPerPass, typename RowLayout>
RAFT_KERNEL last_filter_kernel(const T* in,
                               const IdxT* in_idx,
                               uintptr_t bufs,
                               size_t offset,
                               T* out,
                               IdxT* out_idx,
                               const IdxT len,
                               const IdxT* len_i,
                               const IdxT k,
                               Counter<T, IdxT>* counters,
                               const bool select_min)
{
  const size_t batch_id = blockIdx.y;  // size_t to avoid multiplication overflow

  Counter<T, IdxT>* counter = counters + batch_id;
  IdxT previous_len         = counter->previous_len;

  if (previous_len == 0) { return; }

  const auto [l_len, l_offset] = RowLayout::compute(len, offset, batch_id, len_i);

  const IdxT buf_len = calc_buf_len<T>(len);

  const T* in_buf        = nullptr;
  const IdxT* in_idx_buf = nullptr;
  bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  constexpr int pass      = calc_num_passes<T, BitsPerPass>() - 1;
  constexpr int start_bit = calc_start_bit<T, BitsPerPass>(pass);

  set_buf_pointers(in + l_offset, in_idx + l_offset, bufs, buf_len, pass, in_buf, in_idx_buf);

  if (previous_len > buf_len || in_buf == in + l_offset) {
    in_buf       = in + l_offset;
    in_idx_buf   = in_idx ? (in_idx + l_offset) : nullptr;
    previous_len = l_len;
  }
  out += batch_id * k;
  out_idx += batch_id * k;

  const auto kth_value_bits    = counter->kth_value_bits;
  const IdxT num_of_kth_needed = counter->k;
  IdxT* p_out_cnt              = &counter->out_cnt;
  IdxT* p_out_back_cnt         = &counter->out_back_cnt;

  auto f = [k,
            select_min,
            kth_value_bits,
            num_of_kth_needed,
            p_out_cnt,
            p_out_back_cnt,
            in_idx_buf,
            out,
            out_idx](T value, IdxT i) {
    const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
    if (bits < kth_value_bits) {
      IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
      out[pos]     = value;
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
      if (back_pos < num_of_kth_needed) {
        IdxT pos     = k - 1 - back_pos;
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  };

  vectorized_process(static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x,
                     static_cast<size_t>(blockDim.x) * gridDim.x,
                     in_buf,
                     previous_len,
                     f);
}

template <typename T, typename IdxT, typename S>
_RAFT_DEVICE _RAFT_FORCEINLINE void copy_in_val(
  T* dest, const T* src, S len, IdxT k, const bool select_min)
{
  S idx               = S(threadIdx.x);
  S stride            = S(blockDim.x);
  const T default_val = select_min ? upper_bound<T>() : lower_bound<T>();
  for (S i = idx; i < k; i += stride) {
    dest[i] = i < len ? src[i] : default_val;
  }
}

template <typename T, typename S>
_RAFT_DEVICE _RAFT_FORCEINLINE void copy_in_idx(T* dest, const T* src, S len)
{
  S idx    = S(threadIdx.x);
  S stride = S(blockDim.x);

  for (S i = idx; i < len; i += stride) {
    dest[i] = src ? src[i] : i;
  }
}

/**
 *
 * It is expected to call this kernel multiple times (passes), in each pass we process a radix,
 * going from the most significant towards the least significant bits (MSD).
 *
 * Conceptually, each pass consists of 4 steps:
 *
 * 1. Calculate histogram
 *      First, transform bits into a digit, the value of which is in the range
 *      [0, 2^{BITS_PER_PASS}-1]. Then count the frequency of each digit value and the result is a
 *      histogram. That is, histogram[i] contains the count of inputs having value i.
 *
 * 2. Scan the histogram
 *      Inclusive prefix sum is computed for the histogram. After this step, histogram[i] contains
 *      the count of inputs having value <= i.
 *
 * 3. Find the bucket j of the histogram that the k-th value falls into
 *
 * 4. Filtering
 *      Input elements whose digit value <j are the top-k elements. We put them into the result
 *      array out. The number of such elements is histogram[j-1]. Since the k-th value must be in
 *      the bucket j, we write all elements in bucket j into a intermediate buffer out_buf. For the
 *      next pass, these elements are used as input, and we would like to find the
 *      (k - histogram[j-1])-th value among them. That is, the k in the next pass is set to
 *      (k - histogram[j-1]).
 *
 * In the implementation, the filtering step is delayed to the next pass so the filtering and
 * histogram computation are fused. In this way, inputs are read once rather than twice.
 *
 * During the filtering step, we won't write candidates (elements in bucket j) to `out_buf` if the
 * number of candidates is larger than the length of `out_buf` (this could happen when the leading
 * bits of input values are almost the same). And then in the next pass, inputs are read from `in`
 * rather than from `in_buf`. The benefit is that we can save the cost of writing candidates and
 * their indices.
 */
template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          bool fused_last_filter,
          typename RowLayout>
RAFT_KERNEL radix_kernel(const T* in,
                         const IdxT* in_idx,
                         uintptr_t bufs,
                         size_t offset,
                         T* out,
                         IdxT* out_idx,
                         Counter<T, IdxT>* counters,
                         IdxT* histograms,
                         const IdxT len,
                         const IdxT* len_i,
                         const IdxT k,
                         const bool select_min,
                         const int pass)
{
  const size_t batch_id = blockIdx.y;
  auto counter          = counters + batch_id;
  IdxT current_k;
  IdxT previous_len;
  IdxT current_len;

  const auto [l_len, l_offset] = RowLayout::compute(len, offset, batch_id, len_i);

  if (pass == 0) {
    current_k    = k;
    previous_len = l_len;
    // Need to do this so setting counter->previous_len for the next pass is correct.
    // This value is meaningless for pass 0, but it's fine because pass 0 won't be the
    // last pass in this implementation so pass 0 won't hit the "if (pass ==
    // num_passes - 1)" branch.
    // Maybe it's better to reload counter->previous_len and use it rather than
    // current_len in last_filter()
    current_len = l_len;
  } else {
    current_k    = counter->k;
    current_len  = counter->len;
    previous_len = counter->previous_len;
  }
  if constexpr (!RowLayout::is_uniform) {
    if (pass == 0 && l_len <= k) {
      copy_in_val(out + batch_id * k, in + l_offset, l_len, k, select_min);
      copy_in_idx(out_idx + batch_id * k, (in_idx ? (in_idx + l_offset) : nullptr), l_len);
      if (threadIdx.x == 0) {
        counter->previous_len = 0;
        counter->len          = 0;
      }
      __syncthreads();
      return;
    }
  }

  if (current_len == 0) { return; }

  // When k=len, early_stop will be true at pass 0. It means filter_and_histogram() should handle
  // correctly the case that pass=0 and early_stop=true. However, this special case of k=len is
  // handled in other way in select_k() so such case is not possible here.
  const bool early_stop = (current_len == current_k);
  const IdxT buf_len    = calc_buf_len<T>(len);

  const T* in_buf;
  const IdxT* in_idx_buf;
  T* out_buf;
  IdxT* out_idx_buf;
  bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  set_buf_pointers(in + l_offset,
                   (in_idx ? (in_idx + l_offset) : nullptr),
                   bufs,
                   buf_len,
                   pass,
                   in_buf,
                   in_idx_buf,
                   out_buf,
                   out_idx_buf);

  // "previous_len > buf_len" means previous pass skips writing buffer
  if (pass == 0 || pass == 1 || previous_len > buf_len) {
    in_buf       = in + l_offset;
    in_idx_buf   = in_idx ? (in_idx + l_offset) : nullptr;
    previous_len = l_len;
  }

  // in case we have individual len for each query defined we want to make sure
  // that we only iterate valid elements.
  if (len_i != nullptr) {
    const IdxT max_len = max(l_len, k);
    if (max_len < previous_len) previous_len = max_len;
  }

  // "current_len > buf_len" means current pass will skip writing buffer
  if (pass == 0 || current_len > buf_len) {
    out_buf     = nullptr;
    out_idx_buf = nullptr;
  }
  out += batch_id * k;
  out_idx += batch_id * k;

  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  auto histogram            = histograms + batch_id * num_buckets;

  filter_and_histogram<T, IdxT, BitsPerPass>(in_buf,
                                             in_idx_buf,
                                             out_buf,
                                             out_idx_buf,
                                             out,
                                             out_idx,
                                             previous_len,
                                             counter,
                                             histogram,
                                             select_min,
                                             pass,
                                             early_stop);
  __threadfence();
  // Ensure every thread in this block has finished its writes before finished_block_cnt is updated.
  // Otherwise, the last block (isLastBlock) will read an incomplete histogram.
  __syncthreads();

  bool isLastBlock = false;
  if (threadIdx.x == 0) {
    unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
    isLastBlock           = (finished == (gridDim.x - 1));
  }
  if (__syncthreads_or(isLastBlock)) {
    if (early_stop) {
      if (threadIdx.x == 0) {
        // `last_filter_kernel()` requires setting previous_len
        counter->previous_len = 0;
        counter->len          = 0;
      }
      return;
    }

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();
    choose_bucket<T, IdxT, BitsPerPass>(counter, histogram, current_k, pass);
    __syncthreads();

    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
    // reset for next pass
    if (pass != num_passes - 1) {
      for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        histogram[i] = 0;
      }
    }
    if (threadIdx.x == 0) {
      // `last_filter_kernel()` requires setting previous_len even in the last pass
      counter->previous_len = current_len;
      // not necessary for the last pass, but put it here anyway
      counter->filter_cnt = 0;
    }

    if constexpr (fused_last_filter) {
      if (pass == num_passes - 1) {
        last_filter<T, IdxT, BitsPerPass>(out_buf ? out_buf : in_buf,
                                          out_idx_buf ? out_idx_buf : in_idx_buf,
                                          out,
                                          out_idx,
                                          out_buf ? current_len : l_len,
                                          k,
                                          counter,
                                          select_min,
                                          pass);
      }
    }
  }
}

template <typename T, typename IdxT, int BlockSize, typename Kernel>
int calc_chunk_size(int batch_size, IdxT len, int sm_cnt, Kernel kernel, bool one_block)
{
  int active_blocks;
  RAFT_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, kernel, BlockSize, 0));

  // The chunk size is chosen so that there is enough workload to fully utilize GPU.
  // One full wave contains (sm_cnt * active_blocks) blocks, and 10 waves is an empirically safe
  // estimation of enough workload. It also counteracts imbalance if some blocks run slower
  // than others.
  constexpr int num_waves = 10;
  int chunk_size;
  if (one_block) {
    // For one-block version, one block processes one instance in the chunk. Just ensure that there
    // are enough blocks.
    chunk_size = num_waves * sm_cnt * active_blocks;
  } else {
    // One instance in the chunk contains len items and is processed by multiple blocks.
    // The total number of items in a chunk (chunk_size * len) should be large enough that every
    // thread has enough items to processes. So set it to num_waves * "max num of active threads"
    // (sm_cnt * active_blocks * BlockSize) * items_per_thread.
    //
    // Also, the upper bound of the total number of items in a chunk is:
    // 10 (num_waves) * ~100 (sm_cnt) * 2048 (active_blocks*BlockSize) * 32 (items_per_thread) =64M.
    // So temporary buffer size required for one chunk won't be too large.
    constexpr int items_per_thread = 32;
    chunk_size =
      std::max<int>(1, num_waves * sm_cnt * active_blocks * BlockSize * items_per_thread / len);
  }
  return std::min(chunk_size, batch_size);
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
unsigned calc_grid_dim(int batch_size, IdxT len, int sm_cnt)
{
  static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);

  int active_blocks;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks,
    radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, select::dense_layout<IdxT>>,
    BlockSize,
    0));
  active_blocks *= sm_cnt;

  IdxT best_num_blocks         = 0;
  float best_tail_wave_penalty = 1.0f;
  const IdxT max_num_blocks    = ceildiv<IdxT>(len, VECTORIZED_READ_SIZE / sizeof(T) * BlockSize);
  for (int num_waves = 1;; ++num_waves) {
    IdxT num_blocks = std::min(
      max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
    IdxT items_per_thread  = ceildiv<IdxT>(len, num_blocks * BlockSize);
    items_per_thread       = alignTo<IdxT>(items_per_thread, VECTORIZED_READ_SIZE / sizeof(T));
    num_blocks             = ceildiv<IdxT>(len, items_per_thread * BlockSize);
    float actual_num_waves = static_cast<float>(num_blocks) * batch_size / active_blocks;
    float tail_wave_penalty =
      (ceilf(actual_num_waves) - actual_num_waves) / ceilf(actual_num_waves);

    // 0.15 is determined experimentally. It also ensures breaking the loop early,
    // e.g. when num_waves > 7, tail_wave_penalty will always <0.15
    if (tail_wave_penalty < 0.15) {
      best_num_blocks = num_blocks;
      break;
    } else if (tail_wave_penalty < best_tail_wave_penalty) {
      best_num_blocks        = num_blocks;
      best_tail_wave_penalty = tail_wave_penalty;
    }

    if (num_blocks == max_num_blocks) { break; }
  }
  return best_num_blocks;
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize, typename RowLayout>
void radix_topk(bool dry_run,
                const T* in,
                const IdxT* in_idx,
                int batch_size,
                IdxT len,
                IdxT k,
                T* out,
                IdxT* out_idx,
                bool select_min,
                bool fused_last_filter,
                const IdxT* len_i,
                unsigned grid_dim,
                int sm_cnt,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr)
{
  // TODO: is it possible to relax this restriction?
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

  auto kernel = radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, RowLayout>;
  const size_t max_chunk_size =
    calc_chunk_size<T, IdxT, BlockSize>(batch_size, len, sm_cnt, kernel, false);
  if (max_chunk_size != static_cast<size_t>(batch_size)) {
    grid_dim = calc_grid_dim<T, IdxT, BitsPerPass, BlockSize>(max_chunk_size, len, sm_cnt);
  }
  const IdxT buf_len = calc_buf_len<T>(len);

  size_t req_buf = max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  rmm::device_uvector<Counter<T, IdxT>> counters(max_chunk_size, stream, mr);
  rmm::device_uvector<IdxT> histograms(max_chunk_size * num_buckets, stream, mr);

  rmm::device_buffer bufs(max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT)), stream, mr);

  if (dry_run) { return; }

  for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
    int chunk_size = std::min(max_chunk_size, batch_size - offset);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(counters.data(), 0, counters.size() * sizeof(Counter<T, IdxT>), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms.data(), 0, histograms.size() * sizeof(IdxT), stream));
    auto kernel = radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, RowLayout>;

    T* chunk_out            = out + offset * k;
    IdxT* chunk_out_idx     = out_idx + offset * k;
    const IdxT* chunk_len_i = len_i ? (len_i + offset) : nullptr;

    dim3 blocks(grid_dim, chunk_size);
    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();

    for (int pass = 0; pass < num_passes; ++pass) {
      if (fused_last_filter && pass == num_passes - 1) {
        kernel = radix_kernel<T, IdxT, BitsPerPass, BlockSize, true, RowLayout>;
      }

      kernel<<<blocks, BlockSize, 0, stream>>>(in,
                                               in_idx,
                                               reinterpret_cast<uintptr_t>(bufs.data()),
                                               offset,
                                               chunk_out,
                                               chunk_out_idx,
                                               counters.data(),
                                               histograms.data(),
                                               len,
                                               chunk_len_i,
                                               k,
                                               select_min,
                                               pass);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }

    if (!fused_last_filter) {
      last_filter_kernel<T, IdxT, BitsPerPass, RowLayout>
        <<<blocks, BlockSize, 0, stream>>>(in,
                                           in_idx,
                                           reinterpret_cast<uintptr_t>(bufs.data()),
                                           offset,
                                           chunk_out,
                                           chunk_out_idx,
                                           len,
                                           chunk_len_i,
                                           k,
                                           counters.data(),
                                           select_min);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }
}

// The following a few functions are for the one-block version, which uses single thread block for
// each row of a batch.
template <typename T, typename IdxT, int BitsPerPass>
_RAFT_DEVICE void filter_and_histogram_for_one_block(const T* in_buf,
                                                     const IdxT* in_idx_buf,
                                                     T* out_buf,
                                                     IdxT* out_idx_buf,
                                                     T* out,
                                                     IdxT* out_idx,
                                                     const IdxT previous_len,
                                                     Counter<T, IdxT>* counter,
                                                     IdxT* histogram,
                                                     bool select_min,
                                                     int pass)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram[i] = 0;
  }
  IdxT* p_filter_cnt = &counter->filter_cnt;
  if (threadIdx.x == 0) { *p_filter_cnt = 0; }
  __syncthreads();

  const int start_bit = calc_start_bit<T, BitsPerPass>(pass);
  const unsigned mask = calc_mask<T, BitsPerPass>(pass);

  if (pass == 0) {
    auto f = [histogram, select_min, start_bit, mask](T value, IdxT) {
      int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
      atomicAdd(histogram + bucket, static_cast<IdxT>(1));
    };
    vectorized_process(threadIdx.x, blockDim.x, in_buf, previous_len, f);
  } else if (!out_buf) {
    // not use vectorized_process here because it increases #registers a lot
    const auto kth_value_bits    = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value            = in_buf[i];
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1));
      }
    }
  } else {
    // not use vectorized_process here because it increases #registers a lot
    IdxT* p_out_cnt              = &counter->out_cnt;
    const auto kth_value_bits    = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value            = in_buf[i];
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
        out_buf[pos]     = value;
        out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;

        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1));
      } else if (previous_bits < kth_value_bits) {
        IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize, typename RowLayout>
RAFT_KERNEL radix_topk_one_block_kernel(const T* in,
                                        const IdxT* in_idx,
                                        const IdxT len,
                                        const IdxT* len_i,
                                        const IdxT k,
                                        T* out,
                                        IdxT* out_idx,
                                        const bool select_min,
                                        uintptr_t bufs,
                                        size_t offset)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  const size_t batch_id = blockIdx.x;  // size_t to avoid multiplication overflow

  const auto [l_len, l_offset] = RowLayout::compute(len, offset, batch_id, len_i);

  if (threadIdx.x == 0) {
    counter.k              = k;
    counter.len            = l_len;
    counter.previous_len   = l_len;
    counter.kth_value_bits = 0;
    counter.out_cnt        = 0;
    counter.out_back_cnt   = 0;
  }
  __syncthreads();

  in += l_offset;
  if (in_idx) { in_idx += l_offset; }
  out += batch_id * k;
  out_idx += batch_id * k;
  const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);
  bufs += batch_id * buf_len * 2 * (sizeof(T) + sizeof(IdxT));

  if constexpr (!RowLayout::is_uniform) {
    if (l_len <= k) {
      copy_in_val(out, in, l_len, k, select_min);
      copy_in_idx(out_idx, in_idx, l_len);
      __syncthreads();
      return;
    }
  }

  constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
  for (int pass = 0; pass < num_passes; ++pass) {
    const T* in_buf;
    const IdxT* in_idx_buf;
    T* out_buf;
    IdxT* out_idx_buf;
    set_buf_pointers(in, in_idx, bufs, buf_len, pass, in_buf, in_idx_buf, out_buf, out_idx_buf);

    const IdxT current_len = counter.len;
    const IdxT current_k   = counter.k;
    IdxT previous_len      = counter.previous_len;
    if (previous_len > buf_len) {
      in_buf       = in;
      in_idx_buf   = in_idx;
      previous_len = len;
    }
    if (current_len > buf_len) {
      // so "out_buf==nullptr" denotes skipping writing buffer in current pass
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }

    // in case we have individual len for each query defined we want to make sure
    // that we only iterate valid elements.
    if (len_i != nullptr) {
      const IdxT max_len = max(l_len, k);
      if (max_len < previous_len) previous_len = max_len;
    }

    filter_and_histogram_for_one_block<T, IdxT, BitsPerPass>(in_buf,
                                                             in_idx_buf,
                                                             out_buf,
                                                             out_idx_buf,
                                                             out,
                                                             out_idx,
                                                             previous_len,
                                                             &counter,
                                                             histogram,
                                                             select_min,
                                                             pass);
    __syncthreads();

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();

    choose_bucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
    if (threadIdx.x == 0) { counter.previous_len = current_len; }
    __syncthreads();

    if (counter.len == counter.k || pass == num_passes - 1) {
      last_filter<T, IdxT, BitsPerPass>(out_buf ? out_buf : in,
                                        out_buf ? out_idx_buf : in_idx,
                                        out,
                                        out_idx,
                                        out_buf ? current_len : l_len,
                                        k,
                                        &counter,
                                        select_min,
                                        pass);
      break;
    }
  }
}

// radix_topk() might use multiple thread blocks for one row of a batch. In contrast, the following
// one-block version uses single thread block for one row of a batch, so intermediate data, like
// counters and global histograms, can be kept in shared memory and cheap sync operations can be
// used. It's used when len is relatively small or when the number of blocks per row calculated by
// `calc_grid_dim()` is 1.
template <typename T, typename IdxT, int BitsPerPass, int BlockSize, typename RowLayout>
void radix_topk_one_block(bool dry_run,
                          const T* in,
                          const IdxT* in_idx,
                          int batch_size,
                          IdxT len,
                          IdxT k,
                          T* out,
                          IdxT* out_idx,
                          bool select_min,
                          const IdxT* len_i,
                          int sm_cnt,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);

  auto kernel        = radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize, RowLayout>;
  const IdxT buf_len = calc_buf_len<T, IdxT, unsigned>(len);
  const size_t max_chunk_size =
    calc_chunk_size<T, IdxT, BlockSize>(batch_size, len, sm_cnt, kernel, true);

  rmm::device_buffer bufs(max_chunk_size * buf_len * 2 * (sizeof(T) + sizeof(IdxT)), stream, mr);

  if (dry_run) { return; }

  for (size_t offset = 0; offset < static_cast<size_t>(batch_size); offset += max_chunk_size) {
    int chunk_size          = std::min(max_chunk_size, batch_size - offset);
    const IdxT* chunk_len_i = len_i ? (len_i + offset) : nullptr;
    kernel<<<chunk_size, BlockSize, 0, stream>>>(in,
                                                 in_idx,
                                                 len,
                                                 chunk_len_i,
                                                 k,
                                                 out + offset * k,
                                                 out_idx + offset * k,
                                                 select_min,
                                                 reinterpret_cast<uintptr_t>(bufs.data()),
                                                 offset);
  }
}

}  // namespace impl

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_keys` as a row-major matrix with len columns and
 * batch_size rows, then this function selects k smallest/largest values in each row and fills
 * in the row-major matrix `out` of size (batch_size, k).
 *
 * Note, the output is NOT sorted within the groups of `k` selected elements.
 *
 * Reference:
 * Jingrong Zhang, Akira Naruse, Xipeng Li, and Yong Wang. 2023. Parallel Top-K Algorithms on GPU:
 * A Comprehensive Study and New Methods. In The International Conference for High Performance
 * Computing, Networking, Storage and Analysis (SC ’23), November 12–17, 2023, Denver, CO, USA.
 * ACM, New York, NY, USA. https://doi.org/10.1145/3581784.3607062
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 * @tparam BitsPerPass
 *   The size of the radix;
 *   it affects the number of passes and number of buckets.
 * @tparam BlockSize
 *   Number of threads in a kernel thread block.
 * @tparam RowLayout
 *   Row-layout policy controlling how per-row offsets and lengths are computed.
 *   Use `select::dense_layout<IdxT>` for dense matrices (all rows have the same length).
 *   Use `select::csr_layout<IdxT>` for CSR matrices (row lengths come from an indptr array).
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
 * @param fused_last_filter
 *   when it's true, the last filter is fused into the kernel in the last pass and only one thread
 *   block will do the filtering; when false, a standalone filter kernel with multiple thread
 *   blocks is called. The later case is preferable when leading bits of input data are almost the
 *   same. That is, when the value range of input data is narrow. In such case, there could be a
 *   large number of inputs for the last filter, hence using multiple thread blocks is beneficial.
 * @param len_i
 *   Optional array whose interpretation depends on `RowLayout`:
 *   For `dense_layout`, this is unused (pass nullptr).
 *   For `csr_layout`, this is the CSR indptr array of size `batch_size + 1`.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize, typename RowLayout>
void select_k(raft::resources const& res,
              const T* in,
              const IdxT* in_idx,
              int batch_size,
              IdxT len,
              IdxT k,
              T* out,
              IdxT* out_idx,
              bool select_min,
              bool fused_last_filter,
              const IdxT* len_i)
{
  RAFT_EXPECTS(RowLayout::is_uniform || len_i != nullptr,
               "CSR layout requires a non-null indptr array (len_i)!");

  bool dry_run = resource::get_dry_run_flag(res);
  auto stream  = resource::get_cuda_stream(res);
  auto mr      = resource::get_workspace_resource_ref(res);
  if (k == len && RowLayout::is_uniform) {
    if (dry_run) { return; }
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(out, in, sizeof(T) * batch_size * len, cudaMemcpyDeviceToDevice, stream));
    if (in_idx) {
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        out_idx, in_idx, sizeof(IdxT) * batch_size * len, cudaMemcpyDeviceToDevice, stream));
    } else {
      auto out_idx_view =
        raft::make_device_vector_view(out_idx, static_cast<size_t>(len) * batch_size);
      raft::linalg::map_offset(res, out_idx_view, raft::mod_const_op<IdxT>(len));
    }
    return;
  }

  int sm_cnt = resource::get_device_properties(res).multiProcessorCount;

  constexpr int items_per_thread = 32;

  if (len <= BlockSize * items_per_thread) {
    impl::radix_topk_one_block<T, IdxT, BitsPerPass, BlockSize, RowLayout>(
      dry_run, in, in_idx, batch_size, len, k, out, out_idx, select_min, len_i, sm_cnt, stream, mr);
  } else {
    unsigned grid_dim =
      impl::calc_grid_dim<T, IdxT, BitsPerPass, BlockSize>(batch_size, len, sm_cnt);
    if (grid_dim == 1) {
      impl::radix_topk_one_block<T, IdxT, BitsPerPass, BlockSize, RowLayout>(dry_run,
                                                                             in,
                                                                             in_idx,
                                                                             batch_size,
                                                                             len,
                                                                             k,
                                                                             out,
                                                                             out_idx,
                                                                             select_min,
                                                                             len_i,
                                                                             sm_cnt,
                                                                             stream,
                                                                             mr);
    } else {
      impl::radix_topk<T, IdxT, BitsPerPass, BlockSize, RowLayout>(dry_run,
                                                                   in,
                                                                   in_idx,
                                                                   batch_size,
                                                                   len,
                                                                   k,
                                                                   out,
                                                                   out_idx,
                                                                   select_min,
                                                                   fused_last_filter,
                                                                   len_i,
                                                                   grid_dim,
                                                                   sm_cnt,
                                                                   stream,
                                                                   mr);
    }
  }
}

}  // namespace matrix::detail::select::radix
}  // namespace raft
