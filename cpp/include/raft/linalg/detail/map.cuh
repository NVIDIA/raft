/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/input_validation.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>
#include <raft/util/vectorized_kvp.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/tuple>

namespace raft {
namespace linalg::detail {

template <bool PassOffset, typename OutT, typename IdxT, typename Func, typename... InTs>
__device__ __forceinline__ auto map_apply(Func f, const IdxT& offset, const InTs&... ins) -> OutT
{
  if constexpr (PassOffset) {
    return f(offset, ins...);
  } else {
    return f(ins...);
  }
}

template <int R,
          bool PassOffset,
          typename OutT,
          typename IdxT,
          typename Func,
          typename... InTs,
          size_t... Is>
__device__ __forceinline__ void map_kernel_mainloop(
  OutT* out_ptr, IdxT offset, IdxT len, Func f, const InTs*... in_ptrs, std::index_sequence<Is...>)
{
  TxN_t<OutT, R> wide;
  cuda::std::tuple<TxN_t<InTs, R>...> wide_args;
  if (offset + R <= len) {
    (cuda::std::get<Is>(wide_args).load(in_ptrs, offset), ...);
#pragma unroll
    for (int j = 0; j < R; ++j) {
      wide.val.data[j] = map_apply<PassOffset, OutT, IdxT, Func, InTs...>(
        f, offset + j, cuda::std::get<Is>(wide_args).val.data[j]...);
    }
    wide.store(out_ptr, offset);
  }
}

template <int R, bool PassOffset, typename OutT, typename IdxT, typename Func, typename... InTs>
RAFT_KERNEL map_kernel(OutT* out_ptr, IdxT len, Func f, const InTs*... in_ptrs)
{
  const IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (R <= 1) {
    if (tid < len) {
      out_ptr[tid] = map_apply<PassOffset, OutT, IdxT, Func, InTs...>(f, tid, in_ptrs[tid]...);
    }
  } else {
    using align_bytes = Pow2<sizeof(OutT) * size_t(R)>;
    using align_elems = Pow2<R>;

    // how many elements to skip in order to do aligned vectorized store
    const IdxT skip_cnt_left = std::min<IdxT>(IdxT(align_bytes::roundUp(out_ptr) - out_ptr), len);

    // The main loop: process all aligned data
    map_kernel_mainloop<R, PassOffset, OutT, IdxT, Func, InTs...>(
      out_ptr, tid * R + skip_cnt_left, len, f, in_ptrs..., std::index_sequence_for<InTs...>());

    static_assert(WarpSize >= R);
    // Processes the skipped elements on the left
    if (tid < skip_cnt_left) {
      out_ptr[tid] = map_apply<PassOffset, OutT, IdxT, Func, InTs...>(f, tid, in_ptrs[tid]...);
    }
    // Processes the skipped elements on the right
    const IdxT skip_cnt_right = align_elems::mod(len - skip_cnt_left);
    const IdxT remain_i       = len - skip_cnt_right + tid;
    if (remain_i < len) {
      out_ptr[remain_i] =
        map_apply<PassOffset, OutT, IdxT, Func, InTs...>(f, remain_i, in_ptrs[remain_i]...);
    }
  }
}

template <int R, bool PassOffset, typename OutT, typename IdxT, typename Func, typename... InTs>
void map_call(rmm::cuda_stream_view stream, OutT* out_ptr, IdxT len, Func f, const InTs*... in_ptrs)
{
  const IdxT len_vectorized = raft::div_rounding_up_safe<IdxT>(len, R);
  const int threads =
    std::max<int>(WarpSize, std::min<IdxT>(raft::bound_by_power_of_two<IdxT>(len_vectorized), 256));
  const IdxT blocks = raft::div_rounding_up_unsafe<IdxT>(len_vectorized, threads);
  map_kernel<R, PassOffset><<<blocks, threads, 0, stream>>>(out_ptr, len, f, in_ptrs...);
}

constexpr int kCoalescedVectorSize = 16;
constexpr int kSmallInputThreshold = 1024;

struct ratio_selector {
  int ratio;
  int align;
  constexpr inline ratio_selector(int r, int a) : ratio(r), align(a) {}

  template <typename T>
  constexpr static auto ignoring_alignment() -> ratio_selector
  {
    // Types without IOType specializations must use ratio=1 (non-vectorized access)
    if constexpr (!is_vectorizable_type<T>::value) { return ratio_selector{1, 0}; }

    constexpr bool T_evenly_fits_in_cache_line = (kCoalescedVectorSize % sizeof(T)) == 0;

    if constexpr (T_evenly_fits_in_cache_line) {
      return ratio_selector{size_t(kCoalescedVectorSize / sizeof(T)), 0};
    } else {
      return ratio_selector{1, 0};
    }
  }

  template <typename T>
  explicit ratio_selector(const T* ptr)
  {
    constexpr auto s = ignoring_alignment<T>();  // NOLINT

    if constexpr (s.ratio == 1) {
      align = 0;
    } else {
      align = int(Pow2<sizeof(T) * s.ratio>::roundUp(ptr) - ptr);
    }
    ratio = int(s.ratio);
  }
};

constexpr inline auto operator*(const ratio_selector& a, const ratio_selector& b) -> ratio_selector
{
  auto ratio = std::min<int>(a.ratio, b.ratio);
  while ((a.align % ratio) != (b.align % ratio)) {
    ratio >>= 1;
  }
  return ratio_selector{ratio, a.align % ratio};
}

template <int R, bool PassOffset, typename OutT, typename IdxT, typename Func, typename... InTs>
void map_call_rt(
  int r, rmm::cuda_stream_view stream, OutT* out_ptr, IdxT len, Func f, const InTs*... in_ptrs)
{
  if (r >= R) { return map_call<R, PassOffset>(stream, out_ptr, len, f, in_ptrs...); }
  if constexpr (R > 1) {
    return map_call_rt<(R >> 1), PassOffset>(r, stream, out_ptr, len, f, in_ptrs...);
  }
}

template <bool PassOffset, typename OutT, typename IdxT, typename Func, typename... InTs>
void map(rmm::cuda_stream_view stream, OutT* out_ptr, IdxT len, Func f, const InTs*... in_ptrs)
{
  // don't vectorize on small inputs
  if (len <= kSmallInputThreshold) {
    return map_call<1, PassOffset>(stream, out_ptr, len, f, in_ptrs...);
  }
  constexpr int kRatio =
    (ratio_selector::ignoring_alignment<OutT>() * ... * ratio_selector::ignoring_alignment<InTs>())
      .ratio;
  static_assert(kRatio > 0, "Unexpected zero vector size.");
  const int ratio = (ratio_selector(out_ptr) * ... * ratio_selector(in_ptrs)).ratio;
  return map_call_rt<kRatio, PassOffset>(ratio, stream, out_ptr, len, f, in_ptrs...);
}

template <typename OutType,
          typename InType,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InType>>
void map_check_shape(OutType out, InType in)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(in) && out.size() == in.size(),
               "All inputs must be contiguous and have the same size as the output");
}

/**
 * @brief Map a function over a zero or more inputs and optionally a 0-based flat index
 * (element offset).
 *
 * _Performance note_: when possible, this function loads the argument arrays and stores the output
 * array using vectorized cuda load/store instructions. The size of the vectorization depends on the
 * size of the largest input/output element type and on the alignment of all pointers.
 *
 * @tparam PassOffset whether to pass an offset as a first argument to Func
 * @tparam OutType data-type of the result (device_mdspan)
 * @tparam Func the device-lambda performing the actual operation
 * @tparam InTypes data-types of the inputs (device_mdspan)
 *
 * @param[in] res raft::resources
 * @param[out] out the output of the map operation (device_mdspan)
 * @param[in] f device lambda of type
 *                 ([auto offset], InTypes::value_type xs...) -> OutType::value_type
 * @param[in] ins the inputs (each of the same size as the output) (device_mdspan)
 */
template <bool PassOffset,
          typename OutType,
          typename Func,
          typename... InTypes,
          typename = raft::enable_if_output_device_mdspan<OutType>,
          typename = raft::enable_if_input_device_mdspan<InTypes...>>
void map(const raft::resources& res, OutType out, Func f, InTypes... ins)
{
  if (resource::get_dry_run_flag(res)) { return; }
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  (map_check_shape(out, ins), ...);

  map<PassOffset,
      typename OutType::value_type,
      typename OutType::index_type,
      Func,
      typename InTypes::value_type...>(resource::get_cuda_stream(res),
                                       out.data_handle(),
                                       static_cast<typename OutType::index_type>(out.size()),
                                       f,
                                       ins.data_handle()...);
}

}  // namespace linalg::detail
}  // namespace raft
