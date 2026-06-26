/*

 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "select_radix.cuh"
#include "select_warpsort.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/select_k_types.hpp>

#include <cub/device/device_segmented_radix_sort.cuh>

namespace raft {
namespace matrix::detail {

/**
 * Predict the fastest select_k algorithm based on the number of rows/cols/k
 *
 * The body of this method is automatically generated, using a DecisionTree
 * to predict the fastest algorithm based off of thousands of trial runs
 * on different values of rows/cols/k. The decision tree is converted to c++
 * code, which is cut and paste below.
 *
 * NOTE: The code to generate is in cpp/scripts/heuristics/select_k, running the
 * 'generate_heuristic' notebook there will replace the body of this function
 * with the latest learned heuristic
 */
inline SelectAlgo choose_select_k_algorithm(size_t rows, size_t cols, int k)
{
  if (k > 256) {
    if (cols > 16862) {
      if (rows > 1020) {
        return SelectAlgo::kRadix11bitsExtraPass;
      } else {
        return SelectAlgo::kRadix11bits;
      }
    } else {
      return SelectAlgo::kRadix11bitsExtraPass;
    }
  } else {
    if (k > 2) {
      if (cols > 22061) {
        return SelectAlgo::kWarpDistributedShm;
      } else {
        if (rows > 198) {
          return SelectAlgo::kWarpDistributedShm;
        } else {
          return SelectAlgo::kWarpImmediate;
        }
      }
    } else {
      return SelectAlgo::kWarpImmediate;
    }
  }
}

/**
 * Performs a segmented sorting of a keys array with respect to
 * the segments of a values array.
 * @tparam KeyT
 * @tparam ValT
 * @param handle
 * @param values
 * @param keys
 * @param n_segments
 * @param k
 * @param select_min
 */
template <typename KeyT, typename ValT>
void segmented_sort_by_key(raft::resources const& handle,
                           KeyT* keys,
                           ValT* values,
                           size_t n_segments,
                           size_t n_elements,
                           const ValT* offsets,
                           bool asc)
{
  auto stream = resource::get_cuda_stream(handle);
  auto mr     = resource::get_workspace_resource_ref(handle);
  auto out_inds =
    raft::make_device_mdarray<ValT, ValT>(handle, mr, raft::make_extents<ValT>(n_elements));
  auto out_dists =
    raft::make_device_mdarray<KeyT, ValT>(handle, mr, raft::make_extents<ValT>(n_elements));

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  if (asc) {
    cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                             temp_storage_bytes,
                                             keys,
                                             out_dists.data_handle(),
                                             values,
                                             out_inds.data_handle(),
                                             n_elements,
                                             n_segments,
                                             offsets,
                                             offsets + 1,
                                             0,
                                             sizeof(ValT) * 8,
                                             stream);
  } else {
    cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
                                                       temp_storage_bytes,
                                                       keys,
                                                       out_dists.data_handle(),
                                                       values,
                                                       out_inds.data_handle(),
                                                       n_elements,
                                                       n_segments,
                                                       offsets,
                                                       offsets + 1,
                                                       0,
                                                       sizeof(ValT) * 8,
                                                       stream);
  }

  auto d_temp_storage = raft::make_device_mdarray<char, size_t>(
    handle, mr, raft::make_extents<size_t>(temp_storage_bytes));

  if (resource::get_dry_run_flag(handle)) { return; }

  if (asc) {
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortPairs((void*)d_temp_storage.data_handle(),
                                             temp_storage_bytes,
                                             keys,
                                             out_dists.data_handle(),
                                             values,
                                             out_inds.data_handle(),
                                             n_elements,
                                             n_segments,
                                             offsets,
                                             offsets + 1,
                                             0,
                                             sizeof(ValT) * 8,
                                             stream);

  } else {
    // Run sorting operation
    cub::DeviceSegmentedRadixSort::SortPairsDescending((void*)d_temp_storage.data_handle(),
                                                       temp_storage_bytes,
                                                       keys,
                                                       out_dists.data_handle(),
                                                       values,
                                                       out_inds.data_handle(),
                                                       n_elements,
                                                       n_segments,
                                                       offsets,
                                                       offsets + 1,
                                                       0,
                                                       sizeof(ValT) * 8,
                                                       stream);
  }

  raft::copy(values, out_inds.data_handle(), out_inds.size(), stream);
  raft::copy(keys, out_dists.data_handle(), out_dists.size(), stream);
}

template <typename KeyT, typename ValT>
void segmented_sort_by_key(raft::resources const& handle,
                           raft::device_vector_view<const ValT, ValT> offsets,
                           raft::device_vector_view<KeyT, ValT> keys,
                           raft::device_vector_view<ValT, ValT> values,
                           bool asc)
{
  RAFT_EXPECTS(keys.size() == values.size(),
               "Keys and values must contain the same number of elements.");
  segmented_sort_by_key<KeyT, ValT>(handle,
                                    keys.data_handle(),
                                    values.data_handle(),
                                    offsets.size() - 1,
                                    keys.size(),
                                    offsets.data_handle(),
                                    asc);
}

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_val` as a row-major matrix with `len` columns and
 * `batch_size` rows, then this function selects `k` smallest/largest values in each row and fills
 * in the row-major matrix `out_val` of size (batch_size, k).
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 *
 * @param[in] handle container of reusable resources
 * @param[in] in_val
 *   contiguous device array of inputs of size (len * batch_size);
 *   these are compared and selected.
 * @param[in] in_idx
 *   contiguous device array of inputs of size (len * batch_size);
 *   typically, these are indices of the corresponding in_val.
 * @param batch_size
 *   number of input rows, i.e. the batch size.
 * @param len
 *   length of a single input array (row); also sometimes referred as n_cols.
 *   Invariant: len >= k.
 * @param k
 *   the number of outputs to select in each input row.
 * @param[out] out_val
 *   contiguous device array of outputs of size (k * batch_size);
 *   the k smallest/largest values from each row of the `in_val`.
 * @param[out] out_idx
 *   contiguous device array of outputs of size (k * batch_size);
 *   the payload selected together with `out_val`.
 * @param select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param[in] sorted
 *   whether to make sure selected pairs are sorted by value
 * @param[in] algo
 *   the selection algorithm to use
 * @param[in] len_i
 *   array of size (batch_size) providing lengths for each individual row
 *   only radix select-k supported
 */
template <typename T, typename IdxT>
void select_k(raft::resources const& handle,
              const T* in_val,
              const IdxT* in_idx,
              size_t batch_size,
              size_t len,
              int k,
              T* out_val,
              IdxT* out_idx,
              bool select_min,
              bool sorted       = false,
              SelectAlgo algo   = SelectAlgo::kAuto,
              const IdxT* len_i = nullptr)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "matrix::select_k(batch_size = %zu, len = %zu, k = %d)", batch_size, len, k);

  if (algo == SelectAlgo::kAuto) { algo = choose_select_k_algorithm(batch_size, len, k); }

  using row_layout = select::dense_layout<IdxT>;

  switch (algo) {
    case SelectAlgo::kRadix8bits:
    case SelectAlgo::kRadix11bits:
    case SelectAlgo::kRadix11bitsExtraPass: {
      if (algo == SelectAlgo::kRadix8bits) {
        detail::select::radix::select_k<T, IdxT, 8, 512, row_layout>(handle,
                                                                     in_val,
                                                                     in_idx,
                                                                     batch_size,
                                                                     len,
                                                                     k,
                                                                     out_val,
                                                                     out_idx,
                                                                     select_min,
                                                                     true,  // fused_last_filter
                                                                     len_i);
      } else {
        bool fused_last_filter = algo == SelectAlgo::kRadix11bits;
        detail::select::radix::select_k<T, IdxT, 11, 512, row_layout>(handle,
                                                                      in_val,
                                                                      in_idx,
                                                                      batch_size,
                                                                      len,
                                                                      k,
                                                                      out_val,
                                                                      out_idx,
                                                                      select_min,
                                                                      fused_last_filter,
                                                                      len_i);
      }
      if (sorted) {
        auto offsets = make_device_mdarray<IdxT, IdxT>(
          handle, resource::get_workspace_resource_ref(handle), make_extents<IdxT>(batch_size + 1));
        raft::linalg::map_offset(handle, offsets.view(), mul_const_op<IdxT>(k));

        auto keys = raft::make_device_vector_view<T, IdxT>(out_val, (IdxT)(batch_size * k));
        auto vals = raft::make_device_vector_view<IdxT, IdxT>(out_idx, (IdxT)(batch_size * k));

        segmented_sort_by_key<T, IdxT>(
          handle, raft::make_const_mdspan(offsets.view()), keys, vals, select_min);
      }
      return;
    }
    case SelectAlgo::kWarpDistributed:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_distributed, row_layout>(
          handle, in_val, in_idx, batch_size, len, k, out_val, out_idx, select_min);
    case SelectAlgo::kWarpDistributedShm:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_distributed_ext, row_layout>(
          handle, in_val, in_idx, batch_size, len, k, out_val, out_idx, select_min);
    case SelectAlgo::kWarpAuto:
      return detail::select::warpsort::select_k<T, IdxT, row_layout>(
        handle, in_val, in_idx, batch_size, len, k, out_val, out_idx, select_min);
    case SelectAlgo::kWarpImmediate:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_immediate, row_layout>(
          handle, in_val, in_idx, batch_size, len, k, out_val, out_idx, select_min);
    case SelectAlgo::kWarpFiltered:
      return detail::select::warpsort::
        select_k_impl<T, IdxT, detail::select::warpsort::warp_sort_filtered, row_layout>(
          handle, in_val, in_idx, batch_size, len, k, out_val, out_idx, select_min);
    default: RAFT_FAIL("K-selection Algorithm not supported.");
  }
}
}  // namespace matrix::detail
}  // namespace raft
