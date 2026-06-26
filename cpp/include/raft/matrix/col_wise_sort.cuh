/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __COL_WISE_SORT_H
#define __COL_WISE_SORT_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/matrix/detail/columnWiseSort.cuh>

namespace raft {
namespace matrix {

/**
 * @brief sort columns within each row of row-major input matrix and return sorted indexes
 * modelled as key-value sort with key being input matrix and value being index of values
 * @param in: input matrix
 * @param out: output value(index) matrix
 * @param n_rows: number rows of input matrix
 * @param n_columns: number columns of input matrix
 * @param bAllocWorkspace: check returned value, if true allocate workspace passed in workspaceSize
 * @param workspacePtr: pointer to workspace memory
 * @param workspaceSize: Size of workspace to be allocated
 * @param stream: cuda stream to execute prim on
 * @param sortedKeys: Optional, output matrix for sorted keys (input)
 */
template <typename InType, typename OutType>
void sort_cols_per_row(const InType* in,
                       OutType* out,
                       int n_rows,
                       int n_columns,
                       bool& bAllocWorkspace,
                       void* workspacePtr,
                       size_t& workspaceSize,
                       cudaStream_t stream,
                       InType* sortedKeys = nullptr)
{
  detail::sortColumnsPerRow<InType, OutType>(false,
                                             in,
                                             out,
                                             n_rows,
                                             n_columns,
                                             bAllocWorkspace,
                                             workspacePtr,
                                             workspaceSize,
                                             stream,
                                             sortedKeys);
}

/**
 * @defgroup col_wise_sort Sort rows within each column
 * @{
 */

/**
 * @brief sort columns within each row of row-major input matrix and return sorted indexes
 * modelled as key-value sort with key being input matrix and value being index of values
 * @tparam in_t: element type of input matrix
 * @tparam out_t: element type of output matrix
 * @tparam matrix_idx_t: integer type for matrix indexing
 * @tparam sorted_keys_t: std::optional<raft::device_matrix_view<in_t, matrix_idx_t,
 * raft::row_major>> @c sorted_keys_opt
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output value(index) matrix
 * @param[out] sorted_keys_opt: std::optional, output matrix for sorted keys (input)
 */
template <typename in_t, typename out_t, typename matrix_idx_t, typename sorted_keys_t>
void sort_cols_per_row(raft::resources const& handle,
                       raft::device_matrix_view<const in_t, matrix_idx_t, raft::row_major> in,
                       raft::device_matrix_view<out_t, matrix_idx_t, raft::row_major> out,
                       sorted_keys_t&& sorted_keys_opt)
{
  std::optional<raft::device_matrix_view<in_t, matrix_idx_t, raft::row_major>> sorted_keys =
    std::forward<sorted_keys_t>(sorted_keys_opt);

  RAFT_EXPECTS(in.extent(1) == out.extent(1) && in.extent(0) == out.extent(0),
               "Input and output matrices must have the same shape.");

  if (sorted_keys.has_value()) {
    RAFT_EXPECTS(in.extent(1) == sorted_keys.value().extent(1) &&
                   in.extent(0) == sorted_keys.value().extent(0),
                 "Input and `sorted_keys` matrices must have the same shape.");
  }

  bool dry_run          = resource::get_dry_run_flag(handle);
  size_t workspace_size = 0;
  bool alloc_workspace  = false;

  in_t* keys = sorted_keys.has_value() ? sorted_keys.value().data_handle() : nullptr;

  detail::sortColumnsPerRow<in_t, out_t>(dry_run,
                                         in.data_handle(),
                                         out.data_handle(),
                                         in.extent(0),
                                         in.extent(1),
                                         alloc_workspace,
                                         (void*)nullptr,
                                         workspace_size,
                                         resource::get_cuda_stream(handle),
                                         keys);

  if (alloc_workspace) {
    auto workspace = raft::make_device_vector<char>(handle, workspace_size);

    if (dry_run) { return; }

    detail::sortColumnsPerRow<in_t, out_t>(dry_run,
                                           in.data_handle(),
                                           out.data_handle(),
                                           in.extent(0),
                                           in.extent(1),
                                           alloc_workspace,
                                           (void*)workspace.data_handle(),
                                           workspace_size,
                                           resource::get_cuda_stream(handle),
                                           keys);
  }
}

/**
 * @brief Overload of `sort_keys_per_row` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `sort_keys_per_row`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 3>>
void sort_cols_per_row(Args... args)
{
  sort_cols_per_row(std::forward<Args>(args)..., std::nullopt);
}

/** @} */  // end of group col_wise_sort

};  // namespace matrix
}  // namespace raft
#endif
