/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/matrix/detail/matrix.cuh>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace matrix {

/**
 * @defgroup matrix_reverse Matrix reverse
 * @{
 */

/**
 * @brief Reverse the columns of a matrix in place (i.e. first column and
 * last column are swapped)
 * @param[in] handle: raft handle
 * @param[inout] inout: input and output matrix
 */
template <typename m_t, typename idx_t, typename layout_t>
void col_reverse(raft::resources const& handle,
                 raft::device_matrix_view<m_t, idx_t, layout_t> inout)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(raft::is_row_or_column_major(inout), "Unsupported matrix layout");
  if (raft::is_col_major(inout)) {
    detail::colReverse(
      inout.data_handle(), inout.extent(0), inout.extent(1), resource::get_cuda_stream(handle));
  } else {
    detail::rowReverse(
      inout.data_handle(), inout.extent(1), inout.extent(0), resource::get_cuda_stream(handle));
  }
}

/**
 * @brief Reverse the rows of a matrix in place (i.e. first row and last
 * row are swapped)
 * @param[in] handle: raft handle
 * @param[inout] inout: input and output matrix
 */
template <typename m_t, typename idx_t, typename layout_t>
void row_reverse(raft::resources const& handle,
                 raft::device_matrix_view<m_t, idx_t, layout_t> inout)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(raft::is_row_or_column_major(inout), "Unsupported matrix layout");
  if (raft::is_col_major(inout)) {
    detail::rowReverse(
      inout.data_handle(), inout.extent(0), inout.extent(1), resource::get_cuda_stream(handle));
  } else {
    detail::colReverse(
      inout.data_handle(), inout.extent(1), inout.extent(0), resource::get_cuda_stream(handle));
  }
}
/** @} */  // end group matrix_reverse

}  // namespace matrix
}  // namespace raft
