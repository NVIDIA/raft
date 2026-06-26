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
 * @defgroup matrix_copy Matrix copy operations
 * @{
 */

/**
 * @brief Copy selected rows of the input matrix into contiguous space.
 *
 * On exit out[i + k*n_rows] = in[indices[i] + k*n_rows],
 * where i = 0..n_rows_indices-1, and k = 0..n_cols-1.
 *
 * @param[in] handle raft handle
 * @param[in] in input matrix
 * @param[out] out output matrix
 * @param[in] indices of the rows to be copied
 */
template <typename m_t, typename idx_t, typename layout>
void copy_rows(raft::resources const& handle,
               raft::device_matrix_view<const m_t, idx_t, layout> in,
               raft::device_matrix_view<m_t, idx_t, layout> out,
               raft::device_vector_view<const idx_t, idx_t> indices)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(in.extent(1) == out.extent(1),
               "Input and output matrices must have same number of columns");
  RAFT_EXPECTS(indices.extent(0) == out.extent(0),
               "Number of rows in output matrix must equal number of indices");
  detail::copyRows(in.data_handle(),
                   in.extent(0),
                   in.extent(1),
                   out.data_handle(),
                   indices.data_handle(),
                   indices.extent(0),
                   resource::get_cuda_stream(handle),
                   raft::is_row_major(in));
}

/**
 * @brief copy matrix operation for row major matrices.
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix
 */
template <typename m_t, typename matrix_idx_t>
void copy(raft::resources const& handle,
          raft::device_matrix_view<const m_t, matrix_idx_t, row_major> in,
          raft::device_matrix_view<m_t, matrix_idx_t, row_major> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(in.extent(0) == out.extent(0) && in.extent(1) == out.extent(1),
               "Input and output matrix shapes must match.");

  raft::copy_async(out.data_handle(),
                   in.data_handle(),
                   in.extent(0) * out.extent(1),
                   resource::get_cuda_stream(handle));
}

/**
 * @brief copy matrix operation for column major matrices.
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix
 */
template <typename m_t, typename matrix_idx_t>
void copy(raft::resources const& handle,
          raft::device_matrix_view<const m_t, matrix_idx_t, col_major> in,
          raft::device_matrix_view<m_t, matrix_idx_t, col_major> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(in.extent(0) == out.extent(0) && in.extent(1) == out.extent(1),
               "Input and output matrix shapes must match.");

  raft::copy_async(out.data_handle(),
                   in.data_handle(),
                   in.extent(0) * out.extent(1),
                   resource::get_cuda_stream(handle));
}

/**
 * @brief copy matrix operation for column major matrices. First n_rows and
 * n_cols of input matrix "in" is copied to "out" matrix.
 * @param handle: raft handle for managing resources
 * @param in: input matrix
 * @param out: output matrix
 */
template <typename m_t, typename idx_t>
void trunc_zero_origin(raft::resources const& handle,
                       raft::device_matrix_view<const m_t, idx_t, col_major> in,
                       raft::device_matrix_view<m_t, idx_t, col_major> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(out.extent(0) <= in.extent(0) && out.extent(1) <= in.extent(1),
               "Output matrix must have less or equal number of rows and columns");

  detail::truncZeroOrigin<m_t, idx_t>(in.data_handle(),
                                      in.extent(0),
                                      out.data_handle(),
                                      out.extent(0),
                                      out.extent(1),
                                      resource::get_cuda_stream(handle));
}

/** @} */  // end of group matrix_copy

}  // namespace matrix
}  // namespace raft
