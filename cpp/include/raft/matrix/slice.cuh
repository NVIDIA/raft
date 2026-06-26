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
 * @defgroup matrix_slice Matrix slicing
 * @{
 */

template <typename idx_t>
struct slice_coordinates {
  idx_t row1;  ///< row coordinate of the top-left point of the wanted area (0-based)
  idx_t col1;  ///< column coordinate of the top-left point of the wanted area (0-based)
  idx_t row2;  ///< row coordinate of the bottom-right point of the wanted area (1-based)
  idx_t col2;  ///< column coordinate of the bottom-right point of the wanted area (1-based)

  slice_coordinates(idx_t row1_, idx_t col1_, idx_t row2_, idx_t col2_)
    : row1(row1_), col1(col1_), row2(row2_), col2(col2_)
  {
  }
};

/**
 * @brief Slice a matrix (in-place)
 * @tparam m_t type of matrix elements
 * @tparam idx_t integer type used for indexing
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix
 * @param[in] coords: coordinates of the wanted slice
 * example: Slice the 2nd and 3rd columns of a 4x3 matrix: slice(handle, in, out, {0, 1, 4, 3});
 */
template <typename m_t, typename idx_t, typename layout_t>
void slice(raft::resources const& handle,
           raft::device_matrix_view<const m_t, idx_t, layout_t> in,
           raft::device_matrix_view<m_t, idx_t, layout_t> out,
           slice_coordinates<idx_t> coords)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Matrix layout must be row- or column-major");
  RAFT_EXPECTS(coords.row2 > coords.row1, "row2 must be > row1");
  RAFT_EXPECTS(coords.col2 > coords.col1, "col2 must be > col1");
  RAFT_EXPECTS(coords.row1 >= 0, "row1 must be >= 0");
  RAFT_EXPECTS(coords.row2 <= in.extent(0), "row2 must be <= number of rows in the input matrix");
  RAFT_EXPECTS(coords.col1 >= 0, "col1 must be >= 0");
  RAFT_EXPECTS(coords.col2 <= in.extent(1),
               "col2 must be <= number of columns in the input matrix");

  detail::sliceMatrix(in.data_handle(),
                      in.extent(0),
                      in.extent(1),
                      out.data_handle(),
                      coords.row1,
                      coords.col1,
                      coords.row2,
                      coords.col2,
                      raft::is_row_major(in),
                      resource::get_cuda_stream(handle));
}

/** @} */  // end group matrix_slice

}  // namespace matrix
}  // namespace raft
