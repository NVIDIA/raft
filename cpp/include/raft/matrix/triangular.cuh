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

namespace raft {
namespace matrix {

/**
 * @defgroup matrix_triangular Extract Matrix Triangles
 * @{
 */

/**
 * @brief Copy the upper triangular part of a matrix to another
 * @param[in] handle: raft handle
 * @param[in] src: input matrix with a size of n_rows x n_cols
 * @param[out] dst: output matrix with a size of kxk, k = min(n_rows, n_cols)
 */
template <typename m_t, typename idx_t>
void upper_triangular(raft::resources const& handle,
                      raft::device_matrix_view<const m_t, idx_t, col_major> src,
                      raft::device_matrix_view<m_t, idx_t, col_major> dst)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  auto k = std::min(src.extent(0), src.extent(1));
  RAFT_EXPECTS(k == dst.extent(0) && k == dst.extent(1),
               "dst should be of size kxk, k = min(n_rows, n_cols)");
  detail::copyUpperTriangular(src.data_handle(),
                              dst.data_handle(),
                              src.extent(0),
                              src.extent(1),
                              resource::get_cuda_stream(handle));
}
/** @} */  // end group matrix_triangular

}  // namespace matrix
}  // namespace raft
