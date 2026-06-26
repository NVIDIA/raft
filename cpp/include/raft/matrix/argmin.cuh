/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/matrix/detail/math.cuh>

namespace raft {
namespace matrix {

/**
 * @defgroup argmin Argmin operation
 * @{
 */

/**
 * @brief Argmin: find the col idx with minimum value for each row
 * @param[in] handle: raft handle
 * @param[in] in: input matrix of size (n_rows, n_cols)
 * @param[out] out: output vector of size n_rows
 */
template <typename math_t, typename idx_t, typename matrix_idx_t>
void argmin(raft::resources const& handle,
            raft::device_matrix_view<const math_t, matrix_idx_t, row_major> in,
            raft::device_vector_view<idx_t, matrix_idx_t> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(out.extent(0) == in.extent(0),
               "Size of output vector must equal number of rows in input matrix.");
  detail::argmin(in.data_handle(),
                 in.extent(1),
                 in.extent(0),
                 out.data_handle(),
                 resource::get_cuda_stream(handle));
}

/** @} */  // end of group argmin

}  // namespace matrix
}  // namespace raft
