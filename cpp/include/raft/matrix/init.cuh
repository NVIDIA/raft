/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/linalg/map.cuh>
#include <raft/matrix/detail/math.cuh>

namespace raft {
namespace matrix {

/**
 * @defgroup matrix_init Matrix initialization operations
 * @{
 */

/**
 * @brief set values to scalar in matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam extents dimension and indexing type used for the input
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[in] in input matrix
 * @param[out] out output matrix. The result is stored in the out matrix
 * @param[in] scalar scalar value to fill matrix elements
 */
template <typename math_t, typename extents, typename layout>
void fill(raft::resources const& handle,
          raft::device_mdspan<const math_t, extents, layout> in,
          raft::device_mdspan<math_t, extents, layout> out,
          raft::host_scalar_view<math_t> scalar)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Data layout not supported");
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must be the same size.");
  RAFT_EXPECTS(scalar.data_handle() != nullptr, "Empty scalar");
  detail::setValue(out.data_handle(),
                   in.data_handle(),
                   *(scalar.data_handle()),
                   in.size(),
                   resource::get_cuda_stream(handle));
}

/**
 * @brief set values to scalar in matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam extents dimension and indexing type used for the input
 * @tparam layout_t layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[inout] inout input matrix
 * @param[in] scalar scalar value to fill matrix elements
 */
template <typename math_t, typename extents, typename layout>
void fill(raft::resources const& handle,
          raft::device_mdspan<math_t, extents, layout> inout,
          math_t scalar)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  linalg::map(handle, inout, raft::const_op{scalar});
}

/** @} */  // end of group matrix_init

}  // namespace matrix
}  // namespace raft
