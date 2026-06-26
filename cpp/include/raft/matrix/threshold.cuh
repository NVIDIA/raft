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
 * @defgroup matrix_threshold Matrix thesholding
 * @{
 */

/**
 * @brief sets the small values to zero based on a defined threshold
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix. The result is stored in the out matrix
 * @param[in] thres threshold to set values to zero
 */
template <typename math_t, typename idx_t, typename layout>
void zero_small_values(raft::resources const& handle,
                       raft::device_matrix_view<const math_t, idx_t, layout> in,
                       raft::device_matrix_view<math_t, idx_t, layout> out,
                       math_t thres = 1e-15)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(in.size() == out.size(), "Input and output matrices must have same size");
  detail::setSmallValuesZero(
    out.data_handle(), in.data_handle(), in.size(), resource::get_cuda_stream(handle), thres);
}

/**
 * @brief sets the small values to zero in-place based on a defined threshold
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @param handle: raft handle
 * @param inout: input matrix and also the result is stored
 * @param thres: threshold
 */
template <typename math_t, typename idx_t, typename layout>
void zero_small_values(raft::resources const& handle,
                       raft::device_matrix_view<math_t, idx_t, layout> inout,
                       math_t thres = 1e-15)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::setSmallValuesZero(
    inout.data_handle(), inout.size(), resource::get_cuda_stream(handle), thres);
}

/** @} */  // end group matrix_threshold

}  // namespace matrix
}  // namespace raft
