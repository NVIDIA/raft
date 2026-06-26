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
#include <raft/matrix/detail/matrix.cuh>
#include <raft/matrix/matrix_types.hpp>

namespace raft {
namespace matrix {

/**
 * @defgroup matrix_print Matrix print operations
 * @{
 */

/**
 * @brief Prints the data stored in GPU memory
 * @tparam m_t type of matrix elements
 * @tparam idx_t integer type used for indexing
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[in] separators: horizontal and vertical separator characters
 */
template <typename m_t, typename idx_t>
void print(raft::resources const& handle,
           raft::device_matrix_view<const m_t, idx_t, col_major> in,
           print_separators& separators)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::print(in.data_handle(),
                in.extent(0),
                in.extent(1),
                separators.horizontal,
                separators.vertical,
                resource::get_cuda_stream(handle));
}

/** @} */  // end group matrix_print
}  // namespace matrix
}  // namespace raft
