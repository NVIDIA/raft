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
 * @defgroup matrix_norm Matrix Norm Operations
 * @{
 */

/**
 * @brief Get the L2/F-norm of a matrix
 * @param[in] handle: raft handle
 * @param[in] in: input matrix/vector with totally size elements
 * @returns matrix l2 norm
 */
template <typename m_t, typename idx_t>
m_t l2_norm(raft::resources const& handle, raft::device_mdspan<const m_t, idx_t> in)
{
  if (resource::get_dry_run_flag(handle)) { return {}; }
  return detail::getL2Norm(handle, in.data_handle(), in.size(), resource::get_cuda_stream(handle));
}

/** @} */  // end of group matrix_norm

}  // namespace matrix
}  // namespace raft
