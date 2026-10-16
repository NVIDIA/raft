/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __MVG_H
#define __MVG_H

#pragma once

#include "detail/multi_variable_gaussian.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/random/random_types.hpp>

#include <rmm/mr/per_device_resource.hpp>

namespace raft {
namespace random {

/**
 * \defgroup multi_variable_gaussian Compute multi-variable Gaussian
 * @{
 */

template <typename ValueType>
void multi_variable_gaussian(raft::resources const& handle,
                             cuda::mr::device_resource_ref mem_resource,
                             std::optional<raft::device_vector_view<const ValueType, int>> x,
                             raft::device_matrix_view<ValueType, int, raft::col_major> P,
                             raft::device_matrix_view<ValueType, int, raft::col_major> X,
                             const multi_variable_gaussian_decomposition_method method)
{
  detail::compute_multi_variable_gaussian_impl(handle, mem_resource, x, P, X, method);
}

template <typename ValueType>
void multi_variable_gaussian(raft::resources const& handle,
                             std::optional<raft::device_vector_view<const ValueType, int>> x,
                             raft::device_matrix_view<ValueType, int, raft::col_major> P,
                             raft::device_matrix_view<ValueType, int, raft::col_major> X,
                             const multi_variable_gaussian_decomposition_method method)
{
  detail::compute_multi_variable_gaussian_impl(
    handle, rmm::mr::get_current_device_resource_ref(), x, P, X, method);
}

/** @} */

};  // namespace random
}  // namespace raft
#endif
