/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <cuda/memory_resource>

namespace raft {
namespace mr {

/**
 * @brief Type-erased owning synchronous resource accessible from both host and device.
 *
 * Mirrors the role of rmm::mr::device_memory_resource for the host+device case,
 * but uses concept-based type erasure (no inheritance required from stored resources).
 * Any type satisfying cuda::mr::synchronous_resource with host_accessible and
 * device_accessible properties can be stored.
 */
using host_device_resource =
  cuda::mr::any_synchronous_resource<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Type-erased owning synchronous resource accessible from host only.
 *
 * Mirrors the role of rmm::mr::device_memory_resource for the host-only case.
 */
using host_resource = cuda::mr::any_synchronous_resource<cuda::mr::host_accessible>;

/**
 * @brief Type-erased owning async resource accessible from device.
 *
 * Mirrors the role of rmm::mr::device_memory_resource but uses concept-based
 * type erasure.  Any type satisfying cuda::mr::resource with device_accessible
 * property can be stored.
 *
 * @deprecated Use `cuda::mr::any_device_resource` instead.
 */
using device_resource [[deprecated("Use cuda::mr::any_device_resource instead.")]] =
  cuda::mr::any_device_resource;

/**
 * @brief Alias for a `cuda::mr::synchronous_resource_ref` with the property
 * `cuda::mr::host_accessible`.
 */
using synchronous_host_resource_ref = cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible>;

/**
 * @brief Alias for a `cuda::mr::synchronous_resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 */
using synchronous_host_device_resource_ref =
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, cuda::mr::device_accessible>;

/**
 * @brief Alias for a `cuda::mr::synchronous_resource_ref` with the property
 * `cuda::mr::host_accessible`.
 *
 * @deprecated Use `raft::mr::synchronous_host_resource_ref` instead.
 */
using host_resource_ref [[deprecated("Use raft::mr::synchronous_host_resource_ref instead.")]] =
  synchronous_host_resource_ref;

/**
 * @brief Alias for a `cuda::mr::synchronous_resource_ref` with the properties
 * `cuda::mr::host_accessible` and `cuda::mr::device_accessible`.
 *
 * @deprecated Use `raft::mr::synchronous_host_device_resource_ref` instead.
 */
using host_device_resource_ref
  [[deprecated("Use raft::mr::synchronous_host_device_resource_ref instead.")]] =
    synchronous_host_device_resource_ref;

}  // namespace mr
}  // namespace raft
