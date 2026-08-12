/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/device_setter.hpp>
#include <raft/mr/host_device_resource.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>

#include <gtest/gtest.h>

#include <exception>
#include <utility>

namespace raft::test {

struct device_resource_restore_guard {
  int device_id;
  raft::mr::device_resource resource;

  ~device_resource_restore_guard()
  {
    try {
      rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, std::move(resource));
    } catch (const std::exception& e) {
      ADD_FAILURE() << "Failed to restore device " << device_id << " memory resource: " << e.what();
    } catch (...) {
      ADD_FAILURE() << "Failed to restore device " << device_id
                    << " memory resource: unknown exception";
    }
  }
};

inline auto install_pool_device_resource(int device_id) -> device_resource_restore_guard
{
  auto scoped_device = raft::device_setter{device_id};
  auto upstream      = rmm::mr::get_current_device_resource_ref();
  auto installed_resource =
    raft::mr::device_resource{rmm::mr::pool_memory_resource(upstream, 1 << 20, 2 << 20)};
  auto old_resource = rmm::mr::set_current_device_resource(std::move(installed_resource));
  return device_resource_restore_guard{device_id, std::move(old_resource)};
}

inline auto install_default_device_resource(int device_id) -> device_resource_restore_guard
{
  auto scoped_device = raft::device_setter{device_id};
  return device_resource_restore_guard{device_id, rmm::mr::reset_current_device_resource()};
}

inline auto current_device_uses_pool_resource() -> bool
{
  auto current_mr = rmm::mr::get_current_device_resource_ref();
  return cuda::mr::resource_cast<rmm::mr::pool_memory_resource>(&current_mr) != nullptr;
}

inline auto current_device_uses_default_cuda_resource() -> bool
{
  auto current_mr = rmm::mr::get_current_device_resource_ref();
  return cuda::mr::resource_cast<rmm::mr::cuda_memory_resource>(&current_mr) != nullptr;
}

}  // namespace raft::test
