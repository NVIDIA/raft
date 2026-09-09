/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/logger.hpp>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime_api.h>

#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <mutex>
#include <set>

namespace raft {
auto get_test_device_ids()
{
  auto devices      = std::array<int, 2>{int{}, int{}};
  auto device_count = 0;
  RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
  devices[1] = int{device_count > 1};
  return devices;
}

TEST(DeviceResourcesManager, ObeysSetters)
{
  auto devices = get_test_device_ids();

  auto streams_per_device = 3;
  auto pools_per_device   = 3;
  auto streams_per_pool   = 7;
  auto workspace_limit    = 2048;
  auto workspace_init     = 1024;
  device_resources_manager::set_streams_per_device(streams_per_device);
  device_resources_manager::set_stream_pools_per_device(pools_per_device, streams_per_pool);
  device_resources_manager::set_mem_pool();
  device_resources_manager::set_workspace_allocation_limit(workspace_limit);

  auto unique_streams = std::array<std::set<cudaStream_t>, 2>{};
  auto unique_pools   = std::array<std::set<rmm::cuda_stream_pool const*>, 2>{};

  // Provide lock for counting unique objects
  auto mtx         = std::mutex{};
  auto default_mrs = std::array<bool, 2>{};

  for (auto i = std::size_t{}; i < devices.size(); ++i) {
    auto scoped_device = device_setter{devices[i]};
    auto upstream      = rmm::mr::get_current_device_resource_ref();
    default_mrs[i] = cuda::mr::resource_cast<rmm::mr::cuda_memory_resource>(&upstream) != nullptr;
    device_resources_manager::set_workspace_memory_resource(
      raft::mr::device_resource{
        rmm::mr::pool_memory_resource(upstream, workspace_init, workspace_limit)},
      devices[i]);
  }

  // Suppress the many warnings from testing use of setters after initial
  // get_device_resources call
  auto scoped_log_level =
    rapids_logger::log_level_setter{default_logger(), rapids_logger::level_enum::error};

  omp_set_dynamic(0);
#pragma omp parallel for num_threads(5)
  for (auto i = std::size_t{}; i < 101; ++i) {
    thread_local auto prev_streams = std::array<std::optional<cudaStream_t>, 2>{};
    auto device                    = devices[i % devices.size()];
    auto const& res                = device_resources_manager::get_device_resources(device);

    auto primary_stream  = res.get_stream().get();
    prev_streams[device] = prev_streams[device].value_or(primary_stream);
    // Expect to receive the same stream every time for a given thread
    EXPECT_EQ(*prev_streams[device], primary_stream);

    // Using RAII device setter here to avoid changing device in other tests
    // that depend on a specific device to be set
    auto scoped_device = device_setter{device};
    auto const& res2   = device_resources_manager::get_device_resources();
    // Expect device_resources to default to current device
    EXPECT_EQ(primary_stream, res2.get_stream().get());

    auto const& pool = res.get_stream_pool();
    EXPECT_EQ(streams_per_pool, pool.get_pool_size());

    auto current_mr = rmm::mr::get_current_device_resource_ref();
    auto* mr        = cuda::mr::resource_cast<rmm::mr::pool_memory_resource>(&current_mr);
    if (default_mrs[i % devices.size()]) { EXPECT_NE(mr, nullptr); }

    {
      auto lock = std::unique_lock{mtx};
      unique_streams[device].insert(primary_stream);
      unique_pools[device].insert(&pool);
    }
    // Ensure that setters have no effect after get_device_resources call
    device_resources_manager::set_streams_per_device(streams_per_device + 1);
    device_resources_manager::set_stream_pools_per_device(pools_per_device - 1);
    device_resources_manager::set_mem_pool();
    device_resources_manager::set_workspace_allocation_limit(1024);
    device_resources_manager::set_workspace_memory_resource(
      raft::mr::device_resource{rmm::mr::get_current_device_resource_ref()},
      devices[i % devices.size()]);
  }

  EXPECT_EQ(streams_per_device, unique_streams[devices[0]].size());
  EXPECT_EQ(streams_per_device, unique_streams[devices[1]].size());
  EXPECT_EQ(pools_per_device, unique_pools[devices[0]].size());
  EXPECT_EQ(pools_per_device, unique_pools[devices[1]].size());
}

}  // namespace raft
