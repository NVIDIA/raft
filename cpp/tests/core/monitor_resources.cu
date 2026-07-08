/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/memory_tracking_resources.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>
#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

namespace {

struct device_resource_restore_guard {
  int device_id;
  raft::mr::device_resource resource;

  ~device_resource_restore_guard()
  {
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, std::move(resource));
  }
};

TEST(MemoryTrackingResources, TracksDeviceAllocations)
{
  using namespace std::chrono_literals;

  std::ostringstream oss;
  {
    raft::resources res;
    raft::resource::set_workspace_to_pool_resource(res, 1024 * 1024);

    raft::memory_tracking_resources tracked(res, oss, 1ms);

    auto buf = raft::make_device_mdarray<float>(tracked, raft::make_extents<int>(256));
    raft::resource::sync_stream(tracked);

    std::this_thread::sleep_for(50ms);
  }

  auto output = oss.str();

  EXPECT_NE(output.find("timestamp_us"), std::string::npos);
  EXPECT_NE(output.find("host_current"), std::string::npos);
  EXPECT_NE(output.find("device_current"), std::string::npos);
  EXPECT_NE(output.find("workspace_current"), std::string::npos);

  auto num_lines = std::count(output.begin(), output.end(), '\n');
  EXPECT_GE(num_lines, 3) << "Expected at least 2 data records (allocation + deallocation) "
                             "plus 1 header line; got "
                          << num_lines << " lines" << std::endl
                          << "content: " << std::endl
                          << output;
}

TEST(MemoryTrackingResources, RestoresDeviceResourceOnConstructionDevice)
{
  if (raft::device_setter::get_device_count() < 2) {
    GTEST_SKIP() << "Requires at least 2 CUDA devices";
  }

  auto device0 = 0;
  auto device1 = 1;

  auto device0_guard = [&]() {
    auto scoped_device = raft::device_setter{device0};
    auto upstream      = rmm::mr::get_current_device_resource_ref();
    return device_resource_restore_guard{
      device0,
      rmm::mr::set_current_device_resource(
        raft::mr::device_resource{rmm::mr::pool_memory_resource(upstream, 1 << 20, 2 << 20)})};
  }();

  auto device1_guard = [&]() {
    auto scoped_device = raft::device_setter{device1};
    return device_resource_restore_guard{device1, rmm::mr::reset_current_device_resource()};
  }();

  {
    auto scoped_device = raft::device_setter{device0};
    std::ostringstream oss;
    auto tracked = std::make_unique<raft::memory_tracking_resources>(oss);
    auto wrong_device = raft::device_setter{device1};
    static_cast<void>(wrong_device);
    tracked.reset();
  }

  {
    auto scoped_device = raft::device_setter{device0};
    auto current_mr    = rmm::mr::get_current_device_resource_ref();
    EXPECT_NE(cuda::mr::resource_cast<rmm::mr::pool_memory_resource>(&current_mr), nullptr);
  }

  {
    auto scoped_device = raft::device_setter{device1};
    auto current_mr    = rmm::mr::get_current_device_resource_ref();
    EXPECT_NE(cuda::mr::resource_cast<rmm::mr::cuda_memory_resource>(&current_mr), nullptr);
  }
}

}  // namespace
