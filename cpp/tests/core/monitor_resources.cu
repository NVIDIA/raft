/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/memory_tracking_resources.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <sstream>
#include <string>
#include <thread>

namespace {

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

}  // namespace
