/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/memory_tracking_resources.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {

namespace nvtx = raft::common::nvtx;
using namespace std::chrono_literals;
constexpr std::size_t MiB = std::size_t{1024} * 1024;

TEST(MemoryTrackingResources, TracksDeviceAllocations)
{
  std::ostringstream oss;
  {
    raft::resources res;
    raft::resource::set_workspace_to_pool_resource(res, 1 * MiB);

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

TEST(MemoryTrackingResources, MismatchedRangeLabeling)
{
  const std::string csv_path = "mismatch_range_label.csv";

  {
    raft::resources res;

    raft::memory_tracking_resources tracked(res, csv_path, 1ms);
    {
      nvtx::range r{"1. expect 10 KB"};
      auto matrix = raft::make_host_vector<uint8_t>(tracked, 10 * 1024);
    }
    {
      // Deliberately huge & slow: allocating/freeing 10 GiB of host memory takes
      // several ms, which makes the background sampler lag past this range's end.
      // As a result this allocation's peak is mis-attributed to the NEXT range in
      // the CSV (the range-labeling race discussed in the file header). Source
      // attribution (host) stays correct; only the nvtx_range label is wrong.
      nvtx::range r{"2. expect 10 GiB"};
      auto vector = raft::make_host_vector<uint8_t>(tracked, 10 * 1024 * MiB);
    }
    {
      nvtx::range r{"3. expect 4 MiB"};
      auto matrix = raft::make_host_vector<uint8_t>(tracked, 4 * MiB);
    }
  }  // tracked destroyed here: stops the sampler and flushes the file

  std::cout << "Wrote allocation statistics to " << csv_path << "\n";
}

}  // namespace
