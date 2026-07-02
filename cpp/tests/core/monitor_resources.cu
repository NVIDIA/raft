/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/memory_tracking_resources.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>

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

TEST(MemoryTrackingResources, SamplingSingleThread)
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

TEST(MemoryTrackingResources, RecordingSingleThread)
{
  std::ostringstream oss;
  {
    raft::resources res;
    raft::memory_tracking_resources tracked(res, oss);
    {
      nvtx::range r{"1. expect 10 KB"};
      auto matrix = raft::make_host_vector<uint8_t>(tracked, 10 * 1024);
    }
    {
      nvtx::range r{"2. expect 100 MiB"};
      auto vector = raft::make_host_vector<uint8_t>(tracked, 100 * MiB);
    }
    {
      nvtx::range r{"3. expect 4 MiB"};
      auto matrix = raft::make_host_vector<uint8_t>(tracked, 4 * MiB);
    }
  }  // tracked destroyed here: stops the sampler and flushes the file

  auto output = oss.str();
  EXPECT_NE(output.find("timestamp_us"), std::string::npos);
  EXPECT_NE(output.find("host_current"), std::string::npos);
  EXPECT_NE(output.find("device_current"), std::string::npos);
  EXPECT_NE(output.find("workspace_current"), std::string::npos);
  EXPECT_NE(output.find("event_source"), std::string::npos);
  EXPECT_NE(output.find("event_bytes"), std::string::npos);
  EXPECT_NE(output.find("alloc_range"), std::string::npos);

  auto num_lines                      = std::count(output.begin(), output.end(), '\n');
  constexpr size_t NUM_ALLOCS         = 3;
  constexpr size_t NUM_DEALLOCS       = NUM_ALLOCS;
  constexpr size_t NUM_HEADER_LINES   = 1;
  constexpr size_t NUM_LINES_EXPECTED = NUM_ALLOCS + NUM_DEALLOCS + NUM_HEADER_LINES;
  EXPECT_GE(num_lines, NUM_LINES_EXPECTED)
    << "Expected at least " << NUM_LINES_EXPECTED
    << " data records (allocation + deallocation + header); got " << num_lines << " lines"
    << std::endl
    << "content: " << std::endl
    << output;
}

// ---------------------------------------------------------------------------
// Parallel-thread stress tests
// ---------------------------------------------------------------------------

TEST(MemoryTrackingResources, RecordingParallelThreads)
{
  constexpr int kNumThreads        = 64;
  constexpr int kNumIters          = 200;
  constexpr std::size_t kAllocSize = 256 * 1024;  // 256 KiB

  std::ostringstream oss;
  {
    raft::resources res;
    raft::memory_tracking_resources tracked(res, oss);

    // Lambda captures tracked directly — no base-class cast needed.
    // Each thread tags its allocations with a distinct NVTX range so the
    // CSV output carries per-thread attribution.
    auto run = [&](int t) {
      for (int i = 0; i < kNumIters; ++i) {
        std::string label = std::string("thread-") + std::to_string(t);
        nvtx::range r{label.c_str()};
        auto vec = raft::make_host_vector<uint8_t>(tracked, kAllocSize);
      }
    };

    std::vector<std::thread> workers;
    for (int t = 0; t < kNumThreads; ++t) {
      workers.emplace_back(run, t);
    }
    for (auto& w : workers)
      w.join();
  }  // tracked destroyed: drains queue, flushes CSV

  std::string output = oss.str();
  auto num_lines     = std::count(output.begin(), output.end(), '\n');

  // Each iteration: one alloc row + one dealloc row; plus one header line.
  const int expected_rows = kNumThreads * kNumIters * 2 + 1;
  EXPECT_GE(num_lines, expected_rows)
    << "Expected at least " << expected_rows << " lines; got " << num_lines << "\noutput:\n"
    << output;
}

TEST(MemoryTrackingResources, SamplingParallelThreads)
{
  constexpr int kNumThreads        = 64;
  constexpr int kNumIters          = 200;
  constexpr std::size_t kAllocSize = 256 * 1024;  // 256 KiB

  std::ostringstream oss;
  {
    raft::resources res;
    raft::memory_tracking_resources tracked(res, oss, 1us);

    auto run = [&](int t) {
      for (int i = 0; i < kNumIters; ++i) {
        std::string label = std::string("thread-") + std::to_string(t);
        nvtx::range r{label.c_str()};
        auto vec = raft::make_host_vector<uint8_t>(tracked, kAllocSize);
      }
    };

    std::vector<std::thread> workers;
    for (int t = 0; t < kNumThreads; ++t) {
      workers.emplace_back(run, t);
    }
    for (auto& w : workers)
      w.join();
  }

  std::string output = oss.str();
  auto num_lines     = std::count(output.begin(), output.end(), '\n');

  // Sampling approach drops many rows
  const int max_num_rows = kNumThreads * kNumIters * 2 + 1;
  EXPECT_TRUE(3 < num_lines && num_lines < max_num_rows)
    << "Expected at least several rows. It is expected when many rows are dropped; got "
    << num_lines << "\noutput:\n"
    << output;
}

}  // namespace
