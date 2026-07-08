/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/memory_stats_resources.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>

namespace raft {
namespace {

struct device_resource_restore_guard {
  int device_id;
  raft::mr::device_resource resource;

  ~device_resource_restore_guard()
  {
    rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, std::move(resource));
  }
};

}  // namespace

TEST(MemoryStatsResources, IndependentCounting_DefaultWorkspace)
{
  raft::resources res;

  memory_stats_resources stat_res(res);

  constexpr std::size_t kWsSize     = 1024;
  constexpr std::size_t kGlobalSize = 2048;

  auto ws_ref  = resource::get_workspace_resource_ref(stat_res);
  void* ws_ptr = ws_ref.allocate(cuda::stream_ref{cudaStreamLegacy}, kWsSize);

  auto dev_mr   = rmm::mr::get_current_device_resource_ref();
  void* dev_ptr = dev_mr.allocate(cuda::stream_ref{cudaStreamLegacy}, kGlobalSize);

  auto peak = stat_res.get_bytes_peak();
  EXPECT_EQ(peak.device_workspace, kWsSize);
  EXPECT_EQ(peak.device_global, kGlobalSize);
  EXPECT_EQ(peak.total(), kWsSize + kGlobalSize);

  ws_ref.deallocate(cuda::stream_ref{cudaStreamLegacy}, ws_ptr, kWsSize);
  dev_mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, dev_ptr, kGlobalSize);
}

TEST(MemoryStatsResources, IndependentCounting_WorkspaceSetToGlobal)
{
  raft::resources res;
  resource::set_workspace_to_global_resource(res);

  memory_stats_resources stat_res(res);

  constexpr std::size_t kWsSize     = 1024;
  constexpr std::size_t kGlobalSize = 2048;

  auto ws_ref  = resource::get_workspace_resource_ref(stat_res);
  void* ws_ptr = ws_ref.allocate(cuda::stream_ref{cudaStreamLegacy}, kWsSize);

  auto dev_mr   = rmm::mr::get_current_device_resource_ref();
  void* dev_ptr = dev_mr.allocate(cuda::stream_ref{cudaStreamLegacy}, kGlobalSize);

  auto peak = stat_res.get_bytes_peak();
  EXPECT_EQ(peak.device_workspace, kWsSize);
  EXPECT_EQ(peak.device_global, kGlobalSize);
  EXPECT_EQ(peak.total(), kWsSize + kGlobalSize);

  ws_ref.deallocate(cuda::stream_ref{cudaStreamLegacy}, ws_ptr, kWsSize);
  dev_mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, dev_ptr, kGlobalSize);
}

TEST(MemoryStatsResources, IndependentCounting_PoolWorkspace)
{
  raft::resources res;
  constexpr std::size_t kPoolLimit = 64UL * 1024UL * 1024UL;
  resource::set_workspace_to_pool_resource(res, kPoolLimit);

  memory_stats_resources stat_res(res);

  constexpr std::size_t kWsSize     = 1024;
  constexpr std::size_t kGlobalSize = 2048;

  auto ws_ref  = resource::get_workspace_resource_ref(stat_res);
  void* ws_ptr = ws_ref.allocate(cuda::stream_ref{cudaStreamLegacy}, kWsSize);

  auto dev_mr   = rmm::mr::get_current_device_resource_ref();
  void* dev_ptr = dev_mr.allocate(cuda::stream_ref{cudaStreamLegacy}, kGlobalSize);

  auto peak = stat_res.get_bytes_peak();
  EXPECT_EQ(peak.device_workspace, kWsSize);
  EXPECT_EQ(peak.device_global, kGlobalSize);
  EXPECT_EQ(peak.total(), kWsSize + kGlobalSize);

  ws_ref.deallocate(cuda::stream_ref{cudaStreamLegacy}, ws_ptr, kWsSize);
  dev_mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, dev_ptr, kGlobalSize);
}

TEST(MemoryStatsResources, RestoresDeviceResourceOnConstructionDevice)
{
  if (device_setter::get_device_count() < 2) {
    GTEST_SKIP() << "Requires at least 2 CUDA devices";
  }

  auto device0 = 0;
  auto device1 = 1;

  auto device0_guard = [&]() {
    auto scoped_device = device_setter{device0};
    auto upstream      = rmm::mr::get_current_device_resource_ref();
    return device_resource_restore_guard{
      device0,
      rmm::mr::set_current_device_resource(
        raft::mr::device_resource{rmm::mr::pool_memory_resource(upstream, 1 << 20, 2 << 20)})};
  }();

  auto device1_guard = [&]() {
    auto scoped_device = device_setter{device1};
    return device_resource_restore_guard{device1, rmm::mr::reset_current_device_resource()};
  }();

  {
    auto scoped_device = device_setter{device0};
    raft::resources res;
    auto tracked = std::make_unique<memory_stats_resources>(res);
    auto wrong_device = device_setter{device1};
    static_cast<void>(wrong_device);
    tracked.reset();
  }

  {
    auto scoped_device = device_setter{device0};
    auto current_mr    = rmm::mr::get_current_device_resource_ref();
    EXPECT_NE(cuda::mr::resource_cast<rmm::mr::pool_memory_resource>(&current_mr), nullptr);
  }

  {
    auto scoped_device = device_setter{device1};
    auto current_mr    = rmm::mr::get_current_device_resource_ref();
    EXPECT_NE(cuda::mr::resource_cast<rmm::mr::cuda_memory_resource>(&current_mr), nullptr);
  }
}

}  // namespace raft
