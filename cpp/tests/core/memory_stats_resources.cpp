/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/memory_stats_resources.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <gtest/gtest.h>

#include <cstddef>

namespace raft {

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

}  // namespace raft
