/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/dry_run_resources.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/dry_run_resource.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/mr/host_memory_resource.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace raft::util {

// ===== dry_run_resource tests (device async) =====

TEST(DryRunResource, DeviceAsyncPeakTracking)
{
  auto dev_ref = rmm::mr::get_current_device_resource_ref();
  raft::mr::dry_run_resource<rmm::device_async_resource_ref> dr{dev_ref};
  auto counter = dr.get_counter();

  constexpr std::size_t kSize1 = 100UL * 1024UL * 1024UL;
  constexpr std::size_t kSize2 = 200UL * 1024UL * 1024UL;

  void* p1 = dr.allocate(cuda::stream_ref{cudaStreamLegacy}, kSize1);
  ASSERT_NE(p1, nullptr);
  EXPECT_EQ(counter->get_allocated_bytes(), kSize1);
  EXPECT_EQ(counter->get_peak_bytes(), kSize1);

  void* p2 = dr.allocate(cuda::stream_ref{cudaStreamLegacy}, kSize2);
  EXPECT_EQ(p2, p1);  // same probed pointer for all allocations
  EXPECT_EQ(counter->get_peak_bytes(), kSize1 + kSize2);

  dr.deallocate(cuda::stream_ref{cudaStreamLegacy}, p1, kSize1);
  EXPECT_EQ(counter->get_allocated_bytes(), kSize2);
  EXPECT_EQ(counter->get_peak_bytes(), kSize1 + kSize2);

  dr.deallocate(cuda::stream_ref{cudaStreamLegacy}, p2, kSize2);
  EXPECT_EQ(counter->get_allocated_bytes(), 0UL);
}

TEST(DryRunResource, DeviceAsyncLargeAllocation)
{
  auto dev_ref = rmm::mr::get_current_device_resource_ref();
  raft::mr::dry_run_resource<rmm::device_async_resource_ref> dr{dev_ref};
  auto counter = dr.get_counter();

  constexpr std::size_t kOneGiB = 1024UL * 1024UL * 1024UL;
  void* ptr                     = dr.allocate(cuda::stream_ref{cudaStreamLegacy}, kOneGiB);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(counter->get_allocated_bytes(), kOneGiB);

  dr.deallocate(cuda::stream_ref{cudaStreamLegacy}, ptr, kOneGiB);
  EXPECT_EQ(counter->get_allocated_bytes(), 0UL);
  EXPECT_EQ(counter->get_peak_bytes(), kOneGiB);
}

// ===== dry_run_resource tests (host sync) =====

TEST(DryRunResource, HostSyncPeakTracking)
{
  auto host_ref = raft::mr::get_default_host_resource();
  raft::mr::dry_run_resource<raft::mr::host_resource_ref> dr{host_ref};
  auto counter = dr.get_counter();

  constexpr std::size_t kSize1 = 100UL * 1024UL * 1024UL;
  constexpr std::size_t kSize2 = 200UL * 1024UL * 1024UL;

  void* p1 = dr.allocate_sync(kSize1);
  void* p2 = dr.allocate_sync(kSize2);
  EXPECT_EQ(p1, p2);  // same probed pointer
  EXPECT_EQ(counter->get_peak_bytes(), kSize1 + kSize2);

  dr.deallocate_sync(p1, kSize1);
  dr.deallocate_sync(p2, kSize2);
  EXPECT_EQ(counter->get_allocated_bytes(), 0UL);
  EXPECT_EQ(counter->get_peak_bytes(), kSize1 + kSize2);
}

// ===== dry_run_flag resource tests =====

TEST(DryRunFlag, DefaultIsFalse)
{
  raft::resources res;
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

TEST(DryRunFlag, SetAndGet)
{
  raft::resources res;
  resource::set_dry_run_flag(res, true);
  EXPECT_TRUE(resource::get_dry_run_flag(res));

  resource::set_dry_run_flag(res, false);
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

// ===== dry_run_resources tests =====

TEST(DryRunResources, SetsFlag)
{
  raft::resources res;
  EXPECT_FALSE(resource::get_dry_run_flag(res));
  {
    dry_run_resources dry_res(res);
    EXPECT_TRUE(resource::get_dry_run_flag(dry_res));
    EXPECT_TRUE(resource::get_dry_run_flag(res));
  }
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

TEST(DryRunResources, RestoresGlobalDeviceResource)
{
  auto original_mr = rmm::mr::get_current_device_resource_ref();
  raft::resources res;
  {
    dry_run_resources dry_res(res);
    auto current_mr = rmm::mr::get_current_device_resource_ref();
    EXPECT_NE(current_mr, original_mr);
  }
  EXPECT_EQ(rmm::mr::get_current_device_resource_ref(), original_mr);
}

TEST(DryRunResources, RestoresGlobalHostResource)
{
  auto original_ref = raft::mr::get_default_host_resource();
  raft::resources res;
  {
    dry_run_resources dry_res(res);
    auto current_ref = raft::mr::get_default_host_resource();
    EXPECT_NE(current_ref, original_ref);
  }
  EXPECT_EQ(raft::mr::get_default_host_resource(), original_ref);
}

TEST(DryRunResources, StatsAccuracy)
{
  raft::resources res;
  constexpr std::size_t kAllocSize = 64UL * 1024UL * 1024UL;

  dry_run_resources dry_res(res);

  auto mr   = rmm::mr::get_current_device_resource_ref();
  void* ptr = mr.allocate(cuda::stream_ref{cudaStreamLegacy}, kAllocSize);
  mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, ptr, kAllocSize);

  auto stats = dry_res.get_bytes_peak();
  EXPECT_EQ(stats.device_global, kAllocSize);
}

TEST(DryRunResources, PinnedStatsAccuracy)
{
  raft::resources res;
  constexpr std::size_t kAllocSize = 64UL * 1024UL * 1024UL;

  dry_run_resources dry_res(res);

  auto ref  = resource::get_pinned_memory_resource_ref(dry_res);
  void* ptr = ref.allocate_sync(kAllocSize);
  ref.deallocate_sync(ptr, kAllocSize);

  auto stats = dry_res.get_bytes_peak();
  EXPECT_EQ(stats.host_pinned, kAllocSize);
}

TEST(DryRunResources, ManagedStatsAccuracy)
{
  raft::resources res;
  constexpr std::size_t kAllocSize = 64UL * 1024UL * 1024UL;

  dry_run_resources dry_res(res);

  auto ref  = resource::get_managed_memory_resource_ref(dry_res);
  void* ptr = ref.allocate_sync(kAllocSize);
  ref.deallocate_sync(ptr, kAllocSize);

  auto stats = dry_res.get_bytes_peak();
  EXPECT_EQ(stats.device_managed, kAllocSize);
}

// ===== dry_run_execute tests =====

TEST(DryRunExecute, BasicExecution)
{
  raft::resources res;
  bool action_ran = false;

  auto stats = dry_run_execute(res, [&](raft::resources const& r) {
    action_ran = true;
    EXPECT_TRUE(resource::get_dry_run_flag(r));

    auto mr                     = rmm::mr::get_current_device_resource_ref();
    constexpr std::size_t kSize = 32UL * 1024UL * 1024UL;
    void* ptr                   = mr.allocate(cuda::stream_ref{cudaStreamLegacy}, kSize);
    mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, ptr, kSize);
  });

  EXPECT_TRUE(action_ran);
  EXPECT_FALSE(resource::get_dry_run_flag(res));
  EXPECT_EQ(stats.device_global, 32UL * 1024UL * 1024UL);
}

TEST(DryRunExecute, ExceptionSafety)
{
  raft::resources res;
  auto original_mr   = rmm::mr::get_current_device_resource_ref();
  auto original_host = raft::mr::get_default_host_resource();

  EXPECT_THROW(dry_run_execute(
                 res, [](raft::resources const&) { throw std::runtime_error("test exception"); }),
               std::runtime_error);

  EXPECT_EQ(rmm::mr::get_current_device_resource_ref(), original_mr);
  EXPECT_EQ(raft::mr::get_default_host_resource(), original_host);
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

// ===== Independent-counting tests for dry_run_resources =====

TEST(DryRunResources, IndependentCounting_DefaultWorkspace)
{
  raft::resources res;

  dry_run_resources dry_res(res);

  constexpr std::size_t kWsSize     = 1024;
  constexpr std::size_t kGlobalSize = 2048;

  auto ws_ref  = resource::get_workspace_resource_ref(dry_res);
  void* ws_ptr = ws_ref.allocate(cuda::stream_ref{cudaStreamLegacy}, kWsSize);

  auto dev_mr   = rmm::mr::get_current_device_resource_ref();
  void* dev_ptr = dev_mr.allocate(cuda::stream_ref{cudaStreamLegacy}, kGlobalSize);

  auto peak = dry_res.get_bytes_peak();
  EXPECT_EQ(peak.device_workspace, kWsSize);
  EXPECT_EQ(peak.device_global, kGlobalSize);
  EXPECT_EQ(peak.total(), kWsSize + kGlobalSize);

  ws_ref.deallocate(cuda::stream_ref{cudaStreamLegacy}, ws_ptr, kWsSize);
  dev_mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, dev_ptr, kGlobalSize);
}

TEST(DryRunResources, IndependentCounting_WorkspaceSetToGlobal)
{
  raft::resources res;
  resource::set_workspace_to_global_resource(res);

  dry_run_resources dry_res(res);

  constexpr std::size_t kWsSize     = 1024;
  constexpr std::size_t kGlobalSize = 2048;

  auto ws_ref  = resource::get_workspace_resource_ref(dry_res);
  void* ws_ptr = ws_ref.allocate(cuda::stream_ref{cudaStreamLegacy}, kWsSize);

  auto dev_mr   = rmm::mr::get_current_device_resource_ref();
  void* dev_ptr = dev_mr.allocate(cuda::stream_ref{cudaStreamLegacy}, kGlobalSize);

  auto peak = dry_res.get_bytes_peak();
  EXPECT_EQ(peak.device_workspace, kWsSize);
  EXPECT_EQ(peak.device_global, kGlobalSize);
  EXPECT_EQ(peak.total(), kWsSize + kGlobalSize);

  ws_ref.deallocate(cuda::stream_ref{cudaStreamLegacy}, ws_ptr, kWsSize);
  dev_mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, dev_ptr, kGlobalSize);
}

// ===== Independent-counting tests for memory_stats_resources =====

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

// ===== Nested wrappers test =====

TEST(IndependentCounting, NestedDryRunInStats)
{
  raft::resources res;

  memory_stats_resources stat_res(res);
  dry_run_resources dry_res(stat_res);

  constexpr std::size_t kWsSize     = 1024;
  constexpr std::size_t kGlobalSize = 2048;

  auto ws_ref  = resource::get_workspace_resource_ref(dry_res);
  void* ws_ptr = ws_ref.allocate(cuda::stream_ref{cudaStreamLegacy}, kWsSize);

  auto dev_mr   = rmm::mr::get_current_device_resource_ref();
  void* dev_ptr = dev_mr.allocate(cuda::stream_ref{cudaStreamLegacy}, kGlobalSize);

  auto peak = dry_res.get_bytes_peak();
  EXPECT_EQ(peak.device_workspace, kWsSize);
  EXPECT_EQ(peak.device_global, kGlobalSize);
  EXPECT_EQ(peak.total(), kWsSize + kGlobalSize);

  ws_ref.deallocate(cuda::stream_ref{cudaStreamLegacy}, ws_ptr, kWsSize);
  dev_mr.deallocate(cuda::stream_ref{cudaStreamLegacy}, dev_ptr, kGlobalSize);
}

}  // namespace raft::util
