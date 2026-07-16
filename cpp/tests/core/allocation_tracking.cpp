/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/mr/host_memory_resource.hpp>
#include <raft/mr/notifying_adaptor.hpp>
#include <raft/mr/resource_monitor.hpp>
#include <raft/mr/statistics_adaptor.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <sstream>
#include <string>
#include <thread>

namespace {

TEST(StatisticsResourceAdaptor, TracksSyncAllocations)
{
  auto upstream = raft::mr::get_default_host_resource();
  raft::mr::statistics_adaptor adaptor{upstream};
  auto stats = adaptor.get_stats();

  void* p1 = adaptor.allocate_sync(100);
  ASSERT_NE(p1, nullptr);
  EXPECT_EQ(stats->bytes_current.load(), 100);
  EXPECT_EQ(stats->bytes_peak.load(), 100);
  EXPECT_EQ(stats->num_allocations.load(), 1);
  EXPECT_EQ(stats->bytes_total_allocated.load(), 100);

  void* p2 = adaptor.allocate_sync(200);
  EXPECT_EQ(stats->bytes_current.load(), 300);
  EXPECT_EQ(stats->bytes_peak.load(), 300);
  EXPECT_EQ(stats->num_allocations.load(), 2);

  adaptor.deallocate_sync(p1, 100);
  EXPECT_EQ(stats->bytes_current.load(), 200);
  EXPECT_EQ(stats->bytes_peak.load(), 300);
  EXPECT_EQ(stats->num_deallocations.load(), 1);
  EXPECT_EQ(stats->bytes_total_deallocated.load(), 100);

  adaptor.deallocate_sync(p2, 200);
  EXPECT_EQ(stats->bytes_current.load(), 0);
  EXPECT_EQ(stats->bytes_peak.load(), 300);
  EXPECT_EQ(stats->num_deallocations.load(), 2);
}

TEST(NotifyingResourceAdaptor, NotifiesOnAllocate)
{
  auto n        = std::make_shared<raft::mr::notifier>();
  auto upstream = raft::mr::get_default_host_resource();
  raft::mr::statistics_adaptor stats_adaptor{upstream};
  auto stats = stats_adaptor.get_stats();
  raft::mr::notifying_adaptor adaptor{std::move(stats_adaptor), n};

  void* ptr = adaptor.allocate_sync(64);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(stats->bytes_current.load(), 64);

  adaptor.deallocate_sync(ptr, 64);
  EXPECT_EQ(stats->bytes_current.load(), 0);
}

TEST(AllocationReport, WritesCSVOnDirty)
{
  using namespace std::chrono_literals;

  std::ostringstream oss;
  raft::mr::resource_monitor report(oss, 1ms);

  auto host_stats   = std::make_shared<raft::mr::resource_stats>();
  auto pinned_stats = std::make_shared<raft::mr::resource_stats>();
  report.register_source("host", host_stats);
  report.register_source("pinned", pinned_stats);

  host_stats->record_allocate(1024);
  pinned_stats->record_allocate(2048);
  report.get_notifier()->notify();

  report.start();
  std::this_thread::sleep_for(50ms);
  report.stop();

  auto output = oss.str();

  EXPECT_NE(output.find("timestamp_us"), std::string::npos);
  EXPECT_NE(output.find("host_current"), std::string::npos);
  EXPECT_NE(output.find("pinned_current"), std::string::npos);
  EXPECT_NE(output.find("1024"), std::string::npos);
  EXPECT_NE(output.find("2048"), std::string::npos);
}

TEST(AllocationReport, StartStopIdempotent)
{
  using namespace std::chrono_literals;

  std::ostringstream oss;
  raft::mr::resource_monitor report(oss, 1ms);

  auto stats = std::make_shared<raft::mr::resource_stats>();
  report.register_source("test", stats);

  report.start();
  report.start();

  stats->record_allocate(100);
  report.get_notifier()->notify();
  std::this_thread::sleep_for(50ms);

  report.stop();
  report.stop();

  auto output = oss.str();
  EXPECT_NE(output.find("100"), std::string::npos);
}

TEST(AllocationReport, DestructorCallsStop)
{
  using namespace std::chrono_literals;

  std::ostringstream oss;
  {
    auto stats = std::make_shared<raft::mr::resource_stats>();
    raft::mr::resource_monitor report(oss, 1ms);
    report.register_source("test", stats);

    stats->record_allocate(256);
    report.get_notifier()->notify();

    report.start();
    std::this_thread::sleep_for(50ms);
  }

  auto output = oss.str();
  EXPECT_NE(output.find("256"), std::string::npos);
}

TEST(StackedAdaptors, HostResourceRefProperties)
{
  auto n        = std::make_shared<raft::mr::notifier>();
  auto upstream = raft::mr::get_default_host_resource();

  raft::mr::statistics_adaptor stats_adaptor{upstream};
  auto stats = stats_adaptor.get_stats();
  raft::mr::notifying_adaptor notify_adaptor{std::move(stats_adaptor), n};

  // The stacked adaptor should satisfy synchronous_resource_with<host_accessible>
  // and be usable as raft::mr::host_resource_ref.
  raft::mr::host_resource_ref ref{notify_adaptor};

  void* ptr = ref.allocate_sync(512);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(stats->bytes_current.load(), 512);

  ref.deallocate_sync(ptr, 512);
  EXPECT_EQ(stats->bytes_current.load(), 0);
}

}  // namespace
