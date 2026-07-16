/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/nvtx_range_stack.hpp>
#include <raft/mr/notifying_adaptor.hpp>
#include <raft/mr/statistics_adaptor.hpp>

#include <atomic>
#include <chrono>
#include <memory>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace raft {
namespace mr {

/**
 * @brief Collects allocation statistics from multiple resource_stats sources
 *        and writes CSV snapshots to an output stream from a background thread.
 *
 * Usage:
 *   1. Construct with an output stream and sampling interval
 *      (captures the constructing thread's NVTX range handle).
 *   2. Register each monitored resource via register_source().
 *   3. Call start() to begin sampling in a background thread.
 *   4. Call stop() to end sampling.  The destructor calls stop() if needed.
 *
 * The background thread blocks on notifier::wait() until a
 * notifying_adaptor signals an allocation or deallocation.
 * After being woken, it sleeps for sample_interval to coalesce bursts,
 * then writes one CSV row.
 *
 * start() and stop() are idempotent.
 */
class resource_monitor {
  std::ostream& out_;
  std::chrono::steady_clock::duration sample_interval_;
  std::shared_ptr<notifier> notifier_;
  std::vector<std::pair<std::string, std::shared_ptr<resource_stats>>> sources_;
  std::shared_ptr<const raft::common::nvtx::current_range> nvtx_range_;

  std::chrono::steady_clock::time_point start_time_{std::chrono::steady_clock::now()};
  std::atomic<bool> stop_requested_{false};
  std::thread worker_;

 public:
  /**
   * @param out             Output stream for CSV rows.
   * @param sample_interval Minimum time between successive samples.
   */
  explicit resource_monitor(std::ostream& out, std::chrono::steady_clock::duration sample_interval)
    : out_(out),
      sample_interval_(sample_interval),
      notifier_(std::make_shared<notifier>()),
      nvtx_range_(raft::common::nvtx::thread_local_current_range())
  {
  }

  ~resource_monitor() { stop(); }

  resource_monitor(resource_monitor const&)            = delete;
  resource_monitor& operator=(resource_monitor const&) = delete;

  /**
   * @brief Shared notifier for notifying_adaptor instances.
   *
   * The adaptor calls notify() on every allocation/deallocation;
   * the background sampling thread calls wait() to block until activity occurs.
   */
  [[nodiscard]] auto get_notifier() const noexcept -> std::shared_ptr<notifier>
  {
    return notifier_;
  }

  /**
   * @brief Register a named statistics source.  Must be called before start().
   * @param name  Column-name prefix in the CSV output.
   * @param stats Shared counters updated by a statistics_adaptor.
   */
  void register_source(std::string name, std::shared_ptr<resource_stats> stats)
  {
    sources_.emplace_back(std::move(name), std::move(stats));
  }

  /** @brief Begin sampling in a background thread.  No-op if already started. */
  void start()
  {
    if (worker_.joinable()) { return; }
    stop_requested_.store(false, std::memory_order_relaxed);
    write_header();
    worker_ = std::thread([this] { sample_loop(); });
  }

  /** @brief Stop the background sampling thread.  No-op if not running. */
  void stop()
  {
    if (!worker_.joinable()) { return; }
    stop_requested_.store(true, std::memory_order_release);
    notifier_->notify();
    worker_.join();
    // Write one more row because we may have missed some updates while spinning down.
    // In the worst case, we duplicate one CSV record - not a big deal.
    write_row();
  }

 private:
  void write_header()
  {
    out_ << "timestamp_us";
    for (auto const& [name, _] : sources_) {
      out_ << ',' << name << "_current," << name << "_peak," << name << "_total_alloc," << name
           << "_total_freed";
    }
    out_ << ",nvtx_depth,nvtx_range\n";
    out_.flush();
  }

  void sample_loop()
  {
    while (!stop_requested_.load(std::memory_order_relaxed)) {
      notifier_->wait();  // waits indefinitely until notify() is called
      write_row();
      // sleep for the minimum time interval
      std::this_thread::sleep_for(sample_interval_);
    }
  }

  void write_row()
  {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_time_)
                .count();
    auto [range, depth] = nvtx_range_->get();

    out_ << us;
    for (auto const& [name, stats] : sources_) {
      auto current = stats->bytes_current.load(std::memory_order_relaxed);
      // Reset the peak value to allow tracking local peaks within the sample interval.
      auto peak = stats->bytes_peak.exchange(current, std::memory_order_relaxed);
      // Total alloc+dealloc together allow tracking the allocation throughput and detect leaks.
      auto total_alloc = stats->bytes_total_allocated.load(std::memory_order_relaxed);
      auto total_freed = stats->bytes_total_deallocated.load(std::memory_order_relaxed);

      out_ << ',' << current << ',' << peak << ',' << total_alloc << ',' << total_freed;
    }
    out_ << ',' << depth << ",\"" << range << '"';
    out_ << std::endl;
    out_.flush();
  }
};

}  // namespace mr
}  // namespace raft
