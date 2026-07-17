/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>  // RAFT_EXPECTS
#include <raft/core/logger.hpp>

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <ostream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace raft {
namespace mr {

/**
 * @brief A single allocation or deallocation event, captured on the allocating thread.
 */
struct allocation_event {
  std::chrono::steady_clock::time_point timestamp{};  //< (de)allocation timestamp
  int source_id{0};  //< resource (host, pinned, workspace, etc.) which event belongs to
  std::int64_t current_bytes{0};   //< live bytes after this event (i.e. incl. delta bytes)
  std::int64_t delta_bytes{0};     //< signed delta bytes by this event (+alloc / -free)
  std::size_t nvtx_depth{0};       //< NVTX stack depth at event time
  std::string nvtx_inner_range{};  //< NVTX range name active at event time
  std::string
    nvtx_full_range{};  //< NVTX full path "name#id -> ..." captured at the ALLOCATION time
};

/**
 * @brief Thread-safe multi-producer / single-consumer queue of allocation_events.
 */
class allocation_event_queue {
 public:
  /** @brief Append an event (any thread). */
  void push(allocation_event event)
  {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      events_.push_back(std::move(event));
    }
    cv_.notify_one();
  }

  /**
   * @brief Block until events are available or the queue is stopped, then move
   *        all pending events into `out`.
   *
   * @return true once the queue is stopped AND drained (consumer should exit)
   */
  bool wait_and_take(std::vector<allocation_event>& out)
  {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return stopped_ || !events_.empty(); });
    out.clear();
    out.swap(events_);
    return stopped_ && out.empty();
  }

  /** @brief Signal the consumer to drain and exit. */
  void stop()
  {
    {
      std::lock_guard<std::mutex> lock(mtx_);
      stopped_ = true;
    }
    cv_.notify_all();
  }

 private:
  std::mutex mtx_;
  std::condition_variable cv_;
  std::vector<allocation_event> events_;
  bool stopped_{false};
};

/**
 * @brief Consumes allocation_events from a queue and writes one CSV row per
 *        event from a background thread.
 */
class recording_monitor {
 public:
  explicit recording_monitor(std::ostream& out) : out_(out) {}

  ~recording_monitor() { stop(); }

  recording_monitor(recording_monitor const&)            = delete;
  recording_monitor& operator=(recording_monitor const&) = delete;

  [[nodiscard]] auto get_queue() const noexcept -> std::shared_ptr<allocation_event_queue>
  {
    return queue_;
  }

  /**
   * @brief Register a named source and return its id (column-group index).
   *        Must be called before start().
   */
  auto register_source(std::string name) -> int
  {
    RAFT_EXPECTS(name.find(',') == std::string::npos,
                 "source name must not contain ',' (delimiter). This would break CSV columns: '%s'",
                 name.c_str());
    int source_id = static_cast<int>(sources_.size());
    sources_.push_back(std::move(name));
    source_current_.push_back(0);  // last-known live bytes for this source (carried forward)
    return source_id;
  }

  void start()
  {
    if (worker_.joinable()) { return; }
    write_header();
    // Start the background thread that consumes events from the queue
    // and writes one CSV row per event.
    worker_ = std::thread([this] { run(); });
  }

  void stop()
  {
    if (!worker_.joinable()) { return; }
    queue_->stop();  // drains the queue and causes the worker to exit its loop
    worker_.join();
  }

 private:
  void write_header()
  {
    out_ << "timestamp_us,source";
    for (auto const& name : sources_) {
      out_ << ',' << name << "_current_bytes";
    }
    out_ << ",delta_bytes";
    out_ << ",nvtx_depth,nvtx_inner_range,nvtx_full_range\n";
    out_.flush();
  }

  void run()
  {
    std::vector<allocation_event> batch;
    while (true) {
      bool finished = queue_->wait_and_take(batch);
      for (auto const& event : batch) {
        write_row(event);
      }
      out_.flush();
      if (finished) { break; }
    }
  }

  void write_row(allocation_event const& event)
  {
    if (event.source_id < 0 || static_cast<std::size_t>(event.source_id) >= sources_.size()) {
      RAFT_LOG_WARN("Event source id %d is out-of-bound (number of sources = %zu)",
                    event.source_id,
                    sources_.size());
      return;
    }

    // timestamp since start [us]
    out_ << std::chrono::duration_cast<std::chrono::microseconds>(event.timestamp - start_).count();
    out_ << "," << sources_[event.source_id];
    // live bytes per source (last-known value for each)
    source_current_[event.source_id] = event.current_bytes;
    for (auto const& current_bytes : source_current_) {
      out_ << "," << current_bytes;
    }
    // delta bytes
    out_ << "," << event.delta_bytes;
    // nvtx
    out_ << "," << event.nvtx_depth;
    out_ << ",\"" << event.nvtx_inner_range << "\"";
    out_ << ",\"" << event.nvtx_full_range << "\"\n";
  }

  std::ostream& out_;
  std::shared_ptr<allocation_event_queue> queue_{std::make_shared<allocation_event_queue>()};
  std::vector<std::string> sources_;          // pinned, workspace, host, etc.
  std::vector<std::int64_t> source_current_;  // last-known live bytes per source (carried forward)
  std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
  std::thread worker_;
};

}  // namespace mr
}  // namespace raft
