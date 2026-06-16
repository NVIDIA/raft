/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

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
  int source_id{0};             //< which registered source this belongs to
  std::int64_t current{0};      //< source's live bytes after this event
  std::int64_t total_alloc{0};  //< cumulative bytes allocated (this source)
  std::int64_t total_freed{0};  //< cumulative bytes freed (this source)
  std::size_t nvtx_depth{0};    //< NVTX stack depth at event time
  std::string nvtx_range;       //< NVTX range name active at event time
  std::int64_t event_bytes{0};  //< signed bytes for THIS event (+alloc / -free)
  std::string alloc_range;      //< responsible range path "name#id > ..."
                                //  captured at ALLOCATION time (empty if unknown)
  std::chrono::steady_clock::time_point timestamp{};  //< when the event happened
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
   * @return false once the queue is stopped AND drained (consumer should exit),
   *         true otherwise.
   */
  bool wait_and_take(std::vector<allocation_event>& out)
  {
    std::unique_lock<std::mutex> lock(mtx_);
    cv_.wait(lock, [this] { return stopped_ || !events_.empty(); });
    out.clear();
    out.swap(events_);
    return !(stopped_ && out.empty());
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
class allocation_event_monitor {
 public:
  explicit allocation_event_monitor(std::ostream& out) : out_(out) {}

  ~allocation_event_monitor() { stop(); }

  allocation_event_monitor(allocation_event_monitor const&)            = delete;
  allocation_event_monitor& operator=(allocation_event_monitor const&) = delete;

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
    int id = static_cast<int>(source_names_.size());  // TODO (huuanhhuyn) conflict id?
    source_names_.push_back(std::move(name));
    view_.emplace_back();
    return id;
  }

  void start()
  {
    if (worker_.joinable()) { return; }
    write_header();
    worker_ = std::thread([this] { run(); });
  }

  void stop()
  {
    if (!worker_.joinable()) { return; }
    queue_->stop();  // drains the queue and causes the worker to exit its loop
    worker_.join();
  }

 private:
  struct source_view {
    std::int64_t current{0};
    std::int64_t total_alloc{0};
    std::int64_t total_freed{0};
  };

  void write_header()
  {
    out_ << "timestamp_us";
    for (auto const& name : source_names_) {
      out_ << ',' << name << "_current," << name << "_peak," << name << "_total_alloc," << name
           << "_total_freed";
    }
    out_ << ",nvtx_depth,nvtx_range,event_source,event_bytes,alloc_range\n";
    out_.flush();
  }

  void run()
  {
    std::vector<allocation_event> batch;
    for (;;) {
      bool keep_going = queue_->wait_and_take(batch);
      for (auto const& event : batch) {
        write_row(event);
      }
      out_.flush();
      if (!keep_going) { break; }
    }
  }

  void write_row(allocation_event const& event)
  {
    if (event.source_id >= 0 && event.source_id < static_cast<int>(view_.size())) {
      view_[event.source_id] = source_view{event.current, event.total_alloc, event.total_freed};
    }

    auto us =
      std::chrono::duration_cast<std::chrono::microseconds>(event.timestamp - start_).count();
    out_ << us;
    for (auto const& v : view_) {
      out_ << ',' << v.current << ',' << v.current << ',' << v.total_alloc << ',' << v.total_freed;
    }
    out_ << ',' << event.nvtx_depth << ",\"" << event.nvtx_range << "\"";

    auto const* src_name =
      (event.source_id >= 0 && event.source_id < static_cast<int>(source_names_.size()))
        ? source_names_[event.source_id].c_str()
        : "";
    out_ << ',' << src_name << ',' << event.event_bytes << ",\"" << event.alloc_range << "\"\n";
  }

  std::ostream& out_;
  std::shared_ptr<allocation_event_queue> queue_{std::make_shared<allocation_event_queue>()};
  std::vector<std::string> source_names_;
  std::vector<source_view> view_;
  std::chrono::steady_clock::time_point start_{std::chrono::steady_clock::now()};
  std::thread worker_;
};

}  // namespace mr
}  // namespace raft
