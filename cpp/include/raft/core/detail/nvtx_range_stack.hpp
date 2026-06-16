/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace raft {
namespace common::nvtx {

namespace detail {
struct nvtx_range_name_stack;
}  // namespace detail

/**
 * Shared, read-only handle to the current NVTX range name of a specific thread
 * (set internally by one thread, read publicly by zero or more threads).
 */
class current_range {
  friend detail::nvtx_range_name_stack;

 public:
  /** Read the current range name and stack depth (safe to call from any thread). */
  auto get() const -> std::pair<std::string, std::size_t>
  {
    std::lock_guard lock(mu_);
    return {value_, depth_};
  }

  /**
   * Read the full root->leaf range path with instance ids, formatted as
   * "name#id > name#id > ..." (empty when no range is active).
   * This identifies the exact nvtx range stack responsible for an allocation.
   */
  auto get_path() const -> std::string
  {
    std::lock_guard lock(mu_);
    return path_;
  }

  operator std::string() const
  {
    std::lock_guard lock(mu_);
    return value_;
  }

 private:
  mutable std::mutex mu_;
  std::string value_;
  std::size_t depth_{0};
  std::string path_;

  void set(const char* name, std::size_t depth, std::string path)
  {
    std::lock_guard lock(mu_);
    value_ = name ? name : "";
    depth_ = depth;
    path_  = std::move(path);
  }
};

namespace detail {

RAFT_EXPORT inline std::atomic<std::uint64_t> range_instance_counter{0};

struct nvtx_range_name_stack {
  void push(const char* name)
  {
    ensure_current();
    auto id = range_instance_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    stack_.emplace_back(id, name ? name : "");
    current_->set(stack_.back().second.c_str(), stack_.size(), build_path());
  }

  void pop()
  {
    ensure_current();
    if (!stack_.empty()) { stack_.pop_back(); }
    current_->set(
      stack_.empty() ? nullptr : stack_.back().second.c_str(), stack_.size(), build_path());
  }

  [[nodiscard]] auto current() const -> std::shared_ptr<const current_range>
  {
    ensure_current();
    return current_;
  }

 private:
  void ensure_current() const
  {
    if (!current_) { current_ = std::make_shared<current_range>(); }
  }

  // Serialize the active stack as "name#id > name#id > ..." (outer -> inner).
  [[nodiscard]] auto build_path() const -> std::string
  {
    std::string path;
    for (auto const& [id, name] : stack_) {
      if (!path.empty()) { path += " > "; }
      path += name;
      path += '#';
      path += std::to_string(id);
    }
    return path;
  }

  std::vector<std::pair<std::uint64_t, std::string>> stack_{};
  mutable std::shared_ptr<current_range> current_{std::make_shared<current_range>()};
};

RAFT_EXPORT inline thread_local nvtx_range_name_stack range_name_stack_instance{};

}  // namespace detail

/**
 * Get a read-only handle to this thread's current NVTX range name.
 * Pass the returned shared_ptr to another thread to read this thread's current NVTX range name at
 * any time.
 */
RAFT_EXPORT inline auto thread_local_current_range() -> std::shared_ptr<const current_range>
{
  return detail::range_name_stack_instance.current();
}

}  // namespace common::nvtx
}  // namespace raft
