/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <cstddef>
#include <memory>
#include <mutex>
#include <stack>
#include <string>
#include <utility>

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

  operator std::string() const
  {
    std::lock_guard lock(mu_);
    return value_;
  }

 private:
  mutable std::mutex mu_;
  std::string value_;
  std::size_t depth_{0};

  void set(const char* name, std::size_t depth)
  {
    std::lock_guard lock(mu_);
    value_ = name ? name : "";
    depth_ = depth;
  }
};

namespace detail {

struct nvtx_range_name_stack {
  void push(const char* name)
  {
    stack_.emplace(name);
    current_->set(name, stack_.size());
  }

  void pop()
  {
    if (!stack_.empty()) { stack_.pop(); }
    current_->set(stack_.empty() ? nullptr : stack_.top().c_str(), stack_.size());
  }

  auto current() const -> std::shared_ptr<const current_range> { return current_; }

 private:
  std::stack<std::string> stack_{};
  std::shared_ptr<current_range> current_{std::make_shared<current_range>()};
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
