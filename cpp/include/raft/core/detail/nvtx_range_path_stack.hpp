/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace raft {
namespace common::nvtx {

namespace detail {

/**
 * Process-wide counter producing a unique id to every pushed range, so that
 * two nvtx ranges sharing the same name can differentiate.
 *
 * Motivation: To sample different allocation stats over different runs of the
 * same block of code. User can aggregate them into a distribution of allocations
 * over multiple runs.
 */
RAFT_EXPORT inline std::atomic<std::uint64_t> range_instance_counter{0};

/**
 * @brief Per-thread NVTX range stack that records the full range path with
 *        unique range instance ids.
 */
struct nvtx_full_range_stack {
  void push(const char* name)
  {
    auto id = range_instance_counter.fetch_add(1, std::memory_order_relaxed) + 1;
    stack_.emplace_back(id, name ? name : "");
  }

  void pop()
  {
    if (!stack_.empty()) { stack_.pop_back(); }
  }

  /** Innermost range name and stack depth (empty/0 when no range is active). */
  [[nodiscard]] auto inner_range_and_depth() const noexcept -> std::pair<std::string, std::size_t>
  {
    if (stack_.empty()) { return {"", 0}; }
    return {stack_.back().second, stack_.size()};
  }

  /** Full range path "name#id -> name#id -> ..." (empty when no range is active). */
  [[nodiscard]] auto current_path() const -> std::string
  {
    std::string path;
    for (auto const& [id, name] : stack_) {
      if (!path.empty()) { path += " -> "; }
      path += name + '#' + std::to_string(id);
    }
    return path;
  }

 private:
  // (instance id, range name), outer -> inner (top).
  std::vector<std::pair<std::uint64_t, std::string>> stack_{};
};

RAFT_EXPORT inline thread_local nvtx_full_range_stack full_range_stack_instance{};

}  // namespace detail

/**
 * Mutex-free read of the current thread's innermost NVTX range name and stack depth.
 *
 * ONLY safe to call from the thread that owns this range stack (the current thread).
 */
RAFT_EXPORT inline auto thread_local_inner_range_and_depth() -> std::pair<std::string, std::size_t>
{
  return detail::full_range_stack_instance.inner_range_and_depth();
}

/**
 * Mutex-free read of the current thread's full NVTX range path "name#id -> name#id -> ...".
 *
 * ONLY safe to call from the thread that owns this range stack (the current thread).
 */
RAFT_EXPORT inline auto thread_local_nvtx_full_path() -> std::string
{
  return detail::full_range_stack_instance.current_path();
}

}  // namespace common::nvtx
}  // namespace raft
