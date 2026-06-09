/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/nvtx_range_stack.hpp>   // thread_local_current_range
#include <raft/mr/allocation_event_monitor.hpp>    // allocation_event, allocation_event_queue
#include <raft/mr/statistics_adaptor.hpp>          // resource_stats (atomic counters, reused)

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace raft {
namespace mr {

/**
 * @brief Resource adaptor that records each allocation/deallocation as an event,
 *        capturing the active NVTX range AT THE TIME OF THE EVENT.
 *
 * Forwards all calls to the upstream resource, updates a per-source
 * @ref resource_stats counter, then pushes an @ref allocation_event (carrying a
 * snapshot of that source's counters plus the current NVTX range) onto a shared
 * queue.  A single @ref allocation_event_monitor drains the queue and writes the
 * CSV.
 *
 * This is the producer half of the event-driven replacement for the sampling
 * resource_monitor: because the range is captured here, on the allocating
 * thread, it can never be mis-attributed to a later range.
 *
 * @tparam Upstream  Stored by value.  Use a concrete resource type for owning
 *                   semantics, or a ref type for non-owning semantics.
 */
template <typename Upstream>
class recording_adaptor : public cuda::forward_property<recording_adaptor<Upstream>, Upstream> {
  Upstream upstream_;
  std::shared_ptr<resource_stats> stats_;
  std::shared_ptr<allocation_event_queue> queue_;
  int source_id_;

  // Build and enqueue an event from the current counter snapshot + NVTX range.
  // noexcept: profiling bookkeeping must never disrupt the allocation path, so
  // any failure (e.g. bad_alloc while copying the range string) is swallowed.
  void emit() noexcept
  {
    try {
      allocation_event event;
      event.source_id   = source_id_;
      event.current     = stats_->bytes_current.load(std::memory_order_relaxed);
      event.total_alloc = stats_->bytes_total_allocated.load(std::memory_order_relaxed);
      event.total_freed = stats_->bytes_total_deallocated.load(std::memory_order_relaxed);
      event.timestamp   = std::chrono::steady_clock::now();
      auto range        = raft::common::nvtx::thread_local_current_range()->get();
      event.nvtx_range  = std::move(range.first);
      event.nvtx_depth  = range.second;
      queue_->push(std::move(event));
    } catch (...) {
    }
  }

 public:
  recording_adaptor(Upstream upstream,
                    std::shared_ptr<allocation_event_queue> queue,
                    int source_id)
    : upstream_(std::move(upstream)),
      stats_(std::make_shared<resource_stats>()),
      queue_(std::move(queue)),
      source_id_(source_id)
  {
  }

  /** @brief Access this source's shared counters. */
  [[nodiscard]] auto get_stats() const noexcept -> std::shared_ptr<resource_stats> { return stats_; }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    void* ptr = upstream_.allocate_sync(bytes, alignment);
    stats_->record_allocate(static_cast<std::int64_t>(bytes));
    emit();
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate_sync(ptr, bytes, alignment);
    stats_->record_deallocate(static_cast<std::int64_t>(bytes));
    emit();
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    void* ptr = upstream_.allocate(stream, bytes, alignment);
    stats_->record_allocate(static_cast<std::int64_t>(bytes));
    emit();
    return ptr;
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
    stats_->record_deallocate(static_cast<std::int64_t>(bytes));
    emit();
  }

  [[nodiscard]] bool operator==(recording_adaptor const& other) const noexcept
  {
    return upstream_ == other.upstream_;
  }

  [[nodiscard]] auto upstream_resource() noexcept -> Upstream& { return upstream_; }
  [[nodiscard]] auto upstream_resource() const noexcept -> Upstream const& { return upstream_; }
};

}  // namespace mr
}  // namespace raft
