/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/nvtx_range_stack.hpp>  // thread_local_current_path, thread_local_current_name_and_depth
#include <raft/mr/recording_monitor.hpp>   // allocation_event, allocation_event_queue
#include <raft/mr/statistics_adaptor.hpp>  // resource_stats (atomic counters, reused)

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace raft {
namespace mr {

/**
 * @brief Resource adaptor that records each allocation/deallocation as an event,
 *        and associates it with the active NVTX range AT THE TIME OF THE EVENT.
 */
template <typename Upstream>
class recording_adaptor : public cuda::forward_property<recording_adaptor<Upstream>, Upstream> {
  // Map an allocated address to the nvtx stack range responsible for the allocation.
  // It allows the deallocation event to be tagged with the same range, even if the responsible
  // range has ended by the time of deallocation.
  struct address_range_map {
    std::mutex mtx;
    std::unordered_map<void*, std::string> paths;
  };

  Upstream upstream_;
  std::shared_ptr<resource_stats> stats_;
  std::shared_ptr<allocation_event_queue> queue_;  // stores all alloc and dealloc events
  std::shared_ptr<address_range_map> alloc_map_;
  int source_id_;

  // Record the alloc-time NVTX path for this pointer.
  // Called on the allocating thread — mutex-free NVTX read is safe.
  // The alloc_map_ mutex still protects the shared map from concurrent alloc/dealloc across
  // threads.
  auto record_allocation(void* ptr) noexcept -> std::string
  {
    std::string path = "";
    if (ptr != nullptr) {
      path = raft::common::nvtx::thread_local_current_path();
      if (!path.empty()) {
        std::lock_guard<std::mutex> lock(alloc_map_->mtx);
        alloc_map_->paths[ptr] = path;
      }
    }
    return path;
  }

  // Returns the NVTX path recorded at alloc time for this pointer, then removes it.
  auto forget_allocation(void* ptr) noexcept -> std::string
  {
    std::string path = "";
    std::lock_guard<std::mutex> lock(alloc_map_->mtx);
    auto it = alloc_map_->paths.find(ptr);
    if (it != alloc_map_->paths.end()) {
      path = std::move(it->second);
      alloc_map_->paths.erase(it);
    }
    return path;
  }

  // Enqueue an event.  Called on the allocating/deallocating thread — mutex-free NVTX read.
  void emit(std::string nvtx_full_range, std::int64_t signed_bytes) noexcept
  {
    auto [name, depth] = raft::common::nvtx::thread_local_current_name_and_depth();
    allocation_event event;
    event.source_id       = source_id_;
    event.current         = stats_->bytes_current.load(std::memory_order_relaxed);
    event.total_alloc     = stats_->bytes_total_allocated.load(std::memory_order_relaxed);
    event.total_freed     = stats_->bytes_total_deallocated.load(std::memory_order_relaxed);
    event.timestamp       = std::chrono::steady_clock::now();
    event.event_bytes     = signed_bytes;
    event.nvtx_range      = std::move(name);
    event.nvtx_depth      = depth;
    event.nvtx_full_range = std::move(nvtx_full_range);
    queue_->push(std::move(event));
  }

 public:
  recording_adaptor(Upstream upstream, std::shared_ptr<allocation_event_queue> queue, int source_id)
    : upstream_(std::move(upstream)),
      stats_(std::make_shared<resource_stats>()),
      queue_(std::move(queue)),
      alloc_map_(std::make_shared<address_range_map>()),
      source_id_(source_id)
  {
  }

  /** @brief Access this source's shared counters. */
  [[nodiscard]] auto get_stats() const noexcept -> std::shared_ptr<resource_stats>
  {
    return stats_;
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    void* ptr = upstream_.allocate_sync(bytes, alignment);
    stats_->record_allocate(static_cast<std::int64_t>(bytes));
    emit(record_allocation(ptr), static_cast<std::int64_t>(bytes));
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate_sync(ptr, bytes, alignment);
    stats_->record_deallocate(static_cast<std::int64_t>(bytes));
    emit(forget_allocation(ptr), -static_cast<std::int64_t>(bytes));
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    void* ptr = upstream_.allocate(stream, bytes, alignment);
    stats_->record_allocate(static_cast<std::int64_t>(bytes));
    emit(record_allocation(ptr), static_cast<std::int64_t>(bytes));
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
    emit(forget_allocation(ptr), -static_cast<std::int64_t>(bytes));
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
