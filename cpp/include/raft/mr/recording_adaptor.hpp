/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/nvtx_range_path_stack.hpp>  // thread_local_nvtx_full_path, thread_local_inner_range_and_depth
#include <raft/core/error.hpp>
#include <raft/mr/recording_monitor.hpp>  // allocation_event, allocation_event_queue

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <atomic>
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
  // Map is shared across threads (e.g. one thread allocates and another deallocates)
  struct address_range_map {
    std::mutex mtx;
    std::unordered_map<void*, std::string> paths;
  };

  Upstream upstream_;
  std::shared_ptr<allocation_event_queue> queue_;
  std::shared_ptr<address_range_map> alloc_map_;
  int source_id_;
  std::shared_ptr<std::atomic<std::int64_t>> current_bytes_;

  // Record the alloc-time NVTX path for this pointer.
  // Called on the allocating thread — mutex-free NVTX read is safe.
  // The mutex protects the shared map from concurrent alloc/dealloc threads.
  auto record_allocation(void* ptr) noexcept -> std::string
  {
    std::string path = "";
    if (ptr != nullptr) {
      path = raft::common::nvtx::thread_local_nvtx_full_path();
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

  // Enqueue an event. This is called on the allocating/deallocating thread
  void emit(std::string nvtx_full_range,
            std::int64_t signed_bytes,
            std::chrono::steady_clock::time_point const& timestamp)
  {
    auto [name, depth] = raft::common::nvtx::thread_local_inner_range_and_depth();
    allocation_event event{
      .timestamp = timestamp,
      .source_id = source_id_,
      .current_bytes =
        current_bytes_->fetch_add(signed_bytes, std::memory_order_relaxed) + signed_bytes,
      .delta_bytes      = signed_bytes,
      .nvtx_depth       = depth,
      .nvtx_inner_range = std::move(name),
      .nvtx_full_range  = std::move(nvtx_full_range),
    };
    queue_->push(std::move(event));
  }

 public:
  recording_adaptor(Upstream upstream, std::shared_ptr<allocation_event_queue> queue, int source_id)
    : upstream_(std::move(upstream)),
      queue_(std::move(queue)),
      alloc_map_(std::make_shared<address_range_map>()),
      source_id_(source_id),
      current_bytes_{std::make_shared<std::atomic<std::int64_t>>(0)}
  {
    RAFT_EXPECTS(queue_ != nullptr, "event queue must be initialized");
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    void* ptr                                       = upstream_.allocate_sync(bytes, alignment);
    emit(record_allocation(ptr), static_cast<std::int64_t>(bytes), timestamp);
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    upstream_.deallocate_sync(ptr, bytes, alignment);
    emit(forget_allocation(ptr), -static_cast<std::int64_t>(bytes), timestamp);
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    void* ptr                                       = upstream_.allocate(stream, bytes, alignment);
    emit(record_allocation(ptr), static_cast<std::int64_t>(bytes), timestamp);
    return ptr;
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    std::chrono::steady_clock::time_point timestamp = std::chrono::steady_clock::now();
    upstream_.deallocate(stream, ptr, bytes, alignment);
    emit(forget_allocation(ptr), -static_cast<std::int64_t>(bytes), timestamp);
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
