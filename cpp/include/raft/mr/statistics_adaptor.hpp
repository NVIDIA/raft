/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace raft {
namespace mr {

/**
 * @brief Atomic allocation counters readable from any thread without locking.
 */
struct resource_stats {
  std::atomic<std::int64_t> bytes_current{0};
  std::atomic<std::int64_t> bytes_peak{0};
  std::atomic<std::int64_t> bytes_total_allocated{0};
  std::atomic<std::int64_t> bytes_total_deallocated{0};
  std::atomic<std::int64_t> num_allocations{0};
  std::atomic<std::int64_t> num_deallocations{0};

  void record_allocate(std::int64_t bytes)
  {
    auto cur  = bytes_current.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto peak = bytes_peak.load(std::memory_order_relaxed);
    while (cur > peak && !bytes_peak.compare_exchange_weak(peak, cur, std::memory_order_relaxed)) {}
    bytes_total_allocated.fetch_add(bytes, std::memory_order_relaxed);
    num_allocations.fetch_add(1, std::memory_order_relaxed);
  }

  void record_deallocate(std::int64_t bytes)
  {
    bytes_current.fetch_sub(bytes, std::memory_order_relaxed);
    bytes_total_deallocated.fetch_add(bytes, std::memory_order_relaxed);
    num_deallocations.fetch_add(1, std::memory_order_relaxed);
  }
};

/**
 * @brief Resource adaptor that maintains atomic allocation statistics.
 *
 * The allocations/deallocations are recorded after the fact - not recorded on exception.
 *
 * Forwards all allocations/deallocations to the upstream resource and updates
 * a shared resource_stats object.  The stats are co-owned via shared_ptr so
 * they survive type-erasure of this adaptor.
 *
 * @note Make sure to call get_stats() before type-erasing the adaptor to get the statistics.
 *
 * @tparam Upstream  Stored by value.  Use a concrete resource type for owning
 *                   semantics, or a ref type (e.g. raft::mr::host_resource_ref)
 *                   for non-owning semantics.
 */
template <typename Upstream>
class statistics_adaptor : public cuda::forward_property<statistics_adaptor<Upstream>, Upstream> {
  Upstream upstream_;
  std::shared_ptr<resource_stats> stats_;

 public:
  // Prevent recursive concept satisfaction when Upstream is a __basic_any type (GCC C++20).
  template <typename U, std::enable_if_t<std::is_same_v<std::decay_t<U>, Upstream>, int> = 0>
  explicit statistics_adaptor(U&& upstream)
    : upstream_(std::forward<U>(upstream)), stats_(std::make_shared<resource_stats>())
  {
  }

  // NVCC injects __host__ __device__ on std::shared_ptr special members,
  // which makes the *implicit* or *defaulted* special members __host__
  // __device__ too.  That conflicts with Upstream types whose special
  // members are __host__ only (e.g. rmm::device_async_resource_ref).
  // User-defined bodies (not = default) force plain __host__ execution space.
  statistics_adaptor(statistics_adaptor&& other) noexcept
    : upstream_(std::move(other.upstream_)), stats_(std::move(other.stats_))
  {
  }
  statistics_adaptor(statistics_adaptor const& other)
    : upstream_(other.upstream_), stats_(other.stats_)
  {
  }
  statistics_adaptor& operator=(statistics_adaptor&& other) noexcept
  {
    upstream_ = std::move(other.upstream_);
    stats_    = std::move(other.stats_);
    return *this;
  }
  statistics_adaptor& operator=(statistics_adaptor const& other)
  {
    upstream_ = other.upstream_;
    stats_    = other.stats_;
    return *this;
  }

  /**
   * @brief Get the shared resource_stats object.
   *
   * @return shared pointer to the resource_stats object
   */
  [[nodiscard]] auto get_stats() const noexcept -> std::shared_ptr<resource_stats>
  {
    return stats_;
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    void* ptr = upstream_.allocate_sync(bytes, alignment);
    stats_->record_allocate(static_cast<std::int64_t>(bytes));
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate_sync(ptr, bytes, alignment);
    stats_->record_deallocate(static_cast<std::int64_t>(bytes));
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    void* ptr = upstream_.allocate(stream, bytes, alignment);
    stats_->record_allocate(static_cast<std::int64_t>(bytes));
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
  }

  [[nodiscard]] bool operator==(statistics_adaptor const& other) const noexcept
  {
    return upstream_ == other.upstream_;
  }

  [[nodiscard]] auto upstream_resource() noexcept -> Upstream& { return upstream_; }
  [[nodiscard]] auto upstream_resource() const noexcept -> Upstream const& { return upstream_; }
};

template <typename Upstream>
statistics_adaptor(Upstream) -> statistics_adaptor<Upstream>;

}  // namespace mr
}  // namespace raft
