/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace raft::mr {

namespace detail {

/**
 * @brief Lock-free atomic counter that tracks current and peak allocation bytes.
 */
struct dry_run_memory_counter {
  void record_allocate(std::size_t bytes) noexcept
  {
    auto new_total    = allocated_bytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto current_peak = peak_bytes_.load(std::memory_order_relaxed);
    while (new_total > current_peak &&
           !peak_bytes_.compare_exchange_weak(
             current_peak, new_total, std::memory_order_relaxed, std::memory_order_relaxed)) {}
  }

  void record_deallocate(std::size_t bytes) noexcept
  {
    allocated_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
  }

  [[nodiscard]] auto get_allocated_bytes() const noexcept -> std::size_t
  {
    return allocated_bytes_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] auto get_peak_bytes() const noexcept -> std::size_t
  {
    return peak_bytes_.load(std::memory_order_relaxed);
  }

 private:
  std::atomic<std::size_t> allocated_bytes_{0};
  std::atomic<std::size_t> peak_bytes_{0};
};

/**
 * @brief Minimal RAII container for a single allocation from a memory resource.
 *
 * Stripped-down RAII wrapper: just allocate / deallocate / data().
 * Two constructor overloads cover sync and async resources:
 *   - Sync: (MR, size, alignment) -- calls allocate_sync, destructor calls deallocate_sync
 *   - Async: (MR, stream, size, alignment) -- calls allocate, destructor calls deallocate
 *
 * @tparam MR  Memory resource type, stored by value (use a ref type for non-owning).
 */
template <typename MR>
class probe_container {
  MR mr_;
  void* ptr_;
  std::size_t size_;
  std::size_t alignment_;

 public:
  template <typename M = MR, std::enable_if_t<cuda::mr::synchronous_resource<M>, int> = 0>
  probe_container(MR mr, std::size_t size, std::size_t alignment = alignof(std::max_align_t))
    : mr_(std::move(mr)), ptr_(nullptr), size_(size), alignment_(alignment)
  {
    ptr_ = mr_.allocate_sync(size_, alignment_);
  }

  template <typename M = MR, std::enable_if_t<cuda::mr::resource<M>, int> = 0>
  probe_container(MR mr,
                  cuda::stream_ref stream,
                  std::size_t size,
                  std::size_t alignment = alignof(std::max_align_t))
    : mr_(std::move(mr)), ptr_(nullptr), size_(size), alignment_(alignment)
  {
    ptr_ = mr_.allocate(stream, size_, alignment_);
  }

  ~probe_container()
  {
    if (ptr_ == nullptr) return;
    if constexpr (cuda::mr::resource<MR>) {
      mr_.deallocate(cuda::stream_ref{cudaStreamPerThread}, ptr_, size_, alignment_);
    } else {
      mr_.deallocate_sync(ptr_, size_, alignment_);
    }
  }

  probe_container(probe_container const&)            = delete;
  probe_container& operator=(probe_container const&) = delete;
  probe_container(probe_container&&)                 = delete;
  probe_container& operator=(probe_container&&)      = delete;

  [[nodiscard]] auto data() const noexcept -> void* { return ptr_; }
};

}  // namespace detail

static constexpr std::size_t kDryRunProbeSize = 256;

/**
 * @brief Resource adaptor that returns a single probed pointer for every allocation
 *        and tracks peak usage without holding real memory.
 *
 * Modeled after raft::mr::statistics_adaptor: a single template handles host,
 * device, pinned, and managed resources depending on the Upstream type.
 *
 * Properties are forwarded from Upstream via ADL friend get_property, so
 * dry_run_resource<host_resource_ref> satisfies host_accessible,
 * dry_run_resource<host_device_resource_ref> satisfies host + device accessible,
 * and dry_run_resource<rmm::device_async_resource_ref> satisfies device_accessible.
 *
 * @tparam Upstream  Stored by value.  Use a ref type for non-owning semantics.
 */
template <typename Upstream>
class dry_run_resource : public cuda::forward_property<dry_run_resource<Upstream>, Upstream> {
  Upstream upstream_;

  struct shared_state {
    detail::dry_run_memory_counter counter;
    std::once_flag probe_flag;
    std::unique_ptr<detail::probe_container<Upstream>> probe;
  };
  std::shared_ptr<shared_state> state_;

 public:
  template <typename U, std::enable_if_t<std::is_same_v<std::decay_t<U>, Upstream>, int> = 0>
  explicit dry_run_resource(U&& upstream)
    : upstream_(std::forward<U>(upstream)), state_(std::make_shared<shared_state>())
  {
  }

  // NVCC injects __host__ __device__ on std::shared_ptr special members,
  // which makes the *implicit* or *defaulted* special members __host__
  // __device__ too.  That conflicts with Upstream types whose special
  // members are __host__ only (e.g. rmm::device_async_resource_ref).
  // User-defined bodies (not = default) force plain __host__ execution space.
  dry_run_resource(dry_run_resource&& other) noexcept
    : upstream_(std::move(other.upstream_)), state_(std::move(other.state_))
  {
  }
  dry_run_resource(dry_run_resource const& other) : upstream_(other.upstream_), state_(other.state_)
  {
  }
  dry_run_resource& operator=(dry_run_resource&& other) noexcept
  {
    upstream_ = std::move(other.upstream_);
    state_    = std::move(other.state_);
    return *this;
  }
  dry_run_resource& operator=(dry_run_resource const& other)
  {
    upstream_ = other.upstream_;
    state_    = other.state_;
    return *this;
  }

  [[nodiscard]] auto get_counter() const noexcept -> std::shared_ptr<detail::dry_run_memory_counter>
  {
    return {state_, &state_->counter};
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    std::call_once(state_->probe_flag, [&] {
      state_->probe =
        std::make_unique<detail::probe_container<Upstream>>(upstream_, kDryRunProbeSize, alignment);
    });
    state_->counter.record_allocate(bytes);
    return state_->probe->data();
  }

  void deallocate_sync(void*, std::size_t bytes, std::size_t = alignof(std::max_align_t)) noexcept
  {
    state_->counter.record_deallocate(bytes);
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    std::call_once(state_->probe_flag, [&] {
      state_->probe = std::make_unique<detail::probe_container<Upstream>>(
        upstream_, stream, kDryRunProbeSize, alignment);
    });
    state_->counter.record_allocate(bytes);
    return state_->probe->data();
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void deallocate(cuda::stream_ref,
                  void*,
                  std::size_t bytes,
                  std::size_t = alignof(std::max_align_t)) noexcept
  {
    state_->counter.record_deallocate(bytes);
  }

  [[nodiscard]] bool operator==(dry_run_resource const& other) const noexcept
  {
    return upstream_ == other.upstream_;
  }

  [[nodiscard]] auto upstream_resource() noexcept -> Upstream& { return upstream_; }
  [[nodiscard]] auto upstream_resource() const noexcept -> Upstream const& { return upstream_; }
};

template <typename Upstream>
dry_run_resource(Upstream) -> dry_run_resource<Upstream>;

}  // namespace raft::mr
