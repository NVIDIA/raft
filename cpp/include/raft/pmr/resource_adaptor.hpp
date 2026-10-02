/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <cuda/memory_resource>

#include <cstddef>
#include <memory>
#include <memory_resource>

namespace raft {
namespace pmr {

/**
 * @brief Adapter wrapping std::pmr::memory_resource to satisfy
 *        cuda::mr::synchronous_resource_with<cuda::mr::host_accessible>.
 *
 * Supports both owning and non-owning semantics via a shared_ptr:
 *  - From raw pointer or reference: non-owning (no-op deleter).
 *  - From shared_ptr or unique_ptr: owning (shared ownership or exclusive transfer).
 *
 * Satisfies cuda::mr::synchronous_resource and has_property for host_accessible,
 * so it can be used directly with raft::mr::synchronous_host_resource_ref.
 */
class resource_adaptor {
 public:
  explicit resource_adaptor(std::pmr::memory_resource* upstream) noexcept
    : upstream_(upstream, [](std::pmr::memory_resource*) {})
  {
  }

  resource_adaptor(std::pmr::memory_resource& upstream) noexcept  // NOLINT
    : upstream_(&upstream, [](std::pmr::memory_resource*) {})
  {
  }

  explicit resource_adaptor(std::shared_ptr<std::pmr::memory_resource> upstream) noexcept
    : upstream_(std::move(upstream))
  {
  }

  explicit resource_adaptor(std::unique_ptr<std::pmr::memory_resource> upstream) noexcept
    : upstream_(std::move(upstream))
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    return upstream_->allocate(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_->deallocate(ptr, bytes, alignment);
  }

  [[nodiscard]] bool operator==(resource_adaptor const& other) const noexcept
  {
    return upstream_->is_equal(*other.upstream_);
  }

  [[nodiscard]] bool operator!=(resource_adaptor const& other) const noexcept
  {
    return !(*this == other);
  }

  friend void get_property(resource_adaptor const&, cuda::mr::host_accessible) noexcept {}

  [[nodiscard]] std::pmr::memory_resource* upstream() const noexcept { return upstream_.get(); }

 private:
  std::shared_ptr<std::pmr::memory_resource> upstream_;
};

static_assert(cuda::mr::synchronous_resource_with<resource_adaptor, cuda::mr::host_accessible>,
              "resource_adaptor must satisfy synchronous_resource_with<host_accessible> for "
              "raft::mr::synchronous_host_resource_ref consumption");

}  // namespace pmr
}  // namespace raft
