/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/core/detail/macros.hpp>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/host_container_policy.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>

namespace RAFT_EXPORT raft {
/**
 * @brief A container policy for managed mdarray.
 *
 * Uses synchronous allocation (allocate_sync) via cuda::mr::host_device_resource_ref.
 * No stream; managed memory is accessible from host and device.
 */
template <typename ElementType>
struct managed_container_policy {
  using element_type          = ElementType;
  using container_type        = host_container<element_type, cuda::mr::host_device_resource_ref>;
  using pointer               = typename container_type::pointer;
  using const_pointer         = typename container_type::const_pointer;
  using reference             = typename container_type::reference;
  using const_reference       = typename container_type::const_reference;
  using accessor_policy       = cuda::std::default_accessor<element_type>;
  using const_accessor_policy = cuda::std::default_accessor<element_type const>;

  auto create(raft::resources const& res, size_t n) -> container_type
  {
    return container_type(n, raft::resource::get_managed_memory_resource_ref(res));
  }

  [[nodiscard]] constexpr auto access(container_type& c, size_t n) const noexcept -> reference
  {
    return c[n];
  }
  [[nodiscard]] constexpr auto access(container_type const& c, size_t n) const noexcept
    -> const_reference
  {
    return c[n];
  }

  [[nodiscard]] auto make_accessor_policy() noexcept { return accessor_policy{}; }
  [[nodiscard]] auto make_accessor_policy() const noexcept { return const_accessor_policy{}; }
};

}  // namespace RAFT_EXPORT raft
#else
#include <raft/core/detail/fail_container_policy.hpp>
namespace RAFT_EXPORT raft {

// Provide placeholders that will allow CPU-GPU interoperable codebases to
// compile in non-CUDA mode but which will throw exceptions at runtime on any
// attempt to touch device data

template <typename ElementType>
using managed_container_policy = detail::fail_container_policy<ElementType>;

}  // namespace RAFT_EXPORT raft
#endif
