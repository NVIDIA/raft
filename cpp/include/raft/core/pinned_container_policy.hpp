/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/core/detail/macros.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>

#ifndef RAFT_DISABLE_CUDA
#include <raft/core/resource/pinned_memory_resource.hpp>
#else
#include <raft/core/detail/fail_container_policy.hpp>
#endif

namespace RAFT_EXPORT raft {
#ifndef RAFT_DISABLE_CUDA
/**
 * @brief A container policy for pinned mdarray.
 */
template <typename ElementType>
struct pinned_container_policy {
  using element_type = ElementType;
  using container_type =
    host_container<element_type, raft::mr::synchronous_host_device_resource_ref>;
  using pointer               = typename container_type::pointer;
  using const_pointer         = typename container_type::const_pointer;
  using reference             = typename container_type::reference;
  using const_reference       = typename container_type::const_reference;
  using accessor_policy       = cuda::std::default_accessor<element_type>;
  using const_accessor_policy = cuda::std::default_accessor<element_type const>;

  auto create(raft::resources const& res, size_t n) -> container_type
  {
    return container_type(n, raft::resource::get_pinned_memory_resource_ref(res));
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
#else
template <typename ElementType>
using pinned_container_policy = detail::fail_container_policy<ElementType>;
#endif
}  // namespace RAFT_EXPORT raft
