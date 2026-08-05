/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdarray.hpp>

#include <cstdint>
#include <memory>

namespace raft {

/**
 * @brief A copyable container that wraps an inner (move-only) container in a shared_ptr.
 *
 * All type aliases (pointer, reference, etc.) are forwarded from the inner container,
 * so the shared_container is a drop-in replacement that adds copy semantics via
 * reference-counted shared ownership.
 */
template <typename InnerContainer>
class shared_container {
  std::shared_ptr<InnerContainer> owner_;

 public:
  using value_type      = typename InnerContainer::value_type;
  using size_type       = typename InnerContainer::size_type;
  using reference       = typename InnerContainer::reference;
  using const_reference = typename InnerContainer::const_reference;
  using pointer         = typename InnerContainer::pointer;
  using const_pointer   = typename InnerContainer::const_pointer;
  using iterator        = typename InnerContainer::iterator;
  using const_iterator  = typename InnerContainer::const_iterator;

  shared_container() = default;

  explicit shared_container(InnerContainer&& c)
    : owner_(std::make_shared<InnerContainer>(std::move(c)))
  {
  }

  shared_container(shared_container const&)            = default;
  shared_container(shared_container&&)                 = default;
  shared_container& operator=(shared_container const&) = default;
  shared_container& operator=(shared_container&&)      = default;

  [[nodiscard]] auto data() noexcept -> pointer { return owner_->data(); }
  [[nodiscard]] auto data() const noexcept -> const_pointer { return owner_->data(); }

  template <typename Index>
  auto operator[](Index i) noexcept -> reference
  {
    return (*owner_)[i];
  }
  template <typename Index>
  auto operator[](Index i) const noexcept -> const_reference
  {
    return (*owner_)[i];
  }
};

/**
 * @brief A container policy that wraps any inner container policy, replacing its
 * container_type with shared_container<inner_container_type>.
 *
 * All other type aliases (pointer, reference, accessor_policy, etc.) are forwarded
 * from the inner policy, preserving type identity with the corresponding non-shared mdarray.
 */
template <typename InnerPolicy>
class shared_container_policy {
  InnerPolicy inner_;

 public:
  using element_type          = typename InnerPolicy::element_type;
  using container_type        = shared_container<typename InnerPolicy::container_type>;
  using pointer               = typename InnerPolicy::pointer;
  using const_pointer         = typename InnerPolicy::const_pointer;
  using reference             = typename InnerPolicy::reference;
  using const_reference       = typename InnerPolicy::const_reference;
  using accessor_policy       = typename InnerPolicy::accessor_policy;
  using const_accessor_policy = typename InnerPolicy::const_accessor_policy;

  shared_container_policy() = default;
  explicit shared_container_policy(InnerPolicy inner) : inner_(std::move(inner)) {}

  auto create(raft::resources const& res, size_t n) -> container_type
  {
    return container_type(inner_.create(res, n));
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

  [[nodiscard]] auto make_accessor_policy() noexcept { return inner_.make_accessor_policy(); }
  [[nodiscard]] auto make_accessor_policy() const noexcept { return inner_.make_accessor_policy(); }
};

/**
 * @defgroup shared_mdarray_aliases Shared mdarray type aliases
 * @{
 */

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = device_container_policy<ElementType>>
using shared_device_mdarray = mdarray<ElementType,
                                      Extents,
                                      LayoutPolicy,
                                      device_accessor<shared_container_policy<ContainerPolicy>>>;

template <typename ElementType, typename IndexType = std::uint32_t>
using shared_device_scalar = shared_device_mdarray<ElementType, scalar_extent<IndexType>>;

template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using shared_device_vector =
  shared_device_mdarray<ElementType, vector_extent<IndexType>, LayoutPolicy>;

template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using shared_device_matrix =
  shared_device_mdarray<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = host_container_policy<ElementType>>
using shared_host_mdarray = mdarray<ElementType,
                                    Extents,
                                    LayoutPolicy,
                                    host_accessor<shared_container_policy<ContainerPolicy>>>;

template <typename ElementType, typename IndexType = std::uint32_t>
using shared_host_scalar = shared_host_mdarray<ElementType, scalar_extent<IndexType>>;

template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using shared_host_vector = shared_host_mdarray<ElementType, vector_extent<IndexType>, LayoutPolicy>;

template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using shared_host_matrix = shared_host_mdarray<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/** @} */

/**
 * @defgroup shared_mdarray_factories Shared mdarray factory functions
 * @{
 */

/**
 * @brief Move a regular mdarray into a shared (reference-counted, copyable) mdarray.
 *
 * This is a zero-copy operation: the underlying storage is moved into a shared_ptr.
 * The source mdarray is left in a moved-from state.
 *
 * @tparam ElementType the data type of the elements
 * @tparam Extents defines the shape
 * @tparam LayoutPolicy policy for indexing strides and layout ordering
 * @tparam ContainerPolicy storage and accessor policy
 * @param src the source mdarray (consumed via move)
 * @return a shared mdarray with the same data, shape, and layout
 */
template <typename ElementType, typename Extents, typename LayoutPolicy, typename ContainerPolicy>
auto make_shared_mdarray(mdarray<ElementType, Extents, LayoutPolicy, ContainerPolicy>&& src)
{
  using inner_policy_type   = typename ContainerPolicy::accessor_type;
  using shared_policy_type  = shared_container_policy<inner_policy_type>;
  using shared_cp_type      = host_device_accessor<shared_policy_type, ContainerPolicy::mem_type>;
  using shared_mdarray_type = mdarray<ElementType, Extents, LayoutPolicy, shared_cp_type>;

  using shared_container_t = typename shared_mdarray_type::container_type;

  auto mapping = src.mapping();
  shared_container_t sc{std::move(src).release_container()};
  return shared_mdarray_type(mapping, std::move(sc), shared_cp_type{});
}

/**
 * @brief Create a shared device mdarray.
 * @tparam ElementType the data type of the elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param handle raft::resources
 * @param exts dimensionality of the array (series of integers)
 * @return raft::shared_device_mdarray
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_shared_device_mdarray(raft::resources const& handle, extents<IndexType, Extents...> exts)
{
  using mdarray_t = shared_device_mdarray<ElementType, decltype(exts), LayoutPolicy>;
  typename mdarray_t::mapping_type layout{exts};
  typename mdarray_t::container_policy_type policy{};
  return mdarray_t{handle, layout, policy};
}

/**
 * @brief Create a shared host mdarray.
 * @tparam ElementType the data type of the elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param handle raft::resources
 * @param exts dimensionality of the array (series of integers)
 * @return raft::shared_host_mdarray
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_shared_host_mdarray(raft::resources const& handle, extents<IndexType, Extents...> exts)
{
  using mdarray_t = shared_host_mdarray<ElementType, decltype(exts), LayoutPolicy>;
  typename mdarray_t::mapping_type layout{exts};
  typename mdarray_t::container_policy_type policy{};
  return mdarray_t{handle, layout, policy};
}

/**
 * @brief Create a 2-dim shared device matrix.
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_shared_device_matrix(raft::resources const& handle, IndexType n_rows, IndexType n_cols)
{
  return make_shared_device_mdarray<ElementType, IndexType, LayoutPolicy>(
    handle, make_extents<IndexType>(n_rows, n_cols));
}

/**
 * @brief Create a 1-dim shared device vector.
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_shared_device_vector(raft::resources const& handle, IndexType n)
{
  return make_shared_device_mdarray<ElementType, IndexType, LayoutPolicy>(
    handle, make_extents<IndexType>(n));
}

/**
 * @brief Create a 2-dim shared host matrix.
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_shared_host_matrix(raft::resources const& handle, IndexType n_rows, IndexType n_cols)
{
  return make_shared_host_mdarray<ElementType, IndexType, LayoutPolicy>(
    handle, make_extents<IndexType>(n_rows, n_cols));
}

/**
 * @brief Create a 1-dim shared host vector.
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_shared_host_vector(raft::resources const& handle, IndexType n)
{
  return make_shared_host_mdarray<ElementType, IndexType, LayoutPolicy>(handle,
                                                                        make_extents<IndexType>(n));
}

/** @} */

}  // namespace raft
