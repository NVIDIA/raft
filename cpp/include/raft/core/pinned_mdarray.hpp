/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/pinned_container_policy.hpp>
#include <raft/core/pinned_mdspan.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

#include <cstdint>

namespace RAFT_EXPORT raft {

/**
 * @brief mdarray with pinned container policy
 * @tparam ElementType the data type of the elements
 * @tparam Extents defines the shape
 * @tparam LayoutPolicy policy for indexing strides and layout ordering
 * @tparam ContainerPolicy storage and accessor policy
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = pinned_container_policy<ElementType>>
using pinned_mdarray =
  mdarray<ElementType, Extents, LayoutPolicy, pinned_accessor<ContainerPolicy>>;

/**
 * @brief Shorthand for 0-dim host mdarray (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using pinned_scalar = pinned_mdarray<ElementType, scalar_extent<IndexType>>;

/**
 * @brief Shorthand for 1-dim pinned mdarray.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using pinned_vector = pinned_mdarray<ElementType, vector_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous pinned matrix.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using pinned_matrix = pinned_mdarray<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Create a pinned mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param handle raft::resources
 * @param exts dimensionality of the array (series of integers)
 * @return raft::pinned_mdarray
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_pinned_mdarray(raft::resources const& handle, extents<IndexType, Extents...> exts)
{
  using mdarray_t = pinned_mdarray<ElementType, decltype(exts), LayoutPolicy>;

  typename mdarray_t::mapping_type layout{exts};
  typename mdarray_t::container_policy_type policy{};

  return mdarray_t{handle, layout, policy};
}

/**
 * @brief Create a 2-dim c-contiguous pinned mdarray.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] n_rows number or rows in matrix
 * @param[in] n_cols number of columns in matrix
 * @return raft::pinned_matrix
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_pinned_matrix(raft::resources const& handle, IndexType n_rows, IndexType n_cols)
{
  return make_pinned_mdarray<ElementType, IndexType, LayoutPolicy>(
    handle, make_extents<IndexType>(n_rows, n_cols));
}

/**
 * @brief Create a pinned scalar from v.
 *
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 * @param[in] handle raft handle for managing expensive cuda resources
 * @param[in] v scalar to wrap on pinned
 * @return raft::pinned_scalar
 */
template <typename ElementType, typename IndexType = std::uint32_t>
auto make_pinned_scalar(raft::resources const& handle, ElementType const& v)
{
  scalar_extent<IndexType> extents;
  using policy_t = typename pinned_scalar<ElementType>::container_policy_type;
  policy_t policy{};
  auto scalar = pinned_scalar<ElementType>{handle, extents, policy};
  if (!resource::get_dry_run_flag(handle)) { scalar(0) = v; }
  return scalar;
}

/**
 * @brief Create a 1-dim pinned mdarray.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] handle raft handle for managing expensive cuda resources
 * @param[in] n number of elements in vector
 * @return raft::pinned_vector
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_pinned_vector(raft::resources const& handle, IndexType n)
{
  return make_pinned_mdarray<ElementType, IndexType, LayoutPolicy>(handle,
                                                                   make_extents<IndexType>(n));
}

}  // namespace RAFT_EXPORT raft
