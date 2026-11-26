/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>

#include <cstdint>

namespace RAFT_EXPORT raft {

template <typename AccessorPolicy>
using host_accessor = host_device_accessor<AccessorPolicy, memory_type::host>;

/**
 * @brief std::experimental::mdspan with host tag to avoid accessing incorrect memory location.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = cuda::std::default_accessor<ElementType>>
using host_mdspan = mdspan<ElementType, Extents, LayoutPolicy, host_accessor<AccessorPolicy>>;

template <typename T, bool B>
struct is_host_mdspan : std::false_type {};
template <typename T>
struct is_host_mdspan<T, true> : std::bool_constant<T::accessor_type::is_host_accessible> {};

/**
 * @brief Boolean to determine if template type T is either raft::host_mdspan or a derived type
 */
template <typename T>
using is_host_mdspan_t = is_host_mdspan<T, is_mdspan_v<T>>;

template <typename T>
using is_input_host_mdspan_t = is_host_mdspan<T, is_input_mdspan_v<T>>;

template <typename T>
using is_output_host_mdspan_t = is_host_mdspan<T, is_output_mdspan_v<T>>;

/**
 * @brief Boolean to determine if variadic template types Tn are either raft::host_mdspan or a
 * derived type
 */
template <typename... Tn>
inline constexpr bool is_host_mdspan_v = std::conjunction_v<is_host_mdspan_t<Tn>...>;

template <typename... Tn>
inline constexpr bool is_input_host_mdspan_v = std::conjunction_v<is_input_host_mdspan_t<Tn>...>;

template <typename... Tn>
inline constexpr bool is_output_host_mdspan_v = std::conjunction_v<is_output_host_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_host_mdspan = std::enable_if_t<is_input_mdspan_v<Tn...>>;

template <typename... Tn>
using enable_if_input_host_mdspan = std::enable_if_t<is_input_host_mdspan_v<Tn...>>;

template <typename... Tn>
using enable_if_output_host_mdspan = std::enable_if_t<is_output_host_mdspan_v<Tn...>>;

/**
 * @brief Shorthand for 0-dim host mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using host_scalar_view = host_mdspan<ElementType, scalar_extent<IndexType>>;

/**
 * @brief Shorthand for 1-dim host mdspan.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_vector_view = host_mdspan<ElementType, vector_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous host matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_matrix_view = host_mdspan<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for 128 byte aligned host matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy must be of type layout_{left/right}_padded
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_right_padded<ElementType>,
          typename              = enable_if_layout_padded<ElementType, LayoutPolicy>>
using host_aligned_matrix_view =
  host_mdspan<ElementType,
              matrix_extent<IndexType>,
              LayoutPolicy,
              cuda::std::aligned_accessor<ElementType, detail::alignment::value>>;

/**
 * @brief Create a 2-dim 128 byte aligned mdspan instance for host pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy must be of type layout_{left/right}_padded
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on host to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_right_padded<ElementType>>
auto constexpr make_host_aligned_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
{
  using data_handle_type =
    typename cuda::std::aligned_accessor<ElementType, detail::alignment::value>::data_handle_type;

  static_assert(std::is_same<LayoutPolicy, layout_left_padded<ElementType>>::value ||
                std::is_same<LayoutPolicy, layout_right_padded<ElementType>>::value);
  assert(reinterpret_cast<std::uintptr_t>(ptr) ==
         raft::detail::alignTo(reinterpret_cast<std::uintptr_t>(ptr), detail::alignment::value));
  data_handle_type aligned_pointer = ptr;

  matrix_extent<IndexType> extents{n_rows, n_cols};
  return host_aligned_matrix_view<ElementType, IndexType, LayoutPolicy>{aligned_pointer, extents};
}

/**
 * @brief Create a 0-dim (scalar) mdspan instance for host value.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on host to wrap
 */
template <typename ElementType, typename IndexType = std::uint32_t>
auto constexpr make_host_scalar_view(ElementType* ptr)
{
  scalar_extent<IndexType> extents;
  return host_scalar_view<ElementType, IndexType>{ptr, extents};
}

/**
 * @brief Create a 2-dim c-contiguous mdspan instance for host pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on host to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto constexpr make_host_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
{
  matrix_extent<IndexType> extents{n_rows, n_cols};
  return host_matrix_view<ElementType, IndexType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a 2-dim mdspan instance for host pointer with a strided layout
 *        that is restricted to stride 1 in the trailing dimension. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on host to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 * @param[in] stride leading dimension / stride of data
 */
template <typename ElementType, typename IndexType, typename LayoutPolicy = layout_c_contiguous>
auto constexpr make_host_strided_matrix_view(ElementType* ptr,
                                             IndexType n_rows,
                                             IndexType n_cols,
                                             IndexType stride)
{
  constexpr auto is_row_major = std::is_same_v<LayoutPolicy, layout_c_contiguous>;
  constexpr auto is_col_major = std::is_same_v<LayoutPolicy, layout_f_contiguous>;

  static_assert(is_row_major || is_col_major, "Unsupported layout policy for strided matrix view");

  IndexType stride0 = is_row_major ? (stride > 0 ? stride : n_cols) : 1;
  IndexType stride1 = is_row_major ? 1 : (stride > 0 ? stride : n_rows);

  assert(is_row_major ? stride0 >= n_cols : stride1 >= n_rows);
  matrix_extent<IndexType> extents{n_rows, n_cols};

  auto layout = make_strided_layout(extents, cuda::std::array<IndexType, 2>{stride0, stride1});
  return host_matrix_view<ElementType, IndexType, layout_stride>{ptr, layout};
}

/**
 * @brief Create a 1-dim mdspan instance for host pointer.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on host to wrap
 * @param[in] n number of elements in pointer
 * @return raft::host_vector_view
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto constexpr make_host_vector_view(ElementType* ptr, IndexType n)
{
  return host_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, n};
}
}  // namespace RAFT_EXPORT raft
