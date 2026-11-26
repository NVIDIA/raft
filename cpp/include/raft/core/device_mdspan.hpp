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
using device_accessor = host_device_accessor<AccessorPolicy, memory_type::device>;

/**
 * @brief std::experimental::mdspan with device tag to avoid accessing incorrect memory location.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = cuda::std::default_accessor<ElementType>>
using device_mdspan = mdspan<ElementType, Extents, LayoutPolicy, device_accessor<AccessorPolicy>>;

template <typename T, bool B>
struct is_device_mdspan : std::false_type {};
template <typename T>
struct is_device_mdspan<T, true> : std::bool_constant<T::accessor_type::is_device_accessible> {};

/**
 * @brief Boolean to determine if template type T is either raft::device_mdspan or a derived type
 */
template <typename T>
using is_device_mdspan_t = is_device_mdspan<T, is_mdspan_v<T>>;

template <typename T>
using is_input_device_mdspan_t = is_device_mdspan<T, is_input_mdspan_v<T>>;

template <typename T>
using is_output_device_mdspan_t = is_device_mdspan<T, is_output_mdspan_v<T>>;

/**
 * @brief Boolean to determine if variadic template types Tn are either raft::device_mdspan or a
 * derived type
 */
template <typename... Tn>
inline constexpr bool is_device_mdspan_v = std::conjunction_v<is_device_mdspan_t<Tn>...>;

template <typename... Tn>
inline constexpr bool is_input_device_mdspan_v =
  std::conjunction_v<is_input_device_mdspan_t<Tn>...>;

template <typename... Tn>
inline constexpr bool is_output_device_mdspan_v =
  std::conjunction_v<is_output_device_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_device_mdspan = std::enable_if_t<is_device_mdspan_v<Tn...>>;

template <typename... Tn>
using enable_if_input_device_mdspan = std::enable_if_t<is_input_device_mdspan_v<Tn...>>;

template <typename... Tn>
using enable_if_output_device_mdspan = std::enable_if_t<is_output_device_mdspan_v<Tn...>>;

/**
 * @brief Shorthand for 0-dim host mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using device_scalar_view = device_mdspan<ElementType, scalar_extent<IndexType>>;

/**
 * @brief Shorthand for 1-dim device mdspan.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_vector_view = device_mdspan<ElementType, vector_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous device matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_matrix_view = device_mdspan<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for 128 byte aligned device matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy must be of type layout_{left/right}_padded
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_right_padded<ElementType>,
          typename              = enable_if_layout_padded<ElementType, LayoutPolicy>>
using device_aligned_matrix_view =
  device_mdspan<ElementType,
                matrix_extent<IndexType>,
                LayoutPolicy,
                cuda::std::aligned_accessor<ElementType, detail::alignment::value>>;

/**
 * @brief Create a 2-dim 128 byte aligned mdspan instance for device pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy must be of type layout_{left/right}_padded
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on device to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_right_padded<ElementType>>
auto constexpr make_device_aligned_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
{
  using data_handle_type =
    typename cuda::std::aligned_accessor<ElementType, detail::alignment::value>::data_handle_type;
  static_assert(std::is_same<LayoutPolicy, layout_left_padded<ElementType>>::value ||
                std::is_same<LayoutPolicy, layout_right_padded<ElementType>>::value);
  assert(reinterpret_cast<std::uintptr_t>(ptr) ==
         raft::detail::alignTo(reinterpret_cast<std::uintptr_t>(ptr), detail::alignment::value));

  data_handle_type aligned_pointer = ptr;

  matrix_extent<IndexType> extents{n_rows, n_cols};
  return device_aligned_matrix_view<ElementType, IndexType, LayoutPolicy>{aligned_pointer, extents};
}

/**
 * @brief Create a 0-dim (scalar) mdspan instance for device value.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on device to wrap
 */
template <typename ElementType, typename IndexType = std::uint32_t>
auto constexpr make_device_scalar_view(ElementType* ptr)
{
  scalar_extent<IndexType> extents;
  return device_scalar_view<ElementType, IndexType>{ptr, extents};
}

/**
 * @brief Create a 2-dim c-contiguous mdspan instance for device pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on device to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto constexpr make_device_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
{
  matrix_extent<IndexType> extents{n_rows, n_cols};
  return device_matrix_view<ElementType, IndexType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a 2-dim mdspan instance for device pointer with a strided layout
 *        that is restricted to stride 1 in the trailing dimension. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on device to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 * @param[in] stride leading dimension / stride of data
 */
template <typename ElementType, typename IndexType, typename LayoutPolicy = layout_c_contiguous>
auto constexpr make_device_strided_matrix_view(ElementType* ptr,
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
  return device_matrix_view<ElementType, IndexType, layout_stride>{ptr, layout};
}

/**
 * @brief Create a 1-dim mdspan instance for device pointer.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on device to wrap
 * @param[in] n number of elements in pointer
 * @return raft::device_vector_view
 */
template <typename ElementType, typename IndexType, typename LayoutPolicy = layout_c_contiguous>
auto constexpr make_device_vector_view(ElementType* ptr, IndexType n)
{
  return device_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, n};
}

/**
 * @brief Create a 1-dim mdspan instance for device pointer.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on device to wrap
 * @param[in] mapping The layout mapping to use for this vector
 * @return raft::device_vector_view
 */
template <typename ElementType, typename IndexType, typename LayoutPolicy = layout_c_contiguous>
auto constexpr make_device_vector_view(
  ElementType* ptr,
  const typename LayoutPolicy::template mapping<vector_extent<IndexType>>& mapping)
{
  return device_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, mapping};
}

/**
 * @brief Construct a strided vector layout mapping
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/device_mdspan.hpp>
 *
 *  int n_elements = 10;
 *  int stride = 10;
 *  auto vector = raft::make_device_vector_view(vector_ptr,
 * raft::make_vector_strided_layout(n_elements, stride));
 * @endcode
 *
 * @tparam IndexType the index type of the extents
 * @param[in] n the number of elements in the vector
 * @param[in] stride the stride between elements in the vector
 */
template <typename IndexType>
auto make_vector_strided_layout(IndexType n, IndexType stride)
{
  return make_strided_layout(vector_extent<IndexType>{n}, cuda::std::array<IndexType, 1>{stride});
}
}  // namespace RAFT_EXPORT raft
