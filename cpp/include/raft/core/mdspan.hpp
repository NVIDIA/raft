/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/mdspan_layout_padded.hpp>
#include <raft/core/detail/mdspan_util.cuh>
#include <raft/core/error.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/memory_type.hpp>

#include <cuda/std/mdspan>

#include <algorithm>
#include <cstdint>

namespace RAFT_EXPORT raft {

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = cuda::std::default_accessor<ElementType>>
using mdspan = cuda::std::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

}  // namespace RAFT_EXPORT raft

namespace raft {
namespace detail {

template <typename IntType>
constexpr IntType alignTo(IntType a, IntType b)
{
  return ((a + b - 1) / b) * b;
}

// keeping ByteAlignment as optional to allow testing
template <class ValueType, size_t ByteAlignment = 128>
struct padding {
  static_assert(std::is_same<std::remove_cv_t<ValueType>, ValueType>::value,
                "raft::padding ValueType has to be provided without "
                "const or volatile specifiers.");
  static_assert(ByteAlignment % sizeof(ValueType) == 0 || sizeof(ValueType) % ByteAlignment == 0,
                "raft::padding sizeof(ValueType) has to be multiple or "
                "divider of ByteAlignment.");
  static constexpr size_t value = (ByteAlignment / sizeof(ValueType)) > 1ul
                                    ? (ByteAlignment / sizeof(ValueType))
                                    : 1ul;
};

// alignment fixed to 128 bytes
struct alignment {
  static constexpr size_t value = 128;
};

}  // namespace detail
}  // namespace raft

namespace RAFT_EXPORT raft {
/**
 * @brief Padded layouts that take a padding value directly.
 */
template <size_t PaddingStride = cuda::std::dynamic_extent>
using layout_left_padded_impl = cuda::std::layout_left_padded<PaddingStride>;

template <size_t PaddingStride = cuda::std::dynamic_extent>
using layout_right_padded_impl = cuda::std::layout_right_padded<PaddingStride>;

/**
 * @brief Padded layouts that compute padding from element type.
 */
template <typename ElementType>
using layout_right_padded = cuda::std::layout_right_padded<
  detail::padding<std::remove_cv_t<std::remove_reference_t<ElementType>>>::value>;

template <typename ElementType>
using layout_left_padded = cuda::std::layout_left_padded<
  detail::padding<std::remove_cv_t<std::remove_reference_t<ElementType>>>::value>;

template <typename ElementType, typename LayoutPolicy>
using enable_if_layout_padded =
  std::enable_if_t<std::is_same<LayoutPolicy, layout_left_padded<ElementType>>::value ||
                   std::is_same<LayoutPolicy, layout_right_padded<ElementType>>::value>;

/**
 * Ensure all types listed in the parameter pack `Extents` are integral types.
 * Usage:
 *   put it as the last nameless template parameter of a function:
 *     `typename = ensure_integral_extents<Extents...>`
 */
template <typename... Extents>
using ensure_integral_extents = std::enable_if_t<std::conjunction_v<std::is_integral<Extents>...>>;

/**
 * @brief Template checks and helpers to determine if type T is an std::mdspan
 *         or a derived type
 */

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void __takes_an_mdspan_ptr(mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>*);

template <typename T, typename = void>
struct is_mdspan : std::false_type {};
template <typename T>
struct is_mdspan<T, std::void_t<decltype(__takes_an_mdspan_ptr(std::declval<T*>()))>>
  : std::true_type {};

template <typename T, typename = void>
struct is_input_mdspan : std::false_type {};
template <typename T>
struct is_input_mdspan<T, std::void_t<decltype(__takes_an_mdspan_ptr(std::declval<T*>()))>>
  : std::bool_constant<std::is_const_v<typename T::element_type>> {};

template <typename T, typename = void>
struct is_output_mdspan : std::false_type {};
template <typename T>
struct is_output_mdspan<T, std::void_t<decltype(__takes_an_mdspan_ptr(std::declval<T*>()))>>
  : std::bool_constant<not std::is_const_v<typename T::element_type>> {};

template <typename T>
using is_mdspan_t = is_mdspan<std::remove_const_t<T>>;

template <typename T>
using is_input_mdspan_t = is_input_mdspan<T>;

template <typename T>
using is_output_mdspan_t = is_output_mdspan<T>;

/**
 * @brief Boolean to determine if variadic template types Tn are either
 *          raft::host_mdspan/raft::device_mdspan or their derived types
 */
template <typename... Tn>
inline constexpr bool is_mdspan_v = std::conjunction_v<is_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_mdspan = std::enable_if_t<is_mdspan_v<Tn...>>;

template <typename... Tn>
inline constexpr bool is_input_mdspan_v = std::conjunction_v<is_input_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_input_mdspan = std::enable_if_t<is_input_mdspan_v<Tn...>>;

template <typename... Tn>
inline constexpr bool is_output_mdspan_v = std::conjunction_v<is_output_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_output_mdspan = std::enable_if_t<is_output_mdspan_v<Tn...>>;

// uint division optimization inspired by the CIndexer in cupy.  Division operation is
// slow on both CPU and GPU, especially 64 bit integer.  So here we first try to avoid 64
// bit when the index is smaller, then try to avoid division when it's exp of 2.
template <typename I, typename IndexType, size_t... Extents>
RAFT_INLINE_FUNCTION auto unravel_index_impl(I idx, cuda::std::extents<IndexType, Extents...> shape)
{
  constexpr auto kRank = static_cast<int32_t>(shape.rank());
  std::size_t index[shape.rank()]{0};  // NOLINT
  static_assert(std::is_signed<decltype(kRank)>::value,
                "Don't change the type without changing the for loop.");
  for (int32_t dim = kRank; --dim > 0;) {
    auto s = static_cast<std::remove_const_t<std::remove_reference_t<I>>>(shape.extent(dim));
    if (s & (s - 1)) {
      auto t     = idx / s;
      index[dim] = idx - t * s;
      idx        = t;
    } else {  // exp of 2
      index[dim] = idx & (s - 1);
      idx >>= detail::popc(s - 1);
    }
  }
  index[0] = idx;
  return detail::arr_to_tup(index);
}

/**
 * @brief Create a raft::mdspan
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @tparam is_host_accessible whether the data is accessible on host
 * @tparam is_device_accessible whether the data is accessible on device
 * @param ptr Pointer to the data
 * @param exts dimensionality of the array (series of integers)
 * @return raft::mdspan
 */
template <typename ElementType,
          typename IndexType,
          typename LayoutPolicy     = layout_c_contiguous,
          bool is_host_accessible   = false,
          bool is_device_accessible = true,
          size_t... Extents>
constexpr auto make_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)
{
  using accessor_type = host_device_accessor<
    cuda::std::default_accessor<ElementType>,
    detail::memory_type_from_access<is_host_accessible, is_device_accessible>()>;
  /*using accessor_type = host_device_accessor<cuda::std::default_accessor<ElementType>,
                                             mem_type>; */

  return mdspan<ElementType, decltype(exts), LayoutPolicy, accessor_type>{ptr, exts};
}

/**
 * @brief Create a layout_stride mapping from extents and strides
 * @param[in] extents the dimensionality of the layout
 * @param[in] strides the strides between elements in the layout
 * @return raft::layout_stride::mapping<Extents>
 */
template <typename Extents>
auto make_strided_layout(Extents extents,
                         cuda::std::array<typename Extents::index_type, Extents::rank()> strides)
{
  return layout_stride::mapping<Extents>{extents, strides};
}

/**
 * @brief Create raft::extents to specify dimensionality
 *
 * @tparam IndexType The type of each dimension of the extents
 * @tparam Extents Dimensions (a series of integers)
 * @param exts The desired dimensions
 * @return raft::extents
 */
template <typename IndexType, typename... Extents, typename = ensure_integral_extents<Extents...>>
constexpr auto make_extents(Extents... exts)
{
  return extents<IndexType, ((void)exts, dynamic_extent)...>{exts...};
}

/**
 * @defgroup mdspan_reshape Row- or Col-norm computation
 * @{
 */

/**
 * @brief Flatten raft::mdspan into a 1-dim array view
 *
 * @tparam mdspan_type Expected type raft::host_mdspan or raft::device_mdspan
 * @param mds raft::host_mdspan or raft::device_mdspan object
 * @return raft::host_mdspan or raft::device_mdspan with vector_extent
 *         depending on AccessoryPolicy
 */
template <typename mdspan_type, typename = enable_if_mdspan<mdspan_type>>
auto flatten(mdspan_type mds)
{
  RAFT_EXPECTS(mds.is_exhaustive(), "Input must be contiguous.");

  vector_extent<typename mdspan_type::size_type> ext{mds.size()};

  return cuda::std::mdspan<typename mdspan_type::element_type,
                           decltype(ext),
                           typename mdspan_type::layout_type,
                           typename mdspan_type::accessor_type>(mds.data_handle(), ext);
}

/**
 * @brief Reshape raft::host_mdspan or raft::device_mdspan
 *
 * @tparam mdspan_type Expected type raft::host_mdspan or raft::device_mdspan
 * @tparam IndexType the index type of the extents
 * @tparam Extents raft::extents for dimensions
 * @param mds raft::host_mdspan or raft::device_mdspan object
 * @param new_shape Desired new shape of the input
 * @return raft::host_mdspan or raft::device_mdspan, depending on AccessorPolicy
 */
template <typename mdspan_type,
          typename IndexType,
          size_t... Extents,
          typename = enable_if_mdspan<mdspan_type>>
auto reshape(mdspan_type mds, extents<IndexType, Extents...> new_shape)
{
  RAFT_EXPECTS(mds.is_exhaustive(), "Input must be contiguous.");

  size_t new_size = 1;
  for (size_t i = 0; i < new_shape.rank(); ++i) {
    new_size *= new_shape.extent(i);
  }
  RAFT_EXPECTS(new_size == mds.size(), "Cannot reshape array with size mismatch");

  return cuda::std::mdspan<typename mdspan_type::element_type,
                           decltype(new_shape),
                           typename mdspan_type::layout_type,
                           typename mdspan_type::accessor_type>(mds.data_handle(), new_shape);
}

/* @} */

/**
 * @defgroup mdspan_unravel Unravel mdspan
 * @{
 */

/**
 * \brief Turns linear index into coordinate.  Similar to numpy unravel_index.
 *
 * \code
 *   auto m = make_host_matrix<float>(7, 6);
 *   auto m_v = m.view();
 *   auto coord = unravel_index(2, m.extents(), typename decltype(m)::layout_type{});
 *   std::apply(m_v, coord) = 2;
 * \endcode
 *
 * \param idx    The linear index.
 * \param shape  The shape of the array to use.
 * \param layout Must be `layout_c_contiguous` (row-major) in current implementation.
 *
 * \return A std::tuple that represents the coordinate.
 */
template <typename Idx, typename IndexType, typename LayoutPolicy, size_t... Exts>
RAFT_INLINE_FUNCTION auto unravel_index(Idx idx,
                                        extents<IndexType, Exts...> shape,
                                        LayoutPolicy const& layout)
{
  static_assert(std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(layout)>>,
                               layout_c_contiguous>,
                "Only C layout is supported.");
  static_assert(std::is_integral_v<Idx>, "Index must be integral.");
  auto constexpr kIs64 = sizeof(std::remove_cv_t<std::remove_reference_t<Idx>>) == sizeof(uint64_t);
  if (kIs64 && static_cast<uint64_t>(idx) > cuda::std::numeric_limits<uint32_t>::max()) {
    return unravel_index_impl<uint64_t>(static_cast<uint64_t>(idx), shape);
  } else {
    return unravel_index_impl<uint32_t>(static_cast<uint32_t>(idx), shape);
  }
}

/** @} */

/**
 * @defgroup mdspan_contiguous Whether the strides imply a contiguous layout.
 * @{
 */

/**
 * @brief Whether the strides imply a c-contiguous layout.
 */
template <typename Extents, typename Strides>
[[nodiscard]] auto is_c_contiguous(Extents const& extents, Strides const& strides) -> bool
{
  typename Extents::index_type stride = 1;
  for (auto r = extents.rank(); r > 0; r--) {
    if (stride != strides[r - 1]) { return false; }
    stride *= extents.extent(r - 1);
  }
  return true;
}

/**
 * @brief Whether the strides imply a f-contiguous layout.
 */
template <typename Extents, typename Strides>
[[nodiscard]] auto is_f_contiguous(Extents const& extents, Strides const& strides) -> bool
{
  typename Extents::index_type stride = 1;
  for (typename Extents::rank_type r = 0; r < extents.rank(); r++) {
    if (stride != strides[r]) { return false; }
    stride *= extents.extent(r);
  }
  return true;
}

/** @} */

/**
 * @brief Const accessor specialization for default_accessor
 *
 * @tparam ElementType
 * @param a
 * @return cuda::std::default_accessor<std::add_const_t<ElementType>>
 */
template <class ElementType>
cuda::std::default_accessor<std::add_const_t<ElementType>> accessor_of_const(
  cuda::std::default_accessor<ElementType> a)
{
  return {a};
}

/**
 * @brief Const accessor specialization for host_device_accessor
 *
 * @tparam ElementType the data type of the mdspan elements
 * @tparam MemType the type of memory where the elements are stored.
 * @param a host_device_accessor
 * @return host_device_accessor<cuda::std::default_accessor<std::add_const_t<ElementType>>,
 * MemType>
 */
template <class ElementType, memory_type MemType>
host_device_accessor<cuda::std::default_accessor<std::add_const_t<ElementType>>, MemType>
accessor_of_const(host_device_accessor<cuda::std::default_accessor<ElementType>, MemType> a)
{
  return {a};
}

/**
 * @defgroup mdspan_make_const Convert an mdspan to a const type
 * @{
 */

/**
 * @brief Create a copy of the given mdspan with const element type
 *
 * @tparam ElementType the const-qualified data type of the mdspan elements
 * @tparam Extents raft::extents for dimensions
 * @tparam Layout policy for strides and layout ordering
 * @tparam Accessor Accessor policy for the input and output
 * @param mds raft::mdspan object
 * @return raft::mdspan
 */
template <class ElementType, class Extents, class Layout, class Accessor>
auto make_const_mdspan(mdspan<ElementType, Extents, Layout, Accessor> mds)
{
  auto acc_c = accessor_of_const(mds.accessor());
  return mdspan<std::add_const_t<ElementType>, Extents, Layout, decltype(acc_c)>{
    mds.data_handle(), mds.mapping(), acc_c};
}

/** @} */

}  // namespace RAFT_EXPORT raft
