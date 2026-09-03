/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/span.hpp>
#include <raft/core/mdspan_types.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>

#include <cassert>
#include <cinttypes>  // size_t
#include <cstddef>    // std::byte
#include <type_traits>

namespace RAFT_EXPORT raft {

/**
 * @defgroup span one-dimensional span type
 * @{
 */
/**
 * @brief The span class defined in ISO C++20.  Iterator is defined as plain pointer and
 *        most of the methods have bound check on debug build.
 *
 * @code
 *   rmm::device_uvector<float> uvec(10, cuda::stream_ref{cudaStream_t{cudaStreamDefault}});
 *   auto view = device_span<float>{uvec.data(), uvec.size()};
 * @endcode
 */
template <typename T, bool is_device, std::size_t Extent = dynamic_extent>
class span {
 public:
  using element_type    = T;
  using value_type      = typename std::remove_cv<T>::type;
  using size_type       = std::size_t;
  using difference_type = std::ptrdiff_t;
  using pointer         = T*;
  using const_pointer   = T const*;
  using reference       = T&;
  using const_reference = T const&;

  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = cuda::std::reverse_iterator<const_iterator>;

  /**
   * @brief Default constructor that constructs a span with size 0 and nullptr.
   */
  constexpr span() noexcept = default;

  /**
   * @brief Constructs a span that is a view over the range [first, first + count);
   */
  constexpr span(pointer ptr, size_type count) noexcept : storage_{ptr, count}
  {
    assert(!(Extent != dynamic_extent && count != Extent));
    assert(ptr || count == 0);
  }
  /**
   * @brief Constructs a span that is a view over the range [first, last)
   */
  constexpr span(pointer first, pointer last) noexcept
    : span{first, static_cast<size_type>(cuda::std::distance(first, last))}
  {
  }
  /**
   * @brief Constructs a span that is a view over the array arr.
   */
  template <std::size_t N>
  constexpr span(element_type (&arr)[N]) noexcept : span{&arr[0], N}
  {
  }

  /**
   * @brief Initialize a span class from another one who's underlying type is convertible
   *        to element_type.
   */
  template <class U,
            std::size_t OtherExtent,
            class = typename std::enable_if<
              detail::is_allowed_element_type_conversion_t<U, T>::value &&
              detail::is_allowed_extent_conversion_t<OtherExtent, Extent>::value>>
  constexpr span(const span<U, is_device, OtherExtent>& other) noexcept
    : span{other.data(), other.size()}
  {
  }

  constexpr span(span const& other) noexcept = default;
  constexpr span(span&& other) noexcept      = default;

  constexpr auto operator=(span const& other) noexcept -> span& = default;
  constexpr auto operator=(span&& other) noexcept -> span&      = default;

  constexpr auto begin() const noexcept -> iterator { return data(); }

  constexpr auto end() const noexcept -> iterator { return data() + size(); }

  constexpr auto cbegin() const noexcept -> const_iterator { return data(); }

  constexpr auto cend() const noexcept -> const_iterator { return data() + size(); }

  _RAFT_HOST_DEVICE constexpr auto rbegin() const noexcept -> reverse_iterator
  {
    return reverse_iterator{end()};
  }

  _RAFT_HOST_DEVICE constexpr auto rend() const noexcept -> reverse_iterator
  {
    return reverse_iterator{begin()};
  }

  _RAFT_HOST_DEVICE constexpr auto crbegin() const noexcept -> const_reverse_iterator
  {
    return const_reverse_iterator{cend()};
  }

  _RAFT_HOST_DEVICE constexpr auto crend() const noexcept -> const_reverse_iterator
  {
    return const_reverse_iterator{cbegin()};
  }

  // element access
  constexpr auto front() const -> reference { return (*this)[0]; }

  constexpr auto back() const -> reference { return (*this)[size() - 1]; }

  template <typename Index>
  constexpr auto operator[](Index _idx) const -> reference
  {
    assert(static_cast<size_type>(_idx) < size());
    return data()[_idx];
  }

  constexpr auto data() const noexcept -> pointer { return storage_.data(); }

  // Observers
  [[nodiscard]] constexpr auto size() const noexcept -> size_type { return storage_.size(); }
  [[nodiscard]] constexpr auto size_bytes() const noexcept -> size_type
  {
    return size() * sizeof(T);
  }

  constexpr auto empty() const noexcept { return size() == 0; }

  // Subviews
  template <std::size_t Count>
  constexpr auto first() const -> span<element_type, is_device, Count>
  {
    assert(Count <= size());
    return {data(), Count};
  }

  constexpr auto first(std::size_t _count) const -> span<element_type, is_device, dynamic_extent>
  {
    assert(_count <= size());
    return {data(), _count};
  }

  template <std::size_t Count>
  constexpr auto last() const -> span<element_type, is_device, Count>
  {
    assert(Count <= size());
    return {data() + size() - Count, Count};
  }

  constexpr auto last(std::size_t _count) const -> span<element_type, is_device, dynamic_extent>
  {
    assert(_count <= size());
    return subspan(size() - _count, _count);
  }

  /*!
   * If Count is std::dynamic_extent, r.size() == this->size() - Offset;
   * Otherwise r.size() == Count.
   */
  template <std::size_t Offset, std::size_t Count = dynamic_extent>
  constexpr auto subspan() const
    -> span<element_type, is_device, detail::extent_value_t<Extent, Offset, Count>::value>
  {
    assert((Count == dynamic_extent) ? (Offset <= size()) : (Offset + Count <= size()));
    return {data() + Offset, Count == dynamic_extent ? size() - Offset : Count};
  }

  constexpr auto subspan(size_type _offset, size_type _count = dynamic_extent) const
    -> span<element_type, is_device, dynamic_extent>
  {
    assert((_count == dynamic_extent) ? (_offset <= size()) : (_offset + _count <= size()));
    return {data() + _offset, _count == dynamic_extent ? size() - _offset : _count};
  }

 private:
  detail::span_storage<T, Extent> storage_;
};

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator==(span<T, is_device, X> l, span<U, is_device, Y> r) -> bool
{
  if (l.size() != r.size()) { return false; }
  for (auto l_beg = l.cbegin(), r_beg = r.cbegin(); l_beg != l.cend(); ++l_beg, ++r_beg) {
    if (*l_beg != *r_beg) { return false; }
  }
  return true;
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator!=(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return !(l == r);
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator<(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return detail::lexicographical_compare<
    typename span<T, is_device, X>::iterator,
    typename span<U, is_device, Y>::iterator,
    cuda::std::less<typename span<T, is_device, X>::element_type>>(
    l.begin(), l.end(), r.begin(), r.end());
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator<=(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return !(l > r);
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator>(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return detail::lexicographical_compare<
    typename span<T, is_device, X>::iterator,
    typename span<U, is_device, Y>::iterator,
    cuda::std::greater<typename span<T, is_device, X>::element_type>>(
    l.begin(), l.end(), r.begin(), r.end());
}

template <class T, std::size_t X, class U, std::size_t Y, bool is_device>
constexpr auto operator>=(span<T, is_device, X> l, span<U, is_device, Y> r)
{
  return !(l < r);
}

/**
 * @brief Converts a span into a view of its underlying bytes
 */
template <class T, bool is_device, std::size_t E>
auto as_bytes(span<T, is_device, E> s) noexcept
  -> span<const std::byte, is_device, detail::extent_as_bytes_value_t<T, E>::value>
{
  return {reinterpret_cast<const std::byte*>(s.data()), s.size_bytes()};
}

/**
 * @brief Converts a span into a mutable view of its underlying bytes
 */
template <class T, bool is_device, std::size_t E>
auto as_writable_bytes(span<T, is_device, E> s) noexcept
  -> span<std::byte, is_device, detail::extent_as_bytes_value_t<T, E>::value>
{
  return {reinterpret_cast<std::byte*>(s.data()), s.size_bytes()};
}

/* @} */

}  // namespace RAFT_EXPORT raft
