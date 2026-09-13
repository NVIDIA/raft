/*
 * SPDX-FileCopyrightText: Copyright (2019) Sandia Corporation
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 */
/*
 * Copyright (2019) Sandia Corporation
 *
 * The source code is licensed under the 3-clause BSD license found in the LICENSE file
 * thirdparty/LICENSES/mdarray.license
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/resources.hpp>

#include <stddef.h>

namespace RAFT_EXPORT raft {

/**
 * @defgroup mdarray_apis multi-dimensional memory-owning type
 * @{
 */

/**
 * @brief Interface to implement an owning multi-dimensional array
 *
 * raft::array_interace is an interface to owning container types for mdspan.
 * Check implementation of raft::mdarray which implements raft::array_interface
 * using Curiously Recurring Template Pattern.
 * This interface calls into method `view()` whose implementation is provided by
 * the implementing class. `view()` must return an object of type raft::host_mdspan
 * or raft::device_mdspan or any types derived from the them.
 */
template <typename Base>
class array_interface {
  /**
   * @brief Get an mdspan
   */
  auto view() noexcept { return static_cast<Base*>(this)->view(); }
  /**
   * @brief Get an mdspan<const T>
   */
  auto view() const noexcept { return static_cast<Base*>(this)->view(); }
};

}  // namespace RAFT_EXPORT raft

namespace raft {
namespace detail {
template <typename T, typename = void>
struct is_array_interface : std::false_type {};
template <typename T>
struct is_array_interface<T, std::void_t<decltype(std::declval<T>().view())>>
  : std::bool_constant<is_mdspan_v<decltype(std::declval<T>().view())>> {};

template <typename T>
using is_array_interface_t = is_array_interface<std::remove_const_t<T>>;

/**
 * @\brief Boolean to determine if template type T is raft::array_interface or derived type
 *         or any type that has a member function `view()` that returns either
 *         raft::host_mdspan or raft::device_mdspan
 */
template <typename T>
inline constexpr bool is_array_interface_v = is_array_interface<std::remove_const_t<T>>::value;
}  // namespace detail
}  // namespace raft

namespace RAFT_EXPORT raft {
template <typename...>
struct is_array_interface : std::true_type {};
template <typename T1>
struct is_array_interface<T1> : detail::is_array_interface_t<T1> {};
template <typename T1, typename... Tn>
struct is_array_interface<T1, Tn...> : std::conditional_t<detail::is_array_interface_v<T1>,
                                                          is_array_interface<Tn...>,
                                                          std::false_type> {};
/**
 * @\brief Boolean to determine if variadic template types Tn are raft::array_interface
 *         or derived type or any type that has a member function `view()` that returns either
 *         raft::host_mdspan or raft::device_mdspan
 */
template <typename... Tn>
inline constexpr bool is_array_interface_v = is_array_interface<Tn...>::value;

/**
 * @brief Modified from the c++ mdarray proposal
 *
 *   https://isocpp.org/files/papers/D1684R0.html
 *
 * mdarray is a container type for mdspan with similar template arguments.  However there
 * are some inconsistencies in between them.  We have made some modificiations to fit our
 * needs, which are listed below.
 *
 * - Layout policy is different, the mdarray in raft uses `std::experimental::extent` directly just
 *   like `mdspan`, while the `mdarray` in the reference implementation uses varidic
 *   template.
 *
 * - Most of the constructors from the reference implementation is removed to make sure
 *   CUDA stream is honored. Note that this class is not coupled to CUDA and therefore
 *   will only be used in the case where the device variant is used.
 *
 * - unique_size is not implemented, which is still working in progress in the proposal
 *
 * - For container policy, we adopt the alternative approach documented in the proposal
 *   [sec 2.4.3], which requires an additional make_accessor method for it to be used in
 *   mdspan.  The container policy reference implementation has multiple `access` methods
 *   that accommodate needs for both mdarray and mdspan.  This is more difficult for us
 *   since the policy might contain states that are unwanted inside a CUDA kernel.  Also,
 *   on host we return a proxy to the actual value as `device_ref` so different access
 *   methods will have different return type, which is less desirable.
 *
 * - For the above reasons, copying from other mdarray with different policy type is also
 *   removed.
 */
template <typename ElementType, typename Extents, typename LayoutPolicy, typename ContainerPolicy>
class mdarray
  : public array_interface<mdarray<ElementType, Extents, LayoutPolicy, ContainerPolicy>> {
  static_assert(!std::is_const<ElementType>::value,
                "Element type for container must not be const.");

 public:
  using extents_type = Extents;
  using layout_type  = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;

  using value_type      = std::remove_cv_t<element_type>;
  using index_type      = typename extents_type::index_type;
  using difference_type = std::ptrdiff_t;
  using rank_type       = typename extents_type::rank_type;

  // Naming: ref impl: container_policy_type, proposal: container_policy
  using container_policy_type = ContainerPolicy;
  using container_type        = typename container_policy_type::container_type;

  using pointer         = typename container_policy_type::pointer;
  using const_pointer   = typename container_policy_type::const_pointer;
  using reference       = typename container_policy_type::reference;
  using const_reference = typename container_policy_type::const_reference;

 private:
  template <typename E,
            typename ViewAccessorPolicy =
              std::conditional_t<std::is_const_v<E>,
                                 typename container_policy_type::const_accessor_policy,
                                 typename container_policy_type::accessor_policy>>
  using view_type_impl =
    mdspan<E,
           extents_type,
           layout_type,
           host_device_accessor<ViewAccessorPolicy, container_policy_type::mem_type>>;

 public:
  /**
   * \brief the mdspan type returned by view method.
   */
  using view_type       = view_type_impl<element_type>;
  using const_view_type = view_type_impl<element_type const>;

 public:
  constexpr mdarray(raft::resources const& handle) noexcept(
    std::is_nothrow_default_constructible_v<container_type>)
    : cp_{}, c_{cp_.create(handle, 0)} {};
  constexpr mdarray(mdarray const&) noexcept(std::is_nothrow_copy_constructible_v<container_type>) =
    default;
  constexpr mdarray(mdarray&&) noexcept(std::is_nothrow_move_constructible<container_type>::value) =
    default;

  constexpr auto operator=(mdarray const&) noexcept(
    std::is_nothrow_copy_assignable<container_type>::value) -> mdarray& = default;
  constexpr auto operator=(mdarray&&) noexcept(
    std::is_nothrow_move_assignable<container_type>::value) -> mdarray& = default;

  ~mdarray() noexcept(std::is_nothrow_destructible<container_type>::value) = default;

  /**
   * @brief The only constructor that can create storage, raft::resources is accepted
   * so that the device implementation can make sure the relevant CUDA stream is
   * being used for allocation.
   */
  constexpr mdarray(raft::resources const& handle,
                    mapping_type const& m,
                    container_policy_type const& cp)
    : cp_(cp), map_(m), c_(cp_.create(handle, map_.required_span_size()))
  {
  }

  constexpr mdarray(raft::resources const& handle, mapping_type const& m, container_policy_type& cp)
    : cp_(cp), map_(m), c_(cp_.create(handle, map_.required_span_size()))
  {
  }

  /**
   * @brief Construct from a pre-built container, bypassing cp_.create().
   *
   * Use this when the container has already been constructed externally
   * (e.g. by compressed_mdarray or shared_mdarray factories).
   */
  constexpr mdarray(mapping_type const& m, container_type&& c, container_policy_type const& cp)
    : cp_(cp), map_(m), c_(std::move(c))
  {
  }

  /**
   * @brief Get an mdspan
   */
  auto view() noexcept { return view_type(c_.data(), map_, cp_.make_accessor_policy()); }
  /**
   * @brief Get an mdspan<const T>
   */
  auto view() const noexcept
  {
    return const_view_type(c_.data(), map_, cp_.make_accessor_policy());
  }

  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t { return this->view().size(); }

  [[nodiscard]] auto data_handle() noexcept -> pointer { return c_.data(); }
  [[nodiscard]] constexpr auto data_handle() const noexcept -> const_pointer { return c_.data(); }

  /**
   * @brief Extract the underlying container (only callable on rvalues).
   */
  [[nodiscard]] constexpr auto release_container() && -> container_type { return std::move(c_); }

  /**
   * @brief Indexing operator, use it sparingly since it triggers a device<->host copy.
   */
  template <typename... IndexType>
  auto operator()(IndexType&&... indices)
    -> std::enable_if_t<sizeof...(IndexType) == extents_type::rank() &&
                          (std::is_convertible_v<IndexType, index_type> && ...) &&
                          std::is_constructible_v<extents_type, IndexType...>,
                        /* device policy is not default constructible due to requirement for CUDA
                           stream. */
                        /* std::is_default_constructible_v<container_policy_type> */
                        reference>
  {
    return cp_.access(c_, map_(std::forward<IndexType>(indices)...));
  }

  /**
   * @brief Indexing operator, use it sparingly since it triggers a device<->host copy.
   */
  template <typename... IndexType>
  auto operator()(IndexType&&... indices) const
    -> std::enable_if_t<sizeof...(IndexType) == extents_type::rank() &&
                          (std::is_convertible_v<IndexType, index_type> && ...) &&
                          std::is_constructible_v<extents_type, IndexType...> &&
                          std::is_constructible<mapping_type, extents_type>::value,
                        /* device policy is not default constructible due to requirement for CUDA
                           stream. */
                        /* std::is_default_constructible_v<container_policy_type> */
                        const_reference>
  {
    return cp_.access(c_, map_(std::forward<IndexType>(indices)...));
  }

  // basic_mdarray observers of the domain multidimensional index space (also in basic_mdspan)
  [[nodiscard]] RAFT_INLINE_FUNCTION static constexpr auto rank() noexcept -> rank_type
  {
    return extents_type::rank();
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION static constexpr auto rank_dynamic() noexcept -> rank_type
  {
    return extents_type::rank_dynamic();
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION static constexpr auto static_extent(size_t r) noexcept
    -> index_type
  {
    return extents_type::static_extent(r);
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION constexpr auto extents() const noexcept -> extents_type
  {
    return map_.extents();
  }
  /**
   * @brief the extent of rank r
   */
  [[nodiscard]] RAFT_INLINE_FUNCTION constexpr auto extent(size_t r) const noexcept -> index_type
  {
    return map_.extents().extent(r);
  }
  // mapping
  [[nodiscard]] RAFT_INLINE_FUNCTION constexpr auto mapping() const noexcept -> mapping_type
  {
    return map_;
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION constexpr auto is_unique() const noexcept -> bool
  {
    return map_.is_unique();
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION constexpr auto is_exhaustive() const noexcept -> bool
  {
    return map_.is_exhaustive();
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION constexpr auto is_strided() const noexcept -> bool
  {
    return map_.is_strided();
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION constexpr auto stride(size_t r) const -> index_type
  {
    return map_.stride(r);
  }

  [[nodiscard]] RAFT_INLINE_FUNCTION static constexpr auto is_always_unique() noexcept -> bool
  {
    return mapping_type::is_always_unique();
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION static constexpr auto is_always_exhaustive() noexcept -> bool
  {
    return mapping_type::is_always_exhaustive();
  }
  [[nodiscard]] RAFT_INLINE_FUNCTION static constexpr auto is_always_strided() noexcept -> bool
  {
    return mapping_type::is_always_strided();
  }

 private:
  template <typename, typename, typename, typename>
  friend class mdarray;

 private:
  container_policy_type cp_;
  mapping_type map_;
  container_type c_;
};

/** @} */

/**
 * @defgroup mdarray_reshape Row- or Col-norm computation
 * @{
 */

/**
 * @brief Flatten object implementing raft::array_interface into a 1-dim array view
 *
 * @tparam array_interface_type Expected type implementing raft::array_interface
 * @param mda raft::array_interace implementing object
 * @return Either raft::host_mdspan or raft::device_mdspan with vector_extent
 *         depending on the underlying ContainerPolicy
 */
template <typename array_interface_type,
          std::enable_if_t<is_array_interface_v<array_interface_type>>* = nullptr>
auto flatten(const array_interface_type& mda)
{
  return flatten(mda.view());
}

/**
 * @brief Reshape object implementing raft::array_interface
 *
 * @tparam array_interface_type Expected type implementing raft::array_interface
 * @tparam Extents raft::extents for dimensions
 * @tparam IndexType the index type of the extents
 * @param mda raft::array_interace implementing object
 * @param new_shape Desired new shape of the input
 * @return raft::host_mdspan or raft::device_mdspan, depending on the underlying
 *         ContainerPolicy
 */
template <typename array_interface_type,
          typename IndexType,
          size_t... Extents,
          std::enable_if_t<is_array_interface_v<array_interface_type>>* = nullptr>
auto reshape(const array_interface_type& mda, extents<IndexType, Extents...> new_shape)
{
  return reshape(mda.view(), new_shape);
}

/** @} */

}  // namespace RAFT_EXPORT raft
