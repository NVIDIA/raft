/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/cuda_support.hpp>
#include <raft/core/detail/copy.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/managed_container_policy.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/pinned_container_policy.hpp>
#include <raft/core/stream_view.hpp>
#include <raft/util/variant_utils.hpp>

#include <algorithm>
#include <execution>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/device_ptr.h>
#else
#include <raft/core/copy.hpp>
#endif

namespace RAFT_EXPORT raft {

/**
 * @defgroup mdbuffer_apis multi-dimensional maybe-owning type
 * @{
 */

/**
 * @brief Retrieve a canonical index associated with a given memory type.
 *
 * For variants based on memory type, this index can be used to help keep a
 * consistent ordering of the memory types in the variant.
 */
inline auto constexpr variant_index_from_memory_type(raft::memory_type mem_type)
{
  return static_cast<std::underlying_type_t<raft::memory_type>>(mem_type);
}

/**
 * @brief Retrieve the memory type associated with a canonical index
 */
inline auto constexpr memory_type_from_variant_index(
  std::underlying_type_t<raft::memory_type> index)
{
  return static_cast<raft::memory_type>(index);
}

/**
 * @brief Retrieve a type from a variant based on a given memory type.
 */
template <raft::memory_type MemType, typename Variant>
using alternate_from_mem_type =
  std::variant_alternative_t<variant_index_from_memory_type(MemType) % std::variant_size_v<Variant>,
                             Variant>;

}  // namespace RAFT_EXPORT raft

namespace raft {
namespace detail {
template <typename T, raft::memory_type MemType>
struct memory_type_to_default_policy {};
template <typename T>
struct memory_type_to_default_policy<T, raft::memory_type::host> {
  using type = typename raft::host_container_policy<T>;
};
template <typename T>
struct memory_type_to_default_policy<T, raft::memory_type::device> {
  using type = typename raft::device_container_policy<T>;
};
template <typename T>
struct memory_type_to_default_policy<T, raft::memory_type::managed> {
  using type = typename raft::managed_container_policy<T>;
};
template <typename T>
struct memory_type_to_default_policy<T, raft::memory_type::pinned> {
  using type = typename raft::pinned_container_policy<T>;
};

template <typename T, raft::memory_type MemType>
using memory_type_to_default_policy_t = typename memory_type_to_default_policy<T, MemType>::type;
}  // namespace detail
}  // namespace raft

namespace RAFT_EXPORT raft {
/**
 * @brief A variant of container policies for each memory type which can be
 * used to build the default container policy for a buffer.
 */
template <typename T>
using default_container_policy_variant =
  std::variant<detail::memory_type_to_default_policy_t<T, memory_type_from_variant_index(0)>,
               detail::memory_type_to_default_policy_t<T, memory_type_from_variant_index(1)>,
               detail::memory_type_to_default_policy_t<T, memory_type_from_variant_index(2)>,
               detail::memory_type_to_default_policy_t<T, memory_type_from_variant_index(3)>>;

/**
 * @brief A template used to translate a variant of underlying mdarray
 * container policies into a container policy that can be used by an mdbuffer.
 */
template <typename ElementType,
          typename ContainerPolicyVariant =
            default_container_policy_variant<std::remove_cv_t<ElementType>>>
struct default_buffer_container_policy {
  using element_type = ElementType;
  using value_type   = std::remove_cv_t<element_type>;

 private:
  template <std::size_t index>
  using raw_container_policy_at_index = std::variant_alternative_t<index, ContainerPolicyVariant>;

 public:
  using container_policy_variant =
    std::variant<host_device_accessor<std::variant_alternative_t<0, ContainerPolicyVariant>,
                                      static_cast<memory_type>(0)>,
                 host_device_accessor<std::variant_alternative_t<1, ContainerPolicyVariant>,
                                      static_cast<memory_type>(1)>,
                 host_device_accessor<std::variant_alternative_t<2, ContainerPolicyVariant>,
                                      static_cast<memory_type>(2)>,
                 host_device_accessor<std::variant_alternative_t<3, ContainerPolicyVariant>,
                                      static_cast<memory_type>(3)>>;
  template <raft::memory_type MemType>
  using container_policy = alternate_from_mem_type<MemType, container_policy_variant>;
  using container_type_variant =
    std::variant<typename raw_container_policy_at_index<0>::container_type,
                 typename raw_container_policy_at_index<1>::container_type,
                 typename raw_container_policy_at_index<2>::container_type,
                 typename raw_container_policy_at_index<3>::container_type>;

  template <raft::memory_type MemType>
  using container_type = alternate_from_mem_type<MemType, container_type_variant>;

  using accessor_policy_variant =
    std::variant<host_device_accessor<typename raw_container_policy_at_index<0>::accessor_policy,
                                      static_cast<memory_type>(0)>,
                 host_device_accessor<typename raw_container_policy_at_index<1>::accessor_policy,
                                      static_cast<memory_type>(1)>,
                 host_device_accessor<typename raw_container_policy_at_index<2>::accessor_policy,
                                      static_cast<memory_type>(2)>,
                 host_device_accessor<typename raw_container_policy_at_index<3>::accessor_policy,
                                      static_cast<memory_type>(3)>>;

  template <raft::memory_type MemType>
  using accessor_policy = alternate_from_mem_type<MemType, accessor_policy_variant>;

  using const_accessor_policy_variant = std::variant<
    host_device_accessor<typename raw_container_policy_at_index<0>::const_accessor_policy,
                         static_cast<memory_type>(0)>,
    host_device_accessor<typename raw_container_policy_at_index<1>::const_accessor_policy,
                         static_cast<memory_type>(1)>,
    host_device_accessor<typename raw_container_policy_at_index<2>::const_accessor_policy,
                         static_cast<memory_type>(2)>,
    host_device_accessor<typename raw_container_policy_at_index<3>::const_accessor_policy,
                         static_cast<memory_type>(3)>>;
  template <raft::memory_type MemType>
  using const_accessor_policy = alternate_from_mem_type<MemType, accessor_policy_variant>;

  template <raft::memory_type MemType>
  auto create(raft::resources const& res, size_t n)
  {
    return container_type<MemType>(res, n);
  }

  auto create(raft::resources const& res, size_t n, raft::memory_type mem_type)
  {
    auto result = container_type_variant{};
    switch (mem_type) {
      case raft::memory_type::host: result = create<raft::memory_type::host>(res, n); break;
      case raft::memory_type::device: result = create<raft::memory_type::device>(res, n); break;
      case raft::memory_type::managed: result = create<raft::memory_type::managed>(res, n); break;
      case raft::memory_type::pinned: result = create<raft::memory_type::pinned>(res, n); break;
    }
    return result;
  }

 private:
  template <typename ContainerType>
  auto static constexpr has_stream() -> decltype(std::declval<ContainerType>().stream(), bool())
  {
    return true;
  };
  auto static constexpr has_stream(...) -> bool { return false; };

 public:
  template <memory_type MemType>
  [[nodiscard]] auto make_accessor_policy() noexcept
  {
    return accessor_policy<MemType>{};
  }
  template <memory_type MemType>
  [[nodiscard]] auto make_accessor_policy() const noexcept
  {
    return const_accessor_policy<MemType>{};
  }

  [[nodiscard]] auto make_accessor_policy(memory_type mem_type) noexcept
  {
    auto result = accessor_policy_variant{};
    switch (mem_type) {
      case memory_type::host: result = make_accessor_policy<memory_type::host>(); break;
      case memory_type::device: result = make_accessor_policy<memory_type::device>(); break;
      case memory_type::managed: result = make_accessor_policy<memory_type::managed>(); break;
      case memory_type::pinned: result = make_accessor_policy<memory_type::pinned>(); break;
    }
    return result;
  }
  [[nodiscard]] auto make_accessor_policy(memory_type mem_type) const noexcept
  {
    auto result = const_accessor_policy_variant{};
    switch (mem_type) {
      case memory_type::host: result = make_accessor_policy<memory_type::host>(); break;
      case memory_type::device: result = make_accessor_policy<memory_type::device>(); break;
      case memory_type::managed: result = make_accessor_policy<memory_type::managed>(); break;
      case memory_type::pinned: result = make_accessor_policy<memory_type::pinned>(); break;
    }
    return result;
  }
};

/**
 * @brief A type representing multi-dimensional data which may or may not own
 * its underlying storage. `raft::mdbuffer` is used to conveniently perform
 * copies of data _only_ when necessary to ensure that the data are accessible
 * in the desired memory space and format.
 *
 * When developing functions that interact with the GPU, it is often necessary
 * to ensure that the data are in a particular memory space (e.g. device,
 * host, managed, pinned), but those functions may be called with data that
 * may or may not already be in the desired memory space. For instance, when
 * called in one workflow, the data may have been previously transferred to
 * device, rendering a copy unnecessary. In another, the function may be
 * directly invoked on host data.
 *
 * Even when working strictly with host memory, it is often necessary to
 * ensure that the data are in a particular layout for efficient access (e.g.
 * column major vs row major) or that the the data are of a particular type
 * (e.g. double) even though we wish to call the function with data of
 * another compatible type (e.g. float).
 *
 * `mdbuffer` is a tool for ensuring that the data are represented in exactly
 * the desired format and location while flexibly supporting data which may
 * not already be in that format or location. It does so by providing a
 * non-owning view on data which are already in the required form, but it
 * allocates (owned) memory and performs a copy if and only if it is
 * necessary.
 *
 * Usage example:
 * @code{.cpp}
 * template <typename mdspan_type>
 * void foo_device(raft::resources const& res, mdspan_type data) {
 *   auto buf = raft::mdbuffer{res, raft::mdbuffer{data}, raft::memory_type::device};
 *   // Data in buf is now guaranteed to be accessible from device.
 *   // If it was already accessible from device, no copy was performed. If it
 *   // was not, a copy was performed.
 *
 *   some_kernel<<<...>>>(buf.view<raft::memory_type::device>());
 *
 *   // It is sometimes useful to know whether or not a copy was performed to
 *   // e.g. determine whether the transformed data should be copied back to its original
 *   // location. This can be checked via the `is_owning()` method.
 *   if (buf.is_owning()) {
 *     raft::copy(res, data, buf.view<raft::memory_type::device>());
 *   }
 * }
 * @endcode
 *
 * Note that in this example, the `foo_device` template can be correctly
 * instantiated for both host and device mdspans. Similarly we can use
 * `mdbuffer` to coerce data to a particular memory layout and data-type, as in
 * the following example:
 * @code{.cpp}
 * template <typename mdspan_type>
 * void foo_device(raft::resources const& res, mdspan_type data) {
 *   auto buf = raft::mdbuffer<float, raft::matrix_extent<int>, raft::row_major>{res,
 * raft::mdbuffer{data}, raft::memory_type::device};
 *   // Data in buf is now guaranteed to be accessible from device, and
 *   // represented by floats in row-major order.
 *
 *   some_kernel<<<...>>>(buf.view<raft::memory_type::device>());
 *
 *   // The same check can be used to determine whether or not a copy was
 *   // required, regardless of the cause. I.e. if the data were already on
 *   // device but in column-major order, the is_owning() method would still
 *   // return true because new storage needed to be allocated.
 *   if (buf.is_owning()) {
 *     raft::copy(res, data, buf.view<raft::memory_type::device>());
 *   }
 * }
 * @endcode
 *
 * Note that in this example, the `foo_device` template can accept data of
 * any float-convertible type in any layout and of any memory type and coerce
 * it to the desired device-accessible representation.
 *
 * Because `mdspan` types can be implicitly converted to `mdbuffer`, it is even
 * possible to avoid multiple template instantiations by directly accepting an
 * `mdbuffer` as argument, as in the following example:
 * @code{.cpp}
 * void foo_device(raft::resources const& res, raft::mdbuffer<float, raft::matrix_extent<int>>&&
 * data) { auto buf = raft::mdbuffer{res, data, raft::memory_type::device};
 *   // Data in buf is now guaranteed to be accessible from device.
 *
 *   some_kernel<<<...>>>(buf.view<raft::memory_type::device>());
 * }
 * @endcode
 *
 * In this example, `foo_device` can now accept any row-major mdspan of floats
 * regardless of memory type without requiring separate template instantiations
 * for each type.
 *
 * While the view method takes an optional compile-time memory type parameter,
 * omitting this parameter will return a std::variant of mdspan types. This
 * allows for straightforward runtime dispatching based on the memory type
 * using std::visit, as in the following example:
 *
 * @code{.cpp}
 * void foo(raft::resources const& res, raft::mdbuffer<float, raft::matrix_extent<int>>&& data) {
 *   std::visit([](auto&& view) {
 *     // Do something with the view, including (possibly) dispatching based on
 *     // whether it is a host, device, managed, or pinned mdspan
 *   }, data.view());
 * }
 * @endcode
 *
 * For convenience, runtime memory-type dispatching can also be performed
 * without explicit use of `mdbuffer` using `raft::memory_type_dispatcher`, as
 * described in @ref memory_type_dispatcher. Please see the full documentation
 * of that function template for more extensive discussion of the many ways it
 * can be used. To illustrate its connection to `mdbuffer`, however, consider
 * the following example, which performs a similar task to the above
 * `std::visit` call:
 *
 * @code{.cpp}
 * void foo_device(raft::resources const& res, raft::device_matrix_view<float> data) {
 *   // Implement foo solely for device data
 * };
 *
 * // Call foo with data of any memory type:
 * template <typename mdspan_type>
 * void foo(raft::resources const& res, mdspan_type data) {
 *   raft::memory_type_dispatcher(res,
 *     [&res](raft::device_matrix_view<float> dev_data) {foo_device(res, dev_data);},
 *     data
 *   );
 * }
 * @endcode
 *
 * Here, the `memory_type_dispatcher` implicitly constructs an `mdbuffer` from
 * the input and performs any necessary conversions before passing the input to
 * `foo_device`. While `mdbuffer` does not require the use of
 * `memory_type_dispatcher`, there are many common use cases in which explicit
 * invocations of `mdbuffer` can be elided with `memory_type_dispatcher`.
 *
 * Finally, we should note that `mdbuffer` should almost never be passed as a
 * const reference. To indicate const-ness of the underlying data, the
 * `mdbuffer` should be constructed with a const memory type, but the mdbuffer
 * itself should generally be passed as an rvalue reference in function
 * arguments. Using an `mdbuffer` that is itself `const` is not strictly
 * incorrect, but it indicates a likely misuse of the type.
 *
 * @tparam ElementType element type stored in the buffer
 * @tparam Extents specifies the number of dimensions and their sizes
 * @tparam LayoutPolicy specifies how data should be laid out in memory
 * @tparam ContainerPolicy specifies how data should be allocated if necessary
 * and how it should be accessed. This should very rarely need to be
 * customized. For those cases where it must be customized, it is recommended
 * to instantiate default_buffer_container_policy with a std::variant of
 * container policies for each memory type. Note that the accessor policy of
 * each container policy variant is used as the accessor policy for the mdspan
 * view of the buffer for the corresponding memory type.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = default_buffer_container_policy<ElementType>>
struct mdbuffer {
  using extents_type = Extents;
  using layout_type  = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;

  using value_type      = std::remove_cv_t<element_type>;
  using index_type      = typename extents_type::index_type;
  using difference_type = std::ptrdiff_t;
  using rank_type       = typename extents_type::rank_type;

  using container_policy_type   = ContainerPolicy;
  using accessor_policy_variant = typename ContainerPolicy::accessor_policy_variant;

  template <raft::memory_type MemType>
  using accessor_policy = alternate_from_mem_type<MemType, accessor_policy_variant>;

  using container_type_variant = typename container_policy_type::container_type_variant;

  template <raft::memory_type MemType>
  using container_type = typename container_policy_type::template container_type<MemType>;

  template <memory_type MemType>
  using owning_type = mdarray<value_type,
                              extents_type,
                              layout_type,
                              typename container_policy_type::template container_policy<MemType>>;
  // We use the static cast here to ensure that the memory types appear in the
  // order expected for retrieving the correct variant alternative based on
  // memory type. Even if the memory types are re-arranged in the enum and
  // assigned different values, the logic should remain correct.
  using owning_type_variant = std::variant<owning_type<static_cast<memory_type>(0)>,
                                           owning_type<static_cast<memory_type>(1)>,
                                           owning_type<static_cast<memory_type>(2)>,
                                           owning_type<static_cast<memory_type>(3)>>;

  template <memory_type MemType>
  using view_type = std::conditional_t<std::is_const_v<element_type>,
                                       typename owning_type<MemType>::const_view_type,
                                       typename owning_type<MemType>::view_type>;

  using view_type_variant = std::variant<view_type<static_cast<memory_type>(0)>,
                                         view_type<static_cast<memory_type>(1)>,
                                         view_type<static_cast<memory_type>(2)>,
                                         view_type<static_cast<memory_type>(3)>>;

  template <memory_type MemType>
  using const_view_type         = typename owning_type<MemType>::const_view_type;
  using const_view_type_variant = std::variant<const_view_type<static_cast<memory_type>(0)>,
                                               const_view_type<static_cast<memory_type>(1)>,
                                               const_view_type<static_cast<memory_type>(2)>,
                                               const_view_type<static_cast<memory_type>(3)>>;

  using storage_type_variant = concatenated_variant_t<view_type_variant, owning_type_variant>;

  // Non-owning types are stored first in the variant Thus, if we want to access the
  // owning type corresponding to device memory, we would need to skip over the
  // non-owning types and then go to the index which corresponds to the memory
  // type: is_owning * num_non_owning_types + index = 1 * 4 + 1 = 5
  template <memory_type MemType, bool is_owning>
  using storage_type =
    std::variant_alternative_t<std::size_t{is_owning} * std::variant_size_v<view_type_variant> +
                                 std::size_t{variant_index_from_memory_type(MemType)},
                               storage_type_variant>;

  /**
   * @brief Construct an empty, uninitialized buffer
   */
  constexpr mdbuffer() = default;

 private:
  container_policy_type cp_{};
  storage_type_variant data_{};

  // This template is used to determine whether or not is possible to copy from
  // the mdspan returned by the view method of a FromT type mdbuffer with
  // memory type indicated by FromIndex to the mdspan returned by this mdbuffer
  // at ToIndex
  template <typename FromT, std::size_t FromIndex, std::size_t ToIndex>
  auto static constexpr is_copyable_combination()
  {
    return detail::mdspan_copyable_v<
      decltype(std::declval<std::variant_alternative_t<ToIndex, owning_type_variant>>().view()),
      std::variant_alternative_t<FromIndex, decltype(std::declval<FromT>().view())>>;
  }

  // Using an index_sequence to iterate over the possible memory types of this
  // mdbuffer, we construct an array of bools to determine whether or not the
  // mdspan returned by the view method of a FromT type mdbuffer with memory
  // type indicated by FromIndex can be copied to the mdspan returned by this
  // mdbuffer's view method at each memory type
  template <std::size_t FromIndex, typename FromT, std::size_t... ToIs>
  auto static constexpr get_to_copyable_combinations(std::index_sequence<ToIs...>)
  {
    return std::array{is_copyable_combination<FromT, FromIndex, ToIs>()...};
  }

  // Using an index_sequence to iterate over the possible memory types of the
  // FromT type mdbuffer, we construct an array of arrays indicating whether it
  // is possible to copy from any mdspan that can be returned from the FromT
  // mdbuffer to any mdspan that can be returned from this mdbuffer
  template <typename FromT, std::size_t... FromIs>
  auto static constexpr get_from_copyable_combinations(std::index_sequence<FromIs...>)
  {
    return std::array{get_to_copyable_combinations<FromIs, FromT>(
      std::make_index_sequence<std::variant_size_v<owning_type_variant>>())...};
  }

  // Get an array of arrays indicating whether or not it is possible to copy
  // from any given memory type of a FromT mdbuffer to any memory type of this
  // mdbuffer
  template <typename FromT>
  auto static constexpr get_copyable_combinations()
  {
    return get_from_copyable_combinations<FromT>(
      std::make_index_sequence<std::variant_size_v<decltype(std::declval<FromT>().view())>>());
  }

  template <std::size_t FromIndex, typename FromT, std::size_t... Is>
  auto static constexpr is_copyable_from(std::index_sequence<Is...>)
  {
    return (... || get_copyable_combinations<FromT>()[FromIndex][Is]);
  }

  template <typename FromT, std::size_t... Is>
  auto static constexpr is_copyable_from(bool, std::index_sequence<Is...>)
  {
    return (... || is_copyable_from<Is, FromT>(
                     std::make_index_sequence<std::variant_size_v<owning_type_variant>>()));
  }

  template <typename FromT>
  auto static constexpr is_copyable_from()
  {
    return is_copyable_from<FromT>(
      true,
      std::make_index_sequence<std::variant_size_v<decltype(std::declval<FromT>().view())>>());
  }

  template <typename FromT>
  auto static is_copyable_from(FromT&& other, memory_type mem_type)
  {
    auto static copyable_combinations = get_copyable_combinations<FromT>();
    return copyable_combinations[variant_index_from_memory_type(other.mem_type())]
                                [variant_index_from_memory_type(mem_type)];
  }

  template <typename FromT>
  storage_type_variant static copy_from(raft::resources const& res,
                                        FromT&& other,
                                        memory_type mem_type)
  {
    switch (mem_type) {
      case memory_type::host: {
        return std::visit(
          [&res](auto&& other_view) {
            auto tmp_result = owning_type<memory_type::host>{
              res,
              mapping_type{other_view.extents()},
              typename container_policy_type::template container_policy<memory_type::host>{}};
            raft::copy(res, tmp_result.view(), other_view);
            return tmp_result;
          },
          other.view());
      }
      case memory_type::device: {
        return std::visit(
          [&res](auto&& other_view) {
            auto tmp_result = owning_type<memory_type::device>{
              res,
              mapping_type{other_view.extents()},
              typename container_policy_type::template container_policy<memory_type::device>{}};
            raft::copy(res, tmp_result.view(), other_view);
            return tmp_result;
          },
          other.view());
      }
      case memory_type::managed: {
        return std::visit(
          [&res](auto&& other_view) {
            auto tmp_result = owning_type<memory_type::managed>{
              res,
              mapping_type{other_view.extents()},
              typename container_policy_type::template container_policy<memory_type::managed>{}};
            raft::copy(res, tmp_result.view(), other_view);
            return tmp_result;
          },
          other.view());
      }
      case memory_type::pinned: {
        return std::visit(
          [&res](auto&& other_view) {
            auto tmp_result = owning_type<memory_type::pinned>{
              res,
              mapping_type{other_view.extents()},
              typename container_policy_type::template container_policy<memory_type::pinned>{}};
            raft::copy(res, tmp_result.view(), other_view);
            return tmp_result;
          },
          other.view());
      }
    }
    __builtin_unreachable();
  }

 public:
  /**
   * @brief Construct an mdbuffer wrapping an existing mdspan. The resulting
   * mdbuffer will be non-owning and match the memory type, layout, and
   * element type of the mdspan.
   */
  template <
    typename OtherAccessorPolicy,
    std::enable_if_t<is_type_in_variant_v<OtherAccessorPolicy, accessor_policy_variant>>* = nullptr>
  mdbuffer(mdspan<ElementType, Extents, LayoutPolicy, OtherAccessorPolicy> other) : data_{other}
  {
  }

  /**
   * @brief Construct an mdbuffer of const elements wrapping an existing mdspan
   * with non-const elements. The resulting mdbuffer will be non-owning and match the memory type,
   * layout, and element type of the mdspan.
   */
  template <
    typename OtherElementType,
    typename OtherAccessorPolicy,
    std::enable_if_t<!std::is_same_v<OtherElementType, ElementType> &&
                     std::is_same_v<OtherElementType const, ElementType> &&
                     is_type_in_variant_v<OtherAccessorPolicy, accessor_policy_variant>>* = nullptr>
  mdbuffer(mdspan<OtherElementType, Extents, LayoutPolicy, OtherAccessorPolicy> other)
    : data_{raft::make_const_mdspan(other)}
  {
  }

  /**
   * @brief Construct an mdbuffer to hold an existing mdarray rvalue. The
   * mdarray will be moved into the mdbuffer, and the mdbuffer will be owning.
   */
  template <typename OtherContainerPolicy,
            std::enable_if_t<is_type_in_variant_v<
              host_device_accessor<typename OtherContainerPolicy::accessor_type,
                                   OtherContainerPolicy::mem_type>,
              typename container_policy_type::container_policy_variant>>* = nullptr>
  mdbuffer(mdarray<ElementType, Extents, LayoutPolicy, OtherContainerPolicy>&& other)
    : data_{std::move(other)}
  {
  }

  /**
   * @brief Construct an mdbuffer from an existing mdarray lvalue. An mdspan
   * view will be taken from the mdarray in order to construct the mdbuffer,
   * and the mdbuffer will be non-owning
   */
  template <typename OtherContainerPolicy,
            std::enable_if_t<is_type_in_variant_v<
              host_device_accessor<typename OtherContainerPolicy::accessor_type,
                                   OtherContainerPolicy::mem_type>,
              typename container_policy_type::container_policy_variant>>* = nullptr>
  mdbuffer(mdarray<ElementType, Extents, LayoutPolicy, OtherContainerPolicy>& other)
    : mdbuffer{other.view()}
  {
  }

  /**
   * @brief Construct one mdbuffer from another mdbuffer rvalue with matching
   * element type, extents, layout, and container policy.
   *
   * If the existing mdbuffer is owning and of the correct memory type,
   * the new mdbuffer will take ownership of the underlying memory
   * (preventing a view on memory owned by a moved-from object). The memory
   * type of the new mdbuffer may be specified explicitly, in which case a copy
   * will be performed if and only if it is necessary to do so.
   */
  mdbuffer(raft::resources const& res,
           mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>&& other,
           std::optional<memory_type> specified_mem_type = std::nullopt)
    : data_{[&res, &other, specified_mem_type, this]() {
        auto other_mem_type = other.mem_type();
        auto mem_type       = specified_mem_type.value_or(other_mem_type);
        if (mem_type == other.mem_type()) { return std::move(other.data_); }
        if (!other.is_owning() && has_compatible_accessibility(other_mem_type, mem_type) &&
            !is_host_device_accessible(mem_type)) {
          switch (mem_type) {
            case (memory_type::host): {
              return std::visit(
                [this](auto&& other_view) {
                  return view_type<memory_type::host>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::host>()};
                },
                other.view());
            }
            case (memory_type::device): {
              return std::visit(
                [this](auto&& other_view) {
                  return view_type<memory_type::device>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::device>()};
                },
                other.view());
            }
            case (memory_type::managed): {
              return std::visit(
                [this](auto&& other_view) {
                  return view_type<memory_type::managed>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::managed>()};
                },
                other.view());
            }
            case (memory_type::pinned): {
              return std::visit(
                [this](auto&& other_view) {
                  return view_type<memory_type::pinned>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::pinned>()};
                },
                other.view());
            }
          }
        }
        return copy_from(res, other, mem_type);
      }()}
  {
  }

  /**
   * @brief Construct one mdbuffer from another mdbuffer lvalue with matching
   * element type, extents, layout, and container policy.
   *
   * Unlike when constructing from an rvalue, the new mdbuffer will take a
   * non-owning view whenever possible, since it is assumed that the caller
   * will manage the lifetime of the lvalue input. Note that the mdbuffer
   * passed here must itself be non-const in order to allow this constructor to
   * provide an equivalent view of the underlying data. To indicate const-ness
   * of the underlying data, mdbuffers should be constructed with a const
   * ElementType.
   */
  mdbuffer(raft::resources const& res,
           mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>& other, /* NOLINT */
           std::optional<memory_type> specified_mem_type = std::nullopt)
    : data_{[&res, &other, specified_mem_type, this]() -> storage_type_variant {
        auto mem_type       = specified_mem_type.value_or(other.mem_type());
        auto other_mem_type = other.mem_type();
        if (mem_type == other_mem_type) {
          return std::visit([](auto&& other_view) { return storage_type_variant{other_view}; },
                            other.view());
        }
        if (has_compatible_accessibility(other_mem_type, mem_type) &&
            !is_host_device_accessible(mem_type)) {
          switch (mem_type) {
            case (memory_type::host): {
              return std::visit(
                [this](auto&& other_view) {
                  return view_type<memory_type::host>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::host>()};
                },
                other.view());
            }
            case (memory_type::device): {
              return std::visit(
                [this](auto&& other_view) {
                  return view_type<memory_type::device>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::device>()};
                },
                other.view());
            }
            case (memory_type::managed): {
              return std::visit(
                [this](auto&& other_view) {
                  return view_type<memory_type::managed>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::managed>()};
                },
                other.view());
            }
            case (memory_type::pinned): {
              return std::visit(
                [this](auto&& other_view) {
                  return view_type<memory_type::pinned>{
                    other_view.data_handle(),
                    other_view.mapping(),
                    cp_.template make_accessor_policy<memory_type::pinned>()};
                },
                other.view());
            }
          }
        }
        return copy_from(res, other, mem_type);
      }()}
  {
  }

  /**
   * @brief Construct an mdbuffer from an existing mdbuffer with arbitrary but
   * compatible element type, extents, layout, and container policy. This
   * constructor is used to coerce data to specific element types, layouts,
   * or extents as well as specifying a memory type.
   */
  template <
    typename OtherElementType,
    typename OtherExtents,
    typename OtherLayoutPolicy,
    typename OtherContainerPolicy,
    std::enable_if_t<is_copyable_from<
      mdbuffer<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainerPolicy>>()>* =
      nullptr>
  mdbuffer(
    raft::resources const& res,
    mdbuffer<OtherElementType, OtherExtents, OtherLayoutPolicy, OtherContainerPolicy> const& other,
    std::optional<memory_type> specified_mem_type = std::nullopt)
    : data_{[&res, &other, specified_mem_type]() {
        auto mem_type = specified_mem_type.value_or(other.mem_type());
        // Note: We perform this check at runtime because it is possible for two
        // mdbuffers to have storage types which may be copied to each other for
        // some memory types but not for others. This is an unusual situation, but
        // we still need to guard against it.
        RAFT_EXPECTS(
          is_copyable_from<decltype(other)>(other, mem_type),
          "mdbuffer cannot be constructed from other mdbuffer with indicated memory type");
        return copy_from(res, other, mem_type);
      }()}
  {
  }

  /**
   * @brief Return the memory type of the underlying data referenced by the
   * mdbuffer
   */
  [[nodiscard]] auto constexpr mem_type() const
  {
    return static_cast<memory_type>(data_.index() % std::variant_size_v<owning_type_variant>);
  };

  /**
   * @brief Return a boolean indicating whether or not the mdbuffer owns its
   * storage
   */
  [[nodiscard]] auto constexpr is_owning() const
  {
    return data_.index() >= std::variant_size_v<view_type_variant>;
  };

 private:
  template <typename MemTypeConstant>
  [[nodiscard]] auto view()
  {
    if constexpr (MemTypeConstant::value.has_value()) {
      if (is_owning()) {
        if constexpr (std::is_const_v<element_type>) {
          return std::as_const(std::get<owning_type<MemTypeConstant::value.value()>>(data_)).view();
        } else {
          return std::get<owning_type<MemTypeConstant::value.value()>>(data_).view();
        }
      } else {
        return std::get<view_type<MemTypeConstant::value.value()>>(data_);
      }
    } else {
      return std::visit(
        [](auto&& inner) {
          if constexpr (is_mdspan_v<std::remove_reference_t<decltype(inner)>>) {
            return view_type_variant{inner};
          } else {
            if constexpr (std::is_const_v<element_type>) {
              return view_type_variant{std::as_const(inner).view()};
            } else {
              return view_type_variant{inner.view()};
            }
          }
        },
        data_);
    }
  }

  template <typename MemTypeConstant>
  [[nodiscard]] auto view() const
  {
    if constexpr (MemTypeConstant::value.has_value()) {
      if (is_owning()) {
        return make_const_mdspan(
          std::get<owning_type<MemTypeConstant::value.value()>>(data_).view());
      } else {
        return make_const_mdspan(std::get<view_type<MemTypeConstant::value.value()>>(data_));
      }
    } else {
      return std::visit(
        [](auto&& inner) {
          if constexpr (is_mdspan_v<std::remove_reference_t<decltype(inner)>>) {
            return const_view_type_variant{make_const_mdspan(inner)};
          } else {
            return const_view_type_variant{make_const_mdspan(inner.view())};
          }
        },
        data_);
    }
  }

 public:
  /**
   * @brief Return an mdspan of the indicated memory type representing a view
   * on the stored data. If the mdbuffer does not contain data of the indicated
   * memory type, a std::bad_variant_access will be thrown.
   */
  template <memory_type mem_type>
  [[nodiscard]] auto view()
  {
    return view<memory_type_constant<mem_type>>();
  }
  /**
   * @brief Return an mdspan containing const elements of the indicated memory type representing a
   * view on the stored data. If the mdbuffer does not contain data of the indicated memory type, a
   * std::bad_variant_access will be thrown.
   */
  template <memory_type mem_type>
  [[nodiscard]] auto view() const
  {
    return view<memory_type_constant<mem_type>>();
  }
  /**
   * @brief Return a std::variant representing the possible mdspan types that
   * could be returned as views on the mdbuffer. The variant will contain the mdspan
   * corresponding to its current memory type.
   *
   * This method is useful for writing generic code to handle any memory type
   * that might be contained in an mdbuffer at a particular point in a
   * workflow. By performing a `std::visit` on the returned value, the caller
   * can easily dispatch to the correct code path for the memory type.
   */
  [[nodiscard]] auto view() { return view<memory_type_constant<>>(); }
  /**
   * @brief Return a std::variant representing the possible mdspan types that
   * could be returned as const views on the mdbuffer. The variant will contain the mdspan
   * corresponding to its current memory type.
   *
   * This method is useful for writing generic code to handle any memory type
   * that might be contained in an mdbuffer at a particular point in a
   * workflow. By performing a `std::visit` on the returned value, the caller
   * can easily dispatch to the correct code path for the memory type.
   */
  [[nodiscard]] auto view() const { return view<memory_type_constant<>>(); }
};

/**
 * @brief Template checks and helpers to determine if type T is an mdbuffer
 *         or a derived type
 */

template <typename ElementType, typename Extents, typename LayoutPolicy, typename ContainerPolicy>
void __takes_an_mdbuffer_ptr(mdbuffer<ElementType, Extents, LayoutPolicy, ContainerPolicy>*);

template <typename T, typename = void>
struct is_mdbuffer : std::false_type {};
template <typename T>
struct is_mdbuffer<T, std::void_t<decltype(__takes_an_mdbuffer_ptr(std::declval<T*>()))>>
  : std::true_type {};

template <typename T, typename = void>
struct is_input_mdbuffer : std::false_type {};
template <typename T>
struct is_input_mdbuffer<T, std::void_t<decltype(__takes_an_mdbuffer_ptr(std::declval<T*>()))>>
  : std::bool_constant<std::is_const_v<typename T::element_type>> {};

template <typename T, typename = void>
struct is_output_mdbuffer : std::false_type {};
template <typename T>
struct is_output_mdbuffer<T, std::void_t<decltype(__takes_an_mdbuffer_ptr(std::declval<T*>()))>>
  : std::bool_constant<not std::is_const_v<typename T::element_type>> {};

template <typename T>
using is_mdbuffer_t = is_mdbuffer<std::remove_const_t<T>>;

template <typename T>
using is_input_mdbuffer_t = is_input_mdbuffer<T>;

template <typename T>
using is_output_mdbuffer_t = is_output_mdbuffer<T>;

/**
 * @brief Boolean to determine if variadic template types Tn are
 *          raft::mdbuffer or derived types
 */
template <typename... Tn>
inline constexpr bool is_mdbuffer_v = std::conjunction_v<is_mdbuffer_t<Tn>...>;

template <typename... Tn>
using enable_if_mdbuffer = std::enable_if_t<is_mdbuffer_v<Tn...>>;

template <typename... Tn>
inline constexpr bool is_input_mdbuffer_v = std::conjunction_v<is_input_mdbuffer_t<Tn>...>;

template <typename... Tn>
using enable_if_input_mdbuffer = std::enable_if_t<is_input_mdbuffer_v<Tn...>>;

template <typename... Tn>
inline constexpr bool is_output_mdbuffer_v = std::conjunction_v<is_output_mdbuffer_t<Tn>...>;

template <typename... Tn>
using enable_if_output_mdbuffer = std::enable_if_t<is_output_mdbuffer_v<Tn...>>;

/** @} */

}  // namespace RAFT_EXPORT raft
