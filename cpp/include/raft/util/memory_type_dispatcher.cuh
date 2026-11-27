/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/detail/macros.hpp>
#include <raft/core/mdbuffer.cuh>
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>

#include <type_traits>
#include <utility>

namespace raft {

namespace detail {

template <typename lambda_t, typename arg_t, typename = void>
struct is_callable : std::false_type {};

template <typename lambda_t, typename arg_t>
struct is_callable<lambda_t,
                   arg_t,
                   std::void_t<decltype(std::declval<lambda_t>()(std::declval<arg_t>()))>>
  : std::true_type {};

template <typename lambda_t,
          typename mdbuffer_type,
          memory_type mem_type,
          enable_if_mdbuffer<mdbuffer_type>* = nullptr>
auto static constexpr is_callable_for_memory_type =
  is_callable<lambda_t, decltype(std::declval<mdbuffer_type>().template view<mem_type>())>::value;

}  // namespace detail

/**
 * @defgroup memory_type_dispatcher Dispatch functor based on memory type
 * @{
 */

/**
 * @brief Dispatch to various specializations of a functor which accepts an
 * mdspan based on the mdspan's memory type
 *
 * This function template is used to dispatch to one or more implementations
 * of a function based on memory type. For instance, if a functor has been
 * implemented with an operator that accepts only a `device_mdspan`, input data
 * can be passed to that functor with minimal copies or allocations by wrapping
 * the functor in this template.
 *
 * More specifically, host memory data will be copied to device before being
 * passed to the functor as a `device_mdspan`. Device, managed, and pinned data
 * will be passed directly to the functor as a `device_mdspan`.
 *
 * If the functor's operator were _also_ specialized for `host_mdspan`, then
 * this wrapper would pass an input `host_mdspan` directly to the corresponding
 * specialization.
 *
 * If a functor explicitly specializes for managed/pinned memory and receives
 * managed/pinned input, the corresponding specialization will be invoked. If the functor does not
 * specialize for either, it will preferentially invoke the device
 * specialization if available and then the host specialization. Managed input
 * will never be dispatched to an explicit specialization for pinned memory and
 * vice versa.
 *
 * Dispatching is performed by coercing the input mdspan to an mdbuffer of the
 * correct type. If it is necessary to coerce the input data to a different
 * data type (e.g. floats to doubles) or to a different memory layout, this can
 * be done by passing an explicit mdbuffer type to the `memory_type_dispatcher`
 * template.
 *
 * Usage example:
 * @code{.cpp}
 * // Functor which accepts only a `device_mdspan` or `managed_mdspan` of
 * // doubles in C-contiguous layout. We wish to be able to call this
 * // functor on any compatible data, regardless of data type, memory type,
 * // or layout.
 * struct functor {
 *    auto operator()(device_matrix_view<double> data) {
 *      // Do something with data on device
 *    };
 *    auto operator()(managed_matrix_view<double> data) {
 *      // Do something with data, taking advantage of knowledge that
 *      // underlying memory is managed
 *    };
 * };
 *
 * auto rows = 3;
 * auto cols = 5;
 * auto res = raft::device_resources{};
 *
 * auto host_data = raft::make_host_matrix<double>(rows, cols);
 * // functor{}(host_data.view()); // This would fail to compile
 * auto device_data = raft::make_device_matrix<double>(res, rows, cols);
 * functor{}(device_data.view()); // Functor accepts device mdspan
 * auto managed_data = raft::make_managed_matrix<double>(res, rows, cols);
 * functor{}(managed_data.view()); // Functor accepts managed mdspan
 * auto pinned_data = raft::make_pinned_matrix<double>(res, rows, cols);
 * // functor{}(pinned_data.view()); // This would fail to compile
 * auto float_data = raft::make_device_matrix<float>(res, rows, cols);
 * // functor{}(float_data.view()); // This would fail to compile
 * auto f_data = raft::make_device_matrix<double, int, raft::layout_f_contiguous>(res, rows, cols);
 * // functor{}(f_data.view()); // This would fail to compile
 *
 * // `memory_type_dispatcher` lets us call this functor on all of the above
 * raft::memory_type_dispatcher(res, functor{}, host_data.view());
 * raft::memory_type_dispatcher(res, functor{}, device_data.view());
 * raft::memory_type_dispatcher(res, functor{}, managed_data.view());
 * raft::memory_type_dispatcher(res, functor{}, pinned_data.view());
 * // Here, we use the mdbuffer type template parameter to ensure that the data
 * // type and layout are as expected by the functor
 * raft::memory_type_dispatcher<raft::mdbuffer<double, matrix_extents<int>>>(res, functor{},
 * float_data.view()); raft::memory_type_dispatcher<raft::mdbuffer<double,
 * matrix_extents<int>>>(res, functor{}, f_data.view());
 * @endcode
 *
 * As this example shows, `memory_type_dispatcher` can be used to dispatch any
 * compatible mdspan input to a functor, regardless of the mdspan type(s) that
 * functor supports.
 */
template <typename lambda_t, typename mdbuffer_type, enable_if_mdbuffer<mdbuffer_type>* = nullptr>
decltype(auto) memory_type_dispatcher(raft::resources const& res, lambda_t&& f, mdbuffer_type&& buf)
{
  if (is_host_device_accessible(buf.mem_type())) {
    // First see if functor has been specialized for this exact memory type
    if constexpr (detail::
                    is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::managed>) {
      if (buf.mem_type() == memory_type::managed) {
        return f(buf.template view<memory_type::managed>());
      }
    }
    if constexpr (detail::
                    is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::pinned>) {
      if (buf.mem_type() == memory_type::pinned) {
        return f(buf.template view<memory_type::pinned>());
      }
    }
  }
  // If the functor is specialized for device and the data are
  // device-accessible, use the device specialization
  if constexpr (detail::is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::device>) {
    if (is_device_accessible(buf.mem_type())) {
      return f(mdbuffer{res, buf, memory_type::device}.template view<memory_type::device>());
    }
    // If there is no host specialization, still use the device specialization
    if constexpr (!detail::
                    is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::host>) {
      return f(mdbuffer{res, buf, memory_type::device}.template view<memory_type::device>());
    }
  }

  // If nothing else has worked, use the host specialization
  if constexpr (detail::is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::host>) {
    return f(mdbuffer{res, buf, memory_type::host}.template view<memory_type::host>());
  }

  // In the extremely rare case that the functor has been specialized _only_
  // for either pinned memory, managed memory, or both, and the input data are
  // neither pinned nor managed, we must perform a copy. In this situation, if
  // we have specializations for both pinned and managed memory, we arbitrarily
  // prefer the managed specialization. Note that if the data _are_ either
  // pinned or managed already, we will have already invoked the correct
  // specialization above.
  if constexpr (detail::
                  is_callable_for_memory_type<lambda_t, mdbuffer_type, memory_type::managed>) {
    return f(mdbuffer{res, buf, memory_type::managed}.template view<memory_type::managed>());
  } else if constexpr (detail::is_callable_for_memory_type<lambda_t,
                                                           mdbuffer_type,
                                                           memory_type::pinned>) {
    return f(mdbuffer{res, buf, memory_type::pinned}.template view<memory_type::pinned>());
  }

  // Suppress warning for unreachable loop. In general, it is a desirable thing
  // for this to be unreachable, but some functors may be specialized in such a
  // way that this is not the case.
#pragma nv_diag_suppress 128
  RAFT_FAIL("The given functor could not be invoked on the provided data");
#pragma nv_diag_default 128
}

template <typename lambda_t, typename mdspan_type, enable_if_mdspan<mdspan_type>* = nullptr>
decltype(auto) memory_type_dispatcher(raft::resources const& res, lambda_t&& f, mdspan_type view)
{
  return memory_type_dispatcher(res, std::forward<lambda_t>(f), mdbuffer{view});
}

template <typename mdbuffer_type,
          typename lambda_t,
          typename mdspan_type,
          enable_if_mdbuffer<mdbuffer_type>* = nullptr,
          enable_if_mdspan<mdspan_type>*     = nullptr>
decltype(auto) memory_type_dispatcher(raft::resources const& res, lambda_t&& f, mdspan_type view)
{
  return memory_type_dispatcher(res, std::forward<lambda_t>(f), mdbuffer_type{res, mdbuffer{view}});
}

/** @} */

}  // namespace raft
