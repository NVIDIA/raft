/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdio>
#include <memory>
#include <source_location>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace raft {

namespace detail {

/**
 * @brief Format a cuda_error message with an explicit call-site location.
 *
 * Mirrors SET_ERROR_MSG / RAFT_CUDA_TRY formatting but does not use those macros, so the reported
 * location is the caller's rather than this header. The enclosing function is reported too, since
 * it names the template instantiation that the file and line alone cannot.
 */
inline std::string format_cuda_launch_error(cudaError_t status, std::source_location location)
{
  char const* location_prefix = "CUDA error encountered at: ";
  char const* location_fmt    = "file=%s line=%d function=%s: ";
  char const* fmt             = "call='%s', Reason=%s:%s";
  char const* call            = "cudaLaunchKernelExC";
  char const* file            = location.file_name();
  auto line                   = static_cast<int>(location.line());
  char const* function        = location.function_name();

  int size1 = std::snprintf(nullptr, 0, "%s", location_prefix);
  int size2 = std::snprintf(nullptr, 0, location_fmt, file, line, function);
  int size3 =
    std::snprintf(nullptr, 0, fmt, call, cudaGetErrorName(status), cudaGetErrorString(status));
  if (size1 < 0 || size2 < 0 || size3 < 0) {
    throw raft::exception("Error in snprintf, cannot handle raft exception.");
  }
  auto size = static_cast<std::size_t>(size1 + size2 + size3 + 1);
  std::vector<char> buf(size);
  std::snprintf(buf.data(), static_cast<std::size_t>(size1) + 1, "%s", location_prefix);
  std::snprintf(
    buf.data() + size1, static_cast<std::size_t>(size2) + 1, location_fmt, file, line, function);
  std::snprintf(buf.data() + size1 + size2,
                static_cast<std::size_t>(size3) + 1,
                fmt,
                call,
                cudaGetErrorName(status),
                cudaGetErrorString(status));
  return std::string(buf.data(), buf.data() + size - 1);
}

/**
 * @brief Launch a kernel, copying the launch arguments into parameters first.
 *
 * Taking the address of a copy rather than of the caller's object means passing a constant (e.g. a
 * @c static @c const data member) does not odr-use it, matching the @c <<<>>> launch syntax.
 */
template <typename... Params>
void dispatch(cudaLaunchConfig_t const& config,
              void* kernel,
              std::source_location location,
              Params... params)
{
  std::array<void*, sizeof...(Params)> arg_ptrs{
    {const_cast<void*>(static_cast<void const*>(std::addressof(params)))...}};
  cudaError_t status = cudaLaunchKernelExC(&config, kernel, arg_ptrs.data());
  if (status == cudaSuccess) { return; }
  cudaGetLastError();  // clear sticky error
  throw raft::cuda_error(format_cuda_launch_error(status, location));
}

}  // namespace detail

/**
 * @defgroup kernel_launch Type-checked CUDA kernel launch
 * @{
 */

/**
 * @brief Where a kernel is launched: the stream, the dynamic shared memory size, and the call site
 * to blame for launch errors.
 *
 * Converts implicitly from raft resources or from a stream, so that a launch reads as a single
 * call and the diagnostics of a failed launch point at the launch expression:
 * @code
 *   raft::launch_kernel(res, grid, block, my_kernel, arg0, arg1);
 *   raft::launch_kernel({stream, smem}, grid, block, my_kernel, arg0, arg1);
 * @endcode
 *
 * Copy and move are deleted and @c launch_kernel takes this by value, so the parameter can only be
 * initialized from a prvalue: an instance stored in a variable can never be launched, and the
 * captured location is therefore always the one of the launch expression.
 */
struct launch_on {
 public:
  /**
   * @param[in] res raft resources providing the stream to launch on
   * @param[in] smem dynamic shared memory size in bytes
   * @param[in] loc call site to blame for launch errors; leave at its default
   */
  launch_on(  // NOLINT(google-explicit-constructor)
    resources const& res,
    std::size_t smem         = 0,
    std::source_location loc = std::source_location::current())
    : launch_on{resource::get_cuda_stream(res), smem, loc}
  {
  }

  /**
   * @param[in] stream stream to launch on
   * @param[in] smem dynamic shared memory size in bytes
   * @param[in] loc call site to blame for launch errors; leave at its default
   */
  launch_on(  // NOLINT(google-explicit-constructor)
    rmm::cuda_stream_view stream,
    std::size_t smem         = 0,
    std::source_location loc = std::source_location::current())
    : launch_on{stream.value(), smem, loc}
  {
  }

  /**
   * @param[in] stream stream to launch on
   * @param[in] smem dynamic shared memory size in bytes
   * @param[in] loc call site to blame for launch errors; leave at its default
   */
  launch_on(  // NOLINT(google-explicit-constructor)
    cudaStream_t stream,
    std::size_t smem         = 0,
    std::source_location loc = std::source_location::current())
    : location{loc}
  {
    config.stream           = stream;
    config.dynamicSmemBytes = smem;
  }

  launch_on(launch_on const&)            = delete;
  launch_on& operator=(launch_on const&) = delete;
  launch_on(launch_on&&)                 = delete;
  launch_on& operator=(launch_on&&)      = delete;
  ~launch_on()                           = default;

  /** Call site to blame for launch errors. */
  std::source_location location;
  /** Launch configuration; the grid and block dimensions are filled in by the launch. */
  cudaLaunchConfig_t config{};
};

/**
 * @brief Launch @p kernel with @p args, which already have the kernel parameter types.
 *
 * The launch arguments are checked against the kernel parameters at compile time, and a failed
 * launch throws @c raft::cuda_error blaming the call site:
 * @code
 *   raft::launch_kernel(res, grid, block, my_kernel, arg0, arg1);
 * @endcode
 *
 * The function-pointer parameter type is a non-deduced context derived from @p args, so a
 * partially specified function template (e.g. @c map_kernel<R, PassOffset>) can still convert to a
 * unique @c __global__ pointer by deducing its remaining template parameters from that type.
 * Overload sets that remain ambiguous after that conversion are not supported.
 *
 * @param[in] where stream to launch on, dynamic shared memory size, and the call site
 * @param[in] grid grid dimensions
 * @param[in] block block dimensions
 * @param[in] kernel the @c __global__ function to launch
 * @param[in] args arguments to pass to @p kernel
 */
template <typename... Args>
void launch_kernel(launch_on where,
                   dim3 grid,
                   dim3 block,
                   std::type_identity_t<void (*)(std::remove_cvref_t<Args>...)> kernel,
                   Args&&... args)
{
  where.config.gridDim  = grid;
  where.config.blockDim = block;
  // Let dispatch deduce its by-value parameter types instead of explicitly forwarding Args.
  // In particular, this drops outermost extended qualifiers such as __restrict__ before dispatch
  // takes the address of each parameter copy for cudaLaunchKernelExC.
  detail::dispatch(
    where.config, reinterpret_cast<void*>(kernel), where.location, std::forward<Args>(args)...);
}

/**
 * @brief Launch @p kernel, converting @p args to the kernel parameter types.
 *
 * Handles call sites where an argument merely converts to its parameter (e.g. @c T* to
 * @c const T*), so they do not need casts. @p kernel must name a single specialization here,
 * because its parameter types are what the arguments are converted to.
 *
 * @param[in] where stream to launch on, dynamic shared memory size, and the call site
 * @param[in] grid grid dimensions
 * @param[in] block block dimensions
 * @param[in] kernel the @c __global__ function to launch
 * @param[in] args arguments to convert and pass to @p kernel
 */
template <typename... Params, typename... Args>
requires(sizeof...(Params) == sizeof...(Args) &&
         !(std::is_same_v<std::remove_cvref_t<Args>, Params> &&
           ...)) void launch_kernel(launch_on where,
                                    dim3 grid,
                                    dim3 block,
                                    void (*kernel)(Params...),
                                    Args&&... args)
{
  static_assert((std::is_convertible_v<Args, Params> && ...),
                "Each launch argument must be convertible to the corresponding kernel parameter");

  where.config.gridDim  = grid;
  where.config.blockDim = block;
  // Convert to the kernel parameter types before dispatch, then let dispatch deduce its by-value
  // parameters so outermost extended qualifiers such as __restrict__ are not preserved.
  detail::dispatch(where.config,
                   reinterpret_cast<void*>(kernel),
                   where.location,
                   static_cast<Params>(std::forward<Args>(args))...);
}

/** @} */  // end group kernel_launch

}  // namespace raft
