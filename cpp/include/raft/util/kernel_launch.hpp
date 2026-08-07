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
  char const* call            = "cudaLaunchKernel";
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

inline void throw_on_cuda_launch_error(cudaError_t status, std::source_location location)
{
  if (status == cudaSuccess) { return; }
  cudaGetLastError();  // clear sticky error
  throw raft::cuda_error(format_cuda_launch_error(status, location));
}

}  // namespace detail

/**
 * @brief Temporary object that launches a CUDA kernel with call-site error reporting.
 *
 * Capture @c std::source_location on @ref launch_kernel, then launch via rvalue @c operator().
 * Prefer the one-liner form so diagnostics point at the launch expression:
 * @code
 *   raft::launch_kernel(res, grid, block)(kernel, args...);
 *   raft::launch_kernel(stream, grid, block, smem)(kernel, args...);
 * @endcode
 *
 * The kernel must resolve to a unique @c __global__ function pointer. Partially-specified function
 * templates are OK when the remaining parameters can be deduced from the launch argument types.
 * Overload sets that remain ambiguous after that conversion are not supported.
 */
class kernel_launcher {
 public:
  kernel_launcher(kernel_launcher const&)            = delete;
  kernel_launcher& operator=(kernel_launcher const&) = delete;
  kernel_launcher(kernel_launcher&&)                 = default;
  kernel_launcher& operator=(kernel_launcher&&)      = delete;

  /**
   * @brief Launch @p kernel with @p args, which already have the kernel parameter types.
   *
   * The function-pointer parameter type is a non-deduced context derived from @p args, so a
   * partially specified function template (e.g. @c map_kernel<R, PassOffset>) can still convert to
   * a unique @c __global__ pointer by deducing its remaining template parameters from that type.
   */
  template <typename... Args>
  void operator()(std::type_identity_t<void (*)(std::remove_cvref_t<Args>...)> kernel,
                  Args&&... args) &&
  {
    dispatch_by_value<std::remove_cvref_t<Args>...>(reinterpret_cast<void*>(kernel),
                                                    std::forward<Args>(args)...);
  }

  /**
   * @brief Launch @p kernel, converting @p args to the kernel parameter types.
   *
   * Handles call sites where an argument merely converts to its parameter (e.g. @c T* to
   * @c const T*), so they do not need casts. @p kernel must name a single specialization here,
   * because its parameter types are what the arguments are converted to.
   */
  template <typename... Params, typename... Args>
  requires(sizeof...(Params) == sizeof...(Args) &&
           !(std::is_same_v<std::remove_cvref_t<Args>, Params> && ...)) void
  operator()(void (*kernel)(Params...), Args&&... args) &&
  {
    static_assert((std::is_convertible_v<Args, Params> && ...),
                  "Each launch argument must be convertible to the corresponding kernel parameter");

    dispatch_by_value<Params...>(reinterpret_cast<void*>(kernel), std::forward<Args>(args)...);
  }

 private:
  friend kernel_launcher launch_kernel(
    resources const&, dim3, dim3, std::size_t, std::source_location);
  friend kernel_launcher launch_kernel(
    rmm::cuda_stream_view, dim3, dim3, std::size_t, std::source_location);

  /**
   * @brief Copy the launch arguments into parameters and pass their addresses to @ref dispatch.
   *
   * Taking the address of a copy rather than of the caller's object means passing a constant (e.g.
   * a
   * @c static @c const data member) does not odr-use it, matching the @c <<<>>> launch syntax.
   */
  template <typename... Params>
  void dispatch_by_value(void* kernel, Params... params) const
  {
    std::array<void*, sizeof...(Params)> arg_ptrs{
      {const_cast<void*>(static_cast<void const*>(std::addressof(params)))...}};
    dispatch(kernel, arg_ptrs.data());
  }

  void dispatch(void* kernel, void** arg_ptrs) const
  {
    cudaError_t status =
      cudaLaunchKernel(kernel, grid_, block_, arg_ptrs, shared_mem_bytes_, stream_.value());

    if (status == cudaSuccess) {
#ifndef NDEBUG
      status = cudaStreamSynchronize(stream_.value());
#else
      status = cudaPeekAtLastError();
#endif
    }
    detail::throw_on_cuda_launch_error(status, location_);
  }

  kernel_launcher(rmm::cuda_stream_view stream,
                  dim3 grid,
                  dim3 block,
                  std::size_t shared_mem_bytes,
                  std::source_location location)
    : stream_{stream},
      grid_{grid},
      block_{block},
      shared_mem_bytes_{shared_mem_bytes},
      location_{location}
  {
  }

  rmm::cuda_stream_view stream_;
  dim3 grid_{};
  dim3 block_{};
  std::size_t shared_mem_bytes_{0};
  std::source_location location_{};
};

inline kernel_launcher launch_kernel(
  resources const& res,
  dim3 grid,
  dim3 block,
  std::size_t shared_mem_bytes  = 0,
  std::source_location location = std::source_location::current())
{
  return kernel_launcher{resource::get_cuda_stream(res), grid, block, shared_mem_bytes, location};
}

inline kernel_launcher launch_kernel(
  rmm::cuda_stream_view stream,
  dim3 grid,
  dim3 block,
  std::size_t shared_mem_bytes  = 0,
  std::source_location location = std::source_location::current())
{
  return kernel_launcher{stream, grid, block, shared_mem_bytes, location};
}

}  // namespace raft
