/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

#include <array>
#include <bitset>
#include <cstddef>
#include <memory>
#include <source_location>
#include <type_traits>
#include <utility>

namespace raft {

namespace detail {

/**
 * @brief How a kernel is launched, beyond the stream and the shared memory size.
 *
 * A default-constructed value launches the kernel normally. The flags are an implementation detail
 * of `launch_on`: they are derived from the resources rather than passed at the call site, so that
 * the behavior of a launch can be changed by configuring the handle.
 */
using launch_flags = std::bitset<32>;

/** Do not launch the kernel at all; set from the dry-run flag resource. */
inline constexpr launch_flags kSkipExecution{1U << 0};

/** Synchronize the stream after the launch, blaming the call site for late errors. */
inline constexpr launch_flags kBlocking{1U << 1};

/**
 * @brief Launch a kernel, copying the launch arguments into parameters first.
 *
 * Taking the address of a copy rather than of the caller's object means passing a constant (e.g. a
 * @c static @c const data member) does not odr-use it, matching the @c <<<>>> launch syntax.
 */
template <typename... Params>
void dispatch(cudaLaunchConfig_t const& config,
              void* kernel,
              launch_flags flags,
              std::source_location location,
              Params... params)
{
  if ((flags & kSkipExecution).any()) { return; }
  std::array<void*, sizeof...(Params)> arg_ptrs{
    {const_cast<void*>(static_cast<void const*>(std::addressof(params)))...}};
  raft::check_cuda_error(
    cudaLaunchKernelExC(&config, kernel, arg_ptrs.data()), "cudaLaunchKernelExC", location);
  if ((flags & kBlocking).any()) { raft::interruptible::synchronize(config.stream, location); }
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
 * Launching on raft resources is dry run compliant: the kernel does not run when the handle has the
 * dry run flag set, so such a launch needs no guard of its own. The stream overloads cannot know
 * that, hence their @c kSkipExecution argument.
 *
 * Copy and move are deleted and @c launch_kernel takes this by value, so the parameter can only be
 * initialized from a prvalue: an instance stored in a variable can never be launched, and the
 * captured location is therefore always the one of the launch expression.
 */
struct launch_on {
 public:
  /**
   * Launch on the stream owned by the resources.
   *
   * The launch is skipped when the handle is in dry run mode; do not add a dry-run guard around it
   * (see `docs/source/dry_run_protocol.md`).
   *
   * @param[in] res raft resources providing the stream to launch on
   * @param[in] smem dynamic shared memory size in bytes
   * @param[in] loc call site to blame for launch errors; leave at its default
   */
  launch_on(  // NOLINT(google-explicit-constructor)
    resources const& res,
    std::size_t smem         = 0,
    std::source_location loc = std::source_location::current())
    : launch_on{resource::get_cuda_stream(res).value(),
                smem,
                resource::get_dry_run_flag(res) ? detail::kSkipExecution : detail::launch_flags{},
                loc}
  {
  }

  /**
   * Launch on an explicit stream, which carries no dry-run state of its own.
   *
   * In code reachable from an API taking `raft::resources`, either launch on the resources instead
   * or pass @p kSkipExecution, otherwise the kernel runs in dry-run mode, which must not execute
   * any CUDA work.
   *
   * @param[in] stream stream to launch on
   * @param[in] smem dynamic shared memory size in bytes
   * @param[in] kSkipExecution whether to skip the launch, e.g. a dry-run flag plumbed by the caller
   * @param[in] loc call site to blame for launch errors; leave at its default
   */
  launch_on(  // NOLINT(google-explicit-constructor)
    rmm::cuda_stream_view stream,
    std::size_t smem         = 0,
    bool kSkipExecution      = false,
    std::source_location loc = std::source_location::current())
    : launch_on{stream.value(), smem, kSkipExecution, loc}
  {
  }

  /**
   * Launch on an explicit stream, which carries no dry-run state of its own.
   *
   * In code reachable from an API taking `raft::resources`, either launch on the resources instead
   * or pass @p kSkipExecution, otherwise the kernel runs in dry-run mode, which must not execute
   * any CUDA work.
   *
   * @param[in] stream stream to launch on
   * @param[in] smem dynamic shared memory size in bytes
   * @param[in] kSkipExecution whether to skip the launch, e.g. a dry-run flag plumbed by the caller
   * @param[in] loc call site to blame for launch errors; leave at its default
   */
  launch_on(  // NOLINT(google-explicit-constructor)
    cudaStream_t stream,
    std::size_t smem         = 0,
    bool kSkipExecution      = false,
    std::source_location loc = std::source_location::current())
    : launch_on{stream, smem, kSkipExecution ? detail::kSkipExecution : detail::launch_flags{}, loc}
  {
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
  /** How to launch; derived from the resources rather than given at the call site. */
  detail::launch_flags flags{};

 private:
  /**
   * The flags are private so that they stay a property of the resources: a call site names a
   * stream, a shared memory size and at most a dry-run flag, never a launch mode.
   */
  launch_on(cudaStream_t stream,
            std::size_t smem,
            detail::launch_flags launch_with,
            std::source_location loc)
    : location{loc}, flags{launch_with}
  {
    config.stream           = stream;
    config.dynamicSmemBytes = smem;
  }
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
  detail::dispatch(where.config,
                   reinterpret_cast<void*>(kernel),
                   where.flags,
                   where.location,
                   std::forward<Args>(args)...);
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
                   where.flags,
                   where.location,
                   static_cast<Params>(std::forward<Args>(args))...);
}

/** @} */  // end group kernel_launch

}  // namespace raft
