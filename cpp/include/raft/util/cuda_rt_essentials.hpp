/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
// This file provides a few essential functions that wrap the CUDA runtime API.
// The scope is necessarily limited to ensure that compilation times are
// minimized. Please make sure not to include large / expensive files from here.

#include <raft/core/error.hpp>

#include <cuda_runtime.h>

#include <cstdio>
#include <source_location>

namespace raft {

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
struct cuda_error : public raft::exception {
  explicit cuda_error(char const* const message) : raft::exception(message) {}
  explicit cuda_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @brief Throw raft::cuda_error blaming @p location, unless @p status is cudaSuccess.
 *
 * The function form of RAFT_CUDA_TRY: a utility that checks a CUDA call on behalf of its caller
 * forwards the location it received, so that the reported location is the caller's rather than the
 * utility's.
 *
 * @param[in] status the status returned by a CUDA runtime API call
 * @param[in] call the name of the CUDA runtime API call, as it should appear in the message
 * @param[in] location the call site to blame; leave at its default unless forwarding one
 *
 * @throw raft::cuda_error if @p status is not cudaSuccess
 */
inline void check_cuda_error(cudaError_t status,
                             char const* call,
                             std::source_location location = std::source_location::current())
{
  if (status == cudaSuccess) { return; }
  cudaGetLastError();  // clear the error, so that it does not affect the subsequent calls
  throw cuda_error(format_error_message(location,
                                        "CUDA error encountered at: ",
                                        "call='%s', Reason=%s:%s",
                                        call,
                                        cudaGetErrorName(status),
                                        cudaGetErrorString(status)));
}

}  // namespace raft

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 */
#define RAFT_CUDA_TRY(call) raft::check_cuda_error(call, #call)

/**
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any pending CUDA errors from previous calls. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 */
#ifndef NDEBUG
#define RAFT_CHECK_CUDA(stream) RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
#else
#define RAFT_CHECK_CUDA(stream) RAFT_CUDA_TRY(cudaPeekAtLastError());
#endif

// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define RAFT_CUDA_TRY_NO_THROW(call)                               \
  do {                                                             \
    cudaError_t const status = call;                               \
    if (cudaSuccess != status) {                                   \
      printf("CUDA call='%s' at file=%s line=%d failed with %s\n", \
             #call,                                                \
             __FILE__,                                             \
             __LINE__,                                             \
             cudaGetErrorString(status));                          \
    }                                                              \
  } while (0)
