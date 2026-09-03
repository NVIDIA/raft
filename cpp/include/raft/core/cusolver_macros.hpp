/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __RAFT_RT_CUSOLVER_MACROS_H
#define __RAFT_RT_CUSOLVER_MACROS_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cusolverDn.h>
#include <cusolverSp.h>

#include <type_traits>

#define _CUSOLVER_ERR_TO_STR(err) \
  case err: return #err;

namespace RAFT_EXPORT raft {

/**
 * @ingroup error_handling
 * @{
 */

/**
 * @brief Exception thrown when a cuSOLVER error is encountered.
 */
struct cusolver_error : public raft::exception {
  explicit cusolver_error(char const* const message) : raft::exception(message) {}
  explicit cusolver_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @}
 */

}  // namespace RAFT_EXPORT raft

namespace raft {
namespace linalg {
namespace detail {

inline const char* cusolver_error_to_string(cusolverStatus_t err)
{
  switch (err) {
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_SUCCESS);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_INITIALIZED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ALLOC_FAILED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INVALID_VALUE);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ARCH_MISMATCH);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_EXECUTION_FAILED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_INTERNAL_ERROR);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_ZERO_PIVOT);
    _CUSOLVER_ERR_TO_STR(CUSOLVER_STATUS_NOT_SUPPORTED);
    default: return "CUSOLVER_STATUS_UNKNOWN";
  };
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft

#undef _CUSOLVER_ERR_TO_STR

/**
 * @ingroup assertion
 * @{
 */

/**
 * @brief Error checking macro for cuSOLVER runtime API functions.
 *
 * Invokes a cuSOLVER runtime API function call, if the call does not return
 * CUSolver_STATUS_SUCCESS, throws an exception detailing the cuSOLVER error that occurred
 */
#define RAFT_CUSOLVER_TRY(call)                                                              \
  do {                                                                                       \
    cusolverStatus_t const status = (call);                                                  \
    if (CUSOLVER_STATUS_SUCCESS != status) {                                                 \
      throw raft::cusolver_error(                                                            \
        raft::format_error_message(std::source_location::current(),                          \
                                   "cuSOLVER error encountered at: ",                        \
                                   "call='%s', Reason=%d:%s",                                \
                                   #call,                                                    \
                                   status,                                                   \
                                   raft::linalg::detail::cusolver_error_to_string(status))); \
    }                                                                                        \
  } while (0)

// FIXME: remove after consumer rename
#ifndef CUSOLVER_TRY
#define CUSOLVER_TRY(call) RAFT_CUSOLVER_TRY(call)
#endif

// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define RAFT_CUSOLVER_TRY_NO_THROW(call)                               \
  do {                                                                 \
    cusolverStatus_t const status = call;                              \
    if (CUSOLVER_STATUS_SUCCESS != status) {                           \
      printf("CUSOLVER call='%s' at file=%s line=%d failed with %s\n", \
             #call,                                                    \
             __FILE__,                                                 \
             __LINE__,                                                 \
             raft::linalg::detail::cusolver_error_to_string(status));  \
    }                                                                  \
  } while (0)

/**
 * @}
 */

// FIXME: remove after cuml rename
#ifndef CUSOLVER_CHECK
#define CUSOLVER_CHECK(call) CUSOLVER_TRY(call)
#endif

#ifndef CUSOLVER_CHECK_NO_THROW
#define CUSOLVER_CHECK_NO_THROW(call) CUSOLVER_TRY_NO_THROW(call)
#endif

#endif
