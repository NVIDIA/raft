/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>

#include <cusparse.h>

#define _CUSPARSE_ERR_TO_STR(err) \
  case err: return #err;

// Notes:
//(1.) CUDA_VER_12_4_UP aggregates all the CUDA version selection logic;
//(2.) to enforce a lower version,
//
//`#define CUDA_ENFORCE_LOWER
// #include <raft/sparse/detail/cusparse_wrappers.h>`
//
// (i.e., before including this header)
//
#define CUDA_VER_12_4_UP (CUDART_VERSION >= 12040)

namespace RAFT_EXPORT raft {

/**
 * @ingroup error_handling
 * @{
 */

/**
 * @brief Exception thrown when a cuSparse error is encountered.
 */
struct cusparse_error : public raft::exception {
  explicit cusparse_error(char const* const message) : raft::exception(message) {}
  explicit cusparse_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @}
 */
}  // namespace RAFT_EXPORT raft

namespace raft {
namespace sparse {
namespace detail {

inline const char* cusparse_error_to_string(cusparseStatus_t err)
{
  return cusparseGetErrorString(err);
}

}  // namespace detail
}  // namespace sparse
}  // namespace raft

#undef _CUSPARSE_ERR_TO_STR

/**
 * @ingroup assertion
 * @{
 */

/**
 * @brief Error checking macro for cuSparse runtime API functions.
 *
 * Invokes a cuSparse runtime API function call, if the call does not return
 * CUSPARSE_STATUS_SUCCESS, throws an exception detailing the cuSparse error that occurred
 */
#define RAFT_CUSPARSE_TRY(call)                                                              \
  do {                                                                                       \
    cusparseStatus_t const status = (call);                                                  \
    if (CUSPARSE_STATUS_SUCCESS != status) {                                                 \
      throw raft::cusparse_error(                                                            \
        raft::format_error_message(std::source_location::current(),                          \
                                   "cuSparse error encountered at: ",                        \
                                   "call='%s', Reason=%d:%s",                                \
                                   #call,                                                    \
                                   status,                                                   \
                                   raft::sparse::detail::cusparse_error_to_string(status))); \
    }                                                                                        \
  } while (0)

/**
 * @}
 */

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_TRY
#define CUSPARSE_TRY(call) RAFT_CUSPARSE_TRY(call)
#endif

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(call) CUSPARSE_TRY(call)
#endif

/**
 * @ingroup assertion
 * @{
 */
//@todo: use logger here once logging is enabled
/** check for cusparse runtime API errors but do not assert */
#define RAFT_CUSPARSE_TRY_NO_THROW(call)                           \
  do {                                                             \
    cusparseStatus_t err = call;                                   \
    if (err != CUSPARSE_STATUS_SUCCESS) {                          \
      printf("CUSPARSE call='%s' got errorcode=%d err=%s",         \
             #call,                                                \
             err,                                                  \
             raft::sparse::detail::cusparse_error_to_string(err)); \
    }                                                              \
  } while (0)

/**
 * @}
 */

// FIXME: Remove after consumer rename
#ifndef CUSPARSE_CHECK_NO_THROW
#define CUSPARSE_CHECK_NO_THROW(call) RAFT_CUSPARSE_TRY_NO_THROW(call)
#endif
