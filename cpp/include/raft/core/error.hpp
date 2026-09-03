/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __RAFT_RT_ERROR
#define __RAFT_RT_ERROR

#pragma once

#include <raft/core/detail/macros.hpp>
#if defined(__GNUC__) && __has_include(<cxxabi.h>) && __has_include(<execinfo.h>)
#define ENABLE_COLLECT_CALLSTACK
#endif

#include <cstdarg>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <memory>
#include <source_location>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef ENABLE_COLLECT_CALLSTACK
#include <cxxabi.h>
#include <execinfo.h>

#include <sstream>
#endif

namespace RAFT_EXPORT raft {

namespace detail {

/**
 * @brief Write into @p out the message described by @p location, @p location_prefix and @p fmt.
 *
 * Does not throw, so that the caller can `va_end` its arguments before reporting a failure.
 *
 * @param[out] out receives the formatted message; untouched on failure
 * @param[in] location the call site to blame
 * @param[in] location_prefix text placed in front of the location, e.g. "RAFT failure at "
 * @param[in] fmt printf-style format string of the reason
 * @param[in] args arguments of @p fmt, started and ended by the caller
 * @return whether the message could be formatted
 */
[[nodiscard]] inline auto try_vformat_error_message(std::string& out,
                                                    std::source_location location,
                                                    char const* location_prefix,
                                                    char const* fmt,
                                                    std::va_list args) -> bool
{
  char const* location_fmt = "file=%s line=%d function=%s: ";
  char const* file         = location.file_name();
  auto line                = static_cast<int>(location.line());
  char const* function     = location.function_name();

  std::va_list args_measure;
  va_copy(args_measure, args);
  int size1 = std::snprintf(nullptr, 0, "%s", location_prefix);
  int size2 = std::snprintf(nullptr, 0, location_fmt, file, line, function);
  int size3 = std::vsnprintf(nullptr, 0, fmt, args_measure);
  va_end(args_measure);
  if (size1 < 0 || size2 < 0 || size3 < 0) { return false; }

  auto size = static_cast<std::size_t>(size1 + size2 + size3 + 1); /* +1 for final '\0' */
  std::vector<char> buf(size);
  std::snprintf(buf.data(), static_cast<std::size_t>(size1) + 1, "%s", location_prefix);
  std::snprintf(
    buf.data() + size1, static_cast<std::size_t>(size2) + 1, location_fmt, file, line, function);
  std::vsnprintf(buf.data() + size1 + size2, static_cast<std::size_t>(size3) + 1, fmt, args);
  out.assign(buf.data(), buf.data() + size - 1); /* -1 to remove final '\0' */
  return true;
}

}  // namespace detail

/**
 * @defgroup error_handling Exceptions & Error Handling
 * @{
 */

/** base exception class for the whole of raft */
class exception : public std::exception {
 public:
  /** default ctor */
  explicit exception() noexcept : std::exception(), msg_() {}

  /** copy ctor */
  exception(exception const& src) noexcept : std::exception(), msg_(src.what())
  {
    collect_call_stack();
  }

  /** ctor from an input message */
  explicit exception(std::string const msg) noexcept : std::exception(), msg_(std::move(msg))
  {
    collect_call_stack();
  }

  /** get the message associated with this exception */
  char const* what() const noexcept override { return msg_.c_str(); }

 private:
  /** message associated with this exception */
  std::string msg_;

  /** append call stack info to this exception's message for ease of debug */
  // Courtesy: https://www.gnu.org/software/libc/manual/html_node/Backtraces.html
  void collect_call_stack() noexcept
  {
#ifdef ENABLE_COLLECT_CALLSTACK
    constexpr int kSkipFrames    = 1;
    constexpr int kMaxStackDepth = 64;
    void* stack[kMaxStackDepth];  // NOLINT
    auto depth = backtrace(stack, kMaxStackDepth);
    std::ostringstream oss;
    oss << std::endl << "Obtained " << (depth - kSkipFrames) << " stack frames" << std::endl;
    char** strings = backtrace_symbols(stack, depth);
    if (strings == nullptr) {
      oss << "But no stack trace could be found!" << std::endl;
      msg_ += oss.str();
      return;
    }
    // Courtesy: https://panthema.net/2008/0901-stacktrace-demangled/
    for (int i = kSkipFrames; i < depth; i++) {
      oss << "#" << i << " in ";  // beginning of the backtrace line

      char* mangled_name  = nullptr;
      char* offset_begin  = nullptr;
      char* offset_end    = nullptr;
      auto backtrace_line = strings[i];

      // Find parentheses and +address offset surrounding mangled name
      // e.g. ./module(function+0x15c) [0x8048a6d]
      for (char* p = backtrace_line; *p != 0; p++) {
        if (*p == '(') {
          mangled_name = p;
        } else if (*p == '+') {
          offset_begin = p;
        } else if (*p == ')') {
          offset_end = p;
          break;
        }
      }

      // Attempt to demangle the symbol
      if (mangled_name != nullptr && offset_begin != nullptr && offset_end != nullptr &&
          mangled_name + 1 < offset_begin) {
        // Split the backtrace_line
        *mangled_name++ = 0;
        *offset_begin++ = 0;
        *offset_end++   = 0;

        // Demangle the name part
        int status      = 0;
        char* real_name = abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status);

        if (status == 0) {  // Success: substitute the real name
          oss << backtrace_line << ": " << real_name << " +" << offset_begin << offset_end;
        } else {  // Couldn't demangle
          oss << backtrace_line << ": " << mangled_name << " +" << offset_begin << offset_end;
        }
        free(real_name);
      } else {  // Couldn't match the symbol name
        oss << backtrace_line;
      }
      oss << std::endl;
    }
    free(strings);
    msg_ += oss.str();
#endif
  }
};

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * RAFT_EXPECTS and  RAFT_FAIL macros.
 *
 */
struct logic_error : public raft::exception {
  explicit logic_error(char const* const message) : raft::exception(message) {}
  explicit logic_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @brief Exception thrown when attempting to use CUDA features from a non-CUDA
 * build
 *
 */
struct non_cuda_build_error : public raft::exception {
  explicit non_cuda_build_error(char const* const message) : raft::exception(message) {}
  explicit non_cuda_build_error(std::string const& message) : raft::exception(message) {}
};

/**
 * @brief Format an error message that blames the call site given by @p location.
 *
 * A function that reports errors on behalf of its caller takes a `std::source_location` defaulted
 * to `std::source_location::current()` and forwards it here, so that the message points at the
 * caller rather than at the implementation:
 * @code
 *   void sync(cudaStream_t stream, std::source_location loc = std::source_location::current())
 *   {
 *     auto status = cudaStreamSynchronize(stream);
 *     if (status != cudaSuccess) {
 *       throw raft::cuda_error(raft::format_error_message(
 *         loc, "CUDA error encountered at: ", "Reason=%s", cudaGetErrorString(status)));
 *     }
 *   }
 * @endcode
 *
 * The enclosing function is reported next to the file and the line, since it names the template
 * instantiation that the file and the line alone cannot.
 *
 * @param[in] location the call site to blame
 * @param[in] location_prefix text placed in front of the location, e.g. "RAFT failure at "
 * @param[in] fmt printf-style format string describing the reason of the error
 * @param[in] ... arguments of @p fmt; only types that may be passed through `...` are allowed
 * @return the message, of the form "<prefix>file=<file> line=<line> function=<function>: <reason>"
 */
[[nodiscard]] RAFT_FORMAT_PRINTF(3, 4) inline auto format_error_message(
  std::source_location location, char const* location_prefix, char const* fmt, ...) -> std::string
{
  std::string msg{};
  std::va_list args;
  va_start(args, fmt);
  bool const formatted =
    detail::try_vformat_error_message(msg, location, location_prefix, fmt, args);
  va_end(args);
  if (!formatted) { throw raft::exception("Error in snprintf, cannot handle raft exception."); }
  return msg;
}

/**
 * @}
 */

namespace detail {

/**
 * @brief The implementation of the deprecated SET_ERROR_MSG macro.
 *
 * Same as raft::format_error_message; exists as a separate entry point only to carry the
 * deprecation warning of the macro to its call sites.
 */
#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
[[deprecated(
  "SET_ERROR_MSG is deprecated, use raft::format_error_message("
  "std::source_location::current(), location_prefix, fmt, ...) instead")]]
#endif
RAFT_FORMAT_PRINTF(3, 4) inline auto format_error_message_deprecated(std::source_location location,
                                                                     char const* location_prefix,
                                                                     char const* fmt,
                                                                     ...) -> std::string
{
  std::string msg{};
  std::va_list args;
  va_start(args, fmt);
  bool const formatted = try_vformat_error_message(msg, location, location_prefix, fmt, args);
  va_end(args);
  if (!formatted) { throw raft::exception("Error in snprintf, cannot handle raft exception."); }
  return msg;
}

}  // namespace detail

}  // namespace RAFT_EXPORT raft

// FIXME: Need to be replaced with RAFT_FAIL
/** macro to throw a runtime error */
#define THROW(fmt, ...)                                                                       \
  do {                                                                                        \
    int size1 =                                                                               \
      std::snprintf(nullptr, 0, "exception occurred! file=%s line=%d: ", __FILE__, __LINE__); \
    int size2 = std::snprintf(nullptr, 0, fmt, ##__VA_ARGS__);                                \
    if (size1 < 0 || size2 < 0)                                                               \
      throw raft::exception("Error in snprintf, cannot handle raft exception.");              \
    auto size = size1 + size2 + 1; /* +1 for final '\0' */                                    \
    auto buf  = std::make_unique<char[]>(size_t(size));                                       \
    std::snprintf(buf.get(),                                                                  \
                  size1 + 1 /* +1 for '\0' */,                                                \
                  "exception occurred! file=%s line=%d: ",                                    \
                  __FILE__,                                                                   \
                  __LINE__);                                                                  \
    std::snprintf(buf.get() + size1, size2 + 1 /* +1 for '\0' */, fmt, ##__VA_ARGS__);        \
    std::string msg(buf.get(), buf.get() + size - 1); /* -1 to remove final '\0' */           \
    throw raft::exception(msg);                                                               \
  } while (0)

// FIXME: Need to be replaced with RAFT_EXPECTS
/** macro to check for a conditional and assert on failure */
#define ASSERT(check, fmt, ...)              \
  do {                                       \
    if (!(check)) THROW(fmt, ##__VA_ARGS__); \
  } while (0)

/**
 * Macro to append error message to first argument.
 * This should only be called in contexts where it is OK to throw exceptions!
 *
 * @deprecated use raft::format_error_message instead, which does not need a macro to know the call
 * site and can be forwarded a location captured elsewhere.
 */
#define SET_ERROR_MSG(msg, location_prefix, fmt, ...)                        \
  do {                                                                       \
    msg += raft::detail::format_error_message_deprecated(                    \
      std::source_location::current(), location_prefix, fmt, ##__VA_ARGS__); \
  } while (0)

/**
 * @defgroup assertion Assertion and error macros
 * @{
 */

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when a condition is false
 *
 * @param[in] cond Expression that evaluates to true or false
 * @param[in] fmt String literal description of the reason that cond is expected to be true with
 * optional format tagas
 * @throw raft::logic_error if the condition evaluates to false.
 */
#define RAFT_EXPECTS(cond, fmt, ...)                                               \
  do {                                                                             \
    if (!(cond)) {                                                                 \
      throw raft::logic_error(raft::format_error_message(                          \
        std::source_location::current(), "RAFT failure at ", fmt, ##__VA_ARGS__)); \
    }                                                                              \
  } while (0)

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * @param[in] fmt String literal description of the reason that this code path is erroneous with
 * optional format tagas
 * @throw always throws raft::logic_error
 */
#define RAFT_FAIL(fmt, ...)                                                      \
  do {                                                                           \
    throw raft::logic_error(raft::format_error_message(                          \
      std::source_location::current(), "RAFT failure at ", fmt, ##__VA_ARGS__)); \
  } while (0)

/**
 * @}
 */

#endif
