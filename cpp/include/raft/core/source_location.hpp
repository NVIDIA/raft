/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>

#include <cstdint>
#include <source_location>

namespace RAFT_EXPORT raft {

#if defined(RAFT_COMPACT_DIAGNOSTICS) && RAFT_COMPACT_DIAGNOSTICS
/**
 * @brief File/line-only diagnostic location. current() captures the caller, including in defaults.
 *
 * Deliberately uses no __builtin_FUNCTION, __PRETTY_FUNCTION__, or source_location::current:
 * even an unused function_name() field can retain template-expanded signature strings.
 * Explicit std::source_location arguments remain accepted, but the caller may retain its signature.
 * Use the same mode for every translation unit and consumer of these headers.
 */
class source_location {
 public:
  constexpr source_location() noexcept = default;
  constexpr source_location(std::source_location location) noexcept
    : file_(location.file_name()), line_(location.line())
  {
  }

  static constexpr source_location current(char const* file         = __builtin_FILE(),
                                           std::uint_least32_t line = __builtin_LINE()) noexcept
  {
    return source_location(file, line);
  }

  constexpr char const* file_name() const noexcept { return file_; }
  constexpr std::uint_least32_t line() const noexcept { return line_; }
  constexpr char const* function_name() const noexcept { return ""; }
  constexpr std::uint_least32_t column() const noexcept { return 0; }

 private:
  constexpr source_location(char const* file, std::uint_least32_t line) noexcept
    : file_(file), line_(line)
  {
  }

  char const* file_         = "";
  std::uint_least32_t line_ = 0;
};
#else
using source_location = std::source_location;
#endif

}  // namespace RAFT_EXPORT raft
