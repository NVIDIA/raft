/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/nvtx_range_path_stack.hpp>
#include <raft/core/detail/nvtx_range_stack.hpp>

#include <rmm/cuda_stream_view.hpp>

#ifdef NVTX_ENABLED

#include <nvtx3/nvToolsExt.h>

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace raft {
namespace common::nvtx::detail {

/**
 * @brief An internal struct to to initialize the color generator
 */
struct color_gen {
  /** This determines how many bits of the hash to use for the generator */
  using hash_type = uint16_t;
  /** saturation */
  static inline constexpr float kS = 0.9f;
  /** value */
  static inline constexpr float kV = 0.85f;
  /** golden ratio */
  static inline constexpr float kPhi = 1.61803f;
  /** inverse golden ratio */
  static inline constexpr float kInvPhi = 1.f / kPhi;
};

// all h, s, v are in range [0, 1]
// Ref: http://en.wikipedia.org/wiki/HSL_and_HSV#Converting_to_RGB
inline auto hsv2rgb(float h, float s, float v) -> uint32_t
{
  uint32_t out = 0xff000000u;
  if (s <= 0.0f) { return out; }
  // convert hue from [0, 1] range to [0, 360]
  float h_deg = h * 360.f;
  if (0.f > h_deg || h_deg >= 360.f) h_deg = 0.f;
  h_deg /= 60.f;
  int h_range = static_cast<int>(h_deg);
  float h_mod = h_deg - h_range;
  float x     = v * (1.f - s);
  float y     = v * (1.f - (s * h_mod));
  float z     = v * (1.f - (s * (1.f - h_mod)));
  float r, g, b;
  switch (h_range) {
    case 0:
      r = v;
      g = z;
      b = x;
      break;
    case 1:
      r = y;
      g = v;
      b = x;
      break;
    case 2:
      r = x;
      g = v;
      b = z;
      break;
    case 3:
      r = x;
      g = y;
      b = v;
      break;
    case 4:
      r = z;
      g = x;
      b = v;
      break;
    case 5:
    default:
      r = v;
      g = x;
      b = y;
      break;
  }
  out |= (uint32_t(r * 256.f) << 16);
  out |= (uint32_t(g * 256.f) << 8);
  out |= uint32_t(b * 256.f);
  return out;
}

/**
 * @brief Helper method to generate 'visually distinct' colors.
 * Inspired from https://martin.ankerl.com/2009/12/09/how-to-create-random-colors-programmatically/
 * It calculates a hash of the passed string and uses the result to generate
 * distinct yet deterministic colors.
 * Such a thing is very useful for nvtx markers where the ranges associated
 * with a specific tag should ideally get the same color for the purpose of
 * visualizing it on nsight-systems timeline.
 * @param tag a string used as an input to generate a distinct color.
 * @return returns 32b RGB integer with alpha channel set of 0xff
 */
inline auto generate_next_color(const std::string& tag) -> uint32_t
{
  auto x = static_cast<color_gen::hash_type>(std::hash<std::string>{}(tag));
  auto u = std::numeric_limits<color_gen::hash_type>::max();
  auto h = static_cast<float>(x) / static_cast<float>(u);
  h += color_gen::kInvPhi;
  if (h >= 1.f) h -= 1.f;
  return hsv2rgb(h, color_gen::kS, color_gen::kV);
}

template <typename Domain, typename = Domain>
struct domain_store {
  /* If `Domain::name` does not exist, this default instance is used and throws the error. */
  static_assert(sizeof(Domain) != sizeof(Domain),
                "Type used to identify a domain must contain a static member 'char const* name'");
  static inline auto value() -> const nvtxDomainHandle_t { return nullptr; }
};

template <typename Domain>
struct domain_store<
  Domain,
  /* Check if there exists `Domain::name` */
  std::enable_if_t<
    std::is_same<char const*, typename std::decay<decltype(Domain::name)>::type>::value,
    Domain>> {
  static inline auto value() -> const nvtxDomainHandle_t
  {
    // NB: static modifier ensures the domain is created only once
    static const nvtxDomainHandle_t kValue = nvtxDomainCreateA(Domain::name);
    return kValue;
  }
};

template <typename Domain>
inline void push_range_name(const char* name)
{
  nvtxEventAttributes_t event_attrib = {0};
  event_attrib.version               = NVTX_VERSION;
  event_attrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  event_attrib.colorType             = NVTX_COLOR_ARGB;
  event_attrib.color                 = generate_next_color(name);
  event_attrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;
  event_attrib.message.ascii         = name;
  nvtxDomainRangePushEx(domain_store<Domain>::value(), &event_attrib);
  detail::range_name_stack_instance.push(name);  // tracks inner range and depth, cross-thread
  detail::full_range_stack_instance.push(name);  // tracks full range stack, thread-local
}

template <typename Domain, typename... Args>
inline void push_range(const char* format, Args... args)
{
  if constexpr (sizeof...(args) > 0) {
    int length = std::snprintf(nullptr, 0, format, args...);
    if (length > 0) {
      std::vector<char> buf(length + 1);
      std::snprintf(buf.data(), length + 1, format, args...);
      push_range_name<Domain>(buf.data());
    } else {
      push_range_name<Domain>(format);
    }
  } else {
    push_range_name<Domain>(format);
  }
}

template <typename Domain>
inline void pop_range()
{
  detail::range_name_stack_instance.pop();
  detail::full_range_stack_instance.pop();
  nvtxDomainRangePop(domain_store<Domain>::value());
}

}  // namespace common::nvtx::detail
}  // namespace raft

#else   // NVTX_ENABLED

namespace raft {
namespace common::nvtx::detail {

template <typename Domain, typename... Args>
inline void push_range(const char* format, Args... args)
{
}

template <typename Domain>
inline void pop_range()
{
}

}  // namespace common::nvtx::detail
}  // namespace raft
#endif  // NVTX_ENABLED
