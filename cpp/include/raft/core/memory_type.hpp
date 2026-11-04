/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/core/detail/macros.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#ifndef RAFT_DISABLE_CUDA
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <type_traits>
#include <vector>
#else
#include <raft/core/logger.hpp>
#endif

namespace RAFT_EXPORT raft {
enum class memory_type : std::uint8_t {
  host    = std::uint8_t{0},
  pinned  = std::uint8_t{1},
  device  = std::uint8_t{2},
  managed = std::uint8_t{3}
};

auto constexpr is_device_accessible(memory_type mem_type)
{
  return (mem_type == memory_type::device || mem_type == memory_type::managed ||
          mem_type == memory_type::pinned);
}
auto constexpr is_host_accessible(memory_type mem_type)
{
  return (mem_type == memory_type::host || mem_type == memory_type::managed ||
          mem_type == memory_type::pinned);
}
auto constexpr is_host_device_accessible(memory_type mem_type)
{
  return is_device_accessible(mem_type) && is_host_accessible(mem_type);
}

auto constexpr has_compatible_accessibility(memory_type old_mem_type, memory_type new_mem_type)
{
  return ((!is_device_accessible(new_mem_type) || is_device_accessible(old_mem_type)) &&
          (!is_host_accessible(new_mem_type) || is_host_accessible(old_mem_type)));
}

template <memory_type... mem_types>
struct memory_type_constant {
  static_assert(sizeof...(mem_types) < 2, "At most one memory type can be specified");
  auto static constexpr value = []() {
    auto result = std::optional<memory_type>{};
    if constexpr (sizeof...(mem_types) == 1) { result = std::make_optional(mem_types...); }
    return result;
  }();
};

}  // namespace RAFT_EXPORT raft

namespace raft {
namespace detail {

template <bool is_host_accessible, bool is_device_accessible>
auto constexpr memory_type_from_access()
{
  if constexpr (is_host_accessible && is_device_accessible) {
    return memory_type::managed;
  } else if constexpr (is_host_accessible) {
    return memory_type::host;
  } else if constexpr (is_device_accessible) {
    return memory_type::device;
  }
  static_assert(is_host_accessible || is_device_accessible,
                "Must be either host or device accessible to return a valid memory type");
}

#ifndef RAFT_DISABLE_CUDA
/**
 * @brief Whether `device` reads pageable (unregistered) host memory through the host page tables
 * (ATS, e.g. Grace Hopper / Grace Blackwell).
 *
 * This is a static property of each device, so it is queried once for all devices. The query is
 * deliberately lazy: it runs the first time a pageable pointer is classified, never during static
 * initialization or shared library loading, where CUDA runtime calls are not allowed.
 */
inline auto pageable_memory_uses_host_page_tables(int device) -> bool
{
  static auto const uses_host_page_tables = []() {
    auto device_count = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
    auto result = std::vector<char>(static_cast<std::size_t>(device_count), 0);
    for (auto i = 0; i < device_count; ++i) {
      auto value = 0;
      RAFT_CUDA_TRY(
        cudaDeviceGetAttribute(&value, cudaDevAttrPageableMemoryAccessUsesHostPageTables, i));
      result[static_cast<std::size_t>(i)] = static_cast<char>(value != 0);
    }
    return result;
  }();
  return uses_host_page_tables.at(static_cast<std::size_t>(device)) != 0;
}
#endif

}  // end namespace detail
}  // namespace raft

namespace RAFT_EXPORT raft {

/**
 * @brief How a pointer can be accessed, and the address to use for the device.
 *
 * `device_pointer` is not always the queried pointer: pinned host memory is reached through its
 * device mapping, which can differ from the host address.
 */
template <typename T>
struct pointer_memory_info {
  /** How the memory behind the pointer can be accessed. */
  memory_type type;
  /** The address to use in device code, or nullptr if the memory is not device accessible. */
  T* device_pointer;
};

/**
 * @brief Classify a pointer and get the pointer to use in device code with one pointer query.
 *
 *  - device and managed memory: `device_pointer` is `ptr` itself. This is also the case for an
 *    allocation owned by another device, which `cudaPointerAttributes::devicePointer` reports as
 *    null without peer access even though it is usable once that device is current.
 *  - page-locked host memory (cudaMallocHost, cudaHostRegister): `memory_type::pinned` and the
 *    device mapping, or `memory_type::host` and nullptr when it has no device mapping.
 *  - pageable memory: only device accessible (`memory_type::managed`, `ptr` itself) when the device
 *    reads it through the host page tables (ATS). Otherwise `memory_type::host` and nullptr. On HMM
 *    systems `cudaPointerAttributes::devicePointer` can be non-null for pageable memory, but
 *    device accesses are then serviced through GPU page faults, so it is not device accessible.
 *
 * Costs one `cudaPointerGetAttributes` call. Pageable pointers additionally cost a `cudaGetDevice`
 * call and a lookup in a per device property cache that is filled on first use.
 *
 * @tparam T Pointee type
 * @param ptr Pointer to classify
 */
template <typename T>
auto get_pointer_memory_info(T* ptr) -> pointer_memory_info<T>
{
#ifndef RAFT_DISABLE_CUDA
  auto attrs = cudaPointerAttributes{};
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attrs, ptr));
  switch (attrs.type) {
    case cudaMemoryTypeDevice: return {memory_type::device, ptr};
    case cudaMemoryTypeManaged: return {memory_type::managed, ptr};
    case cudaMemoryTypeHost:
      if (attrs.devicePointer == nullptr) { return {memory_type::host, nullptr}; }
      return {memory_type::pinned, static_cast<T*>(attrs.devicePointer)};
    default: {
      auto device = 0;
      RAFT_CUDA_TRY(cudaGetDevice(&device));
      if (detail::pageable_memory_uses_host_page_tables(device)) {
        return {memory_type::managed, ptr};
      }
      return {memory_type::host, nullptr};
    }
  }
#else
  RAFT_LOG_DEBUG("RAFT compiled without CUDA support, assuming pointer is host pointer");
  return {memory_type::host, nullptr};
#endif
}

/**
 * @brief The memory type of a pointer.
 *
 * @deprecated Use `raft::get_pointer_memory_info`, which also returns the device pointer. The
 * device address of a pointer can differ from the pointer itself, so the memory type alone is not
 * enough to use the memory from device code.
 */
template <typename T>
#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
[[deprecated(
  "memory_type_from_pointer is deprecated, use raft::get_pointer_memory_info, which "
  "also returns the device pointer")]]
#endif
auto memory_type_from_pointer(T* ptr)
{
  return get_pointer_memory_info(ptr).type;
}
}  // namespace RAFT_EXPORT raft
