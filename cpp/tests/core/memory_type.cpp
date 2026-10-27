/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <raft/core/memory_type.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

namespace raft {
TEST(MemoryType, IsDeviceAccessible)
{
  static_assert(!is_device_accessible(memory_type::host));
  static_assert(is_device_accessible(memory_type::device));
  static_assert(is_device_accessible(memory_type::managed));
  static_assert(is_device_accessible(memory_type::pinned));
}

TEST(MemoryType, IsHostAccessible)
{
  static_assert(is_host_accessible(memory_type::host));
  static_assert(!is_host_accessible(memory_type::device));
  static_assert(is_host_accessible(memory_type::managed));
  static_assert(is_host_accessible(memory_type::pinned));
}

TEST(MemoryType, IsHostDeviceAccessible)
{
  static_assert(!is_host_device_accessible(memory_type::host));
  static_assert(!is_host_device_accessible(memory_type::device));
  static_assert(is_host_device_accessible(memory_type::managed));
  static_assert(is_host_device_accessible(memory_type::pinned));
}

#ifndef RAFT_DISABLE_CUDA
TEST(GetPointerMemoryInfo, Pinned)
{
  auto ptr = static_cast<void*>(nullptr);
  cudaMallocHost(&ptr, 1);
  auto info = get_pointer_memory_info(ptr);
  EXPECT_EQ(info.type, memory_type::pinned);
  auto attrs = cudaPointerAttributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attrs, ptr), cudaSuccess);
  EXPECT_NE(info.device_pointer, nullptr);
  EXPECT_EQ(info.device_pointer, attrs.devicePointer);
  cudaFreeHost(ptr);
}

TEST(GetPointerMemoryInfo, RegisteredHost)
{
  constexpr auto size = std::size_t{1} << 16;
  auto* buf           = new char[size];
  ASSERT_EQ(cudaHostRegister(buf, size, cudaHostRegisterMapped), cudaSuccess);
  auto info = get_pointer_memory_info(buf);
  EXPECT_EQ(info.type, memory_type::pinned);
  auto attrs = cudaPointerAttributes{};
  ASSERT_EQ(cudaPointerGetAttributes(&attrs, buf), cudaSuccess);
  EXPECT_NE(info.device_pointer, nullptr);
  EXPECT_EQ(static_cast<void*>(info.device_pointer), attrs.devicePointer);
  cudaHostUnregister(buf);
  delete[] buf;
}

TEST(GetPointerMemoryInfo, Pageable)
{
  // Pageable memory is only device accessible when the device uses the host page tables (ATS).
  // On HMM systems cudaPointerGetAttributes may report a non-null devicePointer for it, but that
  // must not make it device accessible.
  auto* buf = new char[1 << 16];
  int ats   = 0;
  cudaDeviceGetAttribute(&ats, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
  auto info = get_pointer_memory_info(buf);
  if (ats != 0) {
    EXPECT_EQ(info.type, memory_type::managed);
    EXPECT_EQ(info.device_pointer, buf);
  } else {
    EXPECT_EQ(info.type, memory_type::host);
    EXPECT_EQ(info.device_pointer, nullptr);
  }
  delete[] buf;
}

TEST(GetPointerMemoryInfo, Device)
{
  auto ptr = static_cast<void*>(nullptr);
  cudaMalloc(&ptr, 1);
  auto info = get_pointer_memory_info(ptr);
  EXPECT_EQ(info.type, memory_type::device);
  EXPECT_EQ(info.device_pointer, ptr);
  cudaFree(ptr);
}

TEST(GetPointerMemoryInfo, DeviceOwnedByAnotherDevice)
{
  // cudaPointerAttributes::devicePointer is relative to the current device and can be null for an
  // allocation owned by another device, but the allocation is still usable once that device is
  // current, so it must be reported as device memory.
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) { GTEST_SKIP() << "Requires at least two devices"; }
  int current = 0;
  ASSERT_EQ(cudaGetDevice(&current), cudaSuccess);
  auto other = (current + 1) % device_count;
  auto ptr   = static_cast<void*>(nullptr);
  ASSERT_EQ(cudaSetDevice(other), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&ptr, 1), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(current), cudaSuccess);
  auto info = get_pointer_memory_info(ptr);
  EXPECT_EQ(info.type, memory_type::device);
  EXPECT_EQ(info.device_pointer, ptr);
  ASSERT_EQ(cudaSetDevice(other), cudaSuccess);
  cudaFree(ptr);
  ASSERT_EQ(cudaSetDevice(current), cudaSuccess);
}

TEST(GetPointerMemoryInfo, Managed)
{
  auto ptr = static_cast<void*>(nullptr);
  cudaMallocManaged(&ptr, 1);
  auto info = get_pointer_memory_info(ptr);
  EXPECT_EQ(info.type, memory_type::managed);
  EXPECT_EQ(info.device_pointer, ptr);
  cudaFree(ptr);
}

TEST(GetPointerMemoryInfo, ConstPointer)
{
  auto ptr = static_cast<void*>(nullptr);
  cudaMalloc(&ptr, 16 * sizeof(float));
  auto const* const_ptr = static_cast<float const*>(ptr);
  auto info             = get_pointer_memory_info(const_ptr);
  static_assert(std::is_same_v<decltype(info.device_pointer), float const*>);
  EXPECT_EQ(info.type, memory_type::device);
  EXPECT_EQ(info.device_pointer, const_ptr);
  cudaFree(ptr);
}

TEST(MemoryTypeFromPointer, MatchesGetPointerMemoryInfo)
{
  // The deprecated wrapper reports the same classification
  auto ptr = static_cast<void*>(nullptr);
  cudaMalloc(&ptr, 1);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  EXPECT_EQ(memory_type_from_pointer(ptr), get_pointer_memory_info(ptr).type);
#pragma GCC diagnostic pop
  cudaFree(ptr);
}
#endif
}  // namespace raft
