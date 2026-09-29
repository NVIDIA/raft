/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/host_mdarray.hpp>
#include <raft/mr/host_memory_resource.hpp>
#include <raft/mr/mmap_memory_resource.hpp>
#include <raft/pmr/resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <memory_resource>

namespace {

TEST(HostMemoryResource, MmapAllocateDeallocate)
{
  raft::mr::mmap_memory_resource mr;
  void* ptr = mr.allocate_sync(4096);
  ASSERT_NE(ptr, nullptr);
  mr.deallocate_sync(ptr, 4096);
}

TEST(HostMemoryResource, MmapHugePages)
{
  raft::mr::mmap_memory_resource mr{raft::mr::kMmapRequestHugePages};
  void* ptr = mr.allocate_sync(4096);
  ASSERT_NE(ptr, nullptr);
  mr.deallocate_sync(ptr, 4096);
}

TEST(HostMemoryResource, MmapFileBacked)
{
  raft::mr::mmap_memory_resource mr{raft::mr::kMmapRequestHugePages | raft::mr::kMmapFileBacked};
  void* ptr = mr.allocate_sync(4096);
  ASSERT_NE(ptr, nullptr);
  mr.deallocate_sync(ptr, 4096);
}

TEST(HostMemoryResource, MmapViaHostResourceRef)
{
  raft::mr::mmap_memory_resource mr;
  raft::mr::synchronous_host_resource_ref ref{mr};
  void* ptr = ref.allocate_sync(4096);
  ASSERT_NE(ptr, nullptr);
  ref.deallocate_sync(ptr, 4096);
}

TEST(HostMemoryResource, MmapHostMdarray)
{
  raft::mr::mmap_memory_resource mmap_mr;
  raft::resources res;
  auto vec = raft::make_host_mdarray<float>(res, mmap_mr, raft::make_extents<uint32_t>(16u));
  vec(0)   = 42.0f;
  vec(15)  = 7.0f;
  ASSERT_EQ(vec(0), 42.0f);
  ASSERT_EQ(vec(15), 7.0f);
}

TEST(HostMemoryResource, GetDefaultHostResource)
{
  auto ref  = raft::mr::get_default_host_resource();
  void* ptr = ref.allocate_sync(128);
  ASSERT_NE(ptr, nullptr);
  ref.deallocate_sync(ptr, 128);
}

TEST(HostMemoryResource, SetHostResourceMmap)
{
  raft::mr::mmap_memory_resource mmap_mr;
  raft::mr::set_default_host_resource(mmap_mr);

  auto ref  = raft::mr::get_default_host_resource();
  void* ptr = ref.allocate_sync(4096);
  ASSERT_NE(ptr, nullptr);
  ref.deallocate_sync(ptr, 4096);

  // restore default
  raft::mr::set_default_host_resource(raft::mr::new_delete_resource());
}

TEST(HostMemoryResource, StdPmrSyncAdaptor)
{
  std::pmr::synchronized_pool_resource pool{std::pmr::get_default_resource()};
  raft::pmr::resource_adaptor adaptor{&pool};
  void* ptr = adaptor.allocate_sync(256);
  ASSERT_NE(ptr, nullptr);
  adaptor.deallocate_sync(ptr, 256);
}

}  // namespace
