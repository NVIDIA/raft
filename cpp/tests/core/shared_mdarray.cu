/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/shared_mdarray.hpp>

#include <gtest/gtest.h>

namespace raft {

TEST(SharedMDArray, MakeSharedFromDeviceMdarray)
{
  raft::resources handle;

  auto src     = raft::make_device_matrix<float>(handle, 4, 4);
  auto src_ptr = src.data_handle();

  auto shared = raft::make_shared_mdarray(std::move(src));

  ASSERT_EQ(shared.data_handle(), src_ptr);
  ASSERT_EQ(shared.extent(0), 4);
  ASSERT_EQ(shared.extent(1), 4);
  ASSERT_EQ(shared.size(), 16u);
}

TEST(SharedMDArray, CopySharesOwnership)
{
  raft::resources handle;

  auto shared1 = raft::make_shared_mdarray(raft::make_device_matrix<float>(handle, 3, 5));

  auto shared2 = shared1;

  ASSERT_EQ(shared1.data_handle(), shared2.data_handle());

  raft::device_matrix_view<float, int> v1 = shared1.view();
  raft::device_matrix_view<float, int> v2 = shared2.view();
  ASSERT_EQ(v1.data_handle(), v2.data_handle());
  ASSERT_EQ(v1.extent(0), v2.extent(0));
  ASSERT_EQ(v1.extent(1), v2.extent(1));
}

TEST(SharedMDArray, SharedOutlivesOriginal)
{
  raft::resources handle;
  float* ptr = nullptr;

  auto shared1 = raft::make_shared_mdarray(
    raft::make_device_vector<float, std::uint32_t>(handle, std::uint32_t{128}));
  ptr = shared1.data_handle();

  decltype(shared1) survivor(handle);
  survivor = shared1;
  shared1  = decltype(shared1)(handle);

  ASSERT_EQ(survivor.data_handle(), ptr);
  ASSERT_EQ(survivor.extent(0), 128);
}

TEST(SharedMDArray, AllocateDirectly)
{
  raft::resources handle;

  auto shared = raft::make_shared_device_matrix<float>(handle, 10, 20);

  ASSERT_EQ(shared.extent(0), 10);
  ASSERT_EQ(shared.extent(1), 20);
  ASSERT_NE(shared.data_handle(), nullptr);

  auto copy = shared;
  ASSERT_EQ(copy.data_handle(), shared.data_handle());
}

TEST(SharedMDArray, HostSharedMdarray)
{
  raft::resources handle;

  auto src = raft::make_host_vector<float>(handle, 10);
  for (int i = 0; i < 10; i++) {
    src(i) = static_cast<float>(i);
  }

  auto shared = raft::make_shared_mdarray(std::move(src));
  auto copy   = shared;

  ASSERT_EQ(copy.data_handle(), shared.data_handle());
  for (int i = 0; i < 10; i++) {
    ASSERT_EQ(shared(i), static_cast<float>(i));
  }
}

TEST(SharedMDArray, ViewTypeIdentity)
{
  static_assert(std::is_same_v<raft::shared_device_matrix<float, int>::view_type,
                               raft::device_matrix<float, int>::view_type>,
                "shared and regular device_matrix must produce the same view_type");

  static_assert(std::is_same_v<raft::shared_device_matrix<float, int>::const_view_type,
                               raft::device_matrix<float, int>::const_view_type>,
                "shared and regular device_matrix must produce the same const_view_type");
}

}  // namespace raft
