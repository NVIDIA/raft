/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/mdbuffer.cuh>
#include <raft/core/pinned_mdarray.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <variant>

namespace raft {

TEST(MDBuffer, FromHost)
{
  auto res             = device_resources{};
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows  = std::uint32_t{3};
  auto constexpr cols  = std::uint32_t{2};
  auto data = make_host_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});

  auto buffer = mdbuffer(data);
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::host);
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::host>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(),
            std::as_const(buffer).view<memory_type::host>().data_handle());
  EXPECT_EQ(buffer.view().index(), variant_index_from_memory_type(memory_type::host));

  buffer = mdbuffer(data.view());
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::host);
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::host>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(),
            std::as_const(buffer).view<memory_type::host>().data_handle());

  auto original_data_handle = data.data_handle();
  buffer                    = mdbuffer(std::move(data));
  EXPECT_TRUE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::host);
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(), original_data_handle);

  auto buffer2 = mdbuffer(res, buffer);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::host);
  EXPECT_EQ(buffer2.view<memory_type::host>().data_handle(),
            buffer.view<memory_type::host>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::host);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::host);
  EXPECT_EQ(buffer2.view<memory_type::host>().data_handle(),
            buffer.view<memory_type::host>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::device);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::device);
  EXPECT_NE(buffer2.view<memory_type::device>().data_handle(),
            buffer.view<memory_type::host>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::managed);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::managed);
  EXPECT_NE(buffer2.view<memory_type::managed>().data_handle(),
            buffer.view<memory_type::host>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::pinned);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::pinned);
  EXPECT_NE(buffer2.view<memory_type::pinned>().data_handle(),
            buffer.view<memory_type::host>().data_handle());
}

TEST(MDBuffer, FromDevice)
{
  auto res             = device_resources{};
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows  = std::uint32_t{3};
  auto constexpr cols  = std::uint32_t{2};

  execute_with_dry_run_check(
    res,
    [&](raft::resources const& h) {
      auto data = make_device_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
        h, extents<std::uint32_t, depth, rows, cols>{});

      auto buffer = mdbuffer(data);
      EXPECT_FALSE(buffer.is_owning());
      EXPECT_EQ(buffer.mem_type(), memory_type::device);
      EXPECT_EQ(buffer.view<memory_type::device>().data_handle(), data.data_handle());

      buffer = mdbuffer(data.view());
      EXPECT_FALSE(buffer.is_owning());
      EXPECT_EQ(buffer.mem_type(), memory_type::device);

      auto original_data_handle = data.data_handle();
      buffer                    = mdbuffer(std::move(data));
      EXPECT_TRUE(buffer.is_owning());
      EXPECT_EQ(buffer.mem_type(), memory_type::device);
      EXPECT_EQ(buffer.view<memory_type::device>().data_handle(), original_data_handle);

      auto buffer2 = mdbuffer(h, buffer);
      EXPECT_FALSE(buffer2.is_owning());
      EXPECT_EQ(buffer2.mem_type(), memory_type::device);

      buffer2 = mdbuffer(h, buffer, memory_type::host);
      EXPECT_TRUE(buffer2.is_owning());
      EXPECT_EQ(buffer2.mem_type(), memory_type::host);

      buffer2 = mdbuffer(h, buffer, memory_type::device);
      EXPECT_FALSE(buffer2.is_owning());
      EXPECT_EQ(buffer2.mem_type(), memory_type::device);

      buffer2 = mdbuffer(h, buffer, memory_type::managed);
      EXPECT_TRUE(buffer2.is_owning());
      EXPECT_EQ(buffer2.mem_type(), memory_type::managed);

      buffer2 = mdbuffer(h, buffer, memory_type::pinned);
      EXPECT_TRUE(buffer2.is_owning());
      EXPECT_EQ(buffer2.mem_type(), memory_type::pinned);
    },
    alloc_behavior::ARGUMENT_DRIVEN,
    depth * rows * cols * sizeof(int));
}

TEST(MDBuffer, FromManaged)
{
  auto res             = device_resources{};
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows  = std::uint32_t{3};
  auto constexpr cols  = std::uint32_t{2};
  auto data = make_managed_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});

  auto buffer = mdbuffer(data);
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::managed);
  EXPECT_EQ(buffer.view<memory_type::managed>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::managed>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::managed>().data_handle(),
            std::as_const(buffer).view<memory_type::managed>().data_handle());
  EXPECT_EQ(buffer.view().index(), variant_index_from_memory_type(memory_type::managed));

  buffer = mdbuffer(data.view());
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::managed);
  EXPECT_EQ(buffer.view<memory_type::managed>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::managed>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::managed>().data_handle(),
            std::as_const(buffer).view<memory_type::managed>().data_handle());

  auto original_data_handle = data.data_handle();
  buffer                    = mdbuffer(std::move(data));
  EXPECT_TRUE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::managed);
  EXPECT_EQ(buffer.view<memory_type::managed>().data_handle(), original_data_handle);

  auto buffer2 = mdbuffer(res, buffer);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::managed);
  EXPECT_EQ(buffer2.view<memory_type::managed>().data_handle(),
            buffer.view<memory_type::managed>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::host);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::host);
  EXPECT_EQ(buffer2.view<memory_type::host>().data_handle(),
            buffer.view<memory_type::managed>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::device);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::device);
  EXPECT_EQ(buffer2.view<memory_type::device>().data_handle(),
            buffer.view<memory_type::managed>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::managed);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::managed);
  EXPECT_EQ(buffer2.view<memory_type::managed>().data_handle(),
            buffer.view<memory_type::managed>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::pinned);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::pinned);
  EXPECT_NE(buffer2.view<memory_type::pinned>().data_handle(),
            buffer.view<memory_type::managed>().data_handle());
}

TEST(MDBuffer, FromPinned)
{
  auto res             = device_resources{};
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows  = std::uint32_t{3};
  auto constexpr cols  = std::uint32_t{2};
  auto data = make_pinned_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});

  auto buffer = mdbuffer(data);
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::pinned);
  EXPECT_EQ(buffer.view<memory_type::pinned>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::pinned>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::pinned>().data_handle(),
            std::as_const(buffer).view<memory_type::pinned>().data_handle());
  EXPECT_EQ(buffer.view().index(), variant_index_from_memory_type(memory_type::pinned));

  buffer = mdbuffer(data.view());
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::pinned);
  EXPECT_EQ(buffer.view<memory_type::pinned>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::pinned>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::pinned>().data_handle(),
            std::as_const(buffer).view<memory_type::pinned>().data_handle());

  auto original_data_handle = data.data_handle();
  buffer                    = mdbuffer(std::move(data));
  EXPECT_TRUE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::pinned);
  EXPECT_EQ(buffer.view<memory_type::pinned>().data_handle(), original_data_handle);

  auto buffer2 = mdbuffer(res, buffer);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::pinned);
  EXPECT_EQ(buffer2.view<memory_type::pinned>().data_handle(),
            buffer.view<memory_type::pinned>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::host);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::host);
  EXPECT_EQ(buffer2.view<memory_type::host>().data_handle(),
            buffer.view<memory_type::pinned>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::device);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::device);
  EXPECT_EQ(buffer2.view<memory_type::device>().data_handle(),
            buffer.view<memory_type::pinned>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::managed);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::managed);
  EXPECT_NE(buffer2.view<memory_type::managed>().data_handle(),
            buffer.view<memory_type::pinned>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::pinned);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::pinned);
  EXPECT_EQ(buffer2.view<memory_type::pinned>().data_handle(),
            buffer.view<memory_type::pinned>().data_handle());
}

TEST(MDBuffer, ImplicitMdspanConversion)
{
  auto res             = device_resources{};
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows  = std::uint32_t{3};
  auto constexpr cols  = std::uint32_t{2};

  using extents_type  = extents<std::uint32_t, depth, rows, cols>;
  auto shared_extents = extents_type{};

  auto data_host = make_host_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, shared_extents);
  auto data_device =
    make_device_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(res,
                                                                                    shared_extents);
  auto data_managed =
    make_managed_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
      res, shared_extents);
  auto data_pinned =
    make_pinned_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(res,
                                                                                    shared_extents);

  auto test_function = [shared_extents](mdbuffer<int, extents_type>&& buf) {
    std::visit([shared_extents](auto view) { EXPECT_EQ(view.extents(), shared_extents); },
               buf.view());
  };

  test_function(data_host);
  test_function(data_device);
  test_function(data_managed);
  test_function(data_pinned);
  test_function(data_host.view());
  test_function(data_device.view());
  test_function(data_managed.view());
  test_function(data_pinned.view());

  auto test_const_function = [shared_extents](mdbuffer<int const, extents_type>&& buf) {
    std::visit([shared_extents](auto view) { EXPECT_EQ(view.extents(), shared_extents); },
               buf.view());
  };

  test_const_function(data_host.view());
  test_const_function(data_device.view());
  test_const_function(data_managed.view());
  test_const_function(data_pinned.view());
}

}  // namespace raft
