/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "../test_utils.h"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>

#include <gtest/gtest.h>

#include <cstdint>

namespace raft {
TEST(MDSpanCopy, Mdspan3DDeviceDeviceCuda)
{
  auto res                   = device_resources{};
  auto constexpr const depth = std::uint32_t{50};
  auto constexpr const rows  = std::uint32_t{30};
  auto constexpr const cols  = std::uint32_t{20};
  auto in_left = make_device_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto in_right = make_device_mdarray<int, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y, auto&& z) { return x * 7 + y * 11 + z * 13; };

  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        in_left(i, j, k)  = gen_unique_entry(i, j, k);
        in_right(i, j, k) = gen_unique_entry(i, j, k);
      }
    }
  }
  res.sync_stream();

  // Test dtype conversion without transpose
  auto out_long =
    make_device_mdarray<std::int64_t, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
      res, extents<std::uint32_t, depth, rows, cols>{});
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_long.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  execute_with_dry_run_check(
    res,
    [&](raft::resources const& r) { copy(r, out_long.view(), in_left.view()); },
    alloc_behavior::NO_ALLOCATIONS);
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(std::int64_t(out_long(i, j, k)), std::int64_t(gen_unique_entry(i, j, k)));
      }
    }
  }

  // Test transpose
  auto out_left = make_device_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto out_right = make_device_mdarray<int, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});

  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  execute_with_dry_run_check(
    res,
    [&](raft::resources const& r) { copy(r, out_right.view(), in_left.view()); },
    alloc_behavior::NO_ALLOCATIONS);
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(int(out_right(i, j, k)), int(gen_unique_entry(i, j, k)));
      }
    }
  }

  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_left.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  execute_with_dry_run_check(
    res,
    [&](raft::resources const& r) { copy(r, out_left.view(), in_right.view()); },
    alloc_behavior::NO_ALLOCATIONS);
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(int(out_left(i, j, k)), int(gen_unique_entry(i, j, k)));
      }
    }
  }
}

TEST(MDSpanCopy, Mdspan2DDeviceDeviceCuda)
{
  auto res            = device_resources{};
  auto constexpr rows = std::uint32_t{30};
  auto constexpr cols = std::uint32_t{20};
  auto in_left        = make_device_mdarray<float, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto in_right = make_device_mdarray<float, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y) { return x * 7 + y * 11; };

  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      in_left(i, j)  = gen_unique_entry(i, j);
      in_right(i, j) = gen_unique_entry(i, j);
    }
  }

  auto out_left = make_device_mdarray<double, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto out_right = make_device_mdarray<double, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});

  res.sync_stream();

  // Test dtype conversion without transpose
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_right.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_right(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }

  // Test dtype conversion with transpose
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  execute_with_dry_run_check(
    res,
    [&](raft::resources const& r) { copy(r, out_right.view(), in_left.view()); },
    alloc_behavior::NO_ALLOCATIONS);
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_right(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_left.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  execute_with_dry_run_check(
    res,
    [&](raft::resources const& r) { copy(r, out_left.view(), in_right.view()); },
    alloc_behavior::NO_ALLOCATIONS);
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_left(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }
}

TEST(MDSpanCopy, Mdspan2DDeviceDeviceCudaHalfWithTranspose)
{
  auto res            = device_resources{};
  auto constexpr rows = std::uint32_t{30};
  auto constexpr cols = std::uint32_t{20};
  auto in_left        = make_device_mdarray<half, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto in_right = make_device_mdarray<half, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y) { return x * 7 + y * 11; };

  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      in_left(i, j)  = gen_unique_entry(i, j);
      in_right(i, j) = gen_unique_entry(i, j);
    }
  }

  auto out_left = make_device_mdarray<half, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto out_right = make_device_mdarray<half, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});

  res.sync_stream();

  // Test dtype conversion with transpose
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  execute_with_dry_run_check(
    res,
    [&](raft::resources const& r) { copy(r, out_right.view(), in_left.view()); },
    alloc_behavior::NO_ALLOCATIONS);
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(__half2float(out_right(i, j)),
                        __half2float(gen_unique_entry(i, j)),
                        CompareApprox<float>{0.0001}));
    }
  }
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_left.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_left.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(__half2float(out_left(i, j)),
                        __half2float(gen_unique_entry(i, j)),
                        CompareApprox<float>{0.0001}));
    }
  }
}

TEST(MDSpanCopy, Mdspan3DDeviceHostCuda)
{
  auto res                   = device_resources{};
  auto constexpr const depth = std::uint32_t{50};
  auto constexpr const rows  = std::uint32_t{30};
  auto constexpr const cols  = std::uint32_t{20};
  auto in_left = make_device_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto in_right = make_device_mdarray<int, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y, auto&& z) { return x * 7 + y * 11 + z * 13; };

  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        in_left(i, j, k)  = gen_unique_entry(i, j, k);
        in_right(i, j, k) = gen_unique_entry(i, j, k);
      }
    }
  }
  res.sync_stream();

  // Test dtype conversion without transpose
  auto out_long =
    make_host_mdarray<std::int64_t, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
      res, extents<std::uint32_t, depth, rows, cols>{});
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_long.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_long.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(std::int64_t(out_long(i, j, k)), std::int64_t(gen_unique_entry(i, j, k)));
      }
    }
  }

  // Test transpose
  auto out_left = make_host_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto out_right = make_host_mdarray<int, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});

  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_right.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(int(out_right(i, j, k)), int(gen_unique_entry(i, j, k)));
      }
    }
  }

  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_left.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_left.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(int(out_left(i, j, k)), int(gen_unique_entry(i, j, k)));
      }
    }
  }
}

TEST(MDSpanCopy, Mdspan2DDeviceHostCuda)
{
  auto res            = device_resources{};
  auto constexpr rows = std::uint32_t{30};
  auto constexpr cols = std::uint32_t{20};
  auto in_left        = make_host_mdarray<float, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto in_right = make_host_mdarray<float, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y) { return x * 7 + y * 11; };

  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      in_left(i, j)  = gen_unique_entry(i, j);
      in_right(i, j) = gen_unique_entry(i, j);
    }
  }

  auto out_left = make_device_mdarray<double, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto out_right = make_device_mdarray<double, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});

  res.sync_stream();

  // Test dtype conversion without transpose
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_right.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_right(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }

  // Test dtype conversion with transpose
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_right.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_right(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_left.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_left.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_left(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }
}

TEST(MDSpanCopy, Mdspan3DHostDeviceCuda)
{
  auto res                   = device_resources{};
  auto constexpr const depth = std::uint32_t{50};
  auto constexpr const rows  = std::uint32_t{30};
  auto constexpr const cols  = std::uint32_t{20};
  auto in_left = make_device_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto in_right = make_device_mdarray<int, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y, auto&& z) { return x * 7 + y * 11 + z * 13; };

  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        in_left(i, j, k)  = gen_unique_entry(i, j, k);
        in_right(i, j, k) = gen_unique_entry(i, j, k);
      }
    }
  }
  res.sync_stream();

  // Test dtype conversion without transpose
  auto out_long =
    make_device_mdarray<std::int64_t, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
      res, extents<std::uint32_t, depth, rows, cols>{});
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_long.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_long.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(std::int64_t(out_long(i, j, k)), std::int64_t(gen_unique_entry(i, j, k)));
      }
    }
  }

  // Test transpose
  auto out_left = make_device_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto out_right = make_device_mdarray<int, std::uint32_t, layout_f_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});

  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_right.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(int(out_right(i, j, k)), int(gen_unique_entry(i, j, k)));
      }
    }
  }

  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_left.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_left.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        ASSERT_EQ(int(out_left(i, j, k)), int(gen_unique_entry(i, j, k)));
      }
    }
  }
}

TEST(MDSpanCopy, Mdspan2DHostDeviceCuda)
{
  auto res            = device_resources{};
  auto constexpr rows = std::uint32_t{30};
  auto constexpr cols = std::uint32_t{20};
  auto in_left        = make_device_mdarray<float, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto in_right = make_device_mdarray<float, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y) { return x * 7 + y * 11; };

  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      in_left(i, j)  = gen_unique_entry(i, j);
      in_right(i, j) = gen_unique_entry(i, j);
    }
  }

  auto out_left = make_device_mdarray<double, std::uint32_t, layout_c_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});
  auto out_right = make_device_mdarray<double, std::uint32_t, layout_f_contiguous, rows, cols>(
    res, extents<std::uint32_t, rows, cols>{});

  res.sync_stream();

  // Test dtype conversion without transpose
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_right.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_right(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }

  // Test dtype conversion with transpose
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_right.view()), decltype(in_left.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_right.view(), in_left.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_right(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }
  static_assert(
    detail::mdspan_copyable_with_kernel_v<decltype(out_left.view()), decltype(in_right.view())>,
    "Current implementation should use kernel for this copy");
  copy(res, out_left.view(), in_right.view());
  res.sync_stream();
  for (auto i = std::uint32_t{}; i < rows; ++i) {
    for (auto j = std::uint32_t{}; j < cols; ++j) {
      ASSERT_TRUE(match(
        double(out_left(i, j)), double(gen_unique_entry(i, j)), CompareApprox<double>{0.0001}));
    }
  }
}

}  // namespace raft
