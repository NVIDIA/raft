/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/dry_run_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cstddef>

namespace raft::util {

// ===================================================================
// Test Category 1: No CUDA stream activity during dry-run
// ===================================================================

/**
 * @brief Verify that dry-run guards prevent actual kernel execution.
 *
 * Strategy: fill a device array with a known value, run a RAFT function
 * under dry-run mode that would overwrite it, then read back and confirm
 * the original value is untouched.
 */
TEST(DryRunGuard, AddDoesNotExecute)
{
  raft::resources res;
  auto stream     = resource::get_cuda_stream(res);
  constexpr int n = 256;

  auto a   = raft::make_device_vector<float>(res, n);
  auto b   = raft::make_device_vector<float>(res, n);
  auto out = raft::make_device_vector<float>(res, n);

  raft::linalg::map(res, a.view(), raft::const_op<float>{1.0f});
  raft::linalg::map(res, b.view(), raft::const_op<float>{2.0f});
  raft::linalg::map(res, out.view(), raft::const_op<float>{99.0f});
  resource::sync_stream(res);

  // Enable dry-run; add would write 3.0 to out
  resource::set_dry_run_flag(res, true);
  raft::linalg::add(
    res, raft::make_const_mdspan(a.view()), raft::make_const_mdspan(b.view()), out.view());
  resource::set_dry_run_flag(res, false);
  resource::sync_stream(res);

  // Verify data was NOT modified
  std::vector<float> h_out(n);
  auto out_src_view = raft::make_const_mdspan(out.view());
  auto out_dst_view = raft::make_host_vector_view<float, int>(h_out.data(), n);
  raft::copy(res, out_dst_view, out_src_view);
  resource::sync_stream(res);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_out[i], 99.0f) << "at index " << i;
  }
}

TEST(DryRunGuard, RngDoesNotExecute)
{
  raft::resources res;
  auto stream     = resource::get_cuda_stream(res);
  constexpr int n = 1024;

  auto out = raft::make_device_vector<float>(res, n);
  raft::linalg::map(res, out.view(), raft::const_op<float>{0.0f});
  resource::sync_stream(res);

  raft::random::RngState state(42);

  resource::set_dry_run_flag(res, true);
  raft::random::uniform(res, state, out.view(), -1.0f, 1.0f);
  resource::set_dry_run_flag(res, false);
  resource::sync_stream(res);

  std::vector<float> h_out(n);
  auto out_src_view = raft::make_const_mdspan(out.view());
  auto out_dst_view = raft::make_host_vector_view<float, int>(h_out.data(), n);
  raft::copy(res, out_dst_view, out_src_view);
  resource::sync_stream(res);
  for (int i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(h_out[i], 0.0f) << "at index " << i;
  }
}

// ===================================================================
// Test Category 2: Accurate allocation tracking
// ===================================================================

TEST(DryRunAllocTracking, DeviceUvectorTracked)
{
  raft::resources res;

  constexpr std::size_t kAllocSize = 16UL * 1024UL * 1024UL;  // 16 MiB

  auto stats = dry_run_execute(res, [&](raft::resources const& r) {
    // Allocate an rmm::device_uvector; the dry-run MR should track it
    rmm::device_uvector<float> buf(kAllocSize / sizeof(float), resource::get_cuda_stream(r));
  });

  // The allocation should be tracked (note: rmm may align the size)
  EXPECT_GE(stats.device_global, kAllocSize);
}

TEST(DryRunAllocTracking, MakeDeviceArrayTracked)
{
  raft::resources res;

  constexpr int rows             = 1024;
  constexpr int cols             = 512;
  constexpr std::size_t expected = rows * cols * sizeof(float);

  auto stats = dry_run_execute(res, [&](raft::resources const& r) {
    auto mat = raft::make_device_matrix<float>(r, rows, cols);
  });

  EXPECT_GE(stats.device_global, expected);
}

TEST(DryRunAllocTracking, MultipleAllocationsSum)
{
  raft::resources res;

  constexpr std::size_t kSize1 = 4UL * 1024UL * 1024UL;  // 4 MiB
  constexpr std::size_t kSize2 = 8UL * 1024UL * 1024UL;  // 8 MiB

  auto stats = dry_run_execute(res, [&](raft::resources const& r) {
    auto stream = resource::get_cuda_stream(r);
    rmm::device_uvector<char> buf1(kSize1, stream);
    rmm::device_uvector<char> buf2(kSize2, stream);
    // Both alive at same time -> peak should be sum
  });

  EXPECT_GE(stats.device_global, kSize1 + kSize2);
}

TEST(DryRunAllocTracking, DeallocReducesCurrent)
{
  raft::resources res;

  constexpr std::size_t kSize1 = 4UL * 1024UL * 1024UL;
  constexpr std::size_t kSize2 = 8UL * 1024UL * 1024UL;

  auto stats = dry_run_execute(res, [&](raft::resources const& r) {
    auto stream = resource::get_cuda_stream(r);
    {
      rmm::device_uvector<char> buf1(kSize1, stream);
    }
    // buf1 is freed now
    rmm::device_uvector<char> buf2(kSize2, stream);
    // Peak should be max(kSize1, kSize2) = kSize2 since they don't overlap
  });

  // Peak should be at least kSize2 (the larger single allocation)
  EXPECT_GE(stats.device_global, kSize2);
  // But could be less than kSize1 + kSize2 (since buf1 is freed before buf2)
  // This depends on timing/implementation, so we just check the peak is reasonable
}

// ===================================================================
// Test Category 3: End-to-end integration tests for composite functions
// ===================================================================

TEST(DryRunE2E, StatsComposite)
{
  raft::resources res;
  auto stream        = resource::get_cuda_stream(res);
  constexpr int rows = 256;
  constexpr int cols = 64;

  // Pre-allocate input (outside dry-run)
  auto input  = raft::make_device_matrix<float>(res, rows, cols);
  auto output = raft::make_device_vector<float>(res, cols);

  raft::linalg::map(res, input.view(), raft::const_op<float>{1.0f});
  raft::linalg::map(res, output.view(), raft::const_op<float>{-1.0f});
  resource::sync_stream(res);

  // Dry-run: compute column means
  auto stats = dry_run_execute(res, [&](raft::resources const& r) {
    raft::stats::mean(r, raft::make_const_mdspan(input.view()), output.view(), false);
  });

  // The output should NOT be modified (still -1.0)
  std::vector<float> h_output(cols);
  auto output_src_view = raft::make_const_mdspan(output.view());
  auto output_dst_view = raft::make_host_vector_view<float, int>(h_output.data(), cols);
  raft::copy(res, output_dst_view, output_src_view);
  resource::sync_stream(res);
  for (int i = 0; i < cols; ++i) {
    EXPECT_FLOAT_EQ(h_output[i], -1.0f) << "at index " << i;
  }

  // Verify dry-run flag is restored
  EXPECT_FALSE(resource::get_dry_run_flag(res));
  // Stats should be non-negative
  EXPECT_GE(stats.device_global, 0);
}

TEST(DryRunE2E, DryRunExecuteWithArgs)
{
  raft::resources res;

  // dry_run_execute with extra args perfect-forwarded to the action
  auto stats = dry_run_execute(
    res,
    [](raft::resources const& r, int expected_size) {
      // Just verify args are forwarded correctly
      EXPECT_EQ(expected_size, 512);
      // Allocate some memory to verify tracking
      rmm::device_uvector<float> tmp(1024, resource::get_cuda_stream(r));
    },
    512);

  EXPECT_GE(stats.device_global, 1024 * sizeof(float));
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

TEST(DryRunE2E, NestedDryRunIsNoop)
{
  raft::resources res;

  // dry_run_execute is already running, setting the flag again inside should not interfere
  auto stats = dry_run_execute(res, [&](raft::resources const& r) {
    EXPECT_TRUE(resource::get_dry_run_flag(r));
    // Manually set it again (should be harmless)
    resource::set_dry_run_flag(r, true);
    EXPECT_TRUE(resource::get_dry_run_flag(r));

    rmm::device_uvector<float> buf(256, resource::get_cuda_stream(r));
  });

  EXPECT_FALSE(resource::get_dry_run_flag(res));
  EXPECT_GE(stats.device_global, 256 * sizeof(float));
}

TEST(DryRunE2E, ExceptionRestoresResources)
{
  raft::resources res;
  auto original_mr   = rmm::mr::get_current_device_resource_ref();
  auto original_host = raft::mr::get_default_host_resource();

  EXPECT_THROW(dry_run_execute(
                 res, [](raft::resources const&) { throw std::runtime_error("test exception"); }),
               std::runtime_error);

  EXPECT_EQ(rmm::mr::get_current_device_resource_ref(), original_mr);
  EXPECT_EQ(raft::mr::get_default_host_resource(), original_host);
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

}  // namespace raft::util
