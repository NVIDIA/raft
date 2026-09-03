/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>

#include <gtest/gtest.h>

#include <vector>

namespace raft::linalg {

// Matrix dimensions: A (2x3) * B (3x2) = C (2x2)
constexpr int M = 2, N = 2, K = 3;

// Non-trivial alpha and beta constants
float alpha_val = 2.0f;
float beta_val  = 3.0f;

// Input matrices with small integer values (stored as float)
// Matrix A (2x3):
// [ 1,  2,  3]
// [ 4,  5,  6]
std::vector<float> a_host = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

// Matrix B (3x2):
// [ 7,  8]
// [ 9, 10]
// [11, 12]
std::vector<float> b_host = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

// Initial matrix C (2x2):
// [1, 2]
// [3, 4]
std::vector<float> c_host = {1.0f, 2.0f, 3.0f, 4.0f};

// Result matrix (2x2):
// [119, 134]
// [287, 320]
std::vector<float> r_host_allcoefs = {119.0f, 134.0f, 287.0f, 320.0f};

// Result matrix without coefficients (2x2):
// A * B result when alpha=1 (default value), beta=3.0
// [61, 70]
// [148, 166]
std::vector<float> r_host_noalpha = {61.0f, 70.0f, 148.0f, 166.0f};

// Result matrix without coefficients (2x2):
// A * B result when beta=0 (default value), alpha=2.0
// [116, 128]
// [278, 308]
std::vector<float> r_host_nobeta = {116.0f, 128.0f, 278.0f, 308.0f};

// Result matrix without coefficients (2x2):
// A * B result when alpha=1, beta=0 (default values)
// [58, 64]
// [139, 154]
std::vector<float> r_host_nocoefs = {58.0f, 64.0f, 139.0f, 154.0f};

auto get_host_ground_truth(bool use_alpha, bool use_beta)
{
  if (use_alpha && use_beta) {
    return r_host_allcoefs;
  } else if (use_alpha) {
    return r_host_nobeta;
  } else if (use_beta) {
    return r_host_noalpha;
  } else {
    return r_host_nocoefs;
  }
}

void test_gemm_pointer_mode_host(bool use_alpha, bool use_beta)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // Create device matrices
  auto a_device = raft::make_device_matrix<float>(res, M, K);
  auto b_device = raft::make_device_matrix<float>(res, K, N);
  auto c_device = raft::make_device_matrix<float>(res, M, N);

  // Copy data to device
  raft::copy(a_device.data_handle(), a_host.data(), a_host.size(), stream);
  raft::copy(b_device.data_handle(), b_host.data(), b_host.size(), stream);
  raft::copy(c_device.data_handle(), c_host.data(), c_host.size(), stream);

  // Create scalar views for alpha and beta
  auto alpha_scalar = raft::make_host_scalar(res, alpha_val);
  auto beta_scalar  = raft::make_host_scalar(res, beta_val);

  // Perform GEMM: C = alpha * A * B + beta * C
  raft::linalg::gemm(res,
                     a_device.view(),
                     b_device.view(),
                     c_device.view(),
                     use_alpha ? std::make_optional(alpha_scalar.view()) : std::nullopt,
                     use_beta ? std::make_optional(beta_scalar.view()) : std::nullopt);

  // Copy result back to host
  std::vector<float> result(M * N);
  raft::copy(result.data(), c_device.data_handle(), result.size(), stream);
  raft::resource::sync_stream(res);

  // Compare results
  auto gt = get_host_ground_truth(use_alpha, use_beta);
  for (int i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(result[i], gt[i]) << "Mismatch at index " << i;
  }
}

void test_gemm_pointer_mode_device(bool use_alpha, bool use_beta)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // Create device matrices
  auto a_device = raft::make_device_matrix<float>(res, M, K);
  auto b_device = raft::make_device_matrix<float>(res, K, N);
  auto c_device = raft::make_device_matrix<float>(res, M, N);

  // Copy data to device
  raft::copy(a_device.data_handle(), a_host.data(), a_host.size(), stream);
  raft::copy(b_device.data_handle(), b_host.data(), b_host.size(), stream);
  raft::copy(c_device.data_handle(), c_host.data(), c_host.size(), stream);

  // Create scalar views for alpha and beta
  auto alpha_scalar = raft::make_device_scalar(res, alpha_val);
  auto beta_scalar  = raft::make_device_scalar(res, beta_val);

  // Perform GEMM: C = alpha * A * B + beta * C
  raft::linalg::gemm(res,
                     a_device.view(),
                     b_device.view(),
                     c_device.view(),
                     use_alpha ? std::make_optional(alpha_scalar.view()) : std::nullopt,
                     use_beta ? std::make_optional(beta_scalar.view()) : std::nullopt);

  // Copy result back to host
  std::vector<float> result(M * N);
  raft::copy(result.data(), c_device.data_handle(), result.size(), stream);
  raft::resource::sync_stream(res);

  // Compare results
  auto gt = get_host_ground_truth(use_alpha, use_beta);
  for (int i = 0; i < M * N; ++i) {
    EXPECT_FLOAT_EQ(result[i], gt[i]) << "Mismatch at index " << i;
  }
}

TEST(Raft, GemmPointerModeHost) { test_gemm_pointer_mode_host(true, true); }
TEST(Raft, GemmPointerModeHostAlpha) { test_gemm_pointer_mode_host(true, false); }
TEST(Raft, GemmPointerModeHostBeta) { test_gemm_pointer_mode_host(false, true); }
TEST(Raft, GemmPointerModeHostDefaults) { test_gemm_pointer_mode_host(false, false); }
TEST(Raft, GemmPointerModeDevice) { test_gemm_pointer_mode_device(true, true); }
TEST(Raft, GemmPointerModeDeviceAlpha) { test_gemm_pointer_mode_device(true, false); }
TEST(Raft, GemmPointerModeDeviceBeta) { test_gemm_pointer_mode_device(false, true); }
TEST(Raft, GemmPointerModeDeviceDefaults) { test_gemm_pointer_mode_device(false, false); }

TEST(Raft, GemmBatched)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  using index_type                 = int64_t;
  constexpr index_type stride_a    = 8;
  constexpr index_type stride_b    = 8;
  constexpr index_type stride_c    = 5;
  constexpr index_type batch_count = 2;

  // Two column-major A (2 x 3) and B (3 x 2) matrices with padding between batches.
  std::vector<float> a_host = {1, 4, 2, 5, 3, 6, -1, -1, 2, 1, 0, 3, 1, 4, -1, -1};
  std::vector<float> b_host = {7, 9, 11, 8, 10, 12, -1, -1, 1, 0, 2, 3, 1, 4, -1, -1};
  std::vector<float> c_host(stride_c * batch_count, -1);

  auto a_device = raft::make_device_vector<float>(res, a_host.size());
  auto b_device = raft::make_device_vector<float>(res, b_host.size());
  auto c_device = raft::make_device_vector<float>(res, c_host.size());
  raft::copy(a_device.data_handle(), a_host.data(), a_host.size(), stream);
  raft::copy(b_device.data_handle(), b_host.data(), b_host.size(), stream);
  raft::copy(c_device.data_handle(), c_host.data(), c_host.size(), stream);

  auto a_extents = raft::extent_3d<index_type>{batch_count, M, K};
  auto b_extents = raft::extent_3d<index_type>{batch_count, K, N};
  auto c_extents = raft::extent_3d<index_type>{batch_count, M, N};
  auto a_layout =
    raft::make_strided_layout(a_extents, cuda::std::array<index_type, 3>{stride_a, 1, M});
  auto b_layout =
    raft::make_strided_layout(b_extents, cuda::std::array<index_type, 3>{stride_b, 1, K});
  auto c_layout =
    raft::make_strided_layout(c_extents, cuda::std::array<index_type, 3>{stride_c, 1, M});

  auto a_view = raft::device_mdspan<float, decltype(a_extents), raft::layout_stride>{
    a_device.data_handle(), a_layout};
  auto b_view = raft::device_mdspan<float, decltype(b_extents), raft::layout_stride>{
    b_device.data_handle(), b_layout};
  auto c_view = raft::device_mdspan<float, decltype(c_extents), raft::layout_stride>{
    c_device.data_handle(), c_layout};

  std::optional<raft::host_scalar_view<float>> alpha;
  std::optional<raft::host_scalar_view<float>> beta;
  raft::linalg::gemm_batched(
    res, a_view, b_view, c_view, alpha, beta, CUBLAS_COMPUTE_32F_FAST_TF32);

  raft::copy(c_host.data(), c_device.data_handle(), c_host.size(), stream);
  raft::resource::sync_stream(res);

  const std::vector<float> expected = {58, 139, 64, 154, -1, 4, 9, 10, 22, -1};
  EXPECT_EQ(c_host, expected);

  c_host.assign(stride_c * batch_count, -1);
  raft::copy(c_device.data_handle(), c_host.data(), c_host.size(), stream);
  raft::linalg::gemm_batched(res, a_view, b_view, c_view);

  raft::copy(c_host.data(), c_device.data_handle(), c_host.size(), stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(c_host, expected);
}

TEST(Raft, GemmBatchedContiguous)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  using index_type                 = int64_t;
  constexpr index_type batch_count = 2;

  std::vector<float> a_host = {1, 2, 3, 4, 5, 6, 2, 0, 1, 1, 3, 4};
  std::vector<float> b_host = {7, 8, 9, 10, 11, 12, 1, 3, 0, 1, 2, 4};
  std::vector<float> c_host(batch_count * M * N, -1);

  auto a_device = raft::make_device_vector<float>(res, a_host.size());
  auto b_device = raft::make_device_vector<float>(res, b_host.size());
  auto c_device = raft::make_device_vector<float>(res, c_host.size());
  raft::copy(a_device.data_handle(), a_host.data(), a_host.size(), stream);
  raft::copy(b_device.data_handle(), b_host.data(), b_host.size(), stream);
  raft::copy(c_device.data_handle(), c_host.data(), c_host.size(), stream);

  auto a_extents = raft::extent_3d<index_type>{batch_count, M, K};
  auto b_extents = raft::extent_3d<index_type>{batch_count, K, N};
  auto c_extents = raft::extent_3d<index_type>{batch_count, M, N};
  auto a_view    = raft::device_mdspan<float, decltype(a_extents), raft::layout_right>{
    a_device.data_handle(), a_extents};
  auto b_view = raft::device_mdspan<float, decltype(b_extents), raft::layout_right>{
    b_device.data_handle(), b_extents};
  auto c_view = raft::device_mdspan<float, decltype(c_extents), raft::layout_right>{
    c_device.data_handle(), c_extents};

  raft::linalg::gemm_batched(res, a_view, b_view, c_view);

  raft::copy(c_host.data(), c_device.data_handle(), c_host.size(), stream);
  raft::resource::sync_stream(res);

  const std::vector<float> expected = {58, 64, 139, 154, 4, 10, 9, 22};
  EXPECT_EQ(c_host, expected);
}

TEST(Raft, GemmCublasLt136WorkaroundPredicate)
{
  constexpr std::size_t affected_version = 130600;
  const detail::matmul_key_t below_boundary{134217727, 1, 2, 16, 1, 134217727, true, true};
  const detail::matmul_key_t at_boundary{134217728, 1, 2, 16, 1, 134217728, true, true};
  const detail::matmul_key_t above_boundary{134217729, 1, 2, 16, 1, 134217729, true, true};

  const auto needs_workaround = [&](const auto& args) {
    return detail::needs_cublaslt_13_6_workaround(args, affected_version);
  };

  EXPECT_FALSE(needs_workaround(below_boundary));
  EXPECT_TRUE(needs_workaround(at_boundary));
  EXPECT_TRUE(needs_workaround(above_boundary));
  EXPECT_FALSE(detail::needs_cublaslt_13_6_workaround(at_boundary, 130599));
  EXPECT_FALSE(detail::needs_cublaslt_13_6_workaround(at_boundary, 130601));
  EXPECT_FALSE(detail::needs_cublaslt_13_6_workaround(at_boundary, 130700));

  auto different_output    = at_boundary;
  different_output.trans_b = false;
  different_output.n       = 2;
  different_output.ldb     = 7;
  different_output.ldc     = 11;
  EXPECT_TRUE(needs_workaround(different_output));

  const detail::matmul_key_t non_transposed_below{2, 1, 134217727, 16, 134217727, 2, false, false};
  const detail::matmul_key_t non_transposed_at{2, 1, 134217728, 16, 134217728, 2, false, false};
  EXPECT_FALSE(needs_workaround(non_transposed_below));
  EXPECT_TRUE(needs_workaround(non_transposed_at));

  auto invalid_lda = at_boundary;
  invalid_lda.lda  = 0;
  EXPECT_FALSE(needs_workaround(invalid_lda));
}

TEST(Raft, GemmCublasLt136WorkaroundHeuristicArgs)
{
  const auto query_lda = [](uint64_t lda) {
    const detail::matmul_key_t args{134217728, 1, 2, lda, 1, 134217728, true, true};
    return detail::get_cublaslt_13_6_heuristic_args(args).lda;
  };

  EXPECT_EQ(query_lda(12), 13);
  EXPECT_EQ(query_lda(15), 15);
  EXPECT_EQ(query_lda(16), 17);

  const detail::matmul_key_t args{134217728, 1, 2, 16, 1, 134217728, true, true};
  const auto heuristic_args = detail::get_cublaslt_13_6_heuristic_args(args);
  EXPECT_EQ(heuristic_args.m, args.m);
  EXPECT_EQ(heuristic_args.n, args.n);
  EXPECT_EQ(heuristic_args.k, args.k);
  EXPECT_EQ(heuristic_args.ldb, args.ldb);
  EXPECT_EQ(heuristic_args.ldc, args.ldc);
  EXPECT_EQ(heuristic_args.trans_a, args.trans_a);
  EXPECT_EQ(heuristic_args.trans_b, args.trans_b);
}

}  // namespace raft::linalg
