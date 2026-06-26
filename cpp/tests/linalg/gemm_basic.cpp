/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

}  // namespace raft::linalg
