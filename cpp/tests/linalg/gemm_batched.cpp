/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace raft::linalg {

namespace {

constexpr int kBatch = 3;
constexpr int kM     = 4;
constexpr int kK     = 5;
constexpr int kN     = 2;

constexpr float kAlpha = 2.0f;
constexpr float kBeta  = 3.0f;

/** Deterministic small-integer values, so that every result is exactly representable in float. */
auto logical_values(int batch, int rows, int cols, int seed, bool same_for_all_batches = false)
  -> std::vector<float>
{
  std::vector<float> out(static_cast<size_t>(batch) * rows * cols);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        const auto x = seed + (same_for_all_batches ? 0 : 7 * b) + 3 * i + 2 * j;
        out[(static_cast<size_t>(b) * rows + i) * cols + j] = static_cast<float>(x % 11 - 5);
      }
    }
  }
  return out;
}

/** Strides of a batch of row- or column-major matrices. */
auto strides_of(int rows, int cols, bool col_major, int batch_stride) -> cuda::std::array<int, 3>
{
  return col_major ? cuda::std::array<int, 3>{batch_stride, 1, rows}
                   : cuda::std::array<int, 3>{batch_stride, cols, 1};
}

auto buffer_size(int batch, int rows, int cols, const cuda::std::array<int, 3>& strides) -> size_t
{
  return static_cast<size_t>((batch - 1) * strides[0] + (rows - 1) * strides[1] +
                             (cols - 1) * strides[2]) +
         1;
}

/** Lay logical (batch, rows, cols) values out in memory as described by `strides`. */
auto pack(const std::vector<float>& logical,
          int batch,
          int rows,
          int cols,
          const cuda::std::array<int, 3>& strides) -> std::vector<float>
{
  std::vector<float> out(buffer_size(batch, rows, cols, strides), 0.0f);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        out[b * strides[0] + i * strides[1] + j * strides[2]] =
          logical[(static_cast<size_t>(b) * rows + i) * cols + j];
      }
    }
  }
  return out;
}

/** Inverse of `pack`. */
auto unpack(const std::vector<float>& packed,
            int batch,
            int rows,
            int cols,
            const cuda::std::array<int, 3>& strides) -> std::vector<float>
{
  std::vector<float> out(static_cast<size_t>(batch) * rows * cols);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        out[(static_cast<size_t>(b) * rows + i) * cols + j] =
          packed[b * strides[0] + i * strides[1] + j * strides[2]];
      }
    }
  }
  return out;
}

/** Host reference: z[b] = alpha * x[b] * y[b] + beta * z[b], all operands logically row-major. */
auto reference(const std::vector<float>& x,
               const std::vector<float>& y,
               const std::vector<float>& z,
               int batch,
               int m,
               int k,
               int n,
               float alpha,
               float beta) -> std::vector<float>
{
  std::vector<float> out(static_cast<size_t>(batch) * m * n);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int p = 0; p < k; ++p) {
          acc += x[(static_cast<size_t>(b) * m + i) * k + p] *
                 y[(static_cast<size_t>(b) * k + p) * n + j];
        }
        const auto idx = (static_cast<size_t>(b) * m + i) * n + j;
        out[idx]       = alpha * acc + beta * z[idx];
      }
    }
  }
  return out;
}

auto make_strided_view(float* ptr, int batch, int rows, int cols, cuda::std::array<int, 3> strides)
{
  return raft::device_mdspan<float, raft::extent_3d<int>, raft::layout_stride>{
    ptr, raft::make_strided_layout(raft::extent_3d<int>{batch, rows, cols}, strides)};
}

struct gemm_batched_params {
  bool x_col_major;
  bool y_col_major;
  bool z_col_major;
  bool use_alpha;
  bool use_beta;
  bool device_scalars;
  /** Distance between consecutive matrices of X; -1 means tightly packed. */
  int x_batch_stride = -1;
};

void test_gemm_batched(const gemm_batched_params& ps)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  // A zero batch stride broadcasts a single matrix of X over the whole batch, which only matches
  // the reference if every batch of the logical X holds the same values.
  const bool broadcast_x = ps.x_batch_stride == 0;
  auto x_logical         = logical_values(kBatch, kM, kK, 1, broadcast_x);
  auto y_logical         = logical_values(kBatch, kK, kN, 4);
  auto z_logical         = logical_values(kBatch, kM, kN, 9);

  auto x_strides =
    strides_of(kM, kK, ps.x_col_major, ps.x_batch_stride >= 0 ? ps.x_batch_stride : kM * kK);
  auto y_strides = strides_of(kK, kN, ps.y_col_major, kK * kN);
  auto z_strides = strides_of(kM, kN, ps.z_col_major, kM * kN);

  auto x_packed = pack(x_logical, kBatch, kM, kK, x_strides);
  auto y_packed = pack(y_logical, kBatch, kK, kN, y_strides);
  auto z_packed = pack(z_logical, kBatch, kM, kN, z_strides);

  rmm::device_uvector<float> x_device(x_packed.size(), stream);
  rmm::device_uvector<float> y_device(y_packed.size(), stream);
  rmm::device_uvector<float> z_device(z_packed.size(), stream);
  raft::copy(x_device.data(), x_packed.data(), x_packed.size(), stream);
  raft::copy(y_device.data(), y_packed.data(), y_packed.size(), stream);
  raft::copy(z_device.data(), z_packed.data(), z_packed.size(), stream);

  auto x_view = make_strided_view(x_device.data(), kBatch, kM, kK, x_strides);
  auto y_view = make_strided_view(y_device.data(), kBatch, kK, kN, y_strides);
  auto z_view = make_strided_view(z_device.data(), kBatch, kM, kN, z_strides);

  if (ps.device_scalars) {
    auto alpha = raft::make_device_scalar(res, kAlpha);
    auto beta  = raft::make_device_scalar(res, kBeta);
    gemm_batched(res,
                 x_view,
                 y_view,
                 z_view,
                 ps.use_alpha ? std::make_optional(alpha.view()) : std::nullopt,
                 ps.use_beta ? std::make_optional(beta.view()) : std::nullopt);
  } else {
    auto alpha = raft::make_host_scalar(kAlpha);
    auto beta  = raft::make_host_scalar(kBeta);
    gemm_batched(res,
                 x_view,
                 y_view,
                 z_view,
                 ps.use_alpha ? std::make_optional(alpha.view()) : std::nullopt,
                 ps.use_beta ? std::make_optional(beta.view()) : std::nullopt);
  }

  std::vector<float> result_packed(z_packed.size());
  raft::copy(result_packed.data(), z_device.data(), result_packed.size(), stream);
  raft::resource::sync_stream(res);

  auto result = unpack(result_packed, kBatch, kM, kN, z_strides);
  auto gt     = reference(x_logical,
                      y_logical,
                      z_logical,
                      kBatch,
                      kM,
                      kK,
                      kN,
                      ps.use_alpha ? kAlpha : 1.0f,
                      ps.use_beta ? kBeta : 0.0f);
  for (size_t i = 0; i < gt.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], gt[i]) << "Mismatch at index " << i;
  }
}

}  // namespace

class GemmBatchedLayoutTest : public ::testing::TestWithParam<gemm_batched_params> {};

TEST_P(GemmBatchedLayoutTest, MatchesReference) { test_gemm_batched(GetParam()); }

INSTANTIATE_TEST_CASE_P(GemmBatched,
                        GemmBatchedLayoutTest,
                        ::testing::Values(
                          // every combination of per-matrix layouts, with both coefficients
                          gemm_batched_params{false, false, false, true, true, false},
                          gemm_batched_params{false, false, true, true, true, false},
                          gemm_batched_params{false, true, false, true, true, false},
                          gemm_batched_params{false, true, true, true, true, false},
                          gemm_batched_params{true, false, false, true, true, false},
                          gemm_batched_params{true, false, true, true, true, false},
                          gemm_batched_params{true, true, false, true, true, false},
                          gemm_batched_params{true, true, true, true, true, false},
                          // default coefficients
                          gemm_batched_params{false, false, false, false, false, false},
                          gemm_batched_params{false, false, false, true, false, false},
                          gemm_batched_params{false, false, false, false, true, false},
                          gemm_batched_params{true, true, true, false, false, false},
                          // device pointer mode
                          gemm_batched_params{false, false, false, true, true, true},
                          gemm_batched_params{true, true, true, true, true, true},
                          gemm_batched_params{false, false, false, false, false, true},
                          // padding between consecutive matrices of X
                          gemm_batched_params{false, false, false, true, true, false, kM* kK + 3},
                          gemm_batched_params{true, true, true, true, true, false, kM* kK + 3},
                          // a single X broadcast over the whole batch
                          gemm_batched_params{false, false, false, true, true, false, 0},
                          gemm_batched_params{true, false, true, true, true, false, 0}));

// A plain row_major 3D mdarray is a batch of row-major matrices; this is the layout the API is
// expected to be used with most often, so check it goes through without a strided layout.
TEST(GemmBatched, RowMajorMdarray)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  auto x_logical = logical_values(kBatch, kM, kK, 1);
  auto y_logical = logical_values(kBatch, kK, kN, 4);
  auto z_logical = logical_values(kBatch, kM, kN, 9);

  auto x = raft::make_device_mdarray<float, int, raft::row_major>(
    res, raft::extent_3d<int>{kBatch, kM, kK});
  auto y = raft::make_device_mdarray<float, int, raft::row_major>(
    res, raft::extent_3d<int>{kBatch, kK, kN});
  auto z = raft::make_device_mdarray<float, int, raft::row_major>(
    res, raft::extent_3d<int>{kBatch, kM, kN});
  raft::copy(x.data_handle(), x_logical.data(), x_logical.size(), stream);
  raft::copy(y.data_handle(), y_logical.data(), y_logical.size(), stream);
  raft::copy(z.data_handle(), z_logical.data(), z_logical.size(), stream);

  auto alpha = raft::make_host_scalar(kAlpha);
  auto beta  = raft::make_host_scalar(kBeta);
  gemm_batched(res,
               x.view(),
               y.view(),
               z.view(),
               std::make_optional(alpha.view()),
               std::make_optional(beta.view()));

  std::vector<float> result(z_logical.size());
  raft::copy(result.data(), z.data_handle(), result.size(), stream);
  raft::resource::sync_stream(res);

  auto gt = reference(x_logical, y_logical, z_logical, kBatch, kM, kK, kN, kAlpha, kBeta);
  for (size_t i = 0; i < gt.size(); ++i) {
    EXPECT_FLOAT_EQ(result[i], gt[i]) << "Mismatch at index " << i;
  }
}

// The batched result must agree with calling the non-batched gemm once per matrix.
TEST(GemmBatched, MatchesUnbatchedGemm)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  auto x_logical = logical_values(kBatch, kM, kK, 1);
  auto y_logical = logical_values(kBatch, kK, kN, 4);

  auto x = raft::make_device_mdarray<float, int, raft::row_major>(
    res, raft::extent_3d<int>{kBatch, kM, kK});
  auto y = raft::make_device_mdarray<float, int, raft::row_major>(
    res, raft::extent_3d<int>{kBatch, kK, kN});
  auto z_batched = raft::make_device_mdarray<float, int, raft::row_major>(
    res, raft::extent_3d<int>{kBatch, kM, kN});
  auto z_looped = raft::make_device_matrix<float, int, raft::row_major>(res, kBatch * kM, kN);
  raft::copy(x.data_handle(), x_logical.data(), x_logical.size(), stream);
  raft::copy(y.data_handle(), y_logical.data(), y_logical.size(), stream);

  gemm_batched(res, x.view(), y.view(), z_batched.view());

  for (int b = 0; b < kBatch; ++b) {
    auto x_b = raft::make_device_matrix_view<float, int, raft::row_major>(
      x.data_handle() + static_cast<size_t>(b) * kM * kK, kM, kK);
    auto y_b = raft::make_device_matrix_view<float, int, raft::row_major>(
      y.data_handle() + static_cast<size_t>(b) * kK * kN, kK, kN);
    auto z_b = raft::make_device_matrix_view<float, int, raft::row_major>(
      z_looped.data_handle() + static_cast<size_t>(b) * kM * kN, kM, kN);
    gemm(res, x_b, y_b, z_b);
  }

  std::vector<float> batched(static_cast<size_t>(kBatch) * kM * kN);
  std::vector<float> looped(batched.size());
  raft::copy(batched.data(), z_batched.data_handle(), batched.size(), stream);
  raft::copy(looped.data(), z_looped.data_handle(), looped.size(), stream);
  raft::resource::sync_stream(res);

  for (size_t i = 0; i < batched.size(); ++i) {
    EXPECT_FLOAT_EQ(batched[i], looped[i]) << "Mismatch at index " << i;
  }
}

TEST(GemmBatched, RejectsNonContiguousMatrices)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);

  rmm::device_uvector<float> buffer(static_cast<size_t>(kBatch) * kM * kK * 2, stream);
  auto x_view = make_strided_view(
    buffer.data(), kBatch, kM, kK, cuda::std::array<int, 3>{2 * kM * kK, 2 * kK, 2});
  auto y_view =
    make_strided_view(buffer.data(), kBatch, kK, kN, strides_of(kK, kN, false, kK * kN));
  auto z_view =
    make_strided_view(buffer.data(), kBatch, kM, kN, strides_of(kM, kN, false, kM * kN));

  EXPECT_THROW(gemm_batched(res, x_view, y_view, z_view), raft::logic_error);
  raft::resource::sync_stream(res);
}

}  // namespace raft::linalg
