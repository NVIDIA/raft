/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/compressed_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace raft {

TEST(CompressedMDArray, PQGlobalReconstruct)
{
  std::vector<float> codebook_data = {
    0.0f,
    0.0f,
    1.0f,
    0.0f,
    0.0f,
    1.0f,
    1.0f,
    1.0f,
  };

  std::vector<uint8_t> codes_data = {
    0,
    1,
    2,
    3,
    0,
    1,
  };

  auto codebook_view = make_host_matrix_view<const float, uint32_t>(codebook_data.data(), 4, 2);
  auto codes_view    = make_host_matrix_view<const uint8_t, uint32_t>(codes_data.data(), 2, 3);

  auto view = make_pq_host_matrix_view<float, uint32_t>(codebook_view, codes_view, 6);

  ASSERT_EQ(view.extent(0), 2);
  ASSERT_EQ(view.extent(1), 6);

  EXPECT_FLOAT_EQ(view(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(view(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(view(0, 2), 1.0f);
  EXPECT_FLOAT_EQ(view(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(view(0, 4), 0.0f);
  EXPECT_FLOAT_EQ(view(0, 5), 1.0f);

  EXPECT_FLOAT_EQ(view(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(view(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(view(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(view(1, 3), 0.0f);
  EXPECT_FLOAT_EQ(view(1, 4), 1.0f);
  EXPECT_FLOAT_EQ(view(1, 5), 0.0f);
}

TEST(CompressedMDArray, PQSubspaceReconstruct)
{
  // codebook(subspace, component, center) with shape [2, 2, 3]
  std::vector<float> codebook_data = {
    10.0f,
    20.0f,
    30.0f,
    11.0f,
    21.0f,
    31.0f,
    100.0f,
    200.0f,
    300.0f,
    101.0f,
    201.0f,
    301.0f,
  };

  std::vector<uint8_t> codes_data = {
    0,
    2,
    1,
    0,
  };

  using cb_view_t = host_pq_subspace_codebook_view<float>;
  auto codebook_view =
    cb_view_t(codebook_data.data(), make_extents<uint32_t>(uint32_t(2), uint32_t(2), uint32_t(3)));
  auto codes_view = make_host_matrix_view<const uint8_t, uint32_t>(codes_data.data(), 2, 2);

  auto view = make_pq_subspace_host_matrix_view<float, uint32_t>(codebook_view, codes_view, 4);

  ASSERT_EQ(view.extent(0), 2);
  ASSERT_EQ(view.extent(1), 4);

  EXPECT_FLOAT_EQ(view(0, 0), 10.0f);
  EXPECT_FLOAT_EQ(view(0, 1), 11.0f);
  EXPECT_FLOAT_EQ(view(0, 2), 300.0f);
  EXPECT_FLOAT_EQ(view(0, 3), 301.0f);

  EXPECT_FLOAT_EQ(view(1, 0), 20.0f);
  EXPECT_FLOAT_EQ(view(1, 1), 21.0f);
  EXPECT_FLOAT_EQ(view(1, 2), 100.0f);
  EXPECT_FLOAT_EQ(view(1, 3), 101.0f);
}

TEST(CompressedMDArray, SQReconstruct)
{
  float min_val = -1.0f;
  float max_val = 1.0f;

  std::vector<int8_t> codes_data = {
    0,
    127,
    -128,
    64,
    -64,
    0,
  };

  std::vector<float> codebook_data = {min_val, max_val};

  auto codebook_view =
    make_host_vector_view<const float, uint32_t>(codebook_data.data(), uint32_t(2));
  auto codes_view = make_host_matrix_view<const int8_t, uint32_t>(codes_data.data(), 2, 3);

  auto view =
    make_sq_host_matrix_view<float, uint32_t>(codebook_view, codes_view, min_val, max_val);

  ASSERT_EQ(view.extent(0), 2);
  ASSERT_EQ(view.extent(1), 3);

  float scale  = 127.5f;
  float offset = -0.5f;

  EXPECT_NEAR(view(0, 0), (0.0f - offset) / scale, 1e-5f);
  EXPECT_NEAR(view(0, 1), (127.0f - offset) / scale, 1e-5f);
  EXPECT_NEAR(view(0, 2), (-128.0f - offset) / scale, 1e-5f);

  EXPECT_NEAR(view(1, 0), (64.0f - offset) / scale, 1e-5f);
  EXPECT_NEAR(view(1, 1), (-64.0f - offset) / scale, 1e-5f);
  EXPECT_NEAR(view(1, 2), (0.0f - offset) / scale, 1e-5f);
}

TEST(CompressedMDArray, PQHostMdarrayFactory)
{
  raft::resources handle;

  // Create codebook and codes as regular host mdarrays
  auto codebook     = raft::make_host_matrix<float>(handle, std::uint32_t(4), std::uint32_t(2));
  float cb_values[] = {0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  std::copy(cb_values, cb_values + 8, codebook.data_handle());

  auto codes =
    raft::make_host_matrix<std::uint8_t, std::uint32_t>(handle, std::uint32_t(2), std::uint32_t(3));
  uint8_t code_values[] = {0, 1, 2, 3, 0, 1};
  std::copy(code_values, code_values + 6, codes.data_handle());

  // Wrap into a compressed mdarray — dim is derived from codebook/codes shapes
  auto pq_mat = make_pq_host_matrix<float, std::uint32_t>(std::move(codebook), std::move(codes));

  ASSERT_EQ(pq_mat.extent(0), 2);
  ASSERT_EQ(pq_mat.extent(1), 6);

  EXPECT_FLOAT_EQ(pq_mat(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(pq_mat(0, 2), 1.0f);
  EXPECT_FLOAT_EQ(pq_mat(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(pq_mat(1, 1), 1.0f);

  auto v = pq_mat.view();
  EXPECT_FLOAT_EQ(v(0, 4), 0.0f);
  EXPECT_FLOAT_EQ(v(0, 5), 1.0f);
}

TEST(CompressedMDArray, ViewTypeIsConst)
{
  using pq_view_t = pq_host_matrix_view<float, uint32_t>;
  static_assert(std::is_const_v<pq_view_t::element_type>,
                "pq_host_matrix_view element_type must be const");

  using sq_view_t = sq_host_matrix_view<float, uint32_t>;
  static_assert(std::is_const_v<sq_view_t::element_type>,
                "sq_host_matrix_view element_type must be const");

  using pqs_view_t = pq_subspace_host_matrix_view<float, uint32_t>;
  static_assert(std::is_const_v<pqs_view_t::element_type>,
                "pq_subspace_host_matrix_view element_type must be const");
}

}  // namespace raft
