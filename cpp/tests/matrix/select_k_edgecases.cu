/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/select_k.cuh>

#include <gtest/gtest.h>

#include <set>
#include <vector>

namespace raft::matrix {

template <typename DataT = float, typename IdxT = uint32_t>
auto run_max_k(const std::vector<DataT>& h_in, int64_t k)
{
  using raft::matrix::SelectAlgo;

  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  int64_t n_rows = 1;
  int64_t n_cols = h_in.size();

  // Device allocations (explicitly row_major optional)
  auto in_values =
    raft::make_device_matrix<DataT, int64_t, raft::row_major>(handle, n_rows, n_cols);
  auto out_values  = raft::make_device_matrix<DataT, int64_t, raft::row_major>(handle, n_rows, k);
  auto out_indices = raft::make_device_matrix<IdxT, int64_t, raft::row_major>(handle, n_rows, k);

  // Copy input to device
  raft::copy(in_values.data_handle(), h_in.data(), n_cols, stream);

  // Run select_k for largest k
  raft::matrix::select_k<DataT, IdxT>(handle,
                                      in_values.view(),
                                      std::nullopt,
                                      out_values.view(),
                                      out_indices.view(),
                                      /*select_min=*/false,
                                      /*sorted=*/true);

  // Copy results back to host
  std::vector<DataT> h_out(k);
  std::vector<IdxT> h_idx(k);
  raft::copy(h_out.data(), out_values.data_handle(), k, stream);
  raft::copy(h_idx.data(), out_indices.data_handle(), k, stream);
  raft::resource::sync_stream(handle, stream);

  if (k <= n_cols) {
    // Check that all indices in h_idx are unique and valid if there's enough input.
    std::set<IdxT> unique_indices;
    for (auto idx : h_idx) {
      RAFT_EXPECTS(idx < static_cast<IdxT>(n_cols), "Index in h_idx is out of bounds");
      unique_indices.insert(idx);
    }
    RAFT_EXPECTS(unique_indices.size() == h_idx.size(), "Indices in h_idx are not unique");
  }

  return std::make_tuple(h_out, h_idx);
}

TEST(SelectK, EdgeNaN)
{
  int64_t k               = 3;
  std::vector<float> h_in = {0.5f, NAN, NAN, NAN, NAN};
  auto [h_out, h_idx]     = run_max_k(h_in, k);

  // NaN values are the largest values as per IEEE 754
  for (int i = 0; i < k; i++) {
    ASSERT_TRUE(std::isnan(h_out[i])) << "h_out[" << i << "] is not NaN";
    ASSERT_NE(h_idx[i], 0) << "h_idx[" << i << "] is 0";
  }
}

TEST(SelectK, EdgeMinusInfinity)
{
  int64_t k               = 3;
  std::vector<float> h_in = {0.5f, -INFINITY, -INFINITY, -INFINITY, -INFINITY};
  auto [h_out, h_idx]     = run_max_k(h_in, k);

  // First element is the largest in this example
  ASSERT_EQ(h_out[0], 0.5f);
  ASSERT_EQ(h_idx[0], 0);
  // Rest are all -infinity
  for (int i = 1; i < k; i++) {
    ASSERT_TRUE(std::isinf(h_out[i]) && h_out[i] < 0) << "h_out[" << i << "] is not -infinity";
    ASSERT_NE(h_idx[i], 0) << "h_idx[" << i << "] is 0";
  }
}

TEST(SelectK, EdgePlusInfinity)
{
  int64_t k               = 3;
  std::vector<float> h_in = {0.5f, INFINITY, INFINITY, INFINITY, INFINITY};
  auto [h_out, h_idx]     = run_max_k(h_in, k);

  // All values are infinity and the first input element doesn't make it to the output
  for (int i = 0; i < k; i++) {
    ASSERT_TRUE(std::isinf(h_out[i]) && h_out[i] > 0) << "h_out[" << i << "] is not infinity";
    ASSERT_NE(h_idx[i], 0) << "h_idx[" << i << "] is 0";
  }
}

TEST(SelectK, EdgeInfNaN)
{
  int64_t k               = 3;
  std::vector<float> h_in = {0.5f, -INFINITY, NAN, -INFINITY, INFINITY};
  auto [h_out, h_idx]     = run_max_k(h_in, k);

  // The order is well defined here
  ASSERT_TRUE(std::isnan(h_out[0]));
  ASSERT_TRUE(std::isinf(h_out[1]) && h_out[1] > 0);
  ASSERT_EQ(h_out[2], 0.5f);

  ASSERT_EQ(h_idx[0], 2);
  ASSERT_EQ(h_idx[1], 4);
  ASSERT_EQ(h_idx[2], 0);
}

TEST(SelectK, SortedRadixUint64KeyUint32Idx)
{
  raft::device_resources handle;
  auto stream = raft::resource::get_cuda_stream(handle);

  // Upper 32 bits order: idx 0 < idx 2 < idx 3 < idx 1.
  // Lower 32 bits order is reversed (4, 1, 3, 2), so sorting only 32 bits mis-orders outputs.
  constexpr uint64_t v1      = (1ULL << 32) | 4ULL;
  constexpr uint64_t v4      = (4ULL << 32) | 1ULL;
  constexpr uint64_t v2      = (2ULL << 32) | 3ULL;
  constexpr uint64_t v3      = (3ULL << 32) | 2ULL;
  std::vector<uint64_t> h_in = {v1, v4, v2, v3};
  const int64_t n_cols       = h_in.size();
  const int64_t k            = 3;

  auto in_values  = raft::make_device_matrix<uint64_t, int64_t, raft::row_major>(handle, 1, n_cols);
  auto out_values = raft::make_device_matrix<uint64_t, int64_t, raft::row_major>(handle, 1, k);
  auto out_indices = raft::make_device_matrix<uint32_t, int64_t, raft::row_major>(handle, 1, k);
  raft::copy(in_values.data_handle(), h_in.data(), n_cols, stream);

  for (bool select_min : {false, true}) {
    raft::matrix::select_k<uint64_t, uint32_t>(handle,
                                               in_values.view(),
                                               std::nullopt,
                                               out_values.view(),
                                               out_indices.view(),
                                               select_min,
                                               /*sorted=*/true,
                                               SelectAlgo::kRadix8bits);
    std::vector<uint64_t> h_out(k);
    std::vector<uint32_t> h_idx(k);
    raft::copy(h_out.data(), out_values.data_handle(), k, stream);
    raft::copy(h_idx.data(), out_indices.data_handle(), k, stream);
    raft::resource::sync_stream(handle, stream);

    if (select_min) {
      EXPECT_EQ(h_out, (std::vector<uint64_t>{v1, v2, v3}));
      EXPECT_EQ(h_idx, (std::vector<uint32_t>{0, 2, 3}));
    } else {
      EXPECT_EQ(h_out, (std::vector<uint64_t>{v4, v3, v2}));
      EXPECT_EQ(h_idx, (std::vector<uint32_t>{1, 3, 2}));
    }
  }
}

}  // namespace raft::matrix
