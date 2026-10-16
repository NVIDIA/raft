/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/copy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/select_k.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

namespace raft::matrix {

namespace {

// The ids that `kWarpDistributedShmStable` must return, sorted: the best keys, and on an equal key
// the smaller id for select-min and the larger id for select-max.
auto expected_ids(const std::vector<float>& keys,
                  const std::vector<int64_t>& ids,
                  int k,
                  bool select_min) -> std::vector<int64_t>
{
  std::vector<size_t> order(keys.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    if (keys[a] != keys[b]) { return select_min ? keys[a] < keys[b] : keys[a] > keys[b]; }
    return select_min ? ids[a] < ids[b] : ids[a] > ids[b];
  });
  std::vector<int64_t> out;
  for (int i = 0; i < k; i++) {
    out.push_back(ids[order[i]]);
  }
  std::sort(out.begin(), out.end());
  return out;
}

auto stable_select(const raft::device_resources& handle,
                   const std::vector<float>& keys,
                   const std::vector<int64_t>& ids,
                   int k,
                   bool select_min) -> std::vector<int64_t>
{
  auto stream   = raft::resource::get_cuda_stream(handle);
  int64_t len   = keys.size();
  auto in_keys  = raft::make_device_matrix<float, int64_t>(handle, 1, len);
  auto in_ids   = raft::make_device_matrix<int64_t, int64_t>(handle, 1, len);
  auto out_keys = raft::make_device_matrix<float, int64_t>(handle, 1, k);
  auto out_ids  = raft::make_device_matrix<int64_t, int64_t>(handle, 1, k);
  raft::copy(in_keys.data_handle(), keys.data(), len, stream);
  raft::copy(in_ids.data_handle(), ids.data(), len, stream);
  raft::matrix::select_k<float, int64_t>(handle,
                                         raft::make_const_mdspan(in_keys.view()),
                                         std::make_optional(raft::make_const_mdspan(in_ids.view())),
                                         out_keys.view(),
                                         out_ids.view(),
                                         select_min,
                                         /*sorted=*/true,
                                         SelectAlgo::kWarpDistributedShmStable);
  std::vector<int64_t> out(k);
  raft::copy(out.data(), out_ids.data_handle(), k, stream);
  raft::resource::sync_stream(handle, stream);
  std::sort(out.begin(), out.end());
  return out;
}

// Selects from the same (key, id) pairs in two input orders.
void check_order_independent(const std::vector<float>& keys, const std::vector<int64_t>& ids, int k)
{
  raft::device_resources handle;
  std::mt19937 rng(1234);
  for (bool select_min : {true, false}) {
    auto expected = expected_ids(keys, ids, k, select_min);
    for (int order = 0; order < 2; order++) {
      std::vector<size_t> perm(keys.size());
      std::iota(perm.begin(), perm.end(), 0);
      std::shuffle(perm.begin(), perm.end(), rng);
      std::vector<float> perm_keys(keys.size());
      std::vector<int64_t> perm_ids(keys.size());
      for (size_t i = 0; i < perm.size(); i++) {
        perm_keys[i] = keys[perm[i]];
        perm_ids[i]  = ids[perm[i]];
      }
      EXPECT_EQ(stable_select(handle, perm_keys, perm_ids, k, select_min), expected)
        << "select_min=" << select_min << " order=" << order << " len=" << keys.size()
        << " k=" << k;
    }
  }
}

auto distinct_ids(size_t n) -> std::vector<int64_t>
{
  std::vector<int64_t> ids(n);
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), std::mt19937(42));
  return ids;
}

auto from_bits(uint32_t bits) -> float
{
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace

TEST(SelectKStable, AllKeysEqual)
{
  for (size_t len : {4096, 300000}) {
    for (int k : {10, 100}) {
      check_order_independent(std::vector<float>(len, 1.0f), distinct_ids(len), k);
    }
  }
}

TEST(SelectKStable, TiesAtTheKthPlace)
{
  for (size_t len : {4096, 300000}) {
    std::vector<float> keys(len);
    std::mt19937 rng(7);
    for (auto& key : keys) {
      key = float(rng() % 4);
    }
    check_order_independent(keys, distinct_ids(len), 100);
  }
}

// A real key that twiddles to the same bits as the empty-slot key must outrank the empty slots.
// These NaN bit patterns twiddle to the all-ones (select-min) and all-zeros (select-max) keys.
TEST(SelectKStable, RealKeysEqualToTheEmptyKeyWin)
{
  raft::device_resources handle;
  std::vector<int64_t> ids = {11, 3, 7, 5, 9};
  for (bool select_min : {true, false}) {
    float key = from_bits(select_min ? 0x7FFFFFFFu : 0xFFFFFFFFu);
    std::vector<float> keys(ids.size(), key);
    auto got = stable_select(handle, keys, ids, 8, select_min);
    for (int64_t id : ids) {
      EXPECT_NE(std::find(got.begin(), got.end(), id), got.end())
        << "select_min=" << select_min << " missing id " << id;
    }
  }
}

}  // namespace raft::matrix
