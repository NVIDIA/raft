/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/bitmap.cuh>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/linalg/init.cuh>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>

#include <thrust/for_each.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace raft::core {

template <typename index_t>
struct test_spec_bitmap {
  index_t rows;
  index_t cols;
  index_t mask_len;
  index_t query_len;
};

template <typename index_t>
auto operator<<(std::ostream& os, const test_spec_bitmap<index_t>& ss) -> std::ostream&
{
  os << "bitmap{rows: " << ss.rows << ", cols: " << ss.cols << ", mask_len: " << ss.mask_len
     << ", query_len: " << ss.query_len << "}";
  return os;
}

template <typename bitmap_t, typename index_t>
void create_cpu_bitmap(std::vector<bitmap_t>& bitmap,
                       std::vector<index_t>& mask_idx,
                       const index_t rows,
                       const index_t cols)
{
  for (size_t i = 0; i < bitmap.size(); i++) {
    bitmap[i] = ~bitmap_t(0x00);
  }
  constexpr size_t bitmap_element_size = sizeof(bitmap_t) * 8;
  for (size_t i = 0; i < mask_idx.size(); i++) {
    auto row = mask_idx[i] / cols;
    auto col = mask_idx[i] % cols;
    auto idx = row * cols + col;
    bitmap[idx / bitmap_element_size] &= ~(bitmap_t{1} << (idx % bitmap_element_size));
  }
}

template <typename bitmap_t, typename index_t>
void test_cpu_bitmap(const std::vector<bitmap_t>& bitmap,
                     const std::vector<index_t>& queries,
                     std::vector<uint8_t>& result,
                     index_t rows,
                     index_t cols)
{
  constexpr size_t bitmap_element_size = sizeof(bitmap_t) * 8;
  for (size_t i = 0; i < queries.size(); i++) {
    auto row  = queries[i] / cols;
    auto col  = queries[i] % cols;
    auto idx  = row * cols + col;
    result[i] = uint8_t(
      (bitmap[idx / bitmap_element_size] & (bitmap_t{1} << (idx % bitmap_element_size))) != 0);
  }
}

template <typename bitmap_t, typename index_t>
class BitmapTest : public testing::TestWithParam<test_spec_bitmap<index_t>> {
 protected:
  index_t static constexpr const bitmap_element_size = sizeof(bitmap_t) * 8;
  const test_spec_bitmap<index_t> spec;
  std::vector<bitmap_t> bitmap_result;
  std::vector<bitmap_t> bitmap_ref;
  raft::resources res;

 public:
  explicit BitmapTest()
    : spec(testing::TestWithParam<test_spec_bitmap<index_t>>::GetParam()),
      bitmap_result(raft::ceildiv(spec.rows * spec.cols, index_t(bitmap_element_size))),
      bitmap_ref(raft::ceildiv(spec.rows * spec.cols, index_t(bitmap_element_size)))
  {
  }

  void run()
  {
    auto stream = resource::get_cuda_stream(res);

    raft::random::RngState rng(42);
    auto mask_device = raft::make_device_vector<index_t, index_t>(res, spec.mask_len);
    std::vector<index_t> mask_cpu(spec.mask_len);
    raft::random::uniformInt(
      res, rng, mask_device.view(), index_t(0), index_t(spec.rows * spec.cols));
    raft::update_host(mask_cpu.data(), mask_device.data_handle(), mask_device.extent(0), stream);
    resource::sync_stream(res, stream);

    create_cpu_bitmap(bitmap_ref, mask_cpu, spec.rows, spec.cols);

    auto query_device  = raft::make_device_vector<index_t, index_t>(res, spec.query_len);
    auto result_device = raft::make_device_vector<uint8_t, index_t>(res, spec.query_len);
    auto query_cpu     = std::vector<index_t>(spec.query_len);
    auto result_cpu    = std::vector<uint8_t>(spec.query_len);
    auto result_ref    = std::vector<uint8_t>(spec.query_len);

    raft::execute_with_dry_run_check(
      res,
      [&](raft::resources const& h) {
        raft::core::bitset<bitmap_t, index_t> bitset_d(
          h, raft::make_const_mdspan(mask_device.view()), index_t(spec.rows * spec.cols));

        auto bitmap_view_d =
          raft::core::bitmap_view<bitmap_t, index_t>(bitset_d.data(), spec.rows, spec.cols);

        ASSERT_EQ(bitmap_view_d.get_n_rows(), spec.rows);
        ASSERT_EQ(bitmap_view_d.get_n_cols(), spec.cols);

        raft::random::uniformInt(
          h, rng, query_device.view(), index_t(0), index_t(spec.rows * spec.cols));
        raft::copy(h,
                   raft::make_host_vector_view(query_cpu.data(), query_device.extent(0)),
                   raft::make_const_mdspan(query_device.view()));

        auto queries_device_view =
          raft::make_device_vector_view<const index_t>(query_device.data_handle(), spec.query_len);

        raft::linalg::map(
          h,
          result_device.view(),
          [bitmap_view_d] __device__(index_t query) {
            auto row = query / bitmap_view_d.get_n_cols();
            auto col = query % bitmap_view_d.get_n_cols();
            return (uint8_t)(bitmap_view_d.test(row, col));
          },
          queries_device_view);

        raft::copy(h,
                   raft::make_host_vector_view(result_cpu.data(), result_device.extent(0)),
                   raft::make_const_mdspan(result_device.view()));
        resource::sync_stream(h, stream);

        test_cpu_bitmap(bitmap_ref, query_cpu, result_ref, spec.rows, spec.cols);

        if (resource::get_dry_run_flag(h)) { return; }
        ASSERT_TRUE(hostVecMatch(result_cpu, result_ref, Compare<uint8_t>()));

        raft::random::uniformInt(
          h, rng, mask_device.view(), index_t(0), index_t(spec.rows * spec.cols));
        raft::copy(h,
                   raft::make_host_vector_view(mask_cpu.data(), mask_device.extent(0)),
                   raft::make_const_mdspan(mask_device.view()));
        resource::sync_stream(h, stream);

        thrust::for_each_n(raft::resource::get_thrust_policy(h),
                           mask_device.data_handle(),
                           mask_device.extent(0),
                           [bitmap_view_d] __device__(const index_t sample_index) {
                             auto row = sample_index / bitmap_view_d.get_n_cols();
                             auto col = sample_index % bitmap_view_d.get_n_cols();
                             bitmap_view_d.set(row, col, false);
                           });

        raft::copy(h,
                   raft::make_host_vector_view(bitmap_result.data(), bitmap_result.size()),
                   raft::make_device_vector_view<const bitmap_t>(bitmap_view_d.data(),
                                                                 bitmap_result.size()));

        for (size_t i = 0; i < mask_cpu.size(); i++) {
          auto row = mask_cpu[i] / spec.cols;
          auto col = mask_cpu[i] % spec.cols;
          auto idx = row * spec.cols + col;
          bitmap_ref[idx / bitmap_element_size] &= ~(bitmap_t{1} << (idx % bitmap_element_size));
        }
        resource::sync_stream(h, stream);
        ASSERT_TRUE(hostVecMatch(bitmap_ref, bitmap_result, raft::Compare<bitmap_t>()));
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      raft::ceildiv(index_t(spec.rows * spec.cols), bitmap_element_size) * sizeof(bitmap_t));
  }
};

template <typename index_t>
auto inputs_bitmap =
  ::testing::Values(test_spec_bitmap<index_t>{32, 32, 5, 10},
                    test_spec_bitmap<index_t>{100, 100, 30, 10},
                    test_spec_bitmap<index_t>{1024, 1024, 55, 100},
                    test_spec_bitmap<index_t>{10000, 10000, 1000, 1000},
                    test_spec_bitmap<index_t>{1 << 15, 1 << 15, 1 << 3, 1 << 12},
                    test_spec_bitmap<index_t>{1 << 15, 1 << 15, 1 << 24, 1 << 13});

using BitmapTest_Uint32_32 = BitmapTest<uint32_t, uint32_t>;
TEST_P(BitmapTest_Uint32_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitmapTest, BitmapTest_Uint32_32, inputs_bitmap<uint32_t>);

using BitmapTest_Uint64_32 = BitmapTest<uint64_t, uint32_t>;
TEST_P(BitmapTest_Uint64_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitmapTest, BitmapTest_Uint64_32, inputs_bitmap<uint32_t>);

using BitmapTest_Uint32_64 = BitmapTest<uint32_t, uint64_t>;
TEST_P(BitmapTest_Uint32_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitmapTest, BitmapTest_Uint32_64, inputs_bitmap<uint64_t>);

using BitmapTest_Uint64_64 = BitmapTest<uint64_t, uint64_t>;
TEST_P(BitmapTest_Uint64_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitmapTest, BitmapTest_Uint64_64, inputs_bitmap<uint64_t>);

}  // namespace raft::core
