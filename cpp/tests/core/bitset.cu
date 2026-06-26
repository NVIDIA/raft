/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/bitset.cuh>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/linalg/init.cuh>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>

namespace raft::core {

struct test_spec_bitset {
  uint64_t bitset_len;
  uint64_t mask_len;
  uint64_t query_len;
  uint64_t repeat_times;
};

auto operator<<(std::ostream& os, const test_spec_bitset& ss) -> std::ostream&
{
  os << "bitset{bitset_len: " << ss.bitset_len << ", mask_len: " << ss.mask_len
     << ", query_len: " << ss.query_len << ", repeat_times: " << ss.repeat_times << "}";
  return os;
}

template <typename bitset_t, typename index_t>
void add_cpu_bitset(std::vector<bitset_t>& bitset, const std::vector<index_t>& mask_idx)
{
  constexpr size_t bitset_element_size = sizeof(bitset_t) * 8;
  for (size_t i = 0; i < mask_idx.size(); i++) {
    auto idx = mask_idx[i];
    bitset[idx / bitset_element_size] &= ~(bitset_t{1} << (idx % bitset_element_size));
  }
}

template <typename bitset_t, typename index_t>
void create_cpu_bitset(std::vector<bitset_t>& bitset, const std::vector<index_t>& mask_idx)
{
  for (size_t i = 0; i < bitset.size(); i++) {
    bitset[i] = ~bitset_t(0x00);
  }
  add_cpu_bitset(bitset, mask_idx);
}

template <typename bitset_t, typename index_t>
void test_cpu_bitset(const std::vector<bitset_t>& bitset,
                     const std::vector<index_t>& queries,
                     std::vector<uint8_t>& result)
{
  constexpr size_t bitset_element_size = sizeof(bitset_t) * 8;
  for (size_t i = 0; i < queries.size(); i++) {
    result[i] = uint8_t((bitset[queries[i] / bitset_element_size] &
                         (bitset_t{1} << (queries[i] % bitset_element_size))) != 0);
  }
}

template <typename bitset_t, typename index_t>
void test_cpu_bitset_nbits(const bitset_t* bitset,
                           const std::vector<index_t>& queries,
                           std::vector<uint8_t>& result,
                           unsigned original_nbits_)
{
  constexpr size_t nbits = sizeof(bitset_t) * 8;
  if (original_nbits_ == nbits) {
    for (size_t i = 0; i < queries.size(); i++) {
      result[i] =
        uint8_t((bitset[queries[i] / nbits] & (bitset_t{1} << (queries[i] % nbits))) != 0);
    }
  }
  for (size_t i = 0; i < queries.size(); i++) {
    const index_t sample_index        = queries[i];
    const index_t original_bit_index  = sample_index / original_nbits_;
    const index_t original_bit_offset = sample_index % original_nbits_;
    index_t new_bit_index             = original_bit_index * original_nbits_ / nbits;
    index_t new_bit_offset            = 0;
    if (original_nbits_ > nbits) {
      new_bit_index += original_bit_offset / nbits;
      new_bit_offset = original_bit_offset % nbits;
    } else {
      index_t ratio = nbits / original_nbits_;
      new_bit_offset += (original_bit_index % ratio) * original_nbits_;
      new_bit_offset += original_bit_offset % nbits;
    }
    const bitset_t bit_element = bitset[new_bit_index];
    const bool is_bit_set      = (bit_element & (bitset_t{1} << new_bit_offset)) != 0;

    result[i] = uint8_t(is_bit_set);
  }
}

template <typename bitset_t>
void flip_cpu_bitset(std::vector<bitset_t>& bitset)
{
  for (size_t i = 0; i < bitset.size(); i++) {
    bitset[i] = ~bitset[i];
  }
}

template <typename bitset_t>
void repeat_cpu_bitset(std::vector<bitset_t>& input,
                       size_t input_bits,
                       size_t repeat,
                       std::vector<bitset_t>& output)
{
  const size_t output_bits  = input_bits * repeat;
  const size_t output_units = (output_bits + sizeof(bitset_t) * 8 - 1) / (sizeof(bitset_t) * 8);

  std::memset(output.data(), 0, output_units * sizeof(bitset_t));

  size_t output_bit_index = 0;

  for (size_t r = 0; r < repeat; ++r) {
    for (size_t i = 0; i < input_bits; ++i) {
      size_t input_unit_index = i / (sizeof(bitset_t) * 8);
      size_t input_bit_offset = i % (sizeof(bitset_t) * 8);
      bool bit                = (input[input_unit_index] >> input_bit_offset) & 1;

      size_t output_unit_index = output_bit_index / (sizeof(bitset_t) * 8);
      size_t output_bit_offset = output_bit_index % (sizeof(bitset_t) * 8);

      output[output_unit_index] |= (static_cast<bitset_t>(bit) << output_bit_offset);

      ++output_bit_index;
    }
  }
}

template <typename bitset_t>
double sparsity_cpu_bitset(std::vector<bitset_t>& data, size_t total_bits)
{
  size_t one_count = 0;
  for (size_t i = 0; i < total_bits; ++i) {
    size_t unit_index = i / (sizeof(bitset_t) * 8);
    size_t bit_offset = i % (sizeof(bitset_t) * 8);
    bool bit          = (data[unit_index] >> bit_offset) & 1;
    if (bit == 1) { ++one_count; }
  }
  return static_cast<double>((total_bits - one_count) / (1.0 * total_bits));
}

template <typename bitset_t, typename index_t>
class BitsetTest : public testing::TestWithParam<test_spec_bitset> {
 protected:
  index_t static constexpr const bitset_element_size = sizeof(bitset_t) * 8;
  const test_spec_bitset spec;
  std::vector<bitset_t> bitset_result;
  std::vector<bitset_t> bitset_ref;
  std::vector<bitset_t> bitset_repeat_ref;
  std::vector<bitset_t> bitset_repeat_result;
  raft::resources res;

 public:
  explicit BitsetTest()
    : spec(testing::TestWithParam<test_spec_bitset>::GetParam()),
      bitset_result(raft::ceildiv(spec.bitset_len, uint64_t(bitset_element_size))),
      bitset_ref(raft::ceildiv(spec.bitset_len, uint64_t(bitset_element_size))),
      bitset_repeat_ref(
        raft::ceildiv(spec.bitset_len * spec.repeat_times, uint64_t(bitset_element_size))),
      bitset_repeat_result(
        raft::ceildiv(spec.bitset_len * spec.repeat_times, uint64_t(bitset_element_size)))
  {
  }

  void run()
  {
    auto stream = resource::get_cuda_stream(res);

    // generate input and mask
    raft::random::RngState rng(42);
    auto mask_device = raft::make_device_vector<index_t, index_t>(res, spec.mask_len);
    std::vector<index_t> mask_cpu(spec.mask_len);
    raft::random::uniformInt(res, rng, mask_device.view(), index_t(0), index_t(spec.bitset_len));
    update_host(mask_cpu.data(), mask_device.data_handle(), mask_device.extent(0), stream);
    resource::sync_stream(res, stream);

    auto query_device     = raft::make_device_vector<index_t, index_t>(res, spec.query_len);
    auto result_device    = raft::make_device_vector<uint8_t, index_t>(res, spec.query_len);
    auto query_cpu        = std::vector<index_t>(spec.query_len);
    auto result_cpu       = std::vector<uint8_t>(spec.query_len);
    auto result_ref_nbits = std::vector<uint8_t>(spec.query_len);
    auto result_ref       = std::vector<uint8_t>(spec.query_len);

    auto eval_n_elements =
      bitset_view<bitset_t, index_t>::eval_n_elements(spec.bitset_len * spec.repeat_times);
    auto repeat_device = raft::make_device_vector<bitset_t, index_t>(res, eval_n_elements);

    // Verify dry-run compliance of bitset construction.
    // The entire bitset lifetime must be within the dry-run scope to prevent
    // its device_buffer from outliving the temporary resource wrapper.
    raft::execute_with_dry_run_check(
      res,
      [&](raft::resources const& h) {
        raft::core::bitset<bitset_t, index_t> my_bitset(
          h, raft::make_const_mdspan(mask_device.view()), index_t(spec.bitset_len));

        raft::copy(h,
                   raft::make_host_vector_view(bitset_result.data(), bitset_result.size()),
                   my_bitset.to_mdspan());

        // calculate the reference
        create_cpu_bitset(bitset_ref, mask_cpu);
        resource::sync_stream(h, stream);
        if (!resource::get_dry_run_flag(h)) {
          ASSERT_TRUE(hostVecMatch(bitset_ref, bitset_result, raft::Compare<bitset_t>()));
        }

        // Create queries and verify the test results
        raft::random::uniformInt(h, rng, query_device.view(), index_t(0), index_t(spec.bitset_len));
        raft::copy(h,
                   raft::make_host_vector_view(query_cpu.data(), query_device.extent(0)),
                   raft::make_const_mdspan(query_device.view()));
        my_bitset.test(h, raft::make_const_mdspan(query_device.view()), result_device.view());
        raft::copy(h,
                   raft::make_host_vector_view(result_cpu.data(), result_device.extent(0)),
                   raft::make_const_mdspan(result_device.view()));
        test_cpu_bitset(bitset_ref, query_cpu, result_ref);
        resource::sync_stream(h, stream);
        if (!resource::get_dry_run_flag(h)) {
          ASSERT_TRUE(hostVecMatch(result_cpu, result_ref, Compare<uint8_t>()));
        }

        // Add more sample to the bitset and re-test
        raft::random::uniformInt(h, rng, mask_device.view(), index_t(0), index_t(spec.bitset_len));
        raft::copy(h,
                   raft::make_host_vector_view(mask_cpu.data(), mask_device.extent(0)),
                   raft::make_const_mdspan(mask_device.view()));
        resource::sync_stream(h, stream);
        my_bitset.set(h, mask_device.view());
        raft::copy(h,
                   raft::make_host_vector_view(bitset_result.data(), bitset_result.size()),
                   my_bitset.to_mdspan());

        add_cpu_bitset(bitset_ref, mask_cpu);
        resource::sync_stream(h, stream);
        if (!resource::get_dry_run_flag(h)) {
          ASSERT_TRUE(hostVecMatch(bitset_ref, bitset_result, raft::Compare<bitset_t>()));
        }

        // Reinterpret the bitset as uint8_t, uint32 then uint64_t
        {
          // Test CPU logic
          test_cpu_bitset(bitset_ref, query_cpu, result_ref);
          auto bitset_cpu_uint8 =
            raft::make_host_vector<uint8_t>(h, sizeof(bitset_t) * bitset_ref.size());
          raft::copy(
            h,
            bitset_cpu_uint8.view(),
            raft::make_host_vector_view(reinterpret_cast<const uint8_t*>(bitset_ref.data()),
                                        bitset_cpu_uint8.extent(0)));
          test_cpu_bitset_nbits(
            bitset_cpu_uint8.data_handle(), query_cpu, result_ref_nbits, sizeof(bitset_t) * 8);
          if (!resource::get_dry_run_flag(h)) {
            ASSERT_TRUE(hostVecMatch(result_ref, result_ref_nbits, raft::Compare<uint8_t>()));
          }

          // Test GPU uint8_t, uint32_t, uint64_t
          auto my_bitset_view_uint8_t = raft::core::bitset_view<uint8_t, uint32_t>(
            reinterpret_cast<uint8_t*>(my_bitset.data()), my_bitset.size(), sizeof(bitset_t) * 8);
          raft::linalg::map(
            h,
            result_device.view(),
            [my_bitset_view_uint8_t] __device__(index_t query) {
              return my_bitset_view_uint8_t.test(query);
            },
            raft::make_const_mdspan(query_device.view()));
          raft::copy(h,
                     raft::make_host_vector_view(result_cpu.data(), result_device.extent(0)),
                     raft::make_const_mdspan(result_device.view()));
          resource::sync_stream(h, stream);
          if (!resource::get_dry_run_flag(h)) {
            ASSERT_TRUE(hostVecMatch(result_ref, result_cpu, Compare<uint8_t>()));
          }

          auto my_bitset_view_uint32_t = raft::core::bitset_view<uint32_t, uint32_t>(
            reinterpret_cast<uint32_t*>(my_bitset.data()), my_bitset.size(), sizeof(bitset_t) * 8);
          raft::linalg::map(
            h,
            result_device.view(),
            [my_bitset_view_uint32_t] __device__(index_t query) {
              return my_bitset_view_uint32_t.test(query);
            },
            raft::make_const_mdspan(query_device.view()));
          raft::copy(h,
                     raft::make_host_vector_view(result_cpu.data(), result_device.extent(0)),
                     raft::make_const_mdspan(result_device.view()));
          resource::sync_stream(h, stream);
          if (!resource::get_dry_run_flag(h)) {
            ASSERT_TRUE(hostVecMatch(result_ref, result_cpu, Compare<uint8_t>()));
          }

          auto my_bitset_view_uint64_t = raft::core::bitset_view<uint64_t, uint32_t>(
            reinterpret_cast<uint64_t*>(my_bitset.data()), my_bitset.size(), sizeof(bitset_t) * 8);
          raft::linalg::map(
            h,
            result_device.view(),
            [my_bitset_view_uint64_t] __device__(index_t query) {
              return my_bitset_view_uint64_t.test(query);
            },
            raft::make_const_mdspan(query_device.view()));
          raft::copy(h,
                     raft::make_host_vector_view(result_cpu.data(), result_device.extent(0)),
                     raft::make_const_mdspan(result_device.view()));
          resource::sync_stream(h, stream);
          if (!resource::get_dry_run_flag(h)) {
            ASSERT_TRUE(hostVecMatch(result_ref, result_cpu, Compare<uint8_t>()));
          }
        }

        // test sparsity, repeat and eval_n_elements
        {
          auto my_bitset_view  = my_bitset.view();
          auto sparsity_result = my_bitset_view.sparsity(h);
          auto sparsity_ref    = sparsity_cpu_bitset(bitset_ref, size_t(spec.bitset_len));
          if (!resource::get_dry_run_flag(h)) { ASSERT_EQ(sparsity_result, sparsity_ref); }

          ASSERT_EQ(bitset_repeat_ref.size(), eval_n_elements);

          RAFT_CUDA_TRY(cudaMemsetAsync(
            repeat_device.data_handle(), 0, eval_n_elements * sizeof(bitset_t), stream));
          repeat_cpu_bitset(
            bitset_ref, size_t(spec.bitset_len), size_t(spec.repeat_times), bitset_repeat_ref);

          my_bitset_view.repeat(h, index_t(spec.repeat_times), repeat_device.data_handle());

          ASSERT_EQ(bitset_repeat_ref.size(), repeat_device.size());
          raft::copy(
            h,
            raft::make_host_vector_view(bitset_repeat_result.data(), repeat_device.extent(0)),
            raft::make_const_mdspan(repeat_device.view()));
          ASSERT_EQ(bitset_repeat_ref.size(), bitset_repeat_result.size());

          if (!resource::get_dry_run_flag(h)) {
            index_t errors                        = 0;
            static constexpr index_t len_per_item = sizeof(bitset_t) * 8;
            bitset_t tail_len = (index_t(spec.bitset_len * spec.repeat_times) % len_per_item);
            bitset_t tail_mask =
              tail_len ? (bitset_t)((bitset_t{1} << tail_len) - bitset_t{1}) : ~bitset_t{0};
            for (index_t i = 0; i < bitset_repeat_ref.size(); i++) {
              if (i == bitset_repeat_ref.size() - 1) {
                errors +=
                  (bitset_repeat_ref[i] & tail_mask) != (bitset_repeat_result[i] & tail_mask);
              } else {
                errors += (bitset_repeat_ref[i] != bitset_repeat_result[i]);
              }
            }
            ASSERT_EQ(errors, 0);

            // recheck the sparsity after repeat
            sparsity_result = sparsity_cpu_bitset(bitset_repeat_result,
                                                  size_t(spec.bitset_len * spec.repeat_times));
            ASSERT_EQ(sparsity_result, sparsity_ref);
          }
        }

        // Flip the bitset and re-test
        auto bitset_count = my_bitset.count(h);
        my_bitset.flip(h);
        if (!resource::get_dry_run_flag(h)) {
          ASSERT_EQ(my_bitset.count(h), spec.bitset_len - bitset_count);
        }
        raft::copy(h,
                   raft::make_host_vector_view(bitset_result.data(), bitset_result.size()),
                   my_bitset.to_mdspan());
        flip_cpu_bitset(bitset_ref);
        resource::sync_stream(h, stream);
        if (!resource::get_dry_run_flag(h)) {
          ASSERT_TRUE(hostVecMatch(bitset_ref, bitset_result, raft::Compare<bitset_t>()));
        }

        // Test count() operations
        my_bitset.reset(h, false);
        ASSERT_EQ(my_bitset.any(h), false);
        ASSERT_EQ(my_bitset.none(h), true);
        raft::linalg::range(query_device.data_handle(), query_device.size(), stream);
        my_bitset.set(h, raft::make_const_mdspan(query_device.view()), true);
        bitset_count = my_bitset.count(h);
        if (!resource::get_dry_run_flag(h)) {
          ASSERT_EQ(bitset_count, query_device.size());
          ASSERT_EQ(my_bitset.any(h), true);
          ASSERT_EQ(my_bitset.none(h), false);
        }
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      raft::ceildiv(spec.bitset_len, uint64_t(bitset_element_size)) * sizeof(bitset_t));
  }
};

auto inputs_bitset = ::testing::Values(test_spec_bitset{32, 5, 10, 101},
                                       test_spec_bitset{100, 30, 10, 13},
                                       test_spec_bitset{1024, 55, 100, 1},
                                       test_spec_bitset{10000, 1000, 1000, 100},
                                       test_spec_bitset{1 << 15, 1 << 3, 1 << 12, 5},
                                       test_spec_bitset{1 << 15, 1 << 24, 1 << 13, 3},
                                       test_spec_bitset{1 << 25, 1 << 23, 1 << 14, 3},
                                       test_spec_bitset{1 << 25, 1 << 23, 1 << 14, 21});

using Uint16_32 = BitsetTest<uint16_t, uint32_t>;
TEST_P(Uint16_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint16_32, inputs_bitset);

using Uint32_32 = BitsetTest<uint32_t, uint32_t>;
TEST_P(Uint32_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint32_32, inputs_bitset);

using Uint64_32 = BitsetTest<uint64_t, uint32_t>;
TEST_P(Uint64_32, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint64_32, inputs_bitset);

using Uint8_64 = BitsetTest<uint8_t, uint64_t>;
TEST_P(Uint8_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint8_64, inputs_bitset);

using Uint32_64 = BitsetTest<uint32_t, uint64_t>;
TEST_P(Uint32_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint32_64, inputs_bitset);

using Uint64_64 = BitsetTest<uint64_t, uint64_t>;
TEST_P(Uint64_64, Run) { run(); }
INSTANTIATE_TEST_CASE_P(BitsetTest, Uint64_64, inputs_bitset);

}  // namespace raft::core
