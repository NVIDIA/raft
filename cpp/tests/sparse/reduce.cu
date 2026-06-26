/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/op/reduce.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename value_t, typename value_idx>
struct SparseReduceInputs {
  std::vector<value_idx> in_rows;
  std::vector<value_idx> in_cols;
  std::vector<value_t> in_vals;
  std::vector<value_idx> out_rows;
  std::vector<value_idx> out_cols;
  std::vector<value_t> out_vals;

  value_idx m;
  value_idx n;
};

template <typename value_t, typename value_idx>
class SparseReduceTest : public ::testing::TestWithParam<SparseReduceInputs<value_t, value_idx>> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<SparseReduceInputs<value_t, value_idx>>::GetParam();
  }

  void Run()
  {
    raft::resources handle;

    auto stream = resource::get_cuda_stream(handle);

    rmm::device_uvector<value_idx> in_rows(params.in_rows.size(), stream);
    rmm::device_uvector<value_idx> in_cols(params.in_cols.size(), stream);
    rmm::device_uvector<value_t> in_vals(params.in_vals.size(), stream);
    rmm::device_uvector<value_idx> out_rows(params.out_rows.size(), stream);
    rmm::device_uvector<value_idx> out_cols(params.out_cols.size(), stream);
    rmm::device_uvector<value_t> out_vals(params.out_vals.size(), stream);

    raft::update_device(in_rows.data(), params.in_rows.data(), params.in_rows.size(), stream);
    raft::update_device(in_cols.data(), params.in_cols.data(), params.in_cols.size(), stream);
    raft::update_device(in_vals.data(), params.in_vals.data(), params.in_vals.size(), stream);
    raft::update_device(out_rows.data(), params.out_rows.data(), params.out_rows.size(), stream);
    raft::update_device(out_cols.data(), params.out_cols.data(), params.out_cols.size(), stream);
    raft::update_device(out_vals.data(), params.out_vals.data(), params.out_vals.size(), stream);

    raft::sparse::COO<value_t, value_idx, value_idx> out(stream);
    // min_alloc: internal workspace of max_duplicates (diff array + CUB scan workspace).
    // The COO output itself is not tracked because `out` was created outside the wrapper.
    auto min_alloc = (params.in_rows.size() + 1) * sizeof(value_idx);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        raft::sparse::op::max_duplicates(h,
                                         out,
                                         in_rows.data(),
                                         in_cols.data(),
                                         in_vals.data(),
                                         (value_idx)params.in_rows.size(),
                                         (value_idx)params.m,
                                         (value_idx)params.n);
      },
      raft::alloc_behavior::DATA_DRIVEN,
      min_alloc);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ASSERT_TRUE(raft::devArrMatch<value_idx>(
      out_rows.data(), out.rows(), out.nnz, raft::Compare<value_idx>()));
    ASSERT_TRUE(raft::devArrMatch<value_idx>(
      out_cols.data(), out.cols(), out.nnz, raft::Compare<value_idx>()));
    ASSERT_TRUE(
      raft::devArrMatch<value_t>(out_vals.data(), out.vals(), out.nnz, raft::Compare<value_t>()));
  }

  void TearDown() override {}

 protected:
  SparseReduceInputs<value_t, value_idx> params;
  value_idx *in_rows, *in_cols, *out_rows, *out_cols;
  value_t *in_vals, *out_vals;
};

using SparseReduceTestF = SparseReduceTest<float, int>;
TEST_P(SparseReduceTestF, Result) { Run(); }

// Max reduce expects COO to be sorted already
const std::vector<SparseReduceInputs<float, int>> max_reduce_inputs_f = {
  {// input rows/cols/vals
   {0, 0, 0, 0, 1, 1, 1, 2, 3, 3, 3},
   {1, 1, 1, 2, 0, 3, 3, 0, 2, 3, 3},
   {3.0, 50.0, 0.0, 2.0, 40.0, 2.0, 1.0, 4.0, 1.0, 0.0, 30.0},

   // output rows/cols/vals
   {0, 0, 1, 1, 2, 3, 3},
   {1, 2, 0, 3, 0, 2, 3},
   {50.0, 2.0, 40.0, 2.0, 4.0, 1.0, 30.0},
   4,
   4},
};

INSTANTIATE_TEST_CASE_P(SparseReduceTest,
                        SparseReduceTestF,
                        ::testing::ValuesIn(max_reduce_inputs_f));

}  // namespace sparse
}  // namespace raft
