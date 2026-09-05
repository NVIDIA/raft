/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

namespace raft {
namespace sparse {

template <typename T>
struct SparseSortInput {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
class SparseSortTest : public ::testing::TestWithParam<SparseSortInput<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseSortInput<T> params;
};

const std::vector<SparseSortInput<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseSortTest<float> COOSort;
TEST_P(COOSort, Result)
{
  params = ::testing::TestWithParam<SparseSortInput<float>>::GetParam();
  raft::random::RngState r(params.seed);
  raft::resources h;
  auto stream = resource::get_cuda_stream(h).get();

  rmm::device_uvector<int> in_rows(params.nnz, stream);
  rmm::device_uvector<int> in_cols(params.nnz, stream);
  rmm::device_uvector<int> verify(params.nnz, stream);
  rmm::device_uvector<float> in_vals(params.nnz, stream);

  uniform(h, r, in_vals.data(), params.nnz, float(-1.0), float(1.0));

  auto in_rows_h = std::make_unique<int[]>(params.nnz);
  auto in_cols_h = std::make_unique<int[]>(params.nnz);
  auto verify_h  = std::make_unique<int[]>(params.nnz);

  for (int i = 0; i < params.nnz; i++) {
    in_rows_h[i] = params.nnz - i - 1;
    verify_h[i]  = i;
    in_cols_h[i] = i;
  }

  raft::update_device(in_rows.data(), in_rows_h.get(), params.nnz, stream);

  raft::update_device(in_cols.data(), in_cols_h.get(), params.nnz, stream);
  raft::update_device(verify.data(), verify_h.get(), params.nnz, stream);

  op::coo_sort(
    params.m, params.n, params.nnz, in_rows.data(), in_cols.data(), in_vals.data(), stream);

  ASSERT_TRUE(raft::devArrMatch<int>(
    verify.data(), in_rows.data(), params.nnz, raft::Compare<int>(), stream));
}

INSTANTIATE_TEST_CASE_P(SparseSortTest, COOSort, ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
