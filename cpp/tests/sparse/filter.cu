/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace sparse {

template <typename T>
struct SparseFilterInputs {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
class SparseFilterTests : public ::testing::TestWithParam<SparseFilterInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseFilterInputs<T> params;
};

const std::vector<SparseFilterInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseFilterTests<float> COORemoveZeros;
TEST_P(COORemoveZeros, Result)
{
  raft::resources h;
  auto stream = resource::get_cuda_stream(h);
  params      = ::testing::TestWithParam<SparseFilterInputs<float>>::GetParam();

  float* in_h_vals = new float[params.nnz];

  COO<float> in(stream, params.nnz, 5, 5);

  raft::random::RngState r(params.seed);
  uniform(h, r, in.vals(), params.nnz, float(-1.0), float(1.0));

  raft::update_host(in_h_vals, in.vals(), params.nnz, stream);

  in_h_vals[0] = 0;
  in_h_vals[2] = 0;
  in_h_vals[3] = 0;

  int* in_h_rows = new int[params.nnz];
  int* in_h_cols = new int[params.nnz];

  for (int i = 0; i < params.nnz; i++) {
    in_h_rows[i] = params.nnz - i - 1;
    in_h_cols[i] = i;
  }

  raft::update_device(in.rows(), in_h_rows, params.nnz, stream);
  raft::update_device(in.cols(), in_h_cols, params.nnz, stream);
  raft::update_device(in.vals(), in_h_vals, params.nnz, stream);

  op::coo_sort<float>(&in, stream);

  int out_rows_ref_h[2] = {0, 3};
  int out_cols_ref_h[2] = {4, 1};

  float* out_vals_ref_h = (float*)malloc(2 * sizeof(float));
  out_vals_ref_h[0]     = in_h_vals[4];
  out_vals_ref_h[1]     = in_h_vals[1];

  COO<float> out_ref(stream, 2, 5, 5);
  COO<float> out(stream);

  raft::update_device(out_ref.rows(), *&out_rows_ref_h, 2, stream);
  raft::update_device(out_ref.cols(), *&out_cols_ref_h, 2, stream);
  raft::update_device(out_ref.vals(), out_vals_ref_h, 2, stream);

  op::coo_remove_zeros<float>(&in, &out, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  ASSERT_TRUE(raft::devArrMatch<int>(out_ref.rows(), out.rows(), 2, raft::Compare<int>()));
  ASSERT_TRUE(raft::devArrMatch<int>(out_ref.cols(), out.cols(), 2, raft::Compare<int>()));
  ASSERT_TRUE(raft::devArrMatch<float>(out_ref.vals(), out.vals(), 2, raft::Compare<float>()));

  free(out_vals_ref_h);

  delete[] in_h_rows;
  delete[] in_h_cols;
  delete[] in_h_vals;
}

INSTANTIATE_TEST_CASE_P(SparseFilterTests, COORemoveZeros, ::testing::ValuesIn(inputsf));

typedef SparseFilterTests<float> COORemoveScalarView;
TEST_P(COORemoveScalarView, ResultView)
{
  raft::resources h;
  auto stream = resource::get_cuda_stream(h);
  params      = ::testing::TestWithParam<SparseFilterInputs<float>>::GetParam();

  rmm::device_uvector<int> in_rows(params.nnz, stream);
  rmm::device_uvector<int> in_cols(params.nnz, stream);
  rmm::device_uvector<float> in_vals(params.nnz, stream);

  raft::random::RngState r(params.seed);
  uniform(h, r, in_vals.data(), params.nnz, float(-1.0), float(1.0));

  float* in_h_vals = new float[params.nnz];
  raft::update_host(in_h_vals, in_vals.data(), params.nnz, stream);

  in_h_vals[0] = 0;
  in_h_vals[2] = 0;
  in_h_vals[3] = 0;

  int* in_h_rows = new int[params.nnz];
  int* in_h_cols = new int[params.nnz];

  for (int i = 0; i < params.nnz; i++) {
    in_h_rows[i] = params.nnz - i - 1;
    in_h_cols[i] = i;
  }

  raft::update_device(in_rows.data(), in_h_rows, params.nnz, stream);
  raft::update_device(in_cols.data(), in_h_cols, params.nnz, stream);
  raft::update_device(in_vals.data(), in_h_vals, params.nnz, stream);

  op::coo_sort<float>(5, 5, params.nnz, in_rows.data(), in_cols.data(), in_vals.data(), stream);

  auto coo_structure =
    raft::make_device_coordinate_structure_view(in_rows.data(), in_cols.data(), params.nnz, 5, 5);

  auto in_view =
    raft::make_device_coo_matrix_view<const float, int, int, int>(in_vals.data(), coo_structure);

  auto out_matrix = raft::make_device_coo_matrix<float, int, int, int>(h, 5, 5);

  auto scalar = raft::make_host_scalar<float>(0.0f);

  raft::execute_with_dry_run_check(
    h,
    [&](raft::resources const& h) {
      op::coo_remove_scalar<128, float, int, int>(h, in_view, scalar.view(), out_matrix);
    },
    raft::alloc_behavior::DATA_DRIVEN,
    2 * 5 * sizeof(int));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  auto out_nnz = out_matrix.structure_view().get_nnz();
  ASSERT_EQ(out_nnz, 2);

  int out_rows_h[2];
  int out_cols_h[2];
  float out_vals_h[2];

  raft::update_host(out_rows_h, out_matrix.structure_view().get_rows().data(), 2, stream);
  raft::update_host(out_cols_h, out_matrix.structure_view().get_cols().data(), 2, stream);
  raft::update_host(out_vals_h, out_matrix.view().get_elements().data(), 2, stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  ASSERT_EQ(out_rows_h[0], 0);
  ASSERT_EQ(out_cols_h[0], 4);
  ASSERT_EQ(out_rows_h[1], 3);
  ASSERT_EQ(out_cols_h[1], 1);

  delete[] in_h_rows;
  delete[] in_h_cols;
  delete[] in_h_vals;
}

INSTANTIATE_TEST_CASE_P(SparseFilterTests, COORemoveScalarView, ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft
