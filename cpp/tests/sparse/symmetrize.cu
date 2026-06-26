/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <iostream>

namespace raft {
namespace sparse {

template <typename value_idx, typename value_t, typename nnz_t>
RAFT_KERNEL assert_symmetry(
  value_idx* rows, value_idx* cols, value_t* vals, nnz_t nnz, value_idx* sum)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid >= nnz) return;

  atomicAdd(sum, rows[tid]);
  atomicAdd(sum, -1 * cols[tid]);
}

template <typename value_idx, typename value_t>
struct SparseSymmetrizeInputs {
  value_idx n_cols;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;
};

template <typename value_idx, typename value_t>
::std::ostream& operator<<(::std::ostream& os,
                           const SparseSymmetrizeInputs<value_idx, value_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t, typename nnz_t>
class SparseSymmetrizeTest
  : public ::testing::TestWithParam<SparseSymmetrizeInputs<value_idx, value_t>> {
 public:
  SparseSymmetrizeTest()
    : params(::testing::TestWithParam<SparseSymmetrizeInputs<value_idx, value_t>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      indptr(0, stream),
      indices(0, stream),
      data(0, stream)
  {
  }

 protected:
  void make_data()
  {
    std::vector<value_idx> indptr_h  = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h      = params.data_h;

    indptr.resize(indptr_h.size(), stream);
    indices.resize(indices_h.size(), stream);
    data.resize(data_h.size(), stream);

    update_device(indptr.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(indices.data(), indices_h.data(), indices_h.size(), stream);
    update_device(data.data(), data_h.data(), data_h.size(), stream);
  }

  void SetUp() override
  {
    make_data();

    value_idx m = params.indptr_h.size() - 1;
    value_idx n = params.n_cols;
    nnz_t nnz   = params.indices_h.size();

    rmm::device_uvector<value_idx> coo_rows(nnz, stream);

    raft::sparse::convert::csr_to_coo(indptr.data(), m, coo_rows.data(), nnz, stream);

    raft::sparse::COO<value_t, value_idx, nnz_t> out(stream);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        raft::sparse::linalg::symmetrize(
          h, coo_rows.data(), indices.data(), data.data(), m, n, coo_rows.size(), out);
      },
      raft::alloc_behavior::DATA_DRIVEN,
      nnz * 2 * (2 * sizeof(value_idx) + sizeof(value_t)));

    rmm::device_scalar<value_idx> sum(stream);
    sum.set_value_to_zero_async(stream);

    assert_symmetry<<<raft::ceildiv(out.nnz, (nnz_t)256), 256, 0, stream>>>(
      out.rows(), out.cols(), out.vals(), (nnz_t)out.nnz, sum.data());

    sum_h = sum.value(stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  // input data
  rmm::device_uvector<value_idx> indptr, indices;
  rmm::device_uvector<value_t> data;

  value_idx sum_h;

  SparseSymmetrizeInputs<value_idx, value_t> params;
};

template <typename T>
struct COOSymmetrizeInputs {
  int nnz;
  int n_rows;
  int n_cols;
  std::vector<int> in_rows_h;
  std::vector<int> in_cols_h;
  std::vector<T> in_vals_h;
  std::vector<int> exp_rows_h;
  std::vector<int> exp_cols_h;
  std::vector<float> exp_vals_h;
};

template <typename T>
class COOSymmetrizeTest : public ::testing::TestWithParam<COOSymmetrizeInputs<T>> {
 public:
  COOSymmetrizeTest() : params(::testing::TestWithParam<COOSymmetrizeInputs<T>>::GetParam()) {}

 protected:
  void SetUp() override {}

  void TearDown() override {}

  COOSymmetrizeInputs<T> params;
};

typedef COOSymmetrizeTest<float> COOSymmetrizeView;
TEST_P(COOSymmetrizeView, ResultView)
{
  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);

  rmm::device_uvector<int> in_rows(params.nnz, stream);
  rmm::device_uvector<int> in_cols(params.nnz, stream);
  rmm::device_uvector<float> in_vals(params.nnz, stream);

  raft::update_device(in_rows.data(), params.in_rows_h.data(), params.nnz, stream);
  raft::update_device(in_cols.data(), params.in_cols_h.data(), params.nnz, stream);
  raft::update_device(in_vals.data(), params.in_vals_h.data(), params.nnz, stream);

  auto coo_structure_in = raft::make_device_coordinate_structure_view(
    in_rows.data(), in_cols.data(), params.n_rows, params.n_cols, params.nnz);

  auto in_view = raft::make_device_coo_matrix_view((const float*)in_vals.data(), coo_structure_in);

  auto out_matrix =
    raft::make_device_coo_matrix<float, int, int, int>(handle, params.n_rows, params.n_cols);

  linalg::coo_symmetrize(
    handle, in_view, out_matrix, [] __device__(int row, int col, float val, float trans) {
      return val + trans;
    });

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  auto out_nnz = out_matrix.structure_view().get_nnz();
  ASSERT_EQ(out_nnz, params.nnz * 2);

  std::vector<int> out_rows_h(out_nnz);
  std::vector<int> out_cols_h(out_nnz);
  std::vector<float> out_vals_h(out_nnz);

  raft::update_host(
    out_rows_h.data(), out_matrix.structure_view().get_rows().data(), out_nnz, stream);
  raft::update_host(
    out_cols_h.data(), out_matrix.structure_view().get_cols().data(), out_nnz, stream);
  raft::update_host(out_vals_h.data(), out_matrix.view().get_elements().data(), out_nnz, stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  ASSERT_TRUE(raft::devArrMatch<int>(out_matrix.structure_view().get_rows().data(),
                                     params.exp_rows_h.data(),
                                     out_nnz,
                                     raft::Compare<int>()));
  ASSERT_TRUE(raft::devArrMatch<int>(out_matrix.structure_view().get_cols().data(),
                                     params.exp_cols_h.data(),
                                     out_nnz,
                                     raft::Compare<int>()));
  ASSERT_TRUE(raft::devArrMatch<float>(out_matrix.view().get_elements().data(),
                                       params.exp_vals_h.data(),
                                       out_nnz,
                                       raft::Compare<float>()));
}

const std::vector<COOSymmetrizeInputs<float>> inputsf = {
  // first test fails without fix in #2582
  {
    2,             // nnz
    2,             // n_rows
    2,             // n_cols
    {0, 1},        // in_rows
    {0, 0},        // in_cols
    {0.0, 1.0},    // in_vals
    {0, 0, 0, 1},  // out_rows
    {0, 0, 1, 0},  // out_cols
    {0, 0, 1, 1}   // out_vals
  },
  {
    8,                                                                          // nnz
    4,                                                                          // n_rows
    4,                                                                          // n_cols
    {0, 0, 1, 1, 2, 2, 3, 3},                                                   // in_rows
    {1, 3, 2, 3, 0, 1, 0, 2},                                                   // in_cols
    {0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5},                                   // in_vals
    {1, 0, 0, 0, 1, 3, 1, 0, 0, 2, 2, 0, 3, 2, 3, 0},                           // out_rows
    {0, 1, 3, 0, 2, 1, 3, 0, 2, 0, 1, 0, 0, 3, 2, 0},                           // out_cols
    {0.5, 0.5, 1.5, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5, 0, 1.5, 0.5, 0.5, 0.0}  // out_vals
  }};

INSTANTIATE_TEST_CASE_P(COOSymmetrizeTest, COOSymmetrizeView, ::testing::ValuesIn(inputsf));

const std::vector<SparseSymmetrizeInputs<int, float>> symm_inputs_fint = {
  // Test n_clusters == n_points
  {
    2,
    {0, 2, 4, 6, 8},
    {0, 1, 0, 1, 0, 1, 0, 1},
    {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f},
  },
  {2,
   {0, 2, 4, 6, 8},
   {0, 1, 0, 1, 0, 1, 0, 1},  // indices
   {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f}},

};

typedef SparseSymmetrizeTest<int, float, uint64_t> SparseSymmetrizeTestF_int;
TEST_P(SparseSymmetrizeTestF_int, Result) { ASSERT_TRUE(sum_h == 0); }

INSTANTIATE_TEST_CASE_P(SparseSymmetrizeTest,
                        SparseSymmetrizeTestF_int,
                        ::testing::ValuesIn(symm_inputs_fint));

}  // namespace sparse
}  // namespace raft
