/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/linalg/add.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename Type_f, typename Index_>
struct CSRMatrixVal {
  std::vector<Index_> row_ind;
  std::vector<Index_> row_ind_ptr;
  std::vector<Type_f> values;
};

template <typename Type_f, typename Index_>
struct CSRAddInputs {
  CSRMatrixVal<Type_f, Index_> matrix_a;
  CSRMatrixVal<Type_f, Index_> matrix_b;
  CSRMatrixVal<Type_f, Index_> matrix_verify;
};

template <typename Type_f, typename Index_>
class CSRAddTest : public ::testing::TestWithParam<CSRAddInputs<Type_f, Index_>> {
 public:
  CSRAddTest()
    : params(::testing::TestWithParam<CSRAddInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      ind_a(params.matrix_a.row_ind.size(), stream),
      ind_ptr_a(params.matrix_a.row_ind_ptr.size(), stream),
      values_a(params.matrix_a.row_ind_ptr.size(), stream),
      ind_b(params.matrix_a.row_ind.size(), stream),
      ind_ptr_b(params.matrix_b.row_ind_ptr.size(), stream),
      values_b(params.matrix_b.row_ind_ptr.size(), stream),
      ind_verify(params.matrix_a.row_ind.size(), stream),
      ind_ptr_verify(params.matrix_verify.row_ind_ptr.size(), stream),
      values_verify(params.matrix_verify.row_ind_ptr.size(), stream),
      ind_result(params.matrix_a.row_ind.size(), stream),
      ind_ptr_result(params.matrix_verify.row_ind_ptr.size(), stream),
      values_result(params.matrix_verify.row_ind_ptr.size(), stream)
  {
  }

 protected:
  void SetUp() override
  {
    n_rows     = params.matrix_a.row_ind.size();
    nnz_a      = params.matrix_a.row_ind_ptr.size();
    nnz_b      = params.matrix_b.row_ind_ptr.size();
    nnz_result = params.matrix_verify.row_ind_ptr.size();
  }

  void Run()
  {
    raft::update_device(ind_a.data(), params.matrix_a.row_ind.data(), n_rows, stream);
    raft::update_device(ind_ptr_a.data(), params.matrix_a.row_ind_ptr.data(), nnz_a, stream);
    raft::update_device(values_a.data(), params.matrix_a.values.data(), nnz_a, stream);

    raft::update_device(ind_b.data(), params.matrix_b.row_ind.data(), n_rows, stream);
    raft::update_device(ind_ptr_b.data(), params.matrix_b.row_ind_ptr.data(), nnz_b, stream);
    raft::update_device(values_b.data(), params.matrix_b.values.data(), nnz_b, stream);

    raft::update_device(ind_verify.data(), params.matrix_verify.row_ind.data(), n_rows, stream);
    raft::update_device(
      ind_ptr_verify.data(), params.matrix_verify.row_ind_ptr.data(), nnz_result, stream);
    raft::update_device(
      values_verify.data(), params.matrix_verify.values.data(), nnz_result, stream);

    Index_ nnz = linalg::csr_add_calc_inds<Type_f>(ind_a.data(),
                                                   ind_ptr_a.data(),
                                                   values_a.data(),
                                                   nnz_a,
                                                   ind_b.data(),
                                                   ind_ptr_b.data(),
                                                   values_b.data(),
                                                   nnz_b,
                                                   n_rows,
                                                   ind_result.data(),
                                                   stream);

    ASSERT_TRUE(nnz == nnz_result);
    ASSERT_TRUE(raft::devArrMatch<Index_>(
      ind_verify.data(), ind_result.data(), n_rows, raft::Compare<Index_>(), stream));

    linalg::csr_add_finalize<Type_f>(ind_a.data(),
                                     ind_ptr_a.data(),
                                     values_a.data(),
                                     nnz_a,
                                     ind_b.data(),
                                     ind_ptr_b.data(),
                                     values_b.data(),
                                     nnz_b,
                                     n_rows,
                                     ind_result.data(),
                                     ind_ptr_result.data(),
                                     values_result.data(),
                                     stream);

    ASSERT_TRUE(raft::devArrMatch<Index_>(
      ind_ptr_verify.data(), ind_ptr_result.data(), nnz, raft::Compare<Index_>(), stream));
    ASSERT_TRUE(raft::devArrMatch<Type_f>(
      values_verify.data(), values_result.data(), nnz, raft::Compare<Type_f>(), stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  CSRAddInputs<Type_f, Index_> params;
  Index_ n_rows, nnz_a, nnz_b, nnz_result;
  rmm::device_uvector<Index_> ind_a, ind_b, ind_verify, ind_result, ind_ptr_a, ind_ptr_b,
    ind_ptr_verify, ind_ptr_result;
  rmm::device_uvector<Type_f> values_a, values_b, values_verify, values_result;
};

using CSRAddTestF = CSRAddTest<float, int>;
TEST_P(CSRAddTestF, Result) { Run(); }

using CSRAddTestD = CSRAddTest<double, int>;
TEST_P(CSRAddTestD, Result) { Run(); }

const std::vector<CSRAddInputs<float, int>> csradd_inputs_f = {
  {{{0, 4, 8, 9},
    {1, 2, 3, 4, 1, 2, 3, 5, 0, 1},
    {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0}},
   {{0, 4, 8, 9},
    {1, 2, 5, 4, 0, 2, 3, 5, 1, 0},
    {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0}},
   {{0, 5, 10, 12},
    {1, 2, 3, 4, 5, 1, 2, 3, 5, 0, 0, 1, 1, 0},
    {2.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}},
};
const std::vector<CSRAddInputs<double, int>> csradd_inputs_d = {
  {{{0, 4, 8, 9},
    {1, 2, 3, 4, 1, 2, 3, 5, 0, 1},
    {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0}},
   {{0, 4, 8, 9},
    {1, 2, 5, 4, 0, 2, 3, 5, 1, 0},
    {1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0}},
   {{0, 5, 10, 12},
    {1, 2, 3, 4, 5, 1, 2, 3, 5, 0, 0, 1, 1, 0},
    {2.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}}},
};

INSTANTIATE_TEST_CASE_P(SparseAddTest, CSRAddTestF, ::testing::ValuesIn(csradd_inputs_f));
INSTANTIATE_TEST_CASE_P(SparseAddTest, CSRAddTestD, ::testing::ValuesIn(csradd_inputs_d));

}  // namespace sparse
}  // namespace raft
