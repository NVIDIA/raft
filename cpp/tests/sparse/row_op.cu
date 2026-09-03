/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/op/row_op.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename Type_f, typename Index_>
struct CSRRowOpInputs {
  std::vector<Index_> ex_scan;
  std::vector<Type_f> verify;
};

/** Wrapper to call csr_row_op because the enclosing function of a __device__
 *  lambda cannot have private ot protected access within the class. */
template <typename Type_f, typename Index_>
void csr_row_op_wrapper(
  const Index_* row_ind, Index_ n_rows, Index_ nnz, Type_f* result, cudaStream_t stream)
{
  op::csr_row_op<Index_>(
    row_ind,
    n_rows,
    nnz,
    [result] __device__(Index_ row, Index_ start_idx, Index_ stop_idx) {
      for (Index_ i = start_idx; i < stop_idx; i++)
        result[i] = row;
    },
    stream);
}

template <typename Type_f, typename Index_>
class CSRRowOpTest : public ::testing::TestWithParam<CSRRowOpInputs<Type_f, Index_>> {
 public:
  CSRRowOpTest()
    : params(::testing::TestWithParam<CSRRowOpInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      verify(params.verify.size(), stream),
      ex_scan(params.ex_scan.size(), stream),
      result(params.verify.size(), stream)
  {
  }

 protected:
  void SetUp() override
  {
    n_rows = params.ex_scan.size();
    nnz    = params.verify.size();
  }

  void Run()
  {
    raft::update_device(ex_scan.data(), params.ex_scan.data(), n_rows, stream);
    raft::update_device(verify.data(), params.verify.data(), nnz, stream);

    csr_row_op_wrapper<Type_f, Index_>(ex_scan.data(), n_rows, nnz, result.data(), stream);

    ASSERT_TRUE(raft::devArrMatch<Type_f>(
      verify.data(), result.data(), nnz, raft::Compare<Type_f>(), stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  CSRRowOpInputs<Type_f, Index_> params;
  Index_ n_rows, nnz;
  rmm::device_uvector<Index_> ex_scan;
  rmm::device_uvector<Type_f> result, verify;
};

using CSRRowOpTestF = CSRRowOpTest<float, int>;
TEST_P(CSRRowOpTestF, Result) { Run(); }

using CSRRowOpTestD = CSRRowOpTest<double, int>;
TEST_P(CSRRowOpTestD, Result) { Run(); }

const std::vector<CSRRowOpInputs<float, int>> csrrowop_inputs_f = {
  {{0, 4, 8, 9}, {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0}},
};
const std::vector<CSRRowOpInputs<double, int>> csrrowop_inputs_d = {
  {{0, 4, 8, 9}, {0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 3.0}},
};

INSTANTIATE_TEST_CASE_P(SparseRowOpTest, CSRRowOpTestF, ::testing::ValuesIn(csrrowop_inputs_f));
INSTANTIATE_TEST_CASE_P(SparseRowOpTest, CSRRowOpTestD, ::testing::ValuesIn(csrrowop_inputs_d));

}  // namespace sparse
}  // namespace raft
