/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/linalg/norm.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

enum NormalizeMethod { MAX, L1 };

template <typename Type_f, typename Index_>
struct CSRRowNormalizeInputs {
  NormalizeMethod method;
  std::vector<Index_> ex_scan;
  std::vector<Type_f> in_vals;
  std::vector<Type_f> verify;
};

template <typename Type_f, typename Index_>
class CSRRowNormalizeTest : public ::testing::TestWithParam<CSRRowNormalizeInputs<Type_f, Index_>> {
 public:
  CSRRowNormalizeTest()
    : params(::testing::TestWithParam<CSRRowNormalizeInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      in_vals(params.in_vals.size(), stream),
      verify(params.verify.size(), stream),
      ex_scan(params.ex_scan.size(), stream),
      result(params.verify.size(), stream)
  {
  }

 protected:
  void SetUp() override {}

  void Run()
  {
    Index_ n_rows = params.ex_scan.size();
    Index_ nnz    = params.in_vals.size();

    raft::update_device(ex_scan.data(), params.ex_scan.data(), n_rows, stream);
    raft::update_device(in_vals.data(), params.in_vals.data(), nnz, stream);
    raft::update_device(verify.data(), params.verify.data(), nnz, stream);

    switch (params.method) {
      case MAX:
        linalg::csr_row_normalize_max<Type_f>(
          ex_scan.data(), in_vals.data(), nnz, n_rows, result.data(), stream);
        break;
      case L1:
        linalg::csr_row_normalize_l1<Type_f>(
          ex_scan.data(), in_vals.data(), nnz, n_rows, result.data(), stream);
        break;
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    ASSERT_TRUE(
      raft::devArrMatch<Type_f>(verify.data(), result.data(), nnz, raft::Compare<Type_f>()));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  CSRRowNormalizeInputs<Type_f, Index_> params;
  rmm::device_uvector<Index_> ex_scan;
  rmm::device_uvector<Type_f> in_vals, result, verify;
};

using CSRRowNormalizeTestF = CSRRowNormalizeTest<float, int>;
TEST_P(CSRRowNormalizeTestF, Result) { Run(); }

using CSRRowNormalizeTestD = CSRRowNormalizeTest<double, int>;
TEST_P(CSRRowNormalizeTestD, Result) { Run(); }

const std::vector<CSRRowNormalizeInputs<float, int>> csrnormalize_inputs_f = {
  {MAX,
   {0, 4, 8, 9},
   {5.0, 1.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 1.0, 0.0},
   {1.0, 0.2, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1, 0.0}},
  {L1,
   {0, 4, 8, 9},
   {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0},
   {0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1, 0.0}},
};
const std::vector<CSRRowNormalizeInputs<double, int>> csrnormalize_inputs_d = {
  {MAX,
   {0, 4, 8, 9},
   {5.0, 1.0, 0.0, 0.0, 10.0, 1.0, 0.0, 0.0, 1.0, 0.0},
   {1.0, 0.2, 0.0, 0.0, 1.0, 0.1, 0.0, 0.0, 1, 0.0}},
  {L1,
   {0, 4, 8, 9},
   {1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0},
   {0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1, 0.0}},
};

INSTANTIATE_TEST_CASE_P(SparseNormTest,
                        CSRRowNormalizeTestF,
                        ::testing::ValuesIn(csrnormalize_inputs_f));
INSTANTIATE_TEST_CASE_P(SparseNormTest,
                        CSRRowNormalizeTestD,
                        ::testing::ValuesIn(csrnormalize_inputs_d));

}  // namespace sparse
}  // namespace raft
