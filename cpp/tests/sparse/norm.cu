/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/sparse/linalg/norm.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename Type_f, typename Index_>
struct CSRRowNormInputs {
  raft::linalg::NormType norm;
  std::vector<Index_> indptr;
  std::vector<Type_f> data;
  std::vector<Type_f> verify;
};

template <typename Type_f, typename Index_>
class CSRRowNormTest : public ::testing::TestWithParam<CSRRowNormInputs<Type_f, Index_>> {
 public:
  CSRRowNormTest()
    : params(::testing::TestWithParam<CSRRowNormInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.data.size(), stream),
      verify(params.indptr.size() - 1, stream),
      indptr(params.indptr.size(), stream),
      result(params.indptr.size() - 1, stream)
  {
  }

 protected:
  void SetUp() override {}

  void Run()
  {
    Index_ n_rows = params.indptr.size() - 1;
    Index_ nnz    = params.data.size();

    raft::update_device(indptr.data(), params.indptr.data(), n_rows + 1, stream);
    raft::update_device(data.data(), params.data.data(), nnz, stream);
    raft::update_device(verify.data(), params.verify.data(), n_rows, stream);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        linalg::rowNormCsr(h, indptr.data(), data.data(), nnz, n_rows, result.data(), params.norm);
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    ASSERT_TRUE(
      raft::devArrMatch<Type_f>(verify.data(), result.data(), n_rows, raft::Compare<Type_f>()));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  CSRRowNormInputs<Type_f, Index_> params;
  rmm::device_uvector<Index_> indptr;
  rmm::device_uvector<Type_f> data, result, verify;
};

using CSRRowNormTestF = CSRRowNormTest<float, int>;
TEST_P(CSRRowNormTestF, Result) { Run(); }

using CSRRowNormTestD = CSRRowNormTest<double, int>;
TEST_P(CSRRowNormTestD, Result) { Run(); }

const std::vector<CSRRowNormInputs<float, int>> csrnorm_inputs_f = {
  {raft::linalg::NormType::LinfNorm,
   {0, 3, 7, 10},
   {5.0, 1.0, 2.0, 0.0, 10.0, 1.0, 2.0, 1.0, 1.0, 2.0},
   {5.0, 10.0, 2.0}},
  {raft::linalg::NormType::L1Norm,
   {0, 3, 7, 10},
   {5.0, 1.0, 2.0, 0.0, 10.0, 1.0, 2.0, 1.0, 1.0, 2.0},
   {8.0, 13.0, 4.0}},
  {raft::linalg::NormType::L2Norm,
   {0, 3, 7, 10},
   {5.0, 1.0, 2.0, 0.0, 10.0, 1.0, 2.0, 1.0, 1.0, 2.0},
   {30.0, 105.0, 6.0}},
};
const std::vector<CSRRowNormInputs<double, int>> csrnorm_inputs_d = {
  {raft::linalg::NormType::LinfNorm,
   {0, 3, 7, 10},
   {5.0, 1.0, 2.0, 0.0, 10.0, 1.0, 2.0, 1.0, 1.0, 2.0},
   {5.0, 10.0, 2.0}},
  {raft::linalg::NormType::L1Norm,
   {0, 3, 7, 10},
   {5.0, 1.0, 2.0, 0.0, 10.0, 1.0, 2.0, 1.0, 1.0, 2.0},
   {8.0, 13.0, 4.0}},
  {raft::linalg::NormType::L2Norm,
   {0, 3, 7, 10},
   {5.0, 1.0, 2.0, 0.0, 10.0, 1.0, 2.0, 1.0, 1.0, 2.0},
   {30.0, 105.0, 6.0}},
};

INSTANTIATE_TEST_CASE_P(SparseNormTest, CSRRowNormTestF, ::testing::ValuesIn(csrnorm_inputs_f));
INSTANTIATE_TEST_CASE_P(SparseNormTest, CSRRowNormTestD, ::testing::ValuesIn(csrnorm_inputs_d));

}  // namespace sparse
}  // namespace raft
