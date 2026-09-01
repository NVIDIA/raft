/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

struct SPGemmiInputs {
  int n_rows, n_cols;
};

template <typename data_t>
class SPGemmiTest : public ::testing::TestWithParam<SPGemmiInputs> {
 public:
  SPGemmiTest()
    : params(::testing::TestWithParam<SPGemmiInputs>::GetParam()),
      stream(resource::get_cuda_stream(handle).get())
  {
  }

 protected:
  void SetUp() override {}

  void Run()
  {
    // Host problem definition
    float alpha    = 1.0f;
    float beta     = 0.0f;
    int A_num_rows = 5;
    int A_num_cols = 3;
    // int   B_num_rows      = A_num_cols;
    int B_num_cols      = 4;
    int B_nnz           = 9;
    int lda             = A_num_rows;
    int ldc             = A_num_rows;
    int A_size          = lda * A_num_cols;
    int C_size          = ldc * B_num_cols;
    int hB_cscOffsets[] = {0, 3, 4, 7, 9};
    int hB_rows[]       = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    float hB_values[]   = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    float hA[]          = {1.0f,
                           2.0f,
                           3.0f,
                           4.0f,
                           5.0f,
                           6.0f,
                           7.0f,
                           8.0f,
                           9.0f,
                           10.0f,
                           11.0f,
                           12.0f,
                           13.0f,
                           14.0f,
                           15.0f};
    std::vector<float> hC(C_size);
    std::vector<float> hC_expected{23, 26, 29, 32,  35,  24, 28, 32, 36, 40,
                                   71, 82, 93, 104, 115, 48, 56, 64, 72, 80};
    //--------------------------------------------------------------------------
    // Device memory management
    rmm::device_uvector<int> dB_cscOffsets(B_num_cols + 1, stream);
    rmm::device_uvector<int> dB_rows(B_nnz, stream);
    rmm::device_uvector<float> dB_values(B_nnz, stream);
    rmm::device_uvector<float> dA(A_size, stream);
    rmm::device_uvector<float> dC(C_size, stream);
    rmm::device_uvector<float> dCT(C_size, stream);

    raft::update_device(dB_cscOffsets.data(), hB_cscOffsets, B_num_cols + 1, stream);
    raft::update_device(dB_rows.data(), hB_rows, B_nnz, stream);
    raft::update_device(dB_values.data(), hB_values, B_nnz, stream);
    raft::update_device(dA.data(), hA, A_size, stream);
    raft::update_device(dC.data(), hC.data(), C_size, stream);

    //--------------------------------------------------------------------------
    // execute gemmi
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsegemmi(resource::get_cusparse_handle(handle),
                                                          A_num_rows,
                                                          B_num_cols,
                                                          A_num_cols,
                                                          B_nnz,
                                                          &alpha,
                                                          dA.data(),
                                                          lda,
                                                          dB_values.data(),
                                                          dB_cscOffsets.data(),
                                                          dB_rows.data(),
                                                          &beta,
                                                          dC.data(),
                                                          ldc,
                                                          resource::get_cuda_stream(handle).get()));

    //--------------------------------------------------------------------------
    // result check
    raft::update_host(hC.data(), dC.data(), C_size, stream);
    ASSERT_TRUE(hostVecMatch(hC_expected, hC, raft::Compare<float>()));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SPGemmiInputs params;
};

using SPGemmiTestF = SPGemmiTest<float>;
TEST_P(SPGemmiTestF, Result) { Run(); }

using SPGemmiTestD = SPGemmiTest<double>;
TEST_P(SPGemmiTestD, Result) { Run(); }

const std::vector<SPGemmiInputs> csc_inputs_f = {{5, 4}};
const std::vector<SPGemmiInputs> csc_inputs_d = {{5, 4}};

INSTANTIATE_TEST_CASE_P(SparseGemmi, SPGemmiTestF, ::testing::ValuesIn(csc_inputs_f));
INSTANTIATE_TEST_CASE_P(SparseGemmi, SPGemmiTestD, ::testing::ValuesIn(csc_inputs_d));

}  // namespace sparse
}  // namespace raft
