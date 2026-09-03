/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/csr.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename Index_>
struct CSRtoCOOInputs {
  std::vector<Index_> ex_scan;
  std::vector<Index_> verify;
};

template <typename Index_>
class CSRtoCOOTest : public ::testing::TestWithParam<CSRtoCOOInputs<Index_>> {
 public:
  CSRtoCOOTest()
    : params(::testing::TestWithParam<CSRtoCOOInputs<Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      ex_scan(params.ex_scan.size(), stream),
      verify(params.verify.size(), stream),
      result(params.verify.size(), stream)
  {
  }

 protected:
  void SetUp() override {}

  void Run()
  {
    Index_ n_rows = params.ex_scan.size();
    Index_ nnz    = params.verify.size();

    raft::update_device(ex_scan.data(), params.ex_scan.data(), n_rows, stream);
    raft::update_device(verify.data(), params.verify.data(), nnz, stream);

    convert::csr_to_coo<Index_>(ex_scan.data(), n_rows, result.data(), nnz, stream);

    ASSERT_TRUE(
      raft::devArrMatch<Index_>(verify.data(), result.data(), nnz, raft::Compare<float>(), stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  CSRtoCOOInputs<Index_> params;
  rmm::device_uvector<Index_> ex_scan, verify, result;
};

using CSRtoCOOTestI = CSRtoCOOTest<int>;
TEST_P(CSRtoCOOTestI, Result) { Run(); }

using CSRtoCOOTestL = CSRtoCOOTest<int64_t>;
TEST_P(CSRtoCOOTestL, Result) { Run(); }

const std::vector<CSRtoCOOInputs<int>> csrtocoo_inputs_32 = {
  {{0, 0, 2, 2}, {1, 1, 3}},
  {{0, 4, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 2, 3}},
};
const std::vector<CSRtoCOOInputs<int64_t>> csrtocoo_inputs_64 = {
  {{0, 0, 2, 2}, {1, 1, 3}},
  {{0, 4, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 2, 3}},
};

INSTANTIATE_TEST_CASE_P(SparseConvertCOOTest,
                        CSRtoCOOTestI,
                        ::testing::ValuesIn(csrtocoo_inputs_32));
INSTANTIATE_TEST_CASE_P(SparseConvertCOOTest,
                        CSRtoCOOTestL,
                        ::testing::ValuesIn(csrtocoo_inputs_64));

}  // namespace sparse
}  // namespace raft
