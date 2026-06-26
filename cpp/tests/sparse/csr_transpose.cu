/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/linalg/transpose.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cusparse_v2.h>
#include <gtest/gtest.h>

namespace raft {
namespace sparse {

using namespace raft;

template <typename value_idx, typename value_t>
struct CSRTransposeInputs {
  value_idx nrows;
  value_idx ncols;
  value_idx nnz;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_idx> out_indptr_ref_h;
  std::vector<value_idx> out_indices_ref_h;
  std::vector<value_t> out_data_ref_h;
};

template <typename value_idx, typename value_t>
::std::ostream& operator<<(::std::ostream& os, const CSRTransposeInputs<value_idx, value_t>& dims)
{
  return os;
}

template <typename value_idx, typename value_t>
class CSRTransposeTest : public ::testing::TestWithParam<CSRTransposeInputs<value_idx, value_t>> {
 public:
  CSRTransposeTest()
    : params(::testing::TestWithParam<CSRTransposeInputs<value_idx, value_t>>::GetParam()),
      stream(resource::get_cuda_stream(raft_handle)),
      indptr(0, stream),
      indices(0, stream),
      data(0, stream),
      out_indptr_ref(0, stream),
      out_indices_ref(0, stream),
      out_data_ref(0, stream),
      out_indptr(0, stream),
      out_indices(0, stream),
      out_data(0, stream)
  {
    indptr.resize(params.indptr_h.size(), stream);
    indices.resize(params.indices_h.size(), stream);
    data.resize(params.data_h.size(), stream);
    out_indptr_ref.resize(params.out_indptr_ref_h.size(), stream);
    out_indices_ref.resize(params.out_indices_ref_h.size(), stream);
    out_data_ref.resize(params.out_data_ref_h.size(), stream);
    out_indptr.resize(params.out_indptr_ref_h.size(), stream);
    out_indices.resize(params.out_indices_ref_h.size(), stream);
    out_data.resize(params.out_data_ref_h.size(), stream);
  }

 protected:
  void make_data()
  {
    std::vector<value_idx> indptr_h  = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h      = params.data_h;

    update_device(indptr.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(indices.data(), indices_h.data(), indices_h.size(), stream);
    update_device(data.data(), data_h.data(), data_h.size(), stream);

    std::vector<value_idx> out_indptr_ref_h  = params.out_indptr_ref_h;
    std::vector<value_idx> out_indices_ref_h = params.out_indices_ref_h;
    std::vector<value_t> out_data_ref_h      = params.out_data_ref_h;

    update_device(out_indptr_ref.data(), out_indptr_ref_h.data(), out_indptr_ref_h.size(), stream);
    update_device(
      out_indices_ref.data(), out_indices_ref_h.data(), out_indices_ref_h.size(), stream);
    update_device(out_data_ref.data(), out_data_ref_h.data(), out_data_ref_h.size(), stream);
  }

  void SetUp() override
  {
    raft::resources handle;

    make_data();

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        raft::sparse::linalg::csr_transpose(h,
                                            indptr.data(),
                                            indices.data(),
                                            data.data(),
                                            out_indptr.data(),
                                            out_indices.data(),
                                            out_data.data(),
                                            params.nrows,
                                            params.ncols,
                                            params.nnz,
                                            resource::get_cuda_stream(h));
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      1);

    resource::sync_stream(handle, stream);
  }

  void compare()
  {
    ASSERT_TRUE(devArrMatch(out_indptr.data(),
                            out_indptr_ref.data(),
                            params.out_indptr_ref_h.size(),
                            Compare<value_t>()));
    ASSERT_TRUE(devArrMatch(out_indices.data(),
                            out_indices_ref.data(),
                            params.out_indices_ref_h.size(),
                            Compare<value_t>()));
    ASSERT_TRUE(devArrMatch(
      out_data.data(), out_data_ref.data(), params.out_data_ref_h.size(), Compare<value_t>()));
  }

 protected:
  raft::resources raft_handle;
  cudaStream_t stream;

  cusparseHandle_t handle;

  // input data
  rmm::device_uvector<value_idx> indptr, indices;
  rmm::device_uvector<value_t> data;

  // output data
  rmm::device_uvector<value_idx> out_indptr, out_indices;
  rmm::device_uvector<value_t> out_data;

  // expected output data
  rmm::device_uvector<value_idx> out_indptr_ref, out_indices_ref;
  rmm::device_uvector<value_t> out_data_ref;

  CSRTransposeInputs<value_idx, value_t> params;
};

const std::vector<CSRTransposeInputs<int, float>> inputs_i32_f = {
  {
    4,
    2,
    8,
    {0, 2, 4, 6, 8},
    {0, 1, 0, 1, 0, 1, 0, 1},  // indices
    {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f},
    {0, 4, 8},
    {0, 1, 2, 3, 0, 1, 2, 3},  // indices
    {1.0f, 1.0f, 50.0f, 16.0f, 3.0f, 5.0f, 28.0f, 2.0f},
  },
};
typedef CSRTransposeTest<int, float> CSRTransposeTestF;
TEST_P(CSRTransposeTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(CSRTransposeTest, CSRTransposeTestF, ::testing::ValuesIn(inputs_i32_f));

};  // end namespace sparse
};  // end namespace raft
