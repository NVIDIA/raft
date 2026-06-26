/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "add.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/add.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename InT, typename OutT = InT>
class AddTest : public ::testing::TestWithParam<AddInputs<InT, OutT>> {
 public:
  AddTest()
    : params(::testing::TestWithParam<AddInputs<InT, OutT>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      in1(params.len, stream),
      in2(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<AddInputs<InT, OutT>>::GetParam();
    raft::random::RngState r{params.seed};
    int len = params.len;
    uniform(handle, r, in1.data(), len, InT(-1.0), InT(1.0));
    uniform(handle, r, in2.data(), len, InT(-1.0), InT(1.0));
    naiveAddElem<InT, OutT>(out_ref.data(), in1.data(), in2.data(), len, stream);

    auto out_view = raft::make_device_vector_view(out.data(), out.size());
    auto in1_view = raft::make_device_vector_view<const InT>(in1.data(), in1.size());
    auto in2_view = raft::make_device_vector_view<const InT>(in2.data(), in2.size());

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) { add(h, in1_view, in2_view, out_view); },
      raft::alloc_behavior::NO_ALLOCATIONS);
  }

  void compare()
  {
    ASSERT_TRUE(raft::devArrMatch(
      out_ref.data(), out.data(), params.len, raft::CompareApprox<OutT>(params.tolerance), stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  AddInputs<InT, OutT> params;
  rmm::device_uvector<InT> in1;
  rmm::device_uvector<InT> in2;
  rmm::device_uvector<OutT> out_ref;
  rmm::device_uvector<OutT> out;
};

const std::vector<AddInputs<float>> inputsf = {
  {0.000001f, 1024 * 1024, 1234ULL},
  {0.000001f, 1024 * 1024 + 2, 1234ULL},
  {0.000001f, 1024 * 1024 + 1, 1234ULL},
};
typedef AddTest<float> AddTestF;
TEST_P(AddTestF, Result) { compare(); }
INSTANTIATE_TEST_SUITE_P(AddTests, AddTestF, ::testing::ValuesIn(inputsf));

const std::vector<AddInputs<double>> inputsd = {
  {0.00000001, 1024 * 1024, 1234ULL},
  {0.00000001, 1024 * 1024 + 2, 1234ULL},
  {0.00000001, 1024 * 1024 + 1, 1234ULL},
};
typedef AddTest<double> AddTestD;
TEST_P(AddTestD, Result) { compare(); }
INSTANTIATE_TEST_SUITE_P(AddTests, AddTestD, ::testing::ValuesIn(inputsd));

const std::vector<AddInputs<float, double>> inputsfd = {
  {0.00000001, 1024 * 1024, 1234ULL},
  {0.00000001, 1024 * 1024 + 2, 1234ULL},
  {0.00000001, 1024 * 1024 + 1, 1234ULL},
};
typedef AddTest<float, double> AddTestFD;
TEST_P(AddTestFD, Result) { compare(); }
INSTANTIATE_TEST_SUITE_P(AddTests, AddTestFD, ::testing::ValuesIn(inputsfd));

}  // end namespace linalg
}  // end namespace raft
