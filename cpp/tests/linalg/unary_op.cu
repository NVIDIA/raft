/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "unary_op.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename InType, typename IdxType, typename OutType = InType>
class UnaryOpTest : public ::testing::TestWithParam<UnaryOpInputs<InType, IdxType, OutType>> {
 public:
  UnaryOpTest()
    : params(::testing::TestWithParam<UnaryOpInputs<InType, IdxType, OutType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      in(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    auto len = params.len;
    uniform(handle, r, in.data(), len, InType(-1.0), InType(1.0));
    resource::sync_stream(handle, stream);
  }

  virtual void DoTest()
  {
    auto len    = params.len;
    auto scalar = params.scalar;
    naiveScale(out_ref.data(), in.data(), scalar, len, stream);

    auto in_view  = raft::make_device_vector_view<const InType>(in.data(), len);
    auto out_view = raft::make_device_vector_view(out.data(), len);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        unary_op(h,
                 in_view,
                 out_view,
                 raft::compose_op(raft::cast_op<OutType>(), raft::mul_const_op<InType>(scalar)));
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  UnaryOpInputs<InType, IdxType, OutType> params;
  rmm::device_uvector<InType> in;
  rmm::device_uvector<OutType> out_ref, out;
};

// Or else, we get the following compilation error:
// The enclosing parent function ("DoTest") for an extended __device__ lambda cannot have private or
// protected access within its class
template <typename InType, typename IdxType, typename OutType>
void launchWriteOnlyUnaryOp(const raft::resources& handle, OutType* out, InType scalar, IdxType len)
{
  auto out_view = raft::make_device_vector_view(out, len);
  auto op       = [scalar] __device__(OutType * ptr, IdxType idx) {
    *ptr = static_cast<OutType>(scalar * idx);
  };
  write_only_unary_op(handle, out_view, op);
}

template <typename OutType, typename IdxType>
class WriteOnlyUnaryOpTest : public UnaryOpTest<OutType, IdxType, OutType> {
 protected:
  void DoTest() override
  {
    auto len    = this->params.len;
    auto scalar = this->params.scalar;
    naiveScale(this->out_ref.data(), (OutType*)nullptr, scalar, len, this->stream);

    launchWriteOnlyUnaryOp(this->handle, this->out.data(), scalar, len);
    resource::sync_stream(this->handle, this->stream);
  }
};

#define UNARY_OP_TEST(test_type, test_name, inputs)                  \
  typedef RAFT_DEPAREN(test_type) test_name;                         \
  TEST_P(test_name, Result)                                          \
  {                                                                  \
    DoTest();                                                        \
    ASSERT_TRUE(devArrMatch(this->out_ref.data(),                    \
                            this->out.data(),                        \
                            this->params.len,                        \
                            CompareApprox(this->params.tolerance))); \
  }                                                                  \
  INSTANTIATE_TEST_SUITE_P(UnaryOpTests, test_name, ::testing::ValuesIn(inputs))

const std::vector<UnaryOpInputs<float, int>> inputsf_i32 = {{0.000001f, 1024 * 1024, 2.f, 1234ULL}};
UNARY_OP_TEST((UnaryOpTest<float, int>), UnaryOpTestF_i32, inputsf_i32);
UNARY_OP_TEST((WriteOnlyUnaryOpTest<float, int>), WriteOnlyUnaryOpTestF_i32, inputsf_i32);

const std::vector<UnaryOpInputs<float, size_t>> inputsf_i64 = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
UNARY_OP_TEST((UnaryOpTest<float, size_t>), UnaryOpTestF_i64, inputsf_i64);
UNARY_OP_TEST((WriteOnlyUnaryOpTest<float, size_t>), WriteOnlyUnaryOpTestF_i64, inputsf_i64);

const std::vector<UnaryOpInputs<float, int, double>> inputsf_i32_d = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
UNARY_OP_TEST((UnaryOpTest<float, int, double>), UnaryOpTestF_i32_D, inputsf_i32_d);

const std::vector<UnaryOpInputs<double, int>> inputsd_i32 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};
UNARY_OP_TEST((UnaryOpTest<double, int>), UnaryOpTestD_i32, inputsd_i32);
UNARY_OP_TEST((WriteOnlyUnaryOpTest<double, int>), WriteOnlyUnaryOpTestD_i32, inputsd_i32);

const std::vector<UnaryOpInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};
UNARY_OP_TEST((UnaryOpTest<double, size_t>), UnaryOpTestD_i64, inputsd_i64);
UNARY_OP_TEST((WriteOnlyUnaryOpTest<double, size_t>), WriteOnlyUnaryOpTestD_i64, inputsd_i64);

}  // end namespace linalg
}  // end namespace raft
