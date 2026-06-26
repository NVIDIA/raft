/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/ternary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename InType, typename IdxType = int, typename OutType = InType>
struct BinaryOpInputs {
  InType tolerance;
  IdxType len;
  unsigned long long int seed;
};

template <typename InType, typename IdxType = int, typename OutType = InType>
::std::ostream& operator<<(::std::ostream& os, const BinaryOpInputs<InType, IdxType, OutType>& d)
{
  return os;
}

template <typename T>
class ternaryOpTest : public ::testing::TestWithParam<BinaryOpInputs<T>> {
 public:
  ternaryOpTest()
    : params(::testing::TestWithParam<BinaryOpInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      out_add_ref(params.len, stream),
      out_add(params.len, stream),
      out_mul_ref(params.len, stream),
      out_mul(params.len, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState rng(params.seed);
    int len = params.len;
    rmm::device_uvector<T> in1(len, stream);
    rmm::device_uvector<T> in2(len, stream);
    rmm::device_uvector<T> in3(len, stream);

    fill(handle, rng, out_add_ref.data(), len, T(6.0));
    fill(handle, rng, out_mul_ref.data(), len, T(6.0));
    fill(handle, rng, in1.data(), len, T(1.0));
    fill(handle, rng, in2.data(), len, T(2.0));
    fill(handle, rng, in3.data(), len, T(3.0));

    auto add          = [] __device__(T a, T b, T c) { return a + b + c; };
    auto mul          = [] __device__(T a, T b, T c) { return a * b * c; };
    auto out_add_view = raft::make_device_vector_view(out_add.data(), len);
    auto out_mul_view = raft::make_device_vector_view(out_mul.data(), len);
    auto in1_view     = raft::make_device_vector_view<const T>(in1.data(), len);
    auto in2_view     = raft::make_device_vector_view<const T>(in2.data(), len);
    auto in3_view     = raft::make_device_vector_view<const T>(in3.data(), len);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        ternary_op(h, in1_view, in2_view, in3_view, out_add_view, add);
        ternary_op(h, in1_view, in2_view, in3_view, out_mul_view, mul);
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
  }

 protected:
  BinaryOpInputs<T> params;
  raft::resources handle;
  cudaStream_t stream = 0;

  rmm::device_uvector<T> out_add_ref, out_add, out_mul_ref, out_mul;
};

const std::vector<BinaryOpInputs<float>> inputsf = {{0.000001f, 1024 * 1024, 1234ULL},
                                                    {0.000001f, 1024 * 1024 + 2, 1234ULL},
                                                    {0.000001f, 1024 * 1024 + 1, 1234ULL}};
typedef ternaryOpTest<float> ternaryOpTestF;
TEST_P(ternaryOpTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_add_ref.data(), out_add.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(devArrMatch(
    out_mul_ref.data(), out_mul.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ternaryOpTests, ternaryOpTestF, ::testing::ValuesIn(inputsf));

const std::vector<BinaryOpInputs<double>> inputsd = {{0.00000001, 1024 * 1024, 1234ULL},
                                                     {0.00000001, 1024 * 1024 + 2, 1234ULL},
                                                     {0.00000001, 1024 * 1024 + 1, 1234ULL}};
typedef ternaryOpTest<double> ternaryOpTestD;
TEST_P(ternaryOpTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_add_ref.data(), out_add.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
  ASSERT_TRUE(devArrMatch(
    out_mul_ref.data(), out_mul.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ternaryOpTests, ternaryOpTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
