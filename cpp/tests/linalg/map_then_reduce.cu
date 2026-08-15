/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map_reduce.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/functional>

#include <gtest/gtest.h>

#include <limits>

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename MapOp>
RAFT_KERNEL naiveMapReduceKernel(OutType* out, const InType* in, size_t len, MapOp map)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { raft::myAtomicAdd(out, (OutType)map(in[idx])); }
}

template <typename InType, typename OutType, typename MapOp>
void naiveMapReduce(OutType* out, const InType* in, size_t len, MapOp map, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, (size_t)TPB);
  raft::launch_kernel(
    stream, nblks, TPB, naiveMapReduceKernel<InType, OutType, MapOp>, out, in, len, map);
}

template <typename T>
struct MapReduceInputs {
  T tolerance;
  size_t len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MapReduceInputs<T>& dims)
{
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename OutType>
void mapReduceLaunch(
  OutType* out_ref, OutType* out, const InType* in, size_t len, cudaStream_t stream)
{
  naiveMapReduce(out_ref, in, len, raft::identity_op{}, stream);
  mapThenSumReduce(out, len, raft::identity_op{}, 0, in);
}

template <typename InType, typename OutType>
class MapReduceTest : public ::testing::TestWithParam<MapReduceInputs<InType>> {
 public:
  MapReduceTest()
    : params(::testing::TestWithParam<MapReduceInputs<InType>>::GetParam()),
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
    mapReduceLaunch(out_ref.data(), out.data(), in.data(), len, stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MapReduceInputs<InType> params;
  rmm::device_uvector<InType> in;
  rmm::device_uvector<OutType> out_ref, out;
};

const std::vector<MapReduceInputs<float>> inputsf = {{0.001f, 1024 * 1024, 1234ULL}};
typedef MapReduceTest<float, float> MapReduceTestFF;
TEST_P(MapReduceTestFF, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_SUITE_P(MapReduceTests, MapReduceTestFF, ::testing::ValuesIn(inputsf));

typedef MapReduceTest<float, double> MapReduceTestFD;
TEST_P(MapReduceTestFD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_SUITE_P(MapReduceTests, MapReduceTestFD, ::testing::ValuesIn(inputsf));

const std::vector<MapReduceInputs<double>> inputsd = {{0.000001, 1024 * 1024, 1234ULL}};
typedef MapReduceTest<double, double> MapReduceTestDD;
TEST_P(MapReduceTestDD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_SUITE_P(MapReduceTests, MapReduceTestDD, ::testing::ValuesIn(inputsd));

template <typename T>
class MapGenericReduceTest : public ::testing::Test {
  using InType  = typename T::first_type;
  using OutType = typename T::second_type;

 protected:
  MapGenericReduceTest()
    : input(n, resource::get_cuda_stream(handle)), output(resource::get_cuda_stream(handle))
  {
    initInput(input.data(), input.size(), resource::get_cuda_stream(handle));
  }

 public:
  void initInput(InType* input, int n, cudaStream_t stream)
  {
    raft::random::RngState r(137);
    uniform(handle, r, input, n, InType(2), InType(3));
    InType val = 1;
    raft::update_device(input + 42, &val, 1, resource::get_cuda_stream(handle));
    val = 5;
    raft::update_device(input + 337, &val, 1, resource::get_cuda_stream(handle));
  }

  void testMin()
  {
    OutType neutral  = std::numeric_limits<InType>::max();
    auto output_view = raft::make_device_scalar_view(output.data());
    auto input_view  = raft::make_device_vector_view<const InType>(
      input.data(), static_cast<std::uint32_t>(input.size()));
    map_reduce(handle, input_view, output_view, neutral, raft::identity_op{}, cuda::minimum{});
    EXPECT_TRUE(raft::devArrMatch(
      OutType(1), output.data(), 1, raft::Compare<OutType>(), resource::get_cuda_stream(handle)));
  }
  void testMax()
  {
    OutType neutral  = std::numeric_limits<InType>::min();
    auto output_view = raft::make_device_scalar_view(output.data());
    auto input_view  = raft::make_device_vector_view<const InType>(
      input.data(), static_cast<std::uint32_t>(input.size()));
    map_reduce(handle, input_view, output_view, neutral, raft::identity_op{}, cuda::maximum{});
    EXPECT_TRUE(raft::devArrMatch(
      OutType(5), output.data(), 1, raft::Compare<OutType>(), resource::get_cuda_stream(handle)));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  int n = 1237;
  rmm::device_uvector<InType> input;
  rmm::device_scalar<OutType> output;
};

using IoTypePair =
  ::testing::Types<std::pair<float, float>, std::pair<float, double>, std::pair<double, double>>;

TYPED_TEST_CASE(MapGenericReduceTest, IoTypePair);
TYPED_TEST(MapGenericReduceTest, min) { this->testMin(); }
TYPED_TEST(MapGenericReduceTest, max) { this->testMax(); }
}  // end namespace linalg
}  // end namespace raft
