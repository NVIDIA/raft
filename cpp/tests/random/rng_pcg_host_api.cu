/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/util/kernel_launch.hpp>

namespace raft {
namespace random {

// CPT - Calls Per Thread, How many calls to custom_next is made by a single thread
// IPC - Items Per Call, How many items are returned by a single call to custom_next (usually IPC =
// 1 or 2)
template <typename DType, typename ParamType, int CPT, int IPC>
__host__ __device__ void single_thread_fill(DType* buffer,
                                            DeviceState<PCGenerator> r,
                                            ParamType params,
                                            const size_t total_threads,
                                            const size_t len,
                                            const size_t tid)
{
  PCGenerator gen(r, tid);

  for (size_t i = 0; i < CPT; i++) {
    DType val[IPC];
    size_t index = (tid * CPT * IPC) + i * IPC;
    custom_next(gen, val, params, index, total_threads);
    for (int j = 0; j < IPC; j++) {
      if (index + j < len) { buffer[index + j] = val[j]; }
    }
  }
}

template <typename DType, typename ParamType, int CPT, int IPC>
RAFT_KERNEL pcg_device_kernel(DType* buffer,
                              DeviceState<PCGenerator> r,
                              ParamType params,
                              const size_t total_threads,
                              const size_t len)
{
  int tid = int(blockIdx.x) * blockDim.x + threadIdx.x;

  single_thread_fill<DType, ParamType, CPT, IPC>(buffer, r, params, total_threads, len, tid);
}

template <typename ParamType, typename DataType, int CPT, int IPC>
class HostApiTest {
 public:
  HostApiTest() : stream(resource::get_cuda_stream(handle)), d_buffer(0, stream)
  {
    len = total_threads * CPT * IPC;
    d_buffer.resize(len, stream);
    h_buffer.resize(len);
  }
  void FillBuffers(uint64_t seed)
  {
    RngState r(seed, GenPC);
    DeviceState<PCGenerator> d_state(r);

    raft::launch_kernel(handle,
                        n_blocks,
                        n_threads,
                        pcg_device_kernel<DataType, ParamType, CPT, IPC>,
                        d_buffer.data(),
                        d_state,
                        dist_params,
                        total_threads,
                        len);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    for (size_t tid = 0; tid < len; tid++) {
      single_thread_fill<DataType, ParamType, CPT, IPC>(
        h_buffer.data(), d_state, dist_params, total_threads, len, tid);
    }
  }
  void SetParams(ParamType _dist_params) { dist_params = _dist_params; }

  void test()
  {
    ASSERT_TRUE(devArrMatchHost(
      h_buffer.data(), d_buffer.data(), len, raft::CompareApprox<double>(1e-5), stream));
  }
  ParamType dist_params;
  raft::resources handle;
  cudaStream_t stream;

  static const int n_blocks         = 128;
  static const int n_threads        = 64;
  static const size_t total_threads = size_t(n_blocks) * n_threads;

  size_t len;
  rmm::device_uvector<DataType> d_buffer;
  std::vector<DataType> h_buffer;
};

// This Wrapper class is needed because gtest typed test allows single type per class
template <typename T>
class TestW : public testing::Test {
 protected:
  void SetUp() override
  {
    test_obj.SetParams(p);
    test_obj.FillBuffers(seed);
  }

 public:
  void TestFillBuffer() { test_obj.test(); }
  T test_obj;
  using ParamType = decltype(T::dist_params);
  static ParamType p;
  const uint64_t seed = 42;
};

TYPED_TEST_SUITE_P(TestW);

TYPED_TEST_P(TestW, host_api_test) { this->TestFillBuffer(); }

REGISTER_TYPED_TEST_SUITE_P(TestW, host_api_test);

using InvariantT = HostApiTest<InvariantDistParams<int>, int, 16, 1>;
template <>
InvariantDistParams<int> TestW<InvariantT>::p = {.const_val = 123456};

using UniformT = HostApiTest<UniformDistParams<double>, double, 16, 1>;
template <>
UniformDistParams<double> TestW<UniformT>::p = {.start = 0.0, .end = 1.0};

using UniformInt32T = HostApiTest<UniformIntDistParams<uint32_t, uint32_t>, uint32_t, 16, 1>;
template <>
UniformIntDistParams<uint32_t, uint32_t> TestW<UniformInt32T>::p = {
  .start = 0, .end = 100000, .diff = 100000};

using UniformInt64T = HostApiTest<UniformIntDistParams<uint64_t, uint64_t>, uint64_t, 16, 1>;
template <>
UniformIntDistParams<uint64_t, uint64_t> TestW<UniformInt64T>::p = {
  .start = 0, .end = 100000, .diff = 100000};

using NormalT = HostApiTest<NormalDistParams<double>, double, 16, 2>;
template <>
NormalDistParams<double> TestW<NormalT>::p = {.mu = 0.5, .sigma = 0.5};

using NormalIntT = HostApiTest<NormalIntDistParams<uint32_t>, uint32_t, 16, 2>;
template <>
NormalIntDistParams<uint32_t> TestW<NormalIntT>::p = {.mu = 10000000, .sigma = 10000};

using BernoulliT = HostApiTest<BernoulliDistParams<double>, double, 16, 1>;
template <>
BernoulliDistParams<double> TestW<BernoulliT>::p = {.prob = 0.7};

using ScaledBernoulliT = HostApiTest<ScaledBernoulliDistParams<double>, double, 16, 1>;
template <>
ScaledBernoulliDistParams<double> TestW<ScaledBernoulliT>::p = {.prob = 0.7, .scale = 0.5};

using GumbelT = HostApiTest<GumbelDistParams<double>, double, 16, 1>;
template <>
GumbelDistParams<double> TestW<GumbelT>::p = {.mu = 0.7, .beta = 0.5};

using LogNormalT = HostApiTest<LogNormalDistParams<double>, double, 16, 2>;
template <>
LogNormalDistParams<double> TestW<LogNormalT>::p = {.mu = 0.5, .sigma = 0.5};

using LogisticT = HostApiTest<LogisticDistParams<double>, double, 16, 1>;
template <>
LogisticDistParams<double> TestW<LogisticT>::p = {.mu = 0.2, .scale = 0.3};

using ExponentialT = HostApiTest<ExponentialDistParams<double>, double, 16, 1>;
template <>
ExponentialDistParams<double> TestW<ExponentialT>::p = {.lambda = 1.6};

using RayleighT = HostApiTest<RayleighDistParams<double>, double, 16, 1>;
template <>
RayleighDistParams<double> TestW<RayleighT>::p = {.sigma = 1.6};

using LaplaceT = HostApiTest<LaplaceDistParams<double>, double, 16, 1>;
template <>
LaplaceDistParams<double> TestW<LaplaceT>::p = {.mu = 0.2, .scale = 0.3};

using TypeList = testing::Types<InvariantT,
                                UniformT,
                                UniformInt32T,
                                UniformInt64T,
                                NormalT,
                                NormalIntT,
                                BernoulliT,
                                ScaledBernoulliT,
                                GumbelT,
                                LogisticT,
                                ExponentialT,
                                RayleighT,
                                LaplaceT>;

INSTANTIATE_TYPED_TEST_SUITE_P(Rng, TestW, TypeList);

}  // namespace random
}  // namespace raft
