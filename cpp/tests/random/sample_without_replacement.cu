/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <set>
#include <vector>

namespace raft {
namespace random {

using namespace raft::random;

// Terminology:
// SWoR - Sample Without Replacement

template <typename T>
struct SWoRInputs {
  int len, sampledLen;
  int largeWeightIndex;
  T largeWeight;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SWoRInputs<T>& dims)
{
  return os;
}

template <typename T>
class SWoRTest : public ::testing::TestWithParam<SWoRInputs<T>> {
 public:
  SWoRTest()
    : params(::testing::TestWithParam<SWoRInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      in(params.len, stream),
      wts(params.len, stream),
      out(params.sampledLen, stream),
      outIdx(params.sampledLen, stream)
  {
  }

 protected:
  void SetUp() override
  {
    RngState r(params.seed, params.gtype);
    h_outIdx.resize(params.sampledLen);
    uniform(handle, r, in.data(), params.len, T(-1.0), T(1.0));
    uniform(handle, r, wts.data(), params.len, T(1.0), T(2.0));
    if (params.largeWeightIndex >= 0) {
      update_device(wts.data() + params.largeWeightIndex, &params.largeWeight, 1, stream);
    }
    sampleWithoutReplacement(
      handle, r, out.data(), outIdx.data(), in.data(), wts.data(), params.sampledLen, params.len);
    update_host(h_outIdx.data(), outIdx.data(), params.sampledLen, stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SWoRInputs<T> params;
  rmm::device_uvector<T> in, out, wts;
  rmm::device_uvector<int> outIdx;
  std::vector<int> h_outIdx;
};

template <typename T>
class SWoRMdspanTest : public ::testing::TestWithParam<SWoRInputs<T>> {
 public:
  SWoRMdspanTest()
    : params(::testing::TestWithParam<SWoRInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      in(params.len, stream),
      wts(params.len, stream),
      out(params.sampledLen, stream),
      out2(params.sampledLen, stream),
      outIdx(params.sampledLen, stream),
      outIdx2(params.sampledLen, stream)
  {
  }

 protected:
  void SetUp() override
  {
    RngState r(params.seed, params.gtype);
    h_outIdx.resize(params.sampledLen);
    uniform(handle, r, in.data(), params.len, T(-1.0), T(1.0));
    uniform(handle, r, wts.data(), params.len, T(1.0), T(2.0));
    if (params.largeWeightIndex >= 0) {
      update_device(wts.data() + params.largeWeightIndex, &params.largeWeight, 1, stream);
    }
    {
      using index_type  = int;
      using output_view = raft::device_vector_view<T, index_type>;
      output_view out_view{out.data(), out.size()};
      ASSERT_TRUE(out_view.extent(0) == params.sampledLen);

      using output_idxs_view = raft::device_vector_view<index_type, index_type>;
      std::optional<output_idxs_view> outIdx_view{std::in_place, outIdx.data(), outIdx.size()};
      ASSERT_TRUE(outIdx_view.value().extent(0) == params.sampledLen);

      using input_view = raft::device_vector_view<const T, index_type>;
      input_view in_view{in.data(), in.size()};
      ASSERT_TRUE(in_view.extent(0) == params.len);

      using weights_view = raft::device_vector_view<const T, index_type>;
      std::optional<weights_view> wts_view{std::in_place, wts.data(), wts.size()};
      ASSERT_TRUE(wts_view.value().extent(0) == params.len);

      sample_without_replacement(handle, r, in_view, wts_view, out_view, outIdx_view);

      output_view out2_view{out2.data(), out2.size()};
      ASSERT_TRUE(out2_view.extent(0) == params.sampledLen);
      std::optional<output_idxs_view> outIdx2_view{std::in_place, outIdx2.data(), outIdx2.size()};
      ASSERT_TRUE(outIdx2_view.value().extent(0) == params.sampledLen);

      // For now, just test that these calls compile.
      sample_without_replacement(handle, r, in_view, wts_view, out2_view, std::nullopt);
      sample_without_replacement(handle, r, in_view, std::nullopt, out2_view, outIdx2_view);
      sample_without_replacement(handle, r, in_view, std::nullopt, out2_view, std::nullopt);
    }
    update_host(h_outIdx.data(), outIdx.data(), params.sampledLen, stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SWoRInputs<T> params;
  rmm::device_uvector<T> in, out, wts, out2;
  rmm::device_uvector<int> outIdx, outIdx2;
  std::vector<int> h_outIdx;
};

const std::vector<SWoRInputs<float>> inputsf = {{1024, 512, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 1024, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 1024 - 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 1024 - 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 512, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 1024, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 1024 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 1024 - 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 512, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 1024, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 1024 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 1024 + 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 512, 10, 100000.f, GenPhilox, 1234ULL},

                                                {1024, 512, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 1024, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 512 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 1024 - 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 512 + 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 1024 - 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 512, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 1024, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 512 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 1024 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 512 + 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 1024 - 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 512, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 1024, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 512 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 1024 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 512 + 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 1024 + 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 512, 10, 100000.f, GenPC, 1234ULL}};

// This needs to be a macro because it has to live in the scope
// of the class whose name is the first parameter of TEST_P.
//
// We test the following.
//
// 1. Output indices are in the given range.
// 2. Output indices do not repeat.
// 3. If there's a skewed distribution, the top index should
//    correspond to the particular item with a large weight.
#define _RAFT_SWOR_TEST_CONTENTS()                                                                 \
  do {                                                                                             \
    std::set<int> occurrence;                                                                      \
    for (int i = 0; i < params.sampledLen; ++i) {                                                  \
      auto val = h_outIdx[i];                                                                      \
      ASSERT_TRUE(0 <= val && val < params.len)                                                    \
        << "out-of-range index @i=" << i << " val=" << val << " sampledLen=" << params.sampledLen; \
      ASSERT_TRUE(occurrence.find(val) == occurrence.end())                                        \
        << "repeated index @i=" << i << " idx=" << val;                                            \
      occurrence.insert(val);                                                                      \
    }                                                                                              \
    if (params.largeWeightIndex >= 0) {                                                            \
      ASSERT_TRUE((h_outIdx[0] == params.largeWeightIndex) ||                                      \
                  (h_outIdx[1] == params.largeWeightIndex) ||                                      \
                  (h_outIdx[2] == params.largeWeightIndex));                                       \
    }                                                                                              \
  } while (false)

using SWoRTestF = SWoRTest<float>;
TEST_P(SWoRTestF, Result) { _RAFT_SWOR_TEST_CONTENTS(); }
INSTANTIATE_TEST_SUITE_P(SWoRTests, SWoRTestF, ::testing::ValuesIn(inputsf));

using SWoRMdspanTestF = SWoRMdspanTest<float>;
TEST_P(SWoRMdspanTestF, Result) { _RAFT_SWOR_TEST_CONTENTS(); }
INSTANTIATE_TEST_SUITE_P(SWoRTests2, SWoRMdspanTestF, ::testing::ValuesIn(inputsf));

const std::vector<SWoRInputs<double>> inputsd = {{1024, 512, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 1024, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 1024 - 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 1024 - 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 512, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 1024, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 1024 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 1024 - 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 512, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 1024, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 1024 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 1024 + 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 512, 10, 100000.0, GenPhilox, 1234ULL},

                                                 {1024, 512, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 1024, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 512 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 1024 - 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 512 + 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 1024 - 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 512, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 1024, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 512 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 1024 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 512 + 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 1024 - 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 512, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 1024, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 512 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 1024 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 512 + 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 1024 + 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 512, 10, 100000.0, GenPC, 1234ULL}};

using SWoRTestD = SWoRTest<double>;
TEST_P(SWoRTestD, Result) { _RAFT_SWOR_TEST_CONTENTS(); }
INSTANTIATE_TEST_SUITE_P(SWoRTests, SWoRTestD, ::testing::ValuesIn(inputsd));

using SWoRMdspanTestD = SWoRMdspanTest<double>;
TEST_P(SWoRMdspanTestD, Result) { _RAFT_SWOR_TEST_CONTENTS(); }
INSTANTIATE_TEST_SUITE_P(SWoRTests2, SWoRMdspanTestD, ::testing::ValuesIn(inputsd));

}  // namespace random
}  // namespace raft
