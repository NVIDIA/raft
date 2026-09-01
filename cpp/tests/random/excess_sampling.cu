/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <unordered_set>
#include <vector>

namespace raft {
namespace random {

using namespace raft::random;

struct inputs {
  int64_t N;
  int64_t n_samples;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const inputs p)
{
  os << p.N << "/" << p.n_samples;
  return os;
}

template <typename T>
class ExcessSamplingTest : public ::testing::TestWithParam<inputs> {
 public:
  ExcessSamplingTest()
    : params(::testing::TestWithParam<inputs>::GetParam()),
      stream(resource::get_cuda_stream(res).get()),
      state{137ULL}
  {
  }

  void check()
  {
    device_vector<T, int64_t> out =
      raft::random::excess_subsample<T, int64_t>(res, state, params.N, params.n_samples);
    ASSERT_TRUE(out.extent(0) == params.n_samples);

    auto h_out = make_host_vector<T, int64_t>(res, params.n_samples);
    raft::copy(h_out.data_handle(), out.data_handle(), out.size(), stream);
    resource::sync_stream(res, stream);

    std::unordered_set<int> occurrence;
    int64_t sum = 0;
    for (int64_t i = 0; i < params.n_samples; ++i) {
      T val = h_out(i);
      sum += val;
      ASSERT_TRUE(0 <= val && val < params.N)
        << "out-of-range index @i=" << i << " val=" << val << " n_samples=" << params.n_samples;
      ASSERT_TRUE(occurrence.find(val) == occurrence.end())
        << "repeated index @i=" << i << " idx=" << val;
      occurrence.insert(val);
    }
    float avg = sum / (float)params.n_samples;
    if (params.n_samples >= 100 && params.N / params.n_samples < 100) {
      ASSERT_TRUE(raft::match(avg, (params.N - 1) / 2.0f, raft::CompareApprox<float>(0.2)))
        << "non-uniform sample";
    }
  }

 protected:
  inputs params;
  raft::resources res;
  cudaStream_t stream;
  RngState state;
};

const std::vector<inputs> input1 = {{1, 0},
                                    {1, 1},
                                    {10, 0},
                                    {10, 1},
                                    {10, 2},
                                    {10, 10},
                                    {137, 42},
                                    {200, 0},
                                    {200, 1},
                                    {200, 100},
                                    {200, 130},
                                    {200, 200},
                                    {10000, 893},
                                    {10000000000, 1023}};

using ExcessSamplingTestInt64 = ExcessSamplingTest<int64_t>;
TEST_P(ExcessSamplingTestInt64, SamplingTest) { check(); }
INSTANTIATE_TEST_SUITE_P(ExcessSamplingTests, ExcessSamplingTestInt64, ::testing::ValuesIn(input1));

}  // namespace random
}  // namespace raft
