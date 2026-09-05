/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cub/device/device_histogram.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace raft {
namespace random {

/* In this test we generate pseudo-random integers following a probability distribution defined by
 * an array of weights, such that the probability of the integer i is p_i=w_i/sum(w). A histogram of
 * the generated integers is compared to the expected probabilities. The histogram is normalized,
 * i.e divided by the number of drawn integers n=sampled_len*n_repeat. The expected value for the
 * index i of the histogram is E_i=p_i, the standard deviation sigma_i=sqrt(p_i*(1-p_i)/n).
 *
 * Weights are constructed as a sparse vector containing mostly zeros and a small number of non-zero
 * values. The test tolerance used to compare the actual and expected histograms is
 * eps=max(sigma_i). For the test to be relevant, the tolerance must be small w.r.t the non-zero
 * probabilities. Hence, n_repeat, sampled_len and nnz must be chosen accordingly. The test
 * automatically computes the tolerance and will fail if it is estimated too high for the test to be
 * relevant.
 */

template <typename IdxT>
struct RngDiscreteInputs {
  IdxT n_repeat;
  IdxT sampled_len;
  IdxT len;
  IdxT nnz;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename WeightT, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const RngDiscreteInputs<IdxT>& d)
{
  return os << "{" << d.n_repeat << ", " << d.sampled_len << ", " << d.len << ", " << d.nnz << "}";
}

template <typename LabelT, typename IdxT>
void update_count(
  const LabelT* labels, IdxT* count, IdxT sampled_len, IdxT len, const cudaStream_t& stream)
{
  IdxT num_levels  = len + 1;
  IdxT lower_level = 0;
  IdxT upper_level = len;

  rmm::device_uvector<IdxT> temp_count(len, stream);

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    temp_count.data(),
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    sampled_len,
                                                    stream));

  rmm::device_uvector<char> workspace(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    temp_count.data(),
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    sampled_len,
                                                    stream));

  raft::linalg::add(count, count, temp_count.data(), len, stream);
}

template <typename IdxT>
void normalize_count(
  float* histogram, const IdxT* count, float scale, IdxT len, const cudaStream_t& stream)
{
  raft::linalg::unaryOp(
    histogram,
    count,
    len,
    [scale] __device__(const IdxT& cnt) { return static_cast<float>(cnt) / scale; },
    stream);
}

template <typename OutT, typename WeightT, typename IdxT>
class RngDiscreteTest : public ::testing::TestWithParam<RngDiscreteInputs<IdxT>> {
 public:
  RngDiscreteTest()
    : params(::testing::TestWithParam<RngDiscreteInputs<IdxT>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      out(params.sampled_len, stream),
      weights(params.len, stream),
      histogram(params.len, stream),
      exp_histogram(params.len)
  {
  }

 protected:
  void SetUp() override
  {
    tolerance = 0.0f;
    std::vector<WeightT> h_weights(params.len, WeightT{0});
    std::mt19937 gen(params.seed);
    std::uniform_real_distribution dis(WeightT{0.2}, WeightT{2.0});
    WeightT total_weight = WeightT{0};
    for (int i = 0; i < params.nnz; i++) {
      h_weights[i] = dis(gen);
      total_weight += h_weights[i];
    }
    float min_p = 1.f;
    for (int i = 0; i < params.nnz; i++) {
      float p     = static_cast<float>(h_weights[i] / total_weight);
      float n     = static_cast<float>(params.n_repeat * params.sampled_len);
      float sigma = std::sqrt(p * (1.f - p) / n);
      tolerance   = std::max(tolerance, 4.f * sigma);
      min_p       = std::min(min_p, p);
    }
    EXPECT_TRUE(tolerance < 0.5f * min_p) << "Test tolerance (" << tolerance
                                          << ") is too high. Use more samples, more "
                                             "repetitions or less non-zero weights.";
    std::shuffle(h_weights.begin(), h_weights.end(), gen);
    raft::copy(weights.data(), h_weights.data(), params.len, stream);

    RngState r(params.seed, params.gtype);
    raft::device_vector_view<OutT, IdxT> out_view(out.data(), out.size());
    auto weights_view =
      raft::make_device_vector_view<const WeightT, IdxT>(weights.data(), weights.size());

    rmm::device_uvector<IdxT> count(params.len, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(count.data(), 0, params.len * sizeof(IdxT), stream));
    for (int iter = 0; iter < params.n_repeat; iter++) {
      discrete(handle, r, out_view, weights_view);
      update_count(out.data(), count.data(), params.sampled_len, params.len, stream);
    }
    float scale = static_cast<float>(params.sampled_len * params.n_repeat);
    normalize_count(histogram.data(), count.data(), scale, params.len, stream);

    // Compute the expected normalized histogram
    for (IdxT i = 0; i < params.len; i++) {
      exp_histogram[i] = h_weights[i] / total_weight;
    }
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RngDiscreteInputs<IdxT> params;
  float tolerance;
  rmm::device_uvector<OutT> out;
  rmm::device_uvector<WeightT> weights;
  rmm::device_uvector<float> histogram;
  std::vector<float> exp_histogram;
};

const std::vector<RngDiscreteInputs<int>> inputs_i32 = {
  {1, 10000, 5, 5, GenPC, 123ULL},
  {1, 10000, 10, 7, GenPC, 456ULL},
  {1000, 100, 10000, 20, GenPC, 123ULL},
  {1, 10000, 5, 5, GenPhilox, 1234ULL},
};
const std::vector<RngDiscreteInputs<int64_t>> inputs_i64 = {
  {1, 10000, 5, 5, GenPC, 123ULL},
  {1, 10000, 10, 7, GenPC, 456ULL},
  {1000, 100, 10000, 20, GenPC, 123ULL},
  {1, 10000, 5, 5, GenPhilox, 1234ULL},
};

#define RNG_DISCRETE_TEST(test_type, test_name, test_inputs)     \
  typedef RAFT_DEPAREN(test_type) test_name;                     \
  TEST_P(test_name, Result)                                      \
  {                                                              \
    ASSERT_TRUE(devArrMatchHost(exp_histogram.data(),            \
                                histogram.data(),                \
                                exp_histogram.size(),            \
                                CompareApprox<float>(tolerance), \
                                stream));                        \
  }                                                              \
  INSTANTIATE_TEST_CASE_P(ReduceTests, test_name, ::testing::ValuesIn(test_inputs))

RNG_DISCRETE_TEST((RngDiscreteTest<int, float, int>), RngDiscreteTestI32FI32, inputs_i32);
RNG_DISCRETE_TEST((RngDiscreteTest<uint32_t, float, int>), RngDiscreteTestU32FI32, inputs_i32);
RNG_DISCRETE_TEST((RngDiscreteTest<int64_t, float, int>), RngDiscreteTestI64FI32, inputs_i32);
RNG_DISCRETE_TEST((RngDiscreteTest<int, double, int>), RngDiscreteTestI32DI32, inputs_i32);

// Disable IdxT=int64_t test due to CUB error: https://github.com/NVIDIA/cub/issues/192
// RNG_DISCRETE_TEST((RngDiscreteTest<int, float, int64_t>), RngDiscreteTestI32FI64, inputs_i64);

}  // namespace random
}  // namespace raft
