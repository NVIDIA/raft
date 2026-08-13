/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <cub/block/block_reduce.cuh>

#include <gtest/gtest.h>
#include <sys/timeb.h>

#include <memory>

namespace raft {
namespace random {

using namespace raft::random;

enum RandomType {
  RNG_Normal,
  RNG_LogNormal,
  RNG_Uniform,
  RNG_Gumbel,
  RNG_Logistic,
  RNG_Exp,
  RNG_Rayleigh,
  RNG_Laplace
};

template <typename T, int TPB>
RAFT_KERNEL meanKernel(T* out, const T* data, int len)
{
  typedef cub::BlockReduce<T, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  T val   = tid < len ? data[tid] : T(0);
  T x     = BlockReduce(temp_storage).Sum(val);
  __syncthreads();
  T xx = BlockReduce(temp_storage).Sum(val * val);
  __syncthreads();
  if (threadIdx.x == 0) {
    raft::myAtomicAdd(out, x);
    raft::myAtomicAdd(out + 1, xx);
  }
}

template <typename T>
struct RngInputs {
  int len;
  // Meaning of 'start' and 'end' parameter for various distributions
  //
  //         Uniform   Normal/Log-Normal   Gumbel   Logistic   Laplace   Exponential   Rayleigh
  // start    start          mean           mean     mean       mean       lambda       sigma
  // end       end           sigma          beta     scale      scale      Unused       Unused
  T start, end;
  RandomType type;
  GeneratorType gtype;
  uint64_t seed;
};

// In this test we generate pseudo-random values that follow various probability distributions such
// as Normal, Laplace etc. To check the correctness of generated random variates we compute two
// measures, mean and variance from the generated data. The computed values are matched against
// their theoretically expected values for the corresponding distribution. The computed mean and
// variance are statistical variables themselves and follow a Normal distribution. Which means,
// there is 99+% chance that the computed values fall in the 3-sigma (standard deviation) interval
// [theoretical_value - 3*sigma, theoretical_value + 3*sigma]. The values are practically
// guaranteed to fall in the 4-sigma interval. Reference standard deviation of the computed
// mean/variance distribution is calculated here
// https://gist.github.com/vinaydes/cee04f50ff7e3365759603d39b7e079b Maximum standard deviation
// observed here is ~1.5e-2, thus we use this as sigma in our test.
// N O T E: Before adding any new test case below, make sure to calculate standard deviation for the
// test parameters using above notebook.

constexpr int NUM_SIGMA    = 4;
constexpr double MAX_SIGMA = 1.5e-2;

template <typename T>
class RngTest : public ::testing::TestWithParam<RngInputs<T>> {
 public:
  RngTest()
    : params(::testing::TestWithParam<RngInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(0, stream),
      stats(2, stream)
  {
    data.resize(params.len, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(stats.data(), 0, 2 * sizeof(T), stream));
  }

 protected:
  void SetUp() override
  {
    RngState r(params.seed, params.gtype);
    switch (params.type) {
      case RNG_Normal: normal(handle, r, data.data(), params.len, params.start, params.end); break;
      case RNG_LogNormal:
        lognormal(handle, r, data.data(), params.len, params.start, params.end);
        break;
      case RNG_Uniform:
        uniform(handle, r, data.data(), params.len, params.start, params.end);
        break;
      case RNG_Gumbel: gumbel(handle, r, data.data(), params.len, params.start, params.end); break;
      case RNG_Logistic:
        logistic(handle, r, data.data(), params.len, params.start, params.end);
        break;
      case RNG_Exp: exponential(handle, r, data.data(), params.len, params.start); break;
      case RNG_Rayleigh: rayleigh(handle, r, data.data(), params.len, params.start); break;
      case RNG_Laplace:
        laplace(handle, r, data.data(), params.len, params.start, params.end);
        break;
    };
    static const int threads = 128;
    raft::launch_kernel(stream,
                        raft::ceildiv(params.len, threads),
                        threads,
                        meanKernel<T, threads>,
                        stats.data(),
                        data.data(),
                        params.len);
    update_host<T>(h_stats, stats.data(), 2, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    h_stats[0] /= params.len;
    h_stats[1] = (h_stats[1] / params.len) - (h_stats[0] * h_stats[0]);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void getExpectedMeanVar(T meanvar[2])
  {
    switch (params.type) {
      case RNG_Normal:
        meanvar[0] = params.start;
        meanvar[1] = params.end * params.end;
        break;
      case RNG_LogNormal: {
        auto var   = params.end * params.end;
        auto mu    = params.start;
        meanvar[0] = raft::exp(mu + var * T(0.5));
        meanvar[1] = (raft::exp(var) - T(1.0)) * raft::exp(T(2.0) * mu + var);
        break;
      }
      case RNG_Uniform:
        meanvar[0] = (params.start + params.end) * T(0.5);
        meanvar[1] = params.end - params.start;
        meanvar[1] = meanvar[1] * meanvar[1] / T(12.0);
        break;
      case RNG_Gumbel: {
        auto gamma = T(0.577215664901532);
        meanvar[0] = params.start + params.end * gamma;
        meanvar[1] = T(3.1415) * T(3.1415) * params.end * params.end / T(6.0);
        break;
      }
      case RNG_Logistic:
        meanvar[0] = params.start;
        meanvar[1] = T(3.1415) * T(3.1415) * params.end * params.end / T(3.0);
        break;
      case RNG_Exp:
        meanvar[0] = T(1.0) / params.start;
        meanvar[1] = meanvar[0] * meanvar[0];
        break;
      case RNG_Rayleigh:
        meanvar[0] = params.start * raft::sqrt(T(3.1415 / 2.0));
        meanvar[1] = ((T(4.0) - T(3.1415)) / T(2.0)) * params.start * params.start;
        break;
      case RNG_Laplace:
        meanvar[0] = params.start;
        meanvar[1] = T(2.0) * params.end * params.end;
        break;
    };
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RngInputs<T> params;
  rmm::device_uvector<T> data, stats;
  T h_stats[2];  // mean, var
};

template <typename T>
class RngMdspanTest : public ::testing::TestWithParam<RngInputs<T>> {
 public:
  RngMdspanTest()
    : params(::testing::TestWithParam<RngInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(0, stream),
      stats(2, stream)
  {
    data.resize(params.len, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(stats.data(), 0, 2 * sizeof(T), stream));
  }

 protected:
  void SetUp() override
  {
    RngState r(params.seed, params.gtype);

    raft::device_vector_view<T> data_view(data.data(), data.size());
    const auto len = data_view.extent(0);

    switch (params.type) {
      case RNG_Normal: normal(handle, r, data_view, params.start, params.end); break;
      case RNG_LogNormal: lognormal(handle, r, data_view, params.start, params.end); break;
      case RNG_Uniform: uniform(handle, r, data_view, params.start, params.end); break;
      case RNG_Gumbel: gumbel(handle, r, data_view, params.start, params.end); break;
      case RNG_Logistic: logistic(handle, r, data_view, params.start, params.end); break;
      case RNG_Exp: exponential(handle, r, data_view, params.start); break;
      case RNG_Rayleigh: rayleigh(handle, r, data_view, params.start); break;
      case RNG_Laplace: laplace(handle, r, data_view, params.start, params.end); break;
    };
    static const int threads = 128;
    raft::launch_kernel(stream,
                        raft::ceildiv(params.len, threads),
                        threads,
                        meanKernel<T, threads>,
                        stats.data(),
                        data.data(),
                        params.len);
    update_host<T>(h_stats, stats.data(), 2, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    h_stats[0] /= params.len;
    h_stats[1] = (h_stats[1] / params.len) - (h_stats[0] * h_stats[0]);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void getExpectedMeanVar(T meanvar[2])
  {
    switch (params.type) {
      case RNG_Normal:
        meanvar[0] = params.start;
        meanvar[1] = params.end * params.end;
        break;
      case RNG_LogNormal: {
        auto var   = params.end * params.end;
        auto mu    = params.start;
        meanvar[0] = raft::exp(mu + var * T(0.5));
        meanvar[1] = (raft::exp(var) - T(1.0)) * raft::exp(T(2.0) * mu + var);
        break;
      }
      case RNG_Uniform:
        meanvar[0] = (params.start + params.end) * T(0.5);
        meanvar[1] = params.end - params.start;
        meanvar[1] = meanvar[1] * meanvar[1] / T(12.0);
        break;
      case RNG_Gumbel: {
        auto gamma = T(0.577215664901532);
        meanvar[0] = params.start + params.end * gamma;
        meanvar[1] = T(3.1415) * T(3.1415) * params.end * params.end / T(6.0);
        break;
      }
      case RNG_Logistic:
        meanvar[0] = params.start;
        meanvar[1] = T(3.1415) * T(3.1415) * params.end * params.end / T(3.0);
        break;
      case RNG_Exp:
        meanvar[0] = T(1.0) / params.start;
        meanvar[1] = meanvar[0] * meanvar[0];
        break;
      case RNG_Rayleigh:
        meanvar[0] = params.start * raft::sqrt(T(3.1415 / 2.0));
        meanvar[1] = ((T(4.0) - T(3.1415)) / T(2.0)) * params.start * params.start;
        break;
      case RNG_Laplace:
        meanvar[0] = params.start;
        meanvar[1] = T(2.0) * params.end * params.end;
        break;
    };
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RngInputs<T> params;
  rmm::device_uvector<T> data, stats;
  T h_stats[2];  // mean, var
};

const std::vector<RngInputs<float>> inputsf = {
  // Test with Philox
  {1024 * 1024, 3.0f, 1.3f, RNG_Normal, GenPhilox, 1234ULL},
  {1024 * 1024, 1.2f, 0.1f, RNG_LogNormal, GenPhilox, 1234ULL},
  {1024 * 1024, 1.2f, 5.5f, RNG_Uniform, GenPhilox, 1234ULL},
  {1024 * 1024, 0.1f, 1.3f, RNG_Gumbel, GenPhilox, 1234ULL},
  {1024 * 1024, 1.6f, 0.0f, RNG_Exp, GenPhilox, 1234ULL},
  {1024 * 1024, 1.6f, 0.0f, RNG_Rayleigh, GenPhilox, 1234ULL},
  {1024 * 1024, 2.6f, 1.3f, RNG_Laplace, GenPhilox, 1234ULL},
  // Test with PCG
  {1024 * 1024, 3.0f, 1.3f, RNG_Normal, GenPC, 1234ULL},
  {1024 * 1024, 1.2f, 0.1f, RNG_LogNormal, GenPC, 1234ULL},
  {1024 * 1024, 1.2f, 5.5f, RNG_Uniform, GenPC, 1234ULL},
  {1024 * 1024, 0.1f, 1.3f, RNG_Gumbel, GenPC, 1234ULL},
  {1024 * 1024, 1.6f, 0.0f, RNG_Exp, GenPC, 1234ULL},
  {1024 * 1024, 1.6f, 0.0f, RNG_Rayleigh, GenPC, 1234ULL},
  {1024 * 1024, 2.6f, 1.3f, RNG_Laplace, GenPC, 1234ULL}};

#define _RAFT_RNG_TEST_BODY(VALUE_TYPE)                                                           \
  do {                                                                                            \
    VALUE_TYPE meanvar[2];                                                                        \
    getExpectedMeanVar(meanvar);                                                                  \
    ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<VALUE_TYPE>(NUM_SIGMA * MAX_SIGMA))); \
    ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<VALUE_TYPE>(NUM_SIGMA * MAX_SIGMA))); \
  } while (false)

using RngTestF = RngTest<float>;
TEST_P(RngTestF, Result) { _RAFT_RNG_TEST_BODY(float); }
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestF, ::testing::ValuesIn(inputsf));

using RngMdspanTestF = RngMdspanTest<float>;
TEST_P(RngMdspanTestF, Result) { _RAFT_RNG_TEST_BODY(float); }
INSTANTIATE_TEST_SUITE_P(RngMdspanTests, RngMdspanTestF, ::testing::ValuesIn(inputsf));

const std::vector<RngInputs<double>> inputsd = {
  // Test with Philox
  {1024 * 1024, 3.0f, 1.3f, RNG_Normal, GenPhilox, 1234ULL},
  {1024 * 1024, 1.2f, 0.1f, RNG_LogNormal, GenPhilox, 1234ULL},
  {1024 * 1024, 1.2f, 5.5f, RNG_Uniform, GenPhilox, 1234ULL},
  {1024 * 1024, 0.1f, 1.3f, RNG_Gumbel, GenPhilox, 1234ULL},
  {1024 * 1024, 1.6f, 0.0f, RNG_Exp, GenPhilox, 1234ULL},
  {1024 * 1024, 1.6f, 0.0f, RNG_Rayleigh, GenPhilox, 1234ULL},
  {1024 * 1024, 2.6f, 1.3f, RNG_Laplace, GenPhilox, 1234ULL},
  // Test with PCG
  {1024 * 1024, 3.0f, 1.3f, RNG_Normal, GenPC, 1234ULL},
  {1024 * 1024, 1.2f, 0.1f, RNG_LogNormal, GenPC, 1234ULL},
  {1024 * 1024, 1.2f, 5.5f, RNG_Uniform, GenPC, 1234ULL},
  {1024 * 1024, 0.1f, 1.3f, RNG_Gumbel, GenPC, 1234ULL},
  {1024 * 1024, 1.6f, 0.0f, RNG_Exp, GenPC, 1234ULL},
  {1024 * 1024, 1.6f, 0.0f, RNG_Rayleigh, GenPC, 1234ULL},
  {1024 * 1024, 2.6f, 1.3f, RNG_Laplace, GenPC, 1234ULL}};

using RngTestD = RngTest<double>;
TEST_P(RngTestD, Result) { _RAFT_RNG_TEST_BODY(double); }
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestD, ::testing::ValuesIn(inputsd));

using RngMdspanTestD = RngMdspanTest<double>;
TEST_P(RngMdspanTestD, Result) { _RAFT_RNG_TEST_BODY(double); }
INSTANTIATE_TEST_SUITE_P(RngMdspanTests, RngMdspanTestD, ::testing::ValuesIn(inputsd));

// ---------------------------------------------------------------------- //
// Test for expected variance in mean calculations

template <typename T>
T quick_mean(const std::vector<T>& d)
{
  T acc = T(0);
  for (const auto& di : d) {
    acc += di;
  }
  return acc / d.size();
}

template <typename T>
T quick_std(const std::vector<T>& d)
{
  T acc    = T(0);
  T d_mean = quick_mean(d);
  for (const auto& di : d) {
    acc += ((di - d_mean) * (di - d_mean));
  }
  return std::sqrt(acc / (d.size() - 1));
}

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& v)
{
  if (!v.empty()) {
    out << '[';
    std::copy(v.begin(), v.end(), std::ostream_iterator<T>(out, ", "));
    out << "\b\b]";
  }
  return out;
}

// The following tests the two random number generators by checking that the measured mean error is
// close to the well-known analytical result(sigma/sqrt(n_samples)). To compute the mean error, we
// a number of experiments computing the mean, giving us a distribution of the mean itself. The
// mean error is simply the standard deviation of this distribution (the standard deviation of the
// mean).
TEST(Rng, MeanError)
{
  timeb time_struct;
  ftime(&time_struct);
  int seed            = time_struct.millitm;
  int num_samples     = 1024;
  int num_experiments = 1024;
  int len             = num_samples * num_experiments;

  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);

  rmm::device_uvector<float> data(len, stream);
  rmm::device_uvector<float> mean_result(num_experiments, stream);
  rmm::device_uvector<float> std_result(num_experiments, stream);

  for (auto rtype : {GenPhilox, GenPC}) {
    RngState r(seed, rtype);
    normal(handle, r, data.data(), len, 3.3f, 0.23f);
    // uniform(r, data, len, -1.0, 2.0);
    raft::stats::mean<false>(mean_result.data(), data.data(), num_samples, num_experiments, stream);
    raft::stats::stddev<false>(std_result.data(),
                               data.data(),
                               mean_result.data(),
                               num_samples,
                               num_experiments,
                               false,
                               stream);
    std::vector<float> h_mean_result(num_experiments);
    std::vector<float> h_std_result(num_experiments);
    update_host(h_mean_result.data(), mean_result.data(), num_experiments, stream);
    update_host(h_std_result.data(), std_result.data(), num_experiments, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    auto d_mean = quick_mean(h_mean_result);

    // std-dev of mean; also known as mean error
    auto d_std_of_mean            = quick_std(h_mean_result);
    auto d_std                    = quick_mean(h_std_result);
    auto d_std_of_mean_analytical = d_std / std::sqrt(num_samples);

    // std::cout << "measured mean error: " << d_std_of_mean << "\n";
    // std::cout << "expected mean error: " << d_std/std::sqrt(num_samples) << "\n";

    auto diff_expected_vs_measured_mean_error =
      std::abs(d_std_of_mean - d_std / std::sqrt(num_samples));

    ASSERT_TRUE((diff_expected_vs_measured_mean_error / d_std_of_mean_analytical < 0.5))
      << "Failed with seed: " << seed << "\nrtype: " << rtype;
  }

  // std::cout << "mean_res:" << h_mean_result << "\n";
}

template <typename T, int len, int scale>
class ScaledBernoulliTest : public ::testing::Test {
 public:
  ScaledBernoulliTest() : stream(resource::get_cuda_stream(handle)), data(len, stream) {}

 protected:
  void SetUp() override
  {
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    RngState r(42);
    scaled_bernoulli(handle, r, data.data(), len, T(0.5), T(scale));
  }

  void rangeCheck()
  {
    auto h_data = std::make_unique<T[]>(len);
    update_host(h_data.get(), data.data(), len, stream);
    ASSERT_TRUE(std::none_of(
      h_data.get(), h_data.get() + len, [](const T& a) { return a < -scale || a > scale; }));
  }

  raft::resources handle;
  cudaStream_t stream;

  rmm::device_uvector<T> data;
};

template <typename T, int len, int scale>
class ScaledBernoulliMdspanTest : public ::testing::Test {
 public:
  ScaledBernoulliMdspanTest() : stream(resource::get_cuda_stream(handle)), data(len, stream) {}

 protected:
  void SetUp() override
  {
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    RngState r(42);

    raft::device_vector_view<T, int> data_view(data.data(), data.size());
    scaled_bernoulli(handle, r, data_view, T(0.5), T(scale));
  }

  void rangeCheck()
  {
    auto h_data = std::make_unique<T[]>(len);
    update_host(h_data.get(), data.data(), len, stream);
    ASSERT_TRUE(std::none_of(
      h_data.get(), h_data.get() + len, [](const T& a) { return a < -scale || a > scale; }));
  }

  raft::resources handle;
  cudaStream_t stream;

  rmm::device_uvector<T> data;
};

using ScaledBernoulliTest1 = ScaledBernoulliTest<float, 500, 35>;
TEST_F(ScaledBernoulliTest1, RangeCheck) { rangeCheck(); }

using ScaledBernoulliMdspanTest1 = ScaledBernoulliMdspanTest<float, 500, 35>;
TEST_F(ScaledBernoulliMdspanTest1, RangeCheck) { rangeCheck(); }

using ScaledBernoulliTest2 = ScaledBernoulliTest<double, 100, 220>;
TEST_F(ScaledBernoulliTest2, RangeCheck) { rangeCheck(); }

using ScaledBernoulliMdspanTest2 = ScaledBernoulliMdspanTest<double, 100, 220>;
TEST_F(ScaledBernoulliMdspanTest2, RangeCheck) { rangeCheck(); }

template <typename T, int len>
class BernoulliTest : public ::testing::Test {
 public:
  BernoulliTest() : stream(resource::get_cuda_stream(handle)), data(len, stream) {}

 protected:
  void SetUp() override
  {
    RngState r(42);
    bernoulli(handle, r, data.data(), len, T(0.5));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void trueFalseCheck()
  {
    // both true and false values must be present
    bool* h_data = new bool[len];
    update_host(h_data, data.data(), len, stream);
    ASSERT_TRUE(std::any_of(h_data, h_data + len, [](bool a) { return a; }));
    ASSERT_TRUE(std::any_of(h_data, h_data + len, [](bool a) { return !a; }));
    delete[] h_data;
  }

  raft::resources handle;
  cudaStream_t stream;

  rmm::device_uvector<bool> data;
};

template <typename T, int len>
class BernoulliMdspanTest : public ::testing::Test {
 public:
  BernoulliMdspanTest() : stream(resource::get_cuda_stream(handle)), data(len, stream) {}

 protected:
  void SetUp() override
  {
    RngState r(42);

    raft::device_vector_view<bool, int> data_view(data.data(), data.size());

    bernoulli(handle, r, data_view, T(0.5));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void trueFalseCheck()
  {
    // both true and false values must be present
    auto h_data = std::make_unique<bool[]>(len);
    update_host(h_data.get(), data.data(), len, stream);
    ASSERT_TRUE(std::any_of(h_data.get(), h_data.get() + len, [](bool a) { return a; }));
    ASSERT_TRUE(std::any_of(h_data.get(), h_data.get() + len, [](bool a) { return !a; }));
  }

  raft::resources handle;
  cudaStream_t stream;

  rmm::device_uvector<bool> data;
};

using BernoulliTest1 = BernoulliTest<float, 1000>;
TEST_F(BernoulliTest1, TrueFalseCheck) { trueFalseCheck(); }

using BernoulliMdspanTest1 = BernoulliMdspanTest<float, 1000>;
TEST_F(BernoulliMdspanTest1, TrueFalseCheck) { trueFalseCheck(); }

using BernoulliTest2 = BernoulliTest<double, 1000>;
TEST_F(BernoulliTest2, TrueFalseCheck) { trueFalseCheck(); }

using BernoulliMdspanTest2 = BernoulliMdspanTest<double, 1000>;
TEST_F(BernoulliMdspanTest2, TrueFalseCheck) { trueFalseCheck(); }

/** Rng::normalTable tests */
template <typename T>
struct RngNormalTableInputs {
  T tolerance;
  int rows, cols;
  T mu, sigma;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RngNormalTableInputs<T>& dims)
{
  return os;
}

template <typename T>
class RngNormalTableTest : public ::testing::TestWithParam<RngNormalTableInputs<T>> {
 public:
  RngNormalTableTest()
    : params(::testing::TestWithParam<RngNormalTableInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      stats(2, stream),
      mu_vec(params.cols, stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(stats.data(), 0, 2 * sizeof(T), stream));
  }

 protected:
  void SetUp() override
  {
    // Tests are configured with their expected test-values sigma. For example,
    // 4 x sigma indicates the test shouldn't fail 99.9% of the time.
    num_sigma = 10;
    int len   = params.rows * params.cols;
    RngState r(params.seed, params.gtype);
    fill(handle, r, mu_vec.data(), params.cols, params.mu);
    T* sigma_vec = nullptr;
    normalTable(
      handle, r, data.data(), params.rows, params.cols, mu_vec.data(), sigma_vec, params.sigma);
    static const int threads = 128;
    raft::launch_kernel(stream,
                        raft::ceildiv(len, threads),
                        threads,
                        meanKernel<T, threads>,
                        stats.data(),
                        data.data(),
                        len);
    update_host<T>(h_stats, stats.data(), 2, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    h_stats[0] /= len;
    h_stats[1] = (h_stats[1] / len) - (h_stats[0] * h_stats[0]);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void getExpectedMeanVar(T meanvar[2])
  {
    meanvar[0] = params.mu;
    meanvar[1] = params.sigma * params.sigma;
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RngNormalTableInputs<T> params;
  rmm::device_uvector<T> data, stats, mu_vec;
  T h_stats[2];  // mean, var
  int num_sigma;
};

template <typename T>
class RngNormalTableMdspanTest : public ::testing::TestWithParam<RngNormalTableInputs<T>> {
 public:
  RngNormalTableMdspanTest()
    : params(::testing::TestWithParam<RngNormalTableInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      stats(2, stream),
      mu_vec(params.cols, stream)
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(stats.data(), 0, 2 * sizeof(T), stream));
  }

 protected:
  void SetUp() override
  {
    // Tests are configured with their expected test-values sigma. For example,
    // 4 x sigma indicates the test shouldn't fail 99.9% of the time.
    num_sigma = 10;
    int len   = params.rows * params.cols;
    RngState r(params.seed, params.gtype);

    raft::device_matrix_view<T, int, raft::row_major> data_view(
      data.data(), params.rows, params.cols);
    raft::device_vector_view<const T, int> mu_vec_view(mu_vec.data(), params.cols);
    raft::device_vector_view<T, int> mu_vec_nc_view(mu_vec.data(), params.cols);
    std::variant<raft::device_vector_view<const T, int>, T> sigma_var(params.sigma);

    fill(handle, r, params.mu, mu_vec_nc_view);
    normalTable(handle, r, mu_vec_view, sigma_var, data_view);
    static const int threads = 128;
    raft::launch_kernel(stream,
                        raft::ceildiv(len, threads),
                        threads,
                        meanKernel<T, threads>,
                        stats.data(),
                        data.data(),
                        len);
    update_host<T>(h_stats, stats.data(), 2, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    h_stats[0] /= len;
    h_stats[1] = (h_stats[1] / len) - (h_stats[0] * h_stats[0]);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void getExpectedMeanVar(T meanvar[2])
  {
    meanvar[0] = params.mu;
    meanvar[1] = params.sigma * params.sigma;
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RngNormalTableInputs<T> params;
  rmm::device_uvector<T> data, stats, mu_vec;
  T h_stats[2];  // mean, var
  int num_sigma;
};

const std::vector<RngNormalTableInputs<float>> inputsf_t = {
  {0.0055, 32, 1024, 1.f, 1.f, GenPhilox, 1234ULL},
  {0.011, 8, 1024, 1.f, 1.f, GenPhilox, 1234ULL},

  {0.0055, 32, 1024, 1.f, 1.f, GenPC, 1234ULL},
  {0.011, 8, 1024, 1.f, 1.f, GenPC, 1234ULL}};

using RngNormalTableTestF = RngNormalTableTest<float>;
TEST_P(RngNormalTableTestF, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(num_sigma * params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(num_sigma * params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngNormalTableTests, RngNormalTableTestF, ::testing::ValuesIn(inputsf_t));

using RngNormalTableMdspanTestF = RngNormalTableMdspanTest<float>;
TEST_P(RngNormalTableMdspanTestF, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(num_sigma * params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(num_sigma * params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngNormalTableMdspanTests,
                         RngNormalTableMdspanTestF,
                         ::testing::ValuesIn(inputsf_t));

const std::vector<RngNormalTableInputs<double>> inputsd_t = {
  {0.0055, 32, 1024, 1.0, 1.0, GenPhilox, 1234ULL},
  {0.011, 8, 1024, 1.0, 1.0, GenPhilox, 1234ULL},

  {0.0055, 32, 1024, 1.0, 1.0, GenPC, 1234ULL},
  {0.011, 8, 1024, 1.0, 1.0, GenPC, 1234ULL}};

using RngNormalTableTestD = RngNormalTableTest<double>;
TEST_P(RngNormalTableTestD, Result)
{
  double meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<double>(num_sigma * params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<double>(num_sigma * params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngNormalTableTests, RngNormalTableTestD, ::testing::ValuesIn(inputsd_t));

using RngNormalTableMdspanTestD = RngNormalTableMdspanTest<double>;
TEST_P(RngNormalTableMdspanTestD, Result)
{
  double meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<double>(num_sigma * params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<double>(num_sigma * params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngNormalTableMdspanTests,
                         RngNormalTableMdspanTestD,
                         ::testing::ValuesIn(inputsd_t));

struct RngAffineInputs {
  int n;
  unsigned long long seed;
};

class RngAffineTest : public ::testing::TestWithParam<RngAffineInputs> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<RngAffineInputs>::GetParam();
    RngState r(params.seed);
    affine_transform_params(r, params.n, a, b);
  }

  void check()
  {
    ASSERT_TRUE(gcd(a, params.n) == 1);
    ASSERT_TRUE(0 <= b && b < params.n);
  }

 private:
  RngAffineInputs params;
  int a, b;
};  // RngAffineTest

const std::vector<RngAffineInputs> inputs_affine = {
  {100, 123456ULL},
  {100, 1234567890ULL},
  {101, 123456ULL},
  {101, 1234567890ULL},
  {7, 123456ULL},
  {7, 1234567890ULL},
  {2568, 123456ULL},
  {2568, 1234567890ULL},
};
TEST_P(RngAffineTest, Result) { check(); }
INSTANTIATE_TEST_SUITE_P(RngAffineTests, RngAffineTest, ::testing::ValuesIn(inputs_affine));

}  // namespace random
}  // namespace raft
