/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map_reduce.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/random/make_regression.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace raft::random {

template <typename T>
struct MakeRegressionInputs {
  T tolerance;
  int n_samples, n_features, n_informative, n_targets, effective_rank;
  T bias;
  bool shuffle;
  raft::random::GeneratorType gtype;
  uint64_t seed;
};

template <typename T>
class MakeRegressionTest : public ::testing::TestWithParam<MakeRegressionInputs<T>> {
 protected:
  void SetUp() override
  {
    // Noise must be zero to compare the actual and expected values
    T noise = (T)0.0, tail_strength = (T)0.5;

    rmm::device_uvector<T> data(params.n_samples * params.n_features, stream);
    rmm::device_uvector<T> values_cm(params.n_samples * params.n_targets, stream);
    rmm::device_uvector<T> coef(params.n_features * params.n_targets, stream);

    // Create the regression problem
    make_regression(handle,
                    data.data(),
                    values_ret.data(),
                    params.n_samples,
                    params.n_features,
                    params.n_informative,
                    stream.get(),
                    coef.data(),
                    params.n_targets,
                    params.bias,
                    params.effective_rank,
                    tail_strength,
                    noise,
                    params.shuffle,
                    params.seed,
                    params.gtype);

    // FIXME (mfh 2022/09/07) This test passes even if it doesn't call
    // make_regression.  Please see
    // https://github.com/rapidsai/raft/issues/814.

    // Calculate the values from the data and coefficients (column-major)
    T alpha = (T)1.0, beta = (T)0.0;
    raft::linalg::gemm(handle,
                       true,
                       true,
                       params.n_samples,
                       params.n_targets,
                       params.n_features,
                       &alpha,
                       data.data(),
                       params.n_features,
                       coef.data(),
                       params.n_targets,
                       &beta,
                       values_cm.data(),
                       params.n_samples,
                       stream.get());

    // Transpose the values to row-major
    raft::linalg::transpose(
      handle, values_cm.data(), values_prod.data(), params.n_samples, params.n_targets, stream.get());

    // Add the bias
    raft::linalg::addScalar(values_prod.data(),
                            values_prod.data(),
                            params.bias,
                            params.n_samples * params.n_targets,
                            stream.get());

    // Count the number of zeroes in the coefficients
    rmm::device_scalar<int> zc_device(stream);
    raft::linalg::mapReduce(zc_device.data(),
                            coef.size(),
                            0,
                            raft::compose_op{raft::cast_op<int>{}, raft::equal_const_op<T>{0}},
                            raft::add_op{},
                            stream.get(),
                            coef.data());
    zero_count = zc_device.value(stream);
  }

 protected:
  MakeRegressionInputs<T> params{::testing::TestWithParam<MakeRegressionInputs<T>>::GetParam()};
  raft::resources handle;
  rmm::cuda_stream_view stream{resource::get_cuda_stream(handle)};
  rmm::device_uvector<T> values_ret{size_t(params.n_samples) * size_t(params.n_targets), stream};
  rmm::device_uvector<T> values_prod{size_t(params.n_samples) * size_t(params.n_targets), stream};

  int zero_count;
};

typedef MakeRegressionTest<float> MakeRegressionTestF;
const std::vector<MakeRegressionInputs<float>> inputsf_t = {
  {0.01f, 256, 32, 16, 1, -1, 0.f, true, raft::random::GenPC, 1234ULL},
  {0.01f, 1000, 100, 47, 4, 65, 4.2f, true, raft::random::GenPC, 1234ULL},
  {0.01f, 20000, 500, 450, 13, -1, -3.f, false, raft::random::GenPC, 1234ULL},
  {0.01f, 256, 32, 16, 1, -1, 0.f, true, raft::random::GenPhilox, 1234ULL},
  {0.01f, 1000, 100, 47, 4, 65, 4.2f, true, raft::random::GenPhilox, 1234ULL},
  {0.01f, 20000, 500, 450, 13, -1, -3.f, false, raft::random::GenPhilox, 1234ULL}};

TEST_P(MakeRegressionTestF, Result)
{
  ASSERT_TRUE(match(params.n_targets * (params.n_features - params.n_informative),
                    zero_count,
                    raft::Compare<int>()));
  ASSERT_TRUE(devArrMatch(values_ret.data(),
                          values_prod.data(),
                          params.n_samples,
                          params.n_targets,
                          raft::CompareApprox<float>(params.tolerance),
                          stream.get()));
}
INSTANTIATE_TEST_CASE_P(MakeRegressionTests, MakeRegressionTestF, ::testing::ValuesIn(inputsf_t));

typedef MakeRegressionTest<double> MakeRegressionTestD;
const std::vector<MakeRegressionInputs<double>> inputsd_t = {
  {0.01, 256, 32, 16, 1, -1, 0.0, true, raft::random::GenPC, 1234ULL},
  {0.01, 1000, 100, 47, 4, 65, 4.2, true, raft::random::GenPC, 1234ULL},
  {0.01, 20000, 500, 450, 13, -1, -3.0, false, raft::random::GenPC, 1234ULL},
  {0.01, 256, 32, 16, 1, -1, 0.0, true, raft::random::GenPhilox, 1234ULL},
  {0.01, 1000, 100, 47, 4, 65, 4.2, true, raft::random::GenPhilox, 1234ULL},
  {0.01, 20000, 500, 450, 13, -1, -3.0, false, raft::random::GenPhilox, 1234ULL}};

TEST_P(MakeRegressionTestD, Result)
{
  ASSERT_TRUE(match(params.n_targets * (params.n_features - params.n_informative),
                    zero_count,
                    raft::Compare<int>()));
  ASSERT_TRUE(devArrMatch(values_ret.data(),
                          values_prod.data(),
                          params.n_samples,
                          params.n_targets,
                          raft::CompareApprox<double>(params.tolerance),
                          stream.get()));
}
INSTANTIATE_TEST_CASE_P(MakeRegressionTests, MakeRegressionTestD, ::testing::ValuesIn(inputsd_t));

template <typename T>
class MakeRegressionMdspanTest : public ::testing::TestWithParam<MakeRegressionInputs<T>> {
 public:
  MakeRegressionMdspanTest() = default;

 protected:
  void SetUp() override
  {
    // Noise must be zero to compare the actual and expected values
    T noise = (T)0.0, tail_strength = (T)0.5;

    rmm::device_uvector<T> data(params.n_samples * params.n_features, stream);
    rmm::device_uvector<T> values_cm(params.n_samples * params.n_targets, stream);
    rmm::device_uvector<T> coef(params.n_features * params.n_targets, stream);

    using index_type = typename rmm::device_uvector<T>::index_type;
    using matrix_view =
      raft::device_matrix_view<T, raft::matrix_extent<index_type>, raft::row_major>;
    matrix_view out_mat(data.data(), params.n_samples, params.n_features);
    matrix_view values_mat(values_ret.data(), params.n_samples, params.n_targets);
    matrix_view coef_mat(coef.data(), params.n_features, params.n_targets);

    // Create the regression problem
    make_regression(handle,
                    out_mat,
                    values_mat,
                    params.n_informative,
                    coef_mat,
                    params.bias,
                    params.effective_rank,
                    tail_strength,
                    noise,
                    params.shuffle,
                    params.seed,
                    params.gtype);

    // FIXME (mfh 2022/09/07) This test passes even if it doesn't call
    // make_regression.  Please see
    // https://github.com/rapidsai/raft/issues/814.

    // Calculate the values from the data and coefficients (column-major)
    T alpha{};
    T beta{};
    raft::linalg::gemm(handle,
                       true,
                       true,
                       params.n_samples,
                       params.n_targets,
                       params.n_features,
                       &alpha,
                       data.data(),
                       params.n_features,
                       coef.data(),
                       params.n_targets,
                       &beta,
                       values_cm.data(),
                       params.n_samples,
                       stream.get());

    // Transpose the values to row-major
    raft::linalg::transpose(
      handle, values_cm.data(), values_prod.data(), params.n_samples, params.n_targets, stream.get());

    // Add the bias
    raft::linalg::addScalar(values_prod.data(),
                            values_prod.data(),
                            params.bias,
                            params.n_samples * params.n_targets,
                            stream.get());

    // Count the number of zeroes in the coefficients
    rmm::device_scalar<int> zc_device(stream);
    raft::linalg::mapReduce(zc_device.data(),
                            coef.size(),
                            0,
                            raft::compose_op{raft::cast_op<int>{}, raft::equal_const_op<T>{0}},
                            raft::add_op{},
                            stream.get(),
                            coef.data());
    zero_count = zc_device.value(stream);
  }

 private:
  MakeRegressionInputs<T> params{::testing::TestWithParam<MakeRegressionInputs<T>>::GetParam()};
  raft::resources handle;
  rmm::cuda_stream_view stream{resource::get_cuda_stream(handle)};
  rmm::device_uvector<T> values_ret{size_t(params.n_samples) * size_t(params.n_targets), stream};
  rmm::device_uvector<T> values_prod{size_t(params.n_samples) * size_t(params.n_targets), stream};

  int zero_count = -1;
};

using MakeRegressionMdspanTestF = MakeRegressionTest<float>;

TEST_P(MakeRegressionMdspanTestF, Result)
{
  ASSERT_TRUE(match(params.n_targets * (params.n_features - params.n_informative),
                    zero_count,
                    raft::Compare<int>()));
  ASSERT_TRUE(devArrMatch(values_ret.data(),
                          values_prod.data(),
                          params.n_samples,
                          params.n_targets,
                          raft::CompareApprox<float>(params.tolerance),
                          resource::get_cuda_stream(handle).get()));
}
INSTANTIATE_TEST_CASE_P(MakeRegressionMdspanTests,
                        MakeRegressionMdspanTestF,
                        ::testing::ValuesIn(inputsf_t));

using MakeRegressionMdspanTestD = MakeRegressionTest<double>;

TEST_P(MakeRegressionMdspanTestD, Result)
{
  ASSERT_TRUE(match(params.n_targets * (params.n_features - params.n_informative),
                    zero_count,
                    raft::Compare<int>()));
  ASSERT_TRUE(devArrMatch(values_ret.data(),
                          values_prod.data(),
                          params.n_samples,
                          params.n_targets,
                          raft::CompareApprox<double>(params.tolerance),
                          resource::get_cuda_stream(handle).get()));
}
INSTANTIATE_TEST_CASE_P(MakeRegressionMdspanTests,
                        MakeRegressionMdspanTestD,
                        ::testing::ValuesIn(inputsd_t));

}  // end namespace raft::random
