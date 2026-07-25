/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/pca.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <vector>

namespace raft::linalg {

template <typename T>
struct PcaInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  int len2;
  int n_row2;
  int n_col2;
  unsigned long long int seed;
  int algo;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const PcaInputs<T>& dims)
{
  return os;
}

template <typename T>
class PcaTest : public ::testing::TestWithParam<PcaInputs<T>> {
 public:
  PcaTest()
    : params(::testing::TestWithParam<PcaInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      explained_vars(params.n_col, stream),
      explained_vars_ref(params.n_col, stream),
      components(params.n_col * params.n_col, stream),
      components_ref(params.n_col * params.n_col, stream),
      trans_data(params.len, stream),
      trans_data_ref(params.len, stream),
      data(params.len, stream),
      data_back(params.len, stream)
  {
    basicTest();
  }

 protected:
  void basicTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.len;

    std::vector<T> data_h = {1.0, 2.0, 5.0, 4.0, 2.0, 1.0};
    data_h.resize(len);
    raft::update_device(data.data(), data_h.data(), len, stream);

    std::vector<T> trans_data_ref_h = {-2.3231, -0.3517, 2.6748, 0.3979, -0.6571, 0.2592};
    trans_data_ref_h.resize(len);
    raft::update_device(trans_data_ref.data(), trans_data_ref_h.data(), len, stream);

    int n_components = params.n_col;
    int len_comp     = params.n_col * params.n_col;
    rmm::device_uvector<T> explained_var_ratio(params.n_col, stream);
    rmm::device_uvector<T> singular_vals(params.n_col, stream);
    rmm::device_uvector<T> mean(params.n_col, stream);
    rmm::device_uvector<T> noise_vars(1, stream);

    std::vector<T> components_ref_h = {0.8163, 0.5776, -0.5776, 0.8163};
    components_ref_h.resize(len_comp);
    std::vector<T> explained_vars_ref_h = {6.338, 0.3287};
    explained_vars_ref_h.resize(params.n_col);

    raft::update_device(components_ref.data(), components_ref_h.data(), len_comp, stream);
    raft::update_device(
      explained_vars_ref.data(), explained_vars_ref_h.data(), params.n_col, stream);

    std::size_t n_rows = params.n_row;
    std::size_t n_cols = params.n_col;

    paramsPCA prms;
    prms.whiten = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    auto input_view =
      raft::make_device_matrix_view<T, std::size_t, raft::col_major>(data.data(), n_rows, n_cols);
    auto components_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      components.data(), n_components, n_cols);
    auto explained_var_view =
      raft::make_device_vector_view<T, std::size_t>(explained_vars.data(), n_components);
    auto explained_var_ratio_view =
      raft::make_device_vector_view<T, std::size_t>(explained_var_ratio.data(), n_components);
    auto singular_vals_view =
      raft::make_device_vector_view<T, std::size_t>(singular_vals.data(), n_components);
    auto mu_view         = raft::make_device_vector_view<T, std::size_t>(mean.data(), n_cols);
    auto noise_vars_view = raft::make_device_scalar_view<T, std::size_t>(noise_vars.data());

    pca_fit(handle,
            prms,
            input_view,
            components_view,
            explained_var_view,
            explained_var_ratio_view,
            singular_vals_view,
            mu_view,
            noise_vars_view);

    auto trans_data_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      trans_data.data(), n_rows, n_components);

    pca_transform(
      handle, prms, input_view, components_view, singular_vals_view, mu_view, trans_data_view);

    auto data_back_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      data_back.data(), n_rows, n_cols);

    pca_inverse_transform(
      handle, prms, trans_data_view, components_view, singular_vals_view, mu_view, data_back_view);
  }

  template <typename LayoutPolicy>
  void advancedTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.len2;

    std::size_t n_rows       = params.n_row2;
    std::size_t n_cols       = params.n_col2;
    std::size_t n_components = params.n_col2;

    paramsPCA prms;
    prms.whiten = false;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else if (params.algo == 1)
      prms.algorithm = solver::COV_EIG_JACOBI;

    rmm::device_uvector<T> data2(len, stream);
    rmm::device_uvector<T> data2_back(len, stream);
    r.uniform(data2.data(), len, T(-1.0), T(1.0), stream);
    rmm::device_uvector<T> data2_trans(n_rows * n_components, stream);

    int len_comp = params.n_col2 * n_components;
    rmm::device_uvector<T> components2(len_comp, stream);
    rmm::device_uvector<T> explained_vars2(n_components, stream);
    rmm::device_uvector<T> explained_var_ratio2(n_components, stream);
    rmm::device_uvector<T> singular_vals2(n_components, stream);
    rmm::device_uvector<T> mean2(n_cols, stream);
    rmm::device_uvector<T> noise_vars2(1, stream);

    auto input_view =
      raft::make_device_matrix_view<T, std::size_t, LayoutPolicy>(data2.data(), n_rows, n_cols);
    auto trans_view = raft::make_device_matrix_view<T, std::size_t, LayoutPolicy>(
      data2_trans.data(), n_rows, n_components);
    auto comp_view = raft::make_device_matrix_view<T, std::size_t, LayoutPolicy>(
      components2.data(), n_components, n_cols);
    auto ev_view =
      raft::make_device_vector_view<T, std::size_t>(explained_vars2.data(), n_components);
    auto evr_view =
      raft::make_device_vector_view<T, std::size_t>(explained_var_ratio2.data(), n_components);
    auto sv_view =
      raft::make_device_vector_view<T, std::size_t>(singular_vals2.data(), n_components);
    auto mu_view    = raft::make_device_vector_view<T, std::size_t>(mean2.data(), n_cols);
    auto noise_view = raft::make_device_scalar_view<T, std::size_t>(noise_vars2.data());

    pca_fit_transform(handle,
                      prms,
                      input_view,
                      trans_view,
                      comp_view,
                      ev_view,
                      evr_view,
                      sv_view,
                      mu_view,
                      noise_view);

    auto data2_back_view = raft::make_device_matrix_view<T, std::size_t, LayoutPolicy>(
      data2_back.data(), n_rows, n_cols);

    pca_inverse_transform(handle, prms, trans_view, comp_view, sv_view, mu_view, data2_back_view);

    ASSERT_TRUE(devArrMatch(
      data2.data(), data2_back.data(), len, raft::CompareApprox<T>(params.tolerance), stream));
  }

 protected:
  raft::device_resources handle;
  cudaStream_t stream;

  PcaInputs<T> params;

  rmm::device_uvector<T> explained_vars, explained_vars_ref, components, components_ref, trans_data,
    trans_data_ref, data, data_back;
};

const std::vector<PcaInputs<float>> inputsf2 = {
  {0.01f, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01f, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};

const std::vector<PcaInputs<double>> inputsd2 = {
  {0.01, 3 * 2, 3, 2, 1024 * 128, 1024, 128, 1234ULL, 0},
  {0.01, 3 * 2, 3, 2, 256 * 32, 256, 32, 1234ULL, 1}};

typedef PcaTest<float> PcaTestValF;
TEST_P(PcaTestValF, Result)
{
  ASSERT_TRUE(devArrMatch(explained_vars.data(),
                          explained_vars_ref.data(),
                          params.n_col,
                          raft::CompareApprox<float>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef PcaTest<double> PcaTestValD;
TEST_P(PcaTestValD, Result)
{
  ASSERT_TRUE(devArrMatch(explained_vars.data(),
                          explained_vars_ref.data(),
                          params.n_col,
                          raft::CompareApprox<double>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef PcaTest<float> PcaTestLeftVecF;
TEST_P(PcaTestLeftVecF, Result)
{
  ASSERT_TRUE(devArrMatch(components.data(),
                          components_ref.data(),
                          (params.n_col * params.n_col),
                          raft::CompareApprox<float>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef PcaTest<double> PcaTestLeftVecD;
TEST_P(PcaTestLeftVecD, Result)
{
  ASSERT_TRUE(devArrMatch(components.data(),
                          components_ref.data(),
                          (params.n_col * params.n_col),
                          raft::CompareApprox<double>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef PcaTest<float> PcaTestTransDataF;
TEST_P(PcaTestTransDataF, Result)
{
  ASSERT_TRUE(devArrMatch(trans_data.data(),
                          trans_data_ref.data(),
                          (params.n_row * params.n_col),
                          raft::CompareApprox<float>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef PcaTest<double> PcaTestTransDataD;
TEST_P(PcaTestTransDataD, Result)
{
  ASSERT_TRUE(devArrMatch(trans_data.data(),
                          trans_data_ref.data(),
                          (params.n_row * params.n_col),
                          raft::CompareApprox<double>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef PcaTest<float> PcaTestDataVecSmallF;
TEST_P(PcaTestDataVecSmallF, Result)
{
  ASSERT_TRUE(devArrMatch(data.data(),
                          data_back.data(),
                          (params.n_col * params.n_col),
                          raft::CompareApprox<float>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef PcaTest<double> PcaTestDataVecSmallD;
TEST_P(PcaTestDataVecSmallD, Result)
{
  ASSERT_TRUE(devArrMatch(data.data(),
                          data_back.data(),
                          (params.n_col * params.n_col),
                          raft::CompareApprox<double>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef PcaTest<float> PcaTestDataVecF;
TEST_P(PcaTestDataVecF, Result)
{
  this->template advancedTest<raft::row_major>();
  this->template advancedTest<raft::col_major>();
}

typedef PcaTest<double> PcaTestDataVecD;
TEST_P(PcaTestDataVecD, Result)
{
  this->template advancedTest<raft::row_major>();
  this->template advancedTest<raft::col_major>();
}

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestLeftVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestLeftVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecSmallF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecSmallD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestTransDataF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestTransDataD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(PcaTests, PcaTestDataVecD, ::testing::ValuesIn(inputsd2));

}  // end namespace raft::linalg
