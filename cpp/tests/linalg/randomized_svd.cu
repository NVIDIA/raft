/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_resources.hpp>
#include <raft/linalg/rsvd.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T>
struct randomized_svdInputs {
  T tolerance;
  int n_row;
  int n_col;
  int k;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const randomized_svdInputs<T>& dims)
{
  return os;
}

template <typename T>
class randomized_svdTest : public ::testing::TestWithParam<randomized_svdInputs<T>> {
 public:
  randomized_svdTest()
    : params(::testing::TestWithParam<randomized_svdInputs<T>>::GetParam()),
      stream(handle.get_stream().get()),
      data(params.n_row * params.n_col, stream),
      reconst(params.n_row * params.n_col, stream),
      left_eig_vectors_act(params.n_row * params.k, stream),
      right_eig_vectors_act(params.k * params.n_col, stream),
      sing_vals_act(params.k, stream),
      left_eig_vectors_ref(params.n_row * params.n_col, stream),
      right_eig_vectors_ref(params.n_col * params.n_col, stream),
      sing_vals_ref(params.k, stream)
  {
  }

 protected:
  void basicTest()
  {
    int len = params.n_row * params.n_col;
    ASSERT(params.n_row == 5 && params.n_col == 5, "This test only supports nrows=5 && ncols=5!");
    T data_h[] = {0.76420743, 0.61411544, 0.81724151, 0.42040879, 0.03446089,
                  0.03697287, 0.85962444, 0.67584086, 0.45594666, 0.02074835,
                  0.42018265, 0.39204509, 0.12657948, 0.90250559, 0.23076218,
                  0.50339844, 0.92974961, 0.21213988, 0.63962457, 0.58124562,
                  0.58325673, 0.11589871, 0.39831112, 0.21492685, 0.00540355};
    raft::update_device(data.data(), data_h, len, stream);

    T left_eig_vectors_ref_h[] = {0.42823088,
                                  0.59131151,
                                  0.4220887,
                                  0.50441194,
                                  0.18541506,
                                  0.27047497,
                                  -0.17195579,
                                  0.69362791,
                                  -0.43253894,
                                  -0.47860724};

    T right_eig_vectors_ref_h[] = {0.53005494,
                                   0.44104121,
                                   0.40720732,
                                   0.54337293,
                                   0.25189773,
                                   0.5789401,
                                   0.15264214,
                                   -0.45215699,
                                   -0.53184873,
                                   0.3927082};

    T sing_vals_ref_h[] = {2.36539241, 0.81117785, 0.68562255, 0.41390509, 0.01519322};

    raft::update_device(
      left_eig_vectors_ref.data(), left_eig_vectors_ref_h, params.n_row * params.k, stream);
    raft::update_device(
      right_eig_vectors_ref.data(), right_eig_vectors_ref_h, params.k * params.n_col, stream);
    raft::update_device(sing_vals_ref.data(), sing_vals_ref_h, params.k, stream);

    randomized_svd(handle,
                   raft::make_device_matrix_view<const T, uint32_t, raft::col_major>(
                     data.data(), params.n_row, params.n_col),
                   raft::make_device_vector_view<T, uint32_t>(sing_vals_act.data(), params.k),
                   std::make_optional(raft::make_device_matrix_view<T, uint32_t, raft::col_major>(
                     left_eig_vectors_act.data(), params.n_row, params.k)),
                   std::make_optional(raft::make_device_matrix_view<T, uint32_t, raft::col_major>(
                     right_eig_vectors_act.data(), params.k, params.n_col)),
                   2,
                   2);
    handle.sync_stream(stream);
  }

  void apiTest()
  {
    int len = params.n_row * params.n_col;
    ASSERT(params.n_row == 5 && params.n_col == 5, "This test only supports nrows=5 && ncols=5!");
    T data_h[] = {0.76420743, 0.61411544, 0.81724151, 0.42040879, 0.03446089,
                  0.03697287, 0.85962444, 0.67584086, 0.45594666, 0.02074835,
                  0.42018265, 0.39204509, 0.12657948, 0.90250559, 0.23076218,
                  0.50339844, 0.92974961, 0.21213988, 0.63962457, 0.58124562,
                  0.58325673, 0.11589871, 0.39831112, 0.21492685, 0.00540355};
    raft::update_device(data.data(), data_h, len, stream);

    T left_eig_vectors_ref_h[] = {0.42823088,
                                  0.59131151,
                                  0.4220887,
                                  0.50441194,
                                  0.18541506,
                                  0.27047497,
                                  -0.17195579,
                                  0.69362791,
                                  -0.43253894,
                                  -0.47860724};

    T right_eig_vectors_ref_h[] = {0.53005494,
                                   0.44104121,
                                   0.40720732,
                                   0.54337293,
                                   0.25189773,
                                   0.5789401,
                                   0.15264214,
                                   -0.45215699,
                                   -0.53184873,
                                   0.3927082};

    T sing_vals_ref_h[] = {2.36539241, 0.81117785, 0.68562255, 0.41390509, 0.01519322};

    raft::update_device(
      left_eig_vectors_ref.data(), left_eig_vectors_ref_h, params.n_row * params.k, stream);
    raft::update_device(
      right_eig_vectors_ref.data(), right_eig_vectors_ref_h, params.k * params.n_col, stream);
    raft::update_device(sing_vals_ref.data(), sing_vals_ref_h, params.k, stream);
    randomized_svd(handle,
                   raft::make_device_matrix_view<const T, uint32_t, raft::col_major>(
                     data.data(), params.n_row, params.n_col),
                   raft::make_device_vector_view<T, uint32_t>(sing_vals_act.data(), params.k),
                   std::nullopt,
                   std::make_optional(raft::make_device_matrix_view<T, uint32_t, raft::col_major>(
                     right_eig_vectors_act.data(), params.k, params.n_col)),
                   2,
                   2);
    randomized_svd(handle,
                   raft::make_device_matrix_view<const T, uint32_t, raft::col_major>(
                     data.data(), params.n_row, params.n_col),
                   raft::make_device_vector_view<T, uint32_t>(sing_vals_act.data(), params.k),
                   std::make_optional(raft::make_device_matrix_view<T, uint32_t, raft::col_major>(
                     left_eig_vectors_act.data(), params.n_row, params.k)),
                   std::nullopt,
                   2,
                   2);
    randomized_svd(handle,
                   raft::make_device_matrix_view<const T, uint32_t, raft::col_major>(
                     data.data(), params.n_row, params.n_col),
                   raft::make_device_vector_view<T, uint32_t>(sing_vals_act.data(), params.k),
                   std::nullopt,
                   std::nullopt,
                   2,
                   2);
    handle.sync_stream(stream);
  }

  void SetUp() override
  {
    int major = 0;
    int minor = 0;
    cusolverGetProperty(MAJOR_VERSION, &major);
    cusolverGetProperty(MINOR_VERSION, &minor);
    int cusolv_version = major * 1000 + minor * 10;
    if (cusolv_version >= 11050) apiTest();
    basicTest();
  }

 protected:
  raft::device_resources handle;
  cudaStream_t stream;

  randomized_svdInputs<T> params;
  rmm::device_uvector<T> data, left_eig_vectors_act, right_eig_vectors_act, sing_vals_act,
    left_eig_vectors_ref, right_eig_vectors_ref, sing_vals_ref, reconst;
};

const std::vector<randomized_svdInputs<float>> inputsf1  = {{0.0001f, 5, 5, 2, 1234ULL}};
const std::vector<randomized_svdInputs<double>> inputsd1 = {{0.0001, 5, 5, 2, 1234ULL}};

typedef randomized_svdTest<float> randomized_svdTestF;
TEST_P(randomized_svdTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_act.data(),
                                params.k,
                                raft::CompareApproxAbs<float>(params.tolerance)));
  ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_act.data(),
                                params.n_row * params.k,
                                raft::CompareApproxAbs<float>(params.tolerance)));
  ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                right_eig_vectors_act.data(),
                                params.k * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef randomized_svdTest<double> randomized_svdTestD;
TEST_P(randomized_svdTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_act.data(),
                                params.k,
                                raft::CompareApproxAbs<double>(params.tolerance)));
  ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_act.data(),
                                params.n_row * params.k,
                                raft::CompareApproxAbs<double>(params.tolerance)));
  ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                right_eig_vectors_act.data(),
                                params.k * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(randomized_svdTests1, randomized_svdTestF, ::testing::ValuesIn(inputsf1));
INSTANTIATE_TEST_SUITE_P(randomized_svdTests1, randomized_svdTestD, ::testing::ValuesIn(inputsd1));
}  // end namespace linalg
}  // end namespace raft
