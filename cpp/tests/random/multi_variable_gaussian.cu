/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/random/multi_variable_gaussian.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <random>

// mvg.h takes in column-major matrices (as in Fortran)
#define IDX2C(i, j, ld) (j * ld + i)

namespace raft::random {

// helper kernels
/// @todo Duplicate called vctwiseAccumulate in utils.h (Kalman Filters,
// i think that is much better to use., more general)
template <typename T>
RAFT_KERNEL En_KF_accumulate(const int nPoints, const int dim, const T* X, T* x)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int col = idx % dim;
  int row = idx / dim;
  if (col < dim && row < nPoints) raft::myAtomicAdd(x + col, X[idx]);
}

template <typename T>
RAFT_KERNEL En_KF_normalize(const int divider, const int dim, T* x)
{
  int xi = threadIdx.x + blockDim.x * blockIdx.x;
  if (xi < dim) x[xi] = x[xi] / divider;
}

template <typename T>
RAFT_KERNEL En_KF_dif(const int nPoints, const int dim, const T* X, const T* x, T* X_diff)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int col = idx % dim;
  int row = idx / dim;
  if (col < dim && row < nPoints) X_diff[idx] = X[idx] - x[col];
}

// for specialising tests
enum Correlation : unsigned char {
  CORRELATED,  // = 0
  UNCORRELATED
};

template <typename T>
struct MVGInputs {
  T tolerance;
  typename detail::multi_variable_gaussian<T>::Decomposer method;
  Correlation corr;
  int dim, nPoints;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MVGInputs<T>& dims)
{
  return os;
}

template <typename T>
class MVGTest : public ::testing::TestWithParam<MVGInputs<T>> {
 public:
  MVGTest()
    : params(::testing::TestWithParam<MVGInputs<T>>::GetParam()),
      workspace_d(0, resource::get_cuda_stream(handle)),
      P_d(0, resource::get_cuda_stream(handle)),
      x_d(0, resource::get_cuda_stream(handle)),
      X_d(0, resource::get_cuda_stream(handle)),
      Rand_cov(0, resource::get_cuda_stream(handle)),
      Rand_mean(0, resource::get_cuda_stream(handle))
  {
  }

 protected:
  void SetUp() override
  {
    // getting params
    params    = ::testing::TestWithParam<MVGInputs<T>>::GetParam();
    dim       = params.dim;
    nPoints   = params.nPoints;
    method    = params.method;
    corr      = params.corr;
    tolerance = params.tolerance;

    auto cusolverH = resource::get_cusolver_dn_handle(handle);
    auto stream    = resource::get_cuda_stream(handle);

    // preparing to store stuff
    P.resize(dim * dim);
    x.resize(dim);
    X.resize(dim * nPoints);
    P_d.resize(dim * dim, stream);
    X_d.resize(nPoints * dim, stream);
    x_d.resize(dim, stream);
    Rand_cov.resize(dim * dim, stream);
    Rand_mean.resize(dim, stream);

    // generating random mean and cov.
    srand(params.seed);
    for (int j = 0; j < dim; j++)
      x.data()[j] = rand() % 100 + 5.0f;

    // for random Cov. martix
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(0.0, 1.0);

    // P (developing a +ve definite symm matrix)
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < j + 1; i++) {
        T k = distribution(generator);
        if (corr == UNCORRELATED) k = 0.0;
        P.data()[IDX2C(i, j, dim)] = k;
        P.data()[IDX2C(j, i, dim)] = k;
        if (i == j) P.data()[IDX2C(i, j, dim)] += dim;
      }
    }

    // porting inputs to gpu
    raft::update_device(P_d.data(), P.data(), dim * dim, stream);
    raft::update_device(x_d.data(), x.data(), dim, stream);

    // initializing the mvg
    mvg           = new detail::multi_variable_gaussian<T>(handle, dim, method);
    std::size_t o = mvg->get_workspace_size();

    // give the workspace area to mvg
    workspace_d.resize(o, stream);
    mvg->set_workspace(workspace_d.data());

    // get gaussians in X_d | P_d is destroyed.
    mvg->give_gaussian(nPoints, P_d.data(), X_d.data(), x_d.data());

    // saving the mean of the randoms in Rand_mean
    //@todo can be swapped with a API that calculates mean
    RAFT_CUDA_TRY(cudaMemset(Rand_mean.data(), 0, dim * sizeof(T)));
    dim3 block = (64);
    dim3 grid  = (raft::ceildiv(nPoints * dim, (int)block.x));
    raft::launch_kernel(
      handle, grid, block, En_KF_accumulate<T>, nPoints, dim, X_d.data(), Rand_mean.data());
    grid = (raft::ceildiv(dim, (int)block.x));
    raft::launch_kernel(handle, grid, block, En_KF_normalize, nPoints, dim, Rand_mean.data());

    // storing the error wrt random point mean in X_d
    grid = (raft::ceildiv(dim * nPoints, (int)block.x));
    raft::launch_kernel(
      handle, grid, block, En_KF_dif<T>, nPoints, dim, X_d.data(), Rand_mean.data(), X_d.data());

    // finding the cov matrix, placing in Rand_cov
    T alfa = 1.0 / (nPoints - 1), beta = 0.0;

    linalg::gemm(handle,
                 false,
                 true,
                 dim,
                 dim,
                 nPoints,
                 &alfa,
                 X_d.data(),
                 dim,
                 X_d.data(),
                 dim,
                 &beta,
                 Rand_cov.data(),
                 dim,
                 stream);

    // restoring cov provided into P_d
    raft::update_device(P_d.data(), P.data(), dim * dim, stream);
  }

  void TearDown() override
  {
    // deleting mvg
    delete mvg;
  }

 protected:
  raft::resources handle;
  MVGInputs<T> params;
  rmm::device_uvector<T> workspace_d, P_d, x_d, X_d, Rand_cov, Rand_mean;
  std::vector<T> P, x, X;
  int dim, nPoints;
  typename detail::multi_variable_gaussian<T>::Decomposer method;
  Correlation corr;
  detail::multi_variable_gaussian<T>* mvg = NULL;
  T tolerance;
};  // end of MVGTest class

template <typename T>
class MVGMdspanTest : public ::testing::TestWithParam<MVGInputs<T>> {
 private:
  static auto old_enum_to_new_enum(typename detail::multi_variable_gaussian<T>::Decomposer method)
  {
    if (method == detail::multi_variable_gaussian<T>::chol_decomp) {
      return multi_variable_gaussian_decomposition_method::CHOLESKY;
    } else if (method == detail::multi_variable_gaussian<T>::jacobi) {
      return multi_variable_gaussian_decomposition_method::JACOBI;
    } else {
      return multi_variable_gaussian_decomposition_method::QR;
    }
  }

 public:
  MVGMdspanTest()
    : workspace_d(0, resource::get_cuda_stream(handle)),
      P_d(0, resource::get_cuda_stream(handle)),
      x_d(0, resource::get_cuda_stream(handle)),
      X_d(0, resource::get_cuda_stream(handle)),
      Rand_cov(0, resource::get_cuda_stream(handle)),
      Rand_mean(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    params      = ::testing::TestWithParam<MVGInputs<T>>::GetParam();
    dim         = params.dim;
    nPoints     = params.nPoints;
    auto method = old_enum_to_new_enum(params.method);
    corr        = params.corr;
    tolerance   = params.tolerance;

    auto cusolverH = resource::get_cusolver_dn_handle(handle);
    auto stream    = resource::get_cuda_stream(handle);

    P.resize(dim * dim);
    x.resize(dim);
    X.resize(dim * nPoints);
    P_d.resize(dim * dim, stream);
    X_d.resize(nPoints * dim, stream);
    x_d.resize(dim, stream);
    Rand_cov.resize(dim * dim, stream);
    Rand_mean.resize(dim, stream);

    srand(params.seed);
    for (int j = 0; j < dim; j++)
      x.data()[j] = rand() % 100 + 5.0f;

    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(0.0, 1.0);

    // P (symmetric positive definite matrix)
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < j + 1; i++) {
        T k = distribution(generator);
        if (corr == UNCORRELATED) k = 0.0;
        P.data()[IDX2C(i, j, dim)] = k;
        P.data()[IDX2C(j, i, dim)] = k;
        if (i == j) P.data()[IDX2C(i, j, dim)] += dim;
      }
    }

    raft::update_device(P_d.data(), P.data(), dim * dim, stream);
    raft::update_device(x_d.data(), x.data(), dim, stream);

    std::optional<raft::device_vector_view<const T, int>> x_view(std::in_place, x_d.data(), dim);
    raft::device_matrix_view<T, int, raft::col_major> P_view(P_d.data(), dim, dim);
    raft::device_matrix_view<T, int, raft::col_major> X_view(X_d.data(), dim, nPoints);

    raft::random::multi_variable_gaussian(
      handle, rmm::mr::get_current_device_resource_ref(), x_view, P_view, X_view, method);

    // saving the mean of the randoms in Rand_mean
    //@todo can be swapped with a API that calculates mean
    RAFT_CUDA_TRY(cudaMemset(Rand_mean.data(), 0, dim * sizeof(T)));
    dim3 block = (64);
    dim3 grid  = (raft::ceildiv(nPoints * dim, (int)block.x));
    raft::launch_kernel(
      handle, grid, block, En_KF_accumulate<T>, nPoints, dim, X_d.data(), Rand_mean.data());
    grid = (raft::ceildiv(dim, (int)block.x));
    raft::launch_kernel(handle, grid, block, En_KF_normalize, nPoints, dim, Rand_mean.data());

    // storing the error wrt random point mean in X_d
    grid = (raft::ceildiv(dim * nPoints, (int)block.x));
    raft::launch_kernel(
      handle, grid, block, En_KF_dif<T>, nPoints, dim, X_d.data(), Rand_mean.data(), X_d.data());

    // finding the cov matrix, placing in Rand_cov
    T alfa = 1.0 / (nPoints - 1), beta = 0.0;

    linalg::gemm(handle,
                 false,
                 true,
                 dim,
                 dim,
                 nPoints,
                 &alfa,
                 X_d.data(),
                 dim,
                 X_d.data(),
                 dim,
                 &beta,
                 Rand_cov.data(),
                 dim,
                 stream);

    // restoring cov provided into P_d
    raft::update_device(P_d.data(), P.data(), dim * dim, stream);
  }

 protected:
  raft::resources handle;

  MVGInputs<T> params;
  std::vector<T> P, x, X;
  rmm::device_uvector<T> workspace_d, P_d, x_d, X_d, Rand_cov, Rand_mean;
  int dim, nPoints;
  Correlation corr;
  T tolerance;
};  // end of MVGTest class

///@todo find out the reason that Un-correlated covs are giving problems (in qr)
// Declare your inputs
const std::vector<MVGInputs<float>> inputsf = {
  {0.3f,
   detail::multi_variable_gaussian<float>::Decomposer::chol_decomp,
   Correlation::CORRELATED,
   5,
   30000,
   6ULL},
  {0.1f,
   detail::multi_variable_gaussian<float>::Decomposer::chol_decomp,
   Correlation::UNCORRELATED,
   5,
   30000,
   6ULL},
  {0.25f,
   detail::multi_variable_gaussian<float>::Decomposer::jacobi,
   Correlation::CORRELATED,
   5,
   30000,
   6ULL},
  {0.1f,
   detail::multi_variable_gaussian<float>::Decomposer::jacobi,
   Correlation::UNCORRELATED,
   5,
   30000,
   6ULL},
  {0.2f,
   detail::multi_variable_gaussian<float>::Decomposer::qr,
   Correlation::CORRELATED,
   5,
   30000,
   6ULL},
  // { 0.2f,          multi_variable_gaussian<float>::Decomposer::qr,
  // Correlation::UNCORRELATED, 5, 30000, 6ULL}
};
const std::vector<MVGInputs<double>> inputsd = {
  {0.25,
   detail::multi_variable_gaussian<double>::Decomposer::chol_decomp,
   Correlation::CORRELATED,
   10,
   3000000,
   6ULL},
  {0.1,
   detail::multi_variable_gaussian<double>::Decomposer::chol_decomp,
   Correlation::UNCORRELATED,
   10,
   3000000,
   6ULL},
  {0.25,
   detail::multi_variable_gaussian<double>::Decomposer::jacobi,
   Correlation::CORRELATED,
   10,
   3000000,
   6ULL},
  {0.1,
   detail::multi_variable_gaussian<double>::Decomposer::jacobi,
   Correlation::UNCORRELATED,
   10,
   3000000,
   6ULL},
  {0.2,
   detail::multi_variable_gaussian<double>::Decomposer::qr,
   Correlation::CORRELATED,
   10,
   3000000,
   6ULL},
  // { 0.2,          multi_variable_gaussian<double>::Decomposer::qr,
  // Correlation::UNCORRELATED, 10, 3000000, 6ULL}
};

// make the tests
using MVGTestF = MVGTest<float>;
using MVGTestD = MVGTest<double>;
TEST_P(MVGTestF, MeanIsCorrectF)
{
  EXPECT_TRUE(raft::devArrMatch(x_d.data(),
                                Rand_mean.data(),
                                dim,
                                raft::CompareApprox<float>(tolerance),
                                resource::get_cuda_stream(handle)))
    << " in MeanIsCorrect";
}
TEST_P(MVGTestF, CovIsCorrectF)
{
  EXPECT_TRUE(raft::devArrMatch(P_d.data(),
                                Rand_cov.data(),
                                dim,
                                dim,
                                raft::CompareApprox<float>(tolerance),
                                resource::get_cuda_stream(handle)))
    << " in CovIsCorrect";
}
TEST_P(MVGTestD, MeanIsCorrectD)
{
  EXPECT_TRUE(raft::devArrMatch(x_d.data(),
                                Rand_mean.data(),
                                dim,
                                raft::CompareApprox<double>(tolerance),
                                resource::get_cuda_stream(handle)))
    << " in MeanIsCorrect";
}
TEST_P(MVGTestD, CovIsCorrectD)
{
  EXPECT_TRUE(raft::devArrMatch(P_d.data(),
                                Rand_cov.data(),
                                dim,
                                dim,
                                raft::CompareApprox<double>(tolerance),
                                resource::get_cuda_stream(handle)))
    << " in CovIsCorrect";
}

using MVGMdspanTestF = MVGMdspanTest<float>;
using MVGMdspanTestD = MVGMdspanTest<double>;
TEST_P(MVGMdspanTestF, MeanIsCorrectF)
{
  EXPECT_TRUE(raft::devArrMatch(x_d.data(),
                                Rand_mean.data(),
                                dim,
                                raft::CompareApprox<float>(tolerance),
                                resource::get_cuda_stream(handle)))
    << " in MeanIsCorrect";
}
TEST_P(MVGMdspanTestF, CovIsCorrectF)
{
  EXPECT_TRUE(raft::devArrMatch(P_d.data(),
                                Rand_cov.data(),
                                dim,
                                dim,
                                raft::CompareApprox<float>(tolerance),
                                resource::get_cuda_stream(handle)))
    << " in CovIsCorrect";
}
TEST_P(MVGMdspanTestD, MeanIsCorrectD)
{
  EXPECT_TRUE(raft::devArrMatch(x_d.data(),
                                Rand_mean.data(),
                                dim,
                                raft::CompareApprox<double>(tolerance),
                                resource::get_cuda_stream(handle)))
    << " in MeanIsCorrect";
}
TEST_P(MVGMdspanTestD, CovIsCorrectD)
{
  EXPECT_TRUE(raft::devArrMatch(P_d.data(),
                                Rand_cov.data(),
                                dim,
                                dim,
                                raft::CompareApprox<double>(tolerance),
                                resource::get_cuda_stream(handle)))
    << " in CovIsCorrect";
}

// call the tests
INSTANTIATE_TEST_CASE_P(MVGTests, MVGTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(MVGTests, MVGTestD, ::testing::ValuesIn(inputsd));

// call the tests
INSTANTIATE_TEST_CASE_P(MVGMdspanTests, MVGMdspanTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(MVGMdspanTests, MVGMdspanTestD, ::testing::ValuesIn(inputsd));

};  // end of namespace raft::random
