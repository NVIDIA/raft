/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T>
struct GemmLayoutInputs {
  int M;
  int N;
  int K;
  bool zLayout;
  bool xLayout;
  bool yLayout;
  unsigned long long int seed;
};

// Reference GEMM implementation.
template <typename T>
RAFT_KERNEL naiveGemm(
  T* Z, T* X, T* Y, int M, int N, int K, bool isZColMajor, bool isXColMajor, bool isYColMajor)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

  for (int m = tidy; m < M; m += (blockDim.y * gridDim.y)) {
    for (int n = tidx; n < N; n += (blockDim.x * gridDim.x)) {
      T temp = T(0.0);
      for (int k = 0; k < K; k++) {
        int xIndex = isXColMajor ? m + k * M : m * K + k;
        int yIndex = isYColMajor ? k + n * K : k * N + n;
        temp += X[xIndex] * Y[yIndex];
      }
      int zIndex = isZColMajor ? m + n * M : m * N + n;
      Z[zIndex]  = temp;
    }
  }
}

template <typename T>
class GemmLayoutTest : public ::testing::TestWithParam<GemmLayoutInputs<T>> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<GemmLayoutInputs<T>>::GetParam();

    raft::resources handle;
    cudaStream_t stream = resource::get_cuda_stream(handle);

    raft::random::RngState r(params.seed);

    // We compute Z = X * Y and compare against reference result
    // Dimensions of X : M x K
    // Dimensions of Y : K x N
    // Dimensions of Z : M x N

    T* X = NULL;  // Argument X
    T* Y = NULL;  // Argument Y

    size_t xElems = params.M * params.K;
    size_t yElems = params.K * params.N;
    size_t zElems = params.M * params.N;

    RAFT_CUDA_TRY(cudaMalloc(&X, xElems * sizeof(T)));
    RAFT_CUDA_TRY(cudaMalloc(&Y, yElems * sizeof(T)));
    RAFT_CUDA_TRY(cudaMalloc(&refZ, zElems * sizeof(T)));
    RAFT_CUDA_TRY(cudaMalloc(&Z, zElems * sizeof(T)));

    uniform(handle, r, X, xElems, T(-10.0), T(10.0));
    uniform(handle, r, Y, yElems, T(-10.0), T(10.0));

    dim3 blocks(raft::ceildiv<int>(params.M, 128), raft::ceildiv<int>(params.N, 4), 1);
    dim3 threads(128, 4, 1);

    naiveGemm<<<blocks, threads>>>(
      refZ, X, Y, params.M, params.N, params.K, params.zLayout, params.xLayout, params.yLayout);

    auto x_view_row_major = raft::make_device_matrix_view(X, params.M, params.K);
    auto y_view_row_major = raft::make_device_matrix_view(Y, params.K, params.N);
    auto z_view_row_major = raft::make_device_matrix_view(Z, params.M, params.N);

    auto x_view_col_major =
      raft::make_device_matrix_view<T, int, raft::col_major>(X, params.M, params.K);
    auto y_view_col_major =
      raft::make_device_matrix_view<T, int, raft::col_major>(Y, params.K, params.N);
    auto z_view_col_major =
      raft::make_device_matrix_view<T, int, raft::col_major>(Z, params.M, params.N);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.xLayout && params.yLayout && params.zLayout) {
          gemm(h, x_view_col_major, y_view_col_major, z_view_col_major);
        } else if (params.xLayout && params.yLayout && !params.zLayout) {
          gemm(h, x_view_col_major, y_view_col_major, z_view_row_major);
        } else if (params.xLayout && !params.yLayout && params.zLayout) {
          gemm(h, x_view_col_major, y_view_row_major, z_view_col_major);
        } else if (!params.xLayout && params.yLayout && params.zLayout) {
          gemm(h, x_view_row_major, y_view_col_major, z_view_col_major);
        } else if (params.xLayout && !params.yLayout && !params.zLayout) {
          gemm(h, x_view_col_major, y_view_row_major, z_view_row_major);
        } else if (!params.xLayout && params.yLayout && !params.zLayout) {
          gemm(h, x_view_row_major, y_view_col_major, z_view_row_major);
        } else if (!params.xLayout && !params.yLayout && params.zLayout) {
          gemm(h, x_view_row_major, y_view_row_major, z_view_col_major);
        } else if (!params.xLayout && !params.yLayout && !params.zLayout) {
          gemm(h, x_view_row_major, y_view_row_major, z_view_row_major);
        }
      },
      raft::alloc_behavior::NO_ALLOCATIONS);

    resource::sync_stream(handle);

    RAFT_CUDA_TRY(cudaFree(X));
    RAFT_CUDA_TRY(cudaFree(Y));
  }

  void TearDown() override
  {
    RAFT_CUDA_TRY(cudaFree(refZ));
    RAFT_CUDA_TRY(cudaFree(Z));
  }

 protected:
  GemmLayoutInputs<T> params;
  T* refZ = NULL;  // Reference result for comparison
  T* Z    = NULL;  // Computed result
};

const std::vector<GemmLayoutInputs<float>> inputsf = {
  {80, 70, 80, true, true, true, 76430ULL},
  {80, 100, 40, true, true, false, 426646ULL},
  {20, 100, 20, true, false, true, 237703ULL},
  {100, 60, 30, true, false, false, 538004ULL},
  {50, 10, 60, false, true, true, 73012ULL},
  {90, 90, 30, false, true, false, 538147ULL},
  {30, 100, 10, false, false, true, 412352ULL},
  {40, 80, 100, false, false, false, 2979410ULL}};

const std::vector<GemmLayoutInputs<double>> inputsd = {
  {10, 70, 40, true, true, true, 535648ULL},
  {30, 30, 30, true, true, false, 956681ULL},
  {70, 80, 50, true, false, true, 875083ULL},
  {80, 90, 70, true, false, false, 50744ULL},
  {90, 90, 30, false, true, true, 506321ULL},
  {40, 100, 70, false, true, false, 638418ULL},
  {80, 50, 30, false, false, true, 701529ULL},
  {50, 80, 60, false, false, false, 893038ULL}};

typedef GemmLayoutTest<float> GemmLayoutTestF;
TEST_P(GemmLayoutTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(refZ, Z, params.M * params.N, raft::CompareApprox<float>(2e-4)));
}

typedef GemmLayoutTest<double> GemmLayoutTestD;
TEST_P(GemmLayoutTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(refZ, Z, params.M * params.N, raft::CompareApprox<float>(1e-6)));
}

INSTANTIATE_TEST_SUITE_P(GemmLayoutTests, GemmLayoutTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(GemmLayoutTests, GemmLayoutTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
