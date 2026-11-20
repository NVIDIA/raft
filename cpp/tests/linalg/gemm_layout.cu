/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

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

    raft::launch_kernel(cudaStream_t{0},
                        blocks,
                        threads,
                        naiveGemm,
                        refZ,
                        X,
                        Y,
                        params.M,
                        params.N,
                        params.K,
                        params.zLayout,
                        params.xLayout,
                        params.yLayout);

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

// C = alpha * A * B + beta * C + bias, with bias[i] added to row i of C (the cuBLASLt bias
// epilogue), for A and B of type T: float, or half with float accumulation and output. The bias
// starts bias_offset elements into its buffer, so an odd offset leaves it only 4-byte aligned.
template <typename T>
void gemm_bias_test(int bias_offset)
{
  raft::resources handle;
  auto stream       = resource::get_cuda_stream(handle);
  constexpr int m   = 2;
  constexpr int n   = 2;
  constexpr int k   = 3;
  const float alpha = 2.0f;
  const float beta  = 3.0f;
  // Column-major A = [1 2 3; 4 5 6], B = [7 8; 9 10; 11 12], C = [1 2; 3 4] and bias = [10; 20].
  std::vector<T> a_host;
  std::vector<T> b_host;
  for (float v : {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}) {
    a_host.push_back(T(v));
  }
  for (float v : {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f}) {
    b_host.push_back(T(v));
  }
  std::vector<float> c_host    = {1.0f, 3.0f, 2.0f, 4.0f};
  std::vector<float> bias_host = {10.0f, 20.0f};
  // alpha * A * B + beta * C = [119 134; 287 320], plus 10 on row 0 and 20 on row 1.
  std::vector<float> expected = {129.0f, 307.0f, 144.0f, 340.0f};

  auto a    = raft::make_device_vector<T>(handle, m * k);
  auto b    = raft::make_device_vector<T>(handle, k * n);
  auto c    = raft::make_device_vector<float>(handle, m * n);
  auto bias = raft::make_device_vector<float>(handle, bias_offset + m);
  raft::update_device(a.data_handle(), a_host.data(), a_host.size(), stream);
  raft::update_device(b.data_handle(), b_host.data(), b_host.size(), stream);
  raft::update_device(c.data_handle(), c_host.data(), c_host.size(), stream);
  raft::update_device(bias.data_handle() + bias_offset, bias_host.data(), bias_host.size(), stream);

  raft::execute_with_dry_run_check(
    handle,
    [&](raft::resources const& h) {
      gemm(h,
           false,
           false,
           m,
           n,
           k,
           &alpha,
           a.data_handle(),
           m,
           b.data_handle(),
           k,
           &beta,
           c.data_handle(),
           m,
           resource::get_cuda_stream(h).get(),
           static_cast<const float*>(bias.data_handle() + bias_offset));
    },
    raft::alloc_behavior::NO_ALLOCATIONS);

  std::vector<float> result(m * n);
  raft::update_host(result.data(), c.data_handle(), result.size(), stream);
  resource::sync_stream(handle);
  for (int i = 0; i < m * n; ++i) {
    EXPECT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
  }
}

TEST(GemmBias, Float)
{
  gemm_bias_test<float>(0);
  gemm_bias_test<float>(1);
}

TEST(GemmBias, Half)
{
  gemm_bias_test<half>(0);
  gemm_bias_test<half>(1);
}

}  // end namespace linalg
}  // end namespace raft
