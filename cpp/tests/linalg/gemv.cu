/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemv.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T>
struct GemvInputs {
  int n_rows;
  int n_cols;
  int lda;
  bool trans_a;
  unsigned long long int seed;
};

// Reference GEMV implementation.
template <typename T>
RAFT_KERNEL naiveGemv(T* y,
                      const T* A,
                      const T* x,
                      const int n_rows,
                      const int n_cols,
                      const int lda,
                      const bool trans_a)
{
  int istart = blockIdx.x * blockDim.x + threadIdx.x;
  int istep  = blockDim.x * gridDim.x;

  if (!trans_a) {
    for (int i = istart; i < n_rows; i += istep) {
      T t = T(0.0);
      for (int j = 0; j < n_cols; j++) {
        t += A[i + lda * j] * x[j];
      }
      y[i] = t;
    }
  } else {
    for (int i = istart; i < n_cols; i += istep) {
      T t = T(0.0);
      for (int j = 0; j < n_rows; j++) {
        t += A[lda * i + j] * x[j];
      }
      y[i] = t;
    }
  }
}

template <typename T>
class GemvTest : public ::testing::TestWithParam<GemvInputs<T>> {
 protected:
  GemvInputs<T> params;
  rmm::device_uvector<T> refy;  // Reference result for comparison
  rmm::device_uvector<T> y;     // Computed result

 public:
  GemvTest()
    : testing::TestWithParam<GemvInputs<T>>(),
      refy(0, rmm::cuda_stream_default),
      y(0, rmm::cuda_stream_default)
  {
    rmm::cuda_stream_default.synchronize();
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<GemvInputs<T>>::GetParam();

    raft::resources handle;
    cudaStream_t stream = resource::get_cuda_stream(handle);

    raft::random::RngState r(params.seed);

    // We compute y = op(A) * x and compare against reference result
    size_t aElems = params.lda * params.n_cols;
    size_t xElems = params.trans_a ? params.n_rows : params.n_cols;
    size_t yElems = params.trans_a ? params.n_cols : params.n_rows;

    rmm::device_uvector<T> A(aElems, stream);
    rmm::device_uvector<T> x(xElems, stream);
    refy.resize(yElems, stream);
    y.resize(yElems, stream);

    uniform(handle, r, x.data(), xElems, T(-10.0), T(10.0));
    uniform(handle, r, A.data(), aElems, T(-10.0), T(10.0));

    dim3 blocks(raft::ceildiv<int>(yElems, 256), 1, 1);
    dim3 threads(256, 1, 1);

    raft::launch_kernel(handle,
                        blocks,
                        threads,
                        naiveGemv<T>,
                        refy.data(),
                        A.data(),
                        x.data(),
                        params.n_rows,
                        params.n_cols,
                        params.lda,
                        params.trans_a);

    auto A_row_major =
      raft::make_device_matrix_view<const T>(A.data(), params.n_rows, params.n_cols);
    auto A_col_major = raft::make_device_matrix_view<const T, int, raft::col_major>(
      A.data(), params.n_rows, params.n_cols);

    auto x_view = raft::make_device_vector_view<const T, int>(x.data(), xElems);
    auto y_view = raft::make_device_vector_view<T, int>(y.data(), yElems);

    if (params.trans_a) {
      gemv(handle, A_row_major, x_view, y_view);
    } else {
      gemv(handle, A_col_major, x_view, y_view);
    }

    resource::sync_stream(handle);
  }

  void TearDown() override {}
};

const std::vector<GemvInputs<float>> inputsf = {{80, 70, 80, true, 76433ULL},
                                                {80, 100, 80, true, 426646ULL},
                                                {20, 100, 20, true, 37703ULL},
                                                {90, 90, 90, false, 538147ULL},
                                                {30, 100, 30, false, 412352ULL}};

const std::vector<GemvInputs<double>> inputsd = {{10, 70, 10, true, 535648ULL},
                                                 {30, 30, 30, true, 956681ULL},
                                                 {70, 80, 70, true, 875083ULL},
                                                 {90, 90, 90, false, 506321ULL},
                                                 {80, 50, 80, false, 701529ULL}};

typedef GemvTest<float> GemvTestF;
TEST_P(GemvTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(refy.data(),
                                y.data(),
                                params.trans_a ? params.n_cols : params.n_rows,
                                raft::CompareApprox<float>(1e-4)));
}

typedef GemvTest<double> GemvTestD;
TEST_P(GemvTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(refy.data(),
                                y.data(),
                                params.trans_a ? params.n_cols : params.n_rows,
                                raft::CompareApprox<float>(1e-6)));
}

INSTANTIATE_TEST_SUITE_P(GemvTests, GemvTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(GemvTests, GemvTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
