/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/random/rng.cuh>
#include <raft/sparse/linalg/spmm.hpp>
#include <raft/util/cuda_utils.cuh>

#include <thrust/fill.h>

#include <gtest/gtest.h>

namespace raft {
namespace sparse {
namespace linalg {

template <typename T>
struct SpmmInputs {
  bool trans_x;
  bool trans_y;
  int M;
  int N;
  int K;
  int ldy;
  int ldz;
  bool row_major;
  T alpha;
  T beta;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SpmmInputs<T>& params)
{
  os << " trans_x: " << params.trans_x << " trans_y: " << params.trans_y << " M: " << params.M
     << ", N: " << params.N << ", K: " << params.K << ", ldy: " << params.ldy
     << ", ldz: " << params.ldz << ", row_major: " << params.row_major
     << ", alpha: " << params.alpha << ", beta: " << params.beta;
  return os;
}

// Reference GEMM implementation.
template <typename T>
RAFT_KERNEL naiveGemm(bool trans_x,
                      bool trans_y,
                      int M,
                      int N,
                      int K,
                      T alpha,
                      T* X,
                      int ldx,
                      bool x_row_major,
                      T* Y,
                      int ldy,
                      bool y_row_major,
                      T beta,
                      T* Z,
                      int ldz,
                      bool z_row_major)
{
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

  for (int m = tidy; m < M; m += (blockDim.y * gridDim.y)) {
    for (int n = tidx; n < N; n += (blockDim.x * gridDim.x)) {
      T temp = T(0.0);
      for (int k = 0; k < K; k++) {
        int xIndex = x_row_major != trans_x ? m * ldx + k : m + k * ldx;
        int yIndex = y_row_major != trans_y ? k * ldy + n : k + n * ldy;
        temp += X[xIndex] * Y[yIndex];
      }
      int zIndex = z_row_major ? m * ldz + n : m + n * ldz;
      Z[zIndex]  = beta * Z[zIndex] + alpha * temp;
    }
  }
}

template <typename T>
class SpmmTest : public ::testing::TestWithParam<SpmmInputs<T>> {
 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<SpmmInputs<T>>::GetParam();

    cudaStream_t stream = resource::get_cuda_stream(handle);

    // We compute Z = X * Y and compare against reference result
    // Dimensions of X : M x K
    // Dimensions of Y : K x N
    // Dimensions of Z : M x N

    auto [ldx, ldy, ldz, x_size, y_size, z_size] = getXYZStrides();

    RAFT_CUDA_TRY(cudaMalloc(&X, x_size * sizeof(T)));
    RAFT_CUDA_TRY(cudaMalloc(&Y, y_size * sizeof(T)));
    RAFT_CUDA_TRY(cudaMalloc(&Z_ref, z_size * sizeof(T)));
    RAFT_CUDA_TRY(cudaMalloc(&Z, z_size * sizeof(T)));

    raft::random::RngState r(params.seed);
    raft::random::uniform(handle, r, X, x_size, T(-10.0), T(10.0));
    raft::random::uniform(handle, r, Y, y_size, T(-10.0), T(10.0));

    RAFT_CUDA_TRY(
      cudaMalloc(&X_indptr, ((params.trans_x ? params.K : params.M) + 1) * sizeof(int)));
    RAFT_CUDA_TRY(cudaMalloc(&X_indices, z_size * sizeof(int)));
    RAFT_CUDA_TRY(cudaMalloc(&X_data, z_size * sizeof(T)));

    // this will erase all entries below eps and update X accordingly
    X_nnz = generateCsrFromDense(X,
                                 params.trans_x ? params.K : params.M,
                                 params.trans_x ? params.M : params.K,
                                 X_indptr,
                                 X_indices,
                                 X_data);

    resource::sync_stream(handle);
  }

  std::tuple<int, int, int, size_t, size_t, size_t> getXYZStrides()
  {
    // X is always row major
    int ldx = params.trans_x ? params.M : params.K;
    int ldy =
      params.ldy > 0 ? params.ldy : (params.row_major != params.trans_y ? params.N : params.K);
    int ldz       = params.ldz > 0 ? params.ldz : (params.row_major ? params.N : params.M);
    size_t x_size = params.M * params.K;
    size_t y_size = params.row_major != params.trans_y ? params.K * ldy : params.N * ldy;
    size_t z_size = params.row_major ? params.M * ldz : params.N * ldz;

    return {ldx, ldy, ldz, x_size, y_size, z_size};
  }

  void runTest()
  {
    auto stream = resource::get_cuda_stream(handle);

    auto [ldx, ldy, ldz, x_size, y_size, z_size] = getXYZStrides();

    thrust::fill(resource::get_thrust_policy(handle), Z_ref, Z_ref + z_size, T(0.0));
    thrust::fill(resource::get_thrust_policy(handle), Z, Z + z_size, T(0.0));

    // create csr matrix view
    auto X_csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
      X_indptr,
      X_indices,
      params.trans_x ? params.K : params.M,
      params.trans_x ? params.M : params.K,
      X_nnz);
    auto X_csr = raft::device_csr_matrix_view<const T, int, int, int>(
      raft::device_span<const T>(X_data, X_csr_structure.get_nnz()), X_csr_structure);

    auto y_stride_view =
      params.row_major
        ? raft::make_device_strided_matrix_view<const T, int, layout_c_contiguous>(
            Y, params.trans_y ? params.N : params.K, params.trans_y ? params.K : params.N, ldy)
        : raft::make_device_strided_matrix_view<const T, int, layout_f_contiguous>(
            Y, params.trans_y ? params.N : params.K, params.trans_y ? params.K : params.N, ldy);

    auto z_stride_view = params.row_major
                           ? raft::make_device_strided_matrix_view<T, int, layout_c_contiguous>(
                               Z, params.M, params.N, ldz)
                           : raft::make_device_strided_matrix_view<T, int, layout_f_contiguous>(
                               Z, params.M, params.N, ldz);

    T alpha = params.alpha;
    T beta  = params.beta;

    dim3 blocks(raft::ceildiv<int>(params.M, 128), raft::ceildiv<int>(params.N, 4), 1);
    dim3 threads(128, 4, 1);
    naiveGemm<<<blocks, threads, 0, stream>>>(params.trans_x,
                                              params.trans_y,
                                              params.M,
                                              params.N,
                                              params.K,
                                              alpha,
                                              X,
                                              ldx,
                                              true,
                                              Y,
                                              ldy,
                                              params.row_major,
                                              beta,
                                              Z_ref,
                                              ldz,
                                              params.row_major);

    // min_alloc: the actual contiguous span of the strided z matrix (what spmm allocates for z_tmp)
    auto z_span = params.row_major ? (size_t(params.M) - 1) * ldz + params.N
                                   : (size_t(params.N) - 1) * ldz + params.M;
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        spmm(h, params.trans_x, params.trans_y, &alpha, X_csr, y_stride_view, &beta, z_stride_view);
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      z_span * sizeof(T));

    resource::sync_stream(handle, stream);

    ASSERT_TRUE(raft::devArrMatch(Z_ref, Z, z_size, raft::CompareApprox<T>(1e-3f), stream));
  }

  void TearDown() override
  {
    RAFT_CUDA_TRY(cudaFree(Z_ref));
    RAFT_CUDA_TRY(cudaFree(Z));
    RAFT_CUDA_TRY(cudaFree(X));
    RAFT_CUDA_TRY(cudaFree(Y));
    RAFT_CUDA_TRY(cudaFree(X_indptr));
    RAFT_CUDA_TRY(cudaFree(X_indices));
    RAFT_CUDA_TRY(cudaFree(X_data));
  }

  int generateCsrFromDense(T* dense, int n_rows, int n_cols, int* indptr, int* indices, T* data)
  {
    double eps = 1e-4;

    cudaStream_t stream = resource::get_cuda_stream(handle);

    size_t dense_size = n_rows * n_cols;
    std::vector<T> dense_host(dense_size);
    raft::update_host(dense_host.data(), dense, dense_size, stream);
    resource::sync_stream(handle, stream);

    std::vector<int> indptr_host(n_rows + 1);
    std::vector<int> indices_host(dense_size);
    std::vector<T> data_host(dense_size);

    // create csr matrix from dense (with threshold)
    int nnz = 0;
    for (int i = 0; i < n_rows; ++i) {
      indptr_host[i] = nnz;
      for (int j = 0; j < n_cols; ++j) {
        T value = dense_host[i * n_cols + j];
        if (value > eps) {
          indices_host[nnz] = j;
          data_host[nnz]    = value;
          nnz++;
        } else {
          dense_host[i * n_cols + j] = T(0.0);
        }
      }
    }
    indptr_host[n_rows] = nnz;

    raft::update_device(dense, dense_host.data(), dense_size, stream);
    raft::update_device(indptr, indptr_host.data(), n_rows + 1, stream);
    raft::update_device(indices, indices_host.data(), nnz, stream);
    raft::update_device(data, data_host.data(), nnz, stream);
    resource::sync_stream(handle, stream);
    return nnz;
  }

 protected:
  raft::resources handle;
  SpmmInputs<T> params;

  T* X     = NULL;
  T* Y     = NULL;
  T* Z_ref = NULL;
  T* Z     = NULL;

  // CSR
  int* X_indptr  = NULL;
  int* X_indices = NULL;
  T* X_data      = NULL;
  int X_nnz      = 0;
};

// M / N / K / ldy / ldz / y_row_major / z_row_major / seed
const std::vector<SpmmInputs<float>> inputsf = {
  {true, false, 80, 70, 80, 0, 0, true, 1.0, 2.0, 76430ULL},
  {false, false, 80, 100, 40, 0, 0, true, 3.0, 0.0, 426646ULL},
  {false, false, 20, 100, 20, 0, 0, false, 2.0, 2.0, 237703ULL},
  {false, true, 100, 60, 30, 0, 0, false, 1.0, 0.0, 538004ULL},
  {false, false, 80, 70, 80, 106, 0, true, 1.0, 2.0, 76430ULL},
  {false, false, 80, 100, 40, 0, 110, true, 3.0, 0.0, 426646ULL},
  {true, false, 20, 100, 20, 106, 0, false, 2.0, 2.0, 237703ULL},
  {false, false, 100, 60, 30, 0, 110, false, 1.0, 0.0, 538004ULL},
  {false, false, 50, 10, 60, 106, 110, true, 1.0, 1.0, 73012ULL},
  {true, false, 90, 90, 30, 106, 110, true, 1.0, 0.0, 538147ULL},
  {false, false, 30, 100, 10, 106, 110, false, 1.0, 1.0, 412352ULL},
  {false, false, 40, 80, 100, 106, 110, false, 1.0, 0.0, 2979410ULL},
  {true, false, 50, 10, 60, 106, 110, true, 1.0, 1.0, 73012ULL},
  {true, true, 90, 90, 30, 106, 110, true, 1.0, 0.0, 538147ULL},
  {false, true, 30, 100, 10, 106, 110, false, 1.0, 1.0, 412352ULL},
  {true, true, 40, 80, 100, 106, 110, false, 1.0, 0.0, 2979410ULL}};

const std::vector<SpmmInputs<double>> inputsd = {
  {false, false, 10, 70, 40, 0, 0, true, 1.0, 2.0, 535648ULL},
  {false, false, 30, 30, 30, 0, 0, true, 3.0, 0.0, 956681ULL},
  {true, false, 70, 80, 50, 0, 0, false, 2.0, 2.0, 875083ULL},
  {false, false, 80, 90, 70, 0, 0, false, 1.0, 0.0, 50744ULL},
  {false, false, 10, 70, 40, 106, 0, true, 1.0, 2.0, 535648ULL},
  {true, false, 30, 30, 30, 0, 110, true, 3.0, 0.0, 956681ULL},
  {false, false, 70, 80, 50, 106, 0, false, 2.0, 2.0, 875083ULL},
  {false, false, 80, 90, 70, 0, 110, false, 1.0, 0.0, 50744ULL},
  {false, true, 90, 90, 30, 106, 110, true, 1.0, 1.0, 506321ULL},
  {false, false, 40, 100, 70, 106, 110, true, 1.0, 0.0, 638418ULL},
  {false, true, 80, 50, 30, 106, 110, false, 1.0, 1.0, 701529ULL},
  {false, false, 50, 80, 60, 106, 110, false, 1.0, 0.0, 893038ULL},
  {true, false, 90, 90, 30, 106, 110, true, 1.0, 1.0, 506321ULL},
  {true, true, 40, 100, 70, 106, 110, true, 1.0, 0.0, 638418ULL},
  {false, true, 80, 50, 30, 106, 110, false, 1.0, 1.0, 701529ULL},
  {true, true, 50, 80, 60, 106, 110, false, 1.0, 0.0, 893038ULL}};

typedef SpmmTest<float> SpmmTestF;
TEST_P(SpmmTestF, Result) { runTest(); }

typedef SpmmTest<double> SpmmTestD;
TEST_P(SpmmTestD, Result) { runTest(); }

INSTANTIATE_TEST_SUITE_P(SpmmTests, SpmmTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(SpmmTests, SpmmTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
