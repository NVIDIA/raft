/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace random {

template <typename T>
RAFT_KERNEL meanKernel(T* out,
                       int* lens,
                       const T* data,
                       const int* labels,
                       int nrows,
                       int ncols,
                       int nclusters,
                       bool row_major)
{
  int tid   = threadIdx.x + blockIdx.x * blockDim.x;
  int rowid = row_major ? tid / ncols : tid % nrows;
  int colid = row_major ? tid % ncols : tid / nrows;
  if (rowid < nrows && colid < ncols) {
    T val     = data[tid];
    int label = labels[rowid];
    int idx   = row_major ? label * ncols + colid : colid * nclusters + label;
    raft::myAtomicAdd(out + idx * 2, val);
    raft::myAtomicAdd(out + idx * 2 + 1, val * val);
    if (colid == 0) { raft::myAtomicAdd(lens + label, 1); }
  }
}

template <typename T>
RAFT_KERNEL compute_mean_var(
  T* out, const T* stats, int* lens, int nrows, int ncols, bool row_major)
{
  int tid    = threadIdx.x + blockIdx.x * blockDim.x;
  int rowid  = row_major ? tid / ncols : tid % nrows;
  int colid  = row_major ? tid % ncols : tid / nrows;
  int stride = nrows * ncols;
  if (rowid < nrows && colid < ncols) {
    int len           = lens[rowid];
    auto mean         = stats[tid * 2] / len;
    out[tid]          = mean;
    out[tid + stride] = (stats[tid * 2 + 1] / len) - (mean * mean);
  }
}

template <typename T>
struct MakeBlobsInputs {
  T tolerance;
  int rows, cols, n_clusters;
  T std;
  bool shuffle;
  raft::random::GeneratorType gtype;
  uint64_t seed;
};

template <typename T, typename layout>
class MakeBlobsTest : public ::testing::TestWithParam<MakeBlobsInputs<T>> {
 public:
  MakeBlobsTest()
    : params(::testing::TestWithParam<MakeBlobsInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      mu_vec(make_device_matrix<T, int, layout>(handle, params.n_clusters, params.cols)),
      mean_var(make_device_vector<T, int>(handle, 2 * params.n_clusters * params.cols))
  {
  }

 protected:
  void SetUp() override
  {
    // Tests are configured with their expected test-values sigma. For example,
    // 4 x sigma indicates the test shouldn't fail 99.9% of the time.
    num_sigma = 50;
    auto len  = params.rows * params.cols;
    raft::random::RngState r(params.seed, params.gtype);

    auto data   = make_device_matrix<T, int, layout>(handle, params.rows, params.cols);
    auto labels = make_device_vector<int, int>(handle, params.rows);
    auto stats  = make_device_vector<T, int>(handle, 2 * params.n_clusters * params.cols);
    auto lens   = make_device_vector<int, int>(handle, params.n_clusters);

    RAFT_CUDA_TRY(cudaMemsetAsync(stats.data_handle(), 0, stats.extent(0) * sizeof(T), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(lens.data_handle(), 0, lens.extent(0) * sizeof(int), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(mean_var.data_handle(), 0, mean_var.size() * sizeof(T), stream));

    uniform(handle, r, mu_vec.data_handle(), params.cols * params.n_clusters, T(-10.0), T(10.0));

    make_blobs<T, int, layout>(handle,
                               data.view(),
                               labels.view(),
                               params.n_clusters,
                               std::make_optional(mu_vec.view()),
                               std::nullopt,
                               params.std,
                               params.shuffle,
                               T(-10.0),
                               T(10.0),
                               params.seed,
                               params.gtype);

    bool row_major           = std::is_same<layout, raft::layout_c_contiguous>::value;
    static const int threads = 128;
    raft::launch_kernel(stream,
                        raft::ceildiv(len, threads),
                        threads,
                        meanKernel<T>,
                        stats.data_handle(),
                        lens.data_handle(),
                        data.data_handle(),
                        labels.data_handle(),
                        params.rows,
                        params.cols,
                        params.n_clusters,
                        row_major);
    int len1 = params.n_clusters * params.cols;
    raft::launch_kernel(stream,
                        raft::ceildiv(len1, threads),
                        threads,
                        compute_mean_var<T>,
                        mean_var.data_handle(),
                        stats.data_handle(),
                        lens.data_handle(),
                        params.n_clusters,
                        params.cols,
                        row_major);
  }

  void check()
  {
    int len      = params.n_clusters * params.cols;
    auto compare = raft::CompareApprox<T>(num_sigma * params.tolerance);
    ASSERT_TRUE(raft::devArrMatch(mu_vec.data_handle(), mean_var.data_handle(), len, compare));
    ASSERT_TRUE(raft::devArrMatch(params.std, mean_var.data_handle() + len, len, compare));
  }

 protected:
  raft::resources handle;
  MakeBlobsInputs<T> params;
  cudaStream_t stream = 0;

  device_vector<T, int> mean_var;
  device_matrix<T, int, layout> mu_vec;
  int num_sigma;
};

typedef MakeBlobsTest<float, raft::layout_c_contiguous> MakeBlobsTestF_RowMajor;
typedef MakeBlobsTest<float, raft::layout_f_contiguous> MakeBlobsTestF_ColMajor;

const std::vector<MakeBlobsInputs<float>> inputsf_t = {
  {0.0055, 1024, 32, 3, 1.f, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, false, raft::random::GenPC, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, false, raft::random::GenPC, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.f, true, raft::random::GenPC, 1234ULL},
  {0.011, 1024, 8, 3, 1.f, true, raft::random::GenPC, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, false, raft::random::GenPC, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, false, raft::random::GenPC, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.f, true, raft::random::GenPC, 1234ULL},
  {0.011, 5003, 8, 5, 1.f, true, raft::random::GenPC, 1234ULL},
};

TEST_P(MakeBlobsTestF_RowMajor, Result) { check(); }
INSTANTIATE_TEST_CASE_P(MakeBlobsTests, MakeBlobsTestF_RowMajor, ::testing::ValuesIn(inputsf_t));

TEST_P(MakeBlobsTestF_ColMajor, Result) { check(); }
INSTANTIATE_TEST_CASE_P(MakeBlobsTests, MakeBlobsTestF_ColMajor, ::testing::ValuesIn(inputsf_t));

typedef MakeBlobsTest<double, raft::layout_c_contiguous> MakeBlobsTestD_RowMajor;
typedef MakeBlobsTest<double, raft::layout_f_contiguous> MakeBlobsTestD_ColMajor;

const std::vector<MakeBlobsInputs<double>> inputsd_t = {
  {0.0055, 1024, 32, 3, 1.0, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, false, raft::random::GenPC, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, false, raft::random::GenPC, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 1024, 32, 3, 1.0, true, raft::random::GenPC, 1234ULL},
  {0.011, 1024, 8, 3, 1.0, true, raft::random::GenPC, 1234ULL},

  {0.0055, 5003, 32, 5, 1.0, false, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, false, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, false, raft::random::GenPC, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, false, raft::random::GenPC, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, true, raft::random::GenPhilox, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, true, raft::random::GenPhilox, 1234ULL},
  {0.0055, 5003, 32, 5, 1.0, true, raft::random::GenPC, 1234ULL},
  {0.011, 5003, 8, 5, 1.0, true, raft::random::GenPC, 1234ULL},
};
TEST_P(MakeBlobsTestD_RowMajor, Result) { check(); }
INSTANTIATE_TEST_CASE_P(MakeBlobsTests, MakeBlobsTestD_RowMajor, ::testing::ValuesIn(inputsd_t));

TEST_P(MakeBlobsTestD_ColMajor, Result) { check(); }
INSTANTIATE_TEST_CASE_P(MakeBlobsTests, MakeBlobsTestD_ColMajor, ::testing::ValuesIn(inputsd_t));

}  // end namespace random
}  // end namespace raft
