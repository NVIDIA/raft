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

#include <optional>
#include <type_traits>
#include <vector>

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
      stream(resource::get_cuda_stream(handle).get()),
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

template <typename T, typename Layout>
void check_cluster_std_by_label()
{
  // More rows than clusters exercises the per-cluster vector beyond its first two sample rows.
  constexpr int n_rows     = 65;
  constexpr int n_cols     = 2;
  constexpr int n_clusters = 2;
  raft::resources handle;
  auto stream         = resource::get_cuda_stream(handle).get();
  auto data           = make_device_matrix<T, int, Layout>(handle, n_rows, n_cols);
  auto labels         = make_device_vector<int, int>(handle, n_rows);
  auto control        = make_device_matrix<T, int, Layout>(handle, n_rows, n_cols);
  auto control_labels = make_device_vector<int, int>(handle, n_rows);
  auto centers        = make_device_matrix<T, int, Layout>(handle, n_clusters, n_cols);
  auto cluster_std    = make_device_vector<T, int>(handle, n_clusters);

  std::vector<T> host_centers(n_clusters * n_cols, T(0));
  std::vector<T> host_std{T(0), T(0.8)};
  raft::update_device(centers.data_handle(), host_centers.data(), host_centers.size(), stream);
  raft::update_device(cluster_std.data_handle(), host_std.data(), host_std.size(), stream);

  make_blobs<T, int, Layout>(handle,
                             data.view(),
                             labels.view(),
                             n_clusters,
                             std::make_optional(centers.view()),
                             std::make_optional(cluster_std.view()),
                             T(1),
                             false,
                             T(-10),
                             T(10),
                             1234ULL,
                             raft::random::GenPC);
  make_blobs<T, int, Layout>(handle,
                             control.view(),
                             control_labels.view(),
                             n_clusters,
                             std::make_optional(centers.view()),
                             std::nullopt,
                             T(1),
                             false,
                             T(-10),
                             T(10),
                             1234ULL,
                             raft::random::GenPC);

  std::vector<T> host_data(n_rows * n_cols);
  std::vector<T> host_control(n_rows * n_cols);
  std::vector<int> host_labels(n_rows);
  std::vector<int> host_control_labels(n_rows);
  raft::update_host(host_data.data(), data.data_handle(), host_data.size(), stream);
  raft::update_host(host_control.data(), control.data_handle(), host_control.size(), stream);
  raft::update_host(host_labels.data(), labels.data_handle(), host_labels.size(), stream);
  raft::update_host(
    host_control_labels.data(), control_labels.data_handle(), host_control_labels.size(), stream);
  resource::sync_stream(handle);

  bool second_cluster_varies = false;
  constexpr bool row_major   = std::is_same<Layout, raft::layout_c_contiguous>::value;
  for (int row = 0; row < n_rows; ++row) {
    ASSERT_EQ(host_labels[row], row % n_clusters);
    ASSERT_EQ(host_labels[row], host_control_labels[row]);
    for (int col = 0; col < n_cols; ++col) {
      auto offset   = row_major ? row * n_cols + col : col * n_rows + row;
      auto expected = host_control[offset] * host_std[host_labels[row]];
      if (host_labels[row] == 0) {
        EXPECT_EQ(host_data[offset], T(0));
      } else {
        second_cluster_varies |= host_data[offset] != T(0);
      }
      EXPECT_NEAR(host_data[offset], expected, T(1e-5));
    }
  }
  EXPECT_TRUE(second_cluster_varies);
}

TEST(MakeBlobsClusterStd, FloatRowMajor)
{
  check_cluster_std_by_label<float, raft::layout_c_contiguous>();
}

TEST(MakeBlobsClusterStd, FloatColMajor)
{
  check_cluster_std_by_label<float, raft::layout_f_contiguous>();
}

TEST(MakeBlobsClusterStd, DoubleRowMajor)
{
  check_cluster_std_by_label<double, raft::layout_c_contiguous>();
}

TEST(MakeBlobsClusterStd, DoubleColMajor)
{
  check_cluster_std_by_label<double, raft::layout_f_contiguous>();
}

template <typename T, typename Layout>
void check_generated_centers_output()
{
  constexpr int n_rows     = 257;
  constexpr int n_cols     = 3;
  constexpr int n_clusters = 5;
  raft::resources handle;
  auto stream                = resource::get_cuda_stream(handle).get();
  auto data                  = make_device_matrix<T, int, Layout>(handle, n_rows, n_cols);
  auto labels                = make_device_vector<int, int>(handle, n_rows);
  auto data_without_output   = make_device_matrix<T, int, Layout>(handle, n_rows, n_cols);
  auto labels_without_output = make_device_vector<int, int>(handle, n_rows);
  auto generated_centers     = make_device_matrix<T, int, Layout>(handle, n_clusters, n_cols);

  make_blobs<T, int, Layout>(handle,
                             data.view(),
                             labels.view(),
                             n_clusters,
                             std::nullopt,
                             std::nullopt,
                             T(0),
                             true,
                             T(-3),
                             T(3),
                             42ULL,
                             raft::random::GenPC,
                             std::make_optional(generated_centers.view()));
  make_blobs<T, int, Layout>(handle,
                             data_without_output.view(),
                             labels_without_output.view(),
                             n_clusters,
                             std::nullopt,
                             std::nullopt,
                             T(0),
                             true,
                             T(-3),
                             T(3),
                             42ULL,
                             raft::random::GenPC);

  std::vector<T> host_data(n_rows * n_cols);
  std::vector<T> host_data_without_output(n_rows * n_cols);
  std::vector<T> host_centers(n_clusters * n_cols);
  std::vector<int> host_labels(n_rows);
  std::vector<int> host_labels_without_output(n_rows);
  raft::update_host(host_data.data(), data.data_handle(), host_data.size(), stream);
  raft::update_host(host_data_without_output.data(),
                    data_without_output.data_handle(),
                    host_data_without_output.size(),
                    stream);
  raft::update_host(
    host_centers.data(), generated_centers.data_handle(), host_centers.size(), stream);
  raft::update_host(host_labels.data(), labels.data_handle(), host_labels.size(), stream);
  raft::update_host(host_labels_without_output.data(),
                    labels_without_output.data_handle(),
                    host_labels_without_output.size(),
                    stream);
  resource::sync_stream(handle);

  EXPECT_EQ(host_data, host_data_without_output);
  EXPECT_EQ(host_labels, host_labels_without_output);
  constexpr bool row_major = std::is_same<Layout, raft::layout_c_contiguous>::value;
  for (int row = 0; row < n_rows; ++row) {
    for (int col = 0; col < n_cols; ++col) {
      auto data_offset = row_major ? row * n_cols + col : col * n_rows + row;
      auto center_offset =
        row_major ? host_labels[row] * n_cols + col : col * n_clusters + host_labels[row];
      EXPECT_EQ(host_data[data_offset], host_centers[center_offset]);
    }
  }
}

TEST(MakeBlobsGeneratedCenters, FloatRowMajor)
{
  check_generated_centers_output<float, raft::layout_c_contiguous>();
}

TEST(MakeBlobsGeneratedCenters, FloatColMajor)
{
  check_generated_centers_output<float, raft::layout_f_contiguous>();
}

TEST(MakeBlobsGeneratedCenters, DoubleRowMajor)
{
  check_generated_centers_output<double, raft::layout_c_contiguous>();
}

TEST(MakeBlobsGeneratedCenters, DoubleColMajor)
{
  check_generated_centers_output<double, raft::layout_f_contiguous>();
}

TEST(MakeBlobsShuffle, PermutesRowsDeterministically)
{
  constexpr int n_rows     = 257;
  constexpr int n_clusters = 5;
  raft::resources handle;
  auto stream        = resource::get_cuda_stream(handle).get();
  auto data          = make_device_matrix<float, int, raft::layout_c_contiguous>(handle, n_rows, 2);
  auto labels        = make_device_vector<int, int>(handle, n_rows);
  auto repeated_data = make_device_matrix<float, int, raft::layout_c_contiguous>(handle, n_rows, 2);
  auto repeated_labels = make_device_vector<int, int>(handle, n_rows);
  auto unshuffled_data =
    make_device_matrix<float, int, raft::layout_c_contiguous>(handle, n_rows, 2);
  auto unshuffled_labels = make_device_vector<int, int>(handle, n_rows);

  auto generate = [&](auto data_view, auto label_view, bool shuffle) {
    make_blobs<float, int, raft::layout_c_contiguous>(handle,
                                                      data_view,
                                                      label_view,
                                                      n_clusters,
                                                      std::nullopt,
                                                      std::nullopt,
                                                      0.f,
                                                      shuffle,
                                                      -3.f,
                                                      3.f,
                                                      42ULL);
  };
  generate(data.view(), labels.view(), true);
  generate(repeated_data.view(), repeated_labels.view(), true);
  generate(unshuffled_data.view(), unshuffled_labels.view(), false);

  std::vector<float> host_data(n_rows * 2);
  std::vector<float> host_repeated_data(n_rows * 2);
  std::vector<int> host_labels(n_rows);
  std::vector<int> host_repeated_labels(n_rows);
  std::vector<int> host_unshuffled_labels(n_rows);
  raft::update_host(host_data.data(), data.data_handle(), host_data.size(), stream);
  raft::update_host(
    host_repeated_data.data(), repeated_data.data_handle(), host_repeated_data.size(), stream);
  raft::update_host(host_labels.data(), labels.data_handle(), host_labels.size(), stream);
  raft::update_host(host_repeated_labels.data(),
                    repeated_labels.data_handle(),
                    host_repeated_labels.size(),
                    stream);
  raft::update_host(host_unshuffled_labels.data(),
                    unshuffled_labels.data_handle(),
                    host_unshuffled_labels.size(),
                    stream);
  resource::sync_stream(handle);

  EXPECT_EQ(host_data, host_repeated_data);
  EXPECT_EQ(host_labels, host_repeated_labels);
  std::vector<int> shuffled_counts(n_clusters, 0);
  std::vector<int> unshuffled_counts(n_clusters, 0);
  bool differs_from_cyclic = false;
  for (int row = 0; row < n_rows; ++row) {
    ASSERT_GE(host_labels[row], 0);
    ASSERT_LT(host_labels[row], n_clusters);
    EXPECT_EQ(host_unshuffled_labels[row], row % n_clusters);
    ++shuffled_counts[host_labels[row]];
    ++unshuffled_counts[host_unshuffled_labels[row]];
    if (row + n_clusters < n_rows) {
      differs_from_cyclic |= host_labels[row] != host_labels[row + n_clusters];
    }
  }
  EXPECT_EQ(shuffled_counts, unshuffled_counts);
  EXPECT_TRUE(differs_from_cyclic);
}

}  // end namespace random
}  // end namespace raft
