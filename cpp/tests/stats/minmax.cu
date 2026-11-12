/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/minmax.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace raft {
namespace stats {

///@todo: need to add tests for verifying the column subsampling feature

template <typename T>
struct MinMaxInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MinMaxInputs<T>& dims)
{
  return os;
}

template <typename T>
RAFT_KERNEL naiveMinMaxInitKernel(int ncols, T* globalmin, T* globalmax, T init_val)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= ncols) return;
  globalmin[tid] = init_val;
  globalmax[tid] = -init_val;
}

template <typename T>
RAFT_KERNEL naiveMinMaxKernel(const T* data, int nrows, int ncols, T* globalmin, T* globalmax)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int col = tid / nrows;
  if (col < ncols) {
    T val = data[tid];
    if (!isnan(val)) {
      raft::myAtomicMin(&globalmin[col], val);
      raft::myAtomicMax(&globalmax[col], val);
    }
  }
}

template <typename T>
void naiveMinMax(
  const T* data, int nrows, int ncols, T* globalmin, T* globalmax, cudaStream_t stream)
{
  const int TPB = 128;
  int nblks     = raft::ceildiv(ncols, TPB);
  T init_val    = std::numeric_limits<T>::max();
  raft::launch_kernel(
    stream, nblks, TPB, naiveMinMaxInitKernel, ncols, globalmin, globalmax, init_val);
  nblks = raft::ceildiv(nrows * ncols, TPB);
  raft::launch_kernel(
    stream, nblks, TPB, naiveMinMaxKernel, data, nrows, ncols, globalmin, globalmax);
}

template <typename T>
RAFT_KERNEL nanKernel(T* data, const bool* mask, int len, T nan)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= len) return;
  if (!mask[tid]) data[tid] = nan;
}

template <typename T>
class MinMaxTest : public ::testing::TestWithParam<MinMaxInputs<T>> {
 protected:
  MinMaxTest()
    : minmax_act(0, resource::get_cuda_stream(handle)),
      minmax_ref(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    auto stream = resource::get_cuda_stream(handle).get();
    params      = ::testing::TestWithParam<MinMaxInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    int len = params.rows * params.cols;

    rmm::device_uvector<T> data(len, stream);
    rmm::device_uvector<bool> mask(len, stream);
    minmax_act.resize(2 * params.cols, stream);
    minmax_ref.resize(2 * params.cols, stream);

    normal(handle, r, data.data(), len, (T)0.0, (T)1.0);
    T nan_prob = 0.01;
    bernoulli(handle, r, mask.data(), len, nan_prob);
    const int TPB = 256;
    raft::launch_kernel(handle,
                        raft::ceildiv(len, TPB),
                        TPB,
                        nanKernel<T>,
                        data.data(),
                        mask.data(),
                        len,
                        std::numeric_limits<T>::quiet_NaN());
    naiveMinMax(data.data(),
                params.rows,
                params.cols,
                minmax_ref.data(),
                minmax_ref.data() + params.cols,
                stream);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        raft::stats::minmax<T, int>(
          h,
          raft::make_device_matrix_view<const T, int, raft::layout_f_contiguous>(
            data.data(), params.rows, params.cols),
          std::nullopt,
          std::nullopt,
          raft::make_device_vector_view<T, int>(minmax_act.data(), params.cols),
          raft::make_device_vector_view<T, int>(minmax_act.data() + params.cols, params.cols),
          std::nullopt);
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
  }

 protected:
  raft::resources handle;
  MinMaxInputs<T> params;
  rmm::device_uvector<T> minmax_act;
  rmm::device_uvector<T> minmax_ref;
};

const std::vector<MinMaxInputs<float>> inputsf = {
  {0.00001f, 1024, 32, 1234ULL},   {0.00001f, 1024, 64, 1234ULL},  {0.00001f, 1024, 128, 1234ULL},
  {0.00001f, 1024, 256, 1234ULL},  {0.00001f, 1024, 512, 1234ULL}, {0.00001f, 1024, 1024, 1234ULL},
  {0.00001f, 4096, 32, 1234ULL},   {0.00001f, 4096, 64, 1234ULL},  {0.00001f, 4096, 128, 1234ULL},
  {0.00001f, 4096, 256, 1234ULL},  {0.00001f, 4096, 512, 1234ULL}, {0.00001f, 4096, 1024, 1234ULL},
  {0.00001f, 8192, 32, 1234ULL},   {0.00001f, 8192, 64, 1234ULL},  {0.00001f, 8192, 128, 1234ULL},
  {0.00001f, 8192, 256, 1234ULL},  {0.00001f, 8192, 512, 1234ULL}, {0.00001f, 8192, 1024, 1234ULL},
  {0.00001f, 1024, 8192, 1234ULL}, {0.00001f, 1023, 5, 1234ULL},   {0.00001f, 1025, 30, 1234ULL},
  {0.00001f, 2047, 65, 1234ULL},   {0.00001f, 2049, 22, 1234ULL},  {0.00001f, 31, 644, 1234ULL},
  {0.00001f, 33, 999, 1234ULL},    {0.00001f, 1, 1, 1234ULL},      {0.00001f, 7, 23, 1234ULL},
  {0.00001f, 17, 5, 1234ULL}};

const std::vector<MinMaxInputs<double>> inputsd = {
  {0.0000001, 1024, 32, 1234ULL},   {0.0000001, 1024, 64, 1234ULL},
  {0.0000001, 1024, 128, 1234ULL},  {0.0000001, 1024, 256, 1234ULL},
  {0.0000001, 1024, 512, 1234ULL},  {0.0000001, 1024, 1024, 1234ULL},
  {0.0000001, 4096, 32, 1234ULL},   {0.0000001, 4096, 64, 1234ULL},
  {0.0000001, 4096, 128, 1234ULL},  {0.0000001, 4096, 256, 1234ULL},
  {0.0000001, 4096, 512, 1234ULL},  {0.0000001, 4096, 1024, 1234ULL},
  {0.0000001, 8192, 32, 1234ULL},   {0.0000001, 8192, 64, 1234ULL},
  {0.0000001, 8192, 128, 1234ULL},  {0.0000001, 8192, 256, 1234ULL},
  {0.0000001, 8192, 512, 1234ULL},  {0.0000001, 8192, 1024, 1234ULL},
  {0.0000001, 1024, 8192, 1234ULL}, {0.0000001, 1023, 5, 1234ULL},
  {0.0000001, 1025, 30, 1234ULL},   {0.0000001, 2047, 65, 1234ULL},
  {0.0000001, 2049, 22, 1234ULL},   {0.0000001, 31, 644, 1234ULL},
  {0.0000001, 33, 999, 1234ULL},    {0.0000001, 1, 1, 1234ULL},
  {0.0000001, 7, 23, 1234ULL},      {0.0000001, 17, 5, 1234ULL}};

typedef MinMaxTest<float> MinMaxTestF;
TEST_P(MinMaxTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(minmax_ref.data(),
                                minmax_act.data(),
                                2 * params.cols,
                                raft::CompareApprox<float>(params.tolerance)));
}

typedef MinMaxTest<double> MinMaxTestD;
TEST_P(MinMaxTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(minmax_ref.data(),
                                minmax_act.data(),
                                2 * params.cols,
                                raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MinMaxTests, MinMaxTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MinMaxTests, MinMaxTestD, ::testing::ValuesIn(inputsd));

// The sampled columns must be written in the same column-major layout no matter how many
// shared-memory batches the kernel needs, so use more columns than fit in one batch.
template <typename T>
void minmax_sampledcols_test(bool use_colids)
{
  raft::resources handle;
  auto stream     = resource::get_cuda_stream(handle).get();
  const int nrows = 3;
  const int ncols = raft::getSharedMemPerBlock() / int(2 * sizeof(T)) + 7;

  std::vector<T> data_h(size_t(nrows) * ncols);
  // Distinct, exactly representable values, so a misplaced element is always detected.
  for (size_t i = 0; i < data_h.size(); ++i) {
    data_h[i] = T(int64_t(i) - int64_t(data_h.size() / 2));
  }
  std::vector<unsigned> colids_h(ncols);
  for (int c = 0; c < ncols; ++c) {
    colids_h[c] = use_colids ? unsigned(ncols - 1 - c) : unsigned(c);
  }
  std::vector<T> sampled_ref(data_h.size()), min_ref(ncols), max_ref(ncols);
  for (int c = 0; c < ncols; ++c) {
    min_ref[c] = std::numeric_limits<T>::max();
    max_ref[c] = std::numeric_limits<T>::lowest();
    for (int r = 0; r < nrows; ++r) {
      T v                                = data_h[r + size_t(colids_h[c]) * nrows];
      sampled_ref[r + size_t(c) * nrows] = v;
      min_ref[c]                         = std::min(min_ref[c], v);
      max_ref[c]                         = std::max(max_ref[c], v);
    }
  }

  rmm::device_uvector<T> data(data_h.size(), stream);
  rmm::device_uvector<unsigned> colids(ncols, stream);
  rmm::device_uvector<T> globalmin(ncols, stream);
  rmm::device_uvector<T> globalmax(ncols, stream);
  rmm::device_uvector<T> sampled(data_h.size(), stream);
  raft::update_device(data.data(), data_h.data(), data_h.size(), stream);
  raft::update_device(colids.data(), colids_h.data(), ncols, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(sampled.data(), 0, sampled.size() * sizeof(T), stream));

  std::optional<raft::device_vector_view<const unsigned, int>> colids_view = std::nullopt;
  if (use_colids) {
    colids_view = raft::make_device_vector_view<const unsigned, int>(colids.data(), ncols);
  }
  raft::stats::minmax<T, int>(
    handle,
    raft::make_device_matrix_view<const T, int, raft::layout_f_contiguous>(
      data.data(), nrows, ncols),
    std::nullopt,
    colids_view,
    raft::make_device_vector_view<T, int>(globalmin.data(), ncols),
    raft::make_device_vector_view<T, int>(globalmax.data(), ncols),
    std::make_optional(raft::make_device_vector_view<T, int>(sampled.data(), nrows * ncols)));

  ASSERT_TRUE(raft::devArrMatchHost(
    sampled_ref.data(), sampled.data(), sampled.size(), raft::Compare<T>(), stream));
  ASSERT_TRUE(
    raft::devArrMatchHost(min_ref.data(), globalmin.data(), ncols, raft::Compare<T>(), stream));
  ASSERT_TRUE(
    raft::devArrMatchHost(max_ref.data(), globalmax.data(), ncols, raft::Compare<T>(), stream));
}

TEST(MinMaxSampledCols, FloatMultipleBatches) { minmax_sampledcols_test<float>(false); }
TEST(MinMaxSampledCols, FloatMultipleBatchesColids) { minmax_sampledcols_test<float>(true); }
TEST(MinMaxSampledCols, DoubleMultipleBatches) { minmax_sampledcols_test<double>(false); }
TEST(MinMaxSampledCols, DoubleMultipleBatchesColids) { minmax_sampledcols_test<double>(true); }

}  // end namespace stats
}  // end namespace raft
