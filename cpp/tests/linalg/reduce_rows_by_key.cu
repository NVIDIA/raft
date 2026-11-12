/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>

#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

namespace raft {
namespace linalg {

template <typename Type>
RAFT_KERNEL naiveReduceRowsByKeyKernel(const Type* d_A,
                                       int lda,
                                       uint32_t* d_keys,
                                       const Type* d_weight,
                                       char* d_char_keys,
                                       int nrows,
                                       int ncols,
                                       int nkeys,
                                       Type* d_sums)
{
  int c = threadIdx.x + blockIdx.x * blockDim.x;
  if (c >= ncols) return;
  int this_key = threadIdx.y + blockIdx.y * blockDim.y;

  Type sum = 0.0;
  for (int r = 0; r < nrows; r++) {
    if (this_key != d_keys[r]) continue;
    Type wt = 1;
    if (d_weight) wt = d_weight[r];
    sum += d_A[lda * r + c] * wt;
  }
  d_sums[this_key * ncols + c] = sum;
}
template <typename Type>
void naiveReduceRowsByKey(const Type* d_A,
                          int lda,
                          uint32_t* d_keys,
                          const Type* d_weight,
                          char* d_char_keys,
                          int nrows,
                          int ncols,
                          int nkeys,
                          Type* d_sums,
                          cudaStream_t stream)
{
  cudaMemset(d_sums, 0, sizeof(Type) * nkeys * ncols);

  raft::launch_kernel(stream,
                      dim3((ncols + 31) / 32, nkeys),
                      dim3(32, 1),
                      naiveReduceRowsByKeyKernel,
                      d_A,
                      lda,
                      d_keys,
                      d_weight,
                      d_char_keys,
                      nrows,
                      ncols,
                      nkeys,
                      d_sums);
}

template <typename T>
struct ReduceRowsInputs {
  T tolerance;
  int nobs;
  uint32_t cols;
  uint32_t nkeys;
  unsigned long long int seed;
  bool weighted;
  T max_weight;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const ReduceRowsInputs<T>& dims)
{
  return os;
}

template <typename T>
class ReduceRowTest : public ::testing::TestWithParam<ReduceRowsInputs<T>> {
 public:
  ReduceRowTest()
    : params(::testing::TestWithParam<ReduceRowsInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      in(params.nobs * params.cols, stream),
      out(params.nkeys * params.cols, stream),
      out_ref(params.nkeys * params.cols, stream),
      keys(params.nobs, stream),
      scratch_buf(params.nobs, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    raft::random::RngState r_int(params.seed);

    uint32_t nobs  = params.nobs;
    uint32_t cols  = params.cols;
    uint32_t nkeys = params.nkeys;
    uniform(handle, r, in.data(), nobs * cols, T(0.0), T(2.0 / nobs));
    uniformInt(handle, r_int, keys.data(), nobs, (uint32_t)0, nkeys);

    rmm::device_uvector<T> weight(0, stream);
    if (params.weighted) {
      weight.resize(nobs, stream);
      raft::random::RngState r(params.seed);
      uniform(handle, r, weight.data(), nobs, T(1), params.max_weight);
    }

    naiveReduceRowsByKey(in.data(),
                         cols,
                         keys.data(),
                         params.weighted ? weight.data() : nullptr,
                         scratch_buf.data(),
                         nobs,
                         cols,
                         nkeys,
                         out_ref.data(),
                         stream);
    auto input_view = raft::make_device_matrix_view<const T>(
      in.data(), params.cols, static_cast<uint32_t>(params.nobs));
    auto output_view = raft::make_device_matrix_view(out.data(), params.cols, params.nkeys);
    auto keys_view   = raft::make_device_vector_view<const uint32_t>(
      keys.data(), static_cast<uint32_t>(params.nobs));
    auto scratch_buf_view =
      raft::make_device_vector_view(scratch_buf.data(), static_cast<uint32_t>(params.nobs));
    std::optional<raft::device_vector_view<const T>> weights_view;
    if (params.weighted) {
      weights_view.emplace(weight.data(), static_cast<uint32_t>(params.nobs));
    }

    reduce_rows_by_key(
      handle, input_view, keys_view, output_view, params.nkeys, scratch_buf_view, weights_view);
    resource::sync_stream(handle, stream);
  }

 protected:
  ReduceRowsInputs<T> params;
  raft::resources handle;
  cudaStream_t stream = 0;

  int device_count = 0;
  rmm::device_uvector<T> in, out, out_ref;
  rmm::device_uvector<uint32_t> keys;
  rmm::device_uvector<char> scratch_buf;
};

// ReduceRowTestFloat
// 128 Obs, 32 cols, 6 clusters
const std::vector<ReduceRowsInputs<float>> inputsf2 = {{0.000001f, 128, 32, 6, 1234ULL, false},
                                                       {0.000001f, 128, 32, 6, 1234ULL, true, 1.0},
                                                       {0.000001f, 128, 32, 6, 1234ULL, true, 2.0}};
typedef ReduceRowTest<float> ReduceRowTestF;
TEST_P(ReduceRowTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestF, ::testing::ValuesIn(inputsf2));

// ReduceRowTestDouble
// 128 Obs, 32 cols, 6 clusters, double precision
const std::vector<ReduceRowsInputs<double>> inputsd2 = {
  {0.00000001, 128, 32, 6, 1234ULL, false},
  {0.00000001, 128, 32, 6, 1234ULL, true, 2.0},
  {0.00000001, 128, 32, 6, 1234ULL, true, 8.0}};
typedef ReduceRowTest<double> ReduceRowTestD;
TEST_P(ReduceRowTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests, ReduceRowTestD, ::testing::ValuesIn(inputsd2));

// ReduceRowTestFewKeys
// 128 Obs, 32-128 cols, 3-10 clusters
const std::vector<ReduceRowsInputs<float>> inputsf_small_nkey = {
  {0.000001f, 128, 32, 3, 1234ULL, false},
  {0.000001f, 128, 32, 3, 1234ULL, true, 5.0},
  {0.000001f, 128, 32, 3, 1234ULL, true, 8.0},
  {0.00001f, 10000, 128, 10, 1234ULL, true, 4.0}};  // widest few-keys rows
typedef ReduceRowTest<float> ReduceRowTestFewKeys;
TEST_P(ReduceRowTestFewKeys, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestFewKeys,
                        ::testing::ValuesIn(inputsf_small_nkey));

// ReduceRowTestBigSpace
// 512 Obs, 1024 cols, 32 clusters, double precision
const std::vector<ReduceRowsInputs<double>> inputsd_big_space = {
  {0.00000001, 512, 1024, 40, 1234ULL, false},
  {0.00000001, 512, 1024, 40, 1234ULL, true, 4.0},
  {0.00000001, 512, 1024, 40, 1234ULL, true, 16.0}};
typedef ReduceRowTest<double> ReduceRowTestBigSpace;
TEST_P(ReduceRowTestBigSpace, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestBigSpace,
                        ::testing::ValuesIn(inputsd_big_space));

// ReduceRowTestManyObs
// 100000 Obs, 37 cols, 32 clusters
const std::vector<ReduceRowsInputs<float>> inputsf_many_obs = {
  {0.00001f, 100000, 37, 32, 1234ULL, false},
  {0.00001f, 100000, 37, 32, 1234ULL, true, 4.0},
  {0.00001f, 100000, 37, 32, 1234ULL, true, 16.0}};
typedef ReduceRowTest<float> ReduceRowTestManyObs;
TEST_P(ReduceRowTestManyObs, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestManyObs,
                        ::testing::ValuesIn(inputsf_many_obs));

// ReduceRowTestManyClusters
// 100000 Obs, 37 cols, 2048 clusters
const std::vector<ReduceRowsInputs<float>> inputsf_many_cluster = {
  {0.00001f, 100000, 37, 2048, 1234ULL, false},
  {0.00001f, 100000, 37, 2048, 1234ULL, true, 32.0},
  {0.00001f, 100000, 37, 2048, 1234ULL, true, 16.0}};
typedef ReduceRowTest<float> ReduceRowTestManyClusters;
TEST_P(ReduceRowTestManyClusters, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestManyClusters,
                        ::testing::ValuesIn(inputsf_many_cluster));

// ReduceRowTestLargeClustersAndCols
// 100000 Obs, 5000 cols, 3072 clusters: Should trigger the vectorized kernel on GPU newer than 9.0.
const std::vector<ReduceRowsInputs<float>> inputsf_vectorized = {
  {0.00001f, 100000, 5000, 3072, 1234ULL, false},
  {0.00001f, 100000, 5000, 3072, 1234ULL, true, 32.0},
  {0.00001f, 100000, 5000, 3072, 1234ULL, true, 16.0}};
typedef ReduceRowTest<float> ReduceRowTestLargeClustersAndCols;
TEST_P(ReduceRowTestLargeClustersAndCols, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(),
                                out.data(),
                                params.cols * params.nkeys,
                                raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowTestLargeClustersAndCols,
                        ::testing::ValuesIn(inputsf_vectorized));

struct DeterminismInputs {
  int nrows;
  int ncols;
  int nkeys;
  bool weighted;
  bool heavy_hitter;
};

::std::ostream& operator<<(::std::ostream& os, const DeterminismInputs& p)
{
  return os << "nrows=" << p.nrows << " ncols=" << p.ncols << " nkeys=" << p.nkeys
            << " weighted=" << p.weighted << " heavy_hitter=" << p.heavy_hitter;
}

template <typename T>
class ReduceRowsByKeyDeterminism : public ::testing::TestWithParam<DeterminismInputs> {
 public:
  void run()
  {
    const auto p = GetParam();
    auto stream  = resource::get_cuda_stream(handle).get();
    rmm::device_uvector<T> in(static_cast<size_t>(p.nrows) * p.ncols, stream);
    rmm::device_uvector<T> weights(p.nrows, stream);
    rmm::device_uvector<uint32_t> keys(p.nrows, stream);
    rmm::device_uvector<T> sums(static_cast<size_t>(p.nkeys) * p.ncols, stream);
    rmm::device_uvector<T> ref(static_cast<size_t>(p.nkeys) * p.ncols, stream);
    rmm::device_uvector<char> unused(p.nrows, stream);
    raft::random::RngState r(1234ULL);
    // Mixed signs and magnitudes, so that the summation order shows in the result bits.
    uniform(handle, r, in.data(), in.size(), T(-1e3), T(1e3));
    uniform(handle, r, weights.data(), weights.size(), T(0.5), T(2));
    uniformInt(handle, r, keys.data(), p.nrows, uint32_t{0}, static_cast<uint32_t>(p.nkeys));
    if (p.heavy_hitter) {
      // Every other row has key 0.
      raft::linalg::map_offset(
        handle,
        raft::make_device_vector_view(keys.data(), p.nrows),
        [k = keys.data()] __device__(int i) { return i % 2 == 0 ? 0u : k[i]; });
    }
    const T* w = p.weighted ? weights.data() : nullptr;
    naiveReduceRowsByKey(in.data(),
                         p.ncols,
                         keys.data(),
                         w,
                         unused.data(),
                         p.nrows,
                         p.ncols,
                         p.nkeys,
                         ref.data(),
                         stream);

    std::vector<T> first;
    for (int run = 0; run < 4; ++run) {
      reduce_rows_by_key(in.data(),
                         p.ncols,
                         keys.data(),
                         w,
                         unused.data(),
                         p.nrows,
                         p.ncols,
                         p.nkeys,
                         sums.data(),
                         stream,
                         true);
      std::vector<T> out(sums.size());
      raft::update_host(out.data(), sums.data(), sums.size(), stream);
      resource::sync_stream(handle, stream);
      if (run == 0) {
        first = out;
        // Inputs are up to 1e3 * 2 per row, so the absolute error bound scales with the row count.
        const T tol = T(std::is_same_v<T, float> ? 1e-5 : 1e-12) * T(2e3) * T(p.nrows) / T(p.nkeys);
        ASSERT_TRUE(raft::devArrMatch(ref.data(),
                                      sums.data(),
                                      sums.size(),
                                      raft::CompareApprox<T>(std::max(tol, T(1e-3))),
                                      stream));
      } else {
        ASSERT_EQ(0, std::memcmp(first.data(), out.data(), out.size() * sizeof(T)))
          << "results differ between run 0 and run " << run;
      }
    }
  }

 protected:
  raft::resources handle;
};

// One case per path: privatized (few keys), exact bins (float, <= 16 columns), sorted per-key
// (float, few rows per key), sorted chunks (wide rows, double).
const std::vector<DeterminismInputs> determinism_inputs = {
  {100000, 32, 8, true, false},
  {100000, 32, 8, false, true},
  {200000, 4, 10000, true, false},
  {200000, 16, 20000, false, true},
  {200000, 64, 20000, true, false},
  {200000, 64, 20000, false, true},
  {100000, 512, 100, true, false},
  {100000, 1024, 3000, false, true},
  {300000, 37, 64, true, true},
};

using ReduceRowsByKeyDeterminismF = ReduceRowsByKeyDeterminism<float>;
TEST_P(ReduceRowsByKeyDeterminismF, Result) { run(); }
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowsByKeyDeterminismF,
                        ::testing::ValuesIn(determinism_inputs));

using ReduceRowsByKeyDeterminismD = ReduceRowsByKeyDeterminism<double>;
TEST_P(ReduceRowsByKeyDeterminismD, Result) { run(); }
INSTANTIATE_TEST_CASE_P(ReduceRowTests,
                        ReduceRowsByKeyDeterminismD,
                        ::testing::ValuesIn(determinism_inputs));

TEST(ReduceRowsByKeyDryRun, Allocations)
{
  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle).get();
  // Shapes that take each path: privatized, exact bins, sorted per-key, sorted chunks.
  for (auto [nrows, ncols, nkeys] : {std::tuple{100000, 32, 8},
                                     std::tuple{100000, 4, 10000},
                                     std::tuple{100000, 64, 10000},
                                     std::tuple{100000, 1024, 1024}}) {
    rmm::device_uvector<float> in(static_cast<size_t>(nrows) * ncols, stream);
    rmm::device_uvector<float> sums(static_cast<size_t>(nkeys) * ncols, stream);
    rmm::device_uvector<uint32_t> keys(nrows, stream);
    raft::random::RngState r(1234ULL);
    uniformInt(handle, r, keys.data(), nrows, 0u, static_cast<uint32_t>(nkeys));
    RAFT_CUDA_TRY(cudaMemsetAsync(in.data(), 0, in.size() * sizeof(float), stream));
    auto action = [&](raft::resources const& res) {
      detail::reduce_rows_by_key(in.data(),
                                 ncols,
                                 keys.data(),
                                 static_cast<const float*>(nullptr),
                                 static_cast<char*>(nullptr),
                                 nrows,
                                 ncols,
                                 nkeys,
                                 sums.data(),
                                 resource::get_cuda_stream(res).get(),
                                 true,
                                 resource::get_dry_run_flag(res));
    };
    execute_with_dry_run_check(handle, action, alloc_behavior::ARGUMENT_DRIVEN, 1);
  }
}

}  // end namespace linalg
}  // end namespace raft
