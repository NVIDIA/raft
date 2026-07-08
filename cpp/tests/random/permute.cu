/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace raft {
namespace random {

template <typename T>
struct PermInputs {
  int N, D;
  bool needPerms, needShuffle, rowMajor;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const PermInputs<T>& dims)
{
  return os;
}

template <typename T>
class PermTest : public ::testing::TestWithParam<PermInputs<T>> {
 public:
  using test_data_type = T;

 protected:
  PermTest()
    : in(0, resource::get_cuda_stream(handle)),
      out(0, resource::get_cuda_stream(handle)),
      outPerms(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    auto stream = resource::get_cuda_stream(handle);
    params      = ::testing::TestWithParam<PermInputs<T>>::GetParam();
    // forcefully set needPerms, since we need it for unit-testing!
    if (params.needShuffle) { params.needPerms = true; }
    raft::random::RngState r(params.seed);
    int N   = params.N;
    int D   = params.D;
    int len = N * D;
    if (params.needPerms) {
      outPerms.resize(N, stream);
      outPerms_ptr = outPerms.data();
    }
    if (params.needShuffle) {
      in.resize(len, stream);
      out.resize(len, stream);
      in_ptr  = in.data();
      out_ptr = out.data();
      uniform(handle, r, in_ptr, len, T(-1.0), T(1.0));
    }
    permute(outPerms_ptr, out_ptr, in_ptr, D, N, params.rowMajor, stream, params.seed);
    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  PermInputs<T> params;
  rmm::device_uvector<T> in, out;
  T* in_ptr  = nullptr;
  T* out_ptr = nullptr;
  rmm::device_uvector<int> outPerms;
  int* outPerms_ptr = nullptr;
};

template <typename T>
class PermMdspanTest : public ::testing::TestWithParam<PermInputs<T>> {
 public:
  using test_data_type = T;

 protected:
  PermMdspanTest()
    : in(0, resource::get_cuda_stream(handle)),
      out(0, resource::get_cuda_stream(handle)),
      outPerms(0, resource::get_cuda_stream(handle))
  {
  }

 private:
  using index_type = int;

  template <class ElementType, class Layout>
  using matrix_view_t = raft::device_matrix_view<ElementType, index_type, Layout>;

  template <class ElementType>
  using vector_view_t = raft::device_vector_view<ElementType, index_type>;

 protected:
  void SetUp() override
  {
    auto stream = resource::get_cuda_stream(handle);
    params      = ::testing::TestWithParam<PermInputs<T>>::GetParam();
    // forcefully set needPerms, since we need it for unit-testing!
    if (params.needShuffle) { params.needPerms = true; }
    raft::random::RngState r(params.seed);
    int N   = params.N;
    int D   = params.D;
    int len = N * D;
    if (params.needPerms) {
      outPerms.resize(N, stream);
      outPerms_ptr = outPerms.data();
    }
    if (params.needShuffle) {
      in.resize(len, stream);
      out.resize(len, stream);
      in_ptr  = in.data();
      out_ptr = out.data();
      uniform(handle, r, in_ptr, len, T(-1.0), T(1.0));
    }

    auto set_up_views_and_test = [&](auto layout) {
      using layout_type = std::decay_t<decltype(layout)>;

      matrix_view_t<const T, layout_type> in_view(in_ptr, N, D);
      std::optional<matrix_view_t<T, layout_type>> out_view;
      if (out_ptr != nullptr) { out_view.emplace(out_ptr, N, D); }
      std::optional<vector_view_t<index_type>> outPerms_view;
      if (outPerms_ptr != nullptr) { outPerms_view.emplace(outPerms_ptr, N); }

      permute(handle, in_view, outPerms_view, out_view, params.seed);

      // None of these three permute calls should have an effect.
      // The point is to test whether the function can deduce the
      // element type of outPerms if given nullopt.
      std::optional<matrix_view_t<T, layout_type>> out_view_empty;
      std::optional<vector_view_t<index_type>> outPerms_view_empty;
      permute(handle, in_view, std::nullopt, out_view_empty, params.seed);
      permute(handle, in_view, outPerms_view_empty, std::nullopt, params.seed);
      permute(handle, in_view, std::nullopt, std::nullopt, params.seed);
    };

    if (params.rowMajor) {
      set_up_views_and_test(raft::row_major{});
    } else {
      set_up_views_and_test(raft::col_major{});
    }

    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  PermInputs<T> params;
  rmm::device_uvector<T> in, out;
  T* in_ptr  = nullptr;
  T* out_ptr = nullptr;
  rmm::device_uvector<int> outPerms;
  int* outPerms_ptr = nullptr;
};

template <typename T, typename L>
::testing::AssertionResult devArrMatchRange(
  const T* actual, size_t size, T start, L eq_compare, bool doSort = true, cudaStream_t stream = 0)
{
  std::vector<T> act_h(size);
  raft::update_host<T>(&(act_h[0]), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  if (doSort) std::sort(act_h.begin(), act_h.end());
  for (size_t i(0); i < size; ++i) {
    auto act      = act_h[i];
    auto expected = start + i;
    if (!eq_compare(expected, act)) {
      return ::testing::AssertionFailure()
             << "actual=" << act << " != expected=" << expected << " @" << i;
    }
  }
  return ::testing::AssertionSuccess();
}

template <typename T, typename L>
::testing::AssertionResult devArrMatchShuffle(const int* perms,
                                              const T* out,
                                              const T* in,
                                              int D,
                                              int N,
                                              bool rowMajor,
                                              L eq_compare,
                                              cudaStream_t stream = 0)
{
  std::vector<int> h_perms(N);
  raft::update_host<int>(&(h_perms[0]), perms, N, stream);
  std::vector<T> h_out(N * D), h_in(N * D);
  raft::update_host<T>(&(h_out[0]), out, N * D, stream);
  raft::update_host<T>(&(h_in[0]), in, N * D, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < D; ++j) {
      int outPos    = rowMajor ? i * D + j : j * N + i;
      int inPos     = rowMajor ? h_perms[i] * D + j : j * N + h_perms[i];
      auto act      = h_out[outPos];
      auto expected = h_in[inPos];
      if (!eq_compare(expected, act)) {
        return ::testing::AssertionFailure()
               << "actual=" << act << " != expected=" << expected << " @" << i << ", " << j;
      }
    }
  }
  return ::testing::AssertionSuccess();
}

const std::vector<PermInputs<float>> inputsf = {
  // only generate permutations
  {32, 8, true, false, true, 1234ULL},
  {32, 8, true, false, true, 1234567890ULL},
  {1024, 32, true, false, true, 1234ULL},
  {1024, 32, true, false, true, 1234567890ULL},
  {2 * 1024, 32, true, false, true, 1234ULL},
  {2 * 1024, 32, true, false, true, 1234567890ULL},
  {2 * 1024 + 500, 32, true, false, true, 1234ULL},
  {2 * 1024 + 500, 32, true, false, true, 1234567890ULL},
  {100000, 32, true, false, true, 1234ULL},
  {100000, 32, true, false, true, 1234567890ULL},
  {100001, 33, true, false, true, 1234567890ULL},
  // permute and shuffle the data row major
  {32, 8, true, true, true, 1234ULL},
  {32, 8, true, true, true, 1234567890ULL},
  {1024, 32, true, true, true, 1234ULL},
  {1024, 32, true, true, true, 1234567890ULL},
  {2 * 1024, 32, true, true, true, 1234ULL},
  {2 * 1024, 32, true, true, true, 1234567890ULL},
  {2 * 1024 + 500, 32, true, true, true, 1234ULL},
  {2 * 1024 + 500, 32, true, true, true, 1234567890ULL},
  {100000, 32, true, true, true, 1234ULL},
  {100000, 32, true, true, true, 1234567890ULL},
  {100001, 31, true, true, true, 1234567890ULL},
  // permute and shuffle the data column major
  {32, 8, true, true, false, 1234ULL},
  {32, 8, true, true, false, 1234567890ULL},
  {1024, 32, true, true, false, 1234ULL},
  {1024, 32, true, true, false, 1234567890ULL},
  {2 * 1024, 32, true, true, false, 1234ULL},
  {2 * 1024, 32, true, true, false, 1234567890ULL},
  {2 * 1024 + 500, 32, true, true, false, 1234ULL},
  {2 * 1024 + 500, 32, true, true, false, 1234567890ULL},
  {100000, 32, true, true, false, 1234ULL},
  {100000, 32, true, true, false, 1234567890ULL},
  {100001, 33, true, true, false, 1234567890ULL}};

#define _PERMTEST_BODY(DATA_TYPE)                                                     \
  do {                                                                                \
    if (params.needPerms) {                                                           \
      ASSERT_TRUE(devArrMatchRange(outPerms_ptr, params.N, 0, raft::Compare<int>())); \
    }                                                                                 \
    if (params.needShuffle) {                                                         \
      ASSERT_TRUE(devArrMatchShuffle(outPerms_ptr,                                    \
                                     out_ptr,                                         \
                                     in_ptr,                                          \
                                     params.D,                                        \
                                     params.N,                                        \
                                     params.rowMajor,                                 \
                                     raft::Compare<DATA_TYPE>()));                    \
    }                                                                                 \
  } while (false)

using PermTestF = PermTest<float>;
TEST_P(PermTestF, Result)
{
  using test_data_type = PermTestF::test_data_type;
  _PERMTEST_BODY(test_data_type);
}
INSTANTIATE_TEST_CASE_P(PermTests, PermTestF, ::testing::ValuesIn(inputsf));

using PermMdspanTestF = PermMdspanTest<float>;
TEST_P(PermMdspanTestF, Result)
{
  using test_data_type = PermTestF::test_data_type;
  _PERMTEST_BODY(test_data_type);
}
INSTANTIATE_TEST_CASE_P(PermMdspanTests, PermMdspanTestF, ::testing::ValuesIn(inputsf));

const std::vector<PermInputs<double>> inputsd = {
  // only generate permutations
  {32, 8, true, false, true, 1234ULL},
  {32, 8, true, false, true, 1234567890ULL},
  {1024, 32, true, false, true, 1234ULL},
  {1024, 32, true, false, true, 1234567890ULL},
  {2 * 1024, 32, true, false, true, 1234ULL},
  {2 * 1024, 32, true, false, true, 1234567890ULL},
  {2 * 1024 + 500, 32, true, false, true, 1234ULL},
  {2 * 1024 + 500, 32, true, false, true, 1234567890ULL},
  {100000, 32, true, false, true, 1234ULL},
  {100000, 32, true, false, true, 1234567890ULL},
  {100001, 33, true, false, true, 1234567890ULL},
  // permute and shuffle the data row major
  {32, 8, true, true, true, 1234ULL},
  {32, 8, true, true, true, 1234567890ULL},
  {1024, 32, true, true, true, 1234ULL},
  {1024, 32, true, true, true, 1234567890ULL},
  {2 * 1024, 32, true, true, true, 1234ULL},
  {2 * 1024, 32, true, true, true, 1234567890ULL},
  {2 * 1024 + 500, 32, true, true, true, 1234ULL},
  {2 * 1024 + 500, 32, true, true, true, 1234567890ULL},
  {100000, 32, true, true, true, 1234ULL},
  {100000, 32, true, true, true, 1234567890ULL},
  {100001, 31, true, true, true, 1234567890ULL},
  // permute and shuffle the data column major
  {32, 8, true, true, false, 1234ULL},
  {32, 8, true, true, false, 1234567890ULL},
  {1024, 32, true, true, false, 1234ULL},
  {1024, 32, true, true, false, 1234567890ULL},
  {2 * 1024, 32, true, true, false, 1234ULL},
  {2 * 1024, 32, true, true, false, 1234567890ULL},
  {2 * 1024 + 500, 32, true, true, false, 1234ULL},
  {2 * 1024 + 500, 32, true, true, false, 1234567890ULL},
  {100000, 32, true, true, false, 1234ULL},
  {100000, 32, true, true, false, 1234567890ULL},
  {100001, 33, true, true, false, 1234567890ULL}};

using PermTestD = PermTest<double>;
TEST_P(PermTestD, Result)
{
  using test_data_type = PermTestF::test_data_type;
  _PERMTEST_BODY(test_data_type);
}
INSTANTIATE_TEST_CASE_P(PermTests, PermTestD, ::testing::ValuesIn(inputsd));

using PermMdspanTestD = PermMdspanTest<double>;
TEST_P(PermMdspanTestD, Result)
{
  using test_data_type = PermTestF::test_data_type;
  _PERMTEST_BODY(test_data_type);
}
INSTANTIATE_TEST_CASE_P(PermMdspanTests, PermMdspanTestD, ::testing::ValuesIn(inputsd));

/*
 * Randomness test for the Feistel-based permutation via a chi-square
 * goodness-of-fit test.
 *
 * For a random permutation of [0, N), the consecutive differences
 * d[i] = (perm[i+1] - perm[i]) mod N are approximately uniformly distributed
 * over [0, N). We bin them into B equal-width buckets and compare the observed
 * counts against the flat expected count E = (N-1)/B using Pearson's statistic
 *
 *     chi2 = sum_b (O_b - E)^2 / E.
 *
 * Under the uniform (random) hypothesis chi2 follows a chi-square distribution
 * with dof = B-1, so it concentrates around its mean (dof) with std sqrt(2*dof)
 * -- for B = 1000 that is 999 +/- ~45. Structure in the permutation (e.g. a
 * predictable relationship between adjacent outputs) piles the differences into
 * a few buckets and inflates chi2 far above dof.
 *
 * The key is an explicit, fixed parameter, so the permutation and the resulting
 * statistic are fully deterministic (not flaky). We assert chi2 lands in a
 * generous central band around dof; a well-mixing permutation clears this
 * easily, while a non-uniform one falls outside it.
 */
TEST(PermuteRandomness, ConsecutiveDifferencesChiSquareUniform)
{
  // N is a non-power-of-two (prime) so the permutation also cycle-walks.
  const int N        = 100003;
  const uint64_t key = 1234567890ULL;
  const int B        = 1000;  // number of histogram buckets

  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);
  rmm::device_uvector<int> perms(N, stream);
  // null in/out: only the permutation indices are produced.
  permute<float, int, int>(perms.data(),
                           static_cast<float*>(nullptr),
                           static_cast<const float*>(nullptr),
                           /* D = */ 1,
                           N,
                           /* rowMajor = */ true,
                           stream,
                           key);
  std::vector<int> p(N);
  raft::update_host<int>(p.data(), perms.data(), N, stream);
  resource::sync_stream(handle);

  std::vector<long> hist(B, 0);
  const int total = N - 1;
  for (int i = 0; i + 1 < N; ++i) {
    int d = static_cast<int>(((static_cast<long>(p[i + 1]) - p[i]) % N + N) % N);
    int b = static_cast<int>((static_cast<long>(d) * B) / N);  // map [0, N) -> [0, B)
    if (b >= B) b = B - 1;
    hist[b]++;
  }

  const double expected = static_cast<double>(total) / B;
  double chi2           = 0.0;
  for (int b = 0; b < B; ++b) {
    double diff = static_cast<double>(hist[b]) - expected;
    chi2 += diff * diff / expected;
  }

  const double dof = B - 1;  // expected mean of chi2 under uniformity
  // Generous central band (mean +/- ~11 std) to demonstrate uniformity without
  // being flaky. A random permutation sits near dof; structure pushes chi2 high.
  EXPECT_GT(chi2, 0.5 * dof) << "chi2=" << chi2 << " suspiciously low (dof=" << dof << ")";
  EXPECT_LT(chi2, 1.5 * dof) << "chi2=" << chi2 << " too high -- consecutive differences are "
                             << "not uniform, permutation is not random (dof=" << dof << ")";
}

}  // end namespace random
}  // end namespace raft
