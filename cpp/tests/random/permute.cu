/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
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
    permute(outPerms_ptr, out_ptr, in_ptr, D, N, params.rowMajor, stream);
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

      permute(handle, in_view, outPerms_view, out_view);

      // None of these three permute calls should have an effect.
      // The point is to test whether the function can deduce the
      // element type of outPerms if given nullopt.
      std::optional<matrix_view_t<T, layout_type>> out_view_empty;
      std::optional<vector_view_t<index_type>> outPerms_view_empty;
      permute(handle, in_view, std::nullopt, out_view_empty);
      permute(handle, in_view, outPerms_view_empty, std::nullopt);
      permute(handle, in_view, std::nullopt, std::nullopt);
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

// Determinism test for the seeded `permute` overload (cuML issue #7871:
// make_regression was not reproducible with shuffle=True because permute
// drew its affine coefficients from the unseeded global rand()). The same
// seed must yield identical permutation indices across calls.
template <typename T>
class PermSeedTest : public ::testing::TestWithParam<PermInputs<T>> {
 public:
  using test_data_type = T;

 protected:
  PermSeedTest()
    : in(0, resource::get_cuda_stream(handle)),
      out(0, resource::get_cuda_stream(handle)),
      perms1(0, resource::get_cuda_stream(handle)),
      perms2(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    auto stream = resource::get_cuda_stream(handle);
    params      = ::testing::TestWithParam<PermInputs<T>>::GetParam();
    int N       = params.N;
    int D       = params.D;
    int len     = N * D;
    raft::random::RngState r(params.seed);
    in.resize(len, stream);
    out.resize(len, stream);
    perms1.resize(N, stream);
    perms2.resize(N, stream);
    uniform(handle, r, in.data(), len, T(-1.0), T(1.0));

    // Same seed twice -> the two permutations must be identical.
    permute<T, int, int>(
      perms1.data(), out.data(), in.data(), D, N, params.rowMajor, stream, params.seed);
    permute<T, int, int>(
      perms2.data(), out.data(), in.data(), D, N, params.rowMajor, stream, params.seed);
    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  PermInputs<T> params;
  rmm::device_uvector<T> in, out;
  rmm::device_uvector<int> perms1, perms2;
};

using PermSeedTestF = PermSeedTest<float>;
TEST_P(PermSeedTestF, SameSeedIsDeterministic)
{
  using test_data_type = PermSeedTestF::test_data_type;
  auto stream          = resource::get_cuda_stream(handle);
  int N                = params.N;
  std::vector<int> h1(N), h2(N);
  raft::update_host(h1.data(), perms1.data(), N, stream);
  raft::update_host(h2.data(), perms2.data(), N, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  ASSERT_EQ(h1, h2);

  // Different seeds should produce different permutations. Only checked for
  // larger N, where the space of affine permutations is large enough that a
  // collision is astronomically unlikely.
  if (N >= 1024) {
    rmm::device_uvector<int> perms3(N, stream);
    permute<test_data_type, int, int>(
      perms3.data(), out.data(), in.data(), params.D, N, params.rowMajor, stream, params.seed + 1);
    resource::sync_stream(handle);
    std::vector<int> h3(N);
    raft::update_host(h3.data(), perms3.data(), N, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ASSERT_NE(h1, h3);
  }
}
INSTANTIATE_TEST_CASE_P(PermSeedTests, PermSeedTestF, ::testing::ValuesIn(inputsf));

}  // end namespace random
}  // end namespace raft
