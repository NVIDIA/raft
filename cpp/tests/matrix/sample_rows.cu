/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/dry_run_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/sample_rows.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

#include <unordered_set>

namespace raft {
namespace matrix {

struct inputs {
  int N;
  int dim;
  int dim_margin;
  int n_samples;
  bool host;
};

::std::ostream& operator<<(::std::ostream& os, const inputs p)
{
  os << p.N << "#" << p.dim << "#" << p.dim_margin << "#" << p.n_samples
     << (p.host ? "#host" : "#device");
  return os;
}

template <typename T>
class SampleRowsTest : public ::testing::TestWithParam<inputs> {
 public:
  SampleRowsTest()
    : params(::testing::TestWithParam<inputs>::GetParam()),
      ld(params.dim + params.dim_margin),
      stream(resource::get_cuda_stream(res)),
      state{137ULL},
      in(make_device_matrix<T, int64_t>(res, params.N, ld)),
      out(make_device_matrix<T, int64_t>(res, 0, 0)),
      in_h(make_host_matrix<T, int64_t>(res, params.N, ld)),
      out_h(make_host_matrix<T, int64_t>(res, params.n_samples, params.dim))
  {
    raft::random::uniform(res, state, in.data_handle(), in.size(), T(-1.0), T(1.0));
    for (int64_t i = 0; i < params.N; i++) {
      for (int64_t k = 0; k < ld; k++)
        in_h(i, k) = i * 1000 + k;
    }
    raft::copy(in.data_handle(), in_h.data_handle(), in_h.size(), stream);
  }

  void check()
  {
    if (params.dim_margin == 0) {
      if (params.host) {
        out = raft::matrix::sample_rows<T, int64_t>(
          res, state, make_const_mdspan(in_h.view()), (int64_t)params.n_samples);
      } else {
        out = raft::matrix::sample_rows<T, int64_t>(
          res, state, make_const_mdspan(in.view()), (int64_t)params.n_samples);
      }
    } else {
      // Test for strided matrix view input
      if (params.host) {
        auto in_strided_matrix_view = raft::make_host_strided_matrix_view<const T, int64_t>(
          in_h.data_handle(), in_h.extent(0), params.dim, ld);
        out = raft::matrix::sample_rows<T, int64_t>(
          res, state, in_strided_matrix_view, (int64_t)params.n_samples);
      } else {
        auto in_strided_matrix_view = raft::make_device_strided_matrix_view<const T, int64_t>(
          in.data_handle(), in.extent(0), params.dim, ld);
        out = raft::matrix::sample_rows<T, int64_t>(
          res, state, in_strided_matrix_view, (int64_t)params.n_samples);
      }
    }

    raft::copy(out_h.data_handle(), out.data_handle(), out.size(), stream);
    resource::sync_stream(res, stream);

    ASSERT_TRUE(out.extent(0) == params.n_samples);
    ASSERT_TRUE(out.extent(1) == params.dim);

    std::unordered_set<int> occurrence;

    for (int64_t i = 0; i < params.n_samples; ++i) {
      T val = out_h(i, 0) / 1000;
      ASSERT_TRUE(0 <= val && val < params.N)
        << "out-of-range index @i=" << i << " val=" << val << " params=" << params;
      EXPECT_TRUE(occurrence.find(val) == occurrence.end())
        << "repeated index @i=" << i << " idx=" << val << " params=" << params;
      occurrence.insert(val);
      for (int64_t k = 0; k < params.dim; k++) {
        ASSERT_TRUE(raft::match(out_h(i, k), val * 1000 + k, raft::CompareApprox<T>(1e-6)));
      }
    }
  }

 protected:
  inputs params;
  int ld;
  raft::resources res;
  cudaStream_t stream;
  random::RngState state;
  device_matrix<T, int64_t> in, out;
  host_matrix<T, int64_t> in_h, out_h;
};

inline std::vector<inputs> generate_inputs()
{
  std::vector<inputs> input1 =
    raft::util::itertools::product<inputs>({10}, {1, 17, 96}, {0, 3, 190}, {1, 6, 9, 10}, {false});

  std::vector<inputs> input2 = raft::util::itertools::product<inputs>(
    {137}, {1, 17, 128}, {0, 3, 190}, {1, 10, 100, 137}, {false});
  input1.insert(input1.end(), input2.begin(), input2.end());

  input2 = raft::util::itertools::product<inputs>(
    {100000}, {1, 42}, {0, 1, 190}, {1, 137, 1000, 10000, 50000, 62000, 100000}, {false});

  input1.insert(input1.end(), input2.begin(), input2.end());

  int n = input1.size();
  // Add same tests for host data
  for (int i = 0; i < n; i++) {
    inputs x = input1[i];
    x.host   = true;
    input1.push_back(x);
  }
  return input1;
}

const std::vector<inputs> inputs1 = generate_inputs();

using SampleRowsTestInt64 = SampleRowsTest<float>;
TEST_P(SampleRowsTestInt64, SamplingTest) { check(); }
INSTANTIATE_TEST_SUITE_P(SampleRowsTests, SampleRowsTestInt64, ::testing::ValuesIn(inputs1));

// ===== Dry-run tests =====

TEST(SampleRowsDryRun, VoidOverload)
{
  raft::resources res;
  constexpr int64_t n_rows = 1000, n_cols = 64, n_samples = 100;

  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, n_cols);
  auto output  = raft::make_device_matrix<float, int64_t>(res, n_samples, n_cols);

  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    raft::matrix::sample_rows(handle,
                              raft::random::RngState{42ULL},
                              raft::make_const_mdspan(dataset.view()),
                              output.view());
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device allocation";
}

TEST(SampleRowsDryRun, ReturningOverload)
{
  raft::resources res;
  constexpr int64_t n_rows = 1000, n_cols = 64, n_samples = 100;

  auto dataset = raft::make_device_matrix<float, int64_t>(res, n_rows, n_cols);

  auto stats = raft::util::dry_run_execute(res, [&](raft::resources const& handle) {
    auto result = raft::matrix::sample_rows<float, int64_t>(
      handle, raft::random::RngState{42ULL}, raft::make_const_mdspan(dataset.view()), n_samples);
  });

  EXPECT_FALSE(raft::resource::get_dry_run_flag(res));
  EXPECT_GT(stats.device_global, 0) << "Expected non-zero peak device allocation";
}

}  // namespace matrix
}  // namespace raft
