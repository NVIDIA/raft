/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>
#include <raft/solver/nnls.cuh>

#include <cstdint>
#include <optional>
#include <sstream>
#include <vector>

namespace raft::bench::solver {

struct nnls_input {
  int m, n, P;
  bool masked;
};

inline auto operator<<(std::ostream& os, const nnls_input& p) -> std::ostream&
{
  os << p.m << "x" << p.n << "#P=" << p.P << (p.masked ? "#masked" : "");
  return os;
}

/**
 * Signature-attribution style problems: A is a non-negative (m x n) matrix, every b_p is a
 * sparse non-negative combination of its columns plus noise (about a quarter of the weights
 * are non-zero), and, when `masked`, a random 75% of the columns are available to each
 * problem.  Only the solve is timed: A and B are generated once, the Gram matrix and the
 * projections are part of the solver call.
 */
template <typename T>
struct nnls_bench : public fixture {
  explicit nnls_bench(const nnls_input& p)
    : params(p),
      A(raft::make_device_matrix<T, std::int64_t, raft::col_major>(handle, p.m, p.n)),
      B(raft::make_device_matrix<T, std::int64_t, raft::col_major>(handle, p.m, p.P)),
      X(raft::make_device_matrix<T, std::int64_t, raft::col_major>(handle, p.n, p.P)),
      masks(raft::make_device_matrix<std::uint8_t, std::int64_t, raft::col_major>(
        handle, p.n, p.masked ? p.P : 0))
  {
    generate();
  }

  // (a separate method: extended device lambdas are not allowed in constructors)
  void generate()
  {
    const nnls_input& p = params;
    raft::random::RngState rng{1234};
    auto W = raft::make_device_matrix<T, std::int64_t, raft::col_major>(handle, p.n, p.P);
    raft::random::uniform(handle, rng, A.data_handle(), A.size(), T(0), T(1));
    raft::random::uniform(handle, rng, W.data_handle(), W.size(), T(0), T(1));
    raft::random::uniform(handle, rng, B.data_handle(), B.size(), T(0), T(0.02));
    // keep the top quarter of the weights, so the solutions are sparse
    raft::linalg::map(
      handle,
      W.view(),
      [] __device__(T w) { return w > T(0.75) ? (w - T(0.75)) * T(4) : T(0); },
      raft::make_const_mdspan(W.view()));
    T one = 1;
    // B = A W + noise
    raft::linalg::gemm(handle,
                       raft::make_const_mdspan(A.view()),
                       raft::make_const_mdspan(W.view()),
                       B.view(),
                       std::make_optional(raft::make_host_scalar_view(&one)),
                       std::make_optional(raft::make_host_scalar_view(&one)));
    if (p.masked) {
      auto u = raft::make_device_matrix<T, std::int64_t, raft::col_major>(handle, p.n, p.P);
      raft::random::uniform(handle, rng, u.data_handle(), u.size(), T(0), T(1));
      raft::linalg::map(
        handle,
        masks.view(),
        [] __device__(T v) { return static_cast<std::uint8_t>(v < T(0.75)); },
        raft::make_const_mdspan(u.view()));
    }
    handle.sync_stream();
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    std::optional<raft::device_matrix_view<const std::uint8_t, std::int64_t, raft::col_major>> mk;
    if (params.masked) mk = raft::make_const_mdspan(masks.view());
    loop_on_state(state, [&]() {
      raft::solver::nnls<T>(handle,
                            {},
                            raft::make_const_mdspan(A.view()),
                            raft::make_const_mdspan(B.view()),
                            X.view(),
                            mk);
    });
  }

 private:
  nnls_input params;
  raft::device_matrix<T, std::int64_t, raft::col_major> A, B, X;
  raft::device_matrix<std::uint8_t, std::int64_t, raft::col_major> masks;
};

// Mutation-signature tables (m channels x n signatures) with a batch of 2000
// masked samples, a few batch sizes at the middle shapes, and a bit more than production:
// wider n (up to 2048) and a fat square problem.
const std::vector<nnls_input> nnls_inputs = {
  {32, 10, 2000, true},   {48, 10, 2000, true},    {1536, 10, 2000, true},   {4608, 10, 2000, true},
  {77, 11, 2000, true},   {288, 12, 2000, true},   {82, 17, 2000, true},     {96, 29, 2000, true},
  {192, 56, 2000, true},  {95, 66, 2000, true},    {96, 29, 100, true},      {96, 29, 550, true},
  {192, 56, 550, true},   {96, 29, 2000, false},   {95, 66, 2000, false},    {96, 128, 1000, false},
  {96, 512, 1000, false}, {96, 2048, 1000, false}, {1024, 1024, 100, false},
};

RAFT_BENCH_REGISTER(nnls_bench<float>, "", nnls_inputs);
RAFT_BENCH_REGISTER(nnls_bench<double>, "", nnls_inputs);

}  // namespace raft::bench::solver
