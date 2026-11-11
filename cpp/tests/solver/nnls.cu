/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/solver/nnls.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

namespace raft::solver {
namespace {

using index_t = std::int64_t;

/**
 * Optimality is checked directly instead of against a reference solver.  NNLS is convex, so
 * the KKT conditions
 *
 *   x >= 0,  w = A^T (b - A x),  w_j == 0 where x_j > 0,  w_j <= 0 where x_j == 0
 *
 * are necessary and sufficient.  Both halves matter: checking only `w_j <= 0` on the
 * inactive set passes for a solve that never converged on the active set.  Columns a mask
 * disables take no part (and must come back as exactly zero).
 */
template <typename T>
struct kkt_residual {
  T negativity     = 0;  ///< max(0, -x_j): exactly 0
  T stationarity   = 0;  ///< max |w_j| over x_j > 0, relative to max |A^T b|
  T inactive       = 0;  ///< max(0, w_j) over enabled x_j == 0, relative to max |A^T b|
  T masked_leakage = 0;  ///< max |x_j| over disabled columns: exactly 0
  /** False if any x_j, residual or gradient entry is NaN/inf.  std::max swallows a NaN, so
   *  without this flag a NaN solution would read as zero error. */
  bool finite = true;
};

template <typename T>
T at(const std::vector<T>& v, index_t row, index_t col, index_t ld)
{
  return v[col * ld + row];
}

template <typename T>
kkt_residual<T> kkt(const std::vector<T>& A,
                    const std::vector<T>& B,
                    const std::vector<T>& X,
                    const std::vector<std::uint8_t>& masks,
                    index_t m,
                    index_t n,
                    index_t P)
{
  kkt_residual<T> worst;
  std::vector<double> resid(m), w(n);
  for (index_t p = 0; p < P; ++p) {
    for (index_t i = 0; i < m; ++i) {
      double acc = at(B, i, p, m);
      for (index_t j = 0; j < n; ++j)
        acc -= static_cast<double>(at(A, i, j, m)) * at(X, j, p, n);
      worst.finite = worst.finite && std::isfinite(acc);
      resid[i]     = acc;
    }
    double scale = 0;
    for (index_t j = 0; j < n; ++j) {
      double wj = 0, cj = 0;
      for (index_t i = 0; i < m; ++i) {
        wj += static_cast<double>(at(A, i, j, m)) * resid[i];
        cj += static_cast<double>(at(A, i, j, m)) * at(B, i, p, m);
      }
      worst.finite = worst.finite && std::isfinite(wj);
      w[j]         = wj;
      scale        = std::max(scale, std::abs(cj));
    }
    if (scale == 0) scale = 1;
    for (index_t j = 0; j < n; ++j) {
      const double xj = at(X, j, p, n);
      worst.finite    = worst.finite && std::isfinite(xj);
      const bool on   = masks.empty() || masks[p * n + j] != 0;
      if (!on) {
        worst.masked_leakage = std::max<T>(worst.masked_leakage, std::abs(xj));
        continue;
      }
      worst.negativity = std::max<T>(worst.negativity, std::max(0.0, -xj));
      if (xj > 0) {
        worst.stationarity = std::max<T>(worst.stationarity, std::abs(w[j]) / scale);
      } else {
        worst.inactive = std::max<T>(worst.inactive, std::max(0.0, w[j]) / scale);
      }
    }
  }
  return worst;
}

/** Non-negative design matrix; targets are sparse non-negative combinations plus noise, so the
 *  active set stays well short of n -- the regime of signature attribution. */
template <typename T>
struct problem {
  std::vector<T> A, B;
  std::vector<std::uint8_t> masks;
  index_t m, n, P;

  problem(index_t m_, index_t n_, index_t P_, unsigned seed, bool masked = false)
    : m(m_), n(n_), P(P_)
  {
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    std::uniform_int_distribution<index_t> column(0, n - 1);
    A.resize(m * n);
    for (auto& v : A)
      v = static_cast<T>(std::abs(normal(rng)));
    std::vector<double> W(n * P, 0.0);
    for (index_t p = 0; p < P; ++p)
      for (index_t k = 0; k < std::max<index_t>(1, n / 4); ++k)
        W[p * n + column(rng)] = std::abs(normal(rng));
    B.resize(m * P);
    for (index_t p = 0; p < P; ++p)
      for (index_t i = 0; i < m; ++i) {
        double acc = 0.01 * std::abs(normal(rng));
        for (index_t j = 0; j < n; ++j)
          acc += static_cast<double>(at(A, i, j, m)) * W[p * n + j];
        B[p * m + i] = static_cast<T>(acc);
      }
    if (masked) {
      masks.resize(n * P);
      for (auto& v : masks)
        v = (rng() % 3 != 0) ? 1 : 0;
    }
  }
};

/** Device copies of a problem plus the outputs, and the solver call on them. */
template <typename T>
struct device_problem {
  raft::device_resources const& res;
  problem<T> const& pr;
  raft::device_matrix<T, index_t, raft::col_major> A, B, X, fitted;
  raft::device_matrix<std::uint8_t, index_t, raft::col_major> masks;

  device_problem(raft::device_resources const& r, problem<T> const& p)
    : res(r),
      pr(p),
      A(raft::make_device_matrix<T, index_t, raft::col_major>(r, p.m, p.n)),
      B(raft::make_device_matrix<T, index_t, raft::col_major>(r, p.m, p.P)),
      X(raft::make_device_matrix<T, index_t, raft::col_major>(r, p.n, p.P)),
      fitted(raft::make_device_matrix<T, index_t, raft::col_major>(r, p.m, p.P)),
      masks(raft::make_device_matrix<std::uint8_t, index_t, raft::col_major>(
        r, p.n, p.masks.empty() ? 0 : p.P))
  {
    auto stream = raft::resource::get_cuda_stream(r);
    raft::update_device(A.data_handle(), p.A.data(), p.A.size(), stream);
    raft::update_device(B.data_handle(), p.B.data(), p.B.size(), stream);
    if (!p.masks.empty())
      raft::update_device(masks.data_handle(), p.masks.data(), p.masks.size(), stream);
  }

  void solve(raft::resources const& r, nnls_params params = {}, bool with_fitted = false)
  {
    std::optional<raft::device_matrix_view<const std::uint8_t, index_t, raft::col_major>> mk;
    if (!pr.masks.empty()) mk = raft::make_const_mdspan(masks.view());
    std::optional<raft::device_matrix_view<T, index_t, raft::col_major>> fit;
    if (with_fitted) fit = fitted.view();
    nnls<T>(r,
            params,
            raft::make_const_mdspan(A.view()),
            raft::make_const_mdspan(B.view()),
            X.view(),
            mk,
            fit);
  }

  std::vector<T> host(raft::device_matrix<T, index_t, raft::col_major> const& d, index_t rows)
  {
    std::vector<T> h(rows * pr.P);
    raft::update_host(h.data(), d.data_handle(), h.size(), raft::resource::get_cuda_stream(res));
    raft::resource::sync_stream(res);
    return h;
  }
};

template <typename T>
constexpr T kkt_tolerance()
{
  return sizeof(T) == 4 ? T(1e-3) : T(1e-6);
}

template <typename T>
class NnlsTest : public ::testing::Test {
 protected:
  raft::device_resources res;

  /** Solve and assert the KKT conditions; returns the solution. */
  std::vector<T> solve_and_check(
    index_t m, index_t n, index_t P, unsigned seed, bool masked = false)
  {
    problem<T> pr(m, n, P, seed, masked);
    device_problem<T> dp(res, pr);
    dp.solve(res);
    auto X = dp.host(dp.X, n);
    auto k = kkt(pr.A, pr.B, X, pr.masks, m, n, P);
    EXPECT_TRUE(k.finite) << "non-finite solution or residual";
    EXPECT_EQ(k.negativity, T(0)) << "m=" << m << " n=" << n << " P=" << P;
    EXPECT_EQ(k.masked_leakage, T(0)) << "m=" << m << " n=" << n << " P=" << P;
    EXPECT_LT(k.stationarity, kkt_tolerance<T>()) << "m=" << m << " n=" << n << " P=" << P;
    EXPECT_LT(k.inactive, kkt_tolerance<T>()) << "m=" << m << " n=" << n << " P=" << P;
    return X;
  }
};

using Precisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(NnlsTest, Precisions);

TYPED_TEST(NnlsTest, SatisfiesKkt)
{
  // The signature-attribution shapes (small n, many rows), a wide A whose active set
  // saturates the rank bound min(m, n) = m, and a tall one.
  const index_t shapes[][3] = {{32, 10, 200},
                               {96, 29, 100},
                               {95, 66, 64},
                               {192, 56, 64},
                               {1536, 10, 50},
                               {96, 384, 16},
                               {96, 2048, 24},
                               {300, 40, 40}};
  unsigned seed             = 1;
  for (auto& s : shapes)
    this->solve_and_check(s[0], s[1], s[2], seed++);
}

TYPED_TEST(NnlsTest, TinyShapes)
{
  const index_t shapes[][3] = {{1, 1, 1}, {3, 1, 5}, {1, 5, 4}, {5, 5, 1}, {2, 33, 3}, {40, 33, 2}};
  unsigned seed             = 7;
  for (auto& s : shapes)
    this->solve_and_check(s[0], s[1], s[2], seed++);
}

TYPED_TEST(NnlsTest, MasksAreRespected)
{
  this->solve_and_check(96, 29, 100, 11, true);
  this->solve_and_check(96, 66, 40, 12, true);
}

TYPED_TEST(NnlsTest, FittedIsAx)
{
  using T = TypeParam;
  problem<T> pr(96, 29, 30, 21);
  device_problem<T> dp(this->res, pr);
  dp.solve(this->res, {}, true);
  auto X       = dp.host(dp.X, pr.n);
  auto F       = dp.host(dp.fitted, pr.m);
  double worst = 0;
  bool finite  = true;
  for (index_t p = 0; p < pr.P; ++p)
    for (index_t i = 0; i < pr.m; ++i) {
      double acc = 0;
      for (index_t j = 0; j < pr.n; ++j)
        acc += static_cast<double>(at(pr.A, i, j, pr.m)) * at(X, j, p, pr.n);
      const double f = at(F, i, p, pr.m);
      finite         = finite && std::isfinite(f) && std::isfinite(acc);
      worst          = std::max(worst, std::abs(acc - f) / (1.0 + std::abs(acc)));
    }
  EXPECT_TRUE(finite) << "non-finite fitted value";
  EXPECT_LT(worst, kkt_tolerance<T>());
}

TYPED_TEST(NnlsTest, IsDeterministic)
{
  // Two solves of the same input must agree bit for bit: the kernel has no atomics on the
  // result and no run-dependent branching, so a difference is a data race.
  using T = TypeParam;
  for (index_t n : {index_t(29), index_t(384)}) {
    problem<T> pr(96, n, 300, 31 + static_cast<unsigned>(n));
    device_problem<T> dp(this->res, pr);
    dp.solve(this->res);
    auto X1 = dp.host(dp.X, n);
    dp.solve(this->res);
    auto X2 = dp.host(dp.X, n);
    EXPECT_EQ(X1, X2) << "n=" << n;
  }
}

TYPED_TEST(NnlsTest, DegenerateColumnsAreHandled)
{
  // Zero columns and exact duplicates make G_P singular: the pivot test must reject them
  // instead of dividing by a tiny pivot.  The duplicated problem has the same optimum as
  // the de-duplicated one.
  using T = TypeParam;
  problem<T> pr(48, 16, 40, 41);
  for (index_t i = 0; i < pr.m; ++i) {
    pr.A[3 * pr.m + i]  = 0;                   // zero column
    pr.A[9 * pr.m + i]  = pr.A[2 * pr.m + i];  // duplicate of column 2
    pr.A[12 * pr.m + i] = pr.A[2 * pr.m + i];  // and again
  }
  device_problem<T> dp(this->res, pr);
  dp.solve(this->res);
  auto X = dp.host(dp.X, pr.n);
  auto k = kkt(pr.A, pr.B, X, pr.masks, pr.m, pr.n, pr.P);
  EXPECT_TRUE(k.finite) << "non-finite solution or residual";
  EXPECT_EQ(k.negativity, T(0));
  EXPECT_LT(k.stationarity, kkt_tolerance<T>());
  EXPECT_LT(k.inactive, kkt_tolerance<T>());
  for (index_t p = 0; p < pr.P; ++p)
    EXPECT_EQ(at(X, 3, p, pr.n), T(0)) << "zero column activated";
}

TYPED_TEST(NnlsTest, AllZeroMatrixGivesZero)
{
  using T = TypeParam;
  problem<T> pr(20, 8, 5, 51);
  std::fill(pr.A.begin(), pr.A.end(), T(0));
  device_problem<T> dp(this->res, pr);
  dp.solve(this->res);
  for (T v : dp.host(dp.X, pr.n))
    EXPECT_EQ(v, T(0));
}

TYPED_TEST(NnlsTest, HugeRightHandSideScale)
{
  // The gradient noise estimate sums |terms|: it must stay finite for a large b.
  using T = TypeParam;
  problem<T> pr(48, 20, 20, 61);
  const T scale = sizeof(T) == 4 ? T(1e15) : T(1e100);
  for (auto& v : pr.B)
    v *= scale;
  device_problem<T> dp(this->res, pr);
  dp.solve(this->res);
  auto X = dp.host(dp.X, pr.n);
  auto k = kkt(pr.A, pr.B, X, pr.masks, pr.m, pr.n, pr.P);
  EXPECT_TRUE(k.finite) << "non-finite solution or residual";
  EXPECT_LT(k.stationarity, kkt_tolerance<T>());
  EXPECT_LT(k.inactive, kkt_tolerance<T>());
}

TYPED_TEST(NnlsTest, SumOfMagnitudesNearOverflow)
{
  // A = [[1, 1], [0, 1]], b = [2, 1] * s with s near the top of the range: x = [s, s] fits
  // exactly, but after the first step the sum of |G x| terms behind the gradient noise bound
  // exceeds the largest finite value.  The bound must not turn that into -inf and stop.
  using T   = TypeParam;
  const T s = sizeof(T) == 4 ? T(1e38) : T(5.5e307);
  problem<T> pr(2, 2, 1, 81);
  pr.A = {T(1), T(0), T(1), T(1)};  // column-major
  pr.B = {T(2) * s, s};
  device_problem<T> dp(this->res, pr);
  dp.solve(this->res);
  auto X = dp.host(dp.X, pr.n);
  EXPECT_LT(std::abs(static_cast<double>(X[0]) / s - 1.0), kkt_tolerance<T>());
  EXPECT_LT(std::abs(static_cast<double>(X[1]) / s - 1.0), kkt_tolerance<T>());
}

TYPED_TEST(NnlsTest, TinySolutionOfLargeMatrix)
{
  // A = sc * [[2, 1], [1, 0]], b = [1, -0.5]: the optimum is x = [0, 1/sc], which is far
  // below any absolute "is zero" cutoff.  A positive coefficient must not be mistaken for
  // zero (that would re-activate the same columns until the iteration cap).
  using T    = TypeParam;
  const T sc = sizeof(T) == 4 ? T(1e13) : T(1e16);
  problem<T> pr(2, 2, 1, 82);
  pr.A = {T(2) * sc, sc, sc, T(0)};  // column-major
  pr.B = {T(1), T(-0.5)};
  device_problem<T> dp(this->res, pr);
  dp.solve(this->res, nnls_params{1e-6, 5});
  auto X = dp.host(dp.X, pr.n);
  EXPECT_EQ(X[0], T(0));
  EXPECT_LT(std::abs(static_cast<double>(X[1]) * sc - 1.0), kkt_tolerance<T>());
}

TYPED_TEST(NnlsTest, OutputIsFullyWritten)
{
  // Solutions are written in full (zeros included) whatever X held before.
  using T = TypeParam;
  problem<T> pr(96, 29, 64, 71);
  device_problem<T> dp(this->res, pr);
  RAFT_CUDA_TRY(cudaMemsetAsync(dp.X.data_handle(),
                                0xFF,
                                pr.n * pr.P * sizeof(T),
                                raft::resource::get_cuda_stream(this->res).get()));
  dp.solve(this->res);
  for (T v : dp.host(dp.X, pr.n))
    EXPECT_TRUE(v >= T(0)) << "an element of X was left unwritten (NaN) or negative";
}

TYPED_TEST(NnlsTest, MinMnPastTheSharedMemoryLimitIsAClearError)
{
  using T           = TypeParam;
  const index_t big = sizeof(T) == 4 ? 12000 : 7000;  // beyond ~8.1e3 / ~4.8e3 on sm_90
  auto A            = raft::make_device_matrix<T, index_t, raft::col_major>(this->res, big, big);
  auto B            = raft::make_device_matrix<T, index_t, raft::col_major>(this->res, big, 1);
  auto X            = raft::make_device_matrix<T, index_t, raft::col_major>(this->res, big, 1);
  const auto limit  = raft::resource::get_device_properties(this->res).sharedMemPerBlockOptin;
  if (detail::lawson_smem_bytes<T>(static_cast<int>(big), static_cast<int>(big)) <= limit)
    GTEST_SKIP() << "this device's shared memory fits the shape";
  EXPECT_THROW(nnls<T>(this->res,
                       {},
                       raft::make_const_mdspan(A.view()),
                       raft::make_const_mdspan(B.view()),
                       X.view()),
               raft::logic_error);
}

TYPED_TEST(NnlsTest, ShapeMismatchThrows)
{
  using T = TypeParam;
  auto A  = raft::make_device_matrix<T, index_t, raft::col_major>(this->res, 10, 4);
  auto B  = raft::make_device_matrix<T, index_t, raft::col_major>(this->res, 9, 2);
  auto X  = raft::make_device_matrix<T, index_t, raft::col_major>(this->res, 4, 2);
  EXPECT_THROW(nnls<T>(this->res,
                       {},
                       raft::make_const_mdspan(A.view()),
                       raft::make_const_mdspan(B.view()),
                       X.view()),
               raft::logic_error);
}

}  // namespace

TEST(NnlsLarge, Int64ExtentsPastTheGramIntLimit)
{
  // n = 46500: n * n overflows an int (G is 8.6 GB in float32).
  raft::device_resources res;
  constexpr index_t n = 46500, m = 64, P = 2;
  std::size_t free_bytes = 0, total_bytes = 0;
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes < std::size_t(24) << 30) GTEST_SKIP() << "needs ~24 GB of free device memory";
  problem<float> pr(m, n, P, 81);
  device_problem<float> dp(res, pr);
  dp.solve(res);
  auto X = dp.host(dp.X, n);
  auto k = kkt(pr.A, pr.B, X, pr.masks, m, n, P);
  EXPECT_TRUE(k.finite) << "non-finite solution or residual";
  EXPECT_EQ(k.negativity, 0.f);
  EXPECT_LT(k.stationarity, 1e-3f);
  EXPECT_LT(k.inactive, 1e-3f);
}

/**
 * Dry run protocol: allocations are made exactly as in a real run (sized from the arguments
 * only), no CUDA work is executed.  The helper compares the peaks of every memory resource.
 */
TYPED_TEST(NnlsTest, DryRunCompliant)
{
  using T                   = TypeParam;
  const index_t shapes[][3] = {{32, 10, 200}, {96, 29, 100}, {95, 66, 64}, {96, 2048, 8}};
  for (auto& s : shapes) {
    for (bool masked : {false, true}) {
      for (bool with_fitted : {false, true}) {
        problem<T> pr(s[0], s[1], s[2], 91, masked);
        device_problem<T> dp(this->res, pr);
        raft::resource::sync_stream(this->res);
        execute_with_dry_run_check(
          this->res,
          [&](raft::resources const& r) { dp.solve(r, {}, with_fitted); },
          alloc_behavior::ARGUMENT_DRIVEN,
          sizeof(T) * s[1] * (s[1] + s[2]));  // at least the Gram matrix and the projections
      }
    }
  }
}

}  // namespace raft::solver
