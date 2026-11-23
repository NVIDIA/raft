/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>        // raft::min_op, raft::max_op
#include <raft/util/cuda_utils.cuh>       // raft::WarpSize
#include <raft/util/cudart_utils.hpp>     // raft::warp_full_mask
#include <raft/util/pow2_utils.cuh>       // raft::Pow2
#include <raft/util/reduction.cuh>        // raft::warpReduce
#include <raft/util/vectorized.cuh>       // raft::TxN_t
#include <raft/util/warp_primitives.cuh>  // raft::shfl

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace raft::solver::detail {

/** Lanes of a warp, and the bits of one word of the active-set bitset (both powers of two). */
using lawson_warp = raft::Pow2<raft::WarpSize>;
using lawson_word = raft::Pow2<32>;

/**
 * Batched, masked Lawson-Hanson NNLS: one problem per CUDA block.
 *
 * Every problem `min_{x >= 0} ||A x - b_p||` is solved through the normal equations with
 * the Gram matrix G = A^T A and the projections c_p = A^T b_p, both formed by the caller.
 * The outer loop grows the active set P by the column with the largest projected gradient
 * w = c - G x; the inner loop solves G_P s = c_P with an incrementally maintained Cholesky
 * factor of G_P and line-searches away negative coordinates.
 *
 * Lawson-Hanson keeps the active columns linearly independent, so |P| <= rank(A) <=
 * min(m, n) =: np_max, and everything per problem lives in O(np_max) shared memory plus a
 * bit per column; the factor lives in a per-block global slab of np_max x np_max.
 */

/**
 * Cholesky ridge: a pivot is accepted only if the Schur complement of the new column
 * clears it.  Scales with the pivot and with the working precision.  This is a relative
 * rank test, which is what the invariant |P| <= rank(A) needs (adding the ridge to the
 * pivot instead would accept numerically dependent columns).
 */
template <typename T>
__device__ inline T lawson_ridge_eps(T diag)
{
  // ~1 ulp of the working precision for float (the old 1e-7) and ~64 ulp for double (1e-14).
  constexpr T rel = std::numeric_limits<T>::epsilon() * (sizeof(T) == 4 ? T(1) : T(64));
  return rel * (diag > T(0) ? diag : T(1));
}

/** Bytes of statically allocated shared memory of the kernel (the ints `sm_n_active`,
 *  `sm_flag`, `sm_open`, padded to the 16-byte alignment of the dynamic part): they count
 *  against the per-block limit together with the dynamic part. */
constexpr std::size_t kLawsonStaticSmem = 16;

/**
 * Dynamic shared-memory footprint for `n` columns and an active set of at most `np_max`:
 *   T   xp[np_max]    current solution, compact: xp[kk] = x[idx[kk]]
 *   T   s[np_max]     trial solution (also RHS / downdate scratch)
 *   T   y[np_max]     forward-solve state L^-1 c_P, maintained incrementally
 *   T   dinv[np_max]  1 / diag(L)
 *   T   dg[np_max]    diag(L): the factor slab is strictly lower triangular
 *   T   red_val[WarpSize], int red_idx[WarpSize]   per-warp argmax partials
 *   int idx[np_max]   compact list of active-set column indices
 *   int rem[np_max]   positions removed by one line-search step
 *   u32 act[ceil(n / 32)]   active-set bitset over the n columns
 * The projected gradient w = c - G_P x is never stored: it is fused with the argmax that
 * is its only consumer.  G stays in global memory (L2-cached across the grid).
 */
template <typename T>
inline std::size_t lawson_smem_bytes(int n, int np_max)
{
  constexpr std::size_t kVectors = 5;  // xp, s, y, dinv, dg
  const auto np                  = static_cast<std::size_t>(np_max);
  return sizeof(T) * np * kVectors + sizeof(T) * raft::WarpSize + sizeof(int) * raft::WarpSize +
         sizeof(int) * np * 2 +
         sizeof(std::uint32_t) *
           lawson_word::div(lawson_word::roundUp(static_cast<std::size_t>(n)));
}

/** Largest np_max whose footprint, static shared memory included, fits in `limit` bytes
 *  (0 if none); see lawson_smem_bytes. */
template <typename T>
inline int lawson_max_np(std::size_t limit, int n)
{
  const std::size_t fixed = lawson_smem_bytes<T>(n, 0) + kLawsonStaticSmem;
  const std::size_t per   = lawson_smem_bytes<T>(n, 1) - lawson_smem_bytes<T>(n, 0);
  if (limit <= fixed) return 0;
  return static_cast<int>(
    std::min<std::size_t>((limit - fixed) / per, std::numeric_limits<int>::max()));
}

/** Pointers into the dynamic shared memory of one block; see lawson_smem_bytes. */
template <typename T>
struct lawson_smem {
  T* xp;
  T* s;
  T* y;
  T* dinv;
  T* dg;
  T* red_val;
  int* red_idx;
  int* idx;
  int* rem;
  std::uint32_t* act;  // bit j set <=> column j is in the active set
};

/** Carve the dynamic shared memory up; the layout mirrors lawson_smem_bytes. */
template <typename T>
__device__ lawson_smem<T> lawson_smem_layout(unsigned char* smem, int np_max)
{
  lawson_smem<T> S{};
  S.xp      = reinterpret_cast<T*>(smem);  // wider elements first, 4-byte ones after
  S.s       = S.xp + np_max;
  S.y       = S.s + np_max;
  S.dinv    = S.y + np_max;
  S.dg      = S.dinv + np_max;
  S.red_val = S.dg + np_max;
  S.red_idx = reinterpret_cast<int*>(S.red_val + raft::WarpSize);
  S.idx     = S.red_idx + raft::WarpSize;
  S.rem     = S.idx + np_max;
  S.act     = reinterpret_cast<std::uint32_t*>(S.rem + np_max);
  return S;
}

/** Test / set / clear bit `j` of the active-set bitset.  Only warp 0 writes it, between
 *  barriers; the atomic clear in the compaction is the one concurrent writer. */
__device__ inline bool lawson_act_test(const std::uint32_t* act, int j)
{
  return ((act[lawson_word::div(j)] >> lawson_word::mod(j)) & 1u) != 0u;
}
/** Set bit `j` (mark column j active). */
__device__ inline void lawson_act_set(std::uint32_t* act, int j)
{
  act[lawson_word::div(j)] |= (1u << lawson_word::mod(j));
}
/** Clear bit `j` (column j leaves the active set). */
__device__ inline void lawson_act_clear(std::uint32_t* act, int j)
{
  act[lawson_word::div(j)] &= ~(1u << lawson_word::mod(j));
}

/**
 * Projected gradient w = c - G x fused with the argmax over the columns that may enter
 * the active set.  x is zero outside the active set, so only the np active columns of G
 * contribute: w[j] = c[j] - sum_kk G(j, idx[kk]) * xp[kk]; consecutive threads read
 * consecutive (column-major) elements of G.  Inactive columns that `mask` (nullable)
 * enables are eligible.  Each thread returns its running (max, argmax); (-inf, -1) when
 * it saw no eligible column.
 *
 * The gradient is judged against its own rounding error: w is a difference of similar
 * quantities, so near the solution it floors at about sqrt(np) * eps * sum|G x|.  In
 * float32 that exceeds a typical tol, so an absolute test can never pass and the loop
 * activates columns on noise.  A column therefore has to be positive by more than the
 * arithmetic could have invented.
 */
template <typename T, int BlockSize>
__device__ inline void lawson_gradient_partial(const T* __restrict__ G,
                                               const T* __restrict__ c,
                                               const std::uint8_t* __restrict__ mask,
                                               const std::uint32_t* act,
                                               const int* idx,
                                               const T* xp,
                                               int np,
                                               int n,
                                               T& thread_max,
                                               int& thread_idx)
{
  const T noise_scale = std::sqrt(T(np + 1)) * std::numeric_limits<T>::epsilon();
  thread_max          = -std::numeric_limits<T>::infinity();
  thread_idx          = -1;
  // The column loop is unrolled by four to keep four independent G loads in flight; the
  // terms are still accumulated in column order.
  for (int j = threadIdx.x; j < n; j += BlockSize) {
    if (lawson_act_test(act, j)) continue;
    if (mask != nullptr && mask[j] == 0) continue;
    const T cj = c[j];
    T acc      = cj;
    // The rounding-error allowance, accumulated already scaled by `noise_scale` (< 1) so
    // that the sum of magnitudes cannot overflow when the terms themselves do not.
    T allowance              = noise_scale * std::abs(cj);
    const T* __restrict__ Gj = G + j;
    auto accumulate          = [&](T g, T x) {
      const T t = g * x;
      acc -= t;
      allowance = fma(noise_scale, std::abs(t), allowance);
    };
    int kk = 0;
#pragma unroll 4
    for (; kk + 4 <= np; kk += 4) {
      const T g0 = Gj[static_cast<std::size_t>(idx[kk]) * n];
      const T g1 = Gj[static_cast<std::size_t>(idx[kk + 1]) * n];
      const T g2 = Gj[static_cast<std::size_t>(idx[kk + 2]) * n];
      const T g3 = Gj[static_cast<std::size_t>(idx[kk + 3]) * n];
      accumulate(g0, xp[kk]);
      accumulate(g1, xp[kk + 1]);
      accumulate(g2, xp[kk + 2]);
      accumulate(g3, xp[kk + 3]);
    }
    for (; kk < np; ++kk)
      accumulate(Gj[static_cast<std::size_t>(idx[kk]) * n], xp[kk]);
    const T w = acc - allowance;
    if (w > thread_max) {
      thread_max = w;
      thread_idx = j;
    }
  }
}

// ---------------------------------------------------------------------------
// The Cholesky factor and the O(np)-deep recurrences.
//
// Appending a column is a forward substitution against the factor, and every inner
// iteration of the line search is a back substitution; both are chains `np` steps
// deep.  Run one row at a time with block barriers (a global-memory column load,
// a 5-step shuffle reduction, a division and two __syncwarp per step -- ~300-500
// cycles) they are what the solve is made of.  Here they run in ONE warp on 32x32
// panels held in registers instead: the chain step is a single SHFL followed by a
// single FFMA (~37 cycles dependent), and everything bulky -- the panel loads, the
// update of the rows below a panel -- is coalesced streaming work around it.
//
// Layout of the factor (the per-block global slab, column-major, leading dimension
// ld = lawson_slab_ld(np_max)):
//   * STRICTLY lower triangular: the diagonal lives in shared memory (`dg`, and
//     `dinv` = 1 / dg, which both chains multiply by instead of dividing), the
//     upper triangle is zero.  The tile loads then need no predicate on the
//     triangle, and the backward tile can be loaded as float4 / double2 (ld % 4 == 0
//     and the slab base is 16-byte aligned).
//   * `kLawsonSlabPad` extra columns after column np_max-1, so the unguarded loads
//     of the last partial panel stay inside the slab.
//   * The slab is NEVER cleared as a whole.  The allocation may hold anything (the
//     allocator's leftovers, NaNs): the kernel only reads what it wrote, or what
//     lawson_slab_ensure_open zeroed when the factor first grew into the panel.  A
//     clear of the whole slab used to open every launch -- np_max^2 elements per
//     resident block, more than the solve itself for a batch of trivial problems and
//     megabytes of L2-evicting writes at the production shapes.  The only elements
//     that are read without having been written are the ones in the 32 x 32 diagonal
//     block of a panel (its upper triangle, which is zero by construction and never
//     written, and the few entries the vector loads read beyond the last row), and
//     those are cleared one panel at a time, when the first problem of the block reaches
//     it: work proportional to what the batch actually uses.
// ---------------------------------------------------------------------------

/** Extra columns after the last factor column; see above.  The unguarded loads of the last
 *  partial panel go up to the next multiple of four. */
constexpr int kLawsonSlabPad = 4;

/** Leading dimension of a factor slab. */
__host__ __device__ constexpr int lawson_slab_ld(int np_max)
{
  return raft::Pow2<8>::roundUp(np_max);
}

/** Elements of one factor slab. */
__host__ __device__ constexpr std::size_t lawson_slab_elems(int np_max)
{
  return static_cast<std::size_t>(np_max + kLawsonSlabPad) *
         static_cast<std::size_t>(lawson_slab_ld(np_max));
}

/** (value, index) maximum over the warp.  Ties resolve to the lowest lane; every lane
 *  gets the result; the index is -1 when the value is -inf everywhere (no candidate). */
template <typename T>
__device__ inline void lawson_warp_argmax(T& v, int& i)
{
  const T m         = raft::warpReduce(v, raft::max_op{});
  const unsigned eq = __ballot_sync(raft::warp_full_mask(), v == m);
  const int src     = __ffs(eq) - 1;
  i                 = src < 0 ? -1 : raft::shfl(i, src);
  v                 = m;
}

/**
 * Forward substitution L(0:m, 0:m) l = a, in place: `sc[0:m)` holds a on entry and l
 * on exit; `dinv[i] = 1 / L(i,i)`.  One warp, panel by panel: lane = row of the
 * panel, tile column j of L held in register j (a coalesced load), pre-scaled by the
 * row's dinv; the chain step is one shuffle and one FMA.  The rows below the panel
 * are then updated with lane = row, streaming the (coalesced) panel columns against
 * the just-solved values read as shared-memory broadcasts.  The last panel is
 * partial, hence the `valid` / `cnt` guards; the strictly-lower slab (zero upper
 * triangle) makes the rest unconditional.
 */
template <typename T>
__device__ inline void lawson_warp_fwd(
  const T* __restrict__ L, int ld, int m, const T* __restrict__ dinv, T* sc)
{
  const int lane = lawson_warp::mod(threadIdx.x);
  for (int lo = 0; lo < m; lo += raft::WarpSize) {
    const int cnt    = min(static_cast<int>(raft::WarpSize), m - lo);
    const int r      = lo + lane;
    const bool valid = lane < cnt;
    const T di       = valid ? dinv[r] : T(0);
    T tile[raft::WarpSize];
    {
      const T* __restrict__ Lp = L + r + static_cast<std::size_t>(lo) * ld;
#pragma unroll
      for (int j0 = 0; j0 < raft::WarpSize; j0 += 4) {
        if (j0 < cnt) {
#pragma unroll
          for (int j = j0; j < j0 + 4; ++j)
            tile[j] = valid ? Lp[static_cast<std::size_t>(j) * ld] * di : T(0);
        }
      }
    }
    T acc = valid ? sc[r] * di : T(0);
#pragma unroll
    for (int j0 = 0; j0 < raft::WarpSize; j0 += 4) {
      if (j0 < cnt) {
#pragma unroll
        for (int j = j0; j < j0 + 4; ++j) {
          const T lj = raft::shfl(acc, j);
          acc        = fma(-tile[j], lj, acc);
        }
      }
    }
    if (valid) sc[r] = acc;
    __syncwarp();
    const int hi = lo + cnt;
    for (int r2 = hi + lane; r2 < m; r2 += raft::WarpSize) {
      const T* __restrict__ Lr = L + r2 + static_cast<std::size_t>(lo) * ld;
      T a0 = sc[r2], a1 = T(0);
#pragma unroll
      for (int j = 0; j < raft::WarpSize; j += 2) {
        a0 = fma(-Lr[static_cast<std::size_t>(j) * ld], sc[lo + j], a0);
        a1 = fma(-Lr[static_cast<std::size_t>(j + 1) * ld], sc[lo + j + 1], a1);
      }
      sc[r2] = a0 + a1;
    }
    if (hi < m) __syncwarp();
  }
}

/**
 * Back substitution L^T x = y for the leading np x np block; y is copied into `s`
 * first (`s` need not be initialised) and x ends up in `s`.  One warp, bottom-up in
 * panels of 32: lane = COLUMN of the panel (the transposed access of the forward
 * pass), its tile column loaded with float4 / double2.  The contribution of the
 * already-solved rows below the panel is a per-column dot product; the chain inside
 * the panel is again one SHFL + one FFMA per step.
 */
template <typename T>
__device__ inline void lawson_warp_bwd(const T* __restrict__ L,
                                       int ld,
                                       int np,
                                       const T* __restrict__ dinv,
                                       const T* __restrict__ y,
                                       T* s)
{
  constexpr int W = 16 / sizeof(T);  // elements per 16-byte vector (4 for float, 2 for double)
  using V         = raft::TxN_t<T, W>;
  const int lane  = lawson_warp::mod(threadIdx.x);
  for (int j = lane; j < np; j += raft::WarpSize)
    s[j] = y[j];
  __syncwarp();
  const int nt = lawson_warp::div(lawson_warp::roundUp(np));
  for (int k = nt - 1; k >= 0; --k) {
    const int lo              = raft::WarpSize * k;
    const int cnt             = min(static_cast<int>(raft::WarpSize), np - lo);
    const int hi              = lo + cnt;
    const int c               = lo + lane;
    const bool valid          = lane < cnt;
    const T* __restrict__ col = L + static_cast<std::size_t>(c) * ld;
    // The factor is written by this kernel: read it with plain loads (TxN_t::load of a const
    // pointer is an `ld.global.nc`, which is not coherent with writes made during the launch).
    T* col_rw = const_cast<T*>(col);
    T tile[raft::WarpSize];
#pragma unroll
    for (int i0 = 0; i0 < raft::WarpSize; i0 += W) {
      if (i0 < cnt) {
        V q;
        if (valid) {
          q.load(col_rw, lo + i0);
        } else {
          q.fill(T(0));
        }
#pragma unroll
        for (int i = 0; i < W; i++) {
          tile[i0 + i] = q.val.data[i];
        }
      }
    }
    T acc = valid ? s[c] : T(0);
    if (valid && hi < np) {
      // One chain per vector lane; `hi` is a multiple of the warp size and `ld` of eight
      // elements, so the 16-byte vector loads of the column are aligned.
      T p[W] = {};
      int j  = hi;
      V q;
      for (; j + W <= np; j += W) {
        q.load(col_rw, j);
#pragma unroll
        for (int i = 0; i < W; i++) {
          p[i] = fma(q.val.data[i], s[j + i], p[i]);
        }
      }
      for (; j < np; ++j)
        p[0] = fma(col[j], s[j], p[0]);
      T dot = p[0];
#pragma unroll
      for (int i = 1; i < W; i++)
        dot += p[i];
      acc -= dot;
    }
    const T di = valid ? dinv[c] : T(0);
    acc *= di;
#pragma unroll
    for (int i0 = raft::WarpSize - W; i0 >= 0; i0 -= W) {
      if (i0 < cnt) {
#pragma unroll
        for (int i = i0 + W - 1; i >= i0; --i) {
          const T xi = raft::shfl(acc, i);
          acc        = fma(-tile[i] * di, xi, acc);
        }
      }
    }
    if (valid) s[c] = acc;
    __syncwarp();
  }
}

/**
 * Make the elements of a panel's diagonal block, which the tile loads read without the
 * kernel ever having written them, defined: zero the block `[lo, lo + 32) x [lo, lo + 32)`
 * of the slab (clipped to the rows and columns the loads can reach, `roundup(np_max, 4)`)
 * the first time a problem of this block grows the factor to row `lo`.  Called by warp 0 (all
 * 32 lanes) before the append of row `m`; does nothing unless `m` starts a panel that has
 * not been opened yet (`*sm_open` = columns opened so far, shared, one value per block).
 *
 * What the solves read that nothing writes:
 *   * the upper triangle and diagonal of the block: the forward tile (rows `lo + lane`,
 *     columns `lo + j`, j up to the next multiple of four past the panel's size) and the
 *     backward tile (float4 / double2 at rows `lo + i`, whole aligned groups, in column
 *     `lo + lane`) read all of it, and the chains multiply it by shuffled solution entries:
 *     it has to be exactly zero;
 *   * the rest of the last aligned group of rows of a column of the last, partial panel,
 *     below the factor's last row: multiplied by an exact zero in the chain, so it has to be
 *     finite (a NaN left in the allocation would survive that product).  It lies in the block
 *     as well (`np` <= `lo` + 32 and rows are read up to the next multiple of four).
 * The strictly lower triangle is written by the append / downdate code before it is read;
 * rows below the block are the part of the factor that earlier panels' updates read, and are
 * written rows too.  The downdate writes only the strictly lower triangle, so the zeros stay.
 * The block is cleared before the first element of its rows is appended (`m == lo`), and
 * stays open for every later problem of the block, so no live data is ever cleared.
 */
template <typename T>
__device__ inline void lawson_slab_ensure_open(T* L, int ld, int np_max, int m, int* sm_open)
{
  if (lawson_warp::mod(m) != 0) return;  // only the first row of a panel opens it (warp-uniform)
  const int lane = lawson_warp::mod(threadIdx.x);
  // Lane 0 alone reads and writes the shared counter, so there is no race between lanes;
  // everyone learns the value by shuffle.
  int open_to = 0;
  if (lane == 0) open_to = *sm_open;
  open_to = raft::shfl(open_to, 0);
  __syncwarp();  // (the load may be issued by every lane; order it before lane 0's store below)
  if (m < open_to) return;              // already open (an earlier problem of this block got here)
  const int reach = (np_max + 3) & ~3;  // rows / columns the unguarded loads can touch
  const int rows  = min(m + raft::WarpSize, reach);
  const int cols  = min(m + raft::WarpSize, reach);
  if (m + lane < rows)
    for (int cc = m; cc < cols; ++cc)
      L[(m + lane) + static_cast<std::size_t>(cc) * ld] = T(0);
  if (lane == 0) *sm_open = m + raft::WarpSize;
  __syncwarp();  // the cleared block is visible to every lane before the append reads it
}

/**
 * Bordering Cholesky append, executed by warp 0: gather the new Gram column a_12 =
 * G(idx[0:m), j_star), forward-solve L_11 l = a_12, form the dots l.l and l.y, and
 * extend the factor (new row m of the strictly-lower slab, `dg[m]`, `dinv[m]`) and
 * the forward-solve state `y` -- y[m] = (c[j_star] - l.y) / d -- only when the pivot
 * is accepted, so a rejected activation leaves everything intact.  `sc` is O(np)
 * shared scratch.  The result is warp-uniform: true = accepted.
 *
 * Device analogue of raft::linalg::choleskyRank1Update, which is a host/cuBLAS
 * routine and so cannot be called from within a block.  Positive-definiteness is kept
 * by the pivot test (lawson_pivot_accepts): the Schur complement has to clear a
 * relative ridge, which is the rank test the Lawson-Hanson invariant |P| <= rank(A)
 * needs.
 */
template <typename T>
__device__ inline bool lawson_warp_append(T* L,
                                          int ld,
                                          int np,
                                          const T* __restrict__ G,
                                          int n,
                                          const int* idx,
                                          const T* c,
                                          T* y,
                                          T* dg,
                                          T* dinv,
                                          T* sc)
{
  const int lane             = lawson_warp::mod(threadIdx.x);
  const int m                = np - 1;
  const int j_star           = idx[m];
  const T* __restrict__ Gcol = G + static_cast<std::size_t>(j_star) * n;
  const T a22                = Gcol[j_star];
  const T cj                 = c[j_star];
  // Gather with all loads of a group of four in flight before any is consumed.
  for (int base = 0; base < m; base += 128) {
    T av[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int i = base + lane + raft::WarpSize * t;
      av[t]       = (i < m) ? Gcol[idx[i]] : T(0);
    }
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int i = base + lane + raft::WarpSize * t;
      if (i < m) sc[i] = av[t];
    }
  }
  __syncwarp();
  lawson_warp_fwd(L, ld, m, dinv, sc);
  T dll = T(0), dly = T(0);
  for (int j = lane; j < m; j += raft::WarpSize) {
    const T lj                              = sc[j];
    L[m + static_cast<std::size_t>(j) * ld] = lj;
    dll += lj * lj;
    dly += lj * y[j];
  }
  dll           = raft::warpReduce(dll);
  dly           = raft::warpReduce(dly);
  const T d2    = a22 - dll;
  const bool ok = d2 > lawson_ridge_eps(a22);
  if (ok && lane == 0) {
    const T d = std::sqrt(d2);
    dg[m]     = d;
    y[m]      = (cj - dly) / d;
    dinv[m]   = T(1) / d;
  }
  __syncwarp();
  return ok;
}

/**
 * Remove active-set position `p` (0-based, in the current np-ordering) from the
 * np x np Cholesky factor (strictly-lower slab `L`, diagonal `dg`), producing the
 * (np-1) x (np-1) factor of the submatrix with row/column p deleted; executed by
 * warp 0.
 *
 * Deleting an interior row/column reduces to a positive rank-1 Cholesky update of
 * the trailing diagonal block by the below-diagonal part of column p:
 *   M M^T = L33 L33^T + l3p l3p^T,
 * applied with a sequence of Givens rotations (Golub & Van Loan, "deleting a
 * column").  `v` is O(np) scratch holding l3p and the rotated residual.  The columns
 * left of p only shift rows up within a column (lane = column); the trailing block
 * shift L(i,j) <- L(i+1,j+1) decomposes into independent diagonals (lane = diagonal,
 * eight elements in flight); the Givens sweep -- an inherently sequential angle
 * recurrence -- runs with lane 0 computing the rotation and the lanes applying it.
 *
 * The forward-solve state y = L^-1 c_P is downdated in lock-step: y[0:p) is
 * unchanged, the deleted component y[p] is saved as the rotation partner, the
 * trailing y[p+1:] is compacted into y[p:], and the same Givens (c,s) that
 * retriangularise L are applied to (y[p+k], partner) so y stays L^-1 c_P for the
 * shrunken active set.  `dg` and `dinv` follow.
 */
template <typename T>
__device__ inline void lawson_warp_delete_one(
  T* L, int ld, int np, int p, T* y, T* dg, T* dinv, T* v)
{
  const int lane = lawson_warp::mod(threadIdx.x);
  const int q    = np - 1 - p;  // trailing block size
  // Columns left of p: drop row p by shifting rows [p, np-1) up one (lane = column).
  for (int j = lane; j < p; j += raft::WarpSize) {
    T* col = L + static_cast<std::size_t>(j) * ld;
    for (int i = p; i < np - 1; i += 8) {
      T t[8];
#pragma unroll
      for (int u = 0; u < 8; ++u)
        t[u] = (i + u < np - 1) ? col[i + u + 1] : T(0);
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (i + u < np - 1) col[i + u] = t[u];
    }
  }
  // Save l3p = L(p+1 : np-1, p) before the trailing shift overwrites column p.
  for (int i = lane; i < q; i += raft::WarpSize)
    v[i] = L[(p + 1 + i) + static_cast<std::size_t>(p) * ld];
  T partner = T(0);
  if (lane == 0) partner = y[p];
  __syncwarp();
  // Compact y and dg (drop position p); the trailing diagonal entries are rewritten
  // by the sweep below.
  for (int base = p; base < np - 1; base += raft::WarpSize) {
    const int i = base + lane;
    const T val = (i + 1 < np) ? y[i + 1] : T(0);
    const T dv  = (i + 1 < np) ? dg[i + 1] : T(0);
    __syncwarp();
    if (i < np - 1) {
      y[i]  = val;
      dg[i] = dv;
    }
    __syncwarp();
  }
  // Trailing strictly-lower block, up-left shift: L(i,j) <- L(i+1, j+1) along the
  // diagonals d = i - j >= 1.  One lane walks one diagonal from the top, reading each
  // source before this same lane later overwrites it; diagonals never alias across
  // lanes, so the whole shift is a sync-free warp-wide pass.
  for (int d = 1 + lane; d < q; d += raft::WarpSize) {
    for (int k = 0; k <= q - 1 - d; k += 8) {
      T t[8];
#pragma unroll
      for (int u = 0; u < 8; ++u)
        t[u] = (k + u <= q - 1 - d)
                 ? L[(p + d + k + u + 1) + static_cast<std::size_t>(p + k + u + 1) * ld]
                 : T(0);
#pragma unroll
      for (int u = 0; u < 8; ++u)
        if (k + u <= q - 1 - d)
          L[(p + d + k + u) + static_cast<std::size_t>(p + k + u) * ld] = t[u];
    }
  }
  __syncwarp();
  // Positive rank-1 update of the trailing block (rows/cols p..np-2) by v.
  for (int k = 0; k < q; ++k) {
    T c = T(0), s = T(0);
    if (lane == 0) {
      const T Lkk = dg[p + k];
      const T vk  = v[k];
      const T r   = std::sqrt(Lkk * Lkk + vk * vk);
      c           = (r > T(0)) ? (Lkk / r) : T(1);
      s           = (r > T(0)) ? (vk / r) : T(0);
      dg[p + k]   = r;
      dinv[p + k] = (r > T(0)) ? T(1) / r : T(0);
      // Rotate the forward-solve state with the same (c,s): the trailing component
      // y[p+k] pairs with the running partner just like L's column p+k pairs with
      // the v-column.
      const T yk = y[p + k];
      y[p + k]   = c * yk + s * partner;
      partner    = c * partner - s * yk;
    }
    c = raft::shfl(c, 0);
    s = raft::shfl(s, 0);
    for (int t = k + 1 + lane; t < q; t += raft::WarpSize) {
      const int row       = p + t;
      const std::size_t o = row + static_cast<std::size_t>(p + k) * ld;
      const T lik         = L[o];
      const T vi          = v[t];
      L[o]                = c * lik + s * vi;
      v[t]                = c * vi - s * lik;
    }
    __syncwarp();
  }
}

/**
 * Everything of an inner iteration after the back substitution, executed by warp 0:
 * the non-negativity test on s, the line search and the compaction of the active set
 * with the matching factor downdates.  `npc` (the active-set size) is updated in
 * place.  Returns 0 = the trial solution is positive (xp = s, done), 1 = a line-search
 * step was taken and coordinates removed (npc updated; solve again), 2 = stop (no
 * binding coordinate, or the active set emptied).
 */
template <typename T>
__device__ inline int lawson_post_solve(
  T* L, int ld, int& npc, T* xp, T* s, T* y, T* dg, T* dinv, int* idx, int* rem, std::uint32_t* act)
{
  const int lane = lawson_warp::mod(threadIdx.x);
  T mn           = std::numeric_limits<T>::infinity();
  for (int i = lane; i < npc; i += raft::WarpSize)
    mn = min(mn, s[i]);
  mn = raft::warpReduce(mn, raft::min_op{});
  if (mn > T(0)) {
    for (int i = lane; i < npc; i += raft::WarpSize)
      xp[i] = s[i];
    __syncwarp();
    return 0;
  }
  // Largest feasible step alpha = min x/(x - s) over the coordinates with s <= 0,
  // and how many of them are binding.
  T amin  = std::numeric_limits<T>::infinity();
  int cnt = 0;
  for (int i = lane; i < npc; i += raft::WarpSize) {
    const T sj = s[i];
    if (sj <= T(0)) {
      const T xj    = xp[i];
      const T denom = xj - sj;  // strictly positive when xj >= 0 and sj <= 0
      // Numerical safety: if denom is tiny (xj == 0 and sj == 0) treat as non-binding.
      if (denom > T(0)) {
        const T al = xj / denom;
        if (al < amin) amin = al;
        ++cnt;
      }
    }
  }
  amin = raft::warpReduce(amin, raft::min_op{});
  cnt  = raft::warpReduce(cnt);
  if (cnt == 0) return 2;
  for (int i = lane; i < npc; i += raft::WarpSize) {
    const T xi = xp[i];
    const T si = s[i];
    // The coordinate that attained the minimum ratio is the one the step is sized to
    // drive to zero: pin it to exactly zero so the compaction below sees it, instead of
    // testing the rounded post-step value (which lands at +-eps * x about half the time).
    const T denom      = xi - si;
    const bool binding = si <= T(0) && denom > T(0) &&
                         xi <= amin * denom * (T(1) + T(8) * std::numeric_limits<T>::epsilon());
    xp[i] = binding ? T(0) : xi + amin * (si - xi);
  }
  __syncwarp();
  // Compact the active set (idx and the compact solution move together), recording
  // the removed positions in ascending order.  A ballot gives every surviving lane
  // its output slot, so the pass is warp-wide and sync-free; the in-place writes only
  // ever go to slots at or below the lane's own, and a chunk reads all of its inputs
  // before any of them is written.
  int new_n = 0, n_rem = 0;
  for (int base = 0; base < npc; base += raft::WarpSize) {
    const int i = base + lane;
    int jv      = 0;
    T xv        = T(0);
    bool keep = false, drop = false;
    if (i < npc) {
      jv = idx[i];
      xv = xp[i];
      // No absolute cutoff: the solution may legitimately be tiny when A is large.  The
      // binding coordinate was pinned to exactly zero above, so only that and
      // coordinates the step overshot below zero are removed.
      keep = xv > T(0);
      drop = !keep;
    }
    const unsigned mk = __ballot_sync(raft::warp_full_mask(), keep);
    const unsigned md = __ballot_sync(raft::warp_full_mask(), drop);
    const unsigned lt = (1u << lane) - 1u;
    __syncwarp();
    if (keep) {
      idx[new_n + __popc(mk & lt)] = jv;
      xp[new_n + __popc(mk & lt)]  = xv;
    }
    if (drop) {
      rem[n_rem + __popc(md & lt)] = i;
      // Several lanes may clear bits of the same word.
      atomicAnd(&act[lawson_word::div(jv)], ~(1u << lawson_word::mod(jv)));
    }
    new_n += __popc(mk);
    n_rem += __popc(md);
    __syncwarp();
  }
  // Downdate the factor for each removed position, deleting in descending order so
  // earlier (lower-index) deletions stay valid as np shrinks.
  {
    int cur = npc;
    for (int r = n_rem - 1; r >= 0; --r) {
      lawson_warp_delete_one(L, ld, cur, rem[r], y, dg, dinv, s);
      --cur;
    }
  }
  npc = new_n;
  return npc == 0 ? 2 : 1;
}

/** Everything the kernel needs, as raw pointers: it addresses G, C, X and the masks with
 *  64-bit offsets, since n * n (G) and n * P (C, X, masks) can pass 2^31. */
template <typename T>
struct lawson_args {
  const T* G;                 ///< (n, n) column-major Gram matrix
  const T* C;                 ///< (n, P) A^T B, column-major
  const std::uint8_t* masks;  ///< (n, P) column-major support, or nullptr: all eligible
  T* X;                       ///< (n, P) solutions, column-major; written on exit
  T* L_scratch;               ///< gridDim.x slabs of lawson_slab_elems(np_max)
  const T* A;                 ///< (m, n) column-major matrix; read only if rnorm != nullptr
  const T* B;                 ///< (m, P) right-hand sides; read only if rnorm != nullptr
  T* rnorm;                   ///< (P) residual norms ||b_p - A x_p||_2, or nullptr: skipped
  int m;
  int n;
  int P;
  int np_max;  ///< min(m, n): hard cap on the active-set size (rank bound)
  int max_iter;
  int ld_L;  ///< leading dimension of a slab, lawson_slab_ld(np_max)
  T tol;     ///< tolerance on the projected gradient
};

/**
 * Resident threads per SM of the architecture being compiled for.  ptxas rejects a
 * __launch_bounds__ whose BlockSize * MinBlocks exceeds it: 1024 on sm_75; 2048 on
 * sm_70/72/80/90/100/103; 1536 on sm_86/87/88/89/110/120/121.  Unknown (future)
 * architectures get 1536, which merely asks for a smaller register budget there.
 */
__host__ __device__ constexpr int lawson_max_threads_per_sm()
{
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ == 750
  return 1024;
#elif __CUDA_ARCH__ == 700 || __CUDA_ARCH__ == 720 || __CUDA_ARCH__ == 800 || \
  __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1000 || __CUDA_ARCH__ == 1030
  return 2048;
#else
  return 1536;
#endif
#else
  return 2048;
#endif
}

/** MinBlocks clamped so that BlockSize * MinBlocks fits the SM of the target architecture. */
__host__ __device__ constexpr int lawson_clamp_min_blocks(int block, int min_blocks)
{
  const int cap = lawson_max_threads_per_sm() / block;
  return min_blocks < cap ? min_blocks : (cap > 0 ? cap : 1);
}

/**
 * The kernel.  The grid is persistent: `gridDim.x` blocks stride over the P problems
 * (`for p = blockIdx.x; p < P; p += gridDim.x`), so the factor slabs are bounded by
 * hardware occupancy rather than by the batch size.  Per outer iteration:
 *
 *   (A) gradient + argmax on all warps: each warp reduces its own (max, argmax) and
 *       publishes it; every warp then combines the per-warp partials redundantly, which
 *       is cheaper than a second barrier;
 *   (C) the active-set machinery -- bordering append, back substitution, line search,
 *       compaction, Givens downdates -- in warp 0 alone, with __syncwarp and the
 *       register-tile chains of lawson_warp_fwd / lawson_warp_bwd, while the other warps
 *       wait at the closing barrier.
 *
 * That is two block barriers per outer iteration.  `MinBlocks` is the second argument of
 * __launch_bounds__: the warp-0 solver keeps two 32-element register tiles, which makes
 * the kernel ~128 (float) / ~190 (double) registers wide, and registers rather than
 * shared memory decide how many blocks are co-resident.
 *
 * A pivot rejected as numerically dependent ends the problem with the active set found
 * so far.  The slab is never cleared (see lawson_slab_ensure_open).
 */
template <typename T, int BlockSize, int MinBlocks>
__global__ __launch_bounds__(
  BlockSize,
  lawson_clamp_min_blocks(BlockSize, MinBlocks)) void nnls_lawson_kernel(lawson_args<T> args)
{
  const int n          = args.n;
  const int np_max     = args.np_max;
  const T tol          = args.tol;
  const int ldL        = args.ld_L;
  constexpr int kWarps = BlockSize / raft::WarpSize;

  extern __shared__ unsigned char smem_raw[];
  lawson_smem<T> S = lawson_smem_layout<T>(smem_raw, np_max);
  T* L = args.L_scratch + static_cast<std::size_t>(blockIdx.x) * lawson_slab_elems(np_max);

  __shared__ int sm_n_active;
  __shared__ int sm_flag;  // pivot rejected: stop this problem
  __shared__ int sm_open;  // columns of the factor slab whose diagonal blocks are cleared

  const int tid     = threadIdx.x;
  const int lane    = lawson_warp::mod(tid);
  const int warp    = lawson_warp::div(tid);
  const int n_words = lawson_word::div(lawson_word::roundUp(n));

  const int inner_budget_total = 3 * n + 1;

  if (tid == 0) sm_open = 0;  // ordered before its first use by the barrier below

  // 64-bit: with P close to INT_MAX, `p += gridDim.x` would wrap in 32 bits.
  for (std::int64_t p = blockIdx.x; p < args.P; p += gridDim.x) {
    const T* __restrict__ c = args.C + static_cast<std::size_t>(p) * n;
    const std::uint8_t* mask =
      args.masks != nullptr ? args.masks + static_cast<std::size_t>(p) * n : nullptr;

    for (int w = tid; w < n_words; w += BlockSize)
      S.act[w] = 0u;
    if (tid == 0) {
      sm_n_active = 0;
      sm_flag     = 0;
    }
    __syncthreads();
    int np = 0;  // active-set size; identical in every thread at the top of each iteration

    for (int outer = 0; outer < args.max_iter; ++outer) {
      // ---- (A) fused gradient + argmax -----------------------------------------
      {
        T best;
        int bi;
        lawson_gradient_partial<T, BlockSize>(args.G, c, mask, S.act, S.idx, S.xp, np, n, best, bi);
        lawson_warp_argmax(best, bi);
        if (lane == 0) {
          S.red_val[warp] = best;
          S.red_idx[warp] = bi;
        }
      }
      __syncthreads();  // B1: per-warp partials published
      T max_w;
      int j_star;
      {
        T v   = lane < kWarps ? S.red_val[lane] : -std::numeric_limits<T>::infinity();
        int i = lane < kWarps ? S.red_idx[lane] : -1;
        lawson_warp_argmax(v, i);
        max_w  = v;
        j_star = i;
      }
      // `np >= np_max`: every remaining column is linearly dependent on the active ones.
      // np is a register, identical in every thread, so the exit is unanimous.
      if (j_star < 0 || max_w <= tol || np >= np_max) break;

      // ---- (C) active-set update, warp 0 ---------------------------------------
      if (warp == 0) {
        if (lane == 0) {
          lawson_act_set(S.act, j_star);
          S.idx[np] = j_star;
          S.xp[np]  = T(0);
        }
        __syncwarp();
        int npc = np + 1;
        lawson_slab_ensure_open(L, ldL, np_max, np, &sm_open);
        const bool ok =
          lawson_warp_append(L, ldL, npc, args.G, n, S.idx, c, S.y, S.dg, S.dinv, S.s);
        if (!ok) {
          // The column is numerically dependent: undo the activation, stop this problem.
          if (lane == 0) {
            lawson_act_clear(S.act, j_star);
            sm_flag = 1;
          }
          npc = np;
        } else {
          // Inner loop: solve the unconstrained active-set problem and, while any
          // coordinate is negative, take the largest feasible step and drop the
          // coordinates that hit zero until the trial solution is non-negative.
          for (int inner = 0; inner < inner_budget_total; ++inner) {
            lawson_warp_bwd(L, ldL, npc, S.dinv, S.y, S.s);
            const int rc =
              lawson_post_solve(L, ldL, npc, S.xp, S.s, S.y, S.dg, S.dinv, S.idx, S.rem, S.act);
            if (rc != 1) break;
          }
        }
        if (lane == 0) sm_n_active = npc;
      }
      __syncthreads();  // B2: the solver's results (np, idx, xp, act, flag) published
      np = sm_n_active;
      if (sm_flag) break;
    }

    // Write back: zero-fill the column, barrier, then scatter the compact solution.  The
    // second barrier keeps the next problem's re-initialisation from overtaking a slow
    // thread's scatter that still reads S.idx / S.xp.
    T* __restrict__ Xp = args.X + static_cast<std::size_t>(p) * n;
    for (int j = tid; j < n; j += BlockSize)
      Xp[j] = T(0);
    __syncthreads();
    for (int kk = tid; kk < np; kk += BlockSize)
      Xp[S.idx[kk]] = S.xp[kk];
    __syncthreads();

    // Residual norm ||b - A x||_2 from the np active columns only, O(m * np) per problem
    // (a closed form from G and c would cancel badly in float32 for a good fit).  S.idx and
    // S.xp are still intact: the next problem overwrites them only after its first barrier,
    // which every thread reaches after its last read of S.red_val below.
    if (args.rnorm != nullptr) {
      const T* __restrict__ b = args.B + static_cast<std::size_t>(p) * args.m;
      // Sum of squares of (b - A x) / scale and the largest |b - A x| of this thread's rows.
      auto partial = [&](auto scaled, T scale, T& sq, T& mx) {
        sq = T(0);
        mx = T(0);
        // Four rows per thread at a time: independent chains hide the load latency, which
        // dominates for tall A (np is small).
        constexpr int kRows = 4;
        for (std::int64_t i0 = tid; i0 < args.m; i0 += kRows * BlockSize) {  // 64-bit: m <= INT_MAX
          T r[kRows];
#pragma unroll
          for (int u = 0; u < kRows; ++u) {
            const std::int64_t i = i0 + u * BlockSize;
            r[u]                 = i < args.m ? b[i] : T(0);
          }
          for (int kk = 0; kk < np; ++kk) {
            const T* __restrict__ col = args.A + static_cast<std::size_t>(S.idx[kk]) * args.m;
            const T xk                = S.xp[kk];
#pragma unroll
            for (int u = 0; u < kRows; ++u) {
              const std::int64_t i = i0 + u * BlockSize;
              if (i < args.m) r[u] -= col[i] * xk;
            }
          }
#pragma unroll
          for (int u = 0; u < kRows; ++u) {
            if constexpr (decltype(scaled)::value) {
              const T v = r[u] / scale;
              sq += v * v;
            } else {
              sq += r[u] * r[u];
            }
            mx = max(mx, abs(r[u]));
          }
        }
      };
      // Block-wide sum and max, available to every thread; `last` skips the barrier that
      // protects S.red_val from being overwritten by the next use (the caller adds one).
      auto reduce = [&](T sq, T mx, T& total, T& mx_all, bool last) {
        static_assert(2 * kWarps <= raft::WarpSize);
        sq = raft::warpReduce(sq);
        mx = raft::warpReduce(mx, raft::max_op{});
        if (lane == 0) {
          S.red_val[warp]          = sq;
          S.red_val[kWarps + warp] = mx;
        }
        __syncthreads();
        total  = T(0);
        mx_all = T(0);
        for (int w = 0; w < kWarps; ++w) {
          total += S.red_val[w];
          mx_all = max(mx_all, S.red_val[kWarps + w]);
        }
        if (!last) __syncthreads();
      };
      T sq, mx, total, mx_all;
      partial(std::false_type{}, T(1), sq, mx);
      reduce(sq, mx, total, mx_all, true);
      // The plain sum of squares overflows (or loses everything to underflow) when the
      // residual is far from unit scale: redo it relative to the largest entry.  Rare, so
      // the fast path above stays a bare FMA chain.  The decision is block-uniform.
      constexpr T kTiny = std::numeric_limits<T>::min() / std::numeric_limits<T>::epsilon();
      T rn              = std::sqrt(total);
      if (!isfinite(total) || (mx_all > T(0) && total < kTiny)) {
        if (isfinite(mx_all) && mx_all > T(0)) {
          __syncthreads();  // everyone has read S.red_val of the first reduction
          partial(std::true_type{}, mx_all, sq, mx);
          reduce(sq, mx, total, mx_all, true);
          rn = mx_all * std::sqrt(total);
        }
      }
      if (tid == 0) args.rnorm[p] = rn;
    }
  }
}

}  // namespace raft::solver::detail
