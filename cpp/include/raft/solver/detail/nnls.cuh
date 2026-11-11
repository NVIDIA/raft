/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/solver/detail/nnls_lawson.cuh>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <set>
#include <utility>

namespace raft::solver::detail {

template <typename T>
using lawson_kernel_t = void (*)(lawson_args<T>);

/** Launch configuration: block size and `MinBlocks` (the register cap, see the kernel). */
struct lawson_cfg {
  int block;
  int min_blocks;
};

constexpr int kLawsonConfigCount = 3;

/**
 * The menu of launch configurations of each precision.  Shared memory is independent of
 * the block size, so the block size is how many threads cooperate on one problem, and
 * registers decide how many blocks are co-resident.  The work per problem has two parts of
 * opposite shape -- the n x np gradient pass, which wants many threads, and the O(np)-deep
 * recurrences, which run in one warp -- so the best block grows with n.  The smallest
 * blocks also differ in the register cap: a batch that needs more than 8 blocks per SM
 * wants the 16-block cap in float64.
 */
template <typename T>
constexpr lawson_cfg lawson_cfg_at(int i)
{
  constexpr lawson_cfg f32[kLawsonConfigCount] = {{32, 1}, {64, 1}, {256, 8}};
  constexpr lawson_cfg f64[kLawsonConfigCount] = {{32, 1}, {32, 16}, {128, 8}};
  return sizeof(T) == 4 ? f32[i] : f64[i];
}

template <typename T>
lawson_kernel_t<T> lawson_kernel_for(int cfg)
{
  if constexpr (sizeof(T) == 4) {
    switch (cfg) {
      case 0: return &nnls_lawson_kernel<T, 32, 1>;
      case 1: return &nnls_lawson_kernel<T, 64, 1>;
      default: return &nnls_lawson_kernel<T, 256, 8>;
    }
  } else {
    switch (cfg) {
      case 0: return &nnls_lawson_kernel<T, 32, 1>;
      case 1: return &nnls_lawson_kernel<T, 32, 16>;
      default: return &nnls_lawson_kernel<T, 128, 8>;
    }
  }
}

/** Menu index for `n` columns and a batch of `n_problems` on a device with `n_sm` SMs. */
template <typename T>
int lawson_choose_config(int n, int n_problems, int n_sm)
{
  constexpr bool kFloat = sizeof(T) == 4;
  const bool deep       = raft::ceildiv<std::int64_t>(n_problems, n_sm) > 8;
  if (n >= 1024 && (kFloat || n >= 4096 || !deep)) return 2;
  if (kFloat) return (n > 32 && !deep) ? 1 : 0;
  return deep ? 1 : 0;
}

/**
 * Allow `kernel` more dynamic shared memory than the 48 KiB default, to the device's
 * maximum, once per process and device.  The attribute is a property of the function in the
 * CUDA context, so it is raised for good and only when a shape needs it: never lowered
 * (another caller's launch could depend on it) and never on the launch path of the small
 * shapes, where it would be a context-mutating call.
 */
template <typename T>
void lawson_grant_smem(lawson_kernel_t<T> kernel, int device, std::size_t optin_bytes)
{
  static std::mutex mu;
  static std::set<std::pair<const void*, int>> granted;
  std::lock_guard<std::mutex> lock(mu);
  if (!granted.insert({reinterpret_cast<const void*>(kernel), device}).second) { return; }
  // The opt-in limit is for static and dynamic shared memory together.
  cudaFuncAttributes attr{};
  RAFT_CUDA_TRY(cudaFuncGetAttributes(&attr, kernel));
  RAFT_CUDA_TRY(cudaFuncSetAttribute(kernel,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(optin_bytes - attr.sharedSizeBytes)));
}

template <typename T>
void nnls(
  raft::resources const& res,
  double tol,
  int max_iter,
  raft::device_matrix_view<const T, std::int64_t, raft::col_major> A,
  raft::device_matrix_view<const T, std::int64_t, raft::col_major> B,
  raft::device_matrix_view<T, std::int64_t, raft::col_major> X,
  std::optional<raft::device_matrix_view<const std::uint8_t, std::int64_t, raft::col_major>> masks,
  std::optional<raft::device_matrix_view<T, std::int64_t, raft::col_major>> fitted)
{
  using index_t   = std::int64_t;
  const index_t m = A.extent(0), n = A.extent(1), P = B.extent(1);
  RAFT_EXPECTS(m >= 1 && n >= 1 && P >= 1, "nnls: m, n and the number of problems must be >= 1");
  RAFT_EXPECTS(n <= std::numeric_limits<int>::max() && P <= std::numeric_limits<int>::max(),
               "nnls: n and the number of problems must each fit in an int");
  RAFT_EXPECTS(B.extent(0) == m, "nnls: B must have m = %ld rows", static_cast<long>(m));
  RAFT_EXPECTS(X.extent(0) == n && X.extent(1) == P, "nnls: X must be n x P");
  RAFT_EXPECTS(!masks || (masks->extent(0) == n && masks->extent(1) == P),
               "nnls: masks must be n x P");
  RAFT_EXPECTS(!fitted || (fitted->extent(0) == m && fitted->extent(1) == P),
               "nnls: fitted must be m x P");

  // The active set stays linearly independent, so it never holds more than rank(A) <=
  // min(m, n) columns, however large n is.
  const int np_max = static_cast<int>(std::min(m, n));
  const int n_int  = static_cast<int>(n);
  const int P_int  = static_cast<int>(P);

  // Fail early and clearly, before anything is allocated or computed (the Gram matrix of a
  // problem that cannot run is hundreds of MiB and a teraflop).
  const auto& props       = raft::resource::get_device_properties(res);
  const std::size_t smem  = lawson_smem_bytes<T>(n_int, np_max);
  const std::size_t limit = props.sharedMemPerBlockOptin;
  RAFT_EXPECTS(smem <= limit,
               "nnls: min(m, n) = %d is too large for the batched Lawson-Hanson kernel on this "
               "device: its active-set state needs %zu B of shared memory per block (O(min(m, n))) "
               "but a block may use at most %zu B, which allows min(m, n) of up to about %d here "
               "(n = %d). Reduce the problem, e.g. by screening out columns or rows first.",
               np_max,
               smem,
               limit,
               lawson_max_np<T>(limit, n_int),
               n_int);

  // Launch configuration and the number of blocks that can be resident at once.
  int cfg                   = lawson_choose_config<T>(n_int, P_int, props.multiProcessorCount);
  int blocks_per_sm         = 0;
  lawson_kernel_t<T> kernel = nullptr;
  for (; cfg >= 0; --cfg) {
    kernel = lawson_kernel_for<T>(cfg);
    if (smem > static_cast<std::size_t>(props.sharedMemPerBlock)) {
      int device = 0;
      RAFT_CUDA_TRY(cudaGetDevice(&device));
      lawson_grant_smem<T>(kernel, device, limit);
    }
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, lawson_cfg_at<T>(cfg).block, smem));
    if (blocks_per_sm > 0) break;  // (static shared memory differs between configurations)
  }
  RAFT_EXPECTS(blocks_per_sm > 0,
               "nnls: no block of the Lawson-Hanson kernel can be placed on an SM for n = %d, "
               "min(m, n) = %d (%zu B of dynamic shared memory). Reduce min(m, n).",
               n_int,
               np_max,
               smem);
  const int block = lawson_cfg_at<T>(cfg).block;

  // One allocation holds everything: the Gram matrix G = A^T A (n x n) and the projections
  // C = A^T B (n x P), formed once and shared by the whole batch, and one factor slab per
  // resident block.  A smaller block raises the resident count several fold, so the grid is
  // capped at a quarter of the free workspace (the persistent loop handles any grid, taking
  // more passes over the batch instead).  The buffer comes from the workspace when it fits
  // and from the large workspace otherwise: an allocation per call is noise next to a big
  // solve, but at the production shapes it can cost as much as the solve itself.
  auto aligned = [](std::size_t elems) { return (elems + 7) / 8 * 8; };  // 32-64 B boundaries
  const std::size_t g_elems    = aligned(static_cast<std::size_t>(n) * n);
  const std::size_t c_elems    = aligned(static_cast<std::size_t>(n) * P);
  const std::size_t slab_elems = lawson_slab_elems(np_max);
  const std::size_t ws_free    = raft::resource::get_workspace_free_bytes(res);
  long long grid =
    std::min<long long>(P, static_cast<long long>(blocks_per_sm) * props.multiProcessorCount);
  grid = std::max<long long>(1, std::min<long long>(grid, ws_free / 4 / (slab_elems * sizeof(T))));
  const std::size_t total = g_elems + c_elems + static_cast<std::size_t>(grid) * slab_elems;
  auto buffer             = raft::make_device_mdarray<T, index_t>(
    res,
    total * sizeof(T) <= ws_free / 2 ? raft::resource::get_workspace_resource_ref(res)
                                                 : raft::resource::get_large_workspace_resource_ref(res),
    raft::make_extents<index_t>(static_cast<index_t>(total)));
  T* G_ptr    = buffer.data_handle();
  T* C_ptr    = G_ptr + g_elems;
  T* slab_ptr = C_ptr + c_elems;

  // A col-major (m, n) buffer viewed as row-major (n, m) is A^T.
  const bool dry_run = raft::resource::get_dry_run_flag(res);
  if (!dry_run) {
    auto At =
      raft::make_device_matrix_view<const T, index_t, raft::row_major>(A.data_handle(), n, m);
    auto G = raft::make_device_matrix_view<T, index_t, raft::col_major>(G_ptr, n, n);
    auto C = raft::make_device_matrix_view<T, index_t, raft::col_major>(C_ptr, n, P);
    raft::linalg::gemm(res, At, A, G);
    raft::linalg::gemm(res, At, B, C);
  }

  lawson_args<T> args{G_ptr,
                      C_ptr,
                      masks ? masks->data_handle() : nullptr,
                      X.data_handle(),
                      slab_ptr,
                      n_int,
                      P_int,
                      np_max,
                      max_iter > 0 ? max_iter : 3 * n_int + 1,
                      lawson_slab_ld(np_max),
                      static_cast<T>(tol)};
  raft::launch_kernel(
    {res, smem}, static_cast<unsigned>(grid), static_cast<unsigned>(block), kernel, args);

  if (fitted && !dry_run) {
    auto Xc = raft::make_const_mdspan(X);
    raft::linalg::gemm(res, A, Xc, *fitted);
  }
}

}  // namespace raft::solver::detail
