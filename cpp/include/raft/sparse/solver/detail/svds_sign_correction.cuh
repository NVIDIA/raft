/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>

#include <cstdint>
#include <optional>

namespace raft::sparse::solver::detail {

/**
 * @brief CUDA kernel for SVD sign correction.
 *
 * For each component i, finds the element with largest absolute value in either
 * column i of U (if U is not null) or row i of Vt. If that element is negative,
 * flips the sign of U[:, i] (when present) and Vt[i, :] (when present).
 * This ensures deterministic output regardless of the SVD algorithm's internal sign choices.
 *
 * One thread block per component. Uses shared memory reduction to find the argmax.
 *
 * @param U left singular vectors, col-major (m x k), U[:, i] is at U + i*m. May be nullptr.
 * @param Vt right singular vectors, col-major (k x n), Vt[i, :] is row i. May be nullptr.
 * @param m number of rows of U (ignored if U is nullptr)
 * @param n number of columns of Vt (ignored if Vt is nullptr)
 * @param k number of components
 *
 * @note At least one of U / Vt must be non-null. When U is non-null it is used to
 *       derive the signs; otherwise Vt is used.
 */
template <typename ValueTypeT>
RAFT_KERNEL svd_sign_correction_kernel(ValueTypeT* U, ValueTypeT* Vt, int m, int n, int k)
{
  int comp = blockIdx.x;  // one block per component
  if (comp >= k) return;

  extern __shared__ char smem[];
  auto* s_max_val = reinterpret_cast<ValueTypeT*>(smem);
  auto* s_max_idx = reinterpret_cast<int*>(smem + blockDim.x * sizeof(ValueTypeT));

  ValueTypeT local_max_val = 0;
  int local_max_idx        = 0;

  // Source for sign decision: prefer U column; fall back to Vt row
  if (U != nullptr) {
    ValueTypeT* u_col = U + static_cast<int64_t>(comp) * m;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
      ValueTypeT abs_val = raft::abs(u_col[row]);
      if (abs_val > local_max_val) {
        local_max_val = abs_val;
        local_max_idx = row;
      }
    }
  } else {
    // Vt is col-major (k x n); row `comp` is at Vt[comp + j*k] for j=0..n-1
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
      ValueTypeT abs_val = raft::abs(Vt[comp + static_cast<int64_t>(col) * k]);
      if (abs_val > local_max_val) {
        local_max_val = abs_val;
        local_max_idx = col;
      }
    }
  }

  s_max_val[threadIdx.x] = local_max_val;
  s_max_idx[threadIdx.x] = local_max_idx;
  __syncthreads();

  // Tree reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (s_max_val[threadIdx.x + stride] > s_max_val[threadIdx.x]) {
        s_max_val[threadIdx.x] = s_max_val[threadIdx.x + stride];
        s_max_idx[threadIdx.x] = s_max_idx[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  // Thread 0 has the result
  ValueTypeT signed_val;
  if (U != nullptr) {
    signed_val = U[static_cast<int64_t>(comp) * m + s_max_idx[0]];
  } else {
    signed_val = Vt[comp + static_cast<int64_t>(s_max_idx[0]) * k];
  }
  bool flip = (s_max_val[0] > 0) && (signed_val < 0);
  __syncthreads();

  if (!flip) return;

  // Flip U column if present
  if (U != nullptr) {
    ValueTypeT* u_col = U + static_cast<int64_t>(comp) * m;
    for (int row = threadIdx.x; row < m; row += blockDim.x) {
      u_col[row] = -u_col[row];
    }
  }

  // Flip Vt row if present: row `comp` is at Vt[comp + j*k] for j=0..n-1
  if (Vt != nullptr) {
    for (int col = threadIdx.x; col < n; col += blockDim.x) {
      Vt[comp + static_cast<int64_t>(col) * k] = -Vt[comp + static_cast<int64_t>(col) * k];
    }
  }
}

/**
 * @brief Apply deterministic sign correction to SVD output.
 *
 * For each component, ensures the element with largest absolute value in U
 * (or Vt, if U is not present) is positive. Whichever of U / Vt is present is
 * modified in-place to maintain A ≈ U @ diag(S) @ Vt.
 *
 * @param handle raft resources handle
 * @param U optional left singular vectors of shape (m, k), col-major, modified in-place
 * @param Vt optional right singular vectors of shape (k, n), col-major, modified in-place
 *
 * @note If both U and Vt are absent, this is a no-op. When only Vt is present, the
 *       sign convention differs from the both-present case (signs are derived from
 *       `argmax(|Vt[j, :]|)` rather than `argmax(|U[:, j]|)`), but is still
 *       deterministic for fixed inputs.
 */
template <typename ValueTypeT>
void svd_sign_correction(
  raft::resources const& handle,
  std::optional<raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> U,
  std::optional<raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>> Vt)
{
  if (!U && !Vt) return;

  int k = U ? static_cast<int>(U->extent(1)) : static_cast<int>(Vt->extent(0));
  int m = U ? static_cast<int>(U->extent(0)) : 0;
  int n = Vt ? static_cast<int>(Vt->extent(1)) : 0;

  auto stream = raft::resource::get_cuda_stream(handle);

  // threads_per_block must be a power of 2 for the tree reduction in the kernel
  constexpr int threads_per_block = 256;
  int smem_size                   = threads_per_block * (sizeof(ValueTypeT) + sizeof(int));

  ValueTypeT* U_ptr  = U ? U->data_handle() : nullptr;
  ValueTypeT* Vt_ptr = Vt ? Vt->data_handle() : nullptr;

  if (!raft::resource::get_dry_run_flag(handle)) {
    svd_sign_correction_kernel<<<k, threads_per_block, smem_size, stream>>>(U_ptr, Vt_ptr, m, n, k);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

}  // namespace raft::sparse::solver::detail
