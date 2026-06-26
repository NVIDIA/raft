/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cublas_wrappers.hpp"
#include "cusolver_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/binary_op.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename math_t>
void choleskyRank1Update(raft::resources const& handle,
                         math_t* L,
                         int n,
                         int ld,
                         void* workspace,
                         int* n_bytes,
                         cublasFillMode_t uplo,
                         cudaStream_t stream,
                         math_t eps = -1)
{
  // The matrix A' is defined as:
  // A' = [[A_11, A_12]
  //       [A_21, A_22]]
  // where:
  // - A_11 = A, matrix of size (n-1)x(n-1)
  // - A_21[j] = A_12.T[j] = A_new[j] j=0..n-2, vector with (n-1) elements
  // - A_22 = A_new[n-1] scalar.
  //
  // Instead of calculating the Cholelsky decomposition of A' from scratch,
  // we just update L with the new row. The new Cholesky decomposition will be
  // calculated as:
  // L' = [[L_11,    0]
  //       [L_12, L_22]]
  // where L_11 is the Cholesky decomposition of A (size [n-1 x n-1]), and
  // L_12 and L_22 are the new quantities that we need to calculate.

  // We need a workspace in device memory to store a scalar. Additionally, in
  // CUBLAS_FILL_MODE_LOWER we need space for n-1 floats.
  const int align = 256;
  int offset =
    (uplo == CUBLAS_FILL_MODE_LOWER) ? raft::alignTo<int>(sizeof(math_t) * (n - 1), align) : 0;
  if (workspace == nullptr) {
    *n_bytes = offset + 1 * sizeof(math_t);
    return;
  }
  if (resource::get_dry_run_flag(handle)) { return; }
  math_t* s    = reinterpret_cast<math_t*>(((char*)workspace) + offset);
  math_t* L_22 = L + (n - 1) * ld + n - 1;

  math_t* A_new = nullptr;
  math_t* A_row = nullptr;
  if (uplo == CUBLAS_FILL_MODE_UPPER) {
    // A_new is stored as the n-1 th column of L
    A_new = L + (n - 1) * ld;
  } else {
    // If the input is lower triangular, then the new elements of A are stored
    // as the n-th row of L. Since the matrix is column major, this is non
    // contiguous. We copy elements from A_row to a contiguous workspace A_new.
    A_row = L + n - 1;
    A_new = reinterpret_cast<math_t*>(workspace);
    RAFT_CUBLAS_TRY(
      cublasCopy(resource::get_cublas_handle(handle), n - 1, A_row, ld, A_new, 1, stream));
  }
  cublasOperation_t op = (uplo == CUBLAS_FILL_MODE_UPPER) ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (n > 1) {
    // Calculate L_12 = x by solving equation L_11 x = A_12
    math_t alpha = 1;
    RAFT_CUBLAS_TRY(cublastrsm(resource::get_cublas_handle(handle),
                               CUBLAS_SIDE_LEFT,
                               uplo,
                               op,
                               CUBLAS_DIAG_NON_UNIT,
                               n - 1,
                               1,
                               &alpha,
                               L,
                               ld,
                               A_new,
                               n - 1,
                               stream));

    // A_new now stores L_12, we calculate s = L_12 * L_12
    RAFT_CUBLAS_TRY(
      cublasdot(resource::get_cublas_handle(handle), n - 1, A_new, 1, A_new, 1, s, stream));

    if (uplo == CUBLAS_FILL_MODE_LOWER) {
      // Copy back the L_12 elements as the n-th row of L
      RAFT_CUBLAS_TRY(
        cublasCopy(resource::get_cublas_handle(handle), n - 1, A_new, 1, A_row, ld, stream));
    }
  } else {  // n == 1 case
    RAFT_CUDA_TRY(cudaMemsetAsync(s, 0, sizeof(math_t), stream));
  }

  // L_22 = sqrt(A_22 - L_12 * L_12)
  math_t s_host;
  math_t L_22_host;
  raft::update_host(&s_host, s, 1, stream);
  raft::update_host(&L_22_host, L_22, 1, stream);  // L_22 stores A_22
  resource::sync_stream(handle, stream);
  L_22_host = std::sqrt(L_22_host - s_host);

  // Check for numeric error with sqrt. If the matrix is not positive definite or
  // the system is very ill conditioned then the A_22 - L_12 * L_12 can be
  // negative, which would result L_22 = NaN. A small positive eps parameter
  // can be used to prevent this.
  if (eps >= 0 && (std::isnan(L_22_host) || L_22_host < eps)) { L_22_host = eps; }
  ASSERT(!std::isnan(L_22_host), "Error during Cholesky rank one update");
  raft::update_device(L_22, &L_22_host, 1, stream);
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft
