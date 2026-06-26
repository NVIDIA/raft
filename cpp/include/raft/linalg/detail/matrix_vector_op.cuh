/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/matrix/linewise_op.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <bool rowMajor,
          bool bcastAlongRows,
          typename MatT,
          typename Lambda,
          typename VecT,
          typename IdxType = int,
          int TPB          = 256>
void matrixVectorOp(bool dry_run,
                    MatT* out,
                    const MatT* matrix,
                    const VecT* vec,
                    IdxType D,
                    IdxType N,
                    Lambda op,
                    cudaStream_t stream)
{
  if (dry_run) { return; }
  raft::resources handle;
  resource::set_cuda_stream(handle, stream);
  constexpr raft::Apply apply =
    rowMajor == bcastAlongRows ? raft::Apply::ALONG_ROWS : raft::Apply::ALONG_COLUMNS;
  if constexpr (rowMajor) {
    matrix::linewise_op<apply, MatT, IdxType, row_major, Lambda>(
      handle,
      make_device_matrix_view<const MatT, IdxType, row_major>(matrix, N, D),
      make_device_matrix_view<MatT, IdxType, row_major>(out, N, D),
      op,
      make_device_vector_view<const VecT, IdxType>(vec, bcastAlongRows ? N : D));
  } else {
    matrix::linewise_op<apply, MatT, IdxType, col_major, Lambda>(
      handle,
      make_device_matrix_view<const MatT, IdxType, col_major>(matrix, N, D),
      make_device_matrix_view<MatT, IdxType, col_major>(out, N, D),
      op,
      make_device_vector_view<const VecT, IdxType>(vec, bcastAlongRows ? N : D));
  }
}

template <bool rowMajor,
          bool bcastAlongRows,
          typename MatT,
          typename Lambda,
          typename Vec1T,
          typename Vec2T,
          typename IdxType = int,
          int TPB          = 256>
void matrixVectorOp(bool dry_run,
                    MatT* out,
                    const MatT* matrix,
                    const Vec1T* vec1,
                    const Vec2T* vec2,
                    IdxType D,
                    IdxType N,
                    Lambda op,
                    cudaStream_t stream)
{
  if (dry_run) { return; }
  raft::resources handle;
  resource::set_cuda_stream(handle, stream);
  constexpr raft::Apply apply =
    rowMajor == bcastAlongRows ? raft::Apply::ALONG_ROWS : raft::Apply::ALONG_COLUMNS;
  if constexpr (rowMajor) {
    matrix::linewise_op<apply, MatT, IdxType, row_major, Lambda>(
      handle,
      make_device_matrix_view<const MatT, IdxType, row_major>(matrix, N, D),
      make_device_matrix_view<MatT, IdxType, row_major>(out, N, D),
      op,
      make_device_vector_view<const Vec1T, IdxType>(vec1, bcastAlongRows ? N : D),
      make_device_vector_view<const Vec2T, IdxType>(vec2, bcastAlongRows ? N : D));
  } else {
    matrix::linewise_op<apply, MatT, IdxType, col_major, Lambda>(
      handle,
      make_device_matrix_view<const MatT, IdxType, col_major>(matrix, N, D),
      make_device_matrix_view<MatT, IdxType, col_major>(out, N, D),
      op,
      make_device_vector_view<const Vec1T, IdxType>(vec1, bcastAlongRows ? N : D),
      make_device_vector_view<const Vec2T, IdxType>(vec2, bcastAlongRows ? N : D));
  }
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
