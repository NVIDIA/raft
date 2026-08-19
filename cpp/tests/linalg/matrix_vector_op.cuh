/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

namespace raft {
namespace linalg {

template <typename OutT, typename MatT, typename VecT, typename Lambda, typename IdxType = int>
RAFT_KERNEL naiveMatVecKernel(OutT* out,
                              const MatT* mat,
                              const VecT* vec,
                              IdxType D,
                              IdxType N,
                              bool rowMajor,
                              bool bcastAlongRows,
                              Lambda op)
{
  IdxType idx = threadIdx.x + blockIdx.x * blockDim.x;
  IdxType len = N * D;
  IdxType col;
  if (rowMajor && bcastAlongRows) {
    col = idx % D;
  } else if (!rowMajor && !bcastAlongRows) {
    col = idx % N;
  } else if (rowMajor && !bcastAlongRows) {
    col = idx / D;
  } else {
    col = idx / N;
  }
  if (idx < len) { out[idx] = op(mat[idx], vec[col]); }
}

template <typename OutT, typename MatT, typename VecT, typename Lambda, typename IdxType = int>
void naiveMatVec(OutT* out,
                 const MatT* mat,
                 const VecT* vec,
                 IdxType D,
                 IdxType N,
                 bool rowMajor,
                 bool bcastAlongRows,
                 Lambda op,
                 cudaStream_t stream)
{
  static const IdxType TPB = 64;
  IdxType len              = N * D;
  IdxType nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(
    stream, nblks, TPB, naiveMatVecKernel, out, mat, vec, D, N, rowMajor, bcastAlongRows, op);
}

template <typename OutT, typename MatT, typename VecT, typename IdxType = int>
void naiveMatVec(OutT* out,
                 const MatT* mat,
                 const VecT* vec,
                 IdxType D,
                 IdxType N,
                 bool rowMajor,
                 bool bcastAlongRows,
                 OutT scalar,
                 cudaStream_t stream)
{
  naiveMatVec(
    out,
    mat,
    vec,
    D,
    N,
    rowMajor,
    bcastAlongRows,
    [scalar] __device__(MatT a, VecT b) { return (OutT)(a + scalar * b); },
    stream);
}

template <typename OutT,
          typename MatT,
          typename Vec1T,
          typename Vec2T,
          typename Lambda,
          typename IdxType = int>
RAFT_KERNEL naiveMatVecKernel(OutT* out,
                              const MatT* mat,
                              const Vec1T* vec1,
                              const Vec2T* vec2,
                              IdxType D,
                              IdxType N,
                              bool rowMajor,
                              bool bcastAlongRows,
                              Lambda op)
{
  IdxType idx = threadIdx.x + blockIdx.x * blockDim.x;
  IdxType len = N * D;
  IdxType col;
  if (rowMajor && bcastAlongRows) {
    col = idx % D;
  } else if (!rowMajor && !bcastAlongRows) {
    col = idx % N;
  } else if (rowMajor && !bcastAlongRows) {
    col = idx / D;
  } else {
    col = idx / N;
  }
  if (idx < len) { out[idx] = op(mat[idx], vec1[col], vec2[col]); }
}

template <typename OutT,
          typename MatT,
          typename Vec1T,
          typename Vec2T,
          typename Lambda,
          typename IdxType = int>
void naiveMatVec(OutT* out,
                 const MatT* mat,
                 const Vec1T* vec1,
                 const Vec2T* vec2,
                 IdxType D,
                 IdxType N,
                 bool rowMajor,
                 bool bcastAlongRows,
                 Lambda op,
                 cudaStream_t stream)
{
  static const IdxType TPB = 64;
  IdxType len              = N * D;
  IdxType nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(stream,
                      nblks,
                      TPB,
                      naiveMatVecKernel,
                      out,
                      mat,
                      vec1,
                      vec2,
                      D,
                      N,
                      rowMajor,
                      bcastAlongRows,
                      op);
}

template <typename OutT, typename MatT, typename Vec1T, typename Vec2T, typename IdxType = int>
void naiveMatVec(OutT* out,
                 const MatT* mat,
                 const Vec1T* vec1,
                 const Vec2T* vec2,
                 IdxType D,
                 IdxType N,
                 bool rowMajor,
                 bool bcastAlongRows,
                 OutT scalar,
                 cudaStream_t stream)
{
  naiveMatVec(
    out,
    mat,
    vec1,
    vec2,
    D,
    N,
    rowMajor,
    bcastAlongRows,
    [scalar] __device__(MatT a, Vec1T b, Vec2T c) { return (OutT)(a + scalar * b + c); },
    stream);
}

}  // end namespace linalg
}  // end namespace raft
