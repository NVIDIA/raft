/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>

namespace raft {
namespace matrix {
namespace detail {

template <typename math_t>
void power(math_t* in, math_t* out, math_t scalar, int len, cudaStream_t stream)
{
  auto d_src  = in;
  auto d_dest = out;

  raft::linalg::binaryOp(
    d_dest,
    d_src,
    d_src,
    len,
    [=] __device__(math_t a, math_t b) { return scalar * a * b; },
    stream);
}

template <typename math_t>
void power(math_t* inout, math_t scalar, int len, cudaStream_t stream)
{
  power(inout, inout, scalar, len, stream);
}

template <typename math_t>
void power(math_t* inout, int len, cudaStream_t stream)
{
  math_t scalar = 1.0;
  power(inout, scalar, len, stream);
}

template <typename math_t>
void power(math_t* in, math_t* out, int len, cudaStream_t stream)
{
  math_t scalar = 1.0;
  power(in, out, scalar, len, stream);
}

template <typename math_t, typename IdxType = int>
void seqRoot(const math_t* in,
             math_t* out,
             math_t scalar,
             IdxType len,
             cudaStream_t stream,
             bool set_neg_zero = false)
{
  auto d_src  = in;
  auto d_dest = out;

  raft::linalg::unaryOp(
    d_dest,
    d_src,
    len,
    [=] __device__(math_t a) {
      if (set_neg_zero) {
        if (a < math_t(0)) {
          return math_t(0);
        } else {
          return raft::sqrt(a * scalar);
        }
      } else {
        return raft::sqrt(a * scalar);
      }
    },
    stream);
}

template <typename math_t, typename IdxType = int>
void seqRoot(
  math_t* inout, math_t scalar, IdxType len, cudaStream_t stream, bool set_neg_zero = false)
{
  seqRoot(inout, inout, scalar, len, stream, set_neg_zero);
}

template <typename math_t, typename IdxType = int>
void seqRoot(const math_t* in, math_t* out, IdxType len, cudaStream_t stream)
{
  math_t scalar = 1.0;
  seqRoot(in, out, scalar, len, stream);
}

template <typename math_t, typename IdxType = int>
void seqRoot(math_t* inout, IdxType len, cudaStream_t stream)
{
  math_t scalar = 1.0;
  seqRoot(inout, inout, scalar, len, stream);
}

template <typename math_t, typename IdxType = int>
void setSmallValuesZero(
  math_t* out, const math_t* in, IdxType len, cudaStream_t stream, math_t thres = 1e-15)
{
  raft::linalg::unaryOp(
    out,
    in,
    len,
    [=] __device__(math_t a) {
      if (a <= thres && -a <= thres) {
        return math_t(0);
      } else {
        return a;
      }
    },
    stream);
}

template <typename math_t, typename IdxType = int>
void setSmallValuesZero(math_t* inout, IdxType len, cudaStream_t stream, math_t thres = 1e-15)
{
  setSmallValuesZero(inout, inout, len, stream, thres);
}

template <typename math_t, typename IdxType = int>
void reciprocal(const math_t* in,
                math_t* out,
                math_t scalar,
                int len,
                cudaStream_t stream,
                bool setzero = false,
                math_t thres = 1e-15)
{
  auto d_src  = in;
  auto d_dest = out;

  raft::linalg::unaryOp(
    d_dest,
    d_src,
    len,
    [=] __device__(math_t a) { return setzero && (abs(a) <= thres) ? math_t{0} : scalar / a; },
    stream);
}

template <typename math_t, typename IdxType = int>
void reciprocal(math_t* inout,
                math_t scalar,
                IdxType len,
                cudaStream_t stream,
                bool setzero = false,
                math_t thres = 1e-15)
{
  reciprocal(inout, inout, scalar, len, stream, setzero, thres);
}

template <typename math_t, typename IdxType = int>
void reciprocal(math_t* inout, IdxType len, cudaStream_t stream)
{
  math_t scalar = 1.0;
  reciprocal(inout, scalar, len, stream);
}

template <typename math_t, typename IdxType = int>
void reciprocal(math_t* in, math_t* out, IdxType len, cudaStream_t stream)
{
  math_t scalar = 1.0;
  reciprocal(in, out, scalar, len, stream);
}

template <typename math_t>
void setValue(math_t* out, const math_t* in, math_t scalar, int len, cudaStream_t stream = 0)
{
  raft::linalg::unaryOp(out, in, len, raft::const_op(scalar), stream);
}

template <typename math_t, typename IdxType = int>
void ratio(
  raft::resources const& handle, const math_t* src, math_t* dest, IdxType len, cudaStream_t stream)
{
  rmm::device_scalar<math_t> d_sum(stream);
  if (resource::get_dry_run_flag(handle)) { return; }
  auto d_src      = src;
  auto d_dest     = dest;
  auto* d_sum_ptr = d_sum.data();
  raft::linalg::mapThenSumReduce(d_sum_ptr, len, raft::identity_op{}, stream, src);
  raft::linalg::unaryOp(
    d_dest, d_src, len, [=] __device__(math_t a) { return a / (*d_sum_ptr); }, stream);
}

template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryMult(
  Type* data, const Type* vec, IdxType n_row, IdxType n_col, cudaStream_t stream)
{
  raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
    false, data, data, vec, n_col, n_row, raft::mul_op(), stream);
}

template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryMultSkipZero(
  Type* data, const Type* vec, IdxType n_row, IdxType n_col, cudaStream_t stream)
{
  raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
    false,
    data,
    data,
    vec,
    n_col,
    n_row,
    [] __device__(Type a, Type b) {
      if (b == Type(0))
        return a;
      else
        return a * b;
    },
    stream);
}

template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryDiv(
  Type* data, const Type* vec, IdxType n_row, IdxType n_col, cudaStream_t stream)
{
  raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
    false, data, data, vec, n_col, n_row, raft::div_op(), stream);
}

template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryDivSkipZero(Type* data,
                                   const Type* vec,
                                   IdxType n_row,
                                   IdxType n_col,
                                   cudaStream_t stream,
                                   bool return_zero = false)
{
  if (return_zero) {
    raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
      false,
      data,
      data,
      vec,
      n_col,
      n_row,
      [] __device__(Type a, Type b) {
        if (raft::abs(b) < Type(1e-10))
          return Type(0);
        else
          return a / b;
      },
      stream);
  } else {
    raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
      false,
      data,
      data,
      vec,
      n_col,
      n_row,
      [] __device__(Type a, Type b) {
        if (raft::abs(b) < Type(1e-10))
          return a;
        else
          return a / b;
      },
      stream);
  }
}

template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryAdd(
  Type* data, const Type* vec, IdxType n_row, IdxType n_col, cudaStream_t stream)
{
  raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
    false, data, data, vec, n_col, n_row, raft::add_op(), stream);
}

template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinarySub(
  Type* data, const Type* vec, IdxType n_row, IdxType n_col, cudaStream_t stream)
{
  raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
    false, data, data, vec, n_col, n_row, raft::sub_op(), stream);
}

// Computes an argmin/argmax column-wise in a DxN matrix
template <typename RedOp, int TPB, typename T, typename OutT, typename IdxT>
RAFT_KERNEL argReduceKernel(const T* d_in, IdxT D, IdxT N, OutT* out)
{
  typedef cub::
    BlockReduce<cub::KeyValuePair<IdxT, T>, TPB, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>
      BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  using KVP     = cub::KeyValuePair<IdxT, T>;
  IdxT rowStart = static_cast<IdxT>(blockIdx.x) * D;
  KVP thread_data(0, std::is_same_v<RedOp, cub::ArgMax> ? -raft::myInf<T>() : raft::myInf<T>());

  for (IdxT i = threadIdx.x; i < D; i += TPB) {
    IdxT idx    = rowStart + i;
    thread_data = RedOp()(thread_data, KVP(i, d_in[idx]));
  }

  auto maxKV = BlockReduce(temp_storage).Reduce(thread_data, RedOp());

  if (threadIdx.x == 0) { out[blockIdx.x] = maxKV.key; }
}

/**
 * @brief Computes an argmin/argmax coalesced reduction
 *
 * @tparam RedOp Reduction operation (cub::ArgMin or cub::ArgMax)
 * @tparam math_t Value type
 * @tparam out_t Output key type
 * @tparam idx_t Matrix index type
 * @param[in]  in     Input matrix (DxN column-major or NxD row-major)
 * @param[in]  D      Dimension of the axis to reduce along
 * @param[in]  N      Number of reductions
 * @param[out] out    Output keys (N)
 * @param[in]  stream CUDA stream
 */
template <typename RedOp, typename math_t, typename out_t, typename idx_t>
inline void argReduce(const math_t* in, idx_t D, idx_t N, out_t* out, cudaStream_t stream)
{
  if (D <= 32) {
    argReduceKernel<RedOp, 32><<<N, 32, 0, stream>>>(in, D, N, out);
  } else if (D <= 64) {
    argReduceKernel<RedOp, 64><<<N, 64, 0, stream>>>(in, D, N, out);
  } else if (D <= 128) {
    argReduceKernel<RedOp, 128><<<N, 128, 0, stream>>>(in, D, N, out);
  } else {
    argReduceKernel<RedOp, 256><<<N, 256, 0, stream>>>(in, D, N, out);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename math_t, typename out_t, typename idx_t>
void argmin(const math_t* in, idx_t D, idx_t N, out_t* out, cudaStream_t stream)
{
  argReduce<cub::ArgMin>(in, D, N, out, stream);
}

template <typename math_t, typename out_t, typename idx_t>
void argmax(const math_t* in, idx_t D, idx_t N, out_t* out, cudaStream_t stream)
{
  argReduce<cub::ArgMax>(in, D, N, out, stream);
}

// Utility kernel needed for signFlip.
// Computes the argmax(abs(d_in)) column-wise in a DxN matrix followed by
// flipping the sign if the |max| value for each column is negative.
template <typename T, int TPB>
RAFT_KERNEL signFlipKernel(T* d_in, int D, int N)
{
  typedef cub::BlockReduce<cub::KeyValuePair<int, T>, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // compute maxIndex=argMax (with abs()) index for column
  using KVP    = cub::KeyValuePair<int, T>;
  int rowStart = blockIdx.x * D;
  KVP thread_data(0, 0);
  for (int i = threadIdx.x; i < D; i += TPB) {
    int idx     = rowStart + i;
    thread_data = cub::ArgMax()(thread_data, KVP(idx, abs(d_in[idx])));
  }
  auto maxKV = BlockReduce(temp_storage).Reduce(thread_data, cub::ArgMax());

  // flip column sign if d_in[maxIndex] < 0
  __shared__ bool need_sign_flip;
  if (threadIdx.x == 0) { need_sign_flip = d_in[maxKV.key] < T(0); }
  __syncthreads();

  if (need_sign_flip) {
    for (int i = threadIdx.x; i < D; i += TPB) {
      int idx   = rowStart + i;
      d_in[idx] = -d_in[idx];
    }
  }
}

template <typename math_t>
void signFlip(math_t* inout, int n_rows, int n_cols, cudaStream_t stream)
{
  int D     = n_rows;
  int N     = n_cols;
  auto data = inout;
  if (D <= 32) {
    signFlipKernel<math_t, 32><<<N, 32, 0, stream>>>(data, D, N);
  } else if (D <= 64) {
    signFlipKernel<math_t, 64><<<N, 64, 0, stream>>>(data, D, N);
  } else if (D <= 128) {
    signFlipKernel<math_t, 128><<<N, 128, 0, stream>>>(data, D, N);
  } else {
    signFlipKernel<math_t, 256><<<N, 256, 0, stream>>>(data, D, N);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // end namespace detail
}  // end namespace matrix
}  // namespace raft
