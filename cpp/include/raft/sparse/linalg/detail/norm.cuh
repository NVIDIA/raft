/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/op/row_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cusparse_v2.h>
#include <stdio.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

template <int TPB_X = 64, typename T, typename indT>
RAFT_KERNEL csr_row_normalize_l1_kernel(
  // @TODO: This can be done much more parallel by
  // having threads in a warp compute the sum in parallel
  // over each row and then divide the values in parallel.
  const indT* ia,  // csr row ex_scan (sorted by row)
  const T* vals,
  indT nnz,  // array of values and number of non-zeros
  int m,     // num rows in csr
  T* result)
{  // output array

  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  // sum all vals_arr for row and divide each val by sum
  if (row < m) {
    indT start_idx = ia[row];
    indT stop_idx  = 0;
    if (row < m - 1) {
      stop_idx = ia[row + 1];
    } else
      stop_idx = nnz;

    T sum = T(0.0);
    for (indT j = start_idx; j < stop_idx; j++) {
      sum = sum + fabs(vals[j]);
    }

    for (indT j = start_idx; j < stop_idx; j++) {
      if (sum != 0.0) {
        T val     = vals[j];
        result[j] = val / sum;
      } else {
        result[j] = 0.0;
      }
    }
  }
}

/**
 * @brief Perform L1 normalization on the rows of a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T, typename indT>
void csr_row_normalize_l1(const indT* ia,  // csr row ex_scan (sorted by row)
                          const T* vals,
                          indT nnz,  // array of values and number of non-zeros
                          int m,     // num rows in csr
                          T* result,
                          cudaStream_t stream)
{  // output array

  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  raft::launch_kernel(
    stream, grid, blk, csr_row_normalize_l1_kernel<TPB_X, T>, ia, vals, nnz, m, result);
}

template <int TPB_X = 64, typename T>
RAFT_KERNEL csr_row_normalize_max_kernel(
  // @TODO: This can be done much more parallel by
  // having threads in a warp compute the sum in parallel
  // over each row and then divide the values in parallel.
  const int* ia,  // csr row ind array (sorted by row)
  const T* vals,
  int nnz,  // array of values and number of non-zeros
  int m,    // num total rows in csr
  T* result)
{  // output array

  // row-based matrix 1 thread per row
  int row = (blockIdx.x * TPB_X) + threadIdx.x;

  // find max across columns and divide
  if (row < m) {
    int start_idx = ia[row];
    int stop_idx  = 0;
    if (row < m - 1) {
      stop_idx = ia[row + 1];
    } else
      stop_idx = nnz;

    T max = std::numeric_limits<float>::min();
    for (int j = start_idx; j < stop_idx; j++) {
      if (vals[j] > max) max = vals[j];
    }

    // divide nonzeros in current row by max
    for (int j = start_idx; j < stop_idx; j++) {
      if (max != 0.0 && max > std::numeric_limits<float>::min()) {
        T val     = vals[j];
        result[j] = val / max;
      } else {
        result[j] = 0.0;
      }
    }
  }
}

/**
 * @brief Perform L_inf normalization on a given CSR-formatted sparse matrix
 *
 * @param ia: row_ind array
 * @param vals: data array
 * @param nnz: size of data array
 * @param m: size of row_ind array
 * @param result: l1 normalized data array
 * @param stream: cuda stream to use
 */

template <int TPB_X = 64, typename T>
void csr_row_normalize_max(const int* ia,  // csr row ind array (sorted by row)
                           const T* vals,
                           int nnz,  // array of values and number of non-zeros
                           int m,    // num total rows in csr
                           T* result,
                           cudaStream_t stream)
{
  dim3 grid(raft::ceildiv(m, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  raft::launch_kernel(
    stream, grid, blk, csr_row_normalize_max_kernel<TPB_X, T>, ia, vals, nnz, m, result);
}

template <typename Type,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void csr_row_op_wrapper(const IdxType* ia,
                        const Type* data,
                        IdxType nnz,
                        IdxType N,
                        Type init,
                        Type* norm,
                        cudaStream_t stream,
                        MainLambda main_op     = raft::identity_op(),
                        ReduceLambda reduce_op = raft::add_op(),
                        FinalLambda final_op   = raft::identity_op())
{
  op::csr_row_op<IdxType>(
    ia,
    N,
    nnz,
    [data, init, norm, main_op, reduce_op, final_op] __device__(
      IdxType row, IdxType start_idx, IdxType stop_idx) {
      norm[row] = init;
      for (IdxType i = start_idx; i < stop_idx; i++)
        norm[row] = final_op(reduce_op(norm[row], main_op(data[i])));
    },
    stream);
}

template <typename Type, typename IdxType, typename Lambda>
void rowNormCsrCaller(const IdxType* ia,
                      const Type* data,
                      IdxType nnz,
                      IdxType N,
                      Type* norm,
                      raft::linalg::NormType type,
                      Lambda fin_op,
                      cudaStream_t stream)
{
  switch (type) {
    case raft::linalg::NormType::L1Norm:
      csr_row_op_wrapper(
        ia, data, nnz, N, (Type)0, norm, stream, raft::abs_op(), raft::add_op(), fin_op);
      break;
    case raft::linalg::NormType::L2Norm:
      csr_row_op_wrapper(
        ia, data, nnz, N, (Type)0, norm, stream, raft::sq_op(), raft::add_op(), fin_op);
      break;
    case raft::linalg::NormType::LinfNorm:
      csr_row_op_wrapper(
        ia, data, nnz, N, (Type)0, norm, stream, raft::abs_op(), raft::max_op(), fin_op);
      break;
    default: THROW("Unsupported norm type: %d", type);
  };
}

};  // end NAMESPACE detail
};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // namespace raft
