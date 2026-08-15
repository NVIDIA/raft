/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cusparse_v2.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

namespace raft {
namespace sparse {
namespace op {
namespace detail {

template <typename T, int TPB_X = 256, typename Lambda = auto(T, T, T)->void>
RAFT_KERNEL csr_row_op_kernel(const T* row_ind, T n_rows, T nnz, Lambda op)
{
  T row = blockIdx.x * TPB_X + threadIdx.x;
  if (row < n_rows) {
    T start_idx = row_ind[row];
    T stop_idx  = row < n_rows - 1 ? row_ind[row + 1] : nnz;
    op(row, start_idx, stop_idx);
  }
}

/**
 * @brief Perform a custom row operation on a CSR matrix in batches.
 * @tparam T numerical type of row_ind array
 * @tparam TPB_X number of threads per block to use for underlying kernel
 * @tparam Lambda type of custom operation function
 * @param row_ind the CSR row_ind array to perform parallel operations over
 * @param n_rows total number vertices in graph
 * @param nnz number of non-zeros
 * @param op custom row operation functor accepting the row and beginning index.
 * @param stream cuda stream to use
 */
template <typename Index_, int TPB_X = 256, typename Lambda = auto(Index_, Index_, Index_)->void>
void csr_row_op(const Index_* row_ind, Index_ n_rows, Index_ nnz, Lambda op, cudaStream_t stream)
{
  dim3 grid(raft::ceildiv(n_rows, Index_(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  raft::launch_kernel(
    stream, grid, blk, csr_row_op_kernel<Index_, TPB_X, Lambda>, row_ind, n_rows, nnz, op);
}

};  // namespace detail
};  // namespace op
};  // end NAMESPACE sparse
};  // namespace raft
