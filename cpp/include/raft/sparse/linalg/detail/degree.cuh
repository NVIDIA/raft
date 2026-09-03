/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>
#include <raft/util/kernel_launch.hpp>

#include <cuda_runtime.h>

#include <stdio.h>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

/**
 * @brief Count all the rows in the coo row array and place them in the
 * results matrix, indexed by row.
 *
 * @tparam TPB_X: number of threads to use per block
 * @param rows the rows array of the coo matrix
 * @param nnz the size of the rows array
 * @param results array to place results
 */
template <int TPB_X = 64, typename T = int, typename outT, typename nnz_t>
RAFT_KERNEL coo_degree_kernel(const T* rows, nnz_t nnz, outT* results)
{
  nnz_t row = (blockIdx.x * static_cast<nnz_t>(TPB_X)) + threadIdx.x;
  if (row < nnz) { atomicAdd(results + rows[row], (outT)1); }
}

/**
 * @brief Count the number of values for each row
 * @tparam TPB_X: number of threads to use per block
 * @param rows: rows array of the COO matrix
 * @param nnz: size of the rows array
 * @param results: output result array
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T = int, typename outT, typename nnz_t>
void coo_degree(const T* rows, nnz_t nnz, outT* results, cudaStream_t stream)
{
  dim3 grid_rc(raft::ceildiv((nnz_t)nnz, (nnz_t)TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);

  raft::launch_kernel(stream, grid_rc, blk_rc, coo_degree_kernel<TPB_X>, rows, nnz, results);
}

template <int TPB_X = 64, typename T, typename idx_t, typename nnz_t>
RAFT_KERNEL coo_degree_nz_kernel(const idx_t* rows, const T* vals, nnz_t nnz, idx_t* results)
{
  int row = (blockIdx.x * TPB_X) + threadIdx.x;
  if (row < nnz && vals[row] != 0.0) { raft::myAtomicAdd(results + rows[row], 1); }
}

template <int TPB_X = 64, typename T, typename idx_t, typename outT, typename nnz_t>
RAFT_KERNEL coo_degree_scalar_kernel(
  const idx_t* rows, const T* vals, nnz_t nnz, T scalar, outT* results)
{
  nnz_t row = (blockIdx.x * static_cast<nnz_t>(TPB_X)) + threadIdx.x;
  if (row < nnz && vals[row] != scalar) { raft::myAtomicAdd((outT*)results + rows[row], (outT)1); }
}

/**
 * @brief Count the number of values for each row that doesn't match a particular scalar
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param scalar: scalar to match for counting rows
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T, typename idx_t, typename outT, typename nnz_t>
void coo_degree_scalar(
  const idx_t* rows, const T* vals, nnz_t nnz, T scalar, outT* results, cudaStream_t stream = 0)
{
  dim3 grid_rc(raft::ceildiv(nnz, static_cast<nnz_t>(TPB_X)), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);
  raft::launch_kernel(stream,
                      grid_rc,
                      blk_rc,
                      coo_degree_scalar_kernel<TPB_X, T, idx_t, outT, nnz_t>,
                      rows,
                      vals,
                      nnz,
                      scalar,
                      results);
}

/**
 * @brief Count the number of nonzeros for each row
 * @tparam TPB_X: number of threads to use per block
 * @tparam T: the type name of the underlying value arrays
 * @param rows: Input COO row array
 * @param vals: Input COO val arrays
 * @param nnz: size of input COO arrays
 * @param results: output row counts
 * @param stream: cuda stream to use
 */
template <int TPB_X = 64, typename T, typename idx_t, typename nnz_t>
void coo_degree_nz(const idx_t* rows, const T* vals, nnz_t nnz, idx_t* results, cudaStream_t stream)
{
  dim3 grid_rc(raft::ceildiv(nnz, TPB_X), 1, 1);
  dim3 blk_rc(TPB_X, 1, 1);
  raft::launch_kernel(stream,
                      grid_rc,
                      blk_rc,
                      coo_degree_nz_kernel<TPB_X, T, idx_t, nnz_t>,
                      rows,
                      vals,
                      nnz,
                      results);
}

};  // end NAMESPACE detail
};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // namespace raft
