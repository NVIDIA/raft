/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/matrix/init.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/op/reduce.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cusparse_v2.h>
#include <stdio.h>

#include <algorithm>
#include <iostream>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

// TODO: value_idx param needs to be used for this once FAISS is updated to use float32
// for indices so that the index types can be uniform
template <int TPB_X = 128, typename T, typename Lambda, typename IdxT, typename nnz_t>
RAFT_KERNEL coo_symmetrize_kernel(const nnz_t* row_ind,
                                  const IdxT* rows,
                                  const IdxT* cols,
                                  const T* vals,
                                  IdxT* orows,
                                  IdxT* ocols,
                                  T* ovals,
                                  IdxT n,
                                  nnz_t cnnz,
                                  Lambda reduction_op)
{
  IdxT row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < n) {
    nnz_t start_idx = row_ind[row];  // each thread processes one row
    nnz_t stop_idx  = get_stop_idx(row, n, cnnz, row_ind);

    nnz_t row_nnz       = 0;
    nnz_t out_start_idx = start_idx * 2;

    for (nnz_t idx = 0; idx < stop_idx - start_idx; idx++) {
      IdxT cur_row = rows[start_idx + idx];
      IdxT cur_col = cols[start_idx + idx];
      T cur_val    = vals[start_idx + idx];

      IdxT lookup_row = cur_col;
      nnz_t t_start   = row_ind[lookup_row];  // Start at
      nnz_t t_stop    = get_stop_idx(lookup_row, n, cnnz, row_ind);

      T transpose = 0.0;

      bool found_match = false;
      for (nnz_t t_idx = t_start; t_idx < t_stop; t_idx++) {
        // If we find a match, let's get out of the loop. We won't
        // need to modify the transposed value, since that will be
        // done in a different thread.
        if (cols[t_idx] == cur_row && rows[t_idx] == cur_col) {
          // If it exists already, set transposed value to existing value
          transpose   = vals[t_idx];
          found_match = true;
          break;
        }
      }

      // Custom reduction op on value and its transpose, which enables
      // specialized weighting.
      // If only simple X+X.T is desired, this op can just sum
      // the two values.
      T res = reduction_op(cur_row, cur_col, cur_val, transpose);

      // if we didn't find an exact match, we need to add
      // the computed res into our current matrix to guarantee
      // symmetry.
      // Note that if we did find a match, we don't need to
      // compute `res` on it here because it will be computed
      // in a different thread.
      if (!found_match && cur_val != 0.0) {
        orows[out_start_idx + row_nnz] = cur_col;
        ocols[out_start_idx + row_nnz] = cur_row;
        ovals[out_start_idx + row_nnz] = res;
        ++row_nnz;
      }

      if (res != 0.0) {
        orows[out_start_idx + row_nnz] = cur_row;
        ocols[out_start_idx + row_nnz] = cur_col;
        ovals[out_start_idx + row_nnz] = res;
        ++row_nnz;
      }
    }
  }
}

/**
 * @brief takes a COO matrix which may not be symmetric and symmetrizes
 * it, running a custom reduction function against the each value
 * and its transposed value.
 *
 * @param in: Input COO matrix
 * @param out: Output symmetrized COO matrix
 * @param reduction_op: a custom reduction function
 * @param stream: cuda stream to use
 */
template <int TPB_X = 128, typename T, typename IdxT, typename nnz_t, typename Lambda>
void coo_symmetrize(COO<T, IdxT, nnz_t>* in,
                    COO<T, IdxT, nnz_t>* out,
                    Lambda reduction_op,  // two-argument reducer
                    cudaStream_t stream)
{
  dim3 grid(raft::ceildiv(in->n_rows, TPB_X), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  ASSERT(!out->validate_mem(), "Expecting unallocated COO for output");

  rmm::device_uvector<nnz_t> in_row_ind(in->n_rows, stream);

  convert::sorted_coo_to_csr(in, in_row_ind.data(), stream);

  out->allocate(in->nnz * 2, in->n_rows, in->n_cols, true, stream);

  coo_symmetrize_kernel<TPB_X, T><<<grid, blk, 0, stream>>>(in_row_ind.data(),
                                                            in->rows(),
                                                            in->cols(),
                                                            in->vals(),
                                                            out->rows(),
                                                            out->cols(),
                                                            out->vals(),
                                                            in->n_rows,
                                                            in->nnz,
                                                            reduction_op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief takes a COO matrix which may not be symmetric and symmetrizes
 * it, running a custom reduction function against the each value
 * and its transposed value.
 *
 * @param handle: raft resources handle
 * @param in: Input COO matrix
 * @param out: Output symmetrized COO matrix
 * @param reduction_op: a custom reduction function
 */
template <int TPB_X = 128, typename T, typename IdxT, typename nnz_t, typename Lambda>
void coo_symmetrize(raft::resources const& handle,
                    raft::device_coo_matrix_view<const T, IdxT, IdxT, nnz_t> in,
                    raft::device_coo_matrix<T, IdxT, IdxT, nnz_t>& out,
                    Lambda reduction_op)  // two-argument reducer
{
  auto stream = raft::resource::get_cuda_stream(handle);

  auto in_structure = in.structure_view();

  auto in_n_rows = in_structure.get_n_rows();
  auto in_n_cols = in_structure.get_n_cols();
  auto in_nnz    = in_structure.get_nnz();

  auto in_rows = in_structure.get_rows().data();
  auto in_cols = in_structure.get_cols().data();
  auto in_vals = in.get_elements().data();

  dim3 grid(raft::ceildiv(in_n_rows, static_cast<IdxT>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  rmm::device_uvector<nnz_t> in_row_ind(in_n_rows, stream);

  convert::sorted_coo_to_csr(in_rows, in_nnz, in_row_ind.data(), in_n_rows, stream);

  out.initialize_sparsity(in_nnz * 2);

  auto out_structure = out.structure_view();

  auto out_n_rows = out_structure.get_n_rows();
  auto out_n_cols = out_structure.get_n_cols();
  auto out_nnz    = out_structure.get_nnz();

  auto out_rows = out_structure.get_rows().data();
  auto out_cols = out_structure.get_cols().data();
  auto out_vals = out.get_elements().data();

  raft::matrix::fill(
    handle, raft::make_device_vector_view(out_rows, out_nnz), static_cast<IdxT>(0));
  raft::matrix::fill(
    handle, raft::make_device_vector_view(out_cols, out_nnz), static_cast<IdxT>(0));
  raft::matrix::fill(handle, raft::make_device_vector_view(out_vals, out_nnz), static_cast<T>(0.0));

  coo_symmetrize_kernel<TPB_X, T, Lambda, IdxT, nnz_t><<<grid, blk, 0, stream>>>(in_row_ind.data(),
                                                                                 in_rows,
                                                                                 in_cols,
                                                                                 in_vals,
                                                                                 out_rows,
                                                                                 out_cols,
                                                                                 out_vals,
                                                                                 in_n_rows,
                                                                                 in_nnz,
                                                                                 reduction_op);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Find how much space needed in each row.
 * We look through all datapoints and increment the count for each row.
 *
 * @param data: Input knn distances(n, k)
 * @param indices: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param row_sizes: Input empty row sum 1 array(n)
 * @param row_sizes2: Input empty row sum 2 array(n) for faster reduction
 */
template <typename value_idx = int64_t, typename value_t = float>
RAFT_KERNEL symmetric_find_size(const value_t* __restrict__ data,
                                const value_idx* __restrict__ indices,
                                const value_idx n,
                                const int k,
                                value_idx* __restrict__ row_sizes,
                                value_idx* __restrict__ row_sizes2)
{
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;  // for every row
  const auto j   = blockIdx.y * blockDim.y + threadIdx.y;  // for every item in row
  if (row >= n || j >= k) return;

  const auto col = indices[row * k + j];
  if (j % 2)
    atomicAdd(&row_sizes[col], value_idx(1));
  else
    atomicAdd(&row_sizes2[col], value_idx(1));
}

/**
 * @brief Reduce sum(row_sizes) + k
 * Reduction for symmetric_find_size kernel. Allows algo to be faster.
 *
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param row_sizes: Input row sum 1 array(n)
 * @param row_sizes2: Input row sum 2 array(n) for faster reduction
 */
template <typename value_idx>
RAFT_KERNEL reduce_find_size(const value_idx n,
                             const int k,
                             value_idx* __restrict__ row_sizes,
                             const value_idx* __restrict__ row_sizes2)
{
  const auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (i >= n) return;
  row_sizes[i] += (row_sizes2[i] + k);
}

/**
 * @brief Perform data + data.T operation.
 * Can only run once row_sizes from the CSR matrix of data + data.T has been
 * determined.
 *
 * @param edges: Input row sum array(n) after reduction
 * @param data: Input knn distances(n, k)
 * @param indices: Input knn indices(n, k)
 * @param VAL: Output values for data + data.T
 * @param COL: Output column indices for data + data.T
 * @param ROW: Output row indices for data + data.T
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 */
template <typename value_idx = int64_t, typename value_t = float>
RAFT_KERNEL symmetric_sum(value_idx* __restrict__ edges,
                          const value_t* __restrict__ data,
                          const value_idx* __restrict__ indices,
                          value_t* __restrict__ VAL,
                          value_idx* __restrict__ COL,
                          value_idx* __restrict__ ROW,
                          const value_idx n,
                          const int k)
{
  const auto row = blockIdx.x * blockDim.x + threadIdx.x;  // for every row
  const auto j   = blockIdx.y * blockDim.y + threadIdx.y;  // for every item in row
  if (row >= n || j >= k) return;

  const auto col       = indices[row * k + j];
  const auto original  = atomicAdd(&edges[row], value_idx(1));
  const auto transpose = atomicAdd(&edges[col], value_idx(1));

  VAL[transpose] = VAL[original] = data[row * k + j];
  // Notice swapped ROW, COL since transpose
  ROW[original] = row;
  COL[original] = col;

  ROW[transpose] = col;
  COL[transpose] = row;
}

/**
 * @brief Perform data + data.T on raw KNN data.
 * The following steps are invoked:
 * (1) Find how much space needed in each row
 * (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
 * (3) Allocate new space
 * (4) Prepare edges for each new row
 * (5) Perform final data + data.T operation
 * (6) Return summed up VAL, COL, ROW
 *
 * @param knn_indices: Input knn distances(n, k)
 * @param knn_dists: Input knn indices(n, k)
 * @param n: Number of rows
 * @param k: Number of n_neighbors
 * @param out: Output COO Matrix class
 * @param stream: Input cuda stream
 */
template <typename value_idx = int64_t, typename value_t = float, int TPB_X = 32, int TPB_Y = 32>
void from_knn_symmetrize_matrix(const value_idx* __restrict__ knn_indices,
                                const value_t* __restrict__ knn_dists,
                                const value_idx n,
                                const int k,
                                COO<value_t, value_idx>* out,
                                cudaStream_t stream)
{
  // (1) Find how much space needed in each row
  // We look through all datapoints and increment the count for each row.
  const dim3 threadsPerBlock(TPB_X, TPB_Y);
  const dim3 numBlocks(raft::ceildiv(n, (value_idx)TPB_X), raft::ceildiv(k, TPB_Y));

  // Notice n+1 since we can reuse these arrays for transpose_edges, original_edges in step (4)
  rmm::device_uvector<value_idx> row_sizes(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(row_sizes.data(), 0, sizeof(value_idx) * n, stream));

  rmm::device_uvector<value_idx> row_sizes2(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(row_sizes2.data(), 0, sizeof(value_idx) * n, stream));

  symmetric_find_size<<<numBlocks, threadsPerBlock, 0, stream>>>(
    knn_dists, knn_indices, n, k, row_sizes.data(), row_sizes2.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  reduce_find_size<<<raft::ceildiv(n, (value_idx)1024), 1024, 0, stream>>>(
    n, k, row_sizes.data(), row_sizes2.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // (2) Compute final space needed (n*k + sum(row_sizes)) == 2*n*k
  // Notice we don't do any merging and leave the result as 2*NNZ
  const auto NNZ = 2 * n * k;

  // (3) Allocate new space
  out->allocate(NNZ, n, n, true, stream);

  // (4) Prepare edges for each new row
  // This mirrors CSR matrix's row Pointer, were maximum bounds for each row
  // are calculated as the cumulative rolling sum of the previous rows.
  // Notice reusing old row_sizes2 memory
  value_idx* edges                          = row_sizes2.data();
  thrust::device_ptr<value_idx> __edges     = thrust::device_pointer_cast(edges);
  thrust::device_ptr<value_idx> __row_sizes = thrust::device_pointer_cast(row_sizes.data());

  // Rolling cumulative sum
  thrust::exclusive_scan(rmm::exec_policy(stream), __row_sizes, __row_sizes + n, __edges);

  // (5) Perform final data + data.T operation in tandem with memcpying
  symmetric_sum<<<numBlocks, threadsPerBlock, 0, stream>>>(
    edges, knn_dists, knn_indices, out->vals(), out->cols(), out->rows(), n, k);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Symmetrizes a COO matrix
 */
template <typename value_idx, typename value_t, typename nnz_t>
void symmetrize(raft::resources const& handle,
                const value_idx* rows,
                const value_idx* cols,
                const value_t* vals,
                value_idx m,
                value_idx n,
                nnz_t nnz,
                raft::sparse::COO<value_t, value_idx, nnz_t>& out)
{
  auto stream = resource::get_cuda_stream(handle);

  // copy rows to cols and cols to rows
  rmm::device_uvector<value_idx> symm_rows(nnz * 2, stream);
  rmm::device_uvector<value_idx> symm_cols(nnz * 2, stream);
  rmm::device_uvector<value_t> symm_vals(nnz * 2, stream);

  if (!resource::get_dry_run_flag(handle)) {
    raft::copy_async(symm_rows.data(), rows, nnz, stream);
    raft::copy_async(symm_rows.data() + nnz, cols, nnz, stream);
    raft::copy_async(symm_cols.data(), cols, nnz, stream);
    raft::copy_async(symm_cols.data() + nnz, rows, nnz, stream);

    raft::copy_async(symm_vals.data(), vals, nnz, stream);
    raft::copy_async(symm_vals.data() + nnz, vals, nnz, stream);
  }

  raft::sparse::op::coo_sort(handle,
                             (value_idx)m,
                             (value_idx)n,
                             static_cast<nnz_t>(nnz) * 2,
                             symm_rows.data(),
                             symm_cols.data(),
                             symm_vals.data());

  raft::sparse::op::max_duplicates(
    handle, out, symm_rows.data(), symm_cols.data(), symm_vals.data(), nnz * 2, m, n);
}

};  // end NAMESPACE detail
};  // end NAMESPACE linalg
};  // end NAMESPACE sparse
};  // namespace raft
