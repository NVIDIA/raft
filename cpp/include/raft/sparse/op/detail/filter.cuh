/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/linalg/degree.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

#include <cusparse_v2.h>

#include <algorithm>
#include <cstdio>
#include <iostream>

namespace raft {
namespace sparse {
namespace op {
namespace detail {

template <int TPB_X, typename T, typename idx_t, typename nnz_t>
RAFT_KERNEL coo_remove_scalar_kernel(const idx_t* rows,
                                     const idx_t* cols,
                                     const T* vals,
                                     nnz_t nnz,
                                     idx_t* out_rows,
                                     idx_t* out_cols,
                                     T* out_vals,
                                     nnz_t* ex_scan,
                                     nnz_t* cur_ex_scan,
                                     idx_t m,
                                     T scalar)
{
  idx_t row = (blockIdx.x * TPB_X) + threadIdx.x;

  if (row < m) {
    nnz_t start       = cur_ex_scan[row];
    nnz_t stop        = get_stop_idx(row, m, nnz, cur_ex_scan);
    nnz_t cur_out_idx = ex_scan[row];

    for (nnz_t idx = start; idx < stop; idx++) {
      if (vals[idx] != scalar) {
        out_rows[cur_out_idx] = rows[idx];
        out_cols[cur_out_idx] = cols[idx];
        out_vals[cur_out_idx] = vals[idx];
        ++cur_out_idx;
      }
    }
  }
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param rows: input array of rows (size n)
 * @param cols: input array of cols (size n)
 * @param vals: input array of vals (size n)
 * @param nnz: size of current rows/cols/vals arrays
 * @param crows: compressed array of rows
 * @param ccols: compressed array of cols
 * @param cvals: compressed array of vals
 * @param cnnz: array of non-zero counts per row
 * @param cur_cnnz array of counts per row
 * @param scalar: scalar to remove from arrays
 * @param n: number of rows in dense matrix
 * @param d_alloc device allocator for temporary buffers
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T, typename idx_t, typename nnz_t>
void coo_remove_scalar(const idx_t* rows,
                       const idx_t* cols,
                       const T* vals,
                       nnz_t nnz,
                       idx_t* crows,
                       idx_t* ccols,
                       T* cvals,
                       nnz_t* cnnz,
                       nnz_t* cur_cnnz,
                       T scalar,
                       idx_t n,
                       cudaStream_t stream)
{
  rmm::device_uvector<nnz_t> ex_scan(n, stream);
  rmm::device_uvector<nnz_t> cur_ex_scan(n, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(ex_scan.data(), 0, static_cast<nnz_t>(n) * sizeof(nnz_t), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(cur_ex_scan.data(), 0, static_cast<nnz_t>(n) * sizeof(nnz_t), stream));

  thrust::device_ptr<nnz_t> dev_cnnz    = thrust::device_pointer_cast(cnnz);
  thrust::device_ptr<nnz_t> dev_ex_scan = thrust::device_pointer_cast(ex_scan.data());
  thrust::exclusive_scan(rmm::exec_policy(stream), dev_cnnz, dev_cnnz + n, dev_ex_scan);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  thrust::device_ptr<nnz_t> dev_cur_cnnz    = thrust::device_pointer_cast(cur_cnnz);
  thrust::device_ptr<nnz_t> dev_cur_ex_scan = thrust::device_pointer_cast(cur_ex_scan.data());
  thrust::exclusive_scan(rmm::exec_policy(stream), dev_cur_cnnz, dev_cur_cnnz + n, dev_cur_ex_scan);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  dim3 grid(raft::ceildiv(n, static_cast<idx_t>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);

  coo_remove_scalar_kernel<TPB_X, T, idx_t, nnz_t><<<grid, blk, 0, stream>>>(rows,
                                                                             cols,
                                                                             vals,
                                                                             nnz,
                                                                             crows,
                                                                             ccols,
                                                                             cvals,
                                                                             dev_ex_scan.get(),
                                                                             dev_cur_ex_scan.get(),
                                                                             n,
                                                                             scalar);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param scalar: scalar to remove from arrays
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T, typename idx_t, typename nnz_t>
void coo_remove_scalar(COO<T, idx_t, nnz_t>* in,
                       COO<T, idx_t, nnz_t>* out,
                       T scalar,
                       cudaStream_t stream)
{
  rmm::device_uvector<nnz_t> row_count_nz(in->n_rows, stream);
  rmm::device_uvector<nnz_t> row_count(in->n_rows, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(
    row_count_nz.data(), 0, static_cast<nnz_t>(in->n_rows) * sizeof(nnz_t), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(row_count.data(), 0, static_cast<nnz_t>(in->n_rows) * sizeof(nnz_t), stream));

  linalg::coo_degree(in->rows(), in->nnz, row_count.data(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  using nnz_cast_t = std::conditional_t<std::is_same_v<nnz_t, uint64_t>, unsigned long long, nnz_t>;
  linalg::coo_degree_scalar(in->rows(),
                            in->vals(),
                            in->nnz,
                            scalar,
                            reinterpret_cast<nnz_cast_t*>(row_count_nz.data()),
                            stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  thrust::device_ptr<nnz_t> d_row_count_nz = thrust::device_pointer_cast(row_count_nz.data());
  nnz_t out_nnz =
    thrust::reduce(rmm::exec_policy(stream), d_row_count_nz, d_row_count_nz + in->n_rows);

  out->allocate(out_nnz, in->n_rows, in->n_cols, false, stream);

  coo_remove_scalar<TPB_X, T, idx_t, nnz_t>(in->rows(),
                                            in->cols(),
                                            in->vals(),
                                            in->nnz,
                                            out->rows(),
                                            out->cols(),
                                            out->vals(),
                                            row_count_nz.data(),
                                            row_count.data(),
                                            scalar,
                                            in->n_rows,
                                            stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Removes the values matching a particular scalar from a COO formatted sparse matrix.
 *
 * @param handle: device resources
 * @param in: input COO matrix
 * @param scalar: scalar to remove from arrays
 * @param out: output COO matrix
 */
template <int TPB_X, typename T, typename idx_t, typename nnz_t>
void coo_remove_scalar(raft::resources const& handle,
                       raft::device_coo_matrix_view<const T, idx_t, idx_t, nnz_t> in,
                       raft::host_scalar_view<const T> scalar,
                       raft::device_coo_matrix<T, idx_t, idx_t, nnz_t>& out)
{
  auto stream = resource::get_cuda_stream(handle);

  auto in_structure = in.structure_view();

  auto in_n_rows = in_structure.get_n_rows();
  auto in_n_cols = in_structure.get_n_cols();
  auto in_nnz    = in_structure.get_nnz();

  auto in_rows = in_structure.get_rows().data();
  auto in_cols = in_structure.get_cols().data();
  auto in_vals = in.get_elements().data();

  rmm::device_uvector<nnz_t> row_count_nz(in_n_rows, stream);
  rmm::device_uvector<nnz_t> row_count(in_n_rows, stream);

  if (resource::get_dry_run_flag(handle)) {
    // Upper bound on non-dry-run compliant and data-dependent code below (thrust calls, unknown
    // out_nnz <= in_nnz)
    out.initialize_sparsity(in_nnz);
    rmm::device_uvector<nnz_t> ex_scan(in_n_rows, stream);
    rmm::device_uvector<nnz_t> cur_ex_scan(in_n_rows, stream);
    // Upper bound for thrust workspace (reduce + 2x exclusive_scan)
    rmm::device_uvector<char> thrust_ws(3 * (static_cast<size_t>(in_n_rows) * sizeof(nnz_t) + 4096),
                                        stream);
    return;
  }

  RAFT_CUDA_TRY(
    cudaMemsetAsync(row_count_nz.data(), 0, static_cast<nnz_t>(in_n_rows) * sizeof(nnz_t), stream));
  RAFT_CUDA_TRY(
    cudaMemsetAsync(row_count.data(), 0, static_cast<nnz_t>(in_n_rows) * sizeof(nnz_t), stream));

  linalg::coo_degree(in_rows, in_nnz, row_count.data(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  linalg::coo_degree_scalar(
    in_rows, in_vals, in_nnz, scalar(0), (nnz_t*)row_count_nz.data(), stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  thrust::device_ptr<nnz_t> d_row_count_nz = thrust::device_pointer_cast(row_count_nz.data());
  auto out_nnz =
    thrust::reduce(rmm::exec_policy(stream), d_row_count_nz, d_row_count_nz + in_n_rows);

  out.initialize_sparsity(out_nnz);

  auto out_structure = out.structure_view();

  auto out_n_rows = out_structure.get_n_rows();
  auto out_n_cols = out_structure.get_n_cols();
  out_nnz         = out_structure.get_nnz();

  auto out_rows = out_structure.get_rows().data();
  auto out_cols = out_structure.get_cols().data();
  auto out_vals = out.get_elements().data();

  coo_remove_scalar<TPB_X, T, idx_t, nnz_t>(in_rows,
                                            in_cols,
                                            in_vals,
                                            in_nnz,
                                            out_rows,
                                            out_cols,
                                            out_vals,
                                            row_count_nz.data(),
                                            row_count.data(),
                                            scalar(0),
                                            in_n_rows,
                                            stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Removes zeros from a COO formatted sparse matrix.
 *
 * @param in: input COO matrix
 * @param out: output COO matrix
 * @param stream: cuda stream to use
 */
template <int TPB_X, typename T, typename idx_t, typename nnz_t>
void coo_remove_zeros(COO<T, idx_t, nnz_t>* in, COO<T, idx_t, nnz_t>* out, cudaStream_t stream)
{
  coo_remove_scalar<TPB_X, T, idx_t, nnz_t>(in, out, T(0.0), stream);
}

};  // namespace detail
};  // namespace op
};  // end NAMESPACE sparse
};  // namespace raft
