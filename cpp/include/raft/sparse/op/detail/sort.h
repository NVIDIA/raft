/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/tuple>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <cusparse_v2.h>

#include <algorithm>

namespace raft {
namespace sparse {
namespace op {
namespace detail {

struct TupleComp {
  template <typename one, typename two>
  __host__ __device__

    bool
    operator()(const one& t1, const two& t2)
  {
    // sort first by each sample's color,
    if (cuda::std::get<0>(t1) < cuda::std::get<0>(t2)) return true;
    if (cuda::std::get<0>(t1) > cuda::std::get<0>(t2)) return false;

    // then sort by value in descending order
    return cuda::std::get<1>(t1) < cuda::std::get<1>(t2);
  }
};

/**
 * @brief Sorts the arrays that comprise the coo matrix
 * by row and then by column.
 *
 * @param dry_run when true, allocates a best-effort workspace estimate
 *                without launching kernels (for memory tracking)
 * @param m number of rows in coo matrix
 * @param n number of cols in coo matrix
 * @param nnz number of non-zeros
 * @param rows rows array from coo matrix
 * @param cols cols array from coo matrix
 * @param vals vals array from coo matrix
 * @param stream: cuda stream to use
 */
template <typename T, typename IdxT = int, typename nnz_t>
void coo_sort(
  bool dry_run, IdxT m, IdxT n, nnz_t nnz, IdxT* rows, IdxT* cols, T* vals, cudaStream_t stream)
{
  if (dry_run) {
    // Best-effort upper bound for thrust::sort_by_key workspace.
    // Double-buffer estimate for large inputs; minimum 4096 for small inputs
    // where per-allocation alignment overhead dominates.
    auto sort_data_bytes = static_cast<std::size_t>(nnz) * (sizeof(IdxT) * 2 + sizeof(T));
    rmm::device_uvector<char> sort_ws_est(std::max(sort_data_bytes * 2, std::size_t{4096}), stream);
    return;
  }

  auto coo_indices = thrust::make_zip_iterator(cuda::std::make_tuple(rows, cols));

  // get all the colors in contiguous locations so we can map them to warps.
  thrust::sort_by_key(rmm::exec_policy(stream), coo_indices, coo_indices + nnz, vals, TupleComp());
}

/**
 * @brief Sort the underlying COO arrays by row
 * @tparam T: the type name of the underlying value array
 * @param in: COO to sort by row
 * @param stream: the cuda stream to use
 */
template <typename T, typename IdxT = int, typename nnz_t>
void coo_sort(COO<T, IdxT, nnz_t>* const in, cudaStream_t stream)
{
  coo_sort<T, IdxT, nnz_t>(
    false, in->n_rows, in->n_cols, in->nnz, in->rows(), in->cols(), in->vals(), stream);
}

/**
 * Sorts a COO by its weight
 * @tparam value_idx
 * @tparam value_t
 * @param[inout] rows source edges
 * @param[inout] cols dest edges
 * @param[inout] data edge weights
 * @param[in] nnz number of edges in edge list
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename value_idx, typename value_t, typename nnz_t>
void coo_sort_by_weight(
  value_idx* rows, value_idx* cols, value_t* data, nnz_t nnz, cudaStream_t stream)
{
  thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(data);

  auto first = thrust::make_zip_iterator(cuda::std::make_tuple(rows, cols));

  thrust::sort_by_key(rmm::exec_policy(stream), t_data, t_data + nnz, first);
}
};  // namespace detail
};  // namespace op
};  // end NAMESPACE sparse
};  // namespace raft
