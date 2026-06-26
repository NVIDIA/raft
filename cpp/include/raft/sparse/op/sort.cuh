/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __SPARSE_SORT_H
#define __SPARSE_SORT_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/op/detail/sort.h>

namespace raft {
namespace sparse {
namespace op {

/**
 * @brief Sorts the arrays that comprise the coo matrix
 * by row and then by column.
 *
 * @param m number of rows in coo matrix
 * @param n number of cols in coo matrix
 * @param nnz number of non-zeros
 * @param rows rows array from coo matrix
 * @param cols cols array from coo matrix
 * @param vals vals array from coo matrix
 * @param stream: cuda stream to use
 */
template <typename T, typename IdxT = int, typename nnz_t>
void coo_sort(IdxT m, IdxT n, nnz_t nnz, IdxT* rows, IdxT* cols, T* vals, cudaStream_t stream)
{
  detail::coo_sort(false, m, n, nnz, rows, cols, vals, stream);
}

/**
 * @brief Sorts the arrays that comprise the coo matrix
 * by row and then by column (dry-run aware).
 *
 * @param handle raft resources handle
 * @param m number of rows in coo matrix
 * @param n number of cols in coo matrix
 * @param nnz number of non-zeros
 * @param rows rows array from coo matrix
 * @param cols cols array from coo matrix
 * @param vals vals array from coo matrix
 */
template <typename T, typename IdxT = int, typename nnz_t>
void coo_sort(
  raft::resources const& handle, IdxT m, IdxT n, nnz_t nnz, IdxT* rows, IdxT* cols, T* vals)
{
  detail::coo_sort(resource::get_dry_run_flag(handle),
                   m,
                   n,
                   nnz,
                   rows,
                   cols,
                   vals,
                   resource::get_cuda_stream(handle));
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
  detail::coo_sort(in, stream);
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
  detail::coo_sort_by_weight(rows, cols, data, nnz, stream);
}
};  // namespace op
};  // end NAMESPACE sparse
};  // namespace raft

#endif
