/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __DENSE_H
#define __DENSE_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/convert/detail/dense.cuh>

namespace raft {
namespace sparse {
namespace convert {

/**
 * Convert a sparse matrix view to a dense matrix view.
 *
 * Supports both COO and CSR sparse matrix views and row- or column-major dense output.
 *
 * @param[in] handle RAFT resources
 * @param[in] sparse Sparse COO or CSR matrix view
 * @param[out] dense Dense matrix view
 */
template <typename SparseMatrixViewType,
          typename ValueType,
          typename IndexType,
          typename LayoutPolicy>
void sparse_to_dense(raft::resources const& handle,
                     SparseMatrixViewType sparse,
                     raft::device_matrix_view<ValueType, IndexType, LayoutPolicy> dense)
{
  auto structure = sparse.structure_view();
  RAFT_EXPECTS(dense.extent(0) == static_cast<IndexType>(structure.get_n_rows()) &&
                 dense.extent(1) == static_cast<IndexType>(structure.get_n_cols()),
               "Sparse and dense matrix dimensions must match");

  detail::sparse_to_dense(handle, sparse, dense);
}

};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // namespace raft

#endif
