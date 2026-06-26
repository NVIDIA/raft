/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __SPMM_H
#define __SPMM_H

#pragma once

#include <raft/core/copy.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/sparse/linalg/detail/cusparse_utils.hpp>
#include <raft/sparse/linalg/detail/spmm.hpp>

namespace raft {
namespace sparse {
namespace linalg {

/**
 * @brief SPMM function designed for handling all CSR * DENSE
 * combinations of operand layouts for cuSparse.
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * where X is a CSR device matrix view and Y,Z are device matrix views
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of Y and Z
 * @tparam NZType Type of X
 * @tparam LayoutPolicyY layout of Y
 * @tparam LayoutPolicyZ layout of Z
 * @param[in] handle raft handle
 * @param[in] trans_x transpose operation for X
 * @param[in] trans_y transpose operation for Y
 * @param[in] alpha scalar
 * @param[in] x input raft::device_csr_matrix_view
 * @param[in] y input raft::device_matrix_view
 * @param[in] beta scalar
 * @param[inout] z input-output raft::device_matrix_view
 */
template <typename ValueType,
          typename IndexType,
          typename NZType,
          typename LayoutPolicyY,
          typename LayoutPolicyZ>
void spmm(raft::resources const& handle,
          const bool trans_x,
          const bool trans_y,
          const ValueType* alpha,
          raft::device_csr_matrix_view<const ValueType, int, int, NZType> x,
          raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyY> y,
          const ValueType* beta,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyZ> z)
{
  bool is_row_major = detail::is_row_major(y, z);

  // WARNING: The following copy is working around a bug in cusparse which causes an alignment issue
  // and incorrect results. This bug is fixed in CUDA 12.5+ so this workaround shouldn't be removed
  // until that version is supported.
  auto size  = is_row_major ? (z.extent(0) - 1) * z.stride(0) + z.extent(1)
                            : (z.extent(1) - 1) * z.stride(1) + z.extent(0);
  auto z_tmp = raft::make_device_mdarray<ValueType, IndexType>(
    handle,
    raft::resource::get_workspace_resource_ref(handle),
    raft::make_extents<IndexType>(size));

  raft::copy(handle,
             z_tmp.view(),
             raft::make_device_vector_view<const ValueType, IndexType>(z.data_handle(), size));

  auto z_tmp_view =
    is_row_major ? raft::make_device_strided_matrix_view<ValueType, IndexType, layout_c_contiguous>(
                     z_tmp.data_handle(), z.extent(0), z.extent(1), z.stride(0))
                 : raft::make_device_strided_matrix_view<ValueType, IndexType, layout_f_contiguous>(
                     z_tmp.data_handle(), z.extent(0), z.extent(1), z.stride(1));

  auto descr_x = detail::create_descriptor(x);
  auto descr_y = detail::create_descriptor(y);
  auto descr_z = detail::create_descriptor(z_tmp_view);

  detail::spmm(handle, trans_x, trans_y, is_row_major, alpha, descr_x, descr_y, beta, descr_z);

  // WARNING: Do not remove the following copy unless you can, with certainty, say that
  // the underlying cuSPARSE issue affecting CUDA 12.2+ has been resolved.
  raft::copy(handle,
             raft::make_device_vector_view<ValueType, IndexType>(z.data_handle(), size),
             raft::make_const_mdspan(z_tmp.view()));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(descr_x));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr_y));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(descr_z));
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft

#endif
