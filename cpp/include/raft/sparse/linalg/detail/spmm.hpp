/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

/**
 * @brief determine common data layout for both dense matrices
 * @tparam ValueType Data type of Y,Z (float/double)
 * @tparam IndexType Type of Y,Z
 * @tparam LayoutPolicyY layout of Y
 * @tparam LayoutPolicyZ layout of Z
 * @param[in] x input raft::device_matrix_view
 * @param[in] y input raft::device_matrix_view
 * @returns dense matrix descriptor to be used by cuSparse API
 */
template <typename ValueType, typename IndexType, typename LayoutPolicyY, typename LayoutPolicyZ>
bool is_row_major(raft::device_matrix_view<const ValueType, IndexType, LayoutPolicyY>& y,
                  raft::device_matrix_view<ValueType, IndexType, LayoutPolicyZ>& z)
{
  bool is_row_major = z.stride(1) == 1 && y.stride(1) == 1;
  bool is_col_major = z.stride(0) == 1 && y.stride(0) == 1;
  ASSERT(is_row_major || is_col_major, "Both matrices need to be either row or col major");
  return is_row_major;
}

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
 * @param[in] is_row_major data layout of Y,Z
 * @param[in] alpha scalar
 * @param[in] descr_x input sparse descriptor
 * @param[in] descr_y input dense descriptor
 * @param[in] beta scalar
 * @param[out] descr_z output dense descriptor
 */
template <typename ValueType>
void spmm(raft::resources const& handle,
          const bool trans_x,
          const bool trans_y,
          const bool is_row_major,
          const ValueType* alpha,
          cusparseSpMatDescr_t& descr_x,
          cusparseDnMatDescr_t& descr_y,
          const ValueType* beta,
          cusparseDnMatDescr_t& descr_z)
{
  auto opX = trans_x ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opY = trans_y ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto alg = is_row_major ? CUSPARSE_SPMM_CSR_ALG2 : CUSPARSE_SPMM_CSR_ALG1;
  size_t bufferSize;
  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsespmm_bufferSize(resource::get_cusparse_handle(handle),
                                                  opX,
                                                  opY,
                                                  alpha,
                                                  descr_x,
                                                  descr_y,
                                                  beta,
                                                  descr_z,
                                                  alg,
                                                  &bufferSize,
                                                  resource::get_cuda_stream(handle)));

  resource::sync_stream(handle);

  rmm::device_uvector<ValueType> tmp(bufferSize, resource::get_cuda_stream(handle));

  if (resource::get_dry_run_flag(handle)) { return; }

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsespmm(resource::get_cusparse_handle(handle),
                                                       opX,
                                                       opY,
                                                       alpha,
                                                       descr_x,
                                                       descr_y,
                                                       beta,
                                                       descr_z,
                                                       alg,
                                                       tmp.data(),
                                                       resource::get_cuda_stream(handle)));
}

}  // end namespace detail
}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
