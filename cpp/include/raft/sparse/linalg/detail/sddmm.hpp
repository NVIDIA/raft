/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
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
#include <raft/linalg/linalg_types.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>

namespace raft {
namespace sparse {
namespace linalg {
namespace detail {

/**
 * @brief This function performs the multiplication of dense matrix A and dense matrix B,
 * followed by an element-wise multiplication with the sparsity pattern of C.
 * It computes the following equation: C = alpha · (op_a(A) * op_b(B) ∘ spy(C)) + beta · C
 * where A,B are device matrix views and C is a CSR device matrix view
 *
 * @tparam OutputType Data type of input/output matrices (float/double)
 *
 * @param[in] handle raft resource handle
 * @param[in] descr_a input dense descriptor
 * @param[in] descr_b input dense descriptor
 * @param[in/out] descr_c output sparse descriptor
 * @param[in] op_a input Operation op(A)
 * @param[in] op_b input Operation op(B)
 * @param[in] alpha scalar pointer
 * @param[in] beta scalar pointer
 */
template <typename OutputType>
void sddmm(raft::resources const& handle,
           cusparseDnMatDescr_t& descr_a,
           cusparseDnMatDescr_t& descr_b,
           cusparseSpMatDescr_t& descr_c,
           cusparseOperation_t op_a,
           cusparseOperation_t op_b,
           const OutputType* alpha,
           const OutputType* beta)
{
  auto alg = CUSPARSE_SDDMM_ALG_DEFAULT;
  size_t bufferSize;

  RAFT_CUSPARSE_TRY(
    raft::sparse::detail::cusparsesddmm_bufferSize(resource::get_cusparse_handle(handle),
                                                   op_a,
                                                   op_b,
                                                   alpha,
                                                   descr_a,
                                                   descr_b,
                                                   beta,
                                                   descr_c,
                                                   alg,
                                                   &bufferSize,
                                                   resource::get_cuda_stream(handle)));

  resource::sync_stream(handle);

  rmm::device_uvector<uint8_t> tmp(bufferSize, resource::get_cuda_stream(handle));

  if (resource::get_dry_run_flag(handle)) { return; }

  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsesddmm(resource::get_cusparse_handle(handle),
                                                        op_a,
                                                        op_b,
                                                        alpha,
                                                        descr_a,
                                                        descr_b,
                                                        beta,
                                                        descr_c,
                                                        alg,
                                                        reinterpret_cast<void*>(tmp.data()),
                                                        resource::get_cuda_stream(handle)));
}

}  // end namespace detail
}  // end namespace linalg
}  // end namespace sparse
}  // end namespace raft
