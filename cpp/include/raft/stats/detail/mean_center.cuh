/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/vectorized.cuh>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Center the input matrix wrt its mean
 * @tparam rowMajor whether input is row or col major
 * @tparam bcastAlongRows whether to broadcast vector along rows or columns
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param dry_run whether to run in dry-run mode (skip CUDA work)
 * @param out the output mean-centered matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void meanCenter(bool dry_run,
                Type* out,
                const Type* data,
                const Type* mu,
                IdxType D,
                IdxType N,
                cudaStream_t stream)
{
  raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
    dry_run, out, data, mu, D, N, raft::sub_op{}, stream);
}

/**
 * @brief Add the input matrix wrt its mean
 * @tparam rowMajor whether input is row or col major
 * @tparam bcastAlongRows whether to broadcast vector along rows or columns
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param dry_run whether to run in dry-run mode (skip CUDA work)
 * @param out the output mean-added matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void meanAdd(bool dry_run,
             Type* out,
             const Type* data,
             const Type* mu,
             IdxType D,
             IdxType N,
             cudaStream_t stream)
{
  raft::linalg::detail::matrixVectorOp<rowMajor, bcastAlongRows>(
    dry_run, out, data, mu, D, N, raft::add_op{}, stream);
}

};  // end namespace detail
};  // end namespace stats
};  // namespace raft
