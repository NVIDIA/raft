/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam rowMajor whether the input data is row or col major
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param dry_run whether to run in dry-run mode (skip CUDA work)
 * @param std the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, typename Type, typename IdxType = int>
void stddev(bool dry_run,
            Type* std,
            const Type* data,
            const Type* mu,
            IdxType D,
            IdxType N,
            bool sample,
            cudaStream_t stream)
{
  raft::linalg::detail::reduce<rowMajor, false>(
    dry_run, std, data, D, N, Type(0), stream, false, [mu] __device__(Type a, IdxType i) {
      return a * a;
    });
  if (dry_run) { return; }
  Type ratio      = Type(1) / ((sample) ? Type(N - 1) : Type(N));
  Type ratio_mean = sample ? ratio * Type(N) : Type(1);
  raft::linalg::binaryOp(std,
                         std,
                         mu,
                         D,
                         raft::compose_op(raft::sqrt_op(),
                                          raft::abs_op(),
                                          [ratio, ratio_mean] __device__(Type a, Type b) {
                                            return a * ratio - b * b * ratio_mean;
                                          }),
                         stream);
}

/**
 * @brief Compute variance of the input matrix
 *
 * Variance operation is assumed to be performed on a given column.
 *
 * @tparam rowMajor whether the input data is row or col major
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param dry_run whether to run in dry-run mode (skip CUDA work)
 * @param var the output variance vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample variance or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, typename Type, typename IdxType = int>
void vars(bool dry_run,
          Type* var,
          const Type* data,
          const Type* mu,
          IdxType D,
          IdxType N,
          bool sample,
          cudaStream_t stream)
{
  raft::linalg::detail::reduce<rowMajor, false>(
    dry_run, var, data, D, N, Type(0), stream, false, [mu] __device__(Type a, IdxType i) {
      return a * a;
    });
  if (dry_run) { return; }
  Type ratio      = Type(1) / ((sample) ? Type(N - 1) : Type(N));
  Type ratio_mean = sample ? ratio * Type(N) : Type(1);
  raft::linalg::binaryOp(var,
                         var,
                         mu,
                         D,
                         raft::compose_op(raft::abs_op(),
                                          [ratio, ratio_mean] __device__(Type a, Type b) {
                                            return a * ratio - b * b * ratio_mean;
                                          }),
                         stream);
}

}  // namespace detail
}  // namespace stats
}  // namespace raft
