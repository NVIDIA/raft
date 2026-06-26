/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/linalg/reduce.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace stats {
namespace detail {

/**
 * @brief Compute the row-wise weighted mean of the input matrix with a
 * vector of weights
 *
 * @tparam row_major data input matrix is row-major or not
 * @tparam along_rows whether to reduce along rows or columns
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param dry_run whether to run in dry-run mode
 * @param mu the output mean vector
 * @param data the input matrix
 * @param weights weight of size D if along_row is true, else of size N
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream to launch work on
 */
template <bool row_major, bool along_rows, typename Type, typename IdxType = int>
void weightedMean(bool dry_run,
                  Type* mu,
                  const Type* data,
                  const Type* weights,
                  IdxType D,
                  IdxType N,
                  cudaStream_t stream)
{
  // sum the weights & copy back to CPU
  auto weight_size = along_rows ? D : N;
  Type WS          = 0;
  raft::linalg::detail::reduce<false, false>(
    dry_run, mu, weights, (IdxType)1, weight_size, (Type)0, stream);
  if (!dry_run) { raft::update_host(&WS, mu, 1, stream); }

  raft::linalg::detail::reduce<row_major, along_rows>(
    dry_run,
    mu,
    data,
    D,
    N,
    (Type)0,
    stream,
    false,
    [weights] __device__(Type v, IdxType i) { return v * weights[i]; },
    raft::add_op{},
    raft::div_const_op<Type>(WS));
}
};  // end namespace detail
};  // end namespace stats
};  // namespace raft
