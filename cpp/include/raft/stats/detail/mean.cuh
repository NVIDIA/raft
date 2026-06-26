/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace stats {
namespace detail {

template <bool rowMajor, typename Type, typename IdxType = int>
void mean(bool dry_run, Type* mu, const Type* data, IdxType D, IdxType N, cudaStream_t stream)
{
  Type ratio = Type(1) / Type(N);
  raft::linalg::detail::reduce<rowMajor, false>(dry_run,
                                                mu,
                                                data,
                                                D,
                                                N,
                                                Type(0),
                                                stream,
                                                false,
                                                raft::identity_op(),
                                                raft::add_op(),
                                                raft::mul_const_op<Type>(ratio));
}

template <bool rowMajor, typename Type, typename IdxType = int>
[[deprecated]] void mean(
  bool dry_run, Type* mu, const Type* data, IdxType D, IdxType N, bool sample, cudaStream_t stream)
{
  Type ratio = Type(1) / ((sample) ? Type(N - 1) : Type(N));
  raft::linalg::detail::reduce<rowMajor, false>(dry_run,
                                                mu,
                                                data,
                                                D,
                                                N,
                                                Type(0),
                                                stream,
                                                false,
                                                raft::identity_op(),
                                                raft::add_op(),
                                                raft::mul_const_op<Type>(ratio));
}

}  // namespace detail
}  // namespace stats
}  // namespace raft
