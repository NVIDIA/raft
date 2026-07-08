/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>

#include <type_traits>

namespace raft {
namespace stats {
namespace detail {

template <bool rowMajor, typename OutType, typename InType, typename IdxType = int>
void mean(OutType* mu, const InType* data, IdxType D, IdxType N, cudaStream_t stream)
{
  OutType ratio = OutType(1) / OutType(N);
  auto main_op  = [=]() {
    if constexpr (std::is_same_v<InType, OutType>) {
      return raft::identity_op();
    } else {
      return raft::cast_op<OutType>();
    }
  }();
  raft::linalg::reduce<rowMajor, false, InType, OutType>(mu,
                                                         data,
                                                         D,
                                                         N,
                                                         OutType(0),
                                                         stream,
                                                         false,
                                                         main_op,
                                                         raft::add_op(),
                                                         raft::mul_const_op<OutType>(ratio));
}

template <bool rowMajor, typename Type, typename IdxType = int>
[[deprecated]] void mean(
  Type* mu, const Type* data, IdxType D, IdxType N, bool sample, cudaStream_t stream)
{
  Type ratio = Type(1) / ((sample) ? Type(N - 1) : Type(N));
  raft::linalg::reduce<rowMajor, false>(mu,
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
