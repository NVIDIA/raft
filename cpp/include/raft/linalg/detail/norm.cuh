/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/reduce.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <NormType norm_type,
          bool rowMajor,
          typename Type,
          typename IdxType,
          typename Lambda,
          typename OutType = Type>
void rowNormCaller(bool dry_run,
                   OutType* dots,
                   const Type* data,
                   IdxType D,
                   IdxType N,
                   cudaStream_t stream,
                   Lambda fin_op)
{
  if constexpr (norm_type == L1Norm) {
    reduce<rowMajor, true, Type, OutType, IdxType>(
      dry_run, dots, data, D, N, (OutType)0, stream, false, raft::abs_op(), raft::add_op(), fin_op);
  } else if constexpr (norm_type == L2Norm) {
    reduce<rowMajor, true, Type, OutType, IdxType>(
      dry_run, dots, data, D, N, (OutType)0, stream, false, raft::sq_op(), raft::add_op(), fin_op);
  } else if constexpr (norm_type == LinfNorm) {
    reduce<rowMajor, true, Type, OutType, IdxType>(
      dry_run, dots, data, D, N, (OutType)0, stream, false, raft::abs_op(), raft::max_op(), fin_op);
  } else {
    THROW("Unsupported norm type: %d", norm_type);
  }
}

template <NormType norm_type,
          bool rowMajor,
          typename Type,
          typename IdxType,
          typename Lambda,
          typename OutType = Type>
void colNormCaller(bool dry_run,
                   OutType* dots,
                   const Type* data,
                   IdxType D,
                   IdxType N,
                   cudaStream_t stream,
                   Lambda fin_op)
{
  if constexpr (norm_type == L1Norm) {
    reduce<rowMajor, false, Type, OutType, IdxType>(
      dry_run, dots, data, D, N, (OutType)0, stream, false, raft::abs_op(), raft::add_op(), fin_op);
  } else if constexpr (norm_type == L2Norm) {
    reduce<rowMajor, false, Type, OutType, IdxType>(
      dry_run, dots, data, D, N, (OutType)0, stream, false, raft::sq_op(), raft::add_op(), fin_op);
  } else if constexpr (norm_type == LinfNorm) {
    reduce<rowMajor, false, Type, OutType, IdxType>(
      false, dots, data, D, N, (OutType)0, stream, false, raft::abs_op(), raft::max_op(), fin_op);
  } else {
    THROW("Unsupported norm type: %d", norm_type);
  }
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
