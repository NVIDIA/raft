/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __MAP_THEN_REDUCE_H
#define __MAP_THEN_REDUCE_H

#pragma once

#include "detail/map_then_reduce.cuh"

#include <raft/core/detail/macros.hpp>

namespace raft {
namespace linalg {

/**
 * @brief CUDA version of map and then sum reduction operation
 * @tparam InType data-type upon which the math operation will be performed
 * @tparam MapOp the device-lambda performing the actual operation
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Args additional parameters
 * @param out the output sum-reduced value (assumed to be a device pointer)
 * @param len number of elements in the input array
 * @param map the device-lambda
 * @param stream cuda-stream where to launch this kernel
 * @param in the input array
 * @param args additional input arrays
 */

template <typename InType,
          typename MapOp,
          typename IdxType = std::uint32_t,
          int TPB          = 256,
          typename... Args,
          typename OutType = InType>
void mapThenSumReduce(
  OutType* out, IdxType len, MapOp map, cudaStream_t stream, const InType* in, Args... args)
{
  detail::mapThenReduceImpl<InType, OutType, IdxType, MapOp, detail::sum_tag, TPB, Args...>(
    out, len, (OutType)0, map, detail::sum_tag(), stream, in, args...);
}

/**
 * @brief CUDA version of map and then generic reduction operation
 * @tparam InType data-type upon which the math operation will be performed
 * @tparam MapOp the device-lambda performing the actual map operation
 * @tparam ReduceLambda the device-lambda performing the actual reduction
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Args additional parameters
 * @param out the output reduced value (assumed to be a device pointer)
 * @param len number of elements in the input array
 * @param neutral The neutral element of the reduction operation. For example:
 *    0 for sum, 1 for multiply, +Inf for Min, -Inf for Max
 * @param map the device-lambda
 * @param op the reduction device lambda
 * @param stream cuda-stream where to launch this kernel
 * @param in the input array
 * @param args additional input arrays
 */
template <typename InType,
          typename MapOp,
          typename ReduceLambda,
          typename IdxType = std::uint32_t,
          int TPB          = 256,
          typename OutType = InType,
          typename... Args>
[[deprecated("Use function `mapReduce` from `raft/linalg/map_reduce.cuh")]] void mapThenReduce(
  OutType* out,
  size_t len,
  OutType neutral,
  MapOp map,
  ReduceLambda op,
  cudaStream_t stream,
  const InType* in,
  Args... args)
{
  detail::mapThenReduceImpl<InType, OutType, IdxType, MapOp, ReduceLambda, TPB, Args...>(
    out, len, neutral, map, op, stream, in, args...);
}

};  // end namespace linalg
};  // end namespace raft

#endif
