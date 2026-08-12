/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>
#include <raft/util/vectorized.cuh>

#include <cub/block/block_reduce.cuh>

namespace raft {
namespace linalg {
namespace detail {

struct sum_tag {};

template <typename InType, typename OutType, int TPB>
__device__ void reduce(OutType* out, const InType acc, sum_tag)
{
  typedef cub::BlockReduce<InType, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  OutType tmp = BlockReduce(temp_storage).Sum(acc);
  if (threadIdx.x == 0) { raft::myAtomicAdd(out, tmp); }
}

template <typename InType, typename OutType, int TPB, typename ReduceLambda>
__device__ void reduce(OutType* out, const InType acc, ReduceLambda op)
{
  typedef cub::BlockReduce<InType, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  OutType tmp = BlockReduce(temp_storage).Reduce(acc, op);
  if (threadIdx.x == 0) { raft::myAtomicReduce(out, tmp, op); }
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MapOp,
          typename ReduceLambda,
          int TPB,
          typename... Args>
RAFT_KERNEL mapThenReduceKernel(OutType* out,
                                IdxType len,
                                OutType neutral,
                                MapOp map,
                                ReduceLambda op,
                                const InType* in,
                                Args... args)
{
  OutType acc = neutral;
  auto idx    = (threadIdx.x + (blockIdx.x * blockDim.x));

  if (idx < len) { acc = map(in[idx], args[idx]...); }

  __syncthreads();

  reduce<InType, OutType, TPB>(out, acc, op);
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MapOp,
          typename ReduceLambda,
          int TPB,
          typename... Args>
void mapThenReduceImpl(OutType* out,
                       IdxType len,
                       OutType neutral,
                       MapOp map,
                       ReduceLambda op,
                       cudaStream_t stream,
                       const InType* in,
                       Args... args)
{
  raft::update_device(out, &neutral, 1, stream);
  const int nblks = raft::ceildiv(len, IdxType(TPB));
  raft::launch_kernel(
    stream,
    nblks,
    TPB,
    mapThenReduceKernel<InType, OutType, IdxType, MapOp, ReduceLambda, TPB, Args...>,
    out,
    len,
    neutral,
    map,
    op,
    in,
    args...);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
