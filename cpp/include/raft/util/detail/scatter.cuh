/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>
#include <raft/util/vectorized.cuh>

namespace raft {
namespace detail {

template <typename DataT, int VecLen, typename Lambda, typename IdxT>
RAFT_KERNEL scatterKernel(DataT* out, const DataT* in, const IdxT* idx, IdxT len, Lambda op)
{
  typedef TxN_t<DataT, VecLen> DataVec;
  typedef TxN_t<IdxT, VecLen> IdxVec;
  IdxT tid = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  tid *= VecLen;
  if (tid >= len) return;
  IdxVec idxIn;
  idxIn.load(idx, tid);
  DataVec dataIn;
#pragma unroll
  for (int i = 0; i < VecLen; ++i) {
    auto inPos         = idxIn.val.data[i];
    dataIn.val.data[i] = op(in[inPos], tid + i);
  }
  dataIn.store(out, tid);
}

template <typename DataT, int VecLen, typename Lambda, typename IdxT, int TPB>
void scatterImpl(
  DataT* out, const DataT* in, const IdxT* idx, IdxT len, Lambda op, cudaStream_t stream)
{
  const IdxT nblks = raft::ceildiv(VecLen ? len / VecLen : len, (IdxT)TPB);
  raft::launch_kernel(
    stream, nblks, TPB, scatterKernel<DataT, VecLen, Lambda, IdxT>, out, in, idx, len, op);
}

}  // namespace detail
}  // namespace raft
