/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/linalg/add.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

namespace raft {
namespace linalg {

template <typename InT, typename OutT = InT>
RAFT_KERNEL naiveAddElemKernel(OutT* out, const InT* in1, const InT* in2, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = OutT(in1[idx] + in2[idx]); }
}

template <typename InT, typename OutT = InT>
void naiveAddElem(OutT* out, const InT* in1, const InT* in2, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(stream, nblks, TPB, naiveAddElemKernel<InT, OutT>, out, in1, in2, len);
}

template <typename InT, typename OutT = InT>
struct AddInputs {
  OutT tolerance;
  int len;
  unsigned long long int seed;
};

template <typename InT, typename OutT = InT>
::std::ostream& operator<<(::std::ostream& os, const AddInputs<InT, OutT>& dims)
{
  return os;
}

};  // end namespace linalg
};  // end namespace raft
