/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../test_utils.cuh"

#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename IdxType>
RAFT_KERNEL naiveScaleKernel(OutType* out, const InType* in, InType scalar, IdxType len)
{
  IdxType idx = threadIdx.x + ((IdxType)blockIdx.x * (IdxType)blockDim.x);
  if (idx < len) {
    if (in == nullptr) {
      // used for testing write_only_unary_op
      out[idx] = static_cast<OutType>(scalar * idx);
    } else {
      out[idx] = static_cast<OutType>(scalar * in[idx]);
    }
  }
}

template <typename InType, typename IdxType = int, typename OutType = InType>
void naiveScale(OutType* out, const InType* in, InType scalar, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(
    stream, nblks, TPB, naiveScaleKernel<InType, OutType, IdxType>, out, in, scalar, len);
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct UnaryOpInputs {
  OutType tolerance;
  IdxType len;
  InType scalar;
  unsigned long long int seed;
};

template <typename InType, typename IdxType = int, typename OutType = InType>
::std::ostream& operator<<(::std::ostream& os, const UnaryOpInputs<InType, IdxType, OutType>& d)
{
  return os;
}

}  // end namespace linalg
}  // end namespace raft
