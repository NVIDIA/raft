/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../test_utils.cuh"

#include <raft/linalg/binary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename IdxType>
RAFT_KERNEL naiveAddKernel(OutType* out, const InType* in1, const InType* in2, IdxType len)
{
  IdxType idx = threadIdx.x + ((IdxType)blockIdx.x * (IdxType)blockDim.x);
  if (idx < len) { out[idx] = static_cast<OutType>(in1[idx] + in2[idx]); }
}

template <typename InType, typename IdxType = int, typename OutType = InType>
void naiveAdd(OutType* out, const InType* in1, const InType* in2, IdxType len)
{
  static const IdxType TPB = 64;
  IdxType nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(
    cudaStream_t{0}, nblks, TPB, naiveAddKernel<InType, OutType, IdxType>, out, in1, in2, len);
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct BinaryOpInputs {
  InType tolerance;
  IdxType len;
  unsigned long long int seed;
};

template <typename InType, typename IdxType = int, typename OutType = InType>
::std::ostream& operator<<(::std::ostream& os, const BinaryOpInputs<InType, IdxType, OutType>& d)
{
  return os;
}

}  // end namespace linalg
}  // end namespace raft
