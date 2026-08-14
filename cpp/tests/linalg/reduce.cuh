/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>

#include <rmm/device_uvector.hpp>

#include <cublas_v2.h>

namespace raft {
namespace linalg {

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
RAFT_KERNEL naiveCoalescedReductionKernel(OutType* dots,
                                          const InType* data,
                                          IdxType D,
                                          IdxType N,
                                          OutType init,
                                          bool inplace,
                                          MainLambda main_op,
                                          ReduceLambda reduce_op,
                                          FinalLambda fin_op)
{
  OutType acc      = init;
  IdxType rowStart = threadIdx.x + static_cast<IdxType>(blockIdx.x) * blockDim.x;
  if (rowStart < N) {
    for (IdxType i = 0; i < D; ++i) {
      acc = reduce_op(acc, main_op(data[rowStart * D + i], i));
    }
    if (inplace) {
      dots[rowStart] = fin_op(reduce_op(dots[rowStart], acc));
    } else {
      dots[rowStart] = fin_op(acc);
    }
  }
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void naiveCoalescedReduction(OutType* dots,
                             const InType* data,
                             IdxType D,
                             IdxType N,
                             cudaStream_t stream,
                             OutType init,
                             bool inplace           = false,
                             MainLambda main_op     = raft::identity_op(),
                             ReduceLambda reduce_op = raft::add_op(),
                             FinalLambda fin_op     = raft::identity_op())
{
  static const IdxType TPB = 64;
  IdxType nblks            = raft::ceildiv(N, TPB);
  raft::launch_kernel(stream,
                      nblks,
                      TPB,
                      naiveCoalescedReductionKernel,
                      dots,
                      data,
                      D,
                      N,
                      init,
                      inplace,
                      main_op,
                      reduce_op,
                      fin_op);
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
RAFT_KERNEL naiveStridedReductionKernel(OutType* dots,
                                        const InType* data,
                                        IdxType D,
                                        IdxType N,
                                        OutType init,
                                        bool inplace,
                                        MainLambda main_op,
                                        ReduceLambda reduce_op,
                                        FinalLambda fin_op)
{
  OutType acc = init;
  IdxType col = threadIdx.x + static_cast<IdxType>(blockIdx.x) * blockDim.x;
  if (col < D) {
    for (IdxType i = 0; i < N; ++i) {
      acc = reduce_op(acc, main_op(data[i * D + col], i));
    }
    if (inplace) {
      dots[col] = fin_op(reduce_op(dots[col], acc));
    } else {
      dots[col] = fin_op(acc);
    }
  }
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void naiveStridedReduction(OutType* dots,
                           const InType* data,
                           IdxType D,
                           IdxType N,
                           cudaStream_t stream,
                           OutType init,
                           bool inplace           = false,
                           MainLambda main_op     = raft::identity_op(),
                           ReduceLambda reduce_op = raft::add_op(),
                           FinalLambda fin_op     = raft::identity_op())
{
  static const IdxType TPB = 64;
  IdxType nblks            = raft::ceildiv(D, TPB);
  raft::launch_kernel(stream,
                      nblks,
                      TPB,
                      naiveStridedReductionKernel,
                      dots,
                      data,
                      D,
                      N,
                      init,
                      inplace,
                      main_op,
                      reduce_op,
                      fin_op);
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void naiveReduction(OutType* dots,
                    const InType* data,
                    IdxType D,
                    IdxType N,
                    bool rowMajor,
                    bool alongRows,
                    cudaStream_t stream,
                    OutType init,
                    bool inplace           = false,
                    MainLambda main_op     = raft::identity_op(),
                    ReduceLambda reduce_op = raft::add_op(),
                    FinalLambda fin_op     = raft::identity_op())
{
  if (rowMajor && alongRows) {
    naiveCoalescedReduction(dots, data, D, N, stream, init, inplace, main_op, reduce_op, fin_op);
  } else if (rowMajor && !alongRows) {
    naiveStridedReduction(dots, data, D, N, stream, init, inplace, main_op, reduce_op, fin_op);
  } else if (!rowMajor && alongRows) {
    naiveStridedReduction(dots, data, N, D, stream, init, inplace, main_op, reduce_op, fin_op);
  } else {
    naiveCoalescedReduction(dots, data, N, D, stream, init, inplace, main_op, reduce_op, fin_op);
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // end namespace linalg
}  // end namespace raft
