/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __COALESCED_REDUCTION_H
#define __COALESCED_REDUCTION_H

#pragma once

#include "detail/coalesced_reduction.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

namespace raft {
namespace linalg {

/**
 * @brief Compute reduction of the input matrix along the leading dimension
 *
 * @tparam InType the data type of the input
 * @tparam OutType the data type of the output (as well as the data type for
 *  which reduction is performed)
 * @tparam IdxType data type of the indices of the array
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param dots the output reduction vector
 * @param data the input matrix
 * @param D leading dimension of data
 * @param N second dimension data
 * @param init initial value to use for the reduction
 * @param main_op elementwise operation to apply before reduction
 * @param reduce_op binary reduction operation
 * @param final_op elementwise operation to apply before storing results
 * @param inplace reduction result added inplace or overwrites old values?
 * @param stream cuda stream where to launch work
 */
template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReduction(OutType* dots,
                        const InType* data,
                        IdxType D,
                        IdxType N,
                        OutType init,
                        cudaStream_t stream,
                        bool inplace           = false,
                        MainLambda main_op     = raft::identity_op(),
                        ReduceLambda reduce_op = raft::add_op(),
                        FinalLambda final_op   = raft::identity_op())
{
  detail::coalescedReduction<InType, OutType, IdxType>(
    false, dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
}

/**
 * @defgroup coalesced_reduction Coalesced Memory Access Reductions
 * For reducing along rows for col-major and along columns for row-major
 * @{
 */

/**
 * @brief Compute reduction of the input matrix along the leading dimension
 *        This API is to be used when the desired reduction is along the dimension
 *        of the memory layout. For example, a row-major matrix will be reduced
 *        along the columns whereas a column-major matrix will be reduced along
 *        the rows.
 *
 * @tparam InValueType the input data-type of underlying raft::matrix_view
 * @tparam LayoutPolicy The layout of Input/Output (row or col major)
 * @tparam OutValueType the output data-type of underlying raft::matrix_view and reduction
 * @tparam IndexType Integer type used to for addressing
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param handle raft::resources
 * @param[in] data Input of type raft::device_matrix_view
 * @param[out] dots Output of type raft::device_matrix_view
 * @param[in] init initial value to use for the reduction
 * @param[in] inplace reduction result added inplace or overwrites old values?
 * @param[in] main_op fused elementwise operation to apply before reduction
 * @param[in] reduce_op fused binary reduction operation
 * @param[in] final_op fused elementwise operation to apply before storing results
 */
template <typename InValueType,
          typename LayoutPolicy,
          typename OutValueType,
          typename IdxType,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalesced_reduction(raft::resources const& handle,
                         raft::device_matrix_view<const InValueType, IdxType, LayoutPolicy> data,
                         raft::device_vector_view<OutValueType, IdxType> dots,
                         OutValueType init,
                         bool inplace           = false,
                         MainLambda main_op     = raft::identity_op(),
                         ReduceLambda reduce_op = raft::add_op(),
                         FinalLambda final_op   = raft::identity_op())
{
  if constexpr (std::is_same_v<LayoutPolicy, raft::row_major>) {
    RAFT_EXPECTS(static_cast<IdxType>(dots.size()) == data.extent(0),
                 "Output should be equal to number of rows in Input");

    detail::coalescedReduction(resource::get_dry_run_flag(handle),
                               dots.data_handle(),
                               data.data_handle(),
                               data.extent(1),
                               data.extent(0),
                               init,
                               resource::get_cuda_stream(handle),
                               inplace,
                               main_op,
                               reduce_op,
                               final_op);
  } else if constexpr (std::is_same_v<LayoutPolicy, raft::col_major>) {
    RAFT_EXPECTS(static_cast<IdxType>(dots.size()) == data.extent(1),
                 "Output should be equal to number of columns in Input");

    detail::coalescedReduction(resource::get_dry_run_flag(handle),
                               dots.data_handle(),
                               data.data_handle(),
                               data.extent(0),
                               data.extent(1),
                               init,
                               resource::get_cuda_stream(handle),
                               inplace,
                               main_op,
                               reduce_op,
                               final_op);
  }
}

/** @} */  // end of group coalesced_reduction

};  // end namespace linalg
};  // end namespace raft

#endif
