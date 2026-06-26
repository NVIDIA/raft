/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __SQRT_H
#define __SQRT_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/linalg/unary_op.cuh>

namespace raft {
namespace linalg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in the input buffer
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename in_t, typename out_t = in_t, typename IdxType = int>
void sqrt(out_t* out, const in_t* in, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::sqrt_op{}, stream);
}
/** @} */

/**
 * @defgroup sqrt Sqrt Arithmetic
 * @{
 */

/**
 * @brief Elementwise sqrt operation
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @param[in] handle raft::resources
 * @param[in] in     Input
 * @param[out] out    Output
 */
template <typename InType,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void sqrt(raft::resources const& handle, InType in, OutType out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input 1 must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Inputs");

  sqrt<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    resource::get_cuda_stream(handle));
}

/** @} */  // end of group add

};  // end namespace linalg
};  // end namespace raft

#endif
