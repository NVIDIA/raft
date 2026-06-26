/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __DIVIDE_H
#define __DIVIDE_H

#pragma once

#include "detail/divide.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @tparam OutT output data-type upon which the math operation will be performed
 * @tparam InT input data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename InT, typename OutT = InT, typename IdxType = int>
void divideScalar(OutT* out, const InT* in, InT scalar, IdxType len, cudaStream_t stream)
{
  detail::divideScalar(out, in, scalar, len, stream);
}
/** @} */

/**
 * @defgroup divide Division Arithmetic
 * @{
 */

/**
 * @brief Elementwise division of input by host scalar
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param[in] handle raft::resources
 * @param[in] in    Input
 * @param[in] scalar    raft::host_scalar_view
 * @param[out] out    Output
 */
template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void divide_scalar(raft::resources const& handle,
                   InType in,
                   OutType out,
                   raft::host_scalar_view<const typename InType::value_type, ScalarIdxType> scalar)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  divideScalar<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in.data_handle(),
    *scalar.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    resource::get_cuda_stream(handle));
}

/** @} */  // end of group add

};  // end namespace linalg
};  // end namespace raft

#endif
