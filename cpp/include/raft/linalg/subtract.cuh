/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __SUBTRACT_H
#define __SUBTRACT_H

#pragma once

#include "detail/subtract.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief Elementwise scalar subtraction operation on the input buffer
 *
 * @tparam InT     input data-type. Also the data-type upon which the math ops
 *                 will be performed
 * @tparam OutT    output data-type
 * @tparam IdxType Integer type used to for addressing
 *
 * @param out    the output buffer
 * @param in     the input buffer
 * @param scalar the scalar used in the operations
 * @param len    number of elements in the input buffer
 * @param stream cuda stream where to launch work
 */
template <typename InT, typename OutT = InT, typename IdxType = int>
void subtractScalar(OutT* out, const InT* in, InT scalar, IdxType len, cudaStream_t stream)
{
  detail::subtractScalar(out, in, scalar, len, stream);
}

/**
 * @brief Elementwise subtraction operation on the input buffers
 * @tparam InT     input data-type. Also the data-type upon which the math ops
 *                 will be performed
 * @tparam OutT    output data-type
 * @tparam IdxType Integer type used to for addressing
 *
 * @param out    the output buffer
 * @param in1    the first input buffer
 * @param in2    the second input buffer
 * @param len    number of elements in the input buffers
 * @param stream cuda stream where to launch work
 */
template <typename InT, typename OutT = InT, typename IdxType = int>
void subtract(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream)
{
  detail::subtract(out, in1, in2, len, stream);
}

/** Subtract single value pointed by singleScalarDev parameter in device memory from inDev[i] and
 * write result to outDev[i]
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param outDev the output buffer
 * @param inDev the input buffer
 * @param singleScalarDev pointer to the scalar located in device memory
 * @param len number of elements in the input and output buffer
 * @param stream cuda stream
 * @remark block size has not been tuned
 */
template <typename math_t, typename IdxType = int, int TPB = 256>
void subtractDevScalar(math_t* outDev,
                       const math_t* inDev,
                       const math_t* singleScalarDev,
                       IdxType len,
                       cudaStream_t stream)
{
  detail::subtractDevScalar(outDev, inDev, singleScalarDev, len, stream);
}

/**
 * @defgroup sub Subtraction Arithmetic
 * @{
 */

/**
 * @brief Elementwise subtraction operation on the input buffers
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @param handle raft::resources
 * @param[in] in1    First Input
 * @param[in] in2    Second Input
 * @param[out] out    Output
 */
template <typename InType,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void subtract(raft::resources const& handle, InType in1, InType in2, OutType out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in1), "Input 1 must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in2), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  subtract<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in1.data_handle(),
    in2.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    resource::get_cuda_stream(handle));
}

/**
 * @brief Elementwise subtraction of device scalar to input
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param[in] handle raft::resources
 * @param[in] in    Input
 * @param[out] out    Output
 * @param[in] scalar    raft::device_scalar_view
 */
template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void subtract_scalar(
  raft::resources const& handle,
  InType in,
  OutType out,
  raft::device_scalar_view<const typename InType::element_type, ScalarIdxType> scalar)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  subtractDevScalar<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in.data_handle(),
    scalar.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    resource::get_cuda_stream(handle));
}

/**
 * @brief Elementwise subtraction of host scalar to input
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param[in] handle raft::resources
 * @param[in] in    Input
 * @param[out] out    Output
 * @param[in] scalar    raft::host_scalar_view
 */
template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void subtract_scalar(
  raft::resources const& handle,
  InType in,
  OutType out,
  raft::host_scalar_view<const typename InType::element_type, ScalarIdxType> scalar)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  subtractScalar<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in.data_handle(),
    *scalar.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    resource::get_cuda_stream(handle));
}

/** @} */  // end of group subtract

};  // end namespace linalg
};  // end namespace raft

#endif
