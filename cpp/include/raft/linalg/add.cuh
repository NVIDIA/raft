/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __ADD_H
#define __ADD_H

#pragma once

#include "detail/add.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @defgroup arithmetic Arithmetic functions
 * @{
 */

/**
 * @brief Elementwise scalar add operation on the input buffer
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
void addScalar(OutT* out, const InT* in, const InT scalar, IdxType len, cudaStream_t stream)
{
  detail::addScalar(out, in, scalar, len, stream);
}

/**
 * @brief Elementwise add operation on the input buffers
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
void add(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream)
{
  detail::add(out, in1, in2, len, stream);
}

/** Subtract single value pointed by singleScalarDev parameter in device memory from inDev[i] and
 * write result to outDev[i]
 * @tparam InT     input data-type. Also the data-type upon which the math ops
 *                 will be performed
 * @tparam OutT    output data-type
 * @tparam IdxType Integer type used to for addressing
 * @param outDev the output buffer
 * @param inDev the input buffer
 * @param singleScalarDev pointer to the scalar located in device memory
 * @param len number of elements in the input and output buffer
 * @param stream cuda stream
 */
template <typename InT, typename OutT = InT, typename IdxType = int>
void addDevScalar(
  OutT* outDev, const InT* inDev, const InT* singleScalarDev, IdxType len, cudaStream_t stream)
{
  detail::addDevScalar(outDev, inDev, singleScalarDev, len, stream);
}

/** @} */  // end of group add

/**
 * @defgroup add_dense Addition Arithmetic
 * @{
 */

/**
 * @brief Elementwise add operation
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @param[in] handle raft::resources
 * @param[in] in1    First Input
 * @param[in] in2    Second Input
 * @param[out] out    Output
 */
template <typename InType,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void add(raft::resources const& handle, InType in1, InType in2, OutType out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in1), "Input 1 must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in2), "Input 2 must be contiguous");
  RAFT_EXPECTS(out.size() == in1.size() && in1.size() == in2.size(),
               "Size mismatch between Output and Inputs");

  add<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in1.data_handle(),
    in2.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    resource::get_cuda_stream(handle));
}

/**
 * @brief Elementwise addition of device scalar to input
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param[in] handle raft::resources
 * @param[in] in    Input
 * @param[in] scalar    raft::device_scalar_view
 * @param[in] out    Output
 */
template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void add_scalar(raft::resources const& handle,
                InType in,
                OutType out,
                raft::device_scalar_view<const typename InType::value_type, ScalarIdxType> scalar)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  addDevScalar<in_value_t, out_value_t, typename OutType::index_type>(
    out.data_handle(),
    in.data_handle(),
    scalar.data_handle(),
    static_cast<typename OutType::index_type>(out.size()),
    resource::get_cuda_stream(handle));
}

/**
 * @brief Elementwise addition of host scalar to input
 * @tparam InType    Input Type raft::device_mdspan
 * @tparam OutType   Output Type raft::device_mdspan
 * @tparam ScalarIdxType Index Type of scalar
 * @param[in] handle raft::resources
 * @param[in] in    Input
 * @param[in] scalar    raft::host_scalar_view
 * @param[in] out    Output
 */
template <typename InType,
          typename OutType,
          typename ScalarIdxType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void add_scalar(raft::resources const& handle,
                const InType in,
                OutType out,
                raft::host_scalar_view<const typename InType::value_type, ScalarIdxType> scalar)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  using in_value_t  = typename InType::value_type;
  using out_value_t = typename OutType::value_type;

  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == in.size(), "Size mismatch between Output and Input");

  addScalar<in_value_t, out_value_t, typename OutType::index_type>(
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
