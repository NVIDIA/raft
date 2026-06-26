/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __UNARY_OP_H
#define __UNARY_OP_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>

namespace raft {
namespace linalg {

/**
 * @brief perform element-wise unary operation in the input array
 * @tparam InType input data-type
 * @tparam Lambda Device lambda performing the actual operation, with the signature
 *         `OutType func(const InType& val);`
 * @tparam OutType output data-type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads-per-block in the final kernel launched
 * @param[out] out    Output array [on device], dim = [len]
 * @param[in]  in     Input array [on device], dim = [len]
 * @param[in]  len    Number of elements in the input array
 * @param[in]  op     Device lambda
 * @param[in]  stream cuda stream where to launch work
 */
template <typename InType,
          typename Lambda,
          typename IdxType = int,
          typename OutType = InType,
          int TPB          = 256>
void unaryOp(OutType* out, const InType* in, IdxType len, Lambda op, cudaStream_t stream)
{
  return detail::map<false>(stream, out, len, op, in);
}

/**
 * @brief Perform an element-wise unary operation into the output array
 *
 * Compared to `unaryOp()`, this method does not do any reads from any inputs
 *
 * @tparam OutType output data-type
 * @tparam Lambda  Device lambda performing the actual operation, with the signature
 *                 `void func(OutType* outLocationOffset, IdxType idx);`
 *                 where outLocationOffset will be out + idx.
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB     threads-per-block in the final kernel launched
 *
 * @param[out] out    Output array [on device], dim = [len]
 * @param[in]  len    Number of elements in the input array
 * @param[in]  op     Device lambda
 * @param[in]  stream cuda stream where to launch work
 */
template <typename OutType, typename Lambda, typename IdxType = int, int TPB = 256>
void writeOnlyUnaryOp(OutType* out, IdxType len, Lambda op, cudaStream_t stream)
{
  return detail::map<true>(stream, out, len, [op] __device__(IdxType offset) {
    OutType r;
    op(&r, offset);
    return r;
  });
}

/**
 * @defgroup unary_op Element-Wise Unary Operations
 * @{
 */

/**
 * @brief Perform an element-wise unary operation into the output array
 * @tparam InType Input Type raft::device_mdspan
 * @tparam Lambda Device lambda performing the actual operation, with the signature
 *                `out_value_t func(const in_value_t& val);`
 * @tparam OutType Output Type raft::device_mdspan
 * @param[in]  handle The raft handle
 * @param[in]  in     Input
 * @param[out] out    Output
 * @param[in]  op     Device lambda
 */
template <typename InType,
          typename Lambda,
          typename OutType,
          typename = raft::enable_if_input_device_mdspan<InType>,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void unary_op(raft::resources const& handle, InType in, OutType out, Lambda op)
{
  return map(handle, in, out, op);
}

/**
 * @brief Perform an element-wise unary operation on the input index into the output array
 *
 * @note This operation is deprecated. Please use map_offset in `raft/linalg/map.cuh` instead.
 *
 * @tparam OutType Output Type raft::device_mdspan
 * @tparam Lambda  Device lambda performing the actual operation, with the signature
 *                 `void func(out_value_t* out_location, index_t idx);`
 * @param[in]  handle The raft handle
 * @param[out] out    Output
 * @param[in]  op     Device lambda
 */
template <typename OutType,
          typename Lambda,
          typename = raft::enable_if_output_device_mdspan<OutType>>
void write_only_unary_op(const raft::resources& handle, OutType out, Lambda op)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  return writeOnlyUnaryOp(out.data_handle(), out.size(), op, resource::get_cuda_stream(handle));
}

/** @} */  // end of group unary_op

};  // end namespace linalg
};  // end namespace raft

#endif
