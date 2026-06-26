/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __DOT_H
#define __DOT_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>

namespace raft {
namespace linalg {

/**
 * @defgroup dot BLAS dot routine
 * @{
 */

/**
 * @brief Computes the dot product of two vectors.
 * @param[in] handle   raft::resources
 * @param[in] x        First input vector
 * @param[in] y        Second input vector
 * @param[out] out     The output dot product between the x and y vectors.
 */
template <typename ElementType,
          typename IndexType,
          typename ScalarIndexType,
          typename LayoutPolicy1,
          typename LayoutPolicy2>
void dot(raft::resources const& handle,
         raft::device_vector_view<const ElementType, IndexType, LayoutPolicy1> x,
         raft::device_vector_view<const ElementType, IndexType, LayoutPolicy2> y,
         raft::device_scalar_view<ElementType, ScalarIndexType> out)
{
  RAFT_EXPECTS(x.size() == y.size(),
               "Size mismatch between x and y input vectors in raft::linalg::dot");
  if (resource::get_dry_run_flag(handle)) { return; }

  RAFT_CUBLAS_TRY(detail::cublasdot(resource::get_cublas_handle(handle),
                                    x.size(),
                                    x.data_handle(),
                                    x.stride(0),
                                    y.data_handle(),
                                    y.stride(0),
                                    out.data_handle(),
                                    resource::get_cuda_stream(handle)));
}

/**
 * @brief Computes the dot product of two vectors.
 * @param[in] handle   raft::resources
 * @param[in] x        First input vector
 * @param[in] y        Second input vector
 * @param[out] out     The output dot product between the x and y vectors.
 */
template <typename ElementType,
          typename IndexType,
          typename ScalarIndexType,
          typename LayoutPolicy1,
          typename LayoutPolicy2>
void dot(raft::resources const& handle,
         raft::device_vector_view<const ElementType, IndexType, LayoutPolicy1> x,
         raft::device_vector_view<const ElementType, IndexType, LayoutPolicy2> y,
         raft::host_scalar_view<ElementType, ScalarIndexType> out)
{
  RAFT_EXPECTS(x.size() == y.size(),
               "Size mismatch between x and y input vectors in raft::linalg::dot");
  if (resource::get_dry_run_flag(handle)) { return; }

  RAFT_CUBLAS_TRY(detail::cublasdot(resource::get_cublas_handle(handle),
                                    x.size(),
                                    x.data_handle(),
                                    x.stride(0),
                                    y.data_handle(),
                                    y.stride(0),
                                    out.data_handle(),
                                    resource::get_cuda_stream(handle)));
}

/** @} */  // end of group dot

}  // namespace linalg
}  // namespace raft
#endif
