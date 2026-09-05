/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __AXPY_H
#define __AXPY_H

#pragma once

#include "detail/axpy.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft {
namespace linalg {

/**
 * @brief the wrapper of cublas axpy function
 *  It computes the following equation: y = alpha * x + y
 *
 * @tparam T the element type
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @param [in] handle raft handle
 * @param [in] n number of elements in x and y
 * @param [in] alpha host or device scalar
 * @param [in] x vector of length n
 * @param [in] incx stride between consecutive elements of x
 * @param [inout] y vector of length n
 * @param [in] incy stride between consecutive elements of y
 * @param [in] stream
 */
template <typename T, bool DevicePointerMode = false>
void axpy(raft::resources const& handle,
          const int n,
          const T* alpha,
          const T* x,
          const int incx,
          T* y,
          const int incy,
          cudaStream_t stream)
{
  detail::axpy<T, DevicePointerMode>(handle, n, alpha, x, incx, y, incy, stream);
}

/**
 * @defgroup axpy axpy routine
 * @{
 */

/**
 * @brief axpy function
 *  It computes the following equation: y = alpha * x + y
 *
 * @param [in] handle raft::resources
 * @param [in] alpha raft::device_scalar_view
 * @param [in] x Input vector
 * @param [inout] y Output vector
 */
template <typename ElementType,
          typename IndexType,
          typename InLayoutPolicy,
          typename OutLayoutPolicy,
          typename ScalarIdxType>
void axpy(raft::resources const& handle,
          raft::device_scalar_view<const ElementType, ScalarIdxType> alpha,
          raft::device_vector_view<const ElementType, IndexType, InLayoutPolicy> x,
          raft::device_vector_view<ElementType, IndexType, OutLayoutPolicy> y)
{
  RAFT_EXPECTS(y.size() == x.size(), "Size mismatch between Output and Input");

  axpy<ElementType, true>(handle,
                          y.size(),
                          alpha.data_handle(),
                          x.data_handle(),
                          x.stride(0),
                          y.data_handle(),
                          y.stride(0),
                          resource::get_cuda_stream(handle).get());
}

/**
 * @brief axpy function
 *  It computes the following equation: y = alpha * x + y
 * @param [in] handle raft::resources
 * @param [in] alpha raft::device_scalar_view
 * @param [in] x Input vector
 * @param [inout] y Output vector
 */
template <typename ElementType,
          typename IndexType,
          typename InLayoutPolicy,
          typename OutLayoutPolicy,
          typename ScalarIdxType>
void axpy(raft::resources const& handle,
          raft::host_scalar_view<const ElementType, ScalarIdxType> alpha,
          raft::device_vector_view<const ElementType, IndexType, InLayoutPolicy> x,
          raft::device_vector_view<ElementType, IndexType, OutLayoutPolicy> y)
{
  RAFT_EXPECTS(y.size() == x.size(), "Size mismatch between Output and Input");

  axpy<ElementType, false>(handle,
                           y.size(),
                           alpha.data_handle(),
                           x.data_handle(),
                           x.stride(0),
                           y.data_handle(),
                           y.stride(0),
                           resource::get_cuda_stream(handle).get());
}

/** @} */  // end of group axpy

}  // namespace linalg
}  // namespace raft

#endif
