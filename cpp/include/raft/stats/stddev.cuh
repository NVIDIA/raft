/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __STDDEV_H
#define __STDDEV_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/detail/stddev.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam rowMajor: whether the input data is row or col major
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param std the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, typename Type, typename IdxType = int>
void stddev(Type* std,
            const Type* data,
            const Type* mu,
            IdxType D,
            IdxType N,
            bool sample,
            cudaStream_t stream)
{
  detail::stddev<rowMajor>(false, std, data, mu, D, N, sample, stream);
}

/**
 * @brief Compute variance of the input matrix
 *
 * Variance operation is assumed to be performed on a given column.
 *
 * @tparam rowMajor: whether the input data is row or col major
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param var the output stddev vector
 * @param data the input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, typename Type, typename IdxType = int>
void vars(Type* var,
          const Type* data,
          const Type* mu,
          IdxType D,
          IdxType N,
          bool sample,
          cudaStream_t stream)
{
  detail::vars<rowMajor>(false, var, data, mu, D, N, sample, stream);
}

/**
 * @defgroup stats_stddev Standard Deviation
 * @{
 */

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  data the input matrix
 * @param[in]  mu the mean vector
 * @param[out] std the output stddev vector
 * @param[in]  sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 */
template <typename value_t, typename idx_t, typename layout_t>
void stddev(raft::resources const& handle,
            raft::device_matrix_view<const value_t, idx_t, layout_t> data,
            raft::device_vector_view<const value_t, idx_t> mu,
            raft::device_vector_view<value_t, idx_t> std,
            bool sample)
{
  constexpr bool is_row_major = std::is_same_v<layout_t, raft::row_major>;
  constexpr bool is_col_major = std::is_same_v<layout_t, raft::col_major>;
  static_assert(is_row_major || is_col_major,
                "stddev: Layout must be either "
                "raft::row_major or raft::col_major (or one of their aliases)");
  RAFT_EXPECTS(mu.size() == std.size(), "Size mismatch between mu and std");
  RAFT_EXPECTS(mu.extent(0) == data.extent(1), "Size mismatch between data and mu");
  detail::stddev<is_row_major>(resource::get_dry_run_flag(handle),
                               std.data_handle(),
                               data.data_handle(),
                               mu.data_handle(),
                               data.extent(1),
                               data.extent(0),
                               sample,
                               resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_stddev

/**
 * @defgroup stats_variance Variance
 * @{
 */

/**
 * @brief Compute variance of the input matrix
 *
 * Variance operation is assumed to be performed on a given column.
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  data the input matrix
 * @param[in]  mu the mean vector
 * @param[out] var the output stddev vector
 * @param[in]  sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 */
template <typename value_t, typename idx_t, typename layout_t>
void vars(raft::resources const& handle,
          raft::device_matrix_view<const value_t, idx_t, layout_t> data,
          raft::device_vector_view<const value_t, idx_t> mu,
          raft::device_vector_view<value_t, idx_t> var,
          bool sample)
{
  constexpr bool is_row_major = std::is_same_v<layout_t, raft::row_major>;
  constexpr bool is_col_major = std::is_same_v<layout_t, raft::col_major>;
  static_assert(is_row_major || is_col_major,
                "vars: Layout must be either "
                "raft::row_major or raft::col_major (or one of their aliases)");
  RAFT_EXPECTS(mu.size() == var.size(), "Size mismatch between mu and std");
  RAFT_EXPECTS(mu.extent(0) == data.extent(1), "Size mismatch between data and mu");
  detail::vars<is_row_major>(resource::get_dry_run_flag(handle),
                             var.data_handle(),
                             data.data_handle(),
                             mu.data_handle(),
                             data.extent(1),
                             data.extent(0),
                             sample,
                             resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_variance

};  // namespace stats
};  // namespace raft
#endif
