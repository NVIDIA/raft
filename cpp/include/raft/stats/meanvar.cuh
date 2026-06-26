/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __MEANVAR_H
#define __MEANVAR_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/stats/detail/meanvar.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute mean and variance for each column of a given matrix.
 *
 * The operation is performed in a single sweep. Consider using it when you need to compute
 * both mean and variance, or when you need to compute variance but don't have the mean.
 * It's almost twice faster than running `mean` and `vars` sequentially, because all three
 * kernels are memory-bound.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used for addressing
 * @param [out] mean the output mean vector of size D
 * @param [out] var the output variance vector of size D
 * @param [in] data the input matrix of size [N, D]
 * @param [in] D number of columns of data
 * @param [in] N number of rows of data
 * @param [in] sample whether to evaluate sample variance or not. In other words, whether to
 * normalize the variance using N-1 or N, for true or false respectively.
 * @param [in] rowMajor whether the input data is row- or col-major, for true or false respectively.
 * @param [in] stream
 */
template <typename Type, typename IdxType = int>
void meanvar(Type* mean,
             Type* var,
             const Type* data,
             IdxType D,
             IdxType N,
             bool sample,
             bool rowMajor,
             cudaStream_t stream)
{
  detail::meanvar(false, mean, var, data, D, N, sample, rowMajor, stream);
}

/**
 * @defgroup stats_mean_var Mean and Variance
 * @{
 */

/**
 * @brief Compute mean and variance for each column of a given matrix.
 *
 * The operation is performed in a single sweep. Consider using it when you need to compute
 * both mean and variance, or when you need to compute variance but don't have the mean.
 * It's almost twice faster than running `mean` and `vars` sequentially, because all three
 * kernels are memory-bound.
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  data the input matrix of size [N, D]
 * @param[out] mean the output mean vector of size D
 * @param[out] var the output variance vector of size D
 * @param[in]  sample whether to evaluate sample variance or not. In other words, whether to
 * normalize the variance using N-1 or N, for true or false respectively.
 */
template <typename value_t, typename idx_t, typename layout_t>
void meanvar(raft::resources const& handle,
             raft::device_matrix_view<const value_t, idx_t, layout_t> data,
             raft::device_vector_view<value_t, idx_t> mean,
             raft::device_vector_view<value_t, idx_t> var,
             bool sample)
{
  static_assert(
    std::is_same_v<layout_t, raft::row_major> || std::is_same_v<layout_t, raft::col_major>,
    "Data layout not supported");
  RAFT_EXPECTS(data.extent(1) == var.extent(0), "Size mismatch between data and var");
  RAFT_EXPECTS(mean.size() == var.size(), "Size mismatch between mean and var");
  RAFT_EXPECTS(mean.is_exhaustive(), "mean must be contiguous");
  RAFT_EXPECTS(var.is_exhaustive(), "var must be contiguous");
  RAFT_EXPECTS(data.is_exhaustive(), "data must be contiguous");
  detail::meanvar(resource::get_dry_run_flag(handle),
                  mean.data_handle(),
                  var.data_handle(),
                  data.data_handle(),
                  data.extent(1),
                  data.extent(0),
                  sample,
                  std::is_same_v<layout_t, raft::row_major>,
                  resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_mean_var

};  // namespace stats
}  // namespace raft
#endif
