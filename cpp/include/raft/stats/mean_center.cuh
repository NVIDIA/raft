/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __MEAN_CENTER_H
#define __MEAN_CENTER_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/stats/detail/mean_center.cuh>

namespace raft {
namespace stats {

/**
 * @brief Center the input matrix wrt its mean
 * @tparam rowMajor whether input is row or col major
 * @tparam bcastAlongRows whether to broadcast vector along rows or columns
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-centered matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void meanCenter(
  Type* out, const Type* data, const Type* mu, IdxType D, IdxType N, cudaStream_t stream)
{
  detail::meanCenter<rowMajor, bcastAlongRows, Type, IdxType, TPB>(
    false, out, data, mu, D, N, stream);
}

/**
 * @brief Add the input matrix wrt its mean
 * @tparam rowMajor whether input is row or col major
 * @tparam bcastAlongRows whether to broadcast vector along rows or columns
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-added matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, bool bcastAlongRows, typename Type, typename IdxType = int, int TPB = 256>
void meanAdd(Type* out, const Type* data, const Type* mu, IdxType D, IdxType N, cudaStream_t stream)
{
  detail::meanAdd<rowMajor, bcastAlongRows, Type, IdxType, TPB>(false, out, data, mu, D, N, stream);
}

/**
 * @defgroup stats_mean_center Mean Center
 * @{
 */

/**
 * @brief Center the input matrix wrt its mean
 * @tparam apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::Apply
 * @tparam value_t the data type
 * @tparam idx_t index type
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  data input matrix of size nrows * ncols
 * @param[in]  mu the mean vector of size ncols if bcast_along_rows else nrows
 * @param[out] out the output mean-centered matrix
 */
template <Apply apply, typename value_t, typename idx_t, typename layout_t>
void mean_center(raft::resources const& handle,
                 raft::device_matrix_view<const value_t, idx_t, layout_t> data,
                 raft::device_vector_view<const value_t, idx_t> mu,
                 raft::device_matrix_view<value_t, idx_t, layout_t> out)
{
  static_assert(
    std::is_same_v<layout_t, raft::row_major> || std::is_same_v<layout_t, raft::col_major>,
    "Data layout not supported");
  auto mean_vec_size = apply == Apply::ALONG_ROWS ? data.extent(1) : data.extent(0);
  RAFT_EXPECTS(out.extents() == data.extents(), "Size mismatch");
  RAFT_EXPECTS(mean_vec_size == mu.extent(0), "Size mismatch between data and mu");
  RAFT_EXPECTS(out.is_exhaustive(), "out must be contiguous");
  RAFT_EXPECTS(data.is_exhaustive(), "data must be contiguous");
  detail::meanCenter<std::is_same_v<layout_t, raft::row_major>,
                     apply == Apply::ALONG_ROWS,
                     value_t,
                     idx_t>(resource::get_dry_run_flag(handle),
                            out.data_handle(),
                            data.data_handle(),
                            mu.data_handle(),
                            data.extent(1),
                            data.extent(0),
                            resource::get_cuda_stream(handle));
}

/**
 * @brief Add the input matrix wrt its mean
 * @tparam apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::Apply
 * @tparam Type the data type
 * @tparam idx_t index type
 * @tparam layout_t Layout type of the input matrix.
 * @tparam TPB threads per block of the cuda kernel launched
 * @param[in]  handle the raft handle
 * @param[in]  data input matrix of size nrows * ncols
 * @param[in]  mu the mean vector of size ncols if bcast_along_rows else nrows
 * @param[out] out the output mean-centered matrix
 */
template <Apply apply, typename value_t, typename idx_t, typename layout_t>
void mean_add(raft::resources const& handle,
              raft::device_matrix_view<const value_t, idx_t, layout_t> data,
              raft::device_vector_view<const value_t, idx_t> mu,
              raft::device_matrix_view<value_t, idx_t, layout_t> out)
{
  static_assert(
    std::is_same_v<layout_t, raft::row_major> || std::is_same_v<layout_t, raft::col_major>,
    "Data layout not supported");
  auto mean_vec_size = apply == Apply::ALONG_ROWS ? data.extent(1) : data.extent(0);
  RAFT_EXPECTS(out.extents() == data.extents(), "Size mismatch");
  RAFT_EXPECTS(mean_vec_size == mu.extent(0), "Size mismatch between data and mu");
  RAFT_EXPECTS(out.is_exhaustive(), "out must be contiguous");
  RAFT_EXPECTS(data.is_exhaustive(), "data must be contiguous");
  detail::
    meanAdd<std::is_same_v<layout_t, raft::row_major>, apply == Apply::ALONG_ROWS, value_t, idx_t>(
      resource::get_dry_run_flag(handle),
      out.data_handle(),
      data.data_handle(),
      mu.data_handle(),
      data.extent(1),
      data.extent(0),
      resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_mean_center

};  // end namespace stats
};  // namespace raft
#endif
