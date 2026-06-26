/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __HISTOGRAM_H
#define __HISTOGRAM_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/stats/detail/histogram.cuh>
#include <raft/stats/stats_types.hpp>

// This file is a shameless amalgamation of independent works done by
// Lars Nyland and Andy Adinets

///@todo: add cub's histogram as another option

namespace raft {
namespace stats {

/**
 * Default mapper which just returns the value of the data itself
 */
template <typename DataT, typename IdxT>
struct IdentityBinner : public detail::IdentityBinner<DataT, IdxT> {
  IdentityBinner() : detail::IdentityBinner<DataT, IdxT>() {}
};

/**
 * @brief Perform histogram on the input data. It chooses the right load size
 * based on the input data vector length. It also supports large-bin cases
 * using a specialized smem-based hashing technique.
 * @tparam DataT input data type
 * @tparam IdxT data type used to compute indices
 * @tparam BinnerOp takes the input data and computes its bin index
 * @param type histogram implementation type to choose
 * @param bins the output bins (length = ncols * nbins)
 * @param nbins number of bins
 * @param data input data (length = ncols * nrows)
 * @param nrows data array length in each column (or batch)
 * @param ncols number of columns (or batch size)
 * @param stream cuda stream
 * @param binner the operation that computes the bin index of the input data
 *
 * @note signature of BinnerOp is `int func(DataT, IdxT);`
 */
template <typename DataT, typename IdxT = int, typename BinnerOp = IdentityBinner<DataT, IdxT>>
void histogram(HistType type,
               int* bins,
               IdxT nbins,
               const DataT* data,
               IdxT nrows,
               IdxT ncols,
               cudaStream_t stream,
               BinnerOp binner = IdentityBinner<DataT, IdxT>())
{
  detail::histogram<DataT, IdxT, BinnerOp>(type, bins, nbins, data, nrows, ncols, stream, binner);
}

/**
 * @defgroup stats_histogram Histogram
 * @{
 */

/**
 * @brief Perform histogram on the input data. It chooses the right load size
 * based on the input data vector length. It also supports large-bin cases
 * using a specialized smem-based hashing technique.
 * @tparam value_t input data type
 * @tparam idx_t data type used to compute indices
 * @tparam binner_op takes the input data and computes its bin index
 * @param[in]  handle the raft handle
 * @param[in]  type histogram implementation type to choose
 * @param[in]  data input data col-major (length = nrows * ncols)
 * @param[out] bins the output bins col-major (length = nbins * ncols)
 * @param[in]  binner the operation that computes the bin index of the input data
 *
 * @note signature of binner_op is `int func(value_t, IdxT);`
 */
template <typename value_t, typename idx_t, typename binner_op = IdentityBinner<value_t, idx_t>>
void histogram(raft::resources const& handle,
               HistType type,
               raft::device_matrix_view<const value_t, idx_t, raft::col_major> data,
               raft::device_matrix_view<int, idx_t, raft::col_major> bins,
               binner_op binner = IdentityBinner<value_t, idx_t>())
{
  // Seems like neither implementation of histogram does any CUDA allocations.
  // There is one allocation of std::vector inside Seive object in computeHashTableSize,
  // but it doesn't go through sdt::pmr, so isn't counted.
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(std::is_integral_v<idx_t> && data.extent(0) <= std::numeric_limits<int>::max(),
               "Index type not supported");
  RAFT_EXPECTS(bins.extent(1) == data.extent(1), "Size mismatch");
  RAFT_EXPECTS(bins.is_exhaustive(), "bins must be contiguous");
  RAFT_EXPECTS(data.is_exhaustive(), "data must be contiguous");
  detail::histogram<value_t, idx_t, binner_op>(type,
                                               bins.data_handle(),
                                               bins.extent(0),
                                               data.data_handle(),
                                               data.extent(0),
                                               data.extent(1),
                                               resource::get_cuda_stream(handle),
                                               binner);
}

/** @} */  // end group stats_histogram

};  // end namespace stats
};  // namespace raft
#endif
