/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __MINMAX_H
#define __MINMAX_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/stats/detail/minmax.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <limits>
#include <optional>

namespace raft {
namespace stats {

/**
 * @brief Computes min/max across every column of the input matrix, as well as
 * optionally allow to subsample based on the given row/col ID mapping vectors
 *
 * @tparam T the data type
 * @tparam TPB number of threads per block
 * @param data input data
 * @param rowids actual row ID mappings. It is of length nrows. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param colids actual col ID mappings. It is of length ncols. If you want to
 * skip this index lookup entirely, pass nullptr
 * @param nrows number of rows of data to be worked upon. The actual rows of the
 * input "data" can be bigger than this!
 * @param ncols number of cols of data to be worked upon. The actual cols of the
 * input "data" can be bigger than this!
 * @param row_stride stride (in number of elements) between 2 adjacent columns
 * @param globalmin final col-wise global minimum (size = ncols)
 * @param globalmax final col-wise global maximum (size = ncols)
 * @param sampledcols output sampled data. Pass nullptr if you don't need this
 * @param stream cuda stream
 * @note This method makes the following assumptions:
 * 1. input and output matrices are assumed to be col-major
 * 2. ncols is small enough to fit the whole of min/max values across all cols
 *    in shared memory
 */
template <typename T, int TPB = 512>
void minmax(const T* data,
            const unsigned* rowids,
            const unsigned* colids,
            int nrows,
            int ncols,
            int row_stride,
            T* globalmin,
            T* globalmax,
            T* sampledcols,
            cudaStream_t stream)
{
  detail::minmax<T, TPB>(
    data, rowids, colids, nrows, ncols, row_stride, globalmin, globalmax, sampledcols, stream);
}

/**
 * @defgroup stats_minmax Min/Max
 * @{
 */

/**
 * @brief Computes min/max across every column of the input matrix, as well as
 * optionally allow to subsample based on the given row/col ID mapping vectors
 *
 * @tparam value_t Data type of input matrix element.
 * @tparam idx_t Index type of matrix extent.
 * @param[in]  handle the raft handle
 * @param[in]  data input data col-major of size [nrows, ncols], unless rowids or
 * colids length is smaller
 * @param[in]  rowids optional row ID mappings of length nrows. If you want to
 * skip this index lookup entirely, pass std::nullopt
 * @param[in]  colids optional col ID mappings of length ncols. If you want to
 * skip this index lookup entirely, pass std::nullopt
 * @param[out] globalmin final col-wise global minimum (size = ncols)
 * @param[out] globalmax final col-wise global maximum (size = ncols)
 * @param[out] sampledcols output sampled data. Pass std::nullopt if you don't need this
 * @note This method makes the following assumptions:
 * 1. input and output matrices are assumed to be col-major
 * 2. ncols is small enough to fit the whole of min/max values across all cols
 *    in shared memory
 */
template <typename value_t, typename idx_t>
void minmax(raft::resources const& handle,
            raft::device_matrix_view<const value_t, idx_t, raft::col_major> data,
            std::optional<raft::device_vector_view<const unsigned, idx_t>> rowids,
            std::optional<raft::device_vector_view<const unsigned, idx_t>> colids,
            raft::device_vector_view<value_t, idx_t> globalmin,
            raft::device_vector_view<value_t, idx_t> globalmax,
            std::optional<raft::device_vector_view<value_t, idx_t>> sampledcols)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  const unsigned* rowids_ptr = nullptr;
  const unsigned* colids_ptr = nullptr;
  value_t* sampledcols_ptr   = nullptr;
  auto nrows                 = data.extent(0);
  auto ncols                 = data.extent(1);
  auto row_stride            = data.stride(1);
  if (rowids.has_value()) {
    rowids_ptr = rowids.value().data_handle();
    RAFT_EXPECTS(rowids.value().extent(0) <= nrows, "Rowids size is greater than nrows");
    nrows = rowids.value().extent(0);
  }
  if (colids.has_value()) {
    colids_ptr = colids.value().data_handle();
    RAFT_EXPECTS(colids.value().extent(0) <= ncols, "Colids size is greater than ncols");
    ncols = colids.value().extent(0);
  }
  if (sampledcols.has_value()) { sampledcols_ptr = sampledcols.value().data_handle(); }
  RAFT_EXPECTS(globalmin.extent(0) == ncols, "Size mismatch between globalmin and ncols");
  RAFT_EXPECTS(globalmax.extent(0) == ncols, "Size mismatch between globalmax and ncols");
  detail::minmax<value_t>(data.data_handle(),
                          rowids_ptr,
                          colids_ptr,
                          nrows,
                          ncols,
                          row_stride,
                          globalmin.data_handle(),
                          globalmax.data_handle(),
                          sampledcols_ptr,
                          resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_minmax

};  // namespace stats
};  // namespace raft
#endif
