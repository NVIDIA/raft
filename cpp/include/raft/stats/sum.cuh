/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __SUM_H
#define __SUM_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/types.hpp>
#include <raft/linalg/detail/reduce.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace stats {

/**
 * @brief Compute sum of the input matrix
 *
 * Sum operation is assumed to be performed on a given column.
 *
 * @tparam rowMajor: whether the input data is row or col major
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param output the output mean vector
 * @param input the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor, typename Type, typename IdxType = int>
void sum(Type* output, const Type* input, IdxType D, IdxType N, cudaStream_t stream)
{
  // Inline detail::sum: reduce along columns (alongRows=false)
  raft::linalg::detail::reduce<rowMajor, false>(false, output, input, D, N, Type(0), stream);
}

/**
 * @defgroup stats_sum Sum
 * @{
 */

/**
 * @brief Compute sum of the input matrix
 *
 * Sum operation is assumed to be performed on a given column.
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  input the input matrix
 * @param[out] output the output mean vector
 */
template <typename value_t, typename idx_t, typename layout_t>
void sum(raft::resources const& handle,
         raft::device_matrix_view<const value_t, idx_t, layout_t> input,
         raft::device_vector_view<value_t, idx_t> output)
{
  constexpr bool is_row_major = std::is_same_v<layout_t, raft::row_major>;
  constexpr bool is_col_major = std::is_same_v<layout_t, raft::col_major>;
  static_assert(is_row_major || is_col_major,
                "sum: Layout must be either "
                "raft::row_major or raft::col_major (or one of their aliases)");
  RAFT_EXPECTS(input.extent(1) == output.extent(0), "Size mismatch between input and output");
  // Use public reduce API that handles dry-run internally
  // Sum along columns (Apply::ALONG_COLUMNS), reduce_op is add_op (default)
  raft::linalg::reduce<raft::Apply::ALONG_COLUMNS>(handle, input, output, value_t(0));
}

/** @} */  // end group stats_sum

};  // end namespace stats
};  // namespace raft
#endif
