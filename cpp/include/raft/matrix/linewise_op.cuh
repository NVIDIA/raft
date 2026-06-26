/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/types.hpp>
#include <raft/matrix/detail/linewise_op.cuh>

namespace raft {
namespace matrix {

/**
 * @defgroup linewise_op Matrix Linewise Operations
 * @{
 */

/**
 * Run a function over matrix lines (rows or columns) with a variable number
 * row-vectors or column-vectors.
 * The term `line` here signifies that the lines can be either columns or rows,
 * depending on the matrix layout.
 * What matters is if the vectors are applied along lines (indices of vectors correspond to
 * indices within lines), or across lines (indices of vectors correspond to line numbers).
 * @tparam alongLines whether vectors are indices along or across lines.
 * @tparam m_t matrix elements type
 * @tparam idx_t integer type used for indexing
 * @tparam layout layout of the matrix data (must be row or col major)
 * @tparam Lambda type of lambda function used for the operation
 * @tparam vec_t variadic types of device_vector_view vectors (size m if alongRows, size n
 * otherwise)
 * @param[in] handle raft handle for managing resources
 * @param [out] out result of the operation; can be same as `in`; should be aligned the same
 *        as `in` to allow faster vectorized memory transfers.
 * @param [in] in input matrix consisting of `nLines` lines, each `lineLen`-long.
 * @param [in] op the operation applied on each line:
 *    for i in [0..lineLen) and j in [0..nLines):
 *      out[j, i] = op(in[j, i], vec1[i], vec2[i], ... veck[i])   if alongLines = true
 *      out[j, i] = op(in[j, i], vec1[j], vec2[j], ... veck[j])   if alongLines = false
 *    where matrix indexing is row-major ([j, i] = [i + lineLen * j]).
 *      out[i, j] = op(in[i, j], vec1[i], vec2[i], ... veck[i])   if alongLines = true
 *      out[i, j] = op(in[i, j], vec1[j], vec2[j], ... veck[j])   if alongLines = false
 *    where matrix indexing is col-major ([i, j] = [i + lineLen * j]).
 * @param [in] vecs zero or more vectors to be passed as arguments,
 *    size of each vector is `alongLines ? lineLen : nLines`.
 */
template <raft::Apply apply,
          typename m_t,
          typename idx_t,
          typename layout,
          typename Lambda,
          typename... vec_t,
          typename = raft::enable_if_device_mdspan<vec_t...>>
void linewise_op(raft::resources const& handle,
                 raft::device_matrix_view<const m_t, idx_t, layout> in,
                 raft::device_matrix_view<m_t, idx_t, layout> out,
                 Lambda op,
                 vec_t... vecs)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  constexpr auto is_rowmajor = std::is_same_v<layout, row_major>;
  constexpr auto is_colmajor = std::is_same_v<layout, col_major>;

  static_assert(is_rowmajor || is_colmajor,
                "layout for in and out must be either row or col major");

  const idx_t nLines  = is_rowmajor ? in.extent(0) : in.extent(1);
  const idx_t lineLen = is_rowmajor ? in.extent(1) : in.extent(0);

  RAFT_EXPECTS(out.extent(0) == in.extent(0) && out.extent(1) == in.extent(1),
               "Input and output must have the same shape.");

  detail::MatrixLinewiseOp<16, 256>::run<apply == raft::Apply::ALONG_ROWS, m_t, idx_t>(
    out.data_handle(),
    in.data_handle(),
    lineLen,
    nLines,
    op,
    resource::get_cuda_stream(handle),
    vecs.data_handle()...);
}

template <raft::Apply apply,
          typename m_t,
          typename idx_t,
          typename layout,
          typename Lambda,
          typename... vec_t,
          typename = raft::enable_if_device_mdspan<vec_t...>>
void linewise_op(raft::resources const& handle,
                 raft::device_aligned_matrix_view<const m_t, idx_t, layout> in,
                 raft::device_aligned_matrix_view<m_t, idx_t, layout> out,
                 Lambda op,
                 vec_t... vecs)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  constexpr auto is_rowmajor = std::is_same_v<layout, raft::layout_right_padded<m_t>>;
  constexpr auto is_colmajor = std::is_same_v<layout, raft::layout_left_padded<m_t>>;

  static_assert(is_rowmajor || is_colmajor,
                "layout for in and out must be either padded row or col major");

  const idx_t nLines  = is_rowmajor ? in.extent(0) : in.extent(1);
  const idx_t lineLen = is_rowmajor ? in.extent(1) : in.extent(0);

  RAFT_EXPECTS(out.extent(0) == in.extent(0) && out.extent(1) == in.extent(1),
               "Input and output must have the same shape.");

  detail::MatrixLinewiseOp<16, 256>::runPadded<apply == raft::Apply::ALONG_ROWS, m_t, idx_t>(
    out, in, lineLen, nLines, op, resource::get_cuda_stream(handle), vecs.data_handle()...);
}

/** @} */  // end of group linewise_op

}  // namespace matrix
}  // namespace raft
