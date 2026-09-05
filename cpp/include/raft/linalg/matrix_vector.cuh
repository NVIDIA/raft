/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/types.hpp>
#include <raft/matrix/detail/math.cuh>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @defgroup matrix_vector Matrix-Vector Operations
 * @{
 */

/**
 * @brief multiply each row or column of matrix with vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 */
template <Apply apply, typename math_t, typename idx_t, typename layout_t>
void binary_mult(raft::resources const& handle,
                 raft::device_matrix_view<math_t, idx_t, layout_t> data,
                 raft::device_vector_view<const math_t, idx_t> vec)
{
  constexpr auto row_major        = std::is_same_v<layout_t, raft::row_major>;
  constexpr auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryMult<row_major, bcast_along_rows>(
    data.data_handle(),
    vec.data_handle(),
    data.extent(0),
    data.extent(1),
    resource::get_cuda_stream(handle).get());
}

/**
 * @brief multiply each row or column of matrix with vector, skipping zeros in vector
 * @param [in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 */
template <Apply apply, typename math_t, typename idx_t, typename layout_t>
void binary_mult_skip_zero(raft::resources const& handle,
                           raft::device_matrix_view<math_t, idx_t, layout_t> data,
                           raft::device_vector_view<const math_t, idx_t> vec)
{
  constexpr auto row_major        = std::is_same_v<layout_t, raft::row_major>;
  constexpr auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryMultSkipZero<row_major, bcast_along_rows>(
    data.data_handle(),
    vec.data_handle(),
    data.extent(0),
    data.extent(1),
    resource::get_cuda_stream(handle).get());
}

/**
 * @brief divide each row or column of matrix with vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 */
template <Apply apply, typename math_t, typename idx_t, typename layout_t>
void binary_div(raft::resources const& handle,
                raft::device_matrix_view<math_t, idx_t, layout_t> data,
                raft::device_vector_view<const math_t, idx_t> vec)
{
  constexpr auto row_major        = std::is_same_v<layout_t, raft::row_major>;
  constexpr auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryDiv<row_major, bcast_along_rows>(
    data.data_handle(),
    vec.data_handle(),
    data.extent(0),
    data.extent(1),
    resource::get_cuda_stream(handle).get());
}

/**
 * @brief divide each row or column of matrix with vector, skipping zeros in vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 * @param[in] return_zero: result is zero if true and vector value is below threshold, original
 * value if false
 */
template <Apply apply, typename math_t, typename idx_t, typename layout_t>
void binary_div_skip_zero(raft::resources const& handle,
                          raft::device_matrix_view<math_t, idx_t, layout_t> data,
                          raft::device_vector_view<const math_t, idx_t> vec,
                          bool return_zero = false)
{
  constexpr auto row_major        = std::is_same_v<layout_t, raft::row_major>;
  constexpr auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryDivSkipZero<row_major, bcast_along_rows>(
    data.data_handle(),
    vec.data_handle(),
    data.extent(0),
    data.extent(1),
    resource::get_cuda_stream(handle).get(),
    return_zero);
}

/**
 * @brief add each row or column of matrix with vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 */
template <Apply apply, typename math_t, typename idx_t, typename layout_t>
void binary_add(raft::resources const& handle,
                raft::device_matrix_view<math_t, idx_t, layout_t> data,
                raft::device_vector_view<const math_t, idx_t> vec)
{
  constexpr auto row_major        = std::is_same_v<layout_t, raft::row_major>;
  constexpr auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinaryAdd<row_major, bcast_along_rows>(
    data.data_handle(),
    vec.data_handle(),
    data.extent(0),
    data.extent(1),
    resource::get_cuda_stream(handle).get());
}

/**
 * @brief subtract each row or column of matrix with vector
 * @param[in] handle: raft handle for managing library resources
 * @param[inout] data: input matrix, results are in-place
 * @param[in] vec: input vector
 */
template <Apply apply, typename math_t, typename idx_t, typename layout_t>
void binary_sub(raft::resources const& handle,
                raft::device_matrix_view<math_t, idx_t, layout_t> data,
                raft::device_vector_view<const math_t, idx_t> vec)
{
  constexpr auto row_major        = std::is_same_v<layout_t, raft::row_major>;
  constexpr auto bcast_along_rows = apply == Apply::ALONG_ROWS;

  idx_t vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(
    vec.extent(0) == vec_size,
    "If `bcast_along_rows==true`, vector size must equal number of columns in the matrix."
    "If `bcast_along_rows==false`, vector size must equal number of rows in the matrix.");

  matrix::detail::matrixVectorBinarySub<row_major, bcast_along_rows>(
    data.data_handle(),
    vec.data_handle(),
    data.extent(0),
    data.extent(1),
    resource::get_cuda_stream(handle).get());
}

/** @} */  // end of matrix_vector

}  // namespace linalg
}  // namespace raft
