/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __MATRIX_VECTOR_OP_H
#define __MATRIX_VECTOR_OP_H

#pragma once

#include "detail/matrix_vector_op.cuh"
#include "linalg_types.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/types.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief Operations for all the columns or rows with a given vector.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when addresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam rowMajor whether input is row or col major
 * @tparam bcastAlongRows whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns
 * @tparam MatT the matrix type
 * @tparam Lambda a device function which represents a binary operator
 * @tparam VecT the input vector type
 * @tparam IdxType Integer type used to for addressing
 * @param out the output matrix (passing out = matrix makes it in-place)
 * @param matrix the input matrix
 * @param vec the vector
 * @param D number of columns of matrix
 * @param N number of rows of matrix
 * @param op the mathematical operation
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor,
          bool bcastAlongRows,
          typename MatT,
          typename Lambda,
          typename VecT,
          typename IdxType = int>
void matrixVectorOp(MatT* out,
                    const MatT* matrix,
                    const VecT* vec,
                    IdxType D,
                    IdxType N,
                    Lambda op,
                    cudaStream_t stream)
{
  detail::matrixVectorOp<rowMajor, bcastAlongRows>(false, out, matrix, vec, D, N, op, stream);
}

/**
 * @brief Operations for all the columns or rows with the given vectors.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when addresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam rowMajor whether input is row or col major
 * @tparam bcastAlongRows whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns
 * @tparam MatT the matrix type
 * @tparam Lambda a device function which represents a binary operator
 * @tparam Vec1T the first input vector type
 * @tparam Vec2T the second input vector type
 * @tparam IdxType Integer type used to for addressing
 * @param out the output matrix (passing out = matrix makes it in-place)
 * @param matrix the input matrix
 * @param vec1 the first vector
 * @param vec2 the second vector
 * @param D number of columns of matrix
 * @param N number of rows of matrix
 * @param op the mathematical operation
 * @param stream cuda stream where to launch work
 */
template <bool rowMajor,
          bool bcastAlongRows,
          typename MatT,
          typename Lambda,
          typename Vec1T,
          typename Vec2T,
          typename IdxType = int>
void matrixVectorOp(MatT* out,
                    const MatT* matrix,
                    const Vec1T* vec1,
                    const Vec2T* vec2,
                    IdxType D,
                    IdxType N,
                    Lambda op,
                    cudaStream_t stream)
{
  detail::matrixVectorOp<rowMajor, bcastAlongRows>(
    false, out, matrix, vec1, vec2, D, N, op, stream);
}

/**
 * @defgroup matrix_vector_op Matrix Vector Operations
 * @{
 */

/**
 * @brief Operations for all the columns or rows with a given vector.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when addresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::Apply
 * @tparam MatValueType the data-type of the input matrix
 * @tparam VecValueType the data-type of the input vector
 * @tparam LayoutPolicy the layout of input and output (raft::row_major or raft::col_major)
 * @tparam Lambda a device function which represents a binary operator
 * @tparam IndexType Integer used for addressing
 * @param[in] handle raft::resources
 * @param[in] matrix input raft::matrix_view
 * @param[in] vec vector raft::vector_view
 * @param[out] out output raft::matrix_view
 * @param[in] op the mathematical operation
 */
template <Apply apply,
          typename MatValueType,
          typename VecValueType,
          typename LayoutPolicy,
          typename Lambda,
          typename IndexType>
void matrix_vector_op(raft::resources const& handle,
                      raft::device_matrix_view<const MatValueType, IndexType, LayoutPolicy> matrix,
                      raft::device_vector_view<const VecValueType, IndexType> vec,
                      raft::device_matrix_view<MatValueType, IndexType, LayoutPolicy> out,
                      Lambda op)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(matrix), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == matrix.size(), "Size mismatch between Output and Input");

  auto constexpr rowMajor = std::is_same_v<typename decltype(out)::layout_type, raft::row_major>;
  auto constexpr bcastAlongRows = apply == Apply::ALONG_ROWS;

  if constexpr (bcastAlongRows) {
    RAFT_EXPECTS(out.extent(1) == static_cast<IndexType>(vec.size()),
                 "Size mismatch between matrix and vector");
  } else {
    RAFT_EXPECTS(out.extent(0) == static_cast<IndexType>(vec.size()),
                 "Size mismatch between matrix and vector");
  }

  detail::matrixVectorOp<rowMajor, bcastAlongRows>(resource::get_dry_run_flag(handle),
                                                   out.data_handle(),
                                                   matrix.data_handle(),
                                                   vec.data_handle(),
                                                   out.extent(1),
                                                   out.extent(0),
                                                   op,
                                                   resource::get_cuda_stream(handle));
}

/**
 * @brief Operations for all the columns or rows with the given vectors.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when addresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::Apply
 * @tparam MatValueType the data-type of the input and output matrices
 * @tparam Vec1ValueType the data-type of the first input vector
 * @tparam Vec2ValueType the data-type of the second input vector
 * @tparam LayoutPolicy the layout of input and output (raft::row_major or raft::col_major)
 * @tparam Lambda a device function which represents a binary operator
 * @tparam IndexType Integer used for addressing
 * @param handle raft::resources
 * @param matrix input raft::matrix_view
 * @param vec1 the first vector raft::vector_view
 * @param vec2 the second vector raft::vector_view
 * @param out output raft::matrix_view
 * @param op the mathematical operation
 */
template <Apply apply,
          typename MatValueType,
          typename Vec1ValueType,
          typename Vec2ValueType,
          typename LayoutPolicy,
          typename Lambda,
          typename IndexType>
void matrix_vector_op(raft::resources const& handle,
                      raft::device_matrix_view<const MatValueType, IndexType, LayoutPolicy> matrix,
                      raft::device_vector_view<const Vec1ValueType, IndexType> vec1,
                      raft::device_vector_view<const Vec2ValueType, IndexType> vec2,
                      raft::device_matrix_view<MatValueType, IndexType, LayoutPolicy> out,
                      Lambda op)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(matrix), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == matrix.size(), "Size mismatch between Output and Input");

  auto constexpr rowMajor = std::is_same_v<typename decltype(out)::layout_type, raft::row_major>;
  auto constexpr bcastAlongRows = apply == Apply::ALONG_ROWS;

  if constexpr (bcastAlongRows) {
    RAFT_EXPECTS(out.extent(1) == static_cast<IndexType>(vec1.size()),
                 "Size mismatch between matrix and vector");
    RAFT_EXPECTS(out.extent(1) == static_cast<IndexType>(vec2.size()),
                 "Size mismatch between matrix and vector");
  } else {
    RAFT_EXPECTS(out.extent(0) == static_cast<IndexType>(vec1.size()),
                 "Size mismatch between matrix and vector");
    RAFT_EXPECTS(out.extent(0) == static_cast<IndexType>(vec2.size()),
                 "Size mismatch between matrix and vector");
  }

  detail::matrixVectorOp<rowMajor, bcastAlongRows>(resource::get_dry_run_flag(handle),
                                                   out.data_handle(),
                                                   matrix.data_handle(),
                                                   vec1.data_handle(),
                                                   vec2.data_handle(),
                                                   out.extent(1),
                                                   out.extent(0),
                                                   op,
                                                   resource::get_cuda_stream(handle));
}

/** @} */  // end of group matrix_vector_op

};  // end namespace linalg
};  // end namespace raft

#endif
