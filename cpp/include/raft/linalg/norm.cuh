/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __NORM_H
#define __NORM_H

#pragma once

#include "detail/norm.cuh"
#include "linalg_types.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/types.hpp>
#include <raft/linalg/norm_types.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief Compute row-wise norm of the input matrix and perform fin_op lambda
 *
 * Row-wise norm is useful while computing pairwise distance matrix, for
 * example.
 * This is used in many clustering algos like knn, kmeans, dbscan, etc...
 *
 * @tparam norm_type the type of norm to be applied
 * @tparam rowMajor whether the input is row-major or not
 * @tparam Type the data type
 * @tparam Lambda device final lambda
 * @tparam IdxType Integer type used to for addressing
 * @tparam OutType output type, equal to Type by default
 * @param dots the output vector of row-wise dot products
 * @param data the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream where to launch work
 * @param fin_op the final lambda op
 */
template <NormType norm_type,
          bool rowMajor,
          typename Type,
          typename IdxType = int,
          typename Lambda  = raft::identity_op,
          typename OutType = Type>
void rowNorm(OutType* dots,
             const Type* data,
             IdxType D,
             IdxType N,
             cudaStream_t stream,
             Lambda fin_op = raft::identity_op())
{
  detail::rowNormCaller<norm_type, rowMajor>(false, dots, data, D, N, stream, fin_op);
}

/**
 * @brief Compute column-wise norm of the input matrix and perform fin_op
 * @tparam norm_type the type of norm to be applied
 * @tparam rowMajor whether the input is row-major or not
 * @tparam Type the data type
 * @tparam Lambda device final lambda
 * @tparam IdxType Integer type used to for addressing
 * @tparam OutType output type, equal to Type by default
 * @param dots the output vector of column-wise dot products
 * @param data the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream where to launch work
 * @param fin_op the final lambda op
 */
template <NormType norm_type,
          bool rowMajor,
          typename Type,
          typename IdxType = int,
          typename Lambda  = raft::identity_op,
          typename OutType = Type>
void colNorm(OutType* dots,
             const Type* data,
             IdxType D,
             IdxType N,
             cudaStream_t stream,
             Lambda fin_op = raft::identity_op())
{
  detail::colNormCaller<norm_type, rowMajor>(false, dots, data, D, N, stream, fin_op);
}

/**
 * @defgroup norm Row- or Col-norm computation
 * @{
 */

/**
 * @brief Compute norm of the input matrix and perform fin_op
 * @tparam norm_type the type of norm to be applied
 * @tparam apply Whether to apply the norm along rows (raft::Apply::ALONG_ROWS)
 *              or along columns (raft::Apply::ALONG_COLUMNS)
 * @tparam ElementType Input data type
 * @tparam OutType output data type
 * @tparam LayoutPolicy the layout of input (raft::row_major or raft::col_major)
 * @tparam IdxType Integer type used to for addressing
 * @tparam Lambda device final lambda
 * @param[in] handle raft::resources
 * @param[in] in the input raft::device_matrix_view
 * @param[out] out the output raft::device_vector_view
 * @param[in] fin_op the final lambda op
 */
template <NormType norm_type,
          raft::Apply apply,
          typename ElementType,
          typename OutputType,
          typename LayoutPolicy,
          typename IndexType,
          typename Lambda = raft::identity_op>
void norm(raft::resources const& handle,
          raft::device_matrix_view<const ElementType, IndexType, LayoutPolicy> in,
          raft::device_vector_view<OutputType, IndexType> out,
          Lambda fin_op = raft::identity_op())
{
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");

  auto constexpr row_major  = std::is_same_v<LayoutPolicy, raft::row_major>;
  auto constexpr along_rows = apply == raft::Apply::ALONG_ROWS;

  if constexpr (along_rows) {
    RAFT_EXPECTS(static_cast<IndexType>(out.size()) == in.extent(0),
                 "Output should be equal to number of rows in Input");
    detail::rowNormCaller<norm_type, row_major>(resource::get_dry_run_flag(handle),
                                                out.data_handle(),
                                                in.data_handle(),
                                                in.extent(1),
                                                in.extent(0),
                                                resource::get_cuda_stream(handle),
                                                fin_op);
  } else {
    RAFT_EXPECTS(static_cast<IndexType>(out.size()) == in.extent(1),
                 "Output should be equal to number of columns in Input");
    detail::colNormCaller<norm_type, row_major>(resource::get_dry_run_flag(handle),
                                                out.data_handle(),
                                                in.data_handle(),
                                                in.extent(1),
                                                in.extent(0),
                                                resource::get_cuda_stream(handle),
                                                fin_op);
  }
}

/** @} */

};  // end namespace linalg
};  // end namespace raft

#endif
