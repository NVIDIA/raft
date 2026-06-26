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
#include <raft/matrix/detail/gather.cuh>
#include <raft/matrix/detail/gather_inplace.cuh>
#include <raft/util/itertools.hpp>

namespace raft {
namespace matrix {

/**
 * @defgroup matrix_gather Matrix gather operations
 * @{
 */

/**
 * @brief Copies rows from a source matrix into a destination matrix according to a map.
 *
 * For each output row, read the index in the input matrix from the map and copy the row.
 *
 * @tparam InputIteratorT  Input iterator type, for the input matrix (may be a pointer type).
 * @tparam MapIteratorT    Input iterator type, for the map (may be a pointer type).
 * @tparam OutputIteratorT Output iterator type, for the output matrix (may be a pointer type).
 * @tparam IndexT          Index type.
 *
 * @param  in           Input matrix, dim = [N, D] (row-major)
 * @param  D            Number of columns of the input/output matrices
 * @param  N            Number of rows of the input matrix
 * @param  map          Map of row indices to gather, dim = [map_length]
 * @param  map_length   The length of 'map', number of rows of the output matrix
 * @param  out          Output matrix, dim = [map_length, D] (row-major)
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename InputIteratorT, typename MapIteratorT, typename OutputIteratorT, typename IndexT>
void gather(const InputIteratorT in,
            IndexT D,
            IndexT N,
            const MapIteratorT map,
            IndexT map_length,
            OutputIteratorT out,
            cudaStream_t stream)
{
  detail::gather(in, D, N, map, map_length, out, stream);
}

/**
 * @brief Copies rows from a source matrix into a destination matrix according to a transformed map.
 *
 * For each output row, read the index in the input matrix from the map, apply a transformation to
 * this input index and copy the row.
 *
 * @tparam InputIteratorT  Input iterator type, for the input matrix (may be a pointer type).
 * @tparam MapIteratorT    Input iterator type, for the map (may be a pointer type).
 * @tparam MapTransformOp  Unary lambda expression or operator type. MapTransformOp's result type
 *                         must be convertible to IndexT.
 * @tparam OutputIteratorT Output iterator type, for the output matrix (may be a pointer type).
 * @tparam IndexT          Index type.
 *
 * @param  in           Input matrix, dim = [N, D] (row-major)
 * @param  D            Number of columns of the input/output matrices
 * @param  N            Number of rows of the input matrix
 * @param  map          Map of row indices to gather, dim = [map_length]
 * @param  map_length   The length of 'map', number of rows of the output matrix
 * @param  out          Output matrix, dim = [map_length, D] (row-major)
 * @param  transform_op Transformation to apply to map values
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename InputIteratorT,
          typename MapIteratorT,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
void gather(const InputIteratorT in,
            IndexT D,
            IndexT N,
            const MapIteratorT map,
            IndexT map_length,
            OutputIteratorT out,
            MapTransformOp transform_op,
            cudaStream_t stream)
{
  detail::gather(in, D, N, map, map_length, out, transform_op, stream);
}

/**
 * @brief Conditionally copies rows from a source matrix into a destination matrix.
 *
 * For each output row, read the index in the input matrix from the map, read a stencil value, apply
 * a predicate to the stencil value, and if true, copy the row.
 *
 * @tparam InputIteratorT   Input iterator type, for the input matrix (may be a pointer type).
 * @tparam MapIteratorT     Input iterator type, for the map (may be a pointer type).
 * @tparam StencilIteratorT Input iterator type, for the stencil (may be a pointer type).
 * @tparam UnaryPredicateOp Unary lambda expression or operator type. UnaryPredicateOp's result type
 *                          must be convertible to bool type.
 * @tparam OutputIteratorT  Output iterator type, for the output matrix (may be a pointer type).
 * @tparam IndexT           Index type.
 *
 * @param  in           Input matrix, dim = [N, D] (row-major)
 * @param  D            Number of columns of the input/output matrices
 * @param  N            Number of rows of the input matrix
 * @param  map          Map of row indices to gather, dim = [map_length]
 * @param  stencil      Sequence of stencil values, dim = [map_length]
 * @param  map_length   The length of 'map' and 'stencil', number of rows of the output matrix
 * @param  out          Output matrix, dim = [map_length, D] (row-major)
 * @param  pred_op      Predicate to apply to the stencil values
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename UnaryPredicateOp,
          typename OutputIteratorT,
          typename IndexT>
void gather_if(const InputIteratorT in,
               IndexT D,
               IndexT N,
               const MapIteratorT map,
               StencilIteratorT stencil,
               IndexT map_length,
               OutputIteratorT out,
               UnaryPredicateOp pred_op,
               cudaStream_t stream)
{
  detail::gather_if(in, D, N, map, stencil, map_length, out, pred_op, stream);
}

/**
 * @brief Conditionally copies rows according to a transformed map.
 *
 * For each output row, read the index in the input matrix from the map, read a stencil value,
 * apply a predicate to the stencil value, and if true, apply a transformation to the input index
 * and copy the row.
 *
 * @tparam InputIteratorT   Input iterator type, for the input matrix (may be a pointer type).
 * @tparam MapIteratorT     Input iterator type, for the map (may be a pointer type).
 * @tparam MapTransformOp   Unary lambda expression or operator type. MapTransformOp's result type
 *                          must be convertible to IndexT.
 * @tparam StencilIteratorT Input iterator type, for the stencil (may be a pointer type).
 * @tparam UnaryPredicateOp Unary lambda expression or operator type. UnaryPredicateOp's result type
 *                          must be convertible to bool type.
 * @tparam OutputIteratorT  Output iterator type, for the output matrix (may be a pointer type).
 * @tparam IndexT           Index type.
 *
 * @param  in           Input matrix, dim = [N, D] (row-major)
 * @param  D            Number of columns of the input/output matrices
 * @param  N            Number of rows of the input matrix
 * @param  map          Map of row indices to gather, dim = [map_length]
 * @param  stencil      Sequence of stencil values, dim = [map_length]
 * @param  map_length   The length of 'map' and 'stencil', number of rows of the output matrix
 * @param  out          Output matrix, dim = [map_length, D] (row-major)
 * @param  pred_op      Predicate to apply to the stencil values
 * @param  transform_op Transformation to apply to map values
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename UnaryPredicateOp,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
void gather_if(const InputIteratorT in,
               IndexT D,
               IndexT N,
               const MapIteratorT map,
               StencilIteratorT stencil,
               IndexT map_length,
               OutputIteratorT out,
               UnaryPredicateOp pred_op,
               MapTransformOp transform_op,
               cudaStream_t stream)
{
  detail::gather_if(in, D, N, map, stencil, map_length, out, pred_op, transform_op, stream);
}

/**
 * @brief Copies rows from a source matrix into a destination matrix according to a transformed map.
 *
 * For each output row, read the index in the input matrix from the map, apply a transformation to
 * this input index if specified, and copy the row.
 *
 * @tparam matrix_t    Matrix element type
 * @tparam map_t       Integer type of map elements
 * @tparam idx_t       Integer type used for indexing
 * @tparam map_xform_t Unary lambda expression or operator type. MapTransformOp's result type must
 *                     be convertible to idx_t.
 * @param[in]  handle        raft handle for managing resources
 * @param[in]  in            Input matrix, dim = [N, D] (row-major)
 * @param[in]  map           Map of row indices to gather, dim = [map_length]
 * @param[out] out           Output matrix, dim = [map_length, D] (row-major)
 * @param[in]  transform_op  (optional) Transformation to apply to map values
 */
template <typename matrix_t,
          typename map_t,
          typename idx_t,
          typename map_xform_t = raft::identity_op>
void gather(const raft::resources& handle,
            raft::device_matrix_view<const matrix_t, idx_t, row_major> in,
            raft::device_vector_view<const map_t, idx_t> map,
            raft::device_matrix_view<matrix_t, idx_t, row_major> out,
            map_xform_t transform_op = raft::identity_op())
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(out.extent(0) == map.extent(0),
               "Number of rows in output matrix must equal the size of the map vector");
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Number of columns in input and output matrices must be equal.");

  detail::gather(
    const_cast<matrix_t*>(in.data_handle()),  // TODO: There's a better way to handle this
    in.extent(1),
    in.extent(0),
    map.data_handle(),
    map.extent(0),
    out.data_handle(),
    transform_op,
    resource::get_cuda_stream(handle));
}

/**
 * @brief Copies rows from a source matrix into a destination matrix according to a transformed map.
 *
 * For each output row, read the index in the input matrix from the map, apply a transformation to
 * this input index if specified, and copy the row.
 *
 * @tparam matrix_t    Matrix element type
 * @tparam map_t       Integer type of map elements
 * @tparam idx_t       Integer type used for indexing
 * @tparam map_xform_t Unary lambda expression or operator type. MapTransformOp's result type must
 *                     be convertible to idx_t.
 * @param[in]  handle        raft handle for managing resources
 * @param[in]  in            Input matrix, dim = [N, D] (row-major)
 * @param[in]  map           Map of row indices to gather, dim = [map_length]
 * @param[out] out           Output matrix, dim = [map_length, D] (row-major)
 * @param[in]  transform_op  (optional) Transformation to apply to map values
 */
template <typename matrix_t,
          typename map_t,
          typename idx_t,
          typename accessor,
          typename map_xform_t = raft::identity_op>
void gather(
  const raft::resources& handle,
  raft::mdspan<const matrix_t, raft::matrix_extent<idx_t>, raft::layout_stride, accessor> in,
  raft::device_vector_view<const map_t, idx_t> map,
  raft::device_matrix_view<matrix_t, idx_t, row_major> out,
  map_xform_t transform_op = raft::identity_op())
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(out.extent(0) == map.extent(0),
               "Number of rows in output matrix must equal the size of the map vector");
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Number of columns in input and output matrices must be equal.");

  detail::gather(
    const_cast<matrix_t*>(in.data_handle()),  // TODO: There's a better way to handle this
    in.stride(0),
    in.extent(1),
    in.extent(0),
    map.data_handle(),
    map.extent(0),
    out.data_handle(),
    transform_op,
    resource::get_cuda_stream(handle));
}

/**
 * @brief Conditionally copies rows according to a transformed map.
 *
 * For each output row, read the index in the input matrix from the map, read a stencil value,
 * apply a predicate to the stencil value, and if true, apply a transformation if specified to the
 * input index, and copy the row.
 *
 * @tparam matrix_t     Matrix element type
 * @tparam map_t        Integer type of map elements
 * @tparam stencil_t    Value type for stencil (input type for the pred_op)
 * @tparam unary_pred_t Unary lambda expression or operator type. unary_pred_t's result
 *                      type must be convertible to bool type.
 * @tparam map_xform_t  Unary lambda expression or operator type. MapTransformOp's result type must
 *                      be convertible to idx_t.
 * @tparam idx_t        Integer type used for indexing
 * @param[in]  handle        raft handle for managing resources
 * @param[in]  in            Input matrix, dim = [N, D] (row-major)
 * @param[in]  map           Map of row indices to gather, dim = [map_length]
 * @param[in]  stencil       Vector of stencil values, dim = [map_length]
 * @param[out] out           Output matrix, dim = [map_length, D] (row-major)
 * @param[in]  pred_op       Predicate to apply to the stencil values
 * @param[in]  transform_op  (optional) Transformation to apply to map values
 */
template <typename matrix_t,
          typename map_t,
          typename stencil_t,
          typename unary_pred_t,
          typename idx_t,
          typename map_xform_t = raft::identity_op>
void gather_if(const raft::resources& handle,
               raft::device_matrix_view<const matrix_t, idx_t, layout_stride> in,
               raft::device_matrix_view<matrix_t, idx_t, row_major> out,
               raft::device_vector_view<const map_t, idx_t> map,
               raft::device_vector_view<const stencil_t, idx_t> stencil,
               unary_pred_t pred_op,
               map_xform_t transform_op = raft::identity_op())
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(out.extent(0) == map.extent(0),
               "Number of rows in output matrix must equal the size of the map vector");
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Number of columns in input and output matrices must be equal.");
  RAFT_EXPECTS(map.extent(0) == stencil.extent(0),
               "Number of elements in stencil must equal number of elements in map");

  detail::gather_if(const_cast<matrix_t*>(in.data_handle()),
                    in.stride(0),
                    in.extent(1),
                    in.extent(0),
                    map.data_handle(),
                    stencil.data_handle(),
                    map.extent(0),
                    out.data_handle(),
                    pred_op,
                    transform_op,
                    resource::get_cuda_stream(handle));
}

/**
 * @brief Conditionally copies rows according to a transformed map.
 *
 * For each output row, read the index in the input matrix from the map, read a stencil value,
 * apply a predicate to the stencil value, and if true, apply a transformation if specified to the
 * input index, and copy the row.
 *
 * @tparam matrix_t     Matrix element type
 * @tparam map_t        Integer type of map elements
 * @tparam stencil_t    Value type for stencil (input type for the pred_op)
 * @tparam unary_pred_t Unary lambda expression or operator type. unary_pred_t's result
 *                      type must be convertible to bool type.
 * @tparam map_xform_t  Unary lambda expression or operator type. MapTransformOp's result type must
 *                      be convertible to idx_t.
 * @tparam idx_t        Integer type used for indexing
 * @param[in]  handle        raft handle for managing resources
 * @param[in]  in            Input matrix, dim = [N, D] (row-major)
 * @param[in]  map           Map of row indices to gather, dim = [map_length]
 * @param[in]  stencil       Vector of stencil values, dim = [map_length]
 * @param[out] out           Output matrix, dim = [map_length, D] (row-major)
 * @param[in]  pred_op       Predicate to apply to the stencil values
 * @param[in]  transform_op  (optional) Transformation to apply to map values
 */
template <typename matrix_t,
          typename map_t,
          typename stencil_t,
          typename unary_pred_t,
          typename idx_t,
          typename map_xform_t = raft::identity_op>
void gather_if(const raft::resources& handle,
               raft::device_matrix_view<const matrix_t, idx_t, row_major> in,
               raft::device_matrix_view<matrix_t, idx_t, row_major> out,
               raft::device_vector_view<const map_t, idx_t> map,
               raft::device_vector_view<const stencil_t, idx_t> stencil,
               unary_pred_t pred_op,
               map_xform_t transform_op = raft::identity_op())
{
  if (resource::get_dry_run_flag(handle)) { return; }
  RAFT_EXPECTS(out.extent(0) == map.extent(0),
               "Number of rows in output matrix must equal the size of the map vector");
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Number of columns in input and output matrices must be equal.");
  RAFT_EXPECTS(map.extent(0) == stencil.extent(0),
               "Number of elements in stencil must equal number of elements in map");

  detail::gather_if(const_cast<matrix_t*>(in.data_handle()),
                    in.extent(1),
                    in.extent(0),
                    map.data_handle(),
                    stencil.data_handle(),
                    map.extent(0),
                    out.data_handle(),
                    pred_op,
                    transform_op,
                    resource::get_cuda_stream(handle));
}

/**
 * @brief In-place gather elements in a row-major matrix according to a
 * map. The map specifies the new order in which rows of the input matrix are
 * rearranged, i.e. for each output row, read the index in the input matrix
 * from the map, apply a transformation to this input index if specified, and copy the row.
 * map[i]. For example, the matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]] with the
 * map [2, 0, 1] will be transformed to [[7, 8, 9], [1, 2, 3], [4, 5, 6]].
 * Batching is done on columns and an additional scratch space of
 * shape n_rows * cols_batch_size is created. For each batch, chunks
 * of columns from each row are copied into the appropriate location
 * in the scratch space and copied back to the corresponding locations
 * in the input matrix.
 *
 * @tparam matrix_t     Matrix element type
 * @tparam map_t        Integer type of map elements
 * @tparam map_xform_t  Unary lambda expression or operator type. MapTransformOp's result type must
 *                      be convertible to idx_t.
 * @tparam idx_t        Integer type used for indexing
 *
 * @param[in] handle raft handle
 * @param[inout] inout input matrix (n_rows * n_cols)
 * @param[in] map Pointer to the input sequence of gather locations
 * @param[in] col_batch_size (optional) column batch size. Determines the shape of the scratch space
 * (map_length, col_batch_size). When set to zero (default), no batching is done and an additional
 * scratch space of shape (map_lengthm, n_cols) is created.
 * @param[in]  transform_op  (optional) Transformation to apply to map values
 */
template <typename matrix_t,
          typename map_t,
          typename idx_t,
          typename map_xform_t = raft::identity_op>
void gather(raft::resources const& handle,
            raft::device_matrix_view<matrix_t, idx_t, raft::layout_c_contiguous> inout,
            raft::device_vector_view<const map_t, idx_t, raft::layout_c_contiguous> map,
            idx_t col_batch_size     = 0,
            map_xform_t transform_op = raft::identity_op())
{
  detail::gather(handle, inout, map, transform_op, col_batch_size);
}

/**
 * @brief In-place gather elements in a row-major matrix according to a
 * map. The map specifies the new order in which rows of the input matrix are
 * rearranged, i.e. for each output row, read the index in the input matrix
 * from the map, apply a transformation to this input index if specified, and copy the row.
 * map[i]. For example, the matrix [[1, 2, 3], [4, 5, 6], [7, 8, 9]] with the
 * map [2, 0, 1] will be transformed to [[7, 8, 9], [1, 2, 3], [4, 5, 6]].
 * Batching is done on columns and an additional scratch space of
 * shape n_rows * cols_batch_size is created. For each batch, chunks
 * of columns from each row are copied into the appropriate location
 * in the scratch space and copied back to the corresponding locations
 * in the input matrix.
 *
 * @tparam matrix_t     Matrix element type
 * @tparam map_t        Integer type of map elements
 * @tparam map_xform_t  Unary lambda expression or operator type. MapTransformOp's result type must
 *                      be convertible to idx_t.
 * @tparam idx_t        Integer type used for indexing
 *
 * @param[in] handle raft handle
 * @param[inout] inout input matrix (n_rows * n_cols)
 * @param[in] map Pointer to the input sequence of gather locations
 * @param[in] col_batch_size (optional) column batch size. Determines the shape of the scratch space
 * (map_length, col_batch_size). When set to zero (default), no batching is done and an additional
 * scratch space of shape (map_lengthm, n_cols) is created.
 * @param[in]  transform_op  (optional) Transformation to apply to map values
 */
template <typename matrix_t,
          typename map_t,
          typename idx_t,
          typename map_xform_t = raft::identity_op>
void gather(raft::resources const& handle,
            raft::device_matrix_view<matrix_t, idx_t, raft::layout_stride> inout,
            raft::device_vector_view<const map_t, idx_t, raft::layout_c_contiguous> map,
            idx_t col_batch_size     = 0,
            map_xform_t transform_op = raft::identity_op())
{
  detail::gather(handle, inout, map, transform_op, col_batch_size);
}

/** @} */  // end of group matrix_gather

}  // namespace matrix
}  // namespace raft
