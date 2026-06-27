/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map.cuh>
#include <raft/util/fast_int_div.cuh>

#include <cuda/iterator>
#include <thrust/for_each.h>

namespace raft {
namespace matrix {
namespace detail {

template <typename MatrixT, typename MapT, typename MapTransformOp, typename IndexT>
void gatherInplaceImpl(raft::resources const& handle,
                       raft::device_matrix_view<MatrixT, IndexT, raft::layout_stride> inout,
                       raft::device_vector_view<const MapT, IndexT, raft::layout_c_contiguous> map,
                       MapTransformOp transform_op,
                       IndexT batch_size)
{
  IndexT m          = inout.extent(0);
  IndexT n          = inout.extent(1);
  IndexT map_length = map.extent(0);
  IndexT ld         = inout.stride(0);

  // skip in case of 0 length input
  if (map_length <= 0 || m <= 0 || n <= 0 || batch_size < 0) return;

  RAFT_EXPECTS(map_length <= m, "Length of map should be <= number of rows for inplace gather");

  RAFT_EXPECTS(batch_size >= 0, "batch size should be >= 0");

  // re-assign batch_size for default case
  if (batch_size == 0 || batch_size > n) batch_size = n;

  auto exec_policy = resource::get_thrust_policy(handle);

  IndexT n_batches = raft::ceildiv(n, batch_size);

  auto scratch_space = raft::make_device_vector<MatrixT, IndexT>(handle, map_length * batch_size);

  for (IndexT bid = 0; bid < n_batches; bid++) {
    IndexT batch_offset   = bid * batch_size;
    IndexT cols_per_batch = min(batch_size, n - batch_offset);

    auto gather_op = [inout = inout.data_handle(),
                      map   = map.data_handle(),
                      transform_op,
                      batch_offset,
                      map_length,
                      cols_per_batch = raft::util::FastIntDiv<IndexT>(cols_per_batch),
                      n,
                      ld] __device__(auto idx) {
      IndexT row   = idx / cols_per_batch;
      IndexT col   = idx % cols_per_batch;
      MapT map_val = map[row];

      IndexT i_src = transform_op(map_val);
      return inout[i_src * ld + batch_offset + col];
    };
    raft::linalg::map_offset(
      handle,
      raft::make_device_vector_view(scratch_space.data_handle(), map_length * cols_per_batch),
      gather_op);

    auto copy_op = [inout         = inout.data_handle(),
                    map           = map.data_handle(),
                    scratch_space = scratch_space.data_handle(),
                    batch_offset,
                    map_length,
                    cols_per_batch = raft::util::FastIntDiv<IndexT>(cols_per_batch),
                    n,
                    ld] __device__(auto idx) {
      IndexT row                           = idx / cols_per_batch;
      IndexT col                           = idx % cols_per_batch;
      inout[row * ld + batch_offset + col] = scratch_space[idx];
      return;
    };
    auto counting = cuda::make_counting_iterator<IndexT>(0);
    thrust::for_each(exec_policy, counting, counting + map_length * cols_per_batch, copy_op);
  }
}

template <typename MatrixT, typename MapT, typename MapTransformOp, typename IndexT>
void gather(raft::resources const& handle,
            raft::device_matrix_view<MatrixT, IndexT, raft::layout_stride> inout,
            raft::device_vector_view<const MapT, IndexT, raft::layout_c_contiguous> map,
            MapTransformOp transform_op,
            IndexT batch_size)
{
  gatherInplaceImpl(handle, inout, map, transform_op, batch_size);
}

template <typename MatrixT, typename MapT, typename MapTransformOp, typename IndexT>
void gather(raft::resources const& handle,
            raft::device_matrix_view<MatrixT, IndexT, raft::layout_c_contiguous> inout,
            raft::device_vector_view<const MapT, IndexT, raft::layout_c_contiguous> map,
            MapTransformOp transform_op,
            IndexT batch_size)
{
  auto inout_ = raft::make_device_strided_matrix_view<MatrixT, IndexT>(
    inout.data_handle(), inout.extent(0), inout.extent(1), inout.extent(1));
  gatherInplaceImpl(handle, inout_, map, transform_op, batch_size);
}

template <typename MatrixT, typename MapT, typename IndexT>
void gather(raft::resources const& handle,
            raft::device_matrix_view<MatrixT, IndexT, raft::layout_stride> inout,
            raft::device_vector_view<const MapT, IndexT, raft::layout_c_contiguous> map,
            IndexT batch_size)
{
  gatherInplaceImpl(handle, inout, map, raft::identity_op(), batch_size);
}

template <typename MatrixT, typename MapT, typename IndexT>
void gather(raft::resources const& handle,
            raft::device_matrix_view<MatrixT, IndexT, raft::layout_c_contiguous> inout,
            raft::device_vector_view<const MapT, IndexT, raft::layout_c_contiguous> map,
            IndexT batch_size)
{
  auto inout_ = raft::make_device_strided_matrix_view<MatrixT, IndexT>(
    inout.data_handle(), inout.extent(0), inout.extent(1), inout.extent(1));
  gatherInplaceImpl(handle, inout_, map, raft::identity_op(), batch_size);
}

}  // namespace detail
}  // namespace matrix
}  // namespace raft
