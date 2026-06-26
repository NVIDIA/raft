/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/pinned_mdspan.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <omp.h>

#include <functional>

namespace raft {
namespace matrix {
namespace detail {

/** Tiling policy for the gather kernel.
 *
 * The output matrix is considered as a flattened array, an approach that provides much better
 * performance than 1 row per block when D is small. Additionally, each thread works on multiple
 * output elements using an unrolled loop (approx. 30% faster than working on a single element)
 */
template <int tpb, int wpt>
struct gather_policy {
  static constexpr int n_threads       = tpb;
  static constexpr int work_per_thread = wpt;
  static constexpr int stride          = tpb * wpt;
};

/** Conditionally copies rows from the source matrix 'in' into the destination matrix
 * 'out' according to a map (or a transformed map) */
template <typename Policy,
          typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename PredicateOp,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
RAFT_KERNEL gather_kernel(const InputIteratorT in,
                          IndexT ld,
                          IndexT D,
                          IndexT len,
                          const MapIteratorT map,
                          StencilIteratorT stencil,
                          OutputIteratorT out,
                          PredicateOp pred_op,
                          MapTransformOp transform_op)
{
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  typedef typename std::iterator_traits<StencilIteratorT>::value_type StencilValueT;

#pragma unroll
  for (IndexT wid = 0; wid < Policy::work_per_thread; wid++) {
    IndexT tid = threadIdx.x + (Policy::work_per_thread * static_cast<IndexT>(blockIdx.x) + wid) *
                                 Policy::n_threads;
    if (tid < len) {
      IndexT i_dst = tid / D;
      IndexT j     = tid % D;

      MapValueT map_val         = map[i_dst];
      StencilValueT stencil_val = stencil[i_dst];

      bool predicate = pred_op(stencil_val);
      if (predicate) {
        IndexT i_src = transform_op(map_val);
        out[tid]     = in[i_src * ld + j];
      }
    }
  }
}

/**
 * @brief  gather conditionally copies rows from a source matrix into a destination matrix according
 * to a transformed map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  stencil      Pointer to the input sequence of stencil or predicate values
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  pred_op      Predicate to apply to the stencil values
 * @param  transform_op The transformation operation, transforms the map values to IndexT
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename UnaryPredicateOp,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
void gatherImpl(const InputIteratorT in,
                IndexT ld,
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
  // skip in case of 0 length input
  if (map_length <= 0 || N <= 0 || D <= 0) return;

  // map value type
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;

  // stencil value type
  typedef typename std::iterator_traits<StencilIteratorT>::value_type StencilValueT;

  IndexT len        = map_length * D;
  constexpr int TPB = 128;
  const int n_sm    = raft::getMultiProcessorCount();
  // The following empirical heuristics enforce that we keep a good balance between having enough
  // blocks and enough work per thread.
  if (len < static_cast<IndexT>(32 * TPB * n_sm)) {
    using Policy    = gather_policy<TPB, 1>;
    IndexT n_blocks = raft::ceildiv(map_length * D, static_cast<IndexT>(Policy::stride));
    gather_kernel<Policy><<<n_blocks, Policy::n_threads, 0, stream>>>(
      in, ld, D, len, map, stencil, out, pred_op, transform_op);
  } else if (len < static_cast<IndexT>(32 * 4 * TPB * n_sm)) {
    using Policy    = gather_policy<TPB, 4>;
    IndexT n_blocks = raft::ceildiv(map_length * D, static_cast<IndexT>(Policy::stride));
    gather_kernel<Policy><<<n_blocks, Policy::n_threads, 0, stream>>>(
      in, ld, D, len, map, stencil, out, pred_op, transform_op);
  } else {
    using Policy    = gather_policy<TPB, 8>;
    IndexT n_blocks = raft::ceildiv(map_length * D, static_cast<IndexT>(Policy::stride));
    gather_kernel<Policy><<<n_blocks, Policy::n_threads, 0, stream>>>(
      in, ld, D, len, map, stencil, out, pred_op, transform_op);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  ld           Leading dimension of the input matrix 'in'.
 * @param  D            Dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename InputIteratorT, typename MapIteratorT, typename OutputIteratorT, typename IndexT>
void gather(const InputIteratorT in,
            IndexT ld,
            IndexT D,
            IndexT N,
            const MapIteratorT map,
            IndexT map_length,
            OutputIteratorT out,
            cudaStream_t stream)
{
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(
    in, ld, D, N, map, map, map_length, out, raft::const_op(true), raft::identity_op(), stream);
}

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
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
  gather(in, D, D, N, map, map_length, out, stream);
}

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a
 * transformed map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  ld           Leading dimension of the input matrix 'in'.
 * @param  D            Dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  transform_op The transformation operation, transforms the map values to IndexT
 * @param  stream       CUDA stream to launch kernels within
 */
template <typename InputIteratorT,
          typename MapIteratorT,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IndexT>
void gather(const InputIteratorT in,
            IndexT ld,
            IndexT D,
            IndexT N,
            const MapIteratorT map,
            IndexT map_length,
            OutputIteratorT out,
            MapTransformOp transform_op,
            cudaStream_t stream)
{
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(in, ld, D, N, map, map, map_length, out, raft::const_op(true), transform_op, stream);
}

/**
 * @brief  gather copies rows from a source matrix into a destination matrix according to a
 * transformed map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  transform_op The transformation operation, transforms the map values to IndexT
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
  gather(in, D, D, N, map, map_length, out, transform_op, stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  ld           Leading dimension of the input matrix 'in', which in-case of row-major
 * @param  D            Dimension of the input matrix 'in'
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  stencil      Pointer to the input sequence of stencil or predicate values
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
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
               IndexT ld,
               IndexT D,
               IndexT N,
               const MapIteratorT map,
               StencilIteratorT stencil,
               IndexT map_length,
               OutputIteratorT out,
               UnaryPredicateOp pred_op,
               cudaStream_t stream)
{
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(in, ld, D, N, map, stencil, map_length, out, pred_op, raft::identity_op(), stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  stencil      Pointer to the input sequence of stencil or predicate values
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
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
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(in, D, D, N, map, stencil, map_length, out, pred_op, raft::identity_op(), stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a transformed map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT type.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  ld           Leading dimension of the input matrix 'in'
 * @param  D            Dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  stencil      Pointer to the input sequence of stencil or predicate values
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  pred_op      Predicate to apply to the stencil values
 * @param  transform_op The transformation operation, transforms the map values to IndexT
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
               IndexT ld,
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
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(in, ld, D, N, map, stencil, map_length, out, pred_op, transform_op, stream);
}

/**
 * @brief  gather_if conditionally copies rows from a source matrix into a destination matrix
 * according to a transformed map.
 *
 * @tparam InputIteratorT       Random-access iterator type, for reading input matrix (may be a
 * simple pointer type).
 * @tparam MapIteratorT         Random-access iterator type, for reading input map (may be a simple
 * pointer type).
 * @tparam StencilIteratorT     Random-access iterator type, for reading input stencil (may be a
 * simple pointer type).
 * @tparam UnaryPredicateOp     Unary lambda expression or operator type, UnaryPredicateOp's result
 * type must be convertible to bool type.
 * @tparam MapTransformOp       Unary lambda expression or operator type, MapTransformOp's result
 * type must be convertible to IndexT type.
 * @tparam OutputIteratorT      Random-access iterator type, for writing output matrix (may be a
 * simple pointer type).
 * @tparam IndexT               Index type.
 *
 * @param  in           Pointer to the input matrix (assumed to be row-major)
 * @param  D            Leading dimension of the input matrix 'in', which in-case of row-major
 * storage is the number of columns
 * @param  N            Second dimension
 * @param  map          Pointer to the input sequence of gather locations
 * @param  stencil      Pointer to the input sequence of stencil or predicate values
 * @param  map_length   The length of 'map' and 'stencil'
 * @param  out          Pointer to the output matrix (assumed to be row-major)
 * @param  pred_op      Predicate to apply to the stencil values
 * @param  transform_op The transformation operation, transforms the map values to IndexT
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
  typedef typename std::iterator_traits<MapIteratorT>::value_type MapValueT;
  gatherImpl(in, D, D, N, map, stencil, map_length, out, pred_op, transform_op, stream);
}

/**
 * Helper function to gather a set of vectors from a (host) dataset.
 */
template <typename T, typename IdxT, typename Accessor, typename MatIdxT = int64_t>
void gather_buff(
  raft::mdspan<const T, raft::matrix_extent<MatIdxT>, raft::layout_stride, Accessor> dataset,
  host_vector_view<const IdxT, MatIdxT> indices,
  MatIdxT offset,
  pinned_matrix_view<T, MatIdxT> buff)
{
  raft::common::nvtx::range<common::nvtx::domain::raft> fun_scope("gather_host_buff");
  MatIdxT batch_size = std::min<MatIdxT>(buff.extent(0), indices.extent(0) - offset);

#pragma omp for
  for (MatIdxT i = 0; i < batch_size; i++) {
    IdxT in_idx = indices(offset + i);
    for (MatIdxT k = 0; k < buff.extent(1); k++) {
      buff(i, k) = dataset(in_idx, k);
    }
  }
}

template <typename T, typename IdxT, typename MatIdxT = int64_t>
void gather(raft::resources const& res,
            raft::mdspan<const T,
                         raft::matrix_extent<MatIdxT>,
                         raft::layout_stride,
                         raft::host_device_accessor<cuda::std::default_accessor<const T>,
                                                    raft::memory_type::host>> dataset,
            device_vector_view<const IdxT, MatIdxT> indices,
            raft::device_matrix_view<T, MatIdxT> output)
{
  auto dry_run = resource::get_dry_run_flag(res);
  raft::common::nvtx::range<common::nvtx::domain::raft> fun_scope("gather");
  IdxT n_dim        = output.extent(1);
  IdxT n_train      = output.extent(0);
  auto indices_host = raft::make_host_vector<IdxT, MatIdxT>(n_train);
  if (!dry_run) {
    raft::copy(
      indices_host.data_handle(), indices.data_handle(), n_train, resource::get_cuda_stream(res));
  }

  const size_t buffer_size = 32768 * 1024;  // bytes
  const size_t max_batch_size =
    std::min<size_t>(round_up_safe<size_t>(buffer_size / n_dim, 32), n_train);
  RAFT_LOG_DEBUG("Gathering data with batch size %zu", max_batch_size);

  // Gather the vector on the host in tmp buffers. We use two buffers to overlap H2D sync
  // and gathering the data.
  auto out_tmp1 = raft::make_pinned_matrix<T, MatIdxT>(res, max_batch_size, n_dim);
  auto out_tmp2 = raft::make_pinned_matrix<T, MatIdxT>(res, max_batch_size, n_dim);

  if (dry_run) { return; }

  resource::sync_stream(res);

  // Usually a limited number of threads provide sufficient bandwidth for gathering data.
#if defined(_OPENMP)
  int n_threads = std::min(omp_get_max_threads(), 32);
#endif

  // The gather_buff function has a parallel for loop. We start the the omp parallel
  // region here, to avoid repeated overhead within the device_offset loop.
#pragma omp parallel num_threads(n_threads)
  {
    auto view1 = out_tmp1.view();
    auto view2 = out_tmp2.view();
    gather_buff(dataset, make_const_mdspan(indices_host.view()), (MatIdxT)0, view1);
    for (MatIdxT device_offset = 0; device_offset < n_train; device_offset += max_batch_size) {
      MatIdxT batch_size = std::min<IdxT>(max_batch_size, n_train - device_offset);

#pragma omp master
      raft::copy(output.data_handle() + device_offset * n_dim,
                 view1.data_handle(),
                 batch_size * n_dim,
                 resource::get_cuda_stream(res));
      // Start gathering the next batch on the host.
      MatIdxT host_offset = device_offset + batch_size;
      batch_size          = std::min<IdxT>(max_batch_size, n_train - host_offset);
      if (batch_size > 0) {
        gather_buff(dataset, make_const_mdspan(indices_host.view()), host_offset, view2);
      }
#pragma omp master
      resource::sync_stream(res);
#pragma omp barrier
      std::swap(view1, view2);
    }
  }
}

template <typename T, typename IdxT, typename MatIdxT = int64_t>
void gather(raft::resources const& res,
            host_matrix_view<const T, MatIdxT> dataset,
            device_vector_view<const IdxT, MatIdxT> indices,
            raft::device_matrix_view<T, MatIdxT> output)
{
  raft::mdspan<
    const T,
    raft::matrix_extent<MatIdxT>,
    raft::layout_stride,
    raft::host_device_accessor<cuda::std::default_accessor<const T>, raft::memory_type::host>>
    dataset_ = raft::make_host_strided_matrix_view<const T, MatIdxT>(
      dataset.data_handle(), dataset.extent(0), dataset.extent(1), dataset.extent(1));

  gather(res, dataset_, indices, output);
}

}  // namespace detail
}  // namespace matrix
}  // namespace raft
