/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <algorithm>

namespace raft {
namespace spectral {

template <typename vertex_t, typename edge_t, typename weight_t>
void transform_eigen_matrix(raft::resources const& handle,
                            edge_t n,
                            vertex_t nEigVecs,
                            weight_t* eigVecs)
{
  // Allocate before dry-run check to track allocation
  raft::spectral::matrix::vector_t<weight_t> work(handle, nEigVecs * n);

  if (resource::get_dry_run_flag(handle)) { return; }

  auto stream             = resource::get_cuda_stream(handle);
  auto cublas_h           = resource::get_cublas_handle(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  const weight_t zero{0.0};
  const weight_t one{1.0};

  // Whiten eigenvector matrix
  for (auto i = 0; i < nEigVecs; ++i) {
    weight_t mean, std;

    mean = thrust::reduce(thrust_exec_policy,
                          thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                          thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)));
    RAFT_CHECK_CUDA(stream);
    mean /= n;
    thrust::transform(thrust_exec_policy,
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      cuda::make_constant_iterator(mean),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      cuda::std::minus<weight_t>());
    RAFT_CHECK_CUDA(stream);

    // TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublasnrm2(cublas_h, n, eigVecs + IDX(0, i, n), 1, &std, stream));

    std /= std::sqrt(static_cast<weight_t>(n));

    thrust::transform(thrust_exec_policy,
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i + 1, n)),
                      cuda::make_constant_iterator(std),
                      thrust::device_pointer_cast(eigVecs + IDX(0, i, n)),
                      cuda::std::divides<weight_t>());
    RAFT_CHECK_CUDA(stream);
  }

  // Transpose eigenvector matrix
  //   TODO: in-place transpose
  {
    // TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(
      raft::linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

    // TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgeam(cublas_h,
                                                     CUBLAS_OP_T,
                                                     CUBLAS_OP_N,
                                                     nEigVecs,
                                                     n,
                                                     &one,
                                                     eigVecs,
                                                     n,
                                                     &zero,
                                                     (weight_t*)NULL,
                                                     nEigVecs,
                                                     work.raw(),
                                                     nEigVecs,
                                                     stream));

    RAFT_CUDA_TRY(cudaMemcpyAsync(
      eigVecs, work.raw(), nEigVecs * n * sizeof(weight_t), cudaMemcpyDeviceToDevice, stream));
  }
}

namespace {
/// Functor to generate indicator vectors
/** For use in Thrust transform
 */
template <typename index_type_t, typename value_type_t>
struct equal_to_i_op {
  const index_type_t i;

 public:
  equal_to_i_op(index_type_t _i) : i(_i) {}
  template <typename Tuple_>
  __host__ __device__ void operator()(Tuple_ t)
  {
    cuda::std::get<1>(t) = (cuda::std::get<0>(t) == i) ? (value_type_t)1.0 : (value_type_t)0.0;
  }
};
}  // namespace

// Construct indicator vector for ith partition
//
template <typename vertex_t, typename edge_t, typename weight_t, typename nnz_t>
bool construct_indicator(
  raft::resources const& handle,
  edge_t index,
  edge_t n,
  weight_t& clustersize,
  weight_t& partStats,
  vertex_t const* __restrict__ clusters,
  raft::spectral::matrix::vector_t<weight_t>& part_i,
  raft::spectral::matrix::vector_t<weight_t>& Bx,
  raft::spectral::matrix::laplacian_matrix_t<vertex_t, weight_t, nnz_t> const& B)
{
  if (resource::get_dry_run_flag(handle)) { return {}; }
  auto stream             = resource::get_cuda_stream(handle);
  auto cublas_h           = resource::get_cublas_handle(handle);
  auto thrust_exec_policy = resource::get_thrust_policy(handle);

  thrust::for_each(
    thrust_exec_policy,
    thrust::make_zip_iterator(cuda::std::make_tuple(thrust::device_pointer_cast(clusters),
                                                    thrust::device_pointer_cast(part_i.raw()))),
    thrust::make_zip_iterator(cuda::std::make_tuple(thrust::device_pointer_cast(clusters + n),
                                                    thrust::device_pointer_cast(part_i.raw() + n))),
    equal_to_i_op<vertex_t, weight_t>(index));
  RAFT_CHECK_CUDA(stream);

  // Compute size of ith partition
  // TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasdot(
    cublas_h, n, part_i.raw(), 1, part_i.raw(), 1, &clustersize, stream));

  clustersize = round(clustersize);
  if (clustersize < 0.5) { return false; }

  // Compute part stats
  B.mv(1, part_i.raw(), 0, Bx.raw());
  // TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublasdot(cublas_h, n, Bx.raw(), 1, part_i.raw(), 1, &partStats, stream));

  return true;
}

}  // namespace spectral
}  // namespace raft
