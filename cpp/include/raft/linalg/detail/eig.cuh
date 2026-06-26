/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cusolver_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resource/detail/stream_sync_event.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>

namespace raft {
namespace linalg {
namespace detail {

template <typename math_t>
void eigDC_legacy(raft::resources const& handle,
                  const math_t* in,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  math_t* eig_vectors,
                  math_t* eig_vals,
                  cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

  int lwork;
  RAFT_CUSOLVER_TRY(cusolverDnsyevd_bufferSize(cusolverH,
                                               CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_UPPER,
                                               n_rows,
                                               in,
                                               n_cols,
                                               eig_vals,
                                               &lwork));

  // TODO(achirkin): Consider using the workspace resource for these temporary allocations.
  rmm::device_uvector<math_t> d_work(lwork, stream);
  rmm::device_scalar<int> d_dev_info(stream);

  // The workspace is already allocated, no more allocation are foreseeable.
  if (resource::get_dry_run_flag(handle)) { return; }

  raft::matrix::copy(handle,
                     make_device_matrix_view<const math_t>(in, n_rows, n_cols),
                     make_device_matrix_view<math_t>(eig_vectors, n_rows, n_cols));

  RAFT_CUSOLVER_TRY(cusolverDnsyevd(cusolverH,
                                    CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_UPPER,
                                    n_rows,
                                    eig_vectors,
                                    n_cols,
                                    eig_vals,
                                    d_work.data(),
                                    lwork,
                                    d_dev_info.data(),
                                    stream));
  RAFT_CUDA_TRY(cudaGetLastError());

  auto dev_info = d_dev_info.value(stream);
  ASSERT(dev_info == 0,
         "eig.cuh: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");
}

template <typename math_t>
void eigDC(raft::resources const& handle,
           const math_t* in,
           std::size_t n_rows,
           std::size_t n_cols,
           math_t* eig_vectors,
           math_t* eig_vals,
           cudaStream_t stream)
{
  int cusolver_major = -1;
  int cusolver_minor = -1;
  int cusolver_patch = -1;
  RAFT_CUSOLVER_TRY(cusolverGetProperty(MAJOR_VERSION, &cusolver_major));
  RAFT_CUSOLVER_TRY(cusolverGetProperty(MINOR_VERSION, &cusolver_minor));
  RAFT_CUSOLVER_TRY(cusolverGetProperty(PATCH_LEVEL, &cusolver_patch));
  int cusolver_version = cusolver_major * 10000 + cusolver_minor * 100 + cusolver_patch;
  cudaStream_t stream_new;
  cudaEvent_t sync_event = resource::detail::get_cuda_stream_sync_event(handle);
  rmm::cuda_stream stream_new_wrapper;
  if (cusolver_version < 110603) {
    // Use a new stream instead of `cudaStreamPerThread` to avoid cusolver bug # 4580093.
    // Solved in cusolver 11.6.2.40 / CUDA 12.5.
    stream_new = stream_new_wrapper.value();
    RAFT_CUDA_TRY(cudaEventRecord(sync_event, stream));
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream_new, sync_event));
  } else {
    stream_new = stream;
  }
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

  cusolverDnParams_t dn_params = nullptr;
  RAFT_CUSOLVER_TRY(cusolverDnCreateParams(&dn_params));

  size_t workspaceDevice = 0;
  size_t workspaceHost   = 0;
  RAFT_CUSOLVER_TRY(cusolverDnxsyevd_bufferSize(cusolverH,
                                                dn_params,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                CUBLAS_FILL_MODE_UPPER,
                                                static_cast<int64_t>(n_rows),
                                                eig_vectors,
                                                static_cast<int64_t>(n_cols),
                                                eig_vals,
                                                &workspaceDevice,
                                                &workspaceHost,
                                                stream_new));

  rmm::device_uvector<math_t> d_work(workspaceDevice / sizeof(math_t), stream_new);
  rmm::device_scalar<int> d_dev_info(stream_new);
  std::vector<math_t> h_work(workspaceHost / sizeof(math_t));

  if (resource::get_dry_run_flag(handle)) {
    // No more allocations beyond this points, but need to cleanup.
    RAFT_CUSOLVER_TRY(cusolverDnDestroyParams(dn_params));
    return;
  }

  raft::copy(eig_vectors, in, n_rows * n_cols, stream_new);

  RAFT_CUSOLVER_TRY(cusolverDnxsyevd(cusolverH,
                                     dn_params,
                                     CUSOLVER_EIG_MODE_VECTOR,
                                     CUBLAS_FILL_MODE_UPPER,
                                     static_cast<int64_t>(n_rows),
                                     eig_vectors,
                                     static_cast<int64_t>(n_cols),
                                     eig_vals,
                                     d_work.data(),
                                     workspaceDevice,
                                     h_work.data(),
                                     workspaceHost,
                                     d_dev_info.data(),
                                     stream_new));

  RAFT_CUDA_TRY(cudaGetLastError());
  RAFT_CUSOLVER_TRY(cusolverDnDestroyParams(dn_params));
  int dev_info = d_dev_info.value(stream_new);
  ASSERT(dev_info == 0,
         "eig.cuh: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");

  if (cusolver_version < 110603) {
    // Synchronize the created stream with the original stream before return
    RAFT_CUDA_TRY(cudaEventRecord(sync_event, stream_new));
    RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, sync_event));
  }
}

enum EigVecMemUsage { OVERWRITE_INPUT, COPY_INPUT };

template <typename math_t>
void eigSelDC(raft::resources const& handle,
              math_t* in,
              std::size_t n_rows,
              std::size_t n_cols,
              std::size_t n_eig_vals,
              math_t* eig_vectors,
              math_t* eig_vals,
              EigVecMemUsage memUsage,
              cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

  int lwork;
  int h_meig;

  RAFT_CUSOLVER_TRY(cusolverDnsyevdx_bufferSize(cusolverH,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                CUSOLVER_EIG_RANGE_I,
                                                CUBLAS_FILL_MODE_UPPER,
                                                static_cast<int64_t>(n_rows),
                                                in,
                                                static_cast<int64_t>(n_cols),
                                                math_t(0.0),
                                                math_t(0.0),
                                                static_cast<int64_t>(n_cols - n_eig_vals + 1),
                                                static_cast<int64_t>(n_cols),
                                                &h_meig,
                                                eig_vals,
                                                &lwork));

  rmm::device_uvector<math_t> d_work(lwork, stream);
  rmm::device_scalar<int> d_dev_info(stream);
  rmm::device_uvector<math_t> d_eig_vectors(memUsage == COPY_INPUT ? n_rows * n_cols : 0, stream);

  if (resource::get_dry_run_flag(handle)) { return; }

  if (memUsage == OVERWRITE_INPUT) {
    RAFT_CUSOLVER_TRY(cusolverDnsyevdx(cusolverH,
                                       CUSOLVER_EIG_MODE_VECTOR,
                                       CUSOLVER_EIG_RANGE_I,
                                       CUBLAS_FILL_MODE_UPPER,
                                       static_cast<int64_t>(n_rows),
                                       in,
                                       static_cast<int64_t>(n_cols),
                                       math_t(0.0),
                                       math_t(0.0),
                                       static_cast<int64_t>(n_cols - n_eig_vals + 1),
                                       static_cast<int64_t>(n_cols),
                                       &h_meig,
                                       eig_vals,
                                       d_work.data(),
                                       lwork,
                                       d_dev_info.data(),
                                       stream));
  } else if (memUsage == COPY_INPUT) {
    raft::matrix::copy(handle,
                       make_device_matrix_view<const math_t>(in, n_rows, n_cols),
                       make_device_matrix_view(eig_vectors, n_rows, n_cols));

    RAFT_CUSOLVER_TRY(cusolverDnsyevdx(cusolverH,
                                       CUSOLVER_EIG_MODE_VECTOR,
                                       CUSOLVER_EIG_RANGE_I,
                                       CUBLAS_FILL_MODE_UPPER,
                                       static_cast<int64_t>(n_rows),
                                       eig_vectors,
                                       static_cast<int64_t>(n_cols),
                                       math_t(0.0),
                                       math_t(0.0),
                                       static_cast<int64_t>(n_cols - n_eig_vals + 1),
                                       static_cast<int64_t>(n_cols),
                                       &h_meig,
                                       eig_vals,
                                       d_work.data(),
                                       lwork,
                                       d_dev_info.data(),
                                       stream));
  }

  RAFT_CUDA_TRY(cudaGetLastError());

  int dev_info = d_dev_info.value(stream);
  ASSERT(dev_info == 0,
         "eig.cuh: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");

  if (memUsage == OVERWRITE_INPUT) {
    raft::matrix::trunc_zero_origin(
      handle,
      make_device_matrix_view<const math_t, size_t, col_major>(in, n_rows, n_eig_vals),
      make_device_matrix_view<math_t, size_t, col_major>(eig_vectors, n_rows, n_eig_vals));
  } else if (memUsage == COPY_INPUT) {
    raft::matrix::trunc_zero_origin(
      handle,
      make_device_matrix_view<const math_t, size_t, col_major>(
        d_eig_vectors.data(), n_rows, n_eig_vals),
      make_device_matrix_view<math_t, size_t, col_major>(eig_vectors, n_rows, n_eig_vals));
  }
}

template <typename math_t>
void eigJacobi(raft::resources const& handle,
               const math_t* in,
               std::size_t n_rows,
               std::size_t n_cols,
               math_t* eig_vectors,
               math_t* eig_vals,
               cudaStream_t stream,
               math_t tol = 1.e-7,
               int sweeps = 15)
{
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

  syevjInfo_t syevj_params = nullptr;
  RAFT_CUSOLVER_TRY(cusolverDnCreateSyevjInfo(&syevj_params));
  RAFT_CUSOLVER_TRY(cusolverDnXsyevjSetTolerance(syevj_params, tol));
  RAFT_CUSOLVER_TRY(cusolverDnXsyevjSetMaxSweeps(syevj_params, sweeps));

  int lwork;
  RAFT_CUSOLVER_TRY(cusolverDnsyevj_bufferSize(cusolverH,
                                               CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_UPPER,
                                               static_cast<int64_t>(n_rows),
                                               eig_vectors,
                                               static_cast<int64_t>(n_cols),
                                               eig_vals,
                                               &lwork,
                                               syevj_params));

  rmm::device_uvector<math_t> d_work(lwork, stream);
  rmm::device_scalar<int> dev_info(stream);

  if (resource::get_dry_run_flag(handle)) {
    // No more allocations beyond this points, but need to cleanup.
    RAFT_CUSOLVER_TRY(cusolverDnDestroySyevjInfo(syevj_params));
    return;
  }

  raft::matrix::copy(handle,
                     make_device_matrix_view<const math_t>(in, n_rows, n_cols),
                     make_device_matrix_view(eig_vectors, n_rows, n_cols));

  RAFT_CUSOLVER_TRY(cusolverDnsyevj(cusolverH,
                                    CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_UPPER,
                                    static_cast<int64_t>(n_rows),
                                    eig_vectors,
                                    static_cast<int64_t>(n_cols),
                                    eig_vals,
                                    d_work.data(),
                                    lwork,
                                    dev_info.data(),
                                    syevj_params,
                                    stream));

  int executed_sweeps;
  RAFT_CUSOLVER_TRY(cusolverDnXsyevjGetSweeps(cusolverH, syevj_params, &executed_sweeps));

  RAFT_CUDA_TRY(cudaGetLastError());
  RAFT_CUSOLVER_TRY(cusolverDnDestroySyevjInfo(syevj_params));
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft
