/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// for cmath:
#define _USE_MATH_DEFINES

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/types.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/axpy.cuh>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/detail/add.cuh>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/detail/gemv.hpp>
#include <raft/linalg/dot.cuh>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/init.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/norm_types.hpp>
#include <raft/linalg/normalize.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/gather.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/matrix/triangular.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/linalg/detail/cusparse_utils.hpp>
#include <raft/sparse/solver/lanczos_types.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda.h>
#include <thrust/sort.h>

#include <cublasLt.h>
#include <curand.h>
#include <cusparse.h>
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace raft {
namespace sparse::solver::detail {

template <typename T>
RAFT_KERNEL kernel_triangular_populate(T* M, const T* beta, int n)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    // Upper diagonal: M[row + 1, row] in column-major
    if (row < n - 1) { M[(row + 1) * n + row] = beta[row]; }

    // Lower diagonal: M[row - 1, row] in column-major
    if (row > 0) { M[(row - 1) * n + row] = beta[row - 1]; }
  }
}

template <typename T>
RAFT_KERNEL kernel_triangular_beta_k(T* t, const T* beta_k, int k, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < k) {
    // Update the k-th column: t[i, k] -> t[k * n + i] in column-major
    t[tid * n + k] = beta_k[tid];

    // Update the k-th row: t[k, j] -> t[j * n + k] in column-major
    t[k * n + tid] = beta_k[tid];
  }
}

template <typename T>
RAFT_KERNEL kernel_normalize(const T* u, const T* beta, int j, int n, T* v, T* V, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size) {
    if (beta[j] == 0) {
      v[i] = u[i] / 1;
    } else {
      v[i] = u[i] / beta[j];
    }
    V[i + (j + 1) * n] = v[i];
  }
}

template <typename T>
RAFT_KERNEL kernel_clamp_down(T* value, T threshold)
{
  *value = (fabs(*value) < threshold) ? 0 : *value;
}

template <typename T>
RAFT_KERNEL kernel_clamp_down_vector(T* vec, T threshold, int size)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < size) { vec[idx] = (fabs(vec[idx]) < threshold) ? 0 : vec[idx]; }
}

template <typename IndexTypeT, typename ValueTypeT>
void lanczos_solve_ritz(
  raft::resources const& handle,
  raft::device_matrix_view<ValueTypeT, uint32_t, raft::row_major> alpha,
  raft::device_matrix_view<ValueTypeT, uint32_t, raft::row_major> beta,
  std::optional<raft::device_vector_view<ValueTypeT, uint32_t>> beta_k,
  IndexTypeT k,
  LANCZOS_WHICH which,
  int ncv,
  raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> eigenvectors,
  raft::device_vector_view<ValueTypeT> eigenvalues,
  raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major>& eigenvectors_k,
  raft::device_vector_view<ValueTypeT, uint32_t>& eigenvalues_k,
  raft::device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major>& eigenvectors_k_slice,
  raft::device_vector_view<ValueTypeT> sm_eigenvalues,
  raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> sm_eigenvectors)
{
  auto stream           = resource::get_cuda_stream(handle);
  bool const is_dry_run = resource::get_dry_run_flag(handle);

  ValueTypeT zero = 0;
  auto triangular_matrix =
    raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, ncv, ncv);
  raft::matrix::fill(handle, triangular_matrix.view(), zero);

  raft::device_vector_view<const ValueTypeT, uint32_t> alphaVec =
    raft::make_device_vector_view<const ValueTypeT, uint32_t>(alpha.data_handle(), ncv);
  raft::matrix::set_diagonal(handle, alphaVec, triangular_matrix.view());

  if (!is_dry_run) {
    int blockSize = 256;
    int numBlocks = (ncv + blockSize - 1) / blockSize;
    kernel_triangular_populate<ValueTypeT><<<blockSize, numBlocks, 0, stream>>>(
      triangular_matrix.data_handle(), beta.data_handle(), ncv);

    if (beta_k) {
      int threadsPerBlock = 256;
      int blocksPerGrid   = (k + threadsPerBlock - 1) / threadsPerBlock;
      kernel_triangular_beta_k<ValueTypeT><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        triangular_matrix.data_handle(), beta_k.value().data_handle(), (int)k, ncv);
    }
  }

  auto triangular_matrix_view =
    raft::make_device_matrix_view<const ValueTypeT, uint32_t, raft::col_major>(
      triangular_matrix.data_handle(), ncv, ncv);

  raft::linalg::eig_dc(handle, triangular_matrix_view, eigenvectors, eigenvalues);

  IndexTypeT nEigVecs = k;

  auto indices          = raft::make_device_vector<int>(handle, ncv);
  auto selected_indices = raft::make_device_vector<int>(handle, nEigVecs);

  if (which == LANCZOS_WHICH::SA) {
    eigenvectors_k = raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(
      eigenvectors.data_handle(), ncv, nEigVecs);
    eigenvalues_k =
      raft::make_device_vector_view<ValueTypeT, uint32_t>(eigenvalues.data_handle(), nEigVecs);
    eigenvectors_k_slice = raft::make_device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major>(
      eigenvectors.data_handle(), ncv, nEigVecs);
  } else if (which == LANCZOS_WHICH::LA) {
    eigenvectors_k = raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(
      eigenvectors.data_handle() + (ncv - nEigVecs) * ncv, ncv, nEigVecs);
    eigenvalues_k = raft::make_device_vector_view<ValueTypeT, uint32_t>(
      eigenvalues.data_handle() + (ncv - nEigVecs), nEigVecs);
    eigenvectors_k_slice = raft::make_device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major>(
      eigenvectors.data_handle() + (ncv - nEigVecs) * ncv, ncv, nEigVecs);
  } else if (which == LANCZOS_WHICH::SM || which == LANCZOS_WHICH::LM) {
    if (!is_dry_run) {  // TODO: we must be missing some allocations in thrust helpers here
      thrust::sequence(thrust::device, indices.data_handle(), indices.data_handle() + ncv, 0);

      // Sort indices by absolute eigenvalues (magnitude) using a custom comparator
      thrust::sort(thrust::device,
                   indices.data_handle(),
                   indices.data_handle() + ncv,
                   [eigenvalues = eigenvalues.data_handle()] __device__(int a, int b) {
                     return fabsf(eigenvalues[a]) < fabsf(eigenvalues[b]);
                   });

      if (which == LANCZOS_WHICH::SM) {
        // Take the first nEigVecs indices (smallest magnitude)
        raft::copy(selected_indices.data_handle(), indices.data_handle(), nEigVecs, stream);
      } else if (which == LANCZOS_WHICH::LM) {
        // Take the last nEigVecs indices (largest magnitude)
        raft::copy(selected_indices.data_handle(),
                   indices.data_handle() + (ncv - nEigVecs),
                   nEigVecs,
                   stream);
      }

      // Re-sort these indices by algebraic value to maintain algebraic ordering
      thrust::sort(thrust::device,
                   selected_indices.data_handle(),
                   selected_indices.data_handle() + nEigVecs,
                   [eigenvalues = eigenvalues.data_handle()] __device__(int a, int b) {
                     return eigenvalues[a] < eigenvalues[b];
                   });
    }
    raft::matrix::gather(
      handle,
      raft::make_device_matrix_view<const ValueTypeT, uint32_t, raft::row_major>(
        eigenvalues.data_handle(), ncv, 1),
      raft::make_device_vector_view<const int, uint32_t>(selected_indices.data_handle(), nEigVecs),
      raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::row_major>(
        sm_eigenvalues.data_handle(), nEigVecs, 1));
    raft::matrix::gather(
      handle,
      raft::make_device_matrix_view<const ValueTypeT, uint32_t, raft::row_major>(
        eigenvectors.data_handle(), ncv, ncv),
      raft::make_device_vector_view<const int, uint32_t>(selected_indices.data_handle(), nEigVecs),
      raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::row_major>(
        sm_eigenvectors.data_handle(), nEigVecs, ncv));

    eigenvectors_k = raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(
      sm_eigenvectors.data_handle(), ncv, nEigVecs);
    eigenvalues_k =
      raft::make_device_vector_view<ValueTypeT, uint32_t>(sm_eigenvalues.data_handle(), nEigVecs);
    eigenvectors_k_slice = raft::make_device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major>(
      sm_eigenvectors.data_handle(), ncv, nEigVecs);
  }
}

template <typename IndexTypeT, typename ValueTypeT, typename AType>
void lanczos_aux(raft::resources const& handle,
                 AType A,
                 raft::device_matrix_view<ValueTypeT, uint32_t, raft::row_major> V,
                 raft::device_matrix_view<ValueTypeT, uint32_t> u,
                 raft::device_matrix_view<ValueTypeT, uint32_t> alpha,
                 raft::device_matrix_view<ValueTypeT, uint32_t> beta,
                 int start_idx,
                 int end_idx,
                 int ncv,
                 raft::device_matrix_view<ValueTypeT, uint32_t> v,
                 raft::device_matrix_view<ValueTypeT, uint32_t> uu,
                 raft::device_matrix_view<ValueTypeT, uint32_t> vv,
                 std::optional<uint64_t> seed)
{
  // Deterministic when seed is provided
  cusparseSpMVAlg_t spmv_alg;
  if (seed.has_value()) {
    if constexpr (is_device_coo_matrix_view<AType>::value) {
      spmv_alg = CUSPARSE_SPMV_COO_ALG2;
    } else {
      spmv_alg = CUSPARSE_SPMV_CSR_ALG2;
    }
  } else {
    spmv_alg = CUSPARSE_SPMV_ALG_DEFAULT;
  }
  auto stream           = resource::get_cuda_stream(handle);
  bool const is_dry_run = resource::get_dry_run_flag(handle);

  IndexTypeT n  = A.structure_view().get_n_rows();
  auto v_vector = raft::make_device_vector_view<const ValueTypeT>(v.data_handle(), n);
  auto u_vector = raft::make_device_vector_view<const ValueTypeT>(u.data_handle(), n);

  auto cusparse_h                 = resource::get_cusparse_handle(handle);
  cusparseSpMatDescr_t cusparse_A = raft::sparse::linalg::detail::create_descriptor(A);

  cusparseDnVecDescr_t cusparse_v = raft::sparse::linalg::detail::create_descriptor(v_vector);
  cusparseDnVecDescr_t cusparse_u = raft::sparse::linalg::detail::create_descriptor(u_vector);

  ValueTypeT one  = 1;
  ValueTypeT zero = 0;
  size_t bufferSize;
  raft::sparse::detail::cusparsespmv_buffersize(cusparse_h,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &one,
                                                cusparse_A,
                                                cusparse_v,
                                                &zero,
                                                cusparse_u,
                                                spmv_alg,
                                                &bufferSize,
                                                stream);
  auto cusparse_spmv_buffer = raft::make_device_vector<ValueTypeT>(handle, bufferSize);

  if (!is_dry_run) {
    raft::copy(
      v.data_handle(), V.data_handle() + start_idx * V.stride(0), n, stream);  // V(start_idx, 0)
  }

  for (int i = start_idx; i < end_idx; i++) {
    if (!is_dry_run) {
      raft::sparse::detail::cusparsespmv(cusparse_h,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one,
                                         cusparse_A,
                                         cusparse_v,
                                         &zero,
                                         cusparse_u,
                                         spmv_alg,
                                         cusparse_spmv_buffer.data_handle(),
                                         stream);
    }

    auto alpha_i =
      raft::make_device_scalar_view(alpha.data_handle() + i * alpha.stride(1));  // alpha(0, i)
    raft::linalg::dot(handle, v_vector, u_vector, alpha_i);

    raft::matrix::fill(handle, vv, zero);

    auto cublas_h = resource::get_cublas_handle(handle);

    ValueTypeT alpha_i_host = 0;
    ValueTypeT b            = 0;
    ValueTypeT mone         = -1;

    if (!is_dry_run) {
      raft::copy<ValueTypeT>(
        &b, beta.data_handle() + ((i - 1 + ncv) % ncv) * beta.stride(1), 1, stream);
      raft::copy<ValueTypeT>(
        &alpha_i_host, alpha.data_handle() + i * alpha.stride(1), 1, stream);  // alpha(0, i)
    }

    raft::linalg::axpy(handle, n, &alpha_i_host, v.data_handle(), 1, vv.data_handle(), 1, stream);
    raft::linalg::axpy(handle,
                       n,
                       &b,
                       V.data_handle() + (((i - 1 + ncv) % ncv) * V.stride(0)),
                       1,
                       vv.data_handle(),
                       1,
                       stream);
    raft::linalg::axpy(handle, n, &mone, vv.data_handle(), 1, u.data_handle(), 1, stream);

    raft::linalg::gemv(handle,
                       CUBLAS_OP_T,
                       n,
                       i + 1,
                       &one,
                       V.data_handle(),
                       n,
                       u.data_handle(),
                       1,
                       &zero,
                       uu.data_handle(),
                       1,
                       stream);

    raft::linalg::gemv(handle,
                       CUBLAS_OP_N,
                       n,
                       i + 1,
                       &mone,
                       V.data_handle(),
                       n,
                       uu.data_handle(),
                       1,
                       &one,
                       u.data_handle(),
                       1,
                       stream);

    auto uu_i = raft::make_device_scalar_view(uu.data_handle() + uu.stride(1) * i);  // uu(0, i)
    raft::linalg::add(handle, make_const_mdspan(alpha_i), make_const_mdspan(uu_i), alpha_i);

    if (!is_dry_run) {
      kernel_clamp_down<<<1, 1, 0, stream>>>(alpha_i.data_handle(), static_cast<ValueTypeT>(1e-9));
    }

    auto output = raft::make_device_vector_view<ValueTypeT, uint32_t>(
      beta.data_handle() + beta.stride(1) * i, 1);
    auto input = raft::make_device_matrix_view<const ValueTypeT, uint32_t>(u.data_handle(), 1, n);
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle, input, output, raft::sqrt_op());

    if (!is_dry_run) {
      int blockSize = 256;
      int numBlocks = (n + blockSize - 1) / blockSize;

      kernel_clamp_down_vector<<<numBlocks, blockSize, 0, stream>>>(
        u.data_handle(), static_cast<ValueTypeT>(1e-7), n);

      kernel_clamp_down<<<1, 1, 0, stream>>>(beta.data_handle() + beta.stride(1) * i,
                                             static_cast<ValueTypeT>(1e-6));
    }

    if (i >= end_idx - 1) { break; }

    if (!is_dry_run) {
      int threadsPerBlock = 256;
      int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

      kernel_normalize<ValueTypeT><<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        u.data_handle(), beta.data_handle(), i, n, v.data_handle(), V.data_handle(), n);
    }
  }
}

template <typename IndexTypeT, typename ValueTypeT, typename AType>
auto lanczos_smallest(raft::resources const& handle,
                      AType A,
                      int nEigVecs,
                      int maxIter,
                      int restartIter,
                      ValueTypeT tol,
                      LANCZOS_WHICH which,
                      ValueTypeT* eigVals_dev,
                      ValueTypeT* eigVecs_dev,
                      ValueTypeT* v0,
                      std::optional<uint64_t> seed) -> int
{
  // Deterministic when seed is provided
  cusparseSpMVAlg_t spmv_alg;
  if (seed.has_value()) {
    if constexpr (is_device_coo_matrix_view<AType>::value) {
      spmv_alg = CUSPARSE_SPMV_COO_ALG2;
    } else {
      spmv_alg = CUSPARSE_SPMV_CSR_ALG2;
    }
  } else {
    spmv_alg = CUSPARSE_SPMV_ALG_DEFAULT;
  }
  int n           = A.structure_view().get_n_rows();
  int ncv         = restartIter;
  auto stream     = resource::get_cuda_stream(handle);
  bool is_dry_run = resource::get_dry_run_flag(handle);

  auto V = raft::make_device_matrix<ValueTypeT, uint32_t, raft::row_major>(handle, ncv, n);
  auto V_0_view =
    raft::make_device_matrix_view<ValueTypeT, uint32_t>(V.data_handle(), 1, n);  // First Row V[0]
  auto v0_view = raft::make_device_matrix_view<const ValueTypeT, uint32_t>(v0, 1, n);

  auto u        = raft::make_device_matrix<ValueTypeT, uint32_t, raft::row_major>(handle, 1, n);
  auto u_vector = raft::make_device_vector_view<ValueTypeT, uint32_t>(u.data_handle(), n);
  if (!is_dry_run) { raft::copy(u.data_handle(), v0, n, stream); }

  auto cublas_h = resource::get_cublas_handle(handle);
  auto v0nrm    = raft::make_device_vector<ValueTypeT, uint32_t>(handle, 1);
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    handle, v0_view, v0nrm.view(), raft::sqrt_op());

  auto v0_vector_const = raft::make_device_vector_view<const ValueTypeT, uint32_t>(v0, n);

  raft::linalg::unary_op(
    handle, v0_vector_const, V_0_view, [device_scalar = v0nrm.data_handle()] __device__(auto y) {
      return y / *device_scalar;
    });

  auto alpha      = raft::make_device_matrix<ValueTypeT, uint32_t, raft::row_major>(handle, 1, ncv);
  auto beta       = raft::make_device_matrix<ValueTypeT, uint32_t, raft::row_major>(handle, 1, ncv);
  ValueTypeT zero = 0;
  raft::matrix::fill(handle, alpha.view(), zero);
  raft::matrix::fill(handle, beta.view(), zero);

  auto v      = raft::make_device_matrix<ValueTypeT, uint32_t, raft::row_major>(handle, 1, n);
  auto aux_uu = raft::make_device_matrix<ValueTypeT, uint32_t, raft::row_major>(handle, 1, ncv);
  auto vv     = raft::make_device_matrix<ValueTypeT, uint32_t, raft::row_major>(handle, 1, n);

  lanczos_aux<IndexTypeT, ValueTypeT, AType>(handle,
                                             A,
                                             V.view(),
                                             u.view(),
                                             alpha.view(),
                                             beta.view(),
                                             0,
                                             ncv,
                                             ncv,
                                             v.view(),
                                             aux_uu.view(),
                                             vv.view(),
                                             seed);

  auto eigenvectors =
    raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, ncv, ncv);
  auto eigenvalues = raft::make_device_vector<ValueTypeT, uint32_t>(handle, ncv);

  raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> eigenvectors_k;
  raft::device_vector_view<ValueTypeT, uint32_t> eigenvalues_k;
  raft::device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major> eigenvectors_k_slice;

  auto sm_eigenvalues = raft::make_device_vector<ValueTypeT>(handle, nEigVecs);
  auto sm_eigenvectors =
    raft::make_device_matrix<ValueTypeT, uint32_t, raft::col_major>(handle, ncv, nEigVecs);

  lanczos_solve_ritz<IndexTypeT, ValueTypeT>(handle,
                                             alpha.view(),
                                             beta.view(),
                                             std::nullopt,
                                             nEigVecs,
                                             which,
                                             ncv,
                                             eigenvectors.view(),
                                             eigenvalues.view(),
                                             eigenvectors_k,
                                             eigenvalues_k,
                                             eigenvectors_k_slice,
                                             sm_eigenvalues.view(),
                                             sm_eigenvectors.view());

  auto ritz_eigenvectors =
    raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(eigVecs_dev, n, nEigVecs);

  auto V_T =
    raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(V.data_handle(), n, ncv);
  raft::linalg::gemm<ValueTypeT, uint32_t, raft::col_major, raft::col_major, raft::col_major>(
    handle, V_T, eigenvectors_k, ritz_eigenvectors);

  auto s = raft::make_device_vector<ValueTypeT>(handle, nEigVecs);

  auto S_matrix = raft::make_device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major>(
    s.data_handle(), 1, nEigVecs);

  raft::matrix::slice_coordinates<IndexTypeT> coords(ncv - 1, 0, ncv, nEigVecs);
  raft::matrix::slice(handle, make_const_mdspan(eigenvectors_k_slice), S_matrix, coords);

  auto beta_k = raft::make_device_vector<ValueTypeT>(handle, nEigVecs);
  raft::matrix::fill(handle, beta_k.view(), zero);
  auto beta_scalar = raft::make_device_scalar_view<const ValueTypeT>(beta.data_handle() +
                                                                     (ncv - 1) * beta.stride(1));

  raft::linalg::axpy(handle, beta_scalar, raft::make_const_mdspan(s.view()), beta_k.view());

  ValueTypeT res = 0;

  raft::device_vector<ValueTypeT, uint32_t> output =
    raft::make_device_vector<ValueTypeT, uint32_t>(handle, 1);
  raft::device_matrix_view<const ValueTypeT> input =
    raft::make_device_matrix_view<const ValueTypeT>(beta_k.data_handle(), 1, nEigVecs);
  raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
    handle, input, output.view(), raft::sqrt_op());
  if (!is_dry_run) {
    raft::copy(&res, output.data_handle(), 1, stream);
    resource::sync_stream(handle, stream);
  } else {
    // Force exactly one loop iteration so the MR records all allocations inside the loop body.
    res     = tol + 1;
    maxIter = ncv + (ncv - nEigVecs);
  }

  auto uu  = raft::make_device_matrix<ValueTypeT>(handle, 1, nEigVecs);
  int iter = ncv;
  while (res > tol && iter < maxIter) {
    auto beta_view = raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::row_major>(
      beta.data_handle(), 1, nEigVecs);
    raft::matrix::fill(handle, beta_view, zero);

    if (!is_dry_run) {
      raft::copy(alpha.data_handle(), eigenvalues_k.data_handle(), nEigVecs, stream);
    }

    auto x_T =
      raft::make_device_matrix_view<ValueTypeT>(ritz_eigenvectors.data_handle(), nEigVecs, n);

    if (!is_dry_run) { raft::copy(V.data_handle(), x_T.data_handle(), nEigVecs * n, stream); }

    ValueTypeT one  = 1;
    ValueTypeT mone = -1;

    raft::linalg::gemv(handle,
                       CUBLAS_OP_T,
                       n,
                       nEigVecs,
                       &one,
                       V.data_handle(),
                       n,
                       u.data_handle(),
                       1,
                       &zero,
                       uu.data_handle(),
                       1,
                       stream);

    raft::linalg::gemv(handle,
                       CUBLAS_OP_N,
                       n,
                       nEigVecs,
                       &mone,
                       V.data_handle(),
                       n,
                       uu.data_handle(),
                       1,
                       &one,
                       u.data_handle(),
                       1,
                       stream);

    auto V_0_view =
      raft::make_device_matrix_view<ValueTypeT>(V.data_handle() + (nEigVecs * n), 1, n);
    auto V_0_view_vector =
      raft::make_device_vector_view<ValueTypeT, uint32_t>(V_0_view.data_handle(), n);
    auto unrm = raft::make_device_vector<ValueTypeT, uint32_t>(handle, 1);
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle, raft::make_const_mdspan(u.view()), unrm.view(), raft::sqrt_op());

    raft::linalg::unary_op(
      handle,
      raft::make_const_mdspan(u_vector),
      V_0_view,
      [device_scalar = unrm.data_handle()] __device__(auto y) { return y / *device_scalar; });

    auto cusparse_h                 = resource::get_cusparse_handle(handle);
    cusparseSpMatDescr_t cusparse_A = raft::sparse::linalg::detail::create_descriptor(A);

    cusparseDnVecDescr_t cusparse_v =
      raft::sparse::linalg::detail::create_descriptor(V_0_view_vector);
    cusparseDnVecDescr_t cusparse_u = raft::sparse::linalg::detail::create_descriptor(u_vector);

    ValueTypeT zero = 0;
    size_t bufferSize;
    raft::sparse::detail::cusparsespmv_buffersize(cusparse_h,
                                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                  &one,
                                                  cusparse_A,
                                                  cusparse_v,
                                                  &zero,
                                                  cusparse_u,
                                                  spmv_alg,
                                                  &bufferSize,
                                                  stream);
    auto cusparse_spmv_buffer = raft::make_device_vector<ValueTypeT>(handle, bufferSize);

    if (!is_dry_run) {
      raft::sparse::detail::cusparsespmv(cusparse_h,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one,
                                         cusparse_A,
                                         cusparse_v,
                                         &zero,
                                         cusparse_u,
                                         spmv_alg,
                                         cusparse_spmv_buffer.data_handle(),
                                         stream);
    }

    auto alpha_k = raft::make_device_scalar_view<ValueTypeT>(alpha.data_handle() + nEigVecs);

    raft::linalg::dot(
      handle, make_const_mdspan(V_0_view_vector), make_const_mdspan(u_vector), alpha_k);

    raft::linalg::binary_op(handle,
                            make_const_mdspan(u_vector),
                            make_const_mdspan(V_0_view_vector),
                            u_vector,
                            [device_scalar_ptr = alpha_k.data_handle()] __device__(
                              ValueTypeT u_element, ValueTypeT V_0_element) {
                              return u_element - (*device_scalar_ptr) * V_0_element;
                            });

    auto temp = raft::make_device_vector<ValueTypeT, uint32_t>(handle, n);

    auto V_k = raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::row_major>(
      V.data_handle(), nEigVecs, n);
    auto V_k_T =
      raft::make_device_matrix<ValueTypeT, uint32_t, raft::row_major>(handle, n, nEigVecs);

    raft::linalg::transpose(handle, V_k, V_k_T.view());

    raft::linalg::gemv(handle,
                       CUBLAS_OP_N,
                       n,
                       nEigVecs,
                       &one,
                       V_k.data_handle(),
                       n,
                       beta_k.data_handle(),
                       1,
                       &zero,
                       temp.data_handle(),
                       1,
                       stream);

    auto one_scalar = raft::make_device_scalar<ValueTypeT>(handle, 1);
    raft::linalg::binary_op(handle,
                            make_const_mdspan(u_vector),
                            make_const_mdspan(temp.view()),
                            u_vector,
                            [device_scalar_ptr = one_scalar.data_handle()] __device__(
                              ValueTypeT u_element, ValueTypeT temp_element) {
                              return u_element - (*device_scalar_ptr) * temp_element;
                            });

    auto output1 = raft::make_device_vector_view<ValueTypeT, uint32_t>(
      beta.data_handle() + beta.stride(1) * nEigVecs, 1);
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle, raft::make_const_mdspan(u.view()), output1, raft::sqrt_op());

    auto V_kplus1 =
      raft::make_device_vector_view<ValueTypeT>(V.data_handle() + V.stride(0) * (nEigVecs + 1), n);

    raft::linalg::unary_op(
      handle,
      make_const_mdspan(u_vector),
      V_kplus1,
      [device_scalar = (beta.data_handle() + beta.stride(1) * nEigVecs)] __device__(auto y) {
        return y / *device_scalar;
      });

    lanczos_aux<IndexTypeT, ValueTypeT, AType>(handle,
                                               A,
                                               V.view(),
                                               u.view(),
                                               alpha.view(),
                                               beta.view(),
                                               nEigVecs + 1,
                                               ncv,
                                               ncv,
                                               v.view(),
                                               aux_uu.view(),
                                               vv.view(),
                                               seed);
    iter += ncv - nEigVecs;
    lanczos_solve_ritz<IndexTypeT, ValueTypeT>(handle,
                                               alpha.view(),
                                               beta.view(),
                                               beta_k.view(),
                                               nEigVecs,
                                               which,
                                               ncv,
                                               eigenvectors.view(),
                                               eigenvalues.view(),
                                               eigenvectors_k,
                                               eigenvalues_k,
                                               eigenvectors_k_slice,
                                               sm_eigenvalues.view(),
                                               sm_eigenvectors.view());

    auto ritz_eigenvectors = raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(
      eigVecs_dev, n, nEigVecs);

    auto V_T =
      raft::make_device_matrix_view<ValueTypeT, uint32_t, raft::col_major>(V.data_handle(), n, ncv);
    raft::linalg::gemm<ValueTypeT, uint32_t, raft::col_major, raft::col_major, raft::col_major>(
      handle, V_T, eigenvectors_k, ritz_eigenvectors);

    auto S_matrix = raft::make_device_matrix_view<ValueTypeT, IndexTypeT, raft::col_major>(
      s.data_handle(), 1, nEigVecs);

    raft::matrix::slice_coordinates<IndexTypeT> coords(ncv - 1, 0, ncv, nEigVecs);
    raft::matrix::slice(handle, make_const_mdspan(eigenvectors_k_slice), S_matrix, coords);

    raft::matrix::fill(handle, beta_k.view(), zero);

    auto beta_scalar = raft::make_device_scalar_view<const ValueTypeT>(
      beta.data_handle() + beta.stride(1) * (ncv - 1));  // &((beta.view())(0, ncv - 1))

    raft::linalg::axpy(handle, beta_scalar, raft::make_const_mdspan(s.view()), beta_k.view());

    raft::device_vector<ValueTypeT, uint32_t> output2 =
      raft::make_device_vector<ValueTypeT, uint32_t>(handle, 1);
    raft::device_matrix_view<const ValueTypeT> input2 =
      raft::make_device_matrix_view<const ValueTypeT>(beta_k.data_handle(), 1, nEigVecs);
    raft::linalg::norm<raft::linalg::L2Norm, raft::Apply::ALONG_ROWS>(
      handle, input2, output2.view(), raft::sqrt_op());
    if (!is_dry_run) {
      raft::copy(&res, output2.data_handle(), 1, stream);
      resource::sync_stream(handle, stream);
    }
    RAFT_LOG_TRACE("Iteration %f: residual (tolerance) %d", iter, res);
  }

  if (!is_dry_run) {
    raft::copy(eigVals_dev, eigenvalues_k.data_handle(), nEigVecs, stream);
    raft::copy(eigVecs_dev, ritz_eigenvectors.data_handle(), n * nEigVecs, stream);
  }

  return 0;
}

template <typename IndexTypeT, typename ValueTypeT, typename AType>
auto lanczos_compute_eigenpairs(
  raft::resources const& handle,
  lanczos_solver_config<ValueTypeT> const& config,
  AType A,
  std::optional<raft::device_vector_view<ValueTypeT, uint32_t>> v0,
  raft::device_vector_view<ValueTypeT, uint32_t> eigenvalues,
  raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> eigenvectors) -> int
{
  if (v0.has_value()) {
    return lanczos_smallest<IndexTypeT, ValueTypeT, AType>(handle,
                                                           A,
                                                           config.n_components,
                                                           config.max_iterations,
                                                           config.ncv,
                                                           config.tolerance,
                                                           config.which,
                                                           eigenvalues.data_handle(),
                                                           eigenvectors.data_handle(),
                                                           v0->data_handle(),
                                                           config.seed);
  } else {
    // Handle the optional v0 initial Lanczos vector if nullopt is used
    auto n        = A.structure_view().get_n_rows();
    auto temp_v0  = raft::make_device_vector<ValueTypeT, uint32_t>(handle, n);
    uint64_t seed = config.seed.value_or(std::random_device{}());
    raft::random::RngState rng_state(seed);
    raft::random::uniform(handle, rng_state, temp_v0.view(), ValueTypeT{0.0}, ValueTypeT{1.0});
    return lanczos_smallest<IndexTypeT, ValueTypeT, AType>(handle,
                                                           A,
                                                           config.n_components,
                                                           config.max_iterations,
                                                           config.ncv,
                                                           config.tolerance,
                                                           config.which,
                                                           eigenvalues.data_handle(),
                                                           eigenvectors.data_handle(),
                                                           temp_v0.data_handle(),
                                                           config.seed);
  }
}

}  // namespace sparse::solver::detail
}  // namespace raft
