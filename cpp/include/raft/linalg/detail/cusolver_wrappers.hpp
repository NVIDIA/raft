/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/cusolver_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cusolverDn.h>
#include <cusolverSp.h>

#include <type_traits>

namespace raft {
namespace linalg {
namespace detail {

/**
 * @defgroup Getrf cusolver getrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle,
                                 int m,  // NOLINT
                                 int n,
                                 T* A,
                                 int lda,
                                 T* Workspace,
                                 int* devIpiv,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle,  // NOLINT
                                        int m,
                                        int n,
                                        float* A,
                                        int lda,
                                        float* Workspace,
                                        int* devIpiv,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <>
inline cusolverStatus_t cusolverDngetrf(cusolverDnHandle_t handle,  // NOLINT
                                        int m,
                                        int n,
                                        double* A,
                                        int lda,
                                        double* Workspace,
                                        int* devIpiv,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDgetrf(handle, m, n, A, lda, Workspace, devIpiv, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  T* A,
  int lda,
  int* Lwork);

template <>
inline cusolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return cusolverDnSgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

template <>
inline cusolverStatus_t cusolverDngetrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return cusolverDnDgetrf_bufferSize(handle, m, n, A, lda, Lwork);
}

/**
 * @defgroup Getrs cusolver getrs operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,  // NOLINT
                                 cublasOperation_t trans,
                                 int n,
                                 int nrhs,
                                 const T* A,
                                 int lda,
                                 const int* devIpiv,
                                 T* B,
                                 int ldb,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,  // NOLINT
                                        cublasOperation_t trans,
                                        int n,
                                        int nrhs,
                                        const float* A,
                                        int lda,
                                        const int* devIpiv,
                                        float* B,
                                        int ldb,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}

template <>
inline cusolverStatus_t cusolverDngetrs(cusolverDnHandle_t handle,  // NOLINT
                                        cublasOperation_t trans,
                                        int n,
                                        int nrhs,
                                        const double* A,
                                        int lda,
                                        const int* devIpiv,
                                        double* B,
                                        int ldb,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDgetrs(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);
}
/** @} */

/**
 * @defgroup syevd cusolver syevd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  const T* W,
  int* lwork);

template <>
inline cusolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  const float* W,
  int* lwork)
{
  return cusolverDnSsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}

template <>
inline cusolverStatus_t cusolverDnsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  const double* W,
  int* lwork)
{
  return cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork);
}
/** @} */

/**
 * @defgroup syevj cusolver syevj operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevj(cusolverDnHandle_t handle,  // NOLINT
                                 cusolverEigMode_t jobz,
                                 cublasFillMode_t uplo,
                                 int n,
                                 T* A,
                                 int lda,
                                 T* W,
                                 T* work,
                                 int lwork,
                                 int* info,
                                 syevjInfo_t params,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  float* W,
  float* work,
  int lwork,
  int* info,
  syevjInfo_t params,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

template <>
inline cusolverStatus_t cusolverDnsyevj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  double* W,
  double* work,
  int lwork,
  int* info,
  syevjInfo_t params,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevj(handle, jobz, uplo, n, A, lda, W, work, lwork, info, params);
}

template <typename T>
cusolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  const T* W,
  int* lwork,
  syevjInfo_t params);

template <>
inline cusolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  const float* W,
  int* lwork,
  syevjInfo_t params)
{
  return cusolverDnSsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
}

template <>
inline cusolverStatus_t cusolverDnsyevj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  const double* W,
  int* lwork,
  syevjInfo_t params)
{
  return cusolverDnDsyevj_bufferSize(handle, jobz, uplo, n, A, lda, W, lwork, params);
}
/** @} */

/**
 * @defgroup syevd cusolver syevd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,  // NOLINT
                                 cusolverEigMode_t jobz,
                                 cublasFillMode_t uplo,
                                 int n,
                                 T* A,
                                 int lda,
                                 T* W,
                                 T* work,
                                 int lwork,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,  // NOLINT
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo,
                                        int n,
                                        float* A,
                                        int lda,
                                        float* W,
                                        float* work,
                                        int lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnsyevd(cusolverDnHandle_t handle,  // NOLINT
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo,
                                        int n,
                                        double* A,
                                        int lda,
                                        double* W,
                                        double* work,
                                        int lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevd(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo);
}
/** @} */

/**
 * @defgroup syevdx cusolver syevdx operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  const T* A,
  int lda,
  T vl,
  T vu,
  int il,
  int iu,
  int* h_meig,
  const T* W,
  int* lwork);

template <>
inline cusolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  const float* A,
  int lda,
  float vl,
  float vu,
  int il,
  int iu,
  int* h_meig,
  const float* W,
  int* lwork)
{
  return cusolverDnSsyevdx_bufferSize(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, lwork);
}

template <>
inline cusolverStatus_t cusolverDnsyevdx_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  const double* A,
  int lda,
  double vl,
  double vu,
  int il,
  int iu,
  int* h_meig,
  const double* W,
  int* lwork)
{
  return cusolverDnDsyevdx_bufferSize(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, lwork);
}

template <typename T>
cusolverStatus_t cusolverDnsyevdx(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  T* A,
  int lda,
  T vl,
  T vu,
  int il,
  int iu,
  int* h_meig,
  T* W,
  T* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnsyevdx(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  float vl,
  float vu,
  int il,
  int iu,
  int* h_meig,
  float* W,
  float* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSsyevdx(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, work, lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnsyevdx(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  cusolverEigRange_t range,
  cublasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  double vl,
  double vu,
  int il,
  int iu,
  int* h_meig,
  double* W,
  double* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDsyevdx(
    handle, jobz, range, uplo, n, A, lda, vl, vu, il, iu, h_meig, W, work, lwork, devInfo);
}
/** @} */

/**
 * @defgroup svd cusolver svd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngesvd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int* lwork)
{
  if (std::is_same<std::decay_t<T>, float>::value) {
    return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
  } else {
    return cusolverDnDgesvd_bufferSize(handle, m, n, lwork);
  }
}
template <typename T>
cusolverStatus_t cusolverDngesvd(  // NOLINT
  cusolverDnHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  T* A,
  int lda,
  T* S,
  T* U,
  int ldu,
  T* VT,
  int ldvt,
  T* work,
  int lwork,
  T* rwork,
  int* devInfo,
  cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDngesvd(  // NOLINT
  cusolverDnHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  float* A,
  int lda,
  float* S,
  float* U,
  int ldu,
  float* VT,
  int ldvt,
  float* work,
  int lwork,
  float* rwork,
  int* devInfo,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSgesvd(
    handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDngesvd(  // NOLINT
  cusolverDnHandle_t handle,
  signed char jobu,
  signed char jobvt,
  int m,
  int n,
  double* A,
  int lda,
  double* S,
  double* U,
  int ldu,
  double* VT,
  int ldvt,
  double* work,
  int lwork,
  double* rwork,
  int* devInfo,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDgesvd(
    handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}

template <typename T>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const T* A,
  int lda,
  const T* S,
  const T* U,
  int ldu,
  const T* V,
  int ldv,
  int* lwork,
  gesvdjInfo_t params);
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const float* A,
  int lda,
  const float* S,
  const float* U,
  int ldu,
  const float* V,
  int ldv,
  int* lwork,
  gesvdjInfo_t params)
{
  return cusolverDnSgesvdj_bufferSize(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  const double* A,
  int lda,
  const double* S,
  const double* U,
  int ldu,
  const double* V,
  int ldv,
  int* lwork,
  gesvdjInfo_t params)
{
  return cusolverDnDgesvdj_bufferSize(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);
}
template <typename T>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  T* A,
  int lda,
  T* S,
  T* U,
  int ldu,
  T* V,
  int ldv,
  T* work,
  int lwork,
  int* info,
  gesvdjInfo_t params,
  cudaStream_t stream);
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  float* A,
  int lda,
  float* S,
  float* U,
  int ldu,
  float* V,
  int ldv,
  float* work,
  int lwork,
  int* info,
  gesvdjInfo_t params,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}
template <>
inline cusolverStatus_t CUSOLVERAPI cusolverDngesvdj(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverEigMode_t jobz,
  int econ,
  int m,
  int n,
  double* A,
  int lda,
  double* S,
  double* U,
  int ldu,
  double* V,
  int ldv,
  double* work,
  int lwork,
  int* info,
  gesvdjInfo_t params,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDgesvdj(
    handle, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork, info, params);
}

template <typename T>
cusolverStatus_t cusolverDnxgesvdr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  signed char jobu,
  signed char jobv,
  int64_t m,
  int64_t n,
  int64_t k,
  int64_t p,
  int64_t niters,
  const T* a,
  int64_t lda,
  const T* Srand,
  const T* Urand,
  int64_t ldUrand,
  const T* Vrand,
  int64_t ldVrand,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  cudaStream_t stream)
{
  RAFT_EXPECTS(std::is_floating_point_v<T>, "Unsupported data type");
  cudaDataType dataType = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  cusolverDnParams_t dn_params = nullptr;
  RAFT_CUSOLVER_TRY(cusolverDnCreateParams(&dn_params));
  auto result = cusolverDnXgesvdr_bufferSize(handle,
                                             dn_params,
                                             jobu,
                                             jobv,
                                             m,
                                             n,
                                             k,
                                             p,
                                             niters,
                                             dataType,
                                             a,
                                             lda,
                                             dataType,
                                             Srand,
                                             dataType,
                                             Urand,
                                             ldUrand,
                                             dataType,
                                             Vrand,
                                             ldVrand,
                                             dataType,
                                             workspaceInBytesOnDevice,
                                             workspaceInBytesOnHost);
  RAFT_CUSOLVER_TRY(cusolverDnDestroyParams(dn_params));
  return result;
}
template <typename T>
cusolverStatus_t cusolverDnxgesvdr(  // NOLINT
  cusolverDnHandle_t handle,
  signed char jobu,
  signed char jobv,
  int64_t m,
  int64_t n,
  int64_t k,
  int64_t p,
  int64_t niters,
  T* a,
  int64_t lda,
  T* Srand,
  T* Urand,
  int64_t ldUrand,
  T* Vrand,
  int64_t ldVrand,
  void* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  void* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* d_info,
  cudaStream_t stream)
{
  cudaDataType dataType = std::is_same_v<T, float> ? CUDA_R_32F : CUDA_R_64F;
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  cusolverDnParams_t dn_params = nullptr;
  RAFT_CUSOLVER_TRY(cusolverDnCreateParams(&dn_params));
  auto result = cusolverDnXgesvdr(handle,
                                  dn_params,
                                  jobu,
                                  jobv,
                                  m,
                                  n,
                                  k,
                                  p,
                                  niters,
                                  dataType,
                                  a,
                                  lda,
                                  dataType,
                                  Srand,
                                  dataType,
                                  Urand,
                                  ldUrand,
                                  dataType,
                                  Vrand,
                                  ldVrand,
                                  dataType,
                                  bufferOnDevice,
                                  workspaceInBytesOnDevice,
                                  bufferOnHost,
                                  workspaceInBytesOnHost,
                                  d_info);
  RAFT_CUSOLVER_TRY(cusolverDnDestroyParams(dn_params));
  return result;
}

/** @} */

/**
 * @defgroup potrf cusolver potrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasFillMode_t uplo,
  int n,
  T* A,
  int lda,
  int* Lwork);

template <>
inline cusolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasFillMode_t uplo,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return cusolverDnSpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

template <>
inline cusolverStatus_t cusolverDnpotrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasFillMode_t uplo,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return cusolverDnDpotrf_bufferSize(handle, uplo, n, A, lda, Lwork);
}

template <typename T>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        T* A,
                                        int lda,
                                        T* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        float* A,
                                        int lda,
                                        float* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnpotrf(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        double* A,
                                        int lda,
                                        double* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDpotrf(handle, uplo, n, A, lda, Workspace, Lwork, devInfo);
}
/** @} */

/**
 * @defgroup potrfBatched cusolver potrfBatched operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnpotrfBatched(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        T* Aarray[],  // NOLINT
                                        int lda,
                                        int* infoArray,
                                        int batchSize,
                                        cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnpotrfBatched(cusolverDnHandle_t handle,  // NOLINT
                                               cublasFillMode_t uplo,
                                               int n,
                                               float* Aarray[],  // NOLINT
                                               int lda,
                                               int* infoArray,
                                               int batchSize,
                                               cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}

template <>
inline cusolverStatus_t cusolverDnpotrfBatched(cusolverDnHandle_t handle,  // NOLINT
                                               cublasFillMode_t uplo,
                                               int n,
                                               double* Aarray[],  // NOLINT
                                               int lda,
                                               int* infoArray,
                                               int batchSize,
                                               cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, infoArray, batchSize);
}
/** @} */

/**
 * @defgroup potrs cusolver potrs operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,  // NOLINT
                                 cublasFillMode_t uplo,
                                 int n,
                                 int nrhs,
                                 const T* A,
                                 int lda,
                                 T* B,
                                 int ldb,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        int nrhs,
                                        const float* A,
                                        int lda,
                                        float* B,
                                        int ldb,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnpotrs(cusolverDnHandle_t handle,  // NOLINT
                                        cublasFillMode_t uplo,
                                        int n,
                                        int nrhs,
                                        const double* A,
                                        int lda,
                                        double* B,
                                        int ldb,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDpotrs(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo);
}
/** @} */

/**
 * @defgroup geqrf cusolver geqrf operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle,
                                 int m,  // NOLINT
                                 int n,
                                 T* A,
                                 int lda,
                                 T* TAU,
                                 T* Workspace,
                                 int Lwork,
                                 int* devInfo,
                                 cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle,  // NOLINT
                                        int m,
                                        int n,
                                        float* A,
                                        int lda,
                                        float* TAU,
                                        float* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDngeqrf(cusolverDnHandle_t handle,  // NOLINT
                                        int m,
                                        int n,
                                        double* A,
                                        int lda,
                                        double* TAU,
                                        double* Workspace,
                                        int Lwork,
                                        int* devInfo,
                                        cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDgeqrf(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  T* A,
  int lda,
  int* Lwork);
template <>
inline cusolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  float* A,
  int lda,
  int* Lwork)
{
  return cusolverDnSgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
template <>
inline cusolverStatus_t cusolverDngeqrf_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  double* A,
  int lda,
  int* Lwork)
{
  return cusolverDnDgeqrf_bufferSize(handle, m, n, A, lda, Lwork);
}
/** @} */

/**
 * @defgroup orgqr cusolver orgqr operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnorgqr(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  T* A,
  int lda,
  const T* tau,
  T* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream);
template <>
inline cusolverStatus_t cusolverDnorgqr(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  float* A,
  int lda,
  const float* tau,
  float* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}
template <>
inline cusolverStatus_t cusolverDnorgqr(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  double* A,
  int lda,
  const double* tau,
  double* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDorgqr(handle, m, n, k, A, lda, tau, work, lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  const T* A,
  int lda,
  const T* TAU,
  int* lwork);
template <>
inline cusolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* TAU,
  int* lwork)
{
  return cusolverDnSorgqr_bufferSize(handle, m, n, k, A, lda, TAU, lwork);
}
template <>
inline cusolverStatus_t cusolverDnorgqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* TAU,
  int* lwork)
{
  return cusolverDnDorgqr_bufferSize(handle, m, n, k, A, lda, TAU, lwork);
}
/** @} */

/**
 * @defgroup ormqr cusolver ormqr operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnormqr(cusolverDnHandle_t handle,  // NOLINT
                                 cublasSideMode_t side,
                                 cublasOperation_t trans,
                                 int m,
                                 int n,
                                 int k,
                                 const T* A,
                                 int lda,
                                 const T* tau,
                                 T* C,
                                 int ldc,
                                 T* work,
                                 int lwork,
                                 int* devInfo,
                                 cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnormqr(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* tau,
  float* C,
  int ldc,
  float* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnSormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

template <>
inline cusolverStatus_t cusolverDnormqr(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* tau,
  double* C,
  int ldc,
  double* work,
  int lwork,
  int* devInfo,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnDormqr(handle, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
}

template <typename T>
cusolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const T* A,
  int lda,
  const T* tau,
  const T* C,
  int ldc,
  int* lwork);

template <>
inline cusolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const float* A,
  int lda,
  const float* tau,
  const float* C,
  int ldc,
  int* lwork)
{
  return cusolverDnSormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}

template <>
inline cusolverStatus_t cusolverDnormqr_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cublasSideMode_t side,
  cublasOperation_t trans,
  int m,
  int n,
  int k,
  const double* A,
  int lda,
  const double* tau,
  const double* C,
  int ldc,
  int* lwork)
{
  return cusolverDnDormqr_bufferSize(handle, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);
}
/** @} */

/**
 * @defgroup csrqrBatched cusolver batched
 * @{
 */
template <typename T>
cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const T* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes);

template <>
inline cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const float* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes)
{
  return cusolverSpScsrqrBufferInfoBatched(handle,
                                           m,
                                           n,
                                           nnzA,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           batchSize,
                                           info,
                                           internalDataInBytes,
                                           workspaceInBytes);
}

template <>
inline cusolverStatus_t cusolverSpcsrqrBufferInfoBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const double* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  int batchSize,
  csrqrInfo_t info,
  size_t* internalDataInBytes,
  size_t* workspaceInBytes)
{
  return cusolverSpDcsrqrBufferInfoBatched(handle,
                                           m,
                                           n,
                                           nnzA,
                                           descrA,
                                           csrValA,
                                           csrRowPtrA,
                                           csrColIndA,
                                           batchSize,
                                           info,
                                           internalDataInBytes,
                                           workspaceInBytes);
}

template <typename T>
cusolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const T* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const T* b,
  T* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const float* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const float* b,
  float* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverSpSetStream(handle, stream));
  return cusolverSpScsrqrsvBatched(
    handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}

template <>
inline cusolverStatus_t cusolverSpcsrqrsvBatched(  // NOLINT
  cusolverSpHandle_t handle,
  int m,
  int n,
  int nnzA,
  const cusparseMatDescr_t descrA,
  const double* csrValA,
  const int* csrRowPtrA,
  const int* csrColIndA,
  const double* b,
  double* x,
  int batchSize,
  csrqrInfo_t info,
  void* pBuffer,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverSpSetStream(handle, stream));
  return cusolverSpDcsrqrsvBatched(
    handle, m, n, nnzA, descrA, csrValA, csrRowPtrA, csrColIndA, b, x, batchSize, info, pBuffer);
}
/** @} */

/**
 * @defgroup DnXsyevd cusolver DnXsyevd operations
 * @{
 */
template <typename T>
cusolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  const T* A,
  int64_t lda,
  const T* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  const float* A,
  int64_t lda,
  const float* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnXsyevd_bufferSize(handle,
                                     params,
                                     jobz,
                                     uplo,
                                     n,
                                     CUDA_R_32F,
                                     A,
                                     lda,
                                     CUDA_R_32F,
                                     W,
                                     CUDA_R_32F,
                                     workspaceInBytesOnDevice,
                                     workspaceInBytesOnHost);
}

template <>
inline cusolverStatus_t cusolverDnxsyevd_bufferSize(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  const double* A,
  int64_t lda,
  const double* W,
  size_t* workspaceInBytesOnDevice,
  size_t* workspaceInBytesOnHost,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnXsyevd_bufferSize(handle,
                                     params,
                                     jobz,
                                     uplo,
                                     n,
                                     CUDA_R_64F,
                                     A,
                                     lda,
                                     CUDA_R_64F,
                                     W,
                                     CUDA_R_64F,
                                     workspaceInBytesOnDevice,
                                     workspaceInBytesOnHost);
}

template <typename T>
cusolverStatus_t cusolverDnxsyevd(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  T* A,
  int64_t lda,
  T* W,
  T* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  T* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  cudaStream_t stream);

template <>
inline cusolverStatus_t cusolverDnxsyevd(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  float* A,
  int64_t lda,
  float* W,
  float* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  float* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnXsyevd(handle,
                          params,
                          jobz,
                          uplo,
                          n,
                          CUDA_R_32F,
                          A,
                          lda,
                          CUDA_R_32F,
                          W,
                          CUDA_R_32F,
                          bufferOnDevice,
                          workspaceInBytesOnDevice,
                          bufferOnHost,
                          workspaceInBytesOnHost,
                          info);
}

template <>
inline cusolverStatus_t cusolverDnxsyevd(  // NOLINT
  cusolverDnHandle_t handle,
  cusolverDnParams_t params,
  cusolverEigMode_t jobz,
  cublasFillMode_t uplo,
  int64_t n,
  double* A,
  int64_t lda,
  double* W,
  double* bufferOnDevice,
  size_t workspaceInBytesOnDevice,
  double* bufferOnHost,
  size_t workspaceInBytesOnHost,
  int* info,
  cudaStream_t stream)
{
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(handle, stream));
  return cusolverDnXsyevd(handle,
                          params,
                          jobz,
                          uplo,
                          n,
                          CUDA_R_64F,
                          A,
                          lda,
                          CUDA_R_64F,
                          W,
                          CUDA_R_64F,
                          bufferOnDevice,
                          workspaceInBytesOnDevice,
                          bufferOnHost,
                          workspaceInBytesOnHost,
                          info);
}
/** @} */

}  // namespace detail
}  // namespace linalg
}  // namespace raft
