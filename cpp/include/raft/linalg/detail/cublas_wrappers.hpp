/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/cublas_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>

#include <cublas_v2.h>

#include <cstdint>

namespace raft {
namespace linalg {
namespace detail {

/**
 * Assuming the default CUBLAS_POINTER_MODE_HOST, change it to host or device mode
 * temporary for the lifetime of this object.
 */
template <bool DevicePointerMode = false>
class cublas_device_pointer_mode {
 public:
  explicit cublas_device_pointer_mode(cublasHandle_t handle) : handle_(handle)
  {
    if constexpr (DevicePointerMode) {
      RAFT_CUBLAS_TRY(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE));
    }
  }
  auto operator=(const cublas_device_pointer_mode&) -> cublas_device_pointer_mode& = delete;
  auto operator=(cublas_device_pointer_mode&&) -> cublas_device_pointer_mode&      = delete;
  static auto operator new(std::size_t) -> void*                                   = delete;
  static auto operator new[](std::size_t) -> void*                                 = delete;

  ~cublas_device_pointer_mode()
  {
    if constexpr (DevicePointerMode) {
      RAFT_CUBLAS_TRY_NO_THROW(cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_HOST));
    }
  }

 private:
  cublasHandle_t handle_ = nullptr;
};

/**
 * @defgroup Axpy cublas ax+y operations
 * @{
 */
template <typename T>
cublasStatus_t cublasaxpy(cublasHandle_t handle,
                          int n,
                          const T* alpha,
                          const T* x,
                          int incx,
                          T* y,
                          int incy,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublasaxpy(cublasHandle_t handle,
                                 int n,
                                 const float* alpha,
                                 const float* x,
                                 int incx,
                                 float* y,
                                 int incy,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasaxpy(cublasHandle_t handle,
                                 int n,
                                 const double* alpha,
                                 const double* x,
                                 int incx,
                                 double* y,
                                 int incy,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}
/** @} */

/**
 * @defgroup cublas swap operations
 * @{
 */
template <typename T>
cublasStatus_t cublasSwap(
  cublasHandle_t handle, int n, T* x, int incx, T* y, int incy, cudaStream_t stream);

template <>
inline cublasStatus_t cublasSwap(
  cublasHandle_t handle, int n, float* x, int incx, float* y, int incy, cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSswap(handle, n, x, incx, y, incy);
}

template <>
inline cublasStatus_t cublasSwap(
  cublasHandle_t handle, int n, double* x, int incx, double* y, int incy, cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDswap(handle, n, x, incx, y, incy);
}

/** @} */

/**
 * @defgroup cublas copy operations
 * @{
 */
template <typename T>
cublasStatus_t cublasCopy(
  cublasHandle_t handle, int n, const T* x, int incx, T* y, int incy, cudaStream_t stream);

template <>
inline cublasStatus_t cublasCopy(
  cublasHandle_t handle, int n, const float* x, int incx, float* y, int incy, cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasScopy(handle, n, x, incx, y, incy);
}
template <>
inline cublasStatus_t cublasCopy(
  cublasHandle_t handle, int n, const double* x, int incx, double* y, int incy, cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDcopy(handle, n, x, incx, y, incy);
}
/** @} */

/**
 * @defgroup gemv cublas gemv calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgemv(cublasHandle_t handle,
                          cublasOperation_t transA,
                          int m,
                          int n,
                          const T* alfa,
                          const T* A,
                          int lda,
                          const T* x,
                          int incx,
                          const T* beta,
                          T* y,
                          int incy,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublasgemv(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 int m,
                                 int n,
                                 const float* alfa,
                                 const float* A,
                                 int lda,
                                 const float* x,
                                 int incx,
                                 const float* beta,
                                 float* y,
                                 int incy,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSgemv(handle, transA, m, n, alfa, A, lda, x, incx, beta, y, incy);
}

template <>
inline cublasStatus_t cublasgemv(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 int m,
                                 int n,
                                 const double* alfa,
                                 const double* A,
                                 int lda,
                                 const double* x,
                                 int incx,
                                 const double* beta,
                                 double* y,
                                 int incy,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDgemv(handle, transA, m, n, alfa, A, lda, x, incx, beta, y, incy);
}
/** @} */

/**
 * @defgroup ger cublas a(x*y.T) + A calls
 * @{
 */
template <typename T>
cublasStatus_t cublasger(cublasHandle_t handle,
                         int m,
                         int n,
                         const T* alpha,
                         const T* x,
                         int incx,
                         const T* y,
                         int incy,
                         T* A,
                         int lda,
                         cudaStream_t stream);
template <>
inline cublasStatus_t cublasger(cublasHandle_t handle,
                                int m,
                                int n,
                                const float* alpha,
                                const float* x,
                                int incx,
                                const float* y,
                                int incy,
                                float* A,
                                int lda,
                                cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline cublasStatus_t cublasger(cublasHandle_t handle,
                                int m,
                                int n,
                                const double* alpha,
                                const double* x,
                                int incx,
                                const double* y,
                                int incy,
                                double* A,
                                int lda,
                                cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
/** @} */

/**
 * @defgroup gemm cublas gemm calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgemm(cublasHandle_t handle,
                          cublasOperation_t transA,
                          cublasOperation_t transB,
                          int m,
                          int n,
                          int k,
                          const T* alfa,
                          const T* A,
                          int lda,
                          const T* B,
                          int ldb,
                          const T* beta,
                          T* C,
                          int ldc,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublasgemm(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 cublasOperation_t transB,
                                 int m,
                                 int n,
                                 int k,
                                 const float* alfa,
                                 const float* A,
                                 int lda,
                                 const float* B,
                                 int ldb,
                                 const float* beta,
                                 float* C,
                                 int ldc,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSgemm(handle, transA, transB, m, n, k, alfa, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline cublasStatus_t cublasgemm(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 cublasOperation_t transB,
                                 int m,
                                 int n,
                                 int k,
                                 const double* alfa,
                                 const double* A,
                                 int lda,
                                 const double* B,
                                 int ldb,
                                 const double* beta,
                                 double* C,
                                 int ldc,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDgemm(handle, transA, transB, m, n, k, alfa, A, lda, B, ldb, beta, C, ldc);
}
/** @} */

/**
 * @defgroup gemmbatched cublas gemmbatched calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgemmBatched(cublasHandle_t handle,  // NOLINT
                                 cublasOperation_t transa,
                                 cublasOperation_t transb,
                                 int m,
                                 int n,
                                 int k,
                                 const T* alpha,
                                 const T* const Aarray[],  // NOLINT
                                 int lda,
                                 const T* const Barray[],  // NOLINT
                                 int ldb,
                                 const T* beta,
                                 T* Carray[],  // NOLINT
                                 int ldc,
                                 int batchCount,
                                 cudaStream_t stream);

template <>
inline cublasStatus_t cublasgemmBatched(  // NOLINT
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m,
  int n,
  int k,
  const float* alpha,
  const float* const Aarray[],  // NOLINT
  int lda,
  const float* const Barray[],  // NOLINT
  int ldb,
  const float* beta,
  float* Carray[],  // NOLINT
  int ldc,
  int batchCount,
  cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSgemmBatched(handle,
                            transa,
                            transb,
                            m,
                            n,
                            k,
                            alpha,
                            Aarray,
                            lda,
                            Barray,
                            ldb,
                            beta,
                            Carray,
                            ldc,
                            batchCount);
}

template <>
inline cublasStatus_t cublasgemmBatched(  // NOLINT
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m,
  int n,
  int k,
  const double* alpha,
  const double* const Aarray[],  // NOLINT
  int lda,
  const double* const Barray[],  // NOLINT
  int ldb,
  const double* beta,
  double* Carray[],  // NOLINT
  int ldc,
  int batchCount,
  cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDgemmBatched(handle,
                            transa,
                            transb,
                            m,
                            n,
                            k,
                            alpha,
                            Aarray,
                            lda,
                            Barray,
                            ldb,
                            beta,
                            Carray,
                            ldc,
                            batchCount);
}
/** @} */

/**
 * @defgroup gemmbatched cublas gemmbatched calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgemmStridedBatched(  // NOLINT
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m,
  int n,
  int k,
  const T* alpha,
  const T* const Aarray,
  int lda,
  int64_t strideA,
  const T* const Barray,
  int ldb,
  int64_t strideB,
  const T* beta,
  T* Carray,
  int ldc,
  int64_t strideC,
  int batchCount,
  cudaStream_t stream);

template <>
inline cublasStatus_t cublasgemmStridedBatched(  // NOLINT
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m,
  int n,
  int k,
  const float* alpha,
  const float* const Aarray,
  int lda,
  int64_t strideA,
  const float* const Barray,
  int ldb,
  int64_t strideB,
  const float* beta,
  float* Carray,
  int ldc,
  int64_t strideC,
  int batchCount,
  cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSgemmStridedBatched(handle,
                                   transa,
                                   transb,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   Aarray,
                                   lda,
                                   strideA,
                                   Barray,
                                   ldb,
                                   strideB,
                                   beta,
                                   Carray,
                                   ldc,
                                   strideC,
                                   batchCount);
}

template <>
inline cublasStatus_t cublasgemmStridedBatched(  // NOLINT
  cublasHandle_t handle,
  cublasOperation_t transa,
  cublasOperation_t transb,
  int m,
  int n,
  int k,
  const double* alpha,
  const double* const Aarray,
  int lda,
  int64_t strideA,
  const double* const Barray,
  int ldb,
  int64_t strideB,
  const double* beta,
  double* Carray,
  int ldc,
  int64_t strideC,
  int batchCount,
  cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDgemmStridedBatched(handle,
                                   transa,
                                   transb,
                                   m,
                                   n,
                                   k,
                                   alpha,
                                   Aarray,
                                   lda,
                                   strideA,
                                   Barray,
                                   ldb,
                                   strideB,
                                   beta,
                                   Carray,
                                   ldc,
                                   strideC,
                                   batchCount);
}
/** @} */

/**
 * @defgroup solverbatched cublas getrf/gettribatched calls
 * @{
 */

template <typename T>
cublasStatus_t cublasgetrfBatched(cublasHandle_t handle,
                                  int n,         // NOLINT
                                  T* const A[],  // NOLINT
                                  int lda,
                                  int* P,
                                  int* info,
                                  int batchSize,
                                  cudaStream_t stream);

template <>
inline cublasStatus_t cublasgetrfBatched(cublasHandle_t handle,  // NOLINT
                                         int n,
                                         float* const A[],  // NOLINT
                                         int lda,
                                         int* P,
                                         int* info,
                                         int batchSize,
                                         cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

template <>
inline cublasStatus_t cublasgetrfBatched(cublasHandle_t handle,  // NOLINT
                                         int n,
                                         double* const A[],  // NOLINT
                                         int lda,
                                         int* P,
                                         int* info,
                                         int batchSize,
                                         cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

template <typename T>
cublasStatus_t cublasgetriBatched(cublasHandle_t handle,
                                  int n,               // NOLINT
                                  const T* const A[],  // NOLINT
                                  int lda,
                                  const int* P,
                                  T* const C[],  // NOLINT
                                  int ldc,
                                  int* info,
                                  int batchSize,
                                  cudaStream_t stream);

template <>
inline cublasStatus_t cublasgetriBatched(  // NOLINT
  cublasHandle_t handle,
  int n,
  const float* const A[],  // NOLINT
  int lda,
  const int* P,
  float* const C[],  // NOLINT
  int ldc,
  int* info,
  int batchSize,
  cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
}

template <>
inline cublasStatus_t cublasgetriBatched(  // NOLINT
  cublasHandle_t handle,
  int n,
  const double* const A[],  // NOLINT
  int lda,
  const int* P,
  double* const C[],  // NOLINT
  int ldc,
  int* info,
  int batchSize,
  cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
}

/** @} */

/**
 * @defgroup gelsbatched cublas gelsbatched calls
 * @{
 */

template <typename T>
inline cublasStatus_t cublasgelsBatched(cublasHandle_t handle,  // NOLINT
                                        cublasOperation_t trans,
                                        int m,
                                        int n,
                                        int nrhs,
                                        T* Aarray[],  // NOLINT
                                        int lda,
                                        T* Carray[],  // NOLINT
                                        int ldc,
                                        int* info,
                                        int* devInfoArray,
                                        int batchSize,
                                        cudaStream_t stream);

template <>
inline cublasStatus_t cublasgelsBatched(cublasHandle_t handle,  // NOLINT
                                        cublasOperation_t trans,
                                        int m,
                                        int n,
                                        int nrhs,
                                        float* Aarray[],  // NOLINT
                                        int lda,
                                        float* Carray[],  // NOLINT
                                        int ldc,
                                        int* info,
                                        int* devInfoArray,
                                        int batchSize,
                                        cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSgelsBatched(
    handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

template <>
inline cublasStatus_t cublasgelsBatched(cublasHandle_t handle,  // NOLINT
                                        cublasOperation_t trans,
                                        int m,
                                        int n,
                                        int nrhs,
                                        double* Aarray[],  // NOLINT
                                        int lda,
                                        double* Carray[],  // NOLINT
                                        int ldc,
                                        int* info,
                                        int* devInfoArray,
                                        int batchSize,
                                        cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDgelsBatched(
    handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc, info, devInfoArray, batchSize);
}

/** @} */

/**
 * @defgroup geam cublas geam calls
 * @{
 */
template <typename T>
cublasStatus_t cublasgeam(cublasHandle_t handle,
                          cublasOperation_t transA,
                          cublasOperation_t transB,
                          int m,
                          int n,
                          const T* alfa,
                          const T* A,
                          int lda,
                          const T* beta,
                          const T* B,
                          int ldb,
                          T* C,
                          int ldc,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublasgeam(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 cublasOperation_t transB,
                                 int m,
                                 int n,
                                 const float* alfa,
                                 const float* A,
                                 int lda,
                                 const float* beta,
                                 const float* B,
                                 int ldb,
                                 float* C,
                                 int ldc,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSgeam(handle, transA, transB, m, n, alfa, A, lda, beta, B, ldb, C, ldc);
}

template <>
inline cublasStatus_t cublasgeam(cublasHandle_t handle,
                                 cublasOperation_t transA,
                                 cublasOperation_t transB,
                                 int m,
                                 int n,
                                 const double* alfa,
                                 const double* A,
                                 int lda,
                                 const double* beta,
                                 const double* B,
                                 int ldb,
                                 double* C,
                                 int ldc,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDgeam(handle, transA, transB, m, n, alfa, A, lda, beta, B, ldb, C, ldc);
}
/** @} */

/**
 * @defgroup symm cublas symm calls
 * @{
 */
template <typename T>
cublasStatus_t cublassymm(cublasHandle_t handle,
                          cublasSideMode_t side,
                          cublasFillMode_t uplo,
                          int m,
                          int n,
                          const T* alpha,
                          const T* A,
                          int lda,
                          const T* B,
                          int ldb,
                          const T* beta,
                          T* C,
                          int ldc,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublassymm(cublasHandle_t handle,
                                 cublasSideMode_t side,
                                 cublasFillMode_t uplo,
                                 int m,
                                 int n,
                                 const float* alpha,
                                 const float* A,
                                 int lda,
                                 const float* B,
                                 int ldb,
                                 const float* beta,
                                 float* C,
                                 int ldc,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline cublasStatus_t cublassymm(cublasHandle_t handle,
                                 cublasSideMode_t side,
                                 cublasFillMode_t uplo,
                                 int m,
                                 int n,
                                 const double* alpha,
                                 const double* A,
                                 int lda,
                                 const double* B,
                                 int ldb,
                                 const double* beta,
                                 double* C,
                                 int ldc,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
}
/** @} */

/**
 * @defgroup syrk cublas syrk calls
 * @{
 */
template <typename T>
cublasStatus_t cublassyrk(cublasHandle_t handle,
                          cublasFillMode_t uplo,
                          cublasOperation_t trans,
                          int n,
                          int k,
                          const T* alpha,
                          const T* A,
                          int lda,
                          const T* beta,
                          T* C,
                          int ldc,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublassyrk(cublasHandle_t handle,
                                 cublasFillMode_t uplo,
                                 cublasOperation_t trans,
                                 int n,
                                 int k,
                                 const float* alpha,
                                 const float* A,
                                 int lda,
                                 const float* beta,
                                 float* C,
                                 int ldc,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
inline cublasStatus_t cublassyrk(cublasHandle_t handle,
                                 cublasFillMode_t uplo,
                                 cublasOperation_t trans,
                                 int n,
                                 int k,
                                 const double* alpha,
                                 const double* A,
                                 int lda,
                                 const double* beta,
                                 double* C,
                                 int ldc,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
/** @} */

/**
 * @defgroup nrm2 cublas nrm2 calls
 * @{
 */
template <typename T>
cublasStatus_t cublasnrm2(
  cublasHandle_t handle, int n, const T* x, int incx, T* result, cudaStream_t stream);

template <>
inline cublasStatus_t cublasnrm2(
  cublasHandle_t handle, int n, const float* x, int incx, float* result, cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSnrm2(handle, n, x, incx, result);
}

template <>
inline cublasStatus_t cublasnrm2(
  cublasHandle_t handle, int n, const double* x, int incx, double* result, cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDnrm2(handle, n, x, incx, result);
}
/** @} */

template <typename T>
cublasStatus_t cublastrsm(cublasHandle_t handle,
                          cublasSideMode_t side,
                          cublasFillMode_t uplo,
                          cublasOperation_t trans,
                          cublasDiagType_t diag,
                          int m,
                          int n,
                          const T* alpha,
                          const T* A,
                          int lda,
                          T* B,
                          int ldb,
                          cudaStream_t stream);

template <>
inline cublasStatus_t cublastrsm(cublasHandle_t handle,
                                 cublasSideMode_t side,
                                 cublasFillMode_t uplo,
                                 cublasOperation_t trans,
                                 cublasDiagType_t diag,
                                 int m,
                                 int n,
                                 const float* alpha,
                                 const float* A,
                                 int lda,
                                 float* B,
                                 int ldb,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

template <>
inline cublasStatus_t cublastrsm(cublasHandle_t handle,
                                 cublasSideMode_t side,
                                 cublasFillMode_t uplo,
                                 cublasOperation_t trans,
                                 cublasDiagType_t diag,
                                 int m,
                                 int n,
                                 const double* alpha,
                                 const double* A,
                                 int lda,
                                 double* B,
                                 int ldb,
                                 cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
}

/**
 * @defgroup trsmBatched cublas trsmBatched calls
 * @{
 */
template <typename T>
cublasStatus_t cublastrsmBatched(cublasHandle_t handle,  // NOLINT
                                 cublasSideMode_t side,
                                 cublasFillMode_t uplo,
                                 cublasOperation_t trans,
                                 cublasDiagType_t diag,
                                 int m,
                                 int n,
                                 const T* alpha,
                                 const T* const Aarray[],  // NOLINT
                                 int lda,
                                 T* const Barray[],  // NOLINT
                                 int ldb,
                                 int batchCount,
                                 cudaStream_t stream);

template <>
inline cublasStatus_t cublastrsmBatched(cublasHandle_t handle,  // NOLINT
                                        cublasSideMode_t side,
                                        cublasFillMode_t uplo,
                                        cublasOperation_t trans,
                                        cublasDiagType_t diag,
                                        int m,
                                        int n,
                                        const float* alpha,
                                        const float* const Aarray[],  // NOLINT
                                        int lda,
                                        float* const Barray[],  // NOLINT
                                        int ldb,
                                        int batchCount,
                                        cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasStrsmBatched(
    handle, side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount);
}

template <>
inline cublasStatus_t cublastrsmBatched(cublasHandle_t handle,  // NOLINT
                                        cublasSideMode_t side,
                                        cublasFillMode_t uplo,
                                        cublasOperation_t trans,
                                        cublasDiagType_t diag,
                                        int m,
                                        int n,
                                        const double* alpha,
                                        const double* const Aarray[],  // NOLINT
                                        int lda,
                                        double* const Barray[],  // NOLINT
                                        int ldb,
                                        int batchCount,
                                        cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDtrsmBatched(
    handle, side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount);
}
/** @} */

/**
 * @defgroup dot cublas dot calls
 * @{
 */
template <typename T>
cublasStatus_t cublasdot(cublasHandle_t handle,
                         int n,
                         const T* x,
                         int incx,
                         const T* y,
                         int incy,
                         T* result,
                         cudaStream_t stream);

template <>
inline cublasStatus_t cublasdot(cublasHandle_t handle,
                                int n,
                                const float* x,
                                int incx,
                                const float* y,
                                int incy,
                                float* result,
                                cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDotEx(
    handle, n, x, CUDA_R_32F, incx, y, CUDA_R_32F, incy, result, CUDA_R_32F, CUDA_R_32F);
}

template <>
inline cublasStatus_t cublasdot(cublasHandle_t handle,
                                int n,
                                const double* x,
                                int incx,
                                const double* y,
                                int incy,
                                double* result,
                                cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDotEx(
    handle, n, x, CUDA_R_64F, incx, y, CUDA_R_64F, incy, result, CUDA_R_64F, CUDA_R_64F);
}
/** @} */

/**
 * @defgroup setpointermode cublas set pointer mode method
 * @{
 */
// no T dependency...
// template <typename T>
// cublasStatus_t cublassetpointermode(  // NOLINT
//                                         cublasHandle_t  handle,
//                                         cublasPointerMode_t mode,
//                                         cudaStream_t stream);

// template<>
inline cublasStatus_t cublassetpointermode(cublasHandle_t handle,
                                           cublasPointerMode_t mode,
                                           cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSetPointerMode(handle, mode);
}
/** @} */

/**
 * @defgroup scal cublas dot calls
 * @{
 */
template <typename T>
cublasStatus_t cublasscal(
  cublasHandle_t handle, int n, const T* alpha, T* x, int incx, cudaStream_t stream);

template <>
inline cublasStatus_t cublasscal(
  cublasHandle_t handle, int n, const float* alpha, float* x, int incx, cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasSscal(handle, n, alpha, x, incx);
}

template <>
inline cublasStatus_t cublasscal(
  cublasHandle_t handle, int n, const double* alpha, double* x, int incx, cudaStream_t stream)
{
  RAFT_CUBLAS_TRY(cublasSetStream(handle, stream));
  return cublasDscal(handle, n, alpha, x, incx);
}

/** @} */

}  // namespace detail
}  // namespace linalg
}  // namespace raft
