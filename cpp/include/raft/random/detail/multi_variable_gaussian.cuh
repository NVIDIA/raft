/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include "curand_wrappers.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/random_types.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cmath>
#include <cstdio>
#include <memory>
#include <optional>
#include <type_traits>

// mvg.cuh takes in matrices that are column major (as in fortran)
#define IDX2C(i, j, ld) (j * ld + i)

namespace raft {
namespace random {
namespace detail {

enum Filler : unsigned char {
  LOWER,  // = 0
  UPPER   // = 1
};  // used in memseting upper/lower matrix

/**
 * @brief Reset values within the epsilon absolute range to zero
 * @tparam T the data type
 * @param eig the array
 * @param epsilon the range
 * @param size length of the array
 * @param stream cuda stream
 */
template <typename T>
void epsilonToZero(T* eig, T epsilon, int size, cudaStream_t stream)
{
  raft::linalg::unaryOp(
    eig,
    eig,
    size,
    [epsilon] __device__(T in) { return (in < epsilon && in > -epsilon) ? T(0.0) : in; },
    stream);
}

/**
 * @brief Broadcast addition of vector onto a matrix
 * @tparam the data type
 * @param out the output matrix
 * @param in_m the input matrix
 * @param in_v the input vector
 * @param scalar scalar multiplier
 * @param rows number of rows in the input matrix
 * @param cols number of cols in the input matrix
 * @param stream cuda stream
 */
template <typename T>
void matVecAdd(
  T* out, const T* in_m, const T* in_v, T scalar, int rows, int cols, cudaStream_t stream)
{
  raft::linalg::detail::matrixVectorOp<true, true>(
    false,
    out,
    in_m,
    in_v,
    cols,
    rows,
    [=] __device__(T mat, T vec) { return mat + scalar * vec; },
    stream);
}

// helper kernels
template <typename T>
RAFT_KERNEL combined_dot_product(int rows, int cols, const T* W, T* matrix, int* check)
{
  int m_i = threadIdx.x + blockDim.x * blockIdx.x;
  int Wi  = m_i / cols;
  if (m_i < cols * rows) {
    if (W[Wi] >= 0.0)
      matrix[m_i] = pow(W[Wi], 0.5) * (matrix[m_i]);
    else
      check[0] = Wi;  // reports Wi'th eigen values is negative.
  }
}

template <typename T>  // if uplo = 0, lower part of dim x dim matrix set to
// value
RAFT_KERNEL fill_uplo(int dim, Filler uplo, T value, T* A)
{
  int j = threadIdx.x + blockDim.x * blockIdx.x;
  int i = threadIdx.y + blockDim.y * blockIdx.y;
  if (i < dim && j < dim) {
    // making off-diagonals == value
    if (i < j) {
      if (uplo == 1) A[IDX2C(i, j, dim)] = value;
    } else if (i > j) {
      if (uplo == 0) A[IDX2C(i, j, dim)] = value;
    }
  }
}

template <typename T>
class multi_variable_gaussian_impl {
 public:
  enum Decomposer : unsigned char { chol_decomp, jacobi, qr };

 private:
  // adjustable stuff
  const int dim;
  const int nPoints     = 1;
  const double tol      = 1.e-7;
  const T epsilon       = 1.e-12;
  const int max_sweeps  = 100;
  cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
  const Decomposer method;

  // not so much
  T *P = 0, *X = 0, *x = 0, *workspace_decomp = 0, *eig = 0;
  int *info, Lwork, info_h;
  syevjInfo_t syevj_params = NULL;
  curandGenerator_t gen;
  raft::resources const& handle;
  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
  bool deinitilized      = false;

 public:  // functions
  multi_variable_gaussian_impl() = delete;
  multi_variable_gaussian_impl(raft::resources const& handle, const int dim, Decomposer method)
    : handle(handle), dim(dim), method(method)
  {
    auto cusolverHandle = resource::get_cusolver_dn_handle(handle);

    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 28));  // SEED
    if (method == chol_decomp) {
      RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf_bufferSize(
        cusolverHandle, uplo, dim, P, dim, &Lwork));
    } else if (method == jacobi) {  // jacobi init
      RAFT_CUSOLVER_TRY(cusolverDnCreateSyevjInfo(&syevj_params));
      RAFT_CUSOLVER_TRY(cusolverDnXsyevjSetTolerance(syevj_params, tol));
      RAFT_CUSOLVER_TRY(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));
      RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnsyevj_bufferSize(
        cusolverHandle, jobz, uplo, dim, P, dim, eig, &Lwork, syevj_params));
    } else {  // method == qr
      RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnsyevd_bufferSize(
        cusolverHandle, jobz, uplo, dim, P, dim, eig, &Lwork));
    }
  }

  std::size_t get_workspace_size()
  {
    // malloc workspace_decomp
    std::size_t granularity = 256, offset = 0;
    workspace_decomp = (T*)offset;
    offset += raft::alignTo(sizeof(T) * Lwork, granularity);
    eig = (T*)offset;
    offset += raft::alignTo(sizeof(T) * dim, granularity);
    info = (int*)offset;
    offset += raft::alignTo(sizeof(int), granularity);
    return offset;
  }

  void set_workspace(T* workarea)
  {
    workspace_decomp = (T*)((std::size_t)workspace_decomp + (std::size_t)workarea);
    eig              = (T*)((std::size_t)eig + (std::size_t)workarea);
    info             = (int*)((std::size_t)info + (std::size_t)workarea);
  }

  void give_gaussian(const int nPoints, T* P, T* X, const T* x = 0)
  {
    if (resource::get_dry_run_flag(handle)) { return; }
    auto cusolverHandle = resource::get_cusolver_dn_handle(handle);
    auto cudaStream     = resource::get_cuda_stream(handle);
    if (method == chol_decomp) {
      // lower part will contains chol_decomp
      RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf(
        cusolverHandle, uplo, dim, P, dim, workspace_decomp, Lwork, info, cudaStream));
    } else if (method == jacobi) {
      RAFT_CUSOLVER_TRY(
        raft::linalg::detail::cusolverDnsyevj(cusolverHandle,
                                              jobz,
                                              uplo,
                                              dim,
                                              P,
                                              dim,
                                              eig,
                                              workspace_decomp,
                                              Lwork,
                                              info,
                                              syevj_params,
                                              cudaStream));  // vectors stored as cols. & col major
    } else {                                                 // qr
      RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnsyevd(
        cusolverHandle, jobz, uplo, dim, P, dim, eig, workspace_decomp, Lwork, info, cudaStream));
    }
    raft::update_host(&info_h, info, 1, cudaStream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(cudaStream));
    ASSERT(info_h == 0, "mvg: error in syevj/syevd/potrf, info=%d | expected=0", info_h);
    T mean = 0.0, stddv = 1.0;
    // generate nxN gaussian nums in X
    CURAND_CHECK(
      detail::curandGenerateNormal(gen, X, (nPoints * dim) + (nPoints * dim) % 2, mean, stddv));
    T alfa = 1.0, beta = 0.0;
    if (method == chol_decomp) {
      // upper part (0) being filled with 0.0
      dim3 block(32, 32);
      dim3 grid(raft::ceildiv(dim, (int)block.x), raft::ceildiv(dim, (int)block.y));
      fill_uplo<T><<<grid, block, 0, cudaStream>>>(dim, UPPER, (T)0.0, P);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // P is lower triangular chol decomp mtrx
      raft::linalg::gemm(
        handle, false, false, dim, nPoints, dim, &alfa, P, dim, X, dim, &beta, X, dim, cudaStream);
    } else {
      epsilonToZero(eig, epsilon, dim, cudaStream);
      dim3 block(64);
      dim3 grid(raft::ceildiv(dim, (int)block.x));
      RAFT_CUDA_TRY(cudaMemsetAsync(info, 0, sizeof(int), cudaStream));
      grid.x = raft::ceildiv(dim * dim, (int)block.x);
      combined_dot_product<T><<<grid, block, 0, cudaStream>>>(dim, dim, eig, P, info);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // checking if any eigen vals were negative
      raft::update_host(&info_h, info, 1, cudaStream);
      RAFT_CUDA_TRY(cudaStreamSynchronize(cudaStream));
      ASSERT(info_h == 0, "mvg: Cov matrix has %dth Eigenval negative", info_h);

      // Got Q = eigvect*eigvals.sqrt in P, Q*X in X below
      raft::linalg::gemm(
        handle, false, false, dim, nPoints, dim, &alfa, P, dim, X, dim, &beta, X, dim, cudaStream);
    }
    // working to make mean not 0
    // since we are working with column-major, nPoints and dim are swapped
    if (x != NULL) matVecAdd(X, X, x, T(1.0), nPoints, dim, cudaStream);
  }

  void deinit()
  {
    if (deinitilized) return;
    CURAND_CHECK(curandDestroyGenerator(gen));
    RAFT_CUSOLVER_TRY(cusolverDnDestroySyevjInfo(syevj_params));
    deinitilized = true;
  }

  ~multi_variable_gaussian_impl() { deinit(); }
};  // end of multi_variable_gaussian_impl

template <typename ValueType>
class multi_variable_gaussian_setup_token;

template <typename ValueType>
multi_variable_gaussian_setup_token<ValueType> build_multi_variable_gaussian_token_impl(
  raft::resources const& handle,
  rmm::device_async_resource_ref mem_resource,
  const int dim,
  const multi_variable_gaussian_decomposition_method method);

template <typename ValueType>
void compute_multi_variable_gaussian_impl(
  multi_variable_gaussian_setup_token<ValueType>& token,
  std::optional<raft::device_vector_view<const ValueType, int>> x,
  raft::device_matrix_view<ValueType, int, raft::col_major> P,
  raft::device_matrix_view<ValueType, int, raft::col_major> X);

template <typename ValueType>
class multi_variable_gaussian_setup_token {
  template <typename T>
  friend multi_variable_gaussian_setup_token<T> build_multi_variable_gaussian_token_impl(
    raft::resources const& handle,
    rmm::device_async_resource_ref mem_resource,
    const int dim,
    const multi_variable_gaussian_decomposition_method method);

  template <typename T>
  friend void compute_multi_variable_gaussian_impl(
    multi_variable_gaussian_setup_token<T>& token,
    std::optional<raft::device_vector_view<const T, int>> x,
    raft::device_matrix_view<T, int, raft::col_major> P,
    raft::device_matrix_view<T, int, raft::col_major> X);

 private:
  typename multi_variable_gaussian_impl<ValueType>::Decomposer new_enum_to_old_enum(
    multi_variable_gaussian_decomposition_method method)
  {
    if (method == multi_variable_gaussian_decomposition_method::CHOLESKY) {
      return multi_variable_gaussian_impl<ValueType>::chol_decomp;
    } else if (method == multi_variable_gaussian_decomposition_method::JACOBI) {
      return multi_variable_gaussian_impl<ValueType>::jacobi;
    } else {
      return multi_variable_gaussian_impl<ValueType>::qr;
    }
  }

  // Constructor, only for use by friend functions.
  // Hiding this will let us change the implementation in the future.
  multi_variable_gaussian_setup_token(raft::resources const& handle,
                                      rmm::device_async_resource_ref mem_resource,
                                      const int dim,
                                      const multi_variable_gaussian_decomposition_method method)
    : impl_(std::make_unique<multi_variable_gaussian_impl<ValueType>>(
        handle, dim, new_enum_to_old_enum(method))),
      handle_(handle),
      mem_resource_(mem_resource),
      dim_(dim)
  {
  }

  /**
   * @brief Compute the multivariable Gaussian.
   *
   * @param[in]    x vector of dim elements
   * @param[inout] P On input, dim x dim matrix; overwritten on output
   * @param[out]   X dim x nPoints matrix
   */
  void compute(std::optional<raft::device_vector_view<const ValueType, int>> x,
               raft::device_matrix_view<ValueType, int, raft::col_major> P,
               raft::device_matrix_view<ValueType, int, raft::col_major> X)
  {
    const int input_dim = P.extent(0);
    RAFT_EXPECTS(input_dim == dim(),
                 "multi_variable_gaussian: "
                 "P.extent(0) = %d does not match the extent %d "
                 "with which the token was created",
                 input_dim,
                 dim());
    RAFT_EXPECTS(P.extent(0) == P.extent(1),
                 "multi_variable_gaussian: "
                 "P must be square, but P.extent(0) = %d != P.extent(1) = %d",
                 P.extent(0),
                 P.extent(1));
    RAFT_EXPECTS(P.extent(0) == X.extent(0),
                 "multi_variable_gaussian: "
                 "P.extent(0) = %d != X.extent(0) = %d",
                 P.extent(0),
                 X.extent(0));
    const bool x_has_value = x.has_value();
    const int x_extent_0   = x_has_value ? (*x).extent(0) : 0;
    RAFT_EXPECTS(not x_has_value || P.extent(0) == x_extent_0,
                 "multi_variable_gaussian: "
                 "P.extent(0) = %d != x.extent(0) = %d",
                 P.extent(0),
                 x_extent_0);
    const int nPoints      = X.extent(1);
    const ValueType* x_ptr = x_has_value ? (*x).data_handle() : nullptr;

    auto workspace = allocate_workspace();
    impl_->set_workspace(workspace.data());
    impl_->give_gaussian(nPoints, P.data_handle(), X.data_handle(), x_ptr);
  }

 private:
  std::unique_ptr<multi_variable_gaussian_impl<ValueType>> impl_;
  raft::resources const& handle_;
  rmm::device_async_resource_ref mem_resource_;
  int dim_ = 0;

  auto allocate_workspace() const
  {
    const auto num_elements = impl_->get_workspace_size();
    return rmm::device_uvector<ValueType>{
      num_elements, resource::get_cuda_stream(handle_), mem_resource_};
  }

  int dim() const { return dim_; }
};

template <typename ValueType>
multi_variable_gaussian_setup_token<ValueType> build_multi_variable_gaussian_token_impl(
  raft::resources const& handle,
  rmm::device_async_resource_ref mem_resource,
  const int dim,
  const multi_variable_gaussian_decomposition_method method)
{
  return multi_variable_gaussian_setup_token<ValueType>(handle, mem_resource, dim, method);
}

template <typename ValueType>
void compute_multi_variable_gaussian_impl(
  multi_variable_gaussian_setup_token<ValueType>& token,
  std::optional<raft::device_vector_view<const ValueType, int>> x,
  raft::device_matrix_view<ValueType, int, raft::col_major> P,
  raft::device_matrix_view<ValueType, int, raft::col_major> X)
{
  token.compute(x, P, X);
}

template <typename ValueType>
void compute_multi_variable_gaussian_impl(
  raft::resources const& handle,
  rmm::device_async_resource_ref mem_resource,
  std::optional<raft::device_vector_view<const ValueType, int>> x,
  raft::device_matrix_view<ValueType, int, raft::col_major> P,
  raft::device_matrix_view<ValueType, int, raft::col_major> X,
  const multi_variable_gaussian_decomposition_method method)
{
  auto token =
    build_multi_variable_gaussian_token_impl<ValueType>(handle, mem_resource, P.extent(0), method);
  compute_multi_variable_gaussian_impl(token, x, P, X);
}

template <typename T>
class multi_variable_gaussian : public detail::multi_variable_gaussian_impl<T> {
 public:
  // using Decomposer = typename detail::multi_variable_gaussian_impl<T>::Decomposer;
  // using detail::multi_variable_gaussian_impl<T>::Decomposer::chol_decomp;
  // using detail::multi_variable_gaussian_impl<T>::Decomposer::jacobi;
  // using detail::multi_variable_gaussian_impl<T>::Decomposer::qr;

  multi_variable_gaussian() = delete;
  multi_variable_gaussian(raft::resources const& handle,
                          const int dim,
                          typename detail::multi_variable_gaussian_impl<T>::Decomposer method)
    : detail::multi_variable_gaussian_impl<T>{handle, dim, method}
  {
  }

  std::size_t get_workspace_size()
  {
    return detail::multi_variable_gaussian_impl<T>::get_workspace_size();
  }

  void set_workspace(T* workarea)
  {
    detail::multi_variable_gaussian_impl<T>::set_workspace(workarea);
  }

  void give_gaussian(const int nPoints, T* P, T* X, const T* x = 0)
  {
    detail::multi_variable_gaussian_impl<T>::give_gaussian(nPoints, P, X, x);
  }

  void deinit() { detail::multi_variable_gaussian_impl<T>::deinit(); }

  ~multi_variable_gaussian() { deinit(); }
};  // end of multi_variable_gaussian

};  // end of namespace detail
};  // namespace random
}  // namespace raft
