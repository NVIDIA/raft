/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __RSVD_H
#define __RSVD_H

#pragma once

#include "detail/rsvd.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft {
namespace linalg {

/**
 * @brief randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying no. of PCs and
 * upsamples directly
 * @param handle: raft handle
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param k: no. of singular values to be computed
 * @param p: no. of upsamples
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param gen_left_vec: left vector needs to be generated or not?
 * @param gen_right_vec: right vector needs to be generated or not?
 * @param use_jacobi: whether to jacobi solver for decomposition
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param stream cuda stream
 */
template <typename math_t>
void rsvdFixedRank(raft::resources const& handle,
                   math_t* M,
                   int n_rows,
                   int n_cols,
                   math_t* S_vec,
                   math_t* U,
                   math_t* V,
                   int k,
                   int p,
                   bool use_bbt,
                   bool gen_left_vec,
                   bool gen_right_vec,
                   bool use_jacobi,
                   math_t tol,
                   int max_sweeps,
                   cudaStream_t stream)
{
  detail::rsvdFixedRank(handle,
                        M,
                        n_rows,
                        n_cols,
                        S_vec,
                        U,
                        V,
                        k,
                        p,
                        use_bbt,
                        gen_left_vec,
                        gen_right_vec,
                        use_jacobi,
                        tol,
                        max_sweeps,
                        stream);
}

/**
 * @brief randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying the PC and upsampling
 * ratio
 * @param handle: raft handle
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param PC_perc: percentage of singular values to be computed
 * @param UpS_perc: upsampling percentage
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param gen_left_vec: left vector needs to be generated or not?
 * @param gen_right_vec: right vector needs to be generated or not?
 * @param use_jacobi: whether to jacobi solver for decomposition
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param stream cuda stream
 */
template <typename math_t>
void rsvdPerc(raft::resources const& handle,
              math_t* M,
              int n_rows,
              int n_cols,
              math_t* S_vec,
              math_t* U,
              math_t* V,
              math_t PC_perc,
              math_t UpS_perc,
              bool use_bbt,
              bool gen_left_vec,
              bool gen_right_vec,
              bool use_jacobi,
              math_t tol,
              int max_sweeps,
              cudaStream_t stream)
{
  detail::rsvdPerc(handle,
                   M,
                   n_rows,
                   n_cols,
                   S_vec,
                   U,
                   V,
                   PC_perc,
                   UpS_perc,
                   use_bbt,
                   gen_left_vec,
                   gen_right_vec,
                   use_jacobi,
                   tol,
                   max_sweeps,
                   stream);
}

/**
 * @defgroup rsvd Randomized Singular Value Decomposition
 * @{
 */

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using QR decomposition, by specifying no. of PCs and
 * upsamples directly
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @tparam UType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * U_in
 * @tparam VType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * V_in
 * @param[in] handle raft::resources
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] p no. of upsamples
 * @param[out] U_in std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V_in std::optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_fixed_rank(raft::resources const& handle,
                     raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
                     raft::device_vector_view<ValueType, IndexType> S_vec,
                     IndexType p,
                     UType&& U_in,
                     VType&& V_in)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);
  ValueType* U_ptr = nullptr;
  ValueType* V_ptr = nullptr;

  if (U) {
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
    U_ptr = U.value().data_handle();
  }
  if (V) {
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
    V_ptr = V.value().data_handle();
  }

  rsvdFixedRank(handle,
                const_cast<ValueType*>(M.data_handle()),
                M.extent(0),
                M.extent(1),
                S_vec.data_handle(),
                U_ptr,
                V_ptr,
                S_vec.extent(0),
                p,
                false,
                U.has_value(),
                V.has_value(),
                false,
                static_cast<ValueType>(0),
                0,
                resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `rsvd_fixed_rank` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_fixed_rank`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 4>>
void rsvd_fixed_rank(Args... args)
{
  rsvd_fixed_rank(std::forward<Args>(args)..., std::nullopt, std::nullopt);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using symmetric Eigen decomposition, by specifying no. of PCs and
 * upsamples directly. The rectangular input matrix is made square and symmetric using B @ B^T
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @tparam UType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * U_in
 * @tparam VType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * V_in
 * @param[in] handle raft::resources
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] p no. of upsamples
 * @param[out] U_in std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V_in std::optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_fixed_rank_symmetric(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  IndexType p,
  UType&& U_in,
  VType&& V_in)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);
  ValueType* U_ptr = nullptr;
  ValueType* V_ptr = nullptr;

  if (U) {
    U_ptr = U.value().data_handle();
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    V_ptr = V.value().data_handle();
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdFixedRank(handle,
                const_cast<ValueType*>(M.data_handle()),
                M.extent(0),
                M.extent(1),
                S_vec.data_handle(),
                U_ptr,
                V_ptr,
                S_vec.extent(0),
                p,
                true,
                U.has_value(),
                V.has_value(),
                false,
                static_cast<ValueType>(0),
                0,
                resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `rsvd_fixed_rank_symmetric` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_fixed_rank_symmetric`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 4>>
void rsvd_fixed_rank_symmetric(Args... args)
{
  rsvd_fixed_rank_symmetric(std::forward<Args>(args)..., std::nullopt, std::nullopt);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using Jacobi method, by specifying no. of PCs and
 * upsamples directly
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @tparam UType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * U_in
 * @tparam VType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * V_in
 * @param[in] handle raft::resources
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] p no. of upsamples
 * @param[in] tol tolerance for Jacobi-based solvers
 * @param[in] max_sweeps maximum number of sweeps for Jacobi-based solvers
 * @param[out] U_in std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V_in std::optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_fixed_rank_jacobi(raft::resources const& handle,
                            raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
                            raft::device_vector_view<ValueType, IndexType> S_vec,
                            IndexType p,
                            ValueType tol,
                            int max_sweeps,
                            UType&& U_in,
                            VType&& V_in)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);
  ValueType* U_ptr = nullptr;
  ValueType* V_ptr = nullptr;

  if (U) {
    U_ptr = U.value().data_handle();
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    V_ptr = V.value().data_handle();
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdFixedRank(handle,
                const_cast<ValueType*>(M.data_handle()),
                M.extent(0),
                M.extent(1),
                S_vec.data_handle(),
                U_ptr,
                V_ptr,
                S_vec.extent(0),
                p,
                false,
                U.has_value(),
                V.has_value(),
                true,
                tol,
                max_sweeps,
                resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `rsvd_fixed_rank_jacobi` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_fixed_rank_jacobi`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 6>>
void rsvd_fixed_rank_jacobi(Args... args)
{
  rsvd_fixed_rank_jacobi(std::forward<Args>(args)..., std::nullopt, std::nullopt);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using Jacobi method, by specifying no. of PCs and
 * upsamples directly. The rectangular input matrix is made square and symmetric using B @ B^T
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @tparam UType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * U_in
 * @tparam VType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * V_in
 * @param[in] handle raft::resources
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] p no. of upsamples
 * @param[in] tol tolerance for Jacobi-based solvers
 * @param[in] max_sweeps maximum number of sweeps for Jacobi-based solvers
 * @param[out] U_in std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V_in std::optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_fixed_rank_symmetric_jacobi(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  IndexType p,
  ValueType tol,
  int max_sweeps,
  UType&& U_in,
  VType&& V_in)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);
  ValueType* U_ptr = nullptr;
  ValueType* V_ptr = nullptr;

  if (U) {
    U_ptr = U.value().data_handle();
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    V_ptr = V.value().data_handle();
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdFixedRank(handle,
                const_cast<ValueType*>(M.data_handle()),
                M.extent(0),
                M.extent(1),
                S_vec.data_handle(),
                U_ptr,
                V_ptr,
                S_vec.extent(0),
                p,
                true,
                U.has_value(),
                V.has_value(),
                true,
                tol,
                max_sweeps,
                resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `rsvd_fixed_rank_symmetric_jacobi` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_fixed_rank_symmetric_jacobi`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 6>>
void rsvd_fixed_rank_symmetric_jacobi(Args... args)
{
  rsvd_fixed_rank_symmetric_jacobi(std::forward<Args>(args)..., std::nullopt, std::nullopt);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using QR decomposition, by specifying the PC and upsampling
 * ratio
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @tparam UType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * U_in
 * @tparam VType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * V_in
 * @param[in] handle raft::resources
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] PC_perc percentage of singular values to be computed
 * @param[in] UpS_perc upsampling percentage
 * @param[out] U_in std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V_in std::optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_perc(raft::resources const& handle,
               raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
               raft::device_vector_view<ValueType, IndexType> S_vec,
               ValueType PC_perc,
               ValueType UpS_perc,
               UType&& U_in,
               VType&& V_in)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);
  ValueType* U_ptr = nullptr;
  ValueType* V_ptr = nullptr;

  if (U) {
    U_ptr = U.value().data_handle();
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    V_ptr = V.value().data_handle();
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdPerc(handle,
           const_cast<ValueType*>(M.data_handle()),
           M.extent(0),
           M.extent(1),
           S_vec.data_handle(),
           U_ptr,
           V_ptr,
           PC_perc,
           UpS_perc,
           false,
           U.has_value(),
           V.has_value(),
           false,
           static_cast<ValueType>(0),
           0,
           resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `rsvd_perc` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_perc`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 5>>
void rsvd_perc(Args... args)
{
  rsvd_perc(std::forward<Args>(args)..., std::nullopt, std::nullopt);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using symmetric Eigen decomposition, by specifying the PC and upsampling
 * ratio. The rectangular input matrix is made square and symmetric using B @ B^T
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @tparam UType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * U_in
 * @tparam VType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * V_in
 * @param[in] handle raft::resources
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] PC_perc percentage of singular values to be computed
 * @param[in] UpS_perc upsampling percentage
 * @param[out] U_in std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V_in std::optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_perc_symmetric(raft::resources const& handle,
                         raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
                         raft::device_vector_view<ValueType, IndexType> S_vec,
                         ValueType PC_perc,
                         ValueType UpS_perc,
                         UType&& U_in,
                         VType&& V_in)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);
  ValueType* U_ptr = nullptr;
  ValueType* V_ptr = nullptr;

  if (U) {
    U_ptr = U.value().data_handle();
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    V_ptr = V.value().data_handle();
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdPerc(handle,
           const_cast<ValueType*>(M.data_handle()),
           M.extent(0),
           M.extent(1),
           S_vec.data_handle(),
           U_ptr,
           V_ptr,
           PC_perc,
           UpS_perc,
           true,
           U.has_value(),
           V.has_value(),
           false,
           static_cast<ValueType>(0),
           0,
           resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `rsvd_perc_symmetric` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_perc_symmetric`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 5>>
void rsvd_perc_symmetric(Args... args)
{
  rsvd_perc_symmetric(std::forward<Args>(args)..., std::nullopt, std::nullopt);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using Jacobi method, by specifying the PC and upsampling
 * ratio
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @tparam UType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * U_in
 * @tparam VType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * V_in
 * @param[in] handle raft::resources
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] PC_perc percentage of singular values to be computed
 * @param[in] UpS_perc upsampling percentage
 * @param[in] tol tolerance for Jacobi-based solvers
 * @param[in] max_sweeps maximum number of sweeps for Jacobi-based solvers
 * @param[out] U_in std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V_in std::optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_perc_jacobi(raft::resources const& handle,
                      raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
                      raft::device_vector_view<ValueType, IndexType> S_vec,
                      ValueType PC_perc,
                      ValueType UpS_perc,
                      ValueType tol,
                      int max_sweeps,
                      UType&& U_in,
                      VType&& V_in)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);
  ValueType* U_ptr = nullptr;
  ValueType* V_ptr = nullptr;

  if (U) {
    U_ptr = U.value().data_handle();
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    V_ptr = V.value().data_handle();
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdPerc(handle,
           const_cast<ValueType*>(M.data_handle()),
           M.extent(0),
           M.extent(1),
           S_vec.data_handle(),
           U_ptr,
           V_ptr,
           PC_perc,
           UpS_perc,
           false,
           U.has_value(),
           V.has_value(),
           true,
           tol,
           max_sweeps,
           resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `rsvd_perc_jacobi` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_perc_jacobi`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 7>>
void rsvd_perc_jacobi(Args... args)
{
  rsvd_perc_jacobi(std::forward<Args>(args)..., std::nullopt, std::nullopt);
}

/**
 * @brief randomized singular value decomposition (RSVD) on a column major
 * rectangular matrix using Jacobi method, by specifying the PC and upsampling
 * ratio. The rectangular input matrix is made square and symmetric using B @ B^T
 * @tparam ValueType value type of parameters
 * @tparam IndexType index type of parameters
 * @tparam UType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * U_in
 * @tparam VType std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> @c
 * V_in
 * @param[in] handle raft::resources
 * @param[in] M input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S_vec singular values raft::device_vector_view of shape (K)
 * @param[in] PC_perc percentage of singular values to be computed
 * @param[in] UpS_perc upsampling percentage
 * @param[in] tol tolerance for Jacobi-based solvers
 * @param[in] max_sweeps maximum number of sweeps for Jacobi-based solvers
 * @param[out] U_in std::optional left singular values of raft::device_matrix_view with layout
 * raft::col_major
 * @param[out] V_in std::optional right singular values of raft::device_matrix_view with layout
 * raft::col_major
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void rsvd_perc_symmetric_jacobi(
  raft::resources const& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> M,
  raft::device_vector_view<ValueType, IndexType> S_vec,
  ValueType PC_perc,
  ValueType UpS_perc,
  ValueType tol,
  int max_sweeps,
  UType&& U_in,
  VType&& V_in)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U =
    std::forward<UType>(U_in);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V =
    std::forward<VType>(V_in);
  ValueType* U_ptr = nullptr;
  ValueType* V_ptr = nullptr;

  if (U) {
    U_ptr = U.value().data_handle();
    RAFT_EXPECTS(M.extent(0) == U.value().extent(0), "Number of rows in M should be equal to U");
    RAFT_EXPECTS(S_vec.extent(0) == U.value().extent(1),
                 "Number of columns in U should be equal to length of S");
  }
  if (V) {
    V_ptr = V.value().data_handle();
    RAFT_EXPECTS(M.extent(1) == V.value().extent(1), "Number of columns in M should be equal to V");
    RAFT_EXPECTS(S_vec.extent(0) == V.value().extent(0),
                 "Number of rows in V should be equal to length of S");
  }

  rsvdPerc(handle,
           const_cast<ValueType*>(M.data_handle()),
           M.extent(0),
           M.extent(1),
           S_vec.data_handle(),
           U_ptr,
           V_ptr,
           PC_perc,
           UpS_perc,
           true,
           U.has_value(),
           V.has_value(),
           true,
           tol,
           max_sweeps,
           resource::get_cuda_stream(handle).get());
}

/**
 * @brief Overload of `rsvd_perc_symmetric_jacobi` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `rsvd_perc_symmetric_jacobi`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 7>>
void rsvd_perc_symmetric_jacobi(Args... args)
{
  rsvd_perc_symmetric_jacobi(std::forward<Args>(args)..., std::nullopt, std::nullopt);
}

/**
 * @brief randomized singular value decomposition (RSVD) using cusolver
 * @tparam math_t the data type
 * @tparam idx_t index type
 * @param[in]  handle:  raft handle
 * @param[in]  in:      input matrix in col-major format.
 *                      Warning: the content of this matrix is modified by the cuSOLVER routines.
 *                      [dim = n_rows * n_cols]
 * @param[out] S:       array of singular values of input matrix. The rank k must be less than
 * min(m,n). [dim = k]
 * @param[out] U:       optional left singular values of input matrix. Use std::nullopt to not
 * generate it. [dim = n_rows * k]
 * @param[out] V:       optional right singular values of input matrix. Use std::nullopt to not
 * generate it. [dim = k * n_cols]
 * @param[in]  p:       Oversampling. The size of the subspace will be (k + p). (k+p) is less than
 * min(m,n). (Recommended to be at least 2*k)
 * @param[in]  niters:  Number of iteration of power method. (2 is recommended)
 */
template <typename math_t, typename idx_t>
void randomized_svd(const raft::resources& handle,
                    raft::device_matrix_view<const math_t, idx_t, raft::col_major> in,
                    raft::device_vector_view<math_t, idx_t> S,
                    std::optional<raft::device_matrix_view<math_t, idx_t, raft::col_major>> U,
                    std::optional<raft::device_matrix_view<math_t, idx_t, raft::col_major>> V,
                    std::size_t p,
                    std::size_t niters)
{
  auto k                      = S.extent(0);
  math_t* left_sing_vecs_ptr  = nullptr;
  math_t* right_sing_vecs_ptr = nullptr;
  auto gen_U                  = U.has_value();
  auto gen_V                  = V.has_value();
  if (gen_U) {
    RAFT_EXPECTS(in.extent(0) == U.value().extent(0) && k == U.value().extent(1),
                 "U should have dimensions n_rows * k");
    left_sing_vecs_ptr = U.value().data_handle();
  }
  if (gen_V) {
    RAFT_EXPECTS(k == V.value().extent(0) && in.extent(1) == V.value().extent(1),
                 "V should have dimensions k * n_cols");
    right_sing_vecs_ptr = V.value().data_handle();
  }
  detail::randomized_svd(handle,
                         in.data_handle(),
                         in.extent(0),
                         in.extent(1),
                         k,
                         p,
                         niters,
                         S.data_handle(),
                         left_sing_vecs_ptr,
                         right_sing_vecs_ptr,
                         gen_U,
                         gen_V);
}

/**
 * @brief Overload of `randomized_svd` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for the optional arguments.
 *
 * Please see above for documentation of `randomized_svd`.
 */
template <typename math_t, typename idx_t, typename opt_u_vec_t, typename opt_v_vec_t>
void randomized_svd(const raft::resources& handle,
                    raft::device_matrix_view<const math_t, idx_t, raft::col_major> in,
                    raft::device_vector_view<math_t, idx_t> S,
                    opt_u_vec_t&& U,
                    opt_v_vec_t&& V,
                    std::size_t p,
                    std::size_t niters)
{
  std::optional<raft::device_matrix_view<math_t, idx_t, raft::col_major>> opt_u =
    std::forward<opt_u_vec_t>(U);
  std::optional<raft::device_matrix_view<math_t, idx_t, raft::col_major>> opt_v =
    std::forward<opt_v_vec_t>(V);
  randomized_svd(handle, in, S, opt_u, opt_v, p, niters);
}

/** @} */  // end of group rsvd

};  // end namespace linalg
};  // end namespace raft

#endif
