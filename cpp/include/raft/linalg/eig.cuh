/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __EIG_H
#define __EIG_H

#pragma once

#include "detail/eig.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft {
namespace linalg {

/**
 * @brief eig decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @param handle raft handle
 * @param in the input buffer (symmetric matrix that has real eig values and
 * vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param stream cuda stream
 */
template <typename math_t>
void eigDC(raft::resources const& handle,
           const math_t* in,
           std::size_t n_rows,
           std::size_t n_cols,
           math_t* eig_vectors,
           math_t* eig_vals,
           cudaStream_t stream)
{
  detail::eigDC(handle, in, n_rows, n_cols, eig_vectors, eig_vals, stream);
}

using detail::COPY_INPUT;
using detail::EigVecMemUsage;
using detail::OVERWRITE_INPUT;

/**
 * @brief eig sel decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @param handle raft handle
 * @param in the input buffer (symmetric matrix that has real eig values and
 * vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param n_eig_vals: number of eigenvectors to be generated
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param memUsage: the memory selection for eig vector output
 * @param stream cuda stream
 */
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
  detail::eigSelDC(handle, in, n_rows, n_cols, n_eig_vals, eig_vectors, eig_vals, memUsage, stream);
}

/**
 * @brief overloaded function for eig decomp with Jacobi method for the
 * column-major symmetric matrices (in parameter)
 * @param handle: raft handle
 * @param in: input matrix
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param stream: stream on which this function will be run
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the
 * error is below tol
 * @param sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 */
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
  detail::eigJacobi(handle, in, n_rows, n_cols, eig_vectors, eig_vals, stream, tol, sweeps);
}

/**
 * @defgroup eig Eigen Decomposition Methods
 * @{
 */

/**
 * @brief eig decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @tparam ValueType the data-type of input and output
 * @tparam IntegerType Integer used for addressing
 * @param handle raft::resources
 * @param[in] in input raft::device_matrix_view (symmetric matrix that has real eig values and
 * vectors)
 * @param[out] eig_vectors: eigenvectors output of type raft::device_matrix_view
 * @param[out] eig_vals: eigen values output of type raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void eig_dc(raft::resources const& handle,
            raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
            raft::device_matrix_view<ValueType, IndexType, raft::col_major> eig_vectors,
            raft::device_vector_view<ValueType, IndexType> eig_vals)
{
  RAFT_EXPECTS(in.size() == eig_vectors.size(), "Size mismatch between Input and Eigen Vectors");
  RAFT_EXPECTS(eig_vals.size() == in.extent(1), "Size mismatch between Input and Eigen Values");

  eigDC(handle,
        in.data_handle(),
        in.extent(0),
        in.extent(1),
        eig_vectors.data_handle(),
        eig_vals.data_handle(),
        resource::get_cuda_stream(handle).get());
}

/**
 * @brief eig decomp to select top-n eigen values with divide and conquer method
 *        for the column-major symmetric matrices
 * @tparam ValueType the data-type of input and output
 * @tparam IntegerType Integer used for addressing
 * @param[in] handle raft::resources
 * @param[in] in input raft::device_matrix_view (symmetric matrix that has real eig values and
 * vectors)
 * @param[out] eig_vectors: eigenvectors output of type raft::device_matrix_view
 * @param[out] eig_vals: eigen values output of type raft::device_vector_view
 * @param[in] n_eig_vals: number of eigenvectors to be generated
 * @param[in] memUsage: the memory selection for eig vector output
 */
template <typename ValueType, typename IndexType>
void eig_dc_selective(raft::resources const& handle,
                      raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
                      raft::device_matrix_view<ValueType, IndexType, raft::col_major> eig_vectors,
                      raft::device_vector_view<ValueType, IndexType> eig_vals,
                      std::size_t n_eig_vals,
                      EigVecMemUsage memUsage)
{
  RAFT_EXPECTS(eig_vectors.size() == n_eig_vals * in.extent(0),
               "Size mismatch between Input and Eigen Vectors");
  RAFT_EXPECTS(eig_vals.size() == n_eig_vals, "Size mismatch between Input and Eigen Values");

  raft::linalg::eigSelDC(handle,
                         const_cast<ValueType*>(in.data_handle()),
                         in.extent(0),
                         in.extent(1),
                         n_eig_vals,
                         eig_vectors.data_handle(),
                         eig_vals.data_handle(),
                         memUsage,
                         resource::get_cuda_stream(handle).get());
}

/**
 * @brief overloaded function for eig decomp with Jacobi method for the
 * column-major symmetric matrices (in parameter)
 * @tparam ValueType the data-type of input and output
 * @tparam IntegerType Integer used for addressing
 * @param handle raft::resources
 * @param[in] in input raft::device_matrix_view (symmetric matrix that has real eig values and
 * vectors)
 * @param[out] eig_vectors: eigenvectors output of type raft::device_matrix_view
 * @param[out] eig_vals: eigen values output of type raft::device_vector_view
 * @param[in] tol: error tolerance for the jacobi method. Algorithm stops when the
                   Frobenius norm of the absolute error is below tol
 * @param[in] sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 */
template <typename ValueType, typename IndexType>
void eig_jacobi(raft::resources const& handle,
                raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
                raft::device_matrix_view<ValueType, IndexType, raft::col_major> eig_vectors,
                raft::device_vector_view<ValueType, IndexType> eig_vals,
                ValueType tol = 1.e-7,
                int sweeps    = 15)
{
  RAFT_EXPECTS(in.size() == eig_vectors.size(), "Size mismatch between Input and Eigen Vectors");
  RAFT_EXPECTS(eig_vals.size() == in.extent(1), "Size mismatch between Input and Eigen Values");

  eigJacobi(handle,
            in.data_handle(),
            in.extent(0),
            in.extent(1),
            eig_vectors.data_handle(),
            eig_vals.data_handle(),
            resource::get_cuda_stream(handle).get(),
            tol,
            sweeps);
}

/** @} */  // end of eig

};  // end namespace linalg
};  // end namespace raft

#endif
