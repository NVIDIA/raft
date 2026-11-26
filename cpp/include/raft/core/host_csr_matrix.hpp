/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/csr_matrix.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_span.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/sparse_types.hpp>

#include <type_traits>

namespace RAFT_EXPORT raft {

/**
 * \defgroup host_csr_matrix Host CSR Matrix
 * @{
 */

/**
 * Specialization for a sparsity-preserving compressed structure view which uses host memory
 */
template <typename IndptrType, typename IndicesType, typename NZType>
using host_compressed_structure_view =
  compressed_structure_view<IndptrType, IndicesType, NZType, false>;

/**
 * Specialization for a sparsity-owning compressed structure which uses host memory
 */
template <typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = host_container_policy>
using host_compressed_structure =
  compressed_structure<IndptrType, IndicesType, NZType, false, ContainerPolicy>;

/**
 * Specialization for a csr matrix view which uses host memory
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
using host_csr_matrix_view = csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, false>;

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = host_container_policy,
          SparsityType sparsity_type                     = SparsityType::OWNING>
using host_csr_matrix =
  csr_matrix<ElementType, IndptrType, IndicesType, NZType, false, ContainerPolicy, sparsity_type>;

/**
 * Specialization for a sparsity-owning csr matrix which uses host memory
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = host_container_policy>
using host_sparsity_owning_csr_matrix =
  csr_matrix<ElementType, IndptrType, IndicesType, NZType, false, ContainerPolicy>;

/**
 * Specialization for a sparsity-preserving csr matrix which uses host memory
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy = host_container_policy>
using host_sparsity_preserving_csr_matrix = csr_matrix<ElementType,
                                                       IndptrType,
                                                       IndicesType,
                                                       NZType,
                                                       false,
                                                       ContainerPolicy,
                                                       SparsityType::PRESERVING>;

template <typename T>
struct is_host_csr_matrix_view : std::false_type {};

template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
struct is_host_csr_matrix_view<host_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>>
  : std::true_type {};

template <typename T>
constexpr bool is_host_csr_matrix_view_v = is_host_csr_matrix_view<T>::value;

template <typename T>
struct is_host_csr_matrix : std::false_type {};

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          template <typename T> typename ContainerPolicy,
          SparsityType sparsity_type>
struct is_host_csr_matrix<
  host_csr_matrix<ElementType, IndptrType, IndicesType, NZType, ContainerPolicy, sparsity_type>>
  : std::true_type {};

template <typename T>
constexpr bool is_host_csr_matrix_v = is_host_csr_matrix<T>::value;

template <typename T>
constexpr bool is_host_csr_sparsity_owning_v =
  is_host_csr_matrix<T>::value and T::get_sparsity_type() == OWNING;

template <typename T>
constexpr bool is_host_csr_sparsity_preserving_v = std::disjunction_v<
  is_host_csr_matrix_view<T>,
  std::bool_constant<is_host_csr_matrix<T>::value and T::get_sparsity_type() == PRESERVING>>;

/**
 * Create a sparsity-owning sparse matrix in the compressed-sparse row format. sparsity-owning
 * means that all of the underlying vectors (data, indptr, indices) are owned by the csr_matrix
 * instance. If not known up front, the sparsity can be ignored in this factory function and
 * `resize()` invoked on the instance once the sparsity is known.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/host_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 *
 * raft::resources handle;
 * auto csr_matrix = raft::make_host_csr_matrix<float>(handle, n_rows, n_cols);
 * ...
 * // compute expected sparsity
 * ...
 * int nnz = 5000;
 * csr_matrix.initialize_sparsity(nnz);
 * @endcode
 *
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] n_rows total number of rows in the matrix
 * @param[in] n_cols total number of columns in the matrix
 * @param[in] nnz number of non-zeros in the matrix if known [optional]
 * @return a sparsity-owning sparse matrix in compressed (csr) format
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_host_csr_matrix(raft::resources const& handle,
                          IndptrType n_rows,
                          IndicesType n_cols,
                          NZType nnz = 0)
{
  return host_sparsity_owning_csr_matrix<ElementType, IndptrType, IndicesType, NZType>(
    handle, n_rows, n_cols, nnz);
}

/**
 * Create a sparsity-preserving sparse matrix in the compressed-sparse row format.
 * sparsity-preserving means that a view of the csr sparsity is supplied, allowing the values in
 * the sparsity to change but not the sparsity itself. The csr_matrix instance does not own the
 * sparsity, the sparsity must be known up front, and cannot be resized later.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/host_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 *
 * raft::resources handle;
 * auto csr_structure = raft::make_host_compressed_structure(handle, n_rows, n_cols);
 * ...
 * // compute expected sparsity
 * ...
 * int nnz = 5000;
 * csr_structure.initialize_sparsity(nnz);
 * auto csr_matrix = raft::make_host_csr_matrix<float>(handle, csr_structure.view());
 * @endcode

 *
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] structure a sparsity-preserving compressed structural view
 * @return a sparsity-preserving sparse matrix in compressed (csr) format
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_host_csr_matrix(raft::resources const& handle,
                          host_compressed_structure_view<IndptrType, IndicesType, NZType> structure)
{
  return host_sparsity_preserving_csr_matrix<ElementType, IndptrType, IndicesType, NZType>(
    handle, structure);
}

/**
 * Create a non-owning sparse matrix view in the compressed-sparse row format. This is
 * sparsity-preserving, meaning that the underlying sparsity is known and cannot be changed. Use the
 * sparsity-owning csr_matrix if sparsity needs to be mutable.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/host_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following pointer is assumed to reference device memory for a size of nnz
 * float* h_elm_ptr = ...;
 *
 * raft::resources handle;
 * auto csr_structure = raft::make_host_compressed_structure(handle, n_rows, n_cols, nnz);
 * auto csr_matrix_view = raft::make_host_csr_matrix_view(h_elm_ptr, csr_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] ptr a pointer to array of nonzero matrix elements on host (size nnz)
 * @param[in] structure a sparsity-preserving compressed sparse structural view
 * @return a sparsity-preserving csr matrix view
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_host_csr_matrix_view(
  ElementType* ptr, host_compressed_structure_view<IndptrType, IndicesType, NZType> structure)
{
  return host_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>(
    raft::host_span<ElementType>(ptr, structure.get_nnz()), structure);
}

/**
 * Create a non-owning sparse matrix view in the compressed-sparse row format. This is
 * sparsity-preserving, meaning that the underlying sparsity is known and cannot be changed. Use the
 * sparsity-owning csr_matrix if sparsity needs to be mutable.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/host_span.hpp>
 * #include <raft/core/host_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following span is assumed to be of size nnz
 * raft::host_span<float> h_elm_ptr;
 *
 * raft::resources handle;
 * auto csr_structure = raft::make_host_compressed_structure(handle, n_rows, n_cols, nnz);
 * auto csr_matrix_view = raft::make_host_csr_matrix_view(h_elm_ptr, csr_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] elements host span containing array of matrix elements (size nnz)
 * @param[in] structure a sparsity-preserving structural view
 * @return a sparsity-preserving csr matrix view
 */
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType = uint64_t>
auto make_host_csr_matrix_view(
  raft::host_span<ElementType> elements,
  host_compressed_structure_view<IndptrType, IndicesType, NZType> structure)
{
  RAFT_EXPECTS(elements.size() == structure.get_nnz(),
               "Size of elements must be equal to the nnz from the structure");
  return host_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType>(elements, structure);
}

/**
 * Create a sparsity-owning compressed structure. This is not sparsity-preserving, meaning that
 * the underlying sparsity does not need to be known upon construction. When not known up front,
 * the allocation of the underlying indices array is delayed until `resize(nnz)` is invoked.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/host_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * raft::resources handle;
 * auto csr_structure = raft::make_host_compressed_structure(handle, n_rows, n_cols, nnz);
 * ...
 * // compute expected sparsity
 * ...
 * csr_structure.initialize_sparsity(nnz);
 * @endcode
 *
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of cols
 * @param[in] nnz total number of nonzeros, if known
 * @return a sparsity-owning compressed structure instance
 */
template <typename IndptrType, typename IndicesType, typename NZType = uint64_t>
auto make_host_compressed_structure(raft::resources const& handle,
                                    IndptrType n_rows,
                                    IndicesType n_cols,
                                    NZType nnz = 0)
{
  return host_compressed_structure<IndptrType, IndicesType, NZType>(handle, n_rows, n_cols, nnz);
}

/**
 * Create a non-owning sparsity-preserved compressed structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/host_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following pointer is assumed to reference host-accessible memory of size n_rows+1
 * int *indptr = ...;
 *
 * // The following pointer is assumed to reference host-accessible memory of size nnz
 * int *indices = ...;
 *
 * raft::resources handle;
 * auto csr_structure =
 *   raft::make_host_compressed_structure_view(indptr, indices, n_rows, n_cols, nnz);
 * @endcode
 *
 *
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] indptr structural indptr (size n_rows+1)
 * @param[in] indices structural indices (size nnz)
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of columns
 * @param[in] nnz number of non-zeros
 * @return a sparsity-preserving compressed structural view
 */
template <typename IndptrType, typename IndicesType, typename NZType = uint64_t>
auto make_host_compressed_structure_view(
  IndptrType* indptr, IndicesType* indices, IndptrType n_rows, IndicesType n_cols, NZType nnz)
{
  return host_compressed_structure_view<IndptrType, IndicesType, NZType>(
    raft::host_span<IndptrType>(indptr, n_rows + 1),
    raft::host_span<IndicesType>(indices, nnz),
    n_cols);
}

/**
 * Create a non-owning sparsity-preserved compressed structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/host_csr_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following host span is assumed to be of size n_rows+1
 * raft::host_span<int> indptr;
 *
 * // The following host span is assumed to be of size nnz
 * raft::host_span<int> indices;
 *
 * raft::resources handle;
 * auto csr_structure = raft::make_host_compressed_structure_view(indptr, indices, n_cols);
 * @endcode
 *
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam NZType
 * @param[in] indptr structural indptr (size n_rows+1)
 * @param[in] indices structural indices (size nnz)
 * @param[in] n_cols total number of columns
 * @return a sparsity-preserving compressed structural view
 *
 */
template <typename IndptrType, typename IndicesType, typename NZType = uint64_t>
auto make_host_compressed_structure_view(raft::host_span<IndptrType> indptr,
                                         raft::host_span<IndicesType> indices,
                                         IndicesType n_cols)
{
  return host_compressed_structure_view<IndptrType, IndicesType, NZType>(indptr, indices, n_cols);
}

/** @} */

};  // namespace RAFT_EXPORT raft
