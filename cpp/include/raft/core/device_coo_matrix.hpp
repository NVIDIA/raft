/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/coo_matrix.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/sparse_types.hpp>

namespace RAFT_EXPORT raft {

/**
 * \defgroup device_coo_matrix Device COO Matrix
 * @{
 */

/**
 * Specialization for a sparsity-preserving coordinate structure view which uses device memory
 */
template <typename RowType, typename ColType, typename NZType>
using device_coordinate_structure_view = coordinate_structure_view<RowType, ColType, NZType, true>;

/**
 * Specialization for a sparsity-owning coordinate structure which uses device memory
 */
template <typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_container_policy>
using device_coordinate_structure =
  coordinate_structure<RowType, ColType, NZType, true, ContainerPolicy>;

/**
 * Specialization for a coo matrix view which uses device memory
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
using device_coo_matrix_view = coo_matrix_view<ElementType, RowType, ColType, NZType, true>;

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_container_policy,
          SparsityType sparsity_type                     = SparsityType::OWNING>
using device_coo_matrix =
  coo_matrix<ElementType, RowType, ColType, NZType, true, ContainerPolicy, sparsity_type>;

/**
 * Specialization for a sparsity-owning coo matrix which uses device memory
 */
template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_container_policy>
using device_sparsity_owning_coo_matrix =
  coo_matrix<ElementType, RowType, ColType, NZType, true, ContainerPolicy>;

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy = device_container_policy>
using device_sparsity_preserving_coo_matrix = coo_matrix<ElementType,
                                                         RowType,
                                                         ColType,
                                                         NZType,
                                                         true,
                                                         ContainerPolicy,
                                                         SparsityType::PRESERVING>;

template <typename T>
struct is_device_coo_matrix_view : std::false_type {};

template <typename ElementType, typename RowType, typename ColType, typename NZType>
struct is_device_coo_matrix_view<device_coo_matrix_view<ElementType, RowType, ColType, NZType>>
  : std::true_type {};

template <typename T>
constexpr bool is_device_coo_matrix_view_v = is_device_coo_matrix_view<T>::value;

template <typename T>
struct is_device_coo_matrix : std::false_type {};

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          template <typename T> typename ContainerPolicy,
          SparsityType sparsity_type>
struct is_device_coo_matrix<
  device_coo_matrix<ElementType, RowType, ColType, NZType, ContainerPolicy, sparsity_type>>
  : std::true_type {};

template <typename T>
constexpr bool is_device_coo_matrix_v = is_device_coo_matrix<T>::value;

template <typename T>
constexpr bool is_device_coo_sparsity_owning_v =
  is_device_coo_matrix<T>::value and T::get_sparsity_type() == OWNING;

template <typename T>
constexpr bool is_device_coo_sparsity_preserving_v =
  is_device_coo_matrix<T>::value and T::get_sparsity_type() == PRESERVING;

/**
 * Create a sparsity-owning sparse matrix in the coordinate format. sparsity-owning means that
 * all of the underlying vectors (data, indptr, indices) are owned by the coo_matrix instance. If
 * not known up front, the sparsity can be ignored in this factory function and `resize()` invoked
 * on the instance once the sparsity is known.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 *
 * raft::resources handle;
 * auto coo_matrix = raft::make_device_coo_matrix<float, int, int, int>(handle, n_rows, n_cols);
 * ...
 * // compute expected sparsity
 * ...
 * int nnz = 5000;
 * coo_matrix.initialize_sparsity(nnz);
 * @endcode
 *
 * @tparam ElementType
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] handle a raft handle for managing expensive device resources
 * @param[in] n_rows total number of rows in the matrix
 * @param[in] n_cols total number of columns in the matrix
 * @param[in] nnz number of non-zeros in the matrix if known [optional]
 * @return a sparsity-owning sparse matrix in coordinate (coo) format
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_device_coo_matrix(raft::resources const& handle,
                            RowType n_rows,
                            ColType n_cols,
                            NZType nnz = 0)
{
  return device_sparsity_owning_coo_matrix<ElementType, RowType, ColType, NZType>(
    handle, n_rows, n_cols, nnz);
}

/**
 * Create a sparsity-preserving sparse matrix in the coordinate format. sparsity-preserving means
 * that a view of the coo sparsity is supplied, allowing the values in the sparsity to change but
 * not the sparsity itself. The coo_matrix instance does not own the sparsity, the sparsity must
 * be known up front, and cannot be resized later.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 *
 * raft::resources handle;
 * auto coo_structure =
 *   raft::make_device_coordinate_structure<int, int, int>(handle, n_rows, n_cols);
 * ...
 * // compute expected sparsity
 * ...
 * int nnz = 5000;
 * coo_structure.initialize_sparsity(nnz);
 * auto coo_matrix = raft::make_device_coo_matrix<float>(handle, coo_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive device resources
 * @param[in] structure a sparsity-preserving coordinate structural view
 * @return a sparsity-preserving sparse matrix in coordinate (coo) format
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_device_coo_matrix(raft::resources const& handle,
                            device_coordinate_structure_view<RowType, ColType, NZType> structure)
{
  return device_sparsity_preserving_coo_matrix<ElementType, RowType, ColType, NZType>(handle,
                                                                                      structure);
}

/**
 * Create a non-owning sparse matrix view in the coordinate format. This is sparsity-preserving,
 * meaning that the underlying sparsity is known and cannot be changed. Use the sparsity-owning
 * coo_matrix if sparsity needs to be mutable.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following pointer is assumed to reference device memory for a size of nnz
 * float* d_elm_ptr = ...;
 *
 * raft::resources handle;
 * auto coo_structure = raft::make_device_coordinate_structure(handle, n_rows, n_cols, nnz);
 * auto coo_matrix_view = raft::make_device_coo_matrix_view(d_elm_ptr, coo_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] ptr a pointer to array of nonzero matrix elements on device (size nnz)
 * @param[in] structure a sparsity-preserving coordinate structural view
 * @return a sparsity-preserving sparse matrix in coordinate (coo) format
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_device_coo_matrix_view(
  ElementType* ptr, device_coordinate_structure_view<RowType, ColType, NZType> structure)
{
  return device_coo_matrix_view<ElementType, RowType, ColType, NZType>(
    raft::device_span<ElementType>(ptr, structure.get_nnz()), structure);
}

/**
 * Create a non-owning sparse matrix view in the coordinate format. This is sparsity-preserving,
 * meaning that the underlying sparsity is known and cannot be changed. Use the sparsity-owning
 * coo_matrix if sparsity needs to be mutable.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_span.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following span is assumed to be of size nnz
 * raft::device_span<float> d_elm_ptr;
 *
 * raft::resources handle;
 * auto coo_structure = raft::make_device_coordinate_structure(handle, n_rows, n_cols, nnz);
 * auto coo_matrix_view = raft::make_device_coo_matrix_view(d_elm_ptr, coo_structure.view());
 * @endcode
 *
 * @tparam ElementType
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] elements a device span containing nonzero matrix elements (size nnz)
 * @param[in] structure a sparsity-preserving coordinate structural view
 * @return a sparsity-preserving sparse matrix view in coordinate (coo) format
 */
template <typename ElementType, typename RowType, typename ColType, typename NZType>
auto make_device_coo_matrix_view(
  raft::device_span<ElementType> elements,
  device_coordinate_structure_view<RowType, ColType, NZType> structure)
{
  RAFT_EXPECTS(elements.size() == structure.get_nnz(),
               "Size of elements must be equal to the nnz from the structure");
  return device_coo_matrix_view<ElementType, RowType, ColType, NZType>(elements, structure);
}

/**
 * Create a sparsity-owning coordinate structure object. If not known up front, this object can be
 * resized() once the sparsity (number of non-zeros) is known, postponing the allocation of the
 * underlying data arrays.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * raft::resources handle;
 * auto coo_structure = raft::make_device_coordinate_structure(handle, n_rows, n_cols, nnz);
 * ...
 * // compute expected sparsity
 * ...
 * coo_structure.initialize_sparsity(nnz);
 * @endcode
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] handle raft handle for managing expensive resources on device
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of cols
 * @param[in] nnz number of non-zeros
 * @return a sparsity-owning coordinate structure instance
 */
template <typename RowType, typename ColType, typename NZType>
auto make_device_coordinate_structure(raft::resources const& handle,
                                      RowType n_rows,
                                      ColType n_cols,
                                      NZType nnz = 0)
{
  return device_coordinate_structure<RowType, ColType, NZType>(handle, n_rows, n_cols, nnz);
}

/**
 * Create a non-owning sparsity-preserved coordinate structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following pointers are assumed to reference device memory of size nnz
 * int *rows = ...;
 * int *cols = ...;
 *
 * raft::resources handle;
 * auto coo_structure =
 *   raft::make_device_coordinate_structure_view(rows, cols, n_rows, n_cols, nnz);
 * @endcode
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] rows pointer to row indices array on device (size nnz)
 * @param[in] cols pointer to column indices array on device (size nnz)
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of columns
 * @param[in] nnz number of non-zeros
 * @return a sparsity-preserving coordinate structural view
 */
template <typename RowType, typename ColType, typename NZType>
auto make_device_coordinate_structure_view(
  RowType* rows, ColType* cols, RowType n_rows, ColType n_cols, NZType nnz)
{
  return device_coordinate_structure_view<RowType, ColType, NZType>(
    raft::device_span<RowType>(rows, nnz), raft::device_span<ColType>(cols, nnz), n_rows, n_cols);
}

/**
 * Create a non-owning sparsity-preserved coordinate structure view. Sparsity-preserving means that
 * the underlying sparsity is known and cannot be changed. Use the sparsity-owning version if the
 * sparsity is not known up front.
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/device_coo_matrix.hpp>
 *
 * int n_rows = 100000;
 * int n_cols = 10000;
 * int nnz = 5000;
 *
 * // The following device spans are assumed to be of size nnz
 * raft::device_span<int> rows;
 * raft::device_span<int> cols;
 *
 * raft::resources handle;
 * auto coo_structure =
 *   raft::make_device_coordinate_structure_view<int, int, int>(rows, cols, n_rows, n_cols);
 * @endcode
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @param[in] rows a device span containing row indices (size nnz)
 * @param[in] cols a device span containing column indices (size nnz)
 * @param[in] n_rows total number of rows
 * @param[in] n_cols total number of columns
 * @return a sparsity-preserving coordinate structural view
 */
template <typename RowType, typename ColType, typename NZType>
auto make_device_coordinate_structure_view(raft::device_span<RowType> rows,
                                           raft::device_span<ColType> cols,
                                           RowType n_rows,
                                           ColType n_cols)
{
  return device_coordinate_structure_view<RowType, ColType, NZType>(rows, cols, n_rows, n_cols);
}

/** @} */

};  // namespace RAFT_EXPORT raft
