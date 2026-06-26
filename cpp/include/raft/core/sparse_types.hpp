/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/span.hpp>
#include <raft/core/sparse_types.hpp>

namespace RAFT_EXPORT raft {

/**
 * \defgroup sparse_types Sparse API vocabulary
 * @{
 */

enum SparsityType { OWNING, PRESERVING };

/**
 * Maintains metadata about the structure and sparsity of a sparse matrix.
 * @tparam RowType
 * @tparam ColType
 * @tparam NZType
 * @tparam is_device
 */
template <typename RowType, typename ColType, typename NZType, int is_device>
class sparse_structure {
 public:
  using row_type = RowType;
  using col_type = ColType;
  using nnz_type = NZType;

  /**
   * Constructor when sparsity is already known
   * @param n_rows total number of rows in matrix
   * @param n_cols total number of columns in matrix
   * @param nnz sparsity of matrix
   */
  sparse_structure(row_type n_rows, col_type n_cols, nnz_type nnz)
    : n_rows_(n_rows), n_cols_(n_cols), nnz_(nnz) {};

  /**
   * Constructor when sparsity is not yet known
   * @param n_rows total number of rows in matrix
   * @param n_cols total number of columns in matrix
   */
  sparse_structure(row_type n_rows, col_type n_cols) : n_rows_(n_rows), n_cols_(n_cols), nnz_(0) {}

  /**
   * Return the sparsity of the matrix (this will be 0 when sparsity is not yet known)
   * @return sparsity of matrix
   */
  nnz_type get_nnz() { return nnz_; }

  /**
   * Return the total number of rows in the matrix
   * @return total number of rows in the matriz
   */
  row_type get_n_rows() { return n_rows_; }

  /**
   * Return the total number of columns in the matrix
   * @return total number of columns
   */
  col_type get_n_cols() { return n_cols_; }

  /**
   * Initialize the matrix sparsity when it was not known
   * upon construction.
   * @param nnz
   */
  virtual void initialize_sparsity(nnz_type nnz) { nnz_ = nnz; }

 protected:
  row_type n_rows_;
  col_type n_cols_;
  nnz_type nnz_;
};

/**
 * A non-owning view of a sparse matrix, which includes a
 * structure component coupled with its elements/weights
 *
 * @tparam ElementType
 * @tparam sparse_structure
 */
template <typename ElementType, typename StructureType, bool is_device>
class sparse_matrix_view {
 public:
  using element_type        = ElementType;
  using structure_view_type = typename StructureType::view_type;

  sparse_matrix_view(raft::span<ElementType, is_device> element_span,
                     structure_view_type structure_view)
    : element_span_(element_span), structure_view_(structure_view)
  {
    // FIXME: Validate structure sizes match span size.
  }

  /**
   * Return a view of the structure underlying this matrix
   * @return
   */
  structure_view_type structure_view() { return structure_view_; }

  /**
   * Return a span of the nonzero elements of the matrix
   * @return span of the nonzero elements of the matrix
   */
  span<element_type, is_device> get_elements() { return element_span_; }

 protected:
  raft::span<element_type, is_device> element_span_;
  structure_view_type structure_view_;
};

/**
 * TODO: Need to support the following types of configurations:
 * 1. solid: immutable_sparse_matrix_view<const ElementType, const StructureType>
 *      - This is an immutable view type, nothing can change.
 * 2. liquid: sparse_matrix<ElementType, const StructureType>
 *      - sparse_matrix owning container w/ StructureType=immutable view?
 * 3. gas: sparse_matrix<ElementType, StructureType>
 *      - sparse_matrix owning container w/ StructureType owning container?
 */

/**
 * An owning container for a sparse matrix, which includes a
 * structure component coupled with its elements/weights
 * @tparam ElementType
 * @tparam sparse_structure
 * @tparam ContainerPolicy
 */
template <typename ElementType,
          typename StructureType,
          typename ViewType,
          bool is_device,
          template <typename T> typename ContainerPolicy>
class sparse_matrix {
 public:
  using view_type      = ViewType;
  using element_type   = typename view_type::element_type;
  using structure_type = StructureType;
  using row_type       = typename structure_type::row_type;
  using col_type       = typename structure_type::col_type;
  using nnz_type       = typename structure_type::nnz_type;

  using structure_view_type   = typename structure_type::view_type;
  using container_policy_type = ContainerPolicy<element_type>;
  using container_type        = typename container_policy_type::container_type;

  // constructor that owns the data and the structure
  sparse_matrix(raft::resources const& handle,
                row_type n_rows,
                col_type n_cols,
                nnz_type nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : structure_{handle, n_rows, n_cols, nnz}, cp_{}, c_elements_{cp_.create(handle, nnz)} {};

  // Constructor that owns the data but not the structure
  // This constructor is only callable with a `structure_type == *_structure_view`
  // which makes it okay to copy
  sparse_matrix(raft::resources const& handle, structure_type structure) noexcept(
    std::is_nothrow_default_constructible_v<container_type>)
    : structure_{structure}, cp_{}, c_elements_{cp_.create(handle, structure_.get_nnz())} {};

  constexpr sparse_matrix(sparse_matrix const&) noexcept(
    std::is_nothrow_copy_constructible_v<container_type>) = default;
  constexpr sparse_matrix(sparse_matrix&&) noexcept(
    std::is_nothrow_move_constructible<container_type>::value) = default;

  constexpr auto operator=(sparse_matrix const&) noexcept(
    std::is_nothrow_copy_assignable<container_type>::value) -> sparse_matrix& = default;
  constexpr auto operator=(sparse_matrix&&) noexcept(
    std::is_nothrow_move_assignable<container_type>::value) -> sparse_matrix& = default;

  ~sparse_matrix() noexcept(std::is_nothrow_destructible<container_type>::value) = default;

  void initialize_sparsity(nnz_type nnz) { c_elements_.reallocate(nnz); };

  raft::span<ElementType, is_device> get_elements()
  {
    return raft::span<ElementType, is_device>(c_elements_.data(), structure_.get_nnz());
  }

  /**
   * Return a view of the structure underlying this matrix
   * @return
   */
  virtual structure_view_type structure_view() = 0;

  /**
   * Return a sparsity-preserving view of this sparse matrix
   * @return view of this sparse matrix
   */
  view_type view()
  {
    auto struct_view = structure_view();
    auto element_span =
      raft::span<ElementType, is_device>(c_elements_.data(), struct_view.get_nnz());
    return view_type(element_span, struct_view);
  }

 protected:
  structure_type structure_;
  container_policy_type cp_;
  container_type c_elements_;
};

/* @} */

}  // namespace RAFT_EXPORT raft
