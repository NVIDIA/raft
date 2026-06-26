
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
 * \defgroup coo_matrix COO Matrix
 * @{
 */

template <typename RowType, typename ColType, typename NZType, int is_device>
class coordinate_structure_t : public sparse_structure<RowType, ColType, NZType, is_device> {
 public:
  coordinate_structure_t(RowType n_rows, ColType n_cols, NZType nnz)
    : sparse_structure<RowType, ColType, NZType, is_device>(n_rows, n_cols, nnz) {};

  /**
   * Return span containing underlying rows array
   * @return span containing underlying rows array
   */
  virtual span<RowType, is_device> get_rows() = 0;

  /**
   * Return span containing underlying cols array
   * @return span containing underlying cols array
   */
  virtual span<ColType, is_device> get_cols() = 0;
};

/**
 * A non-owning view into a coordinate structure
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam RowType
 * @tparam ColType
 */
template <typename RowType, typename ColType, typename NZType, bool is_device>
class coordinate_structure_view
  : public coordinate_structure_t<RowType, ColType, NZType, is_device> {
 public:
  static constexpr SparsityType sparsity_type = PRESERVING;
  using view_type = coordinate_structure_view<RowType, ColType, NZType, is_device>;
  using row_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::row_type;
  using col_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::col_type;
  using nnz_type  = typename sparse_structure<RowType, ColType, NZType, is_device>::nnz_type;

  coordinate_structure_view(span<row_type, is_device> rows,
                            span<col_type, is_device> cols,
                            row_type n_rows,
                            col_type n_cols)
    : coordinate_structure_t<RowType, ColType, NZType, is_device>(n_rows, n_cols, rows.size()),
      rows_{rows},
      cols_{cols}
  {
  }

  /**
   * Return span containing underlying rows array
   * @return span containing underlying rows array
   */
  span<row_type, is_device> get_rows() override { return rows_; }

  /**
   * Return span containing underlying cols array
   * @return span containing underlying cols array
   */
  span<col_type, is_device> get_cols() override { return cols_; }

 protected:
  raft::span<row_type, is_device> rows_;
  raft::span<col_type, is_device> cols_;
};

/**
 * Represents a sparse coordinate structure (or edge list)
 * which can be used to model a COO matrix.
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam RowType
 * @tparam ColType
 * @tparam ContainerPolicy
 */
template <typename RowType,
          typename ColType,
          typename NZType,
          bool is_device,
          template <typename T> typename ContainerPolicy>
class coordinate_structure : public coordinate_structure_t<RowType, ColType, NZType, is_device> {
 public:
  static constexpr SparsityType sparsity_type = OWNING;
  using sparse_structure_type = coordinate_structure_t<RowType, ColType, NZType, is_device>;
  using row_type              = typename sparse_structure_type::row_type;
  using col_type              = typename sparse_structure_type::col_type;
  using nnz_type              = typename sparse_structure_type::nnz_type;
  using view_type             = coordinate_structure_view<row_type, col_type, nnz_type, is_device>;
  using row_container_policy_type = ContainerPolicy<RowType>;
  using col_container_policy_type = ContainerPolicy<ColType>;
  using row_container_type        = typename row_container_policy_type::container_type;
  using col_container_type        = typename col_container_policy_type::container_type;

  coordinate_structure(
    raft::resources const& handle,
    row_type n_rows,
    col_type n_cols,
    nnz_type nnz = 0) noexcept(std::is_nothrow_default_constructible_v<row_container_type>)
    : coordinate_structure_t<RowType, ColType, NZType, is_device>(n_rows, n_cols, nnz),
      cp_rows_{},
      cp_cols_{},
      c_rows_{cp_rows_.create(handle, nnz)},
      c_cols_{cp_cols_.create(handle, nnz)} {};

  coordinate_structure(coordinate_structure const&) noexcept(
    std::is_nothrow_copy_constructible_v<row_container_type>) = default;
  coordinate_structure(coordinate_structure&&) noexcept(
    std::is_nothrow_move_constructible<row_container_type>::value) = default;

  constexpr auto operator=(coordinate_structure const&) noexcept(
    std::is_nothrow_copy_assignable<row_container_type>::value) -> coordinate_structure& = default;
  constexpr auto operator=(coordinate_structure&&) noexcept(
    std::is_nothrow_move_assignable<row_container_type>::value) -> coordinate_structure& = default;

  ~coordinate_structure() noexcept(std::is_nothrow_destructible<row_container_type>::value) =
    default;

  /**
   * Return a view of the coordinate structure. Structural views are sparsity-preserving
   * so while the structural elements can be updated in a non-const view, the sparsity
   * itself (number of nonzeros) cannot be changed.
   * @return coordinate structure view
   */
  view_type view()
  {
    if (this->get_nnz() == 0) {
      RAFT_LOG_WARN(
        "Cannot create coordinate_structure.view() because it has not been initialized "
        "(sparsity is 0)");
    }
    auto row_span = raft::span<row_type, is_device>(c_rows_.data(), this->get_nnz());
    auto col_span = raft::span<col_type, is_device>(c_cols_.data(), this->get_nnz());
    return view_type(row_span, col_span, this->get_n_rows(), this->get_n_cols());
  }

  /**
   * Return span containing underlying rows array
   * @return span containing underlying rows array
   */
  span<row_type, is_device> get_rows() override
  {
    return raft::span<row_type, is_device>(c_rows_.data(), this->get_n_rows());
  }

  /**
   * Return span containing underlying cols array
   * @return span containing underlying cols array
   */
  span<col_type, is_device> get_cols() override
  {
    return raft::span<col_type, is_device>(c_cols_.data(), this->get_n_cols());
  }

  /**
   * Change the sparsity of the current compressed structure. This will
   * resize the underlying data arrays.
   * @param nnz new sparsity
   */
  void initialize_sparsity(nnz_type nnz)
  {
    sparse_structure_type::initialize_sparsity(nnz);
    c_rows_.reallocate(nnz);
    c_cols_.reallocate(nnz);
  }

 protected:
  row_container_policy_type cp_rows_;
  col_container_policy_type cp_cols_;
  row_container_type c_rows_;
  col_container_type c_cols_;
};

template <typename ElementType, typename RowType, typename ColType, typename NZType, bool is_device>
class coo_matrix_view
  : public sparse_matrix_view<ElementType,
                              coordinate_structure_view<RowType, ColType, NZType, is_device>,
                              is_device> {
 public:
  using element_type = ElementType;
  using row_type     = RowType;
  using col_type     = ColType;
  using nnz_type     = NZType;
  coo_matrix_view(raft::span<ElementType, is_device> element_span,
                  coordinate_structure_view<RowType, ColType, NZType, is_device> structure_view)
    : sparse_matrix_view<ElementType,
                         coordinate_structure_view<RowType, ColType, NZType, is_device>,
                         is_device>(element_span, structure_view)
  {
  }
};

template <typename ElementType,
          typename RowType,
          typename ColType,
          typename NZType,
          bool is_device,
          template <typename T> typename ContainerPolicy,
          SparsityType sparsity_type = SparsityType::OWNING,
          typename structure_type    = std::conditional_t<
               sparsity_type == SparsityType::OWNING,
               coordinate_structure<RowType, ColType, NZType, is_device, ContainerPolicy>,
               coordinate_structure_view<RowType, ColType, NZType, is_device>>>
class coo_matrix
  : public sparse_matrix<ElementType,
                         structure_type,
                         coo_matrix_view<ElementType, RowType, ColType, NZType, is_device>,
                         is_device,
                         ContainerPolicy> {
 public:
  using element_type        = ElementType;
  using row_type            = RowType;
  using col_type            = ColType;
  using nnz_type            = NZType;
  using structure_view_type = typename structure_type::view_type;
  using container_type      = typename ContainerPolicy<ElementType>::container_type;
  using sparse_matrix_type =
    sparse_matrix<ElementType,
                  structure_type,
                  coo_matrix_view<ElementType, RowType, ColType, NZType, is_device>,
                  is_device,
                  ContainerPolicy>;
  static constexpr auto get_sparsity_type() { return sparsity_type; }
  template <SparsityType sparsity_type_ = get_sparsity_type(),
            typename = typename std::enable_if_t<sparsity_type_ == SparsityType::OWNING>>
  coo_matrix(raft::resources const& handle,
             RowType n_rows,
             ColType n_cols,
             NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix_type(handle, n_rows, n_cols, nnz){};

  // Constructor that owns the data but not the structure
  template <SparsityType sparsity_type_ = get_sparsity_type(),
            typename = typename std::enable_if_t<sparsity_type_ == SparsityType::PRESERVING>>
  coo_matrix(raft::resources const& handle, structure_type structure) noexcept(
    std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix_type(handle, structure){};

  /**
   * Initialize the sparsity on this instance if it was not known upon construction
   * Please note this will resize the underlying memory buffers
   * @param nnz new sparsity to initialize.
   */
  template <SparsityType sparsity_type_ = get_sparsity_type(),
            typename = typename std::enable_if_t<sparsity_type_ == SparsityType::OWNING>>
  void initialize_sparsity(NZType nnz)
  {
    sparse_matrix_type::initialize_sparsity(nnz);
    this->structure_.initialize_sparsity(nnz);
  }

  /**
   * Return a view of the structure underlying this matrix
   * @return
   */
  structure_view_type structure_view()
  {
    if constexpr (get_sparsity_type() == SparsityType::OWNING) {
      return this->structure_.view();
    } else {
      return this->structure_;
    }
  }
};

/** @} */

}  // namespace RAFT_EXPORT raft
