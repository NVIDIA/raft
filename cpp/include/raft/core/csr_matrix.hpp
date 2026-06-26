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
 * \defgroup csr_matrix CSR Matrix
 * @{
 */

template <typename IndptrType, typename IndicesType, typename NZType, int is_device>
class compressed_structure_t : public sparse_structure<IndptrType, IndicesType, NZType, is_device> {
 public:
  /**
   * Constructor when sparsity is already known
   * @param n_rows total number of rows in matrix
   * @param n_cols total number of columns in matrix
   * @param nnz sparsity of matrix
   */
  compressed_structure_t(IndptrType n_rows, IndicesType n_cols, NZType nnz)
    : sparse_structure<IndptrType, IndicesType, NZType, is_device>(n_rows, n_cols, nnz) {};

  /**
   * Return span containing underlying indptr array
   * @return span containing underlying indptr array
   */
  virtual span<IndptrType, is_device> get_indptr() = 0;

  /**
   * Return span containing underlying indices array
   * @return span containing underlying indices array
   */
  virtual span<IndicesType, is_device> get_indices() = 0;
};

/**
 * A non-owning view into a compressed sparse structure
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam IndptrType
 * @tparam IndicesType
 */
template <typename IndptrType, typename IndicesType, typename NZType, bool is_device>
class compressed_structure_view
  : public compressed_structure_t<IndptrType, IndicesType, NZType, is_device> {
 public:
  using sparse_structure_type = compressed_structure_t<IndptrType, IndicesType, NZType, is_device>;
  using view_type    = compressed_structure_view<IndptrType, IndicesType, NZType, is_device>;
  using indptr_type  = typename sparse_structure_type::row_type;
  using indices_type = typename sparse_structure_type::col_type;
  using nnz_type     = typename sparse_structure_type::nnz_type;

  compressed_structure_view(span<indptr_type, is_device> indptr,
                            span<indices_type, is_device> indices,
                            indices_type n_cols)
    : sparse_structure_type(indptr.size() - 1, n_cols, indices.size()),
      indptr_(indptr),
      indices_(indices)
  {
  }

  /**
   * Return span containing underlying indptr array
   * @return span containing underlying indptr array
   */
  span<indptr_type, is_device> get_indptr() override { return indptr_; }

  /**
   * Return span containing underlying indices array
   * @return span containing underlying indices array
   */
  span<indices_type, is_device> get_indices() override { return indices_; }

 protected:
  raft::span<indptr_type, is_device> indptr_;
  raft::span<indices_type, is_device> indices_;
};

/**
 * Represents a sparse compressed structure (or adjacency list)
 * which can be used to model both a CSR and CSC matrix.
 *
 * The structure representation does not have a value/weight
 * component so that its const-ness can be varied from it.
 *
 * @tparam IndptrType
 * @tparam IndicesType
 * @tparam ContainerPolicy
 */
template <typename IndptrType,
          typename IndicesType,
          typename NZType,
          bool is_device,
          template <typename T> typename ContainerPolicy>
class compressed_structure
  : public compressed_structure_t<IndptrType, IndicesType, NZType, is_device> {
 public:
  using sparse_structure_type = compressed_structure_t<IndptrType, IndicesType, NZType, is_device>;
  using indptr_type           = typename sparse_structure_type::row_type;
  using indices_type          = typename sparse_structure_type::col_type;
  using nnz_type              = typename sparse_structure_type::nnz_type;
  using view_type = compressed_structure_view<IndptrType, IndicesType, NZType, is_device>;
  using indptr_container_policy_type  = ContainerPolicy<IndptrType>;
  using indices_container_policy_type = ContainerPolicy<IndicesType>;
  using indptr_container_type         = typename indptr_container_policy_type::container_type;
  using indices_container_type        = typename indices_container_policy_type::container_type;

  constexpr compressed_structure(
    raft::resources const& handle,
    IndptrType n_rows,
    IndicesType n_cols,
    NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<indptr_container_type>)
    : sparse_structure_type{n_rows, n_cols, nnz},
      cp_indptr_{},
      cp_indices_{},
      c_indptr_{cp_indptr_.create(handle, n_rows + 1)},
      c_indices_{cp_indices_.create(handle, nnz)} {};

  compressed_structure(compressed_structure const&) noexcept(
    std::is_nothrow_copy_constructible_v<indptr_container_type>) = default;
  compressed_structure(compressed_structure&&) noexcept(
    std::is_nothrow_move_constructible<indptr_container_type>::value) = default;

  constexpr auto operator=(compressed_structure const&) noexcept(
    std::is_nothrow_copy_assignable<indptr_container_type>::value)
    -> compressed_structure& = default;
  constexpr auto operator=(compressed_structure&&) noexcept(
    std::is_nothrow_move_assignable<indptr_container_type>::value)
    -> compressed_structure& = default;

  /**
   * Return span containing underlying indptr array
   * @return span containing underlying indptr array
   */
  span<IndptrType, is_device> get_indptr() override
  {
    return raft::span<IndptrType, is_device>(c_indptr_.data(), this->get_n_rows() + 1);
  }

  /**
   * Return span containing underlying indices array
   * @return span containing underlying indices array
   */
  span<IndicesType, is_device> get_indices() override
  {
    if (this->get_nnz() == 0) {
      RAFT_LOG_WARN("Indices requested for structure that has uninitialized sparsity.");
    }
    return raft::span<IndicesType, is_device>(c_indices_.data(), this->get_nnz());
  }

  ~compressed_structure() noexcept(std::is_nothrow_destructible<indptr_container_type>::value) =
    default;

  /**
   * Return a view of the compressed structure. Structural views are sparsity-preserving
   * so while the structural elements can be updated in a non-const view, the sparsity
   * itself (number of nonzeros) cannot be changed.
   * @return compressed structure view
   */
  view_type view()
  {
    if (this->get_nnz() == 0) {
      RAFT_LOG_WARN(
        "Cannot create compressed_structure.view() because it has not been initialized (sparsity "
        "is 0)");
    }
    auto indptr_span  = raft::span<IndptrType, is_device>(c_indptr_.data(), this->get_n_rows() + 1);
    auto indices_span = raft::span<IndicesType, is_device>(c_indices_.data(), this->get_nnz());
    return view_type(indptr_span, indices_span, this->get_n_cols());
  }

  /**
   * Change the sparsity of the current compressed structure. This will
   * resize the underlying data arrays.
   * @param nnz new sparsity
   */
  void initialize_sparsity(NZType nnz) override
  {
    sparse_structure_type::initialize_sparsity(nnz);
    c_indptr_.reallocate(this->get_n_rows() + 1);
    c_indices_.reallocate(nnz);
  }

 protected:
  indptr_container_policy_type cp_indptr_;
  indices_container_policy_type cp_indices_;
  indptr_container_type c_indptr_;
  indices_container_type c_indices_;
};
template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          bool is_device>
class csr_matrix_view
  : public sparse_matrix_view<ElementType,
                              compressed_structure_view<IndptrType, IndicesType, NZType, is_device>,
                              is_device> {
 public:
  using element_type = ElementType;
  using indptr_type  = IndptrType;
  using indices_type = IndicesType;
  using nnz_type     = NZType;
  csr_matrix_view(
    raft::span<ElementType, is_device> element_span,
    compressed_structure_view<IndptrType, IndicesType, NZType, is_device> structure_view)
    : sparse_matrix_view<ElementType,
                         compressed_structure_view<IndptrType, IndicesType, NZType, is_device>,
                         is_device>(element_span, structure_view) {};
};

template <typename ElementType,
          typename IndptrType,
          typename IndicesType,
          typename NZType,
          bool is_device,
          template <typename T> typename ContainerPolicy,
          SparsityType sparsity_type = SparsityType::OWNING,
          typename structure_type    = std::conditional_t<
               sparsity_type == SparsityType::OWNING,
               compressed_structure<IndptrType, IndicesType, NZType, is_device, ContainerPolicy>,
               compressed_structure_view<IndptrType, IndicesType, NZType, is_device>>>
class csr_matrix
  : public sparse_matrix<ElementType,
                         structure_type,
                         csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
                         is_device,
                         ContainerPolicy> {
 public:
  using element_type        = ElementType;
  using indptr_type         = IndptrType;
  using indices_type        = IndicesType;
  using nnz_type            = NZType;
  using structure_view_type = typename structure_type::view_type;
  static constexpr auto get_sparsity_type() { return sparsity_type; }
  using sparse_matrix_type =
    sparse_matrix<ElementType,
                  structure_type,
                  csr_matrix_view<ElementType, IndptrType, IndicesType, NZType, is_device>,
                  is_device,
                  ContainerPolicy>;
  using container_type = typename ContainerPolicy<ElementType>::container_type;

  template <SparsityType sparsity_type_ = get_sparsity_type(),
            typename = typename std::enable_if_t<sparsity_type_ == SparsityType::OWNING>>
  csr_matrix(raft::resources const& handle,
             IndptrType n_rows,
             IndicesType n_cols,
             NZType nnz = 0) noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix_type(handle, n_rows, n_cols, nnz){};

  // Constructor that owns the data but not the structure

  template <SparsityType sparsity_type_ = get_sparsity_type(),
            typename = typename std::enable_if_t<sparsity_type_ == SparsityType::PRESERVING>>
  csr_matrix(raft::resources const& handle, structure_type structure) noexcept(
    std::is_nothrow_default_constructible_v<container_type>)
    : sparse_matrix_type(handle, structure){};

  /**
   * Initialize the sparsity on this instance if it was not known upon construction
   * Please note this will resize the underlying memory buffers
   * @param nnz new sparsity to initialize.
   */
  template <typename = std::enable_if<sparsity_type == SparsityType::OWNING>>
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
