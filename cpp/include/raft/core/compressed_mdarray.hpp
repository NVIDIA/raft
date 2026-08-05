/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/mdspan.hpp>

#include <cstdint>
#include <type_traits>

namespace raft {

// ============================================================================
// compressed_data_handle: the data_handle_type for compressed mdspan/mdarray
// ============================================================================

template <typename CodebookView, typename CodesView>
struct compressed_data_handle {
  CodebookView codebook;
  CodesView codes;
  size_t base_offset = 0;
};

// ============================================================================
// Reconstruction specs: the only compression-type-specific code
// ============================================================================

/**
 * @brief Scalar quantization reconstruction spec (matches cuvs global min/max SQ).
 *
 * Codebook is a [2] vector holding {min, max}. Scale and offset are derived
 * at construction time. Codes are int8_t in [-128, 127].
 */
template <typename MathT, typename CodebookView, typename CodesView>
struct sq_spec {
  using math_type          = std::remove_const_t<MathT>;
  using codebook_view_type = CodebookView;
  using codes_view_type    = CodesView;

  math_type scale;
  math_type offset;

  RAFT_INLINE_FUNCTION
  static uint32_t dim(compressed_data_handle<CodebookView, CodesView> const& h)
  {
    return static_cast<uint32_t>(h.codes.extent(1));
  }

  RAFT_INLINE_FUNCTION
  math_type reconstruct(compressed_data_handle<CodebookView, CodesView> const& h,
                        size_t row,
                        size_t col) const
  {
    auto code = h.codes(row, col);
    return static_cast<math_type>((static_cast<math_type>(code) - offset) / scale);
  }
};

template <typename MathT, typename CodebookView, typename CodesView>
auto make_sq_spec(MathT min_val, MathT max_val) -> sq_spec<MathT, CodebookView, CodesView>
{
  using T  = std::remove_const_t<MathT>;
  T q_min  = T(-128);
  T q_max  = T(127);
  T range  = static_cast<T>(max_val) - static_cast<T>(min_val);
  T scale  = (range > T(0)) ? ((q_max - q_min) / range) : T(1);
  T offset = q_min - static_cast<T>(min_val) * scale;
  return {scale, offset};
}

/**
 * @brief PQ reconstruction spec with a single global codebook (stateless).
 *
 * Codebook shape: [n_centers, pq_len]. Codes shape: [n_rows, pq_dim].
 * All parameters are derived from the handle views at access time.
 */
template <typename MathT, typename CodebookView, typename CodesView>
struct pq_spec {
  using math_type          = std::remove_const_t<MathT>;
  using codebook_view_type = CodebookView;
  using codes_view_type    = CodesView;

  RAFT_INLINE_FUNCTION
  static uint32_t dim(compressed_data_handle<CodebookView, CodesView> const& h)
  {
    return static_cast<uint32_t>(h.codes.extent(1)) * static_cast<uint32_t>(h.codebook.extent(1));
  }

  RAFT_INLINE_FUNCTION
  static math_type reconstruct(compressed_data_handle<CodebookView, CodesView> const& h,
                               size_t row,
                               size_t col)
  {
    uint32_t pq_len    = static_cast<uint32_t>(h.codebook.extent(1));
    uint32_t subspace  = static_cast<uint32_t>(col) / pq_len;
    uint32_t component = static_cast<uint32_t>(col) % pq_len;
    uint32_t code      = static_cast<uint32_t>(h.codes(row, subspace));
    return static_cast<math_type>(h.codebook(code, component));
  }
};

/**
 * @brief PQ reconstruction spec with per-subspace codebooks (rank-3, stateless).
 *
 * Codebook shape: [pq_dim, pq_len, n_centers] (IVF-PQ convention).
 * Codes shape: [n_rows, pq_dim].
 * All parameters are derived from the handle views at access time.
 */
template <typename MathT, typename CodebookView, typename CodesView>
struct pq_subspace_spec {
  using math_type          = std::remove_const_t<MathT>;
  using codebook_view_type = CodebookView;
  using codes_view_type    = CodesView;

  RAFT_INLINE_FUNCTION
  static uint32_t dim(compressed_data_handle<CodebookView, CodesView> const& h)
  {
    return static_cast<uint32_t>(h.codes.extent(1)) * static_cast<uint32_t>(h.codebook.extent(1));
  }

  RAFT_INLINE_FUNCTION
  static math_type reconstruct(compressed_data_handle<CodebookView, CodesView> const& h,
                               size_t row,
                               size_t col)
  {
    uint32_t pq_len    = static_cast<uint32_t>(h.codebook.extent(1));
    uint32_t subspace  = static_cast<uint32_t>(col) / pq_len;
    uint32_t component = static_cast<uint32_t>(col) % pq_len;
    uint32_t code      = static_cast<uint32_t>(h.codes(row, subspace));
    return static_cast<math_type>(h.codebook(subspace, component, code));
  }
};

// ============================================================================
// compressed_accessor: the accessor_policy for compressed mdspan
// ============================================================================

template <typename MathT, typename ReconstructSpec>
struct compressed_accessor {
  using element_type       = MathT;
  using codebook_view_type = typename ReconstructSpec::codebook_view_type;
  using codes_view_type    = typename ReconstructSpec::codes_view_type;
  using data_handle_type   = compressed_data_handle<codebook_view_type, codes_view_type>;
  using reference          = std::remove_const_t<MathT>;
  using offset_policy      = compressed_accessor;

  ReconstructSpec spec;

  compressed_accessor() = default;
  constexpr explicit compressed_accessor(ReconstructSpec s) : spec(s) {}

  RAFT_INLINE_FUNCTION
  reference access(data_handle_type h, size_t off) const
  {
    size_t d     = spec.dim(h);
    size_t total = h.base_offset + off;
    size_t row   = total / d;
    size_t col   = total % d;
    return spec.reconstruct(h, row, col);
  }

  RAFT_INLINE_FUNCTION
  data_handle_type offset(data_handle_type h, size_t i) const noexcept
  {
    return {h.codebook, h.codes, h.base_offset + i};
  }
};

// ============================================================================
// compressed_container: owns codebook and codes as inner mdarrays
// ============================================================================

/**
 * @brief Owning container that stores a codebook mdarray and a codes mdarray.
 *
 * The container is a simple pair — no reconstruction logic here.
 * The container_policy's access() handles reconstruction via the spec.
 */
template <typename CodebookMdarray, typename CodesMdarray>
class compressed_container {
 public:
  using codebook_view_type = typename CodebookMdarray::const_view_type;
  using codes_view_type    = typename CodesMdarray::const_view_type;

  using value_type      = typename CodebookMdarray::element_type;
  using size_type       = size_t;
  using reference       = std::remove_const_t<value_type>;
  using const_reference = reference;
  using pointer         = compressed_data_handle<codebook_view_type, codes_view_type>;
  using const_pointer   = pointer;

 private:
  CodebookMdarray codebook_;
  CodesMdarray codes_;

 public:
  compressed_container(CodebookMdarray&& codebook, CodesMdarray&& codes)
    : codebook_(std::move(codebook)), codes_(std::move(codes))
  {
  }

  [[nodiscard]] auto data() const noexcept -> pointer
  {
    return {codebook_.view(), codes_.view(), 0};
  }

  [[nodiscard]] auto data() noexcept -> pointer
  {
    return const_cast<compressed_container const*>(this)->data();
  }

  [[nodiscard]] auto codebook() const noexcept -> CodebookMdarray const& { return codebook_; }
  [[nodiscard]] auto codes() const noexcept -> CodesMdarray const& { return codes_; }
};

// ============================================================================
// compressed_container_policy: bridges container and accessor for mdarray
// ============================================================================

template <typename MathT, typename ReconstructSpec, typename CodebookMdarray, typename CodesMdarray>
class compressed_container_policy {
 public:
  using element_type   = MathT;
  using spec_type      = ReconstructSpec;
  using container_type = compressed_container<CodebookMdarray, CodesMdarray>;

  using pointer         = typename container_type::pointer;
  using const_pointer   = typename container_type::const_pointer;
  using reference       = typename container_type::reference;
  using const_reference = typename container_type::const_reference;

  using accessor_policy       = compressed_accessor<MathT, ReconstructSpec>;
  using const_accessor_policy = compressed_accessor<MathT const, ReconstructSpec>;

 private:
  spec_type spec_{};

 public:
  compressed_container_policy() = default;

  explicit compressed_container_policy(spec_type spec) : spec_(spec) {}

  [[nodiscard]] constexpr auto access(container_type& c, size_t n) const noexcept -> reference
  {
    auto acc = accessor_policy{spec_};
    return acc.access(c.data(), n);
  }

  [[nodiscard]] constexpr auto access(container_type const& c, size_t n) const noexcept
    -> const_reference
  {
    auto acc = const_accessor_policy{spec_};
    return acc.access(c.data(), n);
  }

  [[nodiscard]] auto make_accessor_policy() noexcept { return accessor_policy{spec_}; }
  [[nodiscard]] auto make_accessor_policy() const noexcept { return const_accessor_policy{spec_}; }
};

// ============================================================================
// Type aliases: mdspan views
// ============================================================================

// ---- SQ ----

template <typename MathT, typename IdxT = std::uint32_t>
using sq_host_matrix_view =
  host_mdspan<MathT const,
              matrix_extent<IdxT>,
              layout_c_contiguous,
              compressed_accessor<MathT const,
                                  sq_spec<MathT,
                                          host_vector_view<MathT const, std::uint32_t>,
                                          host_matrix_view<int8_t const, IdxT>>>>;

template <typename MathT, typename IdxT = std::uint32_t>
using sq_device_matrix_view =
  device_mdspan<MathT const,
                matrix_extent<IdxT>,
                layout_c_contiguous,
                compressed_accessor<MathT const,
                                    sq_spec<MathT,
                                            device_vector_view<MathT const, std::uint32_t>,
                                            device_matrix_view<int8_t const, IdxT>>>>;

// ---- PQ global ----

template <typename MathT, typename IdxT = std::uint32_t>
using pq_host_matrix_view =
  host_mdspan<MathT const,
              matrix_extent<IdxT>,
              layout_c_contiguous,
              compressed_accessor<MathT const,
                                  pq_spec<MathT,
                                          host_matrix_view<MathT const, std::uint32_t>,
                                          host_matrix_view<std::uint8_t const, IdxT>>>>;

template <typename MathT, typename IdxT = std::uint32_t>
using pq_device_matrix_view =
  device_mdspan<MathT const,
                matrix_extent<IdxT>,
                layout_c_contiguous,
                compressed_accessor<MathT const,
                                    pq_spec<MathT,
                                            device_matrix_view<MathT const, std::uint32_t>,
                                            device_matrix_view<std::uint8_t const, IdxT>>>>;

// ---- PQ per-subspace (rank-3 codebook) ----

template <typename MathT>
using host_pq_subspace_codebook_view =
  host_mdspan<MathT const, extents<std::uint32_t, dynamic_extent, dynamic_extent, dynamic_extent>>;

template <typename MathT>
using device_pq_subspace_codebook_view =
  device_mdspan<MathT const,
                extents<std::uint32_t, dynamic_extent, dynamic_extent, dynamic_extent>>;

template <typename MathT, typename IdxT = std::uint32_t>
using pq_subspace_host_matrix_view =
  host_mdspan<MathT const,
              matrix_extent<IdxT>,
              layout_c_contiguous,
              compressed_accessor<MathT const,
                                  pq_subspace_spec<MathT,
                                                   host_pq_subspace_codebook_view<MathT>,
                                                   host_matrix_view<std::uint8_t const, IdxT>>>>;

template <typename MathT, typename IdxT = std::uint32_t>
using pq_subspace_device_matrix_view = device_mdspan<
  MathT const,
  matrix_extent<IdxT>,
  layout_c_contiguous,
  compressed_accessor<MathT const,
                      pq_subspace_spec<MathT,
                                       device_pq_subspace_codebook_view<MathT>,
                                       device_matrix_view<std::uint8_t const, IdxT>>>>;

// ============================================================================
// Type aliases: inner codebook mdarrays (rank-3 for PQ-subspace)
// ============================================================================

template <typename MathT>
using host_pq_subspace_codebook =
  host_mdarray<MathT, extents<std::uint32_t, dynamic_extent, dynamic_extent, dynamic_extent>>;

template <typename MathT>
using device_pq_subspace_codebook =
  device_mdarray<MathT, extents<std::uint32_t, dynamic_extent, dynamic_extent, dynamic_extent>>;

// ============================================================================
// Type aliases: owning mdarray
// ============================================================================

// ---- SQ ----

template <typename MathT, typename IdxT = std::uint32_t>
using sq_host_matrix =
  host_mdarray<MathT,
               matrix_extent<IdxT>,
               layout_c_contiguous,
               compressed_container_policy<MathT,
                                           sq_spec<MathT,
                                                   host_vector_view<MathT const, std::uint32_t>,
                                                   host_matrix_view<int8_t const, IdxT>>,
                                           host_vector<MathT, std::uint32_t>,
                                           host_matrix<int8_t, IdxT>>>;

template <typename MathT, typename IdxT = std::uint32_t>
using sq_device_matrix =
  device_mdarray<MathT,
                 matrix_extent<IdxT>,
                 layout_c_contiguous,
                 compressed_container_policy<MathT,
                                             sq_spec<MathT,
                                                     device_vector_view<MathT const, std::uint32_t>,
                                                     device_matrix_view<int8_t const, IdxT>>,
                                             device_vector<MathT, std::uint32_t>,
                                             device_matrix<int8_t, IdxT>>>;

// ---- PQ global ----

template <typename MathT, typename IdxT = std::uint32_t>
using pq_host_matrix =
  host_mdarray<MathT,
               matrix_extent<IdxT>,
               layout_c_contiguous,
               compressed_container_policy<MathT,
                                           pq_spec<MathT,
                                                   host_matrix_view<MathT const, std::uint32_t>,
                                                   host_matrix_view<std::uint8_t const, IdxT>>,
                                           host_matrix<MathT, std::uint32_t>,
                                           host_matrix<std::uint8_t, IdxT>>>;

template <typename MathT, typename IdxT = std::uint32_t>
using pq_device_matrix =
  device_mdarray<MathT,
                 matrix_extent<IdxT>,
                 layout_c_contiguous,
                 compressed_container_policy<MathT,
                                             pq_spec<MathT,
                                                     device_matrix_view<MathT const, std::uint32_t>,
                                                     device_matrix_view<std::uint8_t const, IdxT>>,
                                             device_matrix<MathT, std::uint32_t>,
                                             device_matrix<std::uint8_t, IdxT>>>;

// ---- PQ per-subspace ----

template <typename MathT, typename IdxT = std::uint32_t>
using pq_subspace_host_matrix = host_mdarray<
  MathT,
  matrix_extent<IdxT>,
  layout_c_contiguous,
  compressed_container_policy<MathT,
                              pq_subspace_spec<MathT,
                                               host_pq_subspace_codebook_view<MathT>,
                                               host_matrix_view<std::uint8_t const, IdxT>>,
                              host_pq_subspace_codebook<MathT>,
                              host_matrix<std::uint8_t, IdxT>>>;

template <typename MathT, typename IdxT = std::uint32_t>
using pq_subspace_device_matrix = device_mdarray<
  MathT,
  matrix_extent<IdxT>,
  layout_c_contiguous,
  compressed_container_policy<MathT,
                              pq_subspace_spec<MathT,
                                               device_pq_subspace_codebook_view<MathT>,
                                               device_matrix_view<std::uint8_t const, IdxT>>,
                              device_pq_subspace_codebook<MathT>,
                              device_matrix<std::uint8_t, IdxT>>>;

// ============================================================================
// Factory functions: non-owning views
// ============================================================================

template <typename MathT, typename IdxT = std::uint32_t>
auto make_sq_host_matrix_view(host_vector_view<MathT const, std::uint32_t> codebook,
                              host_matrix_view<int8_t const, IdxT> codes,
                              MathT min_val,
                              MathT max_val) -> sq_host_matrix_view<MathT, IdxT>
{
  auto spec = make_sq_spec<MathT,
                           host_vector_view<MathT const, std::uint32_t>,
                           host_matrix_view<int8_t const, IdxT>>(min_val, max_val);

  using accessor_t = typename sq_host_matrix_view<MathT, IdxT>::accessor_type;
  using handle_t   = typename accessor_t::data_handle_type;

  handle_t handle{codebook, codes, 0};
  auto dim    = static_cast<IdxT>(codes.extent(1));
  auto n_rows = static_cast<IdxT>(codes.extent(0));
  auto mapping =
    typename sq_host_matrix_view<MathT, IdxT>::mapping_type{make_extents<IdxT>(n_rows, dim)};

  return sq_host_matrix_view<MathT, IdxT>(handle, mapping, accessor_t{spec});
}

template <typename MathT, typename IdxT = std::uint32_t>
auto make_pq_host_matrix_view(host_matrix_view<MathT const, std::uint32_t> codebook,
                              host_matrix_view<std::uint8_t const, IdxT> codes,
                              uint32_t dim) -> pq_host_matrix_view<MathT, IdxT>
{
  using accessor_t = typename pq_host_matrix_view<MathT, IdxT>::accessor_type;
  using handle_t   = typename accessor_t::data_handle_type;

  handle_t handle{codebook, codes, 0};
  auto n_rows = static_cast<IdxT>(codes.extent(0));
  auto mapping =
    typename pq_host_matrix_view<MathT, IdxT>::mapping_type{make_extents<IdxT>(n_rows, IdxT(dim))};

  return pq_host_matrix_view<MathT, IdxT>(handle, mapping, accessor_t{});
}

template <typename MathT, typename IdxT = std::uint32_t>
auto make_pq_subspace_host_matrix_view(host_pq_subspace_codebook_view<MathT> codebook,
                                       host_matrix_view<std::uint8_t const, IdxT> codes,
                                       uint32_t dim) -> pq_subspace_host_matrix_view<MathT, IdxT>
{
  using accessor_t = typename pq_subspace_host_matrix_view<MathT, IdxT>::accessor_type;
  using handle_t   = typename accessor_t::data_handle_type;

  handle_t handle{codebook, codes, 0};
  auto n_rows  = static_cast<IdxT>(codes.extent(0));
  auto mapping = typename pq_subspace_host_matrix_view<MathT, IdxT>::mapping_type{
    make_extents<IdxT>(n_rows, IdxT(dim))};

  return pq_subspace_host_matrix_view<MathT, IdxT>(handle, mapping, accessor_t{});
}

// ============================================================================
// Factory functions: owning mdarray from existing component mdarrays
// ============================================================================

/**
 * @brief Create a PQ host matrix from existing codebook and codes mdarrays.
 *
 * All parameters (dim, pq_len, n_centers) are derived from the component shapes.
 */
template <typename MathT, typename IdxT = std::uint32_t>
auto make_pq_host_matrix(host_matrix<MathT, std::uint32_t>&& codebook,
                         host_matrix<std::uint8_t, IdxT>&& codes) -> pq_host_matrix<MathT, IdxT>
{
  uint32_t pq_len = static_cast<uint32_t>(codebook.extent(1));
  uint32_t pq_dim = static_cast<uint32_t>(codes.extent(1));
  uint32_t dim    = pq_dim * pq_len;
  auto n_rows     = static_cast<IdxT>(codes.extent(0));

  using container_t = typename pq_host_matrix<MathT, IdxT>::container_type;
  using policy_t    = typename pq_host_matrix<MathT, IdxT>::container_policy_type;
  using mapping_t   = typename pq_host_matrix<MathT, IdxT>::mapping_type;

  auto container = container_t(std::move(codebook), std::move(codes));
  auto mapping   = mapping_t{make_extents<IdxT>(n_rows, IdxT(dim))};
  return pq_host_matrix<MathT, IdxT>(mapping, std::move(container), policy_t{});
}

}  // namespace raft
