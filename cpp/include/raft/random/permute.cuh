/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PERMUTE_H
#define __PERMUTE_H

#pragma once

#include "detail/permute.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

#include <optional>
#include <type_traits>

namespace raft {
namespace random {

namespace permute_impl {

template <typename T, typename InputOutputValueType, typename IdxType, typename Layout>
struct perms_out_view {};

template <typename InputOutputValueType, typename IdxType, typename Layout>
struct perms_out_view<std::nullopt_t, InputOutputValueType, IdxType, Layout> {
  // permsOut won't have a value anyway,
  // so we can pick any integral value type we want.
  using type = raft::device_vector_view<IdxType, IdxType>;
};

template <typename PermutationIndexType,
          typename InputOutputValueType,
          typename IdxType,
          typename Layout>
struct perms_out_view<std::optional<raft::device_vector_view<PermutationIndexType, IdxType>>,
                      InputOutputValueType,
                      IdxType,
                      Layout> {
  using type = raft::device_vector_view<PermutationIndexType, IdxType>;
};

template <typename T, typename InputOutputValueType, typename IdxType, typename Layout>
using perms_out_view_t = typename perms_out_view<T, InputOutputValueType, IdxType, Layout>::type;

}  // namespace permute_impl

/**
 * \defgroup permute Permutation
 * @{
 */

/**
 * @brief Randomly permute the rows of the input matrix.
 *
 * We do not support in-place permutation, so that we can compute
 * in parallel without race conditions.  This function is useful
 * for shuffling input data sets in machine learning algorithms.
 *
 * @tparam InputOutputValueType Type of each element of the input matrix,
 *   and the type of each element of the output matrix (if provided)
 * @tparam IntType Integer type of each element of `permsOut`
 * @tparam IdxType Integer type of the extents of the mdspan parameters
 * @tparam Layout Either `raft::row_major` or `raft::col_major`
 *
 * @param[in] handle RAFT handle containing the CUDA stream
 *   on which to run.
 * @param[in] in input matrix
 * @param[out] permsOut If provided, the indices of the permutation.
 * @param[out] out If provided, the output matrix, containing the
 *   permuted rows of the input matrix `in`.  (Not providing this
 *   is only useful if you provide `permsOut`.)
 *
 * @pre If `permsOut.has_value()` is `true`,
 *   then `(*permsOut).extent(0) == in.extent(0)` is `true`.
 *
 * @pre If `out.has_value()` is `true`,
 *   then `(*out).extents() == in.extents()` is `true`.
 *
 * @note This is NOT a uniform permutation generator!
 *   It only generates a small fraction of all possible random permutations.
 *   If your application needs a high-quality permutation generator,
 *   then we recommend Knuth Shuffle.
 */
template <typename InputOutputValueType, typename IntType, typename IdxType, typename Layout>
void permute(raft::resources const& handle,
             raft::device_matrix_view<const InputOutputValueType, IdxType, Layout> in,
             std::optional<raft::device_vector_view<IntType, IdxType>> permsOut,
             std::optional<raft::device_matrix_view<InputOutputValueType, IdxType, Layout>> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  static_assert(std::is_integral_v<IntType>,
                "permute: The type of each element "
                "of permsOut (if provided) must be an integral type.");
  static_assert(std::is_integral_v<IdxType>,
                "permute: The index type "
                "of each mdspan argument must be an integral type.");
  constexpr bool is_row_major = std::is_same_v<Layout, raft::row_major>;
  constexpr bool is_col_major = std::is_same_v<Layout, raft::col_major>;
  static_assert(is_row_major || is_col_major,
                "permute: Layout must be either "
                "raft::row_major or raft::col_major (or one of their aliases)");

  const bool permsOut_has_value = permsOut.has_value();
  const bool out_has_value      = out.has_value();

  RAFT_EXPECTS(!permsOut_has_value || (*permsOut).extent(0) == in.extent(0),
               "permute: If 'permsOut' is provided, then its extent(0) "
               "must equal the number of rows of the input matrix 'in'.");
  RAFT_EXPECTS(!out_has_value || (*out).extents() == in.extents(),
               "permute: If 'out' is provided, then both its extents "
               "must match the extents of the input matrix 'in'.");

  IntType* permsOut_ptr         = permsOut_has_value ? (*permsOut).data_handle() : nullptr;
  InputOutputValueType* out_ptr = out_has_value ? (*out).data_handle() : nullptr;

  if (permsOut_ptr != nullptr || out_ptr != nullptr) {
    const IdxType N = in.extent(0);
    const IdxType D = in.extent(1);
    detail::permute<InputOutputValueType, IntType, IdxType>(permsOut_ptr,
                                                            out_ptr,
                                                            in.data_handle(),
                                                            D,
                                                            N,
                                                            is_row_major,
                                                            resource::get_cuda_stream(handle));
  }
}

/**
 * @brief Overload of `permute` that compiles if users pass in `std::nullopt`
 *   for either or both of `permsOut` and `out`.
 */
template <typename InputOutputValueType,
          typename IdxType,
          typename Layout,
          typename PermsOutType,
          typename OutType>
void permute(raft::resources const& handle,
             raft::device_matrix_view<const InputOutputValueType, IdxType, Layout> in,
             PermsOutType&& permsOut,
             OutType&& out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  // If PermsOutType is std::optional<device_vector_view<T, IdxType>>
  // for some T, then that type T need not be related to any of the
  // other template parameters.  Thus, we have to deduce it specially.
  using perms_out_view_type = permute_impl::
    perms_out_view_t<std::decay_t<PermsOutType>, InputOutputValueType, IdxType, Layout>;
  using out_view_type = raft::device_matrix_view<InputOutputValueType, IdxType, Layout>;

  static_assert(std::is_same_v<std::decay_t<OutType>, std::nullopt_t> ||
                  std::is_same_v<std::decay_t<OutType>, std::optional<out_view_type>>,
                "permute: The type of 'out' must be either std::optional<"
                "raft::device_matrix_view<InputOutputViewType, IdxType, Layout>>, "
                "or std::nullopt.");

  std::optional<perms_out_view_type> permsOut_arg = std::forward<PermsOutType>(permsOut);
  std::optional<out_view_type> out_arg            = std::forward<OutType>(out);
  permute(handle, in, permsOut_arg, out_arg);
}

/** @} */

/**
 * @brief Legacy overload of `permute` that takes raw arrays instead of mdspan.
 *
 * @tparam Type Type of each element of the input matrix to be permuted
 * @tparam IntType Integer type of each element of the permsOut matrix
 * @tparam IdxType Integer type of the dimensions of the matrices
 * @tparam TPB threads per block (do not use any value other than the default)
 *
 * @param[out] perms If nonnull, the indices of the permutation
 * @param[out] out If nonnull, the output matrix, containing the
 *   permuted rows of the input matrix @c in.  (Not providing this
 *   is only useful if you provide @c perms.)
 * @param[in] in input matrix
 * @param[in] D number of columns in the matrices
 * @param[in] N number of rows in the matrices
 * @param[in] rowMajor true if the matrices are row major,
 *   false if they are column major
 * @param[in] stream CUDA stream on which to run
 */
template <typename Type, typename IntType = int, typename IdxType = int, int TPB = 256>
void permute(IntType* perms,
             Type* out,
             const Type* in,
             IntType D,
             IntType N,
             bool rowMajor,
             cudaStream_t stream)
{
  detail::permute<Type, IntType, IdxType, TPB>(perms, out, in, D, N, rowMajor, stream);
}

};  // namespace random
}  // namespace raft
#endif
