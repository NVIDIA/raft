/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/detail/sample_rows.cuh>
#include <raft/random/rng_state.hpp>

namespace raft {
namespace matrix {

/** @brief Select rows randomly from input and copy to output.
 *
 * The rows are selected randomly. The random sampling method does not guarantee completely unique
 * selection of rows, but it is close to being unique.
 *
 * @param res RAFT resource handle
 * @param random_state
 * @param dataset input dataset
 * @param output subsampled dataset
 */
template <typename T, typename IdxT, typename accessor, typename layout>
void sample_rows(raft::resources const& res,
                 random::RngState random_state,
                 mdspan<const T, matrix_extent<IdxT>, layout, accessor> dataset,
                 raft::device_matrix_view<T, IdxT> output)
{
  static_assert(std::is_same_v<layout, raft::row_major> ||
                std::is_same_v<layout, raft::layout_stride>);
  RAFT_EXPECTS(dataset.extent(1) == output.extent(1),
               "dataset dims must match, but received %ld vs %ld",
               static_cast<long>(dataset.extent(1)),
               static_cast<long>(output.extent(1)));
  detail::sample_rows<T, IdxT>(
    res, random_state, dataset.data_handle(), dataset.stride(0), dataset.extent(0), output);
}

/** @brief Select rows randomly from input and copy to output.
 *
 * The rows are selected randomly. The random sampling method does not guarantee completely unique
 * selection of rows, but it is close to being unique.
 *
 * @param res RAFT resource handle
 * @param random_state
 * @param dataset input dataset
 * @param n_samples number of rows in the returned matrix
 *
 * @return subsampled dataset
 * */
template <typename T, typename IdxT, typename accessor, typename layout>
raft::device_matrix<T, IdxT> sample_rows(
  raft::resources const& res,
  random::RngState random_state,
  mdspan<const T, matrix_extent<IdxT>, layout, accessor> dataset,
  IdxT n_samples)
{
  static_assert(std::is_same_v<layout, raft::row_major> ||
                std::is_same_v<layout, raft::layout_stride>);
  auto output = raft::make_device_matrix<T, IdxT>(res, n_samples, dataset.extent(1));
  sample_rows(res, random_state, dataset, output.view());
  return output;
}

}  // namespace matrix
}  // namespace raft
