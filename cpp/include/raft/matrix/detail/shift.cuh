/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/shift_types.hpp>

namespace raft {
namespace matrix::detail {
enum FillType { CONSTANT, MATRIX, SELF_ID };

template <typename T, typename fill_value, FillType fill_type>
RAFT_KERNEL col_shift_towards_end(
  T* in_out, size_t n_rows, size_t n_cols, size_t k, fill_value value)
{
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows) {
    size_t base_idx = row * n_cols;
    for (size_t target_col = n_cols - 1; target_col >= k; target_col--) {
      in_out[base_idx + target_col] = in_out[base_idx + (target_col - k)];
    }
    if constexpr (fill_type == FillType::CONSTANT) {
      T val = static_cast<T>(value);
      for (size_t i = 0; i < k; i++) {
        in_out[base_idx + i] = val;
      }
    } else if constexpr (fill_type == FillType::MATRIX) {
      const T* values = static_cast<const T*>(value);
      for (size_t i = 0; i < k; i++) {
        in_out[base_idx + i] = values[row * k + i];
      }
    } else {  // FillType::SELF_ID
      for (size_t i = 0; i < k; i++) {
        in_out[base_idx + i] = static_cast<T>(row);
      }
    }
  }
}

template <typename T, typename fill_value, FillType fill_type>
RAFT_KERNEL col_shift_towards_beginning(
  T* in_out, size_t n_rows, size_t n_cols, size_t k, fill_value value)
{
  size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n_rows) {
    size_t base_idx = row * n_cols;
    for (size_t target_col = 0; target_col < n_cols - k; target_col++) {
      in_out[base_idx + target_col] = in_out[base_idx + (target_col + k)];
    }
    size_t base_col = n_cols - k;
    if constexpr (fill_type == FillType::CONSTANT) {
      T val = static_cast<T>(value);
      for (size_t i = 0; i < k; i++) {
        in_out[base_idx + base_col + i] = val;
      }
    } else if constexpr (fill_type == FillType::MATRIX) {
      const T* values = static_cast<const T*>(value);
      for (size_t i = 0; i < k; i++) {
        in_out[base_idx + base_col + i] = values[row * k + i];
      }
    } else {  // FillType::SELF_ID
      for (size_t i = 0; i < k; i++) {
        in_out[base_idx + base_col + i] = static_cast<T>(row);
      }
    }
  }
}

template <typename T, typename fill_value, FillType fill_type>
RAFT_KERNEL row_shift_towards_end(
  T* in_out, size_t n_rows, size_t n_cols, size_t k, fill_value value)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n_cols) {
    for (size_t target_row = n_rows - 1; target_row >= k; target_row--) {
      in_out[target_row * n_cols + col] = in_out[(target_row - k) * n_cols + col];
    }

    if constexpr (fill_type == FillType::CONSTANT) {
      T val = static_cast<T>(value);
      for (size_t i = 0; i < k; i++) {
        in_out[i * n_cols + col] = val;
      }
    } else if constexpr (fill_type == FillType::MATRIX) {
      const T* values = static_cast<const T*>(value);
      for (size_t i = 0; i < k; i++) {
        in_out[i * n_cols + col] = values[i * n_cols + col];
      }
    } else {  // FillType::SELF_ID
      for (size_t i = 0; i < k; i++) {
        in_out[i * n_cols + col] = static_cast<T>(col);
      }
    }
  }
}

template <typename T, typename fill_value, FillType fill_type>
RAFT_KERNEL row_shift_towards_beginning(
  T* in_out, size_t n_rows, size_t n_cols, size_t k, fill_value value)
{
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col < n_cols) {
    for (size_t target_row = 0; target_row < n_rows - k; target_row++) {
      in_out[target_row * n_cols + col] = in_out[(target_row + k) * n_cols + col];
    }
    size_t base_row = n_rows - k;
    if constexpr (fill_type == FillType::CONSTANT) {
      T val = static_cast<T>(value);
      for (size_t i = 0; i < k; i++) {
        in_out[(base_row + i) * n_cols + col] = val;
      }
    } else if constexpr (fill_type == FillType::MATRIX) {
      const T* values = static_cast<const T*>(value);
      for (size_t i = 0; i < k; i++) {
        in_out[(base_row + i) * n_cols + col] = values[i * n_cols + col];
      }
    } else {  // FillType::SELF_ID
      for (size_t i = 0; i < k; i++) {
        in_out[(base_row + i) * n_cols + col] = static_cast<T>(col);
      }
    }
  }
}

template <typename ValueT, typename IdxT, typename fill_value, FillType fill_type>
void shift_dispatch(raft::resources const& handle,
                    raft::device_matrix_view<ValueT, IdxT, row_major> in_out,
                    fill_value value,
                    size_t k,
                    ShiftDirection shift_direction = ShiftDirection::TOWARDS_END,
                    ShiftType shift_type           = ShiftType::COL)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  size_t n_rows = in_out.extent(0);
  size_t n_cols = in_out.extent(1);
  size_t TPB    = 256;
  auto stream   = raft::resource::get_cuda_stream(handle);

  if (shift_type == ShiftType::COL) {
    size_t num_blocks = static_cast<size_t>((n_rows + TPB) / TPB);
    if (shift_direction == ShiftDirection::TOWARDS_BEGINNING) {
      col_shift_towards_beginning<ValueT, fill_value, fill_type>
        <<<num_blocks, TPB, 0, stream>>>(in_out.data_handle(), n_rows, n_cols, k, value);
    } else {  // ShiftDirection::TOWARDS_END
      col_shift_towards_end<ValueT, fill_value, fill_type>
        <<<num_blocks, TPB, 0, stream>>>(in_out.data_handle(), n_rows, n_cols, k, value);
    }
  } else {  // ShiftType::ROW
    size_t num_blocks = static_cast<size_t>((n_cols + TPB) / TPB);
    if (shift_direction == ShiftDirection::TOWARDS_BEGINNING) {
      row_shift_towards_beginning<ValueT, fill_value, fill_type>
        <<<num_blocks, TPB, 0, stream>>>(in_out.data_handle(), n_rows, n_cols, k, value);
    } else {  // ShiftDirection::TOWARDS_END
      row_shift_towards_end<ValueT, fill_value, fill_type>
        <<<num_blocks, TPB, 0, stream>>>(in_out.data_handle(), n_rows, n_cols, k, value);
    }
  }
  raft::resource::sync_stream(handle);
}

template <typename ValueT, typename IdxT>
void shift(raft::resources const& handle,
           raft::device_matrix_view<ValueT, IdxT, row_major> in_out,
           size_t k,
           std::optional<ValueT> val      = std::nullopt,
           ShiftDirection shift_direction = ShiftDirection::TOWARDS_END,
           ShiftType shift_type           = ShiftType::COL)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  if (val.has_value()) {
    shift_dispatch<ValueT, IdxT, ValueT, CONSTANT>(
      handle, in_out, val.value(), k, shift_direction, shift_type);
  } else {
    // using 0 here as a placeholder
    shift_dispatch<ValueT, IdxT, ValueT, SELF_ID>(
      handle, in_out, static_cast<ValueT>(0), k, shift_direction, shift_type);
  }
}

template <typename ValueT, typename IdxT>
void shift(raft::resources const& handle,
           raft::device_matrix_view<ValueT, IdxT, row_major> in_out,
           raft::device_matrix_view<const ValueT, IdxT> values,
           ShiftDirection shift_direction = ShiftDirection::TOWARDS_END,
           ShiftType shift_type           = ShiftType::COL)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  size_t k = shift_type == ShiftType::COL ? values.extent(1) : values.extent(0);
  shift_dispatch<ValueT, IdxT, const ValueT*, MATRIX>(
    handle, in_out, values.data_handle(), k, shift_direction, shift_type);
}
}  // namespace matrix::detail
}  // namespace raft
