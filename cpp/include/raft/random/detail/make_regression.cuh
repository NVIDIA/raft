/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/* Adapted from scikit-learn
 * https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/_samples_generator.py
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/init.cuh>
#include <raft/linalg/qr.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace raft {
namespace random {
namespace detail {

/* Internal auxiliary function to help build the singular profile */
template <typename DataT, typename IdxT>
RAFT_KERNEL _singular_profile_kernel(DataT* out, IdxT n, DataT tail_strength, IdxT rank)
{
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    DataT sval     = static_cast<DataT>(tid) / rank;
    DataT low_rank = ((DataT)1.0 - tail_strength) * raft::exp(-sval * sval);
    DataT tail     = tail_strength * raft::exp((DataT)-0.1 * sval);
    out[tid]       = low_rank + tail;
  }
}

/* Internal auxiliary function to generate a low-rank matrix */
template <typename DataT, typename IdxT>
static void _make_low_rank_matrix(raft::resources const& handle,
                                  DataT* out,
                                  IdxT n_rows,
                                  IdxT n_cols,
                                  IdxT effective_rank,
                                  DataT tail_strength,
                                  raft::random::RngState& r,
                                  cudaStream_t stream)
{
  bool is_dry_run = resource::get_dry_run_flag(handle);
  IdxT n          = std::min(n_rows, n_cols);

  // Generate random (ortho normal) vectors with QR decomposition
  rmm::device_uvector<DataT> rd_mat_0(n_rows * n, stream);
  rmm::device_uvector<DataT> rd_mat_1(n_cols * n, stream);
  if (!is_dry_run) {
    normal(r, rd_mat_0.data(), n_rows * n, (DataT)0.0, (DataT)1.0, stream);
    normal(r, rd_mat_1.data(), n_cols * n, (DataT)0.0, (DataT)1.0, stream);
  }
  rmm::device_uvector<DataT> q0(n_rows * n, stream);
  rmm::device_uvector<DataT> q1(n_cols * n, stream);
  if (!is_dry_run) {
    raft::linalg::qrGetQ(handle, rd_mat_0.data(), q0.data(), n_rows, n, stream);
    raft::linalg::qrGetQ(handle, rd_mat_1.data(), q1.data(), n_cols, n, stream);
  }

  // Build the singular profile by assembling signal and noise components
  rmm::device_uvector<DataT> singular_vec(n, stream);
  if (!is_dry_run) {
    _singular_profile_kernel<<<raft::ceildiv<IdxT>(n, 256), 256, 0, stream>>>(
      singular_vec.data(), n, tail_strength, effective_rank);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  rmm::device_uvector<DataT> singular_mat(n * n, stream);
  if (!is_dry_run) {
    RAFT_CUDA_TRY(cudaMemsetAsync(singular_mat.data(), 0, n * n * sizeof(DataT), stream));
  }

  raft::matrix::set_diagonal(handle,
                             make_device_vector_view<const DataT, IdxT>(singular_vec.data(), n),
                             make_device_matrix_view<DataT, IdxT>(singular_mat.data(), n, n));

  // Generate the column-major matrix
  rmm::device_uvector<DataT> temp_q0s(n_rows * n, stream);
  rmm::device_uvector<DataT> temp_out(n_rows * n_cols, stream);
  DataT alpha = 1.0, beta = 0.0;
  raft::linalg::gemm(handle,
                     false,
                     false,
                     n_rows,
                     n,
                     n,
                     &alpha,
                     q0.data(),
                     n_rows,
                     singular_mat.data(),
                     n,
                     &beta,
                     temp_q0s.data(),
                     n_rows,
                     stream);
  raft::linalg::gemm(handle,
                     false,
                     true,
                     n_rows,
                     n_cols,
                     n,
                     &alpha,
                     temp_q0s.data(),
                     n_rows,
                     q1.data(),
                     n_cols,
                     &beta,
                     temp_out.data(),
                     n_rows,
                     stream);

  // Transpose from column-major to row-major
  raft::linalg::transpose(handle, temp_out.data(), out, n_rows, n_cols, stream);
}

/* Internal auxiliary function to permute rows in the given matrix according
 * to a given permutation vector */
template <typename DataT, typename IdxT>
RAFT_KERNEL _gather2d_kernel(
  DataT* out, const DataT* in, const IdxT* perms, IdxT n_rows, IdxT n_cols)
{
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < n_rows) {
    const DataT* row_in = in + n_cols * perms[tid];
    DataT* row_out      = out + n_cols * tid;

    for (IdxT i = 0; i < n_cols; i++) {
      row_out[i] = row_in[i];
    }
  }
}

template <typename DataT, typename IdxT>
void make_regression_caller(raft::resources const& handle,
                            DataT* out,
                            DataT* values,
                            IdxT n_rows,
                            IdxT n_cols,
                            IdxT n_informative,
                            cudaStream_t stream,
                            DataT* coef                      = nullptr,
                            IdxT n_targets                   = (IdxT)1,
                            DataT bias                       = (DataT)0.0,
                            IdxT effective_rank              = (IdxT)-1,
                            DataT tail_strength              = (DataT)0.5,
                            DataT noise                      = (DataT)0.0,
                            bool shuffle                     = true,
                            uint64_t seed                    = 0ULL,
                            raft::random::GeneratorType type = raft::random::GenPC)
{
  bool is_dry_run = resource::get_dry_run_flag(handle);
  n_informative   = std::min(n_informative, n_cols);

  raft::random::RngState r(seed, type);

  if (effective_rank < 0) {
    // Randomly generate a well conditioned input set
    if (!is_dry_run) { normal(r, out, n_rows * n_cols, (DataT)0.0, (DataT)1.0, stream); }
  } else {
    // Randomly generate a low rank, fat tail input set
    _make_low_rank_matrix(handle, out, n_rows, n_cols, effective_rank, tail_strength, r, stream);
  }

  // Use the right output buffer for the values
  rmm::device_uvector<DataT> tmp_values(shuffle ? n_rows * n_targets : 0, stream);
  DataT* _values = shuffle ? tmp_values.data() : values;
  // Create a column-major matrix of output values only if it has more
  // than 1 column
  rmm::device_uvector<DataT> values_col(n_targets > 1 ? n_rows * n_targets : 0, stream);
  DataT* _values_col = n_targets > 1 ? values_col.data() : _values;

  // Use the right buffer for the coefficients
  rmm::device_uvector<DataT> tmp_coef((coef != nullptr && !shuffle) ? 0 : n_cols * n_targets,
                                      stream);
  DataT* _coef = tmp_coef.size() == 0 ? coef : tmp_coef.data();

  // Generate a ground truth model with only n_informative features
  if (!is_dry_run) {
    uniform(r, _coef, n_informative * n_targets, (DataT)1.0, (DataT)100.0, stream);
    if (coef && n_informative != n_cols) {
      RAFT_CUDA_TRY(cudaMemsetAsync(_coef + n_informative * n_targets,
                                    0,
                                    (n_cols - n_informative) * n_targets * sizeof(DataT),
                                    stream));
    }
  }

  // Compute the output values
  DataT alpha = (DataT)1.0, beta = (DataT)0.0;
  raft::linalg::gemm(handle,
                     true,
                     true,
                     n_rows,
                     n_targets,
                     n_informative,
                     &alpha,
                     out,
                     n_cols,
                     _coef,
                     n_targets,
                     &beta,
                     _values_col,
                     n_rows,
                     stream);

  // Transpose the values from column-major to row-major if needed
  if (n_targets > 1) {
    raft::linalg::transpose(handle, _values_col, _values, n_rows, n_targets, stream);
  }

  if (!is_dry_run) {
    if (bias != 0.0) {
      // Add bias
      raft::linalg::addScalar(_values, _values, bias, n_rows * n_targets, stream);
    }
  }

  rmm::device_uvector<DataT> white_noise(noise != 0.0 ? n_rows * n_targets : 0, stream);
  if (noise != 0.0 && !is_dry_run) {
    // Add white noise
    normal(r, white_noise.data(), n_rows * n_targets, (DataT)0.0, noise, stream);
    raft::linalg::add(_values, _values, white_noise.data(), n_rows * n_targets, stream);
  }

  if (shuffle) {
    rmm::device_uvector<DataT> tmp_out(n_rows * n_cols, stream);
    rmm::device_uvector<IdxT> perms_samples(n_rows, stream);
    rmm::device_uvector<IdxT> perms_features(n_cols, stream);

    if (!is_dry_run) {
      constexpr IdxT Nthreads = 256;

      // Shuffle the samples from out to tmp_out
      raft::random::permute<DataT, IdxT, IdxT>(
        perms_samples.data(), tmp_out.data(), out, n_cols, n_rows, true, stream);
      IdxT nblks_rows = raft::ceildiv<IdxT>(n_rows, Nthreads);
      _gather2d_kernel<<<nblks_rows, Nthreads, 0, stream>>>(
        values, _values, perms_samples.data(), n_rows, n_targets);
      RAFT_CUDA_TRY(cudaPeekAtLastError());

      // Shuffle the features from tmp_out to out
      raft::random::permute<DataT, IdxT, IdxT>(
        perms_features.data(), out, tmp_out.data(), n_rows, n_cols, false, stream);

      // Shuffle the coefficients accordingly
      if (coef != nullptr) {
        IdxT nblks_cols = raft::ceildiv<IdxT>(n_cols, Nthreads);
        _gather2d_kernel<<<nblks_cols, Nthreads, 0, stream>>>(
          coef, _coef, perms_features.data(), n_cols, n_targets);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      }
    }
  }
}

}  // namespace detail
}  // namespace random
}  // namespace raft
