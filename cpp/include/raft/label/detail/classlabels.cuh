/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_select.cuh>

#include <algorithm>

namespace raft {
namespace label {
namespace detail {

/**
 * Get unique class labels.
 *
 * The y array is assumed to store class labels. The unique values are selected
 * from this array.
 *
 * \tparam value_t numeric type of the arrays with class labels
 * \param [in] dry_run if true, perform allocations but skip CUDA work
 * \param [out] unique device array of unique labels, unallocated on entry,
 *   on exit it has size [n_unique]
 * \param [in] y device array of labels, size [n]
 * \param [in] n number of labels
 * \param [in] stream cuda stream
 * \return number of unique labels (upper bound when dry_run is true)
 */
template <typename value_t>
int getUniquelabels(
  bool dry_run, rmm::device_uvector<value_t>& unique, value_t* y, size_t n, cudaStream_t stream)
{
  rmm::device_scalar<int> d_num_selected(stream);
  rmm::device_uvector<value_t> workspace(n, stream);
  size_t bytes  = 0;
  size_t bytes2 = 0;

  // Query how much temporary storage we will need for cub operations
  // and allocate it
  cub::DeviceRadixSort::SortKeys(
    NULL, bytes, y, workspace.data(), n, 0, sizeof(value_t) * 8, stream);
  cub::DeviceSelect::Unique(
    NULL, bytes2, workspace.data(), workspace.data(), d_num_selected.data(), n, stream);
  bytes = std::max(bytes, bytes2);
  rmm::device_uvector<char> cub_storage(bytes, stream);

  if (dry_run) {
    if (unique.size() < n) { unique = rmm::device_uvector<value_t>(n, stream); }
    return static_cast<int>(n);
  }

  // Select Unique classes
  cub::DeviceRadixSort::SortKeys(
    cub_storage.data(), bytes, y, workspace.data(), n, 0, sizeof(value_t) * 8, stream);
  cub::DeviceSelect::Unique(cub_storage.data(),
                            bytes,
                            workspace.data(),
                            workspace.data(),
                            d_num_selected.data(),
                            n,
                            stream);

  int n_unique = d_num_selected.value(stream);
  // Copy unique classes to output
  unique.resize(n_unique, stream);
  raft::copy(unique.data(), workspace.data(), n_unique, stream);

  return n_unique;
}

/**
 * Get unique class labels.
 *
 * The y array is assumed to store class labels. The unique values are selected
 * from this array.
 *
 * \tparam value_t numeric type of the arrays with class labels
 * \param [out] unique device array of unique labels, unallocated on entry,
 *   on exit it has size [n_unique]
 * \param [in] y device array of labels, size [n]
 * \param [in] n number of labels
 * \param [in] stream cuda stream
 * \return number of unique labels
 */
template <typename value_t>
int getUniquelabels(rmm::device_uvector<value_t>& unique, value_t* y, size_t n, cudaStream_t stream)
{
  return getUniquelabels(false, unique, y, n, stream);
}

/**
 * Assign one versus rest labels.
 *
 * The output labels will have values +/-1:
 * y_out = (y == y_unique[idx]) ? +1 : -1;
 *
 * The output type currently is set to value_t, but for SVM in principle we are
 * free to choose other type for y_out (it should represent +/-1, and it is used
 * in floating point arithmetics).
 *
 * \param [in] y device array if input labels, size [n]
 * \param [in] n number of labels
 * \param [in] y_unique device array of unique labels, size [n_classes]
 * \param [in] n_classes number of unique labels
 * \param [out] y_out device array of output labels
 * \param [in] idx index of unique label that should be labeled as 1
 * \param [in] stream cuda stream
 */
template <typename value_t>
void getOvrlabels(
  value_t* y, int n, value_t* y_unique, int n_classes, value_t* y_out, int idx, cudaStream_t stream)
{
  ASSERT(idx < n_classes,
         "Parameter idx should not be larger than the number "
         "of classes");
  raft::linalg::unaryOp(
    y_out,
    y,
    n,
    [idx, y_unique] __device__(value_t y) { return y == y_unique[idx] ? +1 : -1; },
    stream);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// TODO: add one-versus-one selection: select two classes, relabel them to
// +/-1, return array with the new class labels and corresponding indices.

template <typename Type, int TPB_X, typename Lambda>
RAFT_KERNEL map_label_kernel(Type* map_ids,
                             size_t N_labels,
                             Type* in,
                             Type* out,
                             size_t N,
                             Lambda filter_op,
                             bool zero_based = false)
{
  int tid = threadIdx.x + blockIdx.x * TPB_X;
  if (tid < N) {
    if (!filter_op(in[tid])) {
      for (size_t i = 0; i < N_labels; i++) {
        if (in[tid] == map_ids[i]) {
          out[tid] = i + !zero_based;
          break;
        }
      }
    }
  }
}

/**
 * Maps an input array containing a series of numbers into a new array
 * where numbers have been mapped to a monotonically increasing set
 * of labels. This can be useful in machine learning algorithms, for instance,
 * where a given set of labels is not taken from a monotonically increasing
 * set. This can happen if they are filtered or if only a subset of the
 * total labels are used in a dataset. This is also useful in graph algorithms
 * where a set of vertices need to be labeled in a monotonically increasing
 * order.
 * @tparam Type the numeric type of the input and output arrays
 * @tparam Lambda the type of an optional filter function, which determines
 * which items in the array to map.
 * @param out the output monotonic array
 * @param in input label array
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 * @param filter_op an optional function for specifying which values
 * should have monotonically increasing labels applied to them.
 */
template <typename Type, typename Lambda>
void make_monotonic(
  Type* out, Type* in, size_t N, cudaStream_t stream, Lambda filter_op, bool zero_based = false)
{
  static const size_t TPB_X = 256;

  dim3 blocks(raft::ceildiv(N, TPB_X));
  dim3 threads(TPB_X);

  rmm::device_uvector<Type> map_ids(0, stream);
  int num_clusters = getUniquelabels(map_ids, in, N, stream);

  map_label_kernel<Type, TPB_X><<<blocks, threads, 0, stream>>>(
    map_ids.data(), num_clusters, in, out, N, filter_op, zero_based);
}

/**
 * Maps an input array containing a series of numbers into a new array
 * where numbers have been mapped to a monotonically increasing set
 * of labels. This can be useful in machine learning algorithms, for instance,
 * where a given set of labels is not taken from a monotonically increasing
 * set. This can happen if they are filtered or if only a subset of the
 * total labels are used in a dataset. This is also useful in graph algorithms
 * where a set of vertices need to be labeled in a monotonically increasing
 * order.
 * @tparam Type the numeric type of the input and output arrays
 * @tparam Lambda the type of an optional filter function, which determines
 * which items in the array to map.
 * @param out output label array with labels assigned monotonically
 * @param in input label array
 * @param N number of elements in the input array
 * @param stream cuda stream to use
 */
template <typename Type>
void make_monotonic(Type* out, Type* in, size_t N, cudaStream_t stream, bool zero_based = false)
{
  make_monotonic<Type>(out, in, N, stream, raft::const_op(false), zero_based);
}

};  // namespace detail
};  // namespace label
};  // namespace raft
