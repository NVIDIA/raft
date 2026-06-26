/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __CLASS_LABELS_H
#define __CLASS_LABELS_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/label/detail/classlabels.cuh>

namespace raft {
namespace label {

/**
 * Get unique class labels.
 *
 * The y array is assumed to store class labels. The unique values are selected
 * from this array.
 *
 * @tparam value_t numeric type of the arrays with class labels
 * @param [in] handle raft resources handle (dry-run aware)
 * @param [inout] unique output unique labels
 * @param [in] y device array of labels, size [n]
 * @param [in] n number of labels
 * @returns number of unique labels (upper bound in dry-run mode)
 */
template <typename value_t>
int getUniquelabels(raft::resources const& handle,
                    rmm::device_uvector<value_t>& unique,
                    value_t* y,
                    size_t n)
{
  return detail::getUniquelabels<value_t>(
    resource::get_dry_run_flag(handle), unique, y, n, resource::get_cuda_stream(handle));
}

/**
 * Get unique class labels.
 *
 * The y array is assumed to store class labels. The unique values are selected
 * from this array.
 *
 * @tparam value_t numeric type of the arrays with class labels
 * @param [inout] unique output unique labels
 * @param [in] y device array of labels, size [n]
 * @param [in] n number of labels
 * @param [in] stream cuda stream
 * @returns unique device array of unique labels, unallocated on entry,
 *   on exit it has size
 */
template <typename value_t>
int getUniquelabels(rmm::device_uvector<value_t>& unique, value_t* y, size_t n, cudaStream_t stream)
{
  return detail::getUniquelabels<value_t>(unique, y, n, stream);
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
 * @param [in] y device array if input labels, size [n]
 * @param [in] n number of labels
 * @param [in] y_unique device array of unique labels, size [n_classes]
 * @param [in] n_classes number of unique labels
 * @param [out] y_out device array of output labels
 * @param [in] idx index of unique label that should be labeled as 1
 * @param [in] stream cuda stream
 */
template <typename value_t>
void getOvrlabels(
  value_t* y, int n, value_t* y_unique, int n_classes, value_t* y_out, int idx, cudaStream_t stream)
{
  detail::getOvrlabels<value_t>(y, n, y_unique, n_classes, y_out, idx, stream);
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
 * @param[out] out the output monotonic array
 * @param[in] in input label array
 * @param[in] N number of elements in the input array
 * @param[in] stream cuda stream to use
 * @param[in] filter_op an optional function for specifying which values
 * should have monotonically increasing labels applied to them.
 * @param[in] zero_based force monotonic set to start at 0?
 */
template <typename Type, typename Lambda>
void make_monotonic(
  Type* out, Type* in, size_t N, cudaStream_t stream, Lambda filter_op, bool zero_based = false)
{
  detail::make_monotonic<Type, Lambda>(out, in, N, stream, filter_op, zero_based);
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
 * @param[out] out output label array with labels assigned monotonically
 * @param[in] in input label array
 * @param[in] N number of elements in the input array
 * @param[in] stream cuda stream to use
 * @param[in] zero_based force monotonic label set to start at 0?
 */
template <typename Type>
void make_monotonic(Type* out, Type* in, size_t N, cudaStream_t stream, bool zero_based = false)
{
  detail::make_monotonic<Type>(out, in, N, stream, zero_based);
}
};  // namespace label
};  // namespace raft
#endif
