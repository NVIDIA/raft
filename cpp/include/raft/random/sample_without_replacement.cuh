/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/rng_impl.cuh"
#include "rng_state.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

#include <cassert>
#include <optional>
#include <type_traits>
#include <variant>

namespace raft {
namespace random {

namespace sample_without_replacement_impl {
template <typename T>
struct weight_alias {};

template <>
struct weight_alias<std::nullopt_t> {
  using type = double;
};

template <typename ElementType, typename IndexType>
struct weight_alias<std::optional<raft::device_vector_view<ElementType, IndexType>>> {
  using type = typename raft::device_vector_view<ElementType, IndexType>::value_type;
};

template <typename T>
using weight_t = typename weight_alias<T>::type;
}  // namespace sample_without_replacement_impl

/**
 * \defgroup sample_without_replacement Sampling without Replacement
 * @{
 */

/**
 * @brief Sample the input vector without replacement, optionally based on the
 * input weight vector for each element in the array.
 *
 * The implementation is based on the `one-pass sampling` algorithm described in
 * ["Accelerating weighted random sampling without
 * replacement,"](https://www.ethz.ch/content/dam/ethz/special-interest/baug/ivt/ivt-dam/vpl/reports/1101-1200/ab1141.pdf)
 * a technical report by Kirill Mueller.
 *
 * If no input weight vector is provided, then input elements will be
 * sampled uniformly.  Otherwise, the elements sampled from the input
 * vector will always appear in increasing order of their weights as
 * computed using the exponential distribution. So, if you are
 * particular about the order (for e.g., array permutations), then
 * this might not be the right choice.
 *
 * @tparam DataT type of each element of the input array @c in
 * @tparam IdxT type of the dimensions of the arrays; output index type
 * @tparam WeightsVectorType std::optional<raft::device_vector_view<const weight_type, IdxT>> of
 * each elements of the weights array @c weights_opt
 * @tparam OutIndexVectorType std::optional<raft::device_vector_view<IdxT, IdxT>> of output indices
 * @c outIdx_opt
 *
 * @note Please do not specify template parameters explicitly,
 *   as the compiler can deduce them from the arguments.
 *
 * @param[in] handle RAFT handle containing (among other resources)
 *   the CUDA stream on which to run.
 * @param[inout] rng_state Pseudorandom number generator state.
 * @param[in] in Input vector to be sampled.
 * @param[in] weights_opt std::optional weights vector.
 *        If not provided, uniform sampling will be used.
 * @param[out] out Vector of samples from the input vector.
 * @param[out] outIdx_opt std::optional vector of the indices
 *   sampled from the input array.
 *
 * @pre The number of samples `out.extent(0)`
 *   is less than or equal to the number of inputs `in.extent(0)`.
 *
 * @pre The number of weights `wts.extent(0)`
 *   equals the number of inputs `in.extent(0)`.
 */
template <typename DataT, typename IdxT, typename WeightsVectorType, class OutIndexVectorType>
void sample_without_replacement(raft::resources const& handle,
                                RngState& rng_state,
                                raft::device_vector_view<const DataT, IdxT> in,
                                WeightsVectorType&& weights_opt,
                                raft::device_vector_view<DataT, IdxT> out,
                                OutIndexVectorType&& outIdx_opt)
{
  using weight_type = sample_without_replacement_impl::weight_t<
    std::remove_const_t<std::remove_reference_t<WeightsVectorType>>>;

  std::optional<raft::device_vector_view<const weight_type, IdxT>> wts =
    std::forward<WeightsVectorType>(weights_opt);
  std::optional<raft::device_vector_view<IdxT, IdxT>> outIdx =
    std::forward<OutIndexVectorType>(outIdx_opt);

  static_assert(std::is_integral<IdxT>::value, "IdxT must be an integral type.");
  const IdxT sampledLen = out.extent(0);
  const IdxT len        = in.extent(0);
  RAFT_EXPECTS(sampledLen <= len,
               "sampleWithoutReplacement: "
               "sampledLen (out.extent(0)) must be <= len (in.extent(0))");
  RAFT_EXPECTS(len == 0 || in.data_handle() != nullptr,
               "sampleWithoutReplacement: "
               "If in.extents(0) != 0, then in.data_handle() must be nonnull");
  RAFT_EXPECTS(sampledLen == 0 || out.data_handle() != nullptr,
               "sampleWithoutReplacement: "
               "If out.extents(0) != 0, then out.data_handle() must be nonnull");

  const bool outIdx_has_value = outIdx.has_value();
  if (outIdx_has_value) {
    RAFT_EXPECTS((*outIdx).extent(0) == sampledLen,
                 "sampleWithoutReplacement: "
                 "If outIdx is provided, its extent(0) must equal out.extent(0)");
  }
  IdxT* outIdx_ptr = outIdx_has_value ? (*outIdx).data_handle() : nullptr;

  const bool wts_has_value = wts.has_value();
  if (wts_has_value) {
    RAFT_EXPECTS((*wts).extent(0) == len,
                 "sampleWithoutReplacement: "
                 "If wts is provided, its extent(0) must equal in.extent(0)");
  }
  const weight_type* wts_ptr = wts_has_value ? (*wts).data_handle() : nullptr;

  detail::sampleWithoutReplacement(resource::get_dry_run_flag(handle),
                                   rng_state,
                                   out.data_handle(),
                                   outIdx_ptr,
                                   in.data_handle(),
                                   wts_ptr,
                                   sampledLen,
                                   len,
                                   resource::get_cuda_stream(handle));
}

/**
 * @brief Overload of `sample_without_replacement` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 *
 * Please see above for documentation of `sample_without_replacement`.
 */
template <typename... Args, typename = std::enable_if_t<sizeof...(Args) == 5>>
void sample_without_replacement(Args... args)
{
  sample_without_replacement(std::forward<Args>(args)..., std::nullopt);
}

/** @} */

}  // namespace random
}  // namespace raft
