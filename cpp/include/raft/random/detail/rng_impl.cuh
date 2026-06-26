/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/math.hpp>
#include <raft/core/operators.cuh>
#include <raft/linalg/map.cuh>
#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/scatter.cuh>

#include <rmm/device_scalar.hpp>

#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <cuda_fp16.h>

namespace raft {
namespace random {
namespace detail {

/**
 * This macro will invoke function `func` with the correct instantiation of
 * device state as the first parameter, and passes all subsequent macro
 * arguments through to the function.
 * Note that you can call this macro with incomplete template specializations
 * as well as triple chevron kernel calls, see the following code example
 * @code
 * template <int C1, int C2, typename GenType>
 * RAFT_KERNEL my_kernel(DeviceState<GenType> state, int arg1) { ... }
 *
 * template <int C1, typename GenType, int C2 = 2>
 * void foo(DeviceState<GenType> state, int arg1) {
 *   my_kernel<C1, C2><<<1, 1>>>(state, arg1);
 * }
 *
 * RAFT_CALL_RNG_FUNC(rng_state, foo<1>, 5);
 * RAFT_CALL_RNG_FUNC(rng_state, (my_kernel<1, 2><<<1, 1>>>), 5);
 * @endcode
 */
#define RAFT_CALL_RNG_FUNC(rng_state, func, ...)                                    \
  switch ((rng_state).type) {                                                       \
    case raft::random::GeneratorType::GenPhilox: {                                  \
      raft::random::DeviceState<raft::random::PhiloxGenerator> r_phil{(rng_state)}; \
      RAFT_DEPAREN(func)(r_phil, ##__VA_ARGS__);                                    \
      break;                                                                        \
    }                                                                               \
    case raft::random::GeneratorType::GenPC: {                                      \
      raft::random::DeviceState<raft::random::PCGenerator> r_pc{(rng_state)};       \
      RAFT_DEPAREN(func)(r_pc, ##__VA_ARGS__);                                      \
      break;                                                                        \
    }                                                                               \
    default: RAFT_FAIL("Unexpected generator type '%d'", int((rng_state).type));    \
  }

template <int ITEMS_PER_CALL, typename GenType, typename... ArgsT>
void call_rng_kernel(DeviceState<GenType> const& dev_state,
                     RngState& rng_state,
                     cudaStream_t stream,
                     ArgsT... args)
{
  auto n_threads = 256;
  auto n_blocks  = 4 * getMultiProcessorCount();
  rngKernel<ITEMS_PER_CALL><<<n_blocks, n_threads, 0, stream>>>(dev_state, args...);
  rng_state.advance(uint64_t(n_blocks) * n_threads, 16);
}

template <typename OutType, typename LenType = int>
void uniform(
  RngState& rng_state, OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value || std::is_same_v<OutType, half>,
                "Type for 'uniform' can only be floating point!");
  UniformDistParams<OutType> params;
  params.start = start;
  params.end   = end;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void uniformInt(
  RngState& rng_state, OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
{
  static_assert(std::is_integral<OutType>::value, "Type for 'uniformInt' can only be integer!");
  ASSERT(end > start, "'end' must be greater than 'start'");
  if (sizeof(OutType) == 4) {
    UniformIntDistParams<OutType, uint32_t> params;
    params.start = start;
    params.end   = end;
    params.diff  = uint32_t(params.end - params.start);
    RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
  } else {
    UniformIntDistParams<OutType, uint64_t> params;
    params.start = start;
    params.end   = end;
    params.diff  = uint64_t(params.end - params.start);
    RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
  }
}

template <typename OutType, typename LenType = int>
void normal(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value,
                "Type for 'normal' can only be floating point!");
  NormalDistParams<OutType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, rng_state, stream, ptr, len, params);
}

template <typename IntType, typename LenType = int>
void normalInt(
  RngState& rng_state, IntType* ptr, LenType len, IntType mu, IntType sigma, cudaStream_t stream)
{
  static_assert(std::is_integral<IntType>::value, "Type for 'normalInt' can only be integer!");
  NormalIntDistParams<IntType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void normalTable(RngState& rng_state,
                 OutType* ptr,
                 LenType n_rows,
                 LenType n_cols,
                 const OutType* mu_vec,
                 const OutType* sigma_vec,
                 OutType sigma,
                 cudaStream_t stream)
{
  NormalTableDistParams<OutType, LenType> params;
  params.n_rows    = n_rows;
  params.n_cols    = n_cols;
  params.mu_vec    = mu_vec;
  params.sigma     = sigma;
  params.sigma_vec = sigma_vec;
  LenType len      = n_rows * n_cols;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void fill(RngState& rng_state, OutType* ptr, LenType len, OutType val, cudaStream_t stream)
{
  InvariantDistParams<OutType> params;
  params.const_val = val;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename Type, typename OutType = bool, typename LenType = int>
void bernoulli(RngState& rng_state, OutType* ptr, LenType len, Type prob, cudaStream_t stream)
{
  BernoulliDistParams<Type> params;
  params.prob = prob;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void scaled_bernoulli(
  RngState& rng_state, OutType* ptr, LenType len, OutType prob, OutType scale, cudaStream_t stream)
{
  static_assert(std::is_floating_point<OutType>::value,
                "Type for 'scaled_bernoulli' can only be floating point!");
  ScaledBernoulliDistParams<OutType> params;
  params.prob  = prob;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void gumbel(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType beta, cudaStream_t stream)
{
  GumbelDistParams<OutType> params;
  params.mu   = mu;
  params.beta = beta;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void lognormal(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
{
  LogNormalDistParams<OutType> params;
  params.mu    = mu;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<2>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void logistic(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
{
  LogisticDistParams<OutType> params;
  params.mu    = mu;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void exponential(
  RngState& rng_state, OutType* ptr, LenType len, OutType lambda, cudaStream_t stream)
{
  ExponentialDistParams<OutType> params;
  params.lambda = lambda;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void rayleigh(RngState& rng_state, OutType* ptr, LenType len, OutType sigma, cudaStream_t stream)
{
  RayleighDistParams<OutType> params;
  params.sigma = sigma;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename OutType, typename LenType = int>
void laplace(
  RngState& rng_state, OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
{
  LaplaceDistParams<OutType> params;
  params.mu    = mu;
  params.scale = scale;
  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, ptr, len, params);
}

template <typename GenType, typename OutType, typename WeightType, typename IdxType>
void call_sample_with_replacement_kernel(DeviceState<GenType> const& dev_state,
                                         RngState& rng_state,
                                         cudaStream_t stream,
                                         OutType* out,
                                         const WeightType* weights_csum,
                                         IdxType sampledLen,
                                         IdxType len)
{
  IdxType n_threads = 256;
  IdxType n_blocks  = raft::ceildiv(sampledLen, n_threads);
  sample_with_replacement_kernel<<<n_blocks, n_threads, 0, stream>>>(
    dev_state, out, weights_csum, sampledLen, len);
  rng_state.advance(uint64_t(n_blocks) * n_threads, 1);
}

template <typename OutType, typename WeightType, typename IndexType = OutType>
std::enable_if_t<std::is_integral_v<OutType>> discrete(bool dry_run,
                                                       RngState& rng_state,
                                                       OutType* ptr,
                                                       const WeightType* weights,
                                                       IndexType sampledLen,
                                                       IndexType len,
                                                       cudaStream_t stream)
{
  // Compute the cumulative sums of the weights
  size_t temp_storage_bytes = 0;
  rmm::device_uvector<WeightType> weights_csum(len, stream);
  cub::DeviceScan::InclusiveSum(
    nullptr, temp_storage_bytes, weights, weights_csum.data(), len, stream);
  rmm::device_uvector<uint8_t> temp_storage(temp_storage_bytes, stream);

  if (dry_run) { return; }

  cub::DeviceScan::InclusiveSum(
    temp_storage.data(), temp_storage_bytes, weights, weights_csum.data(), len, stream);

  // Sample indices with replacement
  RAFT_CALL_RNG_FUNC(rng_state,
                     call_sample_with_replacement_kernel,
                     rng_state,
                     stream,
                     ptr,
                     weights_csum.data(),
                     sampledLen,
                     len);
}

/** Note the memory space requirements are O(4*len) */
template <typename DataT, typename WeightsT, typename IdxT = int>
void sampleWithoutReplacement(bool dry_run,
                              RngState& rng_state,
                              DataT* out,
                              IdxT* outIdx,
                              const DataT* in,
                              const WeightsT* wts,
                              IdxT sampledLen,
                              IdxT len,
                              cudaStream_t stream)
{
  ASSERT(sampledLen <= len, "sampleWithoutReplacement: 'sampledLen' cant be more than 'len'.");

  rmm::device_uvector<WeightsT> expWts(len, stream);
  rmm::device_uvector<WeightsT> sortedWts(len, stream);
  rmm::device_uvector<IdxT> inIdx(len, stream);
  rmm::device_uvector<IdxT> outIdxBuff(len, stream);
  auto* inIdxPtr = inIdx.data();
  // generate modified weights
  SamplingParams<WeightsT, IdxT> params;
  params.inIdxPtr = inIdxPtr;
  params.wts      = wts;

  // Query workspace size for sortPairs before dry-run check to track allocation
  size_t workspace_size = 0;
  cub::DeviceRadixSort::SortPairs(nullptr,
                                  workspace_size,
                                  expWts.data(),
                                  sortedWts.data(),
                                  inIdxPtr,
                                  outIdxBuff.data(),
                                  (int)len,
                                  0,
                                  sizeof(WeightsT) * 8,
                                  stream);
  rmm::device_uvector<char> workspace(workspace_size, stream);

  if (dry_run) { return; }

  RAFT_CALL_RNG_FUNC(rng_state, call_rng_kernel<1>, rng_state, stream, expWts.data(), len, params);

  ///@todo: use a more efficient partitioning scheme instead of full sort
  // sort the array and pick the top sampledLen items
  IdxT* outIdxPtr = outIdxBuff.data();
  cub::DeviceRadixSort::SortPairs(workspace.data(),
                                  workspace_size,
                                  expWts.data(),
                                  sortedWts.data(),
                                  inIdxPtr,
                                  outIdxPtr,
                                  (int)len,
                                  0,
                                  sizeof(WeightsT) * 8,
                                  stream);
  if (outIdx != nullptr) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      outIdx, outIdxPtr, sizeof(IdxT) * sampledLen, cudaMemcpyDeviceToDevice, stream));
  }
  scatter<DataT, IdxT>(out, in, outIdxPtr, sampledLen, stream);
}

template <typename IdxT>
void affine_transform_params(RngState const& rng_state, IdxT n, IdxT& a, IdxT& b)
{
  // always keep 'a' to be coprime to 'n'
  std::mt19937_64 mt_rng(rng_state.seed + rng_state.base_subsequence);
  a = mt_rng() % n;
  while (gcd(a, n) != 1) {
    ++a;
    if (a >= n) a = 0;
  }
  // the bias term 'b' can be any number in the range of [0, n)
  b = mt_rng() % n;
}

/** @brief Sample without replacement from range 0..N-1.
 *
 * Elements are sampled uniformly.
 * The algorithm will allocate a workspace of size O(4*n_samples) internally.
 *
 * We use max N random numbers. Depending on how large n_samples is w.r.t to N, we
 * either use rejection sampling, or sort the [0..N-1] values using random keys.
 *
 * @tparam IdxT type of indices that we sample
 * @tparam MatIdxT extent type of the returned mdarray
 *
 * @param res RAFT resource handle
 * @param state random number generator state
 * @param N number of elements to sample from. We will sample values in range 0..N-1
 * @param n_samples number of samples to return
 *
 * @return device mdarray with the random samples
 */
template <typename IdxT, typename MatIdxT = IdxT>
auto excess_subsample(raft::resources const& res, RngState& state, IdxT N, IdxT n_samples)
  -> raft::device_vector<IdxT, MatIdxT>
{
  RAFT_EXPECTS(n_samples <= N, "Cannot have more training samples than dataset vectors");

  // Number of samples we'll need to sample (with replacement), to expect 'k'
  // unique samples from 'n' is given by the following equation: log(1 - k/n)/log(1 - 1/n) ref:
  // https://stats.stackexchange.com/questions/296005/the-expected-number-of-unique-elements-drawn-with-replacement
  IdxT n_excess_samples =
    n_samples < N
      ? std::ceil(raft::log(1 - double(n_samples) / double(N)) / (raft::log(1 - 1 / double(N))))
      : N;

  // There is a variance of n_excess_samples, we take 10% more elements.
  n_excess_samples += std::max<IdxT>(0.1 * n_samples, 100);

  bool dry_run = resource::get_dry_run_flag(res);
  auto stream  = resource::get_cuda_stream(res);

  while (true) {
    // n_excess_sampless will be larger than N around k = 0.64*N. When we reach N, then instead of
    // doing rejection sampling, we simply shuffle the range [0..N-1] using N random numbers.
    n_excess_samples = std::min<IdxT>(n_excess_samples, N);
    auto rnd_idx     = raft::make_device_vector<IdxT, IdxT>(res, n_excess_samples);
    auto linear_idx  = raft::make_device_vector<IdxT, IdxT>(res, rnd_idx.size());

    // Workspace size queries (safe with nullptr)
    size_t workspace_size = 0;
    cub::DeviceMergeSort::SortPairs(nullptr,
                                    workspace_size,
                                    rnd_idx.data_handle(),
                                    linear_idx.data_handle(),
                                    rnd_idx.size(),
                                    raft::less_op{},
                                    stream);

    auto keys_out   = raft::make_device_vector<IdxT, IdxT>(res, rnd_idx.size());
    auto values_out = raft::make_device_vector<IdxT, IdxT>(res, rnd_idx.size());
    rmm::device_scalar<IdxT> num_selected(stream);
    size_t worksize2 = 0;
    cub::DeviceSelect::UniqueByKey(nullptr,
                                   worksize2,
                                   rnd_idx.data_handle(),
                                   linear_idx.data_handle(),
                                   keys_out.data_handle(),
                                   values_out.data_handle(),
                                   num_selected.data(),
                                   rnd_idx.size(),
                                   stream);

    workspace_size = std::max(workspace_size, worksize2);
    auto workspace = raft::make_device_vector<char, IdxT>(res, workspace_size);

    if (dry_run) { return raft::make_device_vector<IdxT, IdxT>(res, n_samples); }

    raft::linalg::map_offset(res, linear_idx.view(), identity_op());
    uniformInt(res, state, rnd_idx.data_handle(), rnd_idx.size(), IdxT(0), IdxT(N));

    // Sort indices according to rnd keys
    cub::DeviceMergeSort::SortPairs(workspace.data_handle(),
                                    workspace_size,
                                    rnd_idx.data_handle(),
                                    linear_idx.data_handle(),
                                    rnd_idx.size(),
                                    raft::less_op{},
                                    stream);

    if (rnd_idx.size() == static_cast<size_t>(N)) {
      // We shuffled the linear_idx array by sorting it according to rnd_idx.
      // We return the first n_samples elements.
      if (n_samples == N) { return linear_idx; }
      rnd_idx = raft::make_device_vector<IdxT, IdxT>(res, n_samples);
      raft::copy(rnd_idx.data_handle(), linear_idx.data_handle(), n_samples, stream);
      return rnd_idx;
    }
    // Else we do a rejection sampling (or excess sampling): we generated more random indices than
    // needed and reject the duplicates.
    cub::DeviceSelect::UniqueByKey(workspace.data_handle(),
                                   workspace_size,
                                   rnd_idx.data_handle(),
                                   linear_idx.data_handle(),
                                   keys_out.data_handle(),
                                   values_out.data_handle(),
                                   num_selected.data(),
                                   rnd_idx.size(),
                                   stream);

    IdxT selected = num_selected.value(stream);

    if (selected < n_samples) {
      RAFT_LOG_DEBUG("Subsampling returned with less unique indices (%zu) than requested (%zu)",
                     (size_t)selected,
                     (size_t)n_samples);

      // We just restart the algorithm with 20% increased excess samples
      n_excess_samples += std::max<IdxT>(0.2 * n_excess_samples, 100);
    } else {
      // After duplicates are removed, we need to shuffle back to random order
      cub::DeviceMergeSort::SortPairs(workspace.data_handle(),
                                      workspace_size,
                                      values_out.data_handle(),
                                      keys_out.data_handle(),
                                      selected,
                                      raft::less_op{},
                                      stream);

      values_out = raft::make_device_vector<IdxT, IdxT>(res, n_samples);
      raft::copy(values_out.data_handle(), keys_out.data_handle(), n_samples, stream);
      return values_out;
    }
  }
}

};  // end namespace detail
};  // end namespace random
};  // namespace raft
