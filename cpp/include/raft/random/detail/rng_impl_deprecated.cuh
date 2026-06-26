/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * DISCLAIMER: this file is deprecated and will be removed in a future release
 */

#pragma once

#include "rng_device.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/scatter.cuh>

#include <rmm/device_uvector.hpp>

#include <cub/device/device_radix_sort.cuh>

#include <curand_kernel.h>

#include <random>

namespace raft {
namespace random {
namespace detail {

#define METHOD_DEPR(new_name) [[deprecated("Use the flat '" #new_name "' function instead")]]

class RngImpl {
 public:
  RngImpl(uint64_t seed, GeneratorType _t = GenPhilox)
    : state{seed, 0, _t},
      type(_t),
      // simple heuristic to make sure all SMs will be occupied properly
      // and also not too many initialization calls will be made by each thread
      nBlocks(4 * getMultiProcessorCount())
  {
  }

  template <typename IdxT>
  METHOD_DEPR(affine_transform_params)
  void affine_transform_params(IdxT n, IdxT& a, IdxT& b)
  {
    // always keep 'a' to be coprime to 'n'
    std::mt19937_64 mt_rng(state.seed + state.base_subsequence);
    a = mt_rng() % n;
    while (gcd(a, n) != 1) {
      ++a;
      if (a >= n) a = 0;
    }
    // the bias term 'b' can be any number in the range of [0, n)
    b = mt_rng() % n;
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(uniform)
  void uniform(OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<OutType>::value,
                  "Type for 'uniform' can only be floating point!");
    UniformDistParams<OutType> params;
    params.start = start;
    params.end   = end;
    kernel_dispatch<OutType, LenType, 1, UniformDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(uniformInt)
  void uniformInt(OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
  {
    static_assert(std::is_integral<OutType>::value, "Type for 'uniformInt' can only be integer!");
    ASSERT(end > start, "'end' must be greater than 'start'");
    if (sizeof(OutType) == 4) {
      UniformIntDistParams<OutType, uint32_t> params;
      params.start = start;
      params.end   = end;
      params.diff  = uint32_t(params.end - params.start);
      kernel_dispatch<OutType, LenType, 1, UniformIntDistParams<OutType, uint32_t>>(
        ptr, len, stream, params);
    } else {
      UniformIntDistParams<OutType, uint64_t> params;
      params.start = start;
      params.end   = end;
      params.diff  = uint64_t(params.end - params.start);
      kernel_dispatch<OutType, LenType, 1, UniformIntDistParams<OutType, uint64_t>>(
        ptr, len, stream, params);
    }
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(normal)
  void normal(OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<OutType>::value,
                  "Type for 'normal' can only be floating point!");
    NormalDistParams<OutType> params;
    params.mu    = mu;
    params.sigma = sigma;
    kernel_dispatch<OutType, LenType, 2, NormalDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename IntType, typename LenType = int>
  METHOD_DEPR(normalInt)
  void normalInt(IntType* ptr, LenType len, IntType mu, IntType sigma, cudaStream_t stream)
  {
    static_assert(std::is_integral<IntType>::value, "Type for 'normalInt' can only be integer!");
    NormalIntDistParams<IntType> params;
    params.mu    = mu;
    params.sigma = sigma;
    kernel_dispatch<IntType, LenType, 2, NormalIntDistParams<IntType>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(normalTable)
  void normalTable(OutType* ptr,
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
    kernel_dispatch<OutType, LenType, 2, NormalTableDistParams<OutType, LenType>>(
      ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(fill)
  void fill(OutType* ptr, LenType len, OutType val, cudaStream_t stream)
  {
    InvariantDistParams<OutType> params;
    params.const_val = val;
    kernel_dispatch<OutType, LenType, 1, InvariantDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename Type, typename OutType = bool, typename LenType = int>
  METHOD_DEPR(bernoulli)
  void bernoulli(OutType* ptr, LenType len, Type prob, cudaStream_t stream)
  {
    BernoulliDistParams<Type> params;
    params.prob = prob;
    kernel_dispatch<OutType, LenType, 1, BernoulliDistParams<Type>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(scaled_bernoulli)
  void scaled_bernoulli(OutType* ptr, LenType len, OutType prob, OutType scale, cudaStream_t stream)
  {
    static_assert(std::is_floating_point<OutType>::value,
                  "Type for 'scaled_bernoulli' can only be floating point!");
    ScaledBernoulliDistParams<OutType> params;
    params.prob  = prob;
    params.scale = scale;
    kernel_dispatch<OutType, LenType, 1, ScaledBernoulliDistParams<OutType>>(
      ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(gumbel)
  void gumbel(OutType* ptr, LenType len, OutType mu, OutType beta, cudaStream_t stream)
  {
    GumbelDistParams<OutType> params;
    params.mu   = mu;
    params.beta = beta;
    kernel_dispatch<OutType, LenType, 1, GumbelDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(lognormal)
  void lognormal(OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
  {
    LogNormalDistParams<OutType> params;
    params.mu    = mu;
    params.sigma = sigma;
    kernel_dispatch<OutType, LenType, 2, LogNormalDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(logistic)
  void logistic(OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
  {
    LogisticDistParams<OutType> params;
    params.mu    = mu;
    params.scale = scale;
    kernel_dispatch<OutType, LenType, 1, LogisticDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(exponential)
  void exponential(OutType* ptr, LenType len, OutType lambda, cudaStream_t stream)
  {
    ExponentialDistParams<OutType> params;
    params.lambda = lambda;
    kernel_dispatch<OutType, LenType, 1, ExponentialDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(rayleigh)
  void rayleigh(OutType* ptr, LenType len, OutType sigma, cudaStream_t stream)
  {
    RayleighDistParams<OutType> params;
    params.sigma = sigma;
    kernel_dispatch<OutType, LenType, 1, RayleighDistParams<OutType>>(ptr, len, stream, params);
  }

  template <typename OutType, typename LenType = int>
  METHOD_DEPR(laplace)
  void laplace(OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
  {
    LaplaceDistParams<OutType> params;
    params.mu    = mu;
    params.scale = scale;
    kernel_dispatch<OutType, LenType, 1, LaplaceDistParams<OutType>>(ptr, len, stream, params);
  }

  void advance(uint64_t max_uniq_subsequences_used,
               uint64_t max_numbers_generated_per_subsequence = 0)
  {
    state.advance(max_uniq_subsequences_used, max_numbers_generated_per_subsequence);
  }

  template <typename OutType, typename LenType, int ITEMS_PER_CALL, typename ParamType>
  void kernel_dispatch(OutType* ptr, LenType len, cudaStream_t stream, ParamType params)
  {
    switch (state.type) {
      case GenPhilox:
        fillKernel<OutType, LenType, PhiloxGenerator, ITEMS_PER_CALL>
          <<<nBlocks, nThreads, 0, stream>>>(
            state.seed, state.base_subsequence, 0, ptr, len, params);
        break;
      case GenPC:
        fillKernel<OutType, LenType, PCGenerator, ITEMS_PER_CALL><<<nBlocks, nThreads, 0, stream>>>(
          state.seed, state.base_subsequence, 0, ptr, len, params);
        break;
      default: break;
    }
    // The max_numbers_generated_per_subsequence parameter does not matter for now, using 16 for now
    advance(uint64_t(nBlocks) * nThreads, 16);
    return;
  }

  template <typename DataT, typename WeightsT, typename IdxT = int>
  METHOD_DEPR(sampleWithoutReplacement)
  void sampleWithoutReplacement(raft::resources const& handle,
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

    if (resource::get_dry_run_flag(handle)) { return; }
    kernel_dispatch<WeightsT, IdxT, 1, SamplingParams<WeightsT, IdxT>>(
      expWts.data(), len, stream, params);
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

  RngState state;
  GeneratorType type;
  /** number of blocks to launch */
  int nBlocks;
  static const int nThreads = 256;
};

#undef METHOD_DEPR

};  // end namespace detail
};  // end namespace random
};  // namespace raft
