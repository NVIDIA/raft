/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/integer_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_fp16.h>

#include <curand_kernel.h>

#include <random>

namespace raft {
namespace random {
namespace detail {

/**
 * The device state used to communicate RNG state from host to device.
 * As of now, it is just a templated version of `RngState`.
 */
template <typename GenType>
struct DeviceState {
  using gen_t                    = GenType;
  static constexpr auto GEN_TYPE = gen_t::GEN_TYPE;

  explicit DeviceState(const RngState& rng_state)
    : seed(rng_state.seed), base_subsequence(rng_state.base_subsequence)
  {
  }

  uint64_t seed;
  uint64_t base_subsequence;
};

template <typename OutType>
struct InvariantDistParams {
  OutType const_val;
};

template <typename OutType>
struct UniformDistParams {
  OutType start;
  OutType end;
};

template <typename OutType, typename DiffType>
struct UniformIntDistParams {
  OutType start;
  OutType end;
  DiffType diff;
};

template <typename OutType>
struct NormalDistParams {
  OutType mu;
  OutType sigma;
};

template <typename IntType>
struct NormalIntDistParams {
  IntType mu;
  IntType sigma;
};

template <typename OutType, typename LenType>
struct NormalTableDistParams {
  LenType n_rows;
  LenType n_cols;
  const OutType* mu_vec;
  OutType sigma;
  const OutType* sigma_vec;
};

template <typename OutType>
struct BernoulliDistParams {
  OutType prob;
};

template <typename OutType>
struct ScaledBernoulliDistParams {
  OutType prob;
  OutType scale;
};

template <typename OutType>
struct GumbelDistParams {
  OutType mu;
  OutType beta;
};

template <typename OutType>
struct LogNormalDistParams {
  OutType mu;
  OutType sigma;
};

template <typename OutType>
struct LogisticDistParams {
  OutType mu;
  OutType scale;
};

template <typename OutType>
struct ExponentialDistParams {
  OutType lambda;
};

template <typename OutType>
struct RayleighDistParams {
  OutType sigma;
};

template <typename OutType>
struct LaplaceDistParams {
  OutType mu;
  OutType scale;
};

// Not really a distro, useful for sample without replacement function
template <typename WeightsT, typename IdxT>
struct SamplingParams {
  IdxT* inIdxPtr;
  const WeightsT* wts;
};

template <typename Type>
HDI void box_muller_transform(Type& val1, Type& val2, Type sigma1, Type mu1, Type sigma2, Type mu2)
{
  constexpr Type twoPi  = Type(2.0) * Type(3.141592653589793);
  constexpr Type minus2 = -Type(2.0);
  Type R                = raft::sqrt(minus2 * raft::log(val1));
  Type theta            = twoPi * val2;
  Type s, c;
  raft::sincos(theta, &s, &c);
  val1 = R * c * sigma1 + mu1;
  val2 = R * s * sigma2 + mu2;
}

template <typename Type>
HDI void box_muller_transform(Type& val1, Type& val2, Type sigma1, Type mu1)
{
  box_muller_transform<Type>(val1, val2, sigma1, mu1, sigma1, mu1);
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     InvariantDistParams<OutType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  *val = params.const_val;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     UniformDistParams<OutType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  OutType res;
  gen.next(res);
  *val = (res * (params.end - params.start)) + params.start;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     UniformIntDistParams<OutType, uint32_t> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  uint32_t x = 0;
  uint32_t s = params.diff;
  gen.next(x);
  uint64_t m = uint64_t(x) * s;
  uint32_t l = uint32_t(m);
  if (l < s) {
    uint32_t t = (-s) % s;  // (2^32 - s) mod s
    while (l < t) {
      gen.next(x);
      m = uint64_t(x) * s;
      l = uint32_t(m);
    }
  }
  *val = OutType(m >> 32) + params.start;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     UniformIntDistParams<OutType, uint64_t> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  using raft::wmul_64bit;
  uint64_t x = 0;
  gen.next(x);
  uint64_t s = params.diff;
  uint64_t m_lo, m_hi;
  // m = x * s;
  wmul_64bit(m_hi, m_lo, x, s);
  if (m_lo < s) {
    uint64_t t = (-s) % s;  // (2^64 - s) mod s
    while (m_lo < t) {
      gen.next(x);
      wmul_64bit(m_hi, m_lo, x, s);
    }
  }
  *val = OutType(m_hi) + params.start;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(
  GenType& gen, OutType* val, NormalDistParams<OutType> params, LenType idx = 0, LenType stride = 0)
{
  OutType res1, res2;

  do {
    gen.next(res1);
  } while (res1 == OutType(0.0));

  gen.next(res2);

  box_muller_transform<OutType>(res1, res2, params.sigma, params.mu);
  *val       = res1;
  *(val + 1) = res2;
}

template <typename GenType, typename IntType, typename LenType>
HDI void custom_next(GenType& gen,
                     IntType* val,
                     NormalIntDistParams<IntType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  // Draw a zero-mean deviate in the narrowest type that can carry it, then shift by mu in the
  // output type. Folding mu into the transform would round mu itself and, worse, quantize the
  // deviate away entirely once |mu| exceeds the mantissa of the compute type (2^24 for float,
  // 2^53 for double): at mu = 2e9 the float spacing is 128, so a sigma of 10 vanishes.
  // With mu applied separately, float carries any deviate an integer output can represent, and
  // double is only needed for the wider deviates of 64-bit outputs. Double is 1/64 rate on
  // consumer GPUs, so this matters.
  using compute_t = std::conditional_t<(sizeof(IntType) > 4), double, float>;
  compute_t res1, res2;
  do {
    gen.next(res1);
  } while (res1 == compute_t(0.0));

  gen.next(res2);
  compute_t sigma = static_cast<compute_t>(params.sigma);
  box_muller_transform<compute_t>(res1, res2, sigma, compute_t(0));
  *val       = static_cast<IntType>(res1) + params.mu;
  *(val + 1) = static_cast<IntType>(res2) + params.mu;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     NormalTableDistParams<OutType, LenType> params,
                     LenType idx,
                     LenType stride)
{
  OutType res1, res2;

  do {
    gen.next(res1);
  } while (res1 == OutType(0.0));

  gen.next(res2);
  LenType col1  = idx % params.n_cols;
  LenType col2  = (idx + stride) % params.n_cols;
  OutType mean1 = params.mu_vec[col1];
  OutType mean2 = params.mu_vec[col2];
  OutType sig1  = params.sigma_vec == nullptr ? params.sigma : params.sigma_vec[col1];
  OutType sig2  = params.sigma_vec == nullptr ? params.sigma : params.sigma_vec[col2];
  box_muller_transform<OutType>(res1, res2, sig1, mean1, sig2, mean2);
  *val       = res1;
  *(val + 1) = res2;
}

template <typename GenType, typename OutType, typename Type, typename LenType>
HDI void custom_next(
  GenType& gen, OutType* val, BernoulliDistParams<Type> params, LenType idx = 0, LenType stride = 0)
{
  Type res = 0;
  gen.next(res);
  *val = res < params.prob;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     ScaledBernoulliDistParams<OutType> params,
                     LenType idx,
                     LenType stride)
{
  OutType res = 0;
  gen.next(res);
  *val = res < params.prob ? -params.scale : params.scale;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(
  GenType& gen, OutType* val, GumbelDistParams<OutType> params, LenType idx = 0, LenType stride = 0)
{
  OutType res = 0;

  do {
    gen.next(res);
  } while (res == OutType(0.0));

  *val = params.mu - params.beta * raft::log(-raft::log(res));
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     LogNormalDistParams<OutType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  OutType res1 = 0, res2 = 0;
  do {
    gen.next(res1);
  } while (res1 == OutType(0.0));

  gen.next(res2);
  box_muller_transform<OutType>(res1, res2, params.sigma, params.mu);
  *val       = raft::exp(res1);
  *(val + 1) = raft::exp(res2);
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     LogisticDistParams<OutType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  OutType res;

  do {
    gen.next(res);
  } while (res == OutType(0.0));

  constexpr OutType one = (OutType)1.0;
  *val                  = params.mu - params.scale * raft::log(one / res - one);
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     ExponentialDistParams<OutType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  OutType res;
  gen.next(res);
  constexpr OutType one = (OutType)1.0;
  *val                  = -raft::log(one - res) / params.lambda;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     RayleighDistParams<OutType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  OutType res;
  gen.next(res);

  constexpr OutType one = (OutType)1.0;
  constexpr OutType two = (OutType)2.0;
  *val                  = raft::sqrt(-two * raft::log(one - res)) * params.sigma;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(GenType& gen,
                     OutType* val,
                     LaplaceDistParams<OutType> params,
                     LenType idx    = 0,
                     LenType stride = 0)
{
  OutType res, out;

  do {
    gen.next(res);
  } while (res == OutType(0.0));

  constexpr OutType one     = (OutType)1.0;
  constexpr OutType two     = (OutType)2.0;
  constexpr OutType oneHalf = (OutType)0.5;

  // The <= comparison here means, number of samples going in `if` branch are more by 1 than `else`
  // branch. However it does not matter as for 0.5 both branches evaluate to same result.
  if (res <= oneHalf) {
    out = params.mu + params.scale * raft::log(two * res);
  } else {
    out = params.mu - params.scale * raft::log(two * (one - res));
  }
  *val = out;
}

template <typename GenType, typename OutType, typename LenType>
HDI void custom_next(
  GenType& gen, OutType* val, SamplingParams<OutType, LenType> params, LenType idx, LenType stride)
{
  OutType res;
  gen.next(res);
  params.inIdxPtr[idx]  = idx;
  constexpr OutType one = (OutType)1.0;
  auto exp              = -raft::log(one - res);
  if (params.wts != nullptr) {
    *val = exp / params.wts[idx];
  } else {
    *val = exp;
  }
}

/** Philox-based random number generator */
// Courtesy: Jakub Szuppe
struct PhiloxGenerator {
  static constexpr auto GEN_TYPE = GeneratorType::GenPhilox;

  /**
   * @brief ctor. Initializes the state for RNG
   * @param seed random seed (can be same across all threads)
   * @param subsequence as found in curand docs
   * @param offset as found in curand docs
   */
  DI PhiloxGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    curand_init(seed, subsequence, offset, &philox_state);
  }

  DI PhiloxGenerator(const DeviceState<PhiloxGenerator>& rng_state, const uint64_t subsequence)
  {
    curand_init(rng_state.seed, rng_state.base_subsequence + subsequence, 0, &philox_state);
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @{
   */
  DI uint32_t next_u32()
  {
    uint32_t ret = curand(&(this->philox_state));
    return ret;
  }

  DI uint64_t next_u64()
  {
    uint64_t ret;
    uint32_t a, b;
    a   = next_u32();
    b   = next_u32();
    ret = uint64_t(a) | (uint64_t(b) << 32);
    return ret;
  }

  DI int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  DI int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  DI float next_float()
  {
    float ret;
    uint32_t val = next_u32() >> 8;
    ret          = static_cast<float>(val) / float(uint32_t(1) << 24);
    return ret;
  }

  DI double next_double()
  {
    double ret;
    uint64_t val = next_u64() >> 11;
    ret          = static_cast<double>(val) / double(uint64_t(1) << 53);
    return ret;
  }

  DI half next_half()
  {
    float ret = next_float();
    return __float2half(ret);
  }

  DI void next(float& ret)
  {
    // ret = curand_uniform(&(this->philox_state));
    ret = next_float();
  }

  DI void next(double& ret)
  {
    // ret = curand_uniform_double(&(this->philox_state));
    ret = next_double();
  }

  DI void next(half& ret)
  {
    // ret = curand_uniform_double(&(this->philox_state));
    ret = next_half();
  }

  DI void next(uint32_t& ret) { ret = next_u32(); }
  DI void next(uint64_t& ret) { ret = next_u64(); }
  DI void next(int32_t& ret) { ret = next_i32(); }
  DI void next(int64_t& ret) { ret = next_i64(); }

  /** @} */

 private:
  /** the state for RNG */
  curandStatePhilox4_32_10_t philox_state;
};

/** PCG random number generator */
struct PCGenerator {
  static constexpr auto GEN_TYPE = GeneratorType::GenPC;

  /**
   * @brief ctor. Initializes the PCG
   * @param rng_state is the generator state used for initializing the generator
   * @param subsequence specifies the subsequence to be generated out of 2^64 possible subsequences
   * In a parallel setting, like threads of a CUDA kernel, each thread is required to generate a
   * unique set of random numbers. This can be achieved by initializing the generator with same
   * rng_state for all the threads and distinct values for subsequence.
   */
  HDI PCGenerator(const DeviceState<PCGenerator>& rng_state, const uint64_t subsequence)
  {
    _init_pcg(rng_state.seed, rng_state.base_subsequence + subsequence, subsequence);
  }

  /**
   * @brief ctor. This is lower level constructor for PCG
   * This code is derived from PCG basic code
   * @param seed A 64-bit seed for the generator
   * @param subsequence The id of subsequence that should be generated [0, 2^64-1]
   * @param offset Initial `offset` number of items are skipped from the subsequence
   */
  HDI PCGenerator(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    _init_pcg(seed, subsequence, offset);
  }

  // Based on "Random Number Generation with Arbitrary Strides" F. B. Brown
  // Link https://mcnp.lanl.gov/pdf_files/anl-rn-arb-stride.pdf
  HDI void skipahead(uint64_t offset)
  {
    uint64_t G = 1;
    uint64_t h = 6364136223846793005ULL;
    uint64_t C = 0;
    uint64_t f = inc;
    while (offset) {
      if (offset & 1) {
        G = G * h;
        C = C * h + f;
      }
      f = f * (h + 1);
      h = h * h;
      offset >>= 1;
    }
    pcg_state = pcg_state * G + C;
  }

  /**
   * @defgroup NextRand Generate the next random number
   * @brief This code is derived from PCG basic code
   * @{
   */
  HDI uint32_t next_u32()
  {
    uint32_t ret;
    uint64_t oldstate   = pcg_state;
    pcg_state           = oldstate * 6364136223846793005ULL + inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot        = oldstate >> 59u;
    ret                 = (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    return ret;
  }
  HDI uint64_t next_u64()
  {
    uint64_t ret;
    uint32_t a, b;
    a   = next_u32();
    b   = next_u32();
    ret = uint64_t(a) | (uint64_t(b) << 32);
    return ret;
  }

  HDI int32_t next_i32()
  {
    int32_t ret;
    uint32_t val;
    val = next_u32();
    ret = int32_t(val & 0x7fffffff);
    return ret;
  }

  HDI int64_t next_i64()
  {
    int64_t ret;
    uint64_t val;
    val = next_u64();
    ret = int64_t(val & 0x7fffffffffffffff);
    return ret;
  }

  HDI float next_float()
  {
    float ret;
    uint32_t val = next_u32() >> 8;
    ret          = static_cast<float>(val) / (1U << 24);
    return ret;
  }

  HDI double next_double()
  {
    double ret;
    uint64_t val = next_u64() >> 11;
    ret          = static_cast<double>(val) / (1LU << 53);
    return ret;
  }

  HDI half next_half()
  {
    float ret = next_float();
    return __float2half(ret);
  }

  HDI void next(uint32_t& ret) { ret = next_u32(); }
  HDI void next(uint64_t& ret) { ret = next_u64(); }
  HDI void next(int32_t& ret) { ret = next_i32(); }
  HDI void next(int64_t& ret) { ret = next_i64(); }

  HDI void next(float& ret) { ret = next_float(); }
  HDI void next(double& ret) { ret = next_double(); }
  HDI void next(half& ret) { ret = next_half(); }

  /** @} */

 private:
  HDI void _init_pcg(uint64_t seed, uint64_t subsequence, uint64_t offset)
  {
    pcg_state = uint64_t(0);
    inc       = (subsequence << 1u) | 1u;
    uint32_t discard;
    next(discard);
    pcg_state += seed;
    next(discard);
    skipahead(offset);
  }
  uint64_t pcg_state;
  uint64_t inc;
};

template <int ITEMS_PER_CALL,
          typename OutType,
          typename LenType,
          typename GenType,
          typename ParamType>
RAFT_KERNEL rngKernel(DeviceState<GenType> rng_state, OutType* ptr, LenType len, ParamType params)
{
  LenType tid = threadIdx.x + static_cast<LenType>(blockIdx.x) * blockDim.x;
  GenType gen(rng_state, (uint64_t)tid);
  const LenType stride = gridDim.x * blockDim.x;
  for (LenType idx = tid; idx < len; idx += stride * ITEMS_PER_CALL) {
    OutType val[ITEMS_PER_CALL];
    custom_next(gen, val, params, idx, stride);
#pragma unroll
    for (int i = 0; i < ITEMS_PER_CALL; i++) {
      if ((idx + i * stride) < len) ptr[idx + i * stride] = val[i];
    }
  }
  return;
}

template <typename GenType, typename OutType, typename WeightType, typename IdxType>
RAFT_KERNEL sample_with_replacement_kernel(DeviceState<GenType> rng_state,
                                           OutType* out,
                                           const WeightType* weights_csum,
                                           IdxType sampledLen,
                                           IdxType len)
{
  // todo(lsugy): warp-collaborative binary search

  IdxType tid = threadIdx.x + static_cast<IdxType>(blockIdx.x) * blockDim.x;
  GenType gen(rng_state, static_cast<uint64_t>(tid));

  if (tid < sampledLen) {
    WeightType val_01;
    gen.next(val_01);
    WeightType val_search = val_01 * weights_csum[len - 1];

    // Binary search of the first index for which the cumulative sum of weights is larger than the
    // generated value
    IdxType idx_start = 0;
    IdxType idx_end   = len;
    while (idx_end > idx_start) {
      IdxType idx_middle    = (idx_start + idx_end) / 2;
      WeightType val_middle = weights_csum[idx_middle];
      if (val_search <= val_middle) {
        idx_end = idx_middle;
      } else {
        idx_start = idx_middle + 1;
      }
    }
    out[tid] = static_cast<OutType>(min(idx_start, len - 1));
  }
}

/**
 * This kernel is deprecated and should be removed in a future release
 */
template <typename OutType,
          typename LenType,
          typename GenType,
          int ITEMS_PER_CALL,
          typename ParamType>
RAFT_KERNEL fillKernel(
  uint64_t seed, uint64_t adv_subs, uint64_t offset, OutType* ptr, LenType len, ParamType params)
{
  LenType tid = threadIdx.x + static_cast<LenType>(blockIdx.x) * blockDim.x;
  GenType gen(seed, adv_subs + (uint64_t)tid, offset);
  const LenType stride = gridDim.x * blockDim.x;
  for (LenType idx = tid; idx < len; idx += stride * ITEMS_PER_CALL) {
    OutType val[ITEMS_PER_CALL];
    custom_next(gen, val, params, idx, stride);
#pragma unroll
    for (int i = 0; i < ITEMS_PER_CALL; i++) {
      if ((idx + i * stride) < len) ptr[idx + i * stride] = val[i];
    }
  }
  return;
}

#undef POTENTIAL_DEPR

};  // end namespace detail
};  // end namespace random
};  // namespace raft
