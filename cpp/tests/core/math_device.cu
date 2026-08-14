/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.h"

#include <raft/core/math.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_scalar.hpp>

#include <cuda/std/type_traits>

#include <gtest/gtest.h>

#ifdef _RAFT_HAS_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif

template <typename OutT, typename OpT, typename... Args>
RAFT_KERNEL math_eval_kernel(OutT* out, OpT op, Args... args)
{
  out[0] = op(std::forward<Args>(args)...);
}

template <typename OpT, typename... Args>
auto math_eval(OpT op, Args&&... args)
{
  using OutT  = cuda::std::invoke_result_t<OpT, Args...>;
  auto stream = rmm::cuda_stream_default;
  rmm::device_scalar<OutT> result(stream);
  raft::launch_kernel(
    stream, 1, 1, math_eval_kernel, result.data(), op, std::forward<Args>(args)...);
  return result.value(stream);
}

struct abs_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::abs(in);
  }
};

TEST(MathDevice, Abs)
{
  // Integer abs
  ASSERT_TRUE(
    raft::match(int8_t{123}, math_eval(abs_test_op{}, int8_t{-123}), raft::Compare<int8_t>()));
  ASSERT_TRUE(raft::match(12345, math_eval(abs_test_op{}, -12345), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(12345l, math_eval(abs_test_op{}, -12345l), raft::Compare<long int>()));
  ASSERT_TRUE(raft::match(123451234512345ll,
                          math_eval(abs_test_op{}, -123451234512345ll),
                          raft::Compare<long long int>()));
  // Floating-point abs
  ASSERT_TRUE(
    raft::match(12.34f, math_eval(abs_test_op{}, -12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(12.34, math_eval(abs_test_op{}, -12.34), raft::CompareApprox<double>(0.000001)));
}

struct acos_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::acos(in);
  }
};

TEST(MathDevice, Acos)
{
  ASSERT_TRUE(raft::match(
    std::acos(0.123f), math_eval(acos_test_op{}, 0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::acos(0.123), math_eval(acos_test_op{}, 0.123), raft::CompareApprox<double>(0.000001)));
}

struct asin_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::asin(in);
  }
};

TEST(MathDevice, Asin)
{
  ASSERT_TRUE(raft::match(
    std::asin(0.123f), math_eval(asin_test_op{}, 0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::asin(0.123), math_eval(asin_test_op{}, 0.123), raft::CompareApprox<double>(0.000001)));
}

struct atanh_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::atanh(in);
  }
};

TEST(MathDevice, Atanh)
{
  ASSERT_TRUE(raft::match(
    std::atanh(0.123f), math_eval(atanh_test_op{}, 0.123f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::atanh(0.123), math_eval(atanh_test_op{}, 0.123), raft::CompareApprox<double>(0.000001)));
}

struct cos_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::cos(in);
  }
};

struct cos_test_op_device {
  template <typename Type>
  constexpr RAFT_DEVICE_INLINE_FUNCTION auto operator()(const Type& in) const
  {
#if (__CUDA_ARCH__ < 530)
    if constexpr (std::is_same_v<Type, __half>) {
      return __float2half(raft::cos(__half2float(in)));
    }
#elif (__CUDA_ARCH__ < 800)
    if constexpr (std::is_same_v<Type, nv_bfloat16>) {
      return __float2bfloat16(raft::cos(__bfloat162float(in)));
    } else  // else is there to make sure raft::cos(in) is not compiled with __half / nv_bfloat16
#endif
    return raft::cos(in);
  }
};

TEST(MathDevice, Cos)
{
  ASSERT_TRUE(raft::match(std::cos(12.34f),
                          __half2float(math_eval(cos_test_op_device{}, __float2half(12.34f))),
                          raft::CompareApprox<float>(0.001f)));
  ASSERT_TRUE(
    raft::match(std::cos(12.34f),
                __bfloat162float(math_eval(cos_test_op_device{}, __float2bfloat16(12.34f))),
                raft::CompareApprox<float>(0.01f)));
  ASSERT_TRUE(raft::match(
    std::cos(12.34f), math_eval(cos_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::cos(12.34), math_eval(cos_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct exp_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::exp(in);
  }
};

struct exp_test_op_device {
  template <typename Type>
  constexpr RAFT_DEVICE_INLINE_FUNCTION auto operator()(const Type& in) const
  {
#if (__CUDA_ARCH__ < 530)
    if constexpr (std::is_same_v<Type, __half>) {
      return __float2half(raft::exp(__half2float(in)));
    }
#elif (__CUDA_ARCH__ < 800)
    if constexpr (std::is_same_v<Type, nv_bfloat16>) {
      return __float2bfloat16(raft::exp(__bfloat162float(in)));
    } else  // else is there to make sure raft::exp(in) is not compiled with __half / nv_bfloat16
#endif
    return raft::exp(in);
  }
};

TEST(MathDevice, Exp)
{
  ASSERT_TRUE(raft::match(std::exp(3.4f),
                          __half2float(math_eval(exp_test_op_device{}, __float2half(3.4f))),
                          raft::CompareApprox<float>(0.001f)));
  ASSERT_TRUE(raft::match(std::exp(3.4f),
                          __bfloat162float(math_eval(exp_test_op_device{}, __float2bfloat16(3.4f))),
                          raft::CompareApprox<float>(0.01f)));
  ASSERT_TRUE(raft::match(
    std::exp(3.4f), math_eval(exp_test_op{}, 3.4f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::exp(3.4), math_eval(exp_test_op{}, 3.4), raft::CompareApprox<double>(0.000001)));
}

struct log_test_op_device {
  template <typename Type>
  constexpr RAFT_DEVICE_INLINE_FUNCTION auto operator()(const Type& in) const
  {
#if (__CUDA_ARCH__ < 530)
    if constexpr (std::is_same_v<Type, __half>) {
      return __float2half(raft::log(__half2float(in)));
    }
#elif (__CUDA_ARCH__ < 800)
    if constexpr (std::is_same_v<Type, nv_bfloat16>) {
      return __float2bfloat16(raft::log(__bfloat162float(in)));
    } else  // else is there to make sure raft::log(in) is not compiled with __half / nv_bfloat16
#endif
    return raft::log(in);
  }
};

struct log_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::log(in);
  }
};

TEST(MathDevice, Log)
{
  ASSERT_TRUE(raft::match(std::log(12.34f),
                          __half2float(math_eval(log_test_op_device{}, __float2half(12.34f))),
                          raft::CompareApprox<float>(0.001f)));
  ASSERT_TRUE(
    raft::match(std::log(12.34f),
                __bfloat162float(math_eval(log_test_op_device{}, __float2bfloat16(12.34f))),
                raft::CompareApprox<float>(0.01f)));
  ASSERT_TRUE(raft::match(
    std::log(12.34f), math_eval(log_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::log(12.34), math_eval(log_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct max_test_op {
  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    return raft::max(std::forward<Args>(args)...);
  }
};

TEST(MathDevice, Max2)
{
  ASSERT_TRUE(raft::match(1234, math_eval(max_test_op{}, -1234, 1234), raft::Compare<int>()));
  ASSERT_TRUE(
    raft::match(1234u, math_eval(max_test_op{}, 1234u, 123u), raft::Compare<unsigned int>()));
  ASSERT_TRUE(
    raft::match(1234ll, math_eval(max_test_op{}, -1234ll, 1234ll), raft::Compare<long long int>()));
  ASSERT_TRUE(raft::match(
    1234ull, math_eval(max_test_op{}, 1234ull, 123ull), raft::Compare<unsigned long long int>()));

  ASSERT_TRUE(
    raft::match(12.34f, math_eval(max_test_op{}, -12.34f, 12.34f), raft::Compare<float>()));
  ASSERT_TRUE(raft::match(12.34, math_eval(max_test_op{}, -12.34, 12.34), raft::Compare<double>()));
  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, -12.34f, 12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

TEST(MathDevice, Max3)
{
  ASSERT_TRUE(raft::match(1234, math_eval(max_test_op{}, 1234, 0, -1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1234, math_eval(max_test_op{}, -1234, 1234, 0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1234, math_eval(max_test_op{}, 0, -1234, 1234), raft::Compare<int>()));

  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, 12.34f, 0., -12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, -12.34, 12.34f, 0.), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    12.34, math_eval(max_test_op{}, 0., -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

struct min_test_op {
  template <typename... Args>
  constexpr RAFT_INLINE_FUNCTION auto operator()(Args&&... args) const
  {
    return raft::min(std::forward<Args>(args)...);
  }
};

TEST(MathDevice, Min2)
{
  ASSERT_TRUE(raft::match(-1234, math_eval(min_test_op{}, -1234, 1234), raft::Compare<int>()));
  ASSERT_TRUE(
    raft::match(123u, math_eval(min_test_op{}, 1234u, 123u), raft::Compare<unsigned int>()));
  ASSERT_TRUE(raft::match(
    -1234ll, math_eval(min_test_op{}, -1234ll, 1234ll), raft::Compare<long long int>()));
  ASSERT_TRUE(raft::match(
    123ull, math_eval(min_test_op{}, 1234ull, 123ull), raft::Compare<unsigned long long int>()));

  ASSERT_TRUE(
    raft::match(-12.34f, math_eval(min_test_op{}, -12.34f, 12.34f), raft::Compare<float>()));
  ASSERT_TRUE(
    raft::match(-12.34, math_eval(min_test_op{}, -12.34, 12.34), raft::Compare<double>()));
  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, -12.34f, 12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

TEST(MathDevice, Min3)
{
  ASSERT_TRUE(raft::match(-1234, math_eval(min_test_op{}, 1234, 0, -1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1234, math_eval(min_test_op{}, -1234, 1234, 0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1234, math_eval(min_test_op{}, 0, -1234, 1234), raft::Compare<int>()));

  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, 12.34f, 0., -12.34), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, -12.34, 12.34f, 0.), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(
    -12.34, math_eval(min_test_op{}, 0., -12.34, 12.34f), raft::CompareApprox<double>(0.000001)));
}

struct pow_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& x, const Type& y) const
  {
    return raft::pow(x, y);
  }
};

TEST(MathDevice, Pow)
{
  ASSERT_TRUE(raft::match(std::pow(12.34f, 2.f),
                          math_eval(pow_test_op{}, 12.34f, 2.f),
                          raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(std::pow(12.34, 2.),
                          math_eval(pow_test_op{}, 12.34, 2.),
                          raft::CompareApprox<double>(0.000001)));
}

struct sgn_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::sgn(in);
  }
};

TEST(MathDevice, Sgn)
{
  ASSERT_TRUE(raft::match(-1, math_eval(sgn_test_op{}, -1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(0, math_eval(sgn_test_op{}, 0), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1, math_eval(sgn_test_op{}, 1234), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(-1, math_eval(sgn_test_op{}, -12.34f), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(0, math_eval(sgn_test_op{}, 0.f), raft::Compare<int>()));
  ASSERT_TRUE(raft::match(1, math_eval(sgn_test_op{}, 12.34f), raft::Compare<int>()));
}

struct sin_test_op_device {
  template <typename Type>
  constexpr RAFT_DEVICE_INLINE_FUNCTION auto operator()(const Type& in) const
  {
#if (__CUDA_ARCH__ < 530)
    if constexpr (std::is_same_v<Type, __half>) {
      return __float2half(raft::sin(__half2float(in)));
    }
#elif (__CUDA_ARCH__ < 800)
    if constexpr (std::is_same_v<Type, nv_bfloat16>) {
      return __float2bfloat16(raft::sin(__bfloat162float(in)));
    } else  // else is there to make sure raft::sin(in) is not compiled with __half / nv_bfloat16
#endif
    return raft::sin(in);
  }
};

struct sin_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::sin(in);
  }
};

TEST(MathDevice, Sin)
{
  ASSERT_TRUE(raft::match(std::sin(12.34f),
                          __half2float(math_eval(sin_test_op_device{}, __float2half(12.34f))),
                          raft::CompareApprox<float>(0.01f)));
  ASSERT_TRUE(
    raft::match(std::sin(12.34f),
                __bfloat162float(math_eval(sin_test_op_device{}, __float2bfloat16(12.34f))),
                raft::CompareApprox<float>(0.1f)));
  ASSERT_TRUE(raft::match(
    std::sin(12.34f), math_eval(sin_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::sin(12.34), math_eval(sin_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct sincos_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& x, Type* s, Type* c) const
  {
    raft::sincos(x, s, c);
    return x;  // unused, just to avoid creating another helper
  }
};

TEST(MathDevice, SinCos)
{
  auto stream = rmm::cuda_stream_default;
  float xf    = 12.34f;
  rmm::device_scalar<float> sf(stream);
  rmm::device_scalar<float> cf(stream);
  math_eval(sincos_test_op{}, xf, sf.data(), cf.data());
  ASSERT_TRUE(raft::match(std::sin(12.34f), sf.value(stream), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(std::cos(12.34f), cf.value(stream), raft::CompareApprox<float>(0.0001f)));
  double xd = 12.34f;
  rmm::device_scalar<double> sd(stream);
  rmm::device_scalar<double> cd(stream);
  math_eval(sincos_test_op{}, xd, sd.data(), cd.data());
  ASSERT_TRUE(raft::match(std::sin(12.34), sd.value(stream), raft::CompareApprox<double>(0.0001f)));
  ASSERT_TRUE(raft::match(std::cos(12.34), cd.value(stream), raft::CompareApprox<double>(0.0001f)));
}

struct sqrt_test_op_device {
  template <typename Type>
  constexpr RAFT_DEVICE_INLINE_FUNCTION auto operator()(const Type& in) const
  {
#if (__CUDA_ARCH__ < 530)
    if constexpr (std::is_same_v<Type, __half>) {
      return __float2half(raft::sqrt(__half2float(in)));
    }
#elif (__CUDA_ARCH__ < 800)
    if constexpr (std::is_same_v<Type, nv_bfloat16>) {
      return __float2bfloat16(raft::sqrt(__bfloat162float(in)));
    } else  // else is there to make sure raft::sqrt(in) is not compiled with __half / nv_bfloat16
#endif
    return raft::sqrt(in);
  }
};

struct sqrt_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::sqrt(in);
  }
};

TEST(MathDevice, Sqrt)
{
  ASSERT_TRUE(raft::match(std::sqrt(12.34f),
                          __half2float(math_eval(sqrt_test_op_device{}, __float2half(12.34f))),
                          raft::CompareApprox<float>(0.001f)));
  ASSERT_TRUE(
    raft::match(std::sqrt(12.34f),
                __bfloat162float(math_eval(sqrt_test_op_device{}, __float2bfloat16(12.34f))),
                raft::CompareApprox<float>(0.01f)));
  ASSERT_TRUE(raft::match(
    std::sqrt(12.34f), math_eval(sqrt_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::sqrt(12.34), math_eval(sqrt_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}

struct tanh_test_op {
  template <typename Type>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const Type& in) const
  {
    return raft::tanh(in);
  }
};

TEST(MathDevice, Tanh)
{
  ASSERT_TRUE(raft::match(
    std::tanh(12.34f), math_eval(tanh_test_op{}, 12.34f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::tanh(12.34), math_eval(tanh_test_op{}, 12.34), raft::CompareApprox<double>(0.000001)));
}
