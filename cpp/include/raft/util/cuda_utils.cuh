/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/math.hpp>
#include <raft/core/operators.hpp>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <stdint.h>

#include <type_traits>
// For backward compatibility, we include the follow headers. They contain
// functionality that were previously contained in cuda_utils.cuh
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/reduction.cuh>

namespace raft {

/** Device function to have atomic add support for older archs */
template <typename Type>
DI void myAtomicAdd(Type* address, Type val)
{
  atomicAdd(address, val);
}

// Specialization for int64_t (signed long long)
template <>
DI void myAtomicAdd(int64_t* address, int64_t val)
{
  // Cast to unsigned long long for atomic operation
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, (unsigned long long int)(((int64_t)assumed) + val));
  } while (assumed != old);
}

// Specialization for float4
template <>
DI void myAtomicAdd(float4* address, float4 val)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // Vectorized atomic write is only available from Hopper.
  atomicAdd(address, val);
#else
  float* elements = reinterpret_cast<float*>(address);
  myAtomicAdd(elements, val.x);
  myAtomicAdd(elements + 1, val.y);
  myAtomicAdd(elements + 2, val.z);
  myAtomicAdd(elements + 3, val.w);
#endif
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
// Ref:
// http://on-demand.gputechconf.com/gtc/2013/presentations/S3101-Atomic-Memory-Operations.pdf
template <>
DI void myAtomicAdd(double* address, double val)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;
  do {
    assumed = old;
    old =
      atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
}
#endif

template <typename T, typename ReduceLambda>
DI void myAtomicReduce(T* address, T val, ReduceLambda op);

template <typename ReduceLambda>
DI void myAtomicReduce(double* address, double val, ReduceLambda op)
{
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old             = *address_as_ull, assumed;
  do {
    assumed = old;
    old     = atomicCAS(
      address_as_ull, assumed, __double_as_longlong(op(val, __longlong_as_double(assumed))));
  } while (assumed != old);
}

template <typename ReduceLambda>
DI void myAtomicReduce(float* address, float val, ReduceLambda op)
{
  unsigned int* address_as_uint = (unsigned int*)address;
  unsigned int old              = *address_as_uint, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed, __float_as_uint(op(val, __uint_as_float(assumed))));
  } while (assumed != old);
}

// Needed for atomicCas on ushort
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
template <typename ReduceLambda>
DI void myAtomicReduce(__half* address, __half val, ReduceLambda op)
{
  unsigned short int* address_as_uint = (unsigned short int*)address;
  unsigned short int old              = *address_as_uint, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_uint, assumed, __half_as_ushort(op(val, __ushort_as_half(assumed))));
  } while (assumed != old);
}
#endif

// Needed for nv_bfloat16 support
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
template <typename ReduceLambda>
DI void myAtomicReduce(nv_bfloat16* address, nv_bfloat16 val, ReduceLambda op)
{
  unsigned short int* address_as_uint = (unsigned short int*)address;
  unsigned short int old              = *address_as_uint, assumed;
  do {
    assumed = old;
    old     = atomicCAS(
      address_as_uint, assumed, __bfloat16_as_ushort(op(val, __ushort_as_bfloat16(assumed))));
  } while (assumed != old);
}
#endif

template <typename ReduceLambda>
DI void myAtomicReduce(int* address, int val, ReduceLambda op)
{
  int old = *address, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address, assumed, op(val, assumed));
  } while (assumed != old);
}

template <typename ReduceLambda>
DI void myAtomicReduce(long long* address, long long val, ReduceLambda op)
{
  long long old = *address, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address, assumed, op(val, assumed));
  } while (assumed != old);
}

template <typename ReduceLambda>
DI void myAtomicReduce(unsigned long long* address, unsigned long long val, ReduceLambda op)
{
  unsigned long long old = *address, assumed;
  do {
    assumed = old;
    old     = atomicCAS(address, assumed, op(val, assumed));
  } while (assumed != old);
}

/**
 * @brief Provide atomic min operation.
 * @tparam T: data type for input data (float or double).
 * @param[in] address: address to read old value from, and to atomically update w/ min(old value,
 * val)
 * @param[in] val: new value to compare with old
 */
template <typename T>
DI T myAtomicMin(T* address, T val);

/**
 * @brief Provide atomic max operation.
 * @tparam T: data type for input data (float or double).
 * @param[in] address: address to read old value from, and to atomically update w/ max(old value,
 * val)
 * @param[in] val: new value to compare with old
 */
template <typename T>
DI T myAtomicMax(T* address, T val);

DI float myAtomicMin(float* address, float val)
{
  myAtomicReduce<float(float, float)>(address, val, fminf);
  return *address;
}

DI float myAtomicMax(float* address, float val)
{
  myAtomicReduce<float(float, float)>(address, val, fmaxf);
  return *address;
}

DI double myAtomicMin(double* address, double val)
{
  myAtomicReduce<double(double, double)>(address, val, fmin);
  return *address;
}

DI double myAtomicMax(double* address, double val)
{
  myAtomicReduce<double(double, double)>(address, val, fmax);
  return *address;
}

/**
 * @defgroup Max maximum of two numbers
 * @{
 */
template <typename T>
HDI T myMax(T x, T y);
template <>
[[deprecated("use raft::max from raft/core/math.hpp instead")]] HDI float myMax<float>(float x,
                                                                                       float y)
{
  return fmaxf(x, y);
}
template <>
[[deprecated("use raft::max from raft/core/math.hpp instead")]] HDI double myMax<double>(double x,
                                                                                         double y)
{
  return fmax(x, y);
}
/** @} */

/**
 * @defgroup Min minimum of two numbers
 * @{
 */
template <typename T>
HDI T myMin(T x, T y);
template <>
[[deprecated("use raft::min from raft/core/math.hpp instead")]] HDI float myMin<float>(float x,
                                                                                       float y)
{
  return fminf(x, y);
}
template <>
[[deprecated("use raft::min from raft/core/math.hpp instead")]] HDI double myMin<double>(double x,
                                                                                         double y)
{
  return fmin(x, y);
}
/** @} */

/**
 * @brief Provide atomic min operation.
 * @tparam T: data type for input data (float or double).
 * @param[in] address: address to read old value from, and to atomically update w/ min(old value,
 * val)
 * @param[in] val: new value to compare with old
 */
template <typename T>
DI T myAtomicMin(T* address, T val)
{
  myAtomicReduce(address, val, raft::min_op{});
  return *address;
}

/**
 * @brief Provide atomic max operation.
 * @tparam T: data type for input data (float or double).
 * @param[in] address: address to read old value from, and to atomically update w/ max(old value,
 * val)
 * @param[in] val: new value to compare with old
 */
template <typename T>
DI T myAtomicMax(T* address, T val)
{
  myAtomicReduce(address, val, raft::max_op{});
  return *address;
}

/**
 * @defgroup Exp Exponential function
 * @{
 */
template <typename T>
HDI T myExp(T x);
template <>
[[deprecated("use raft::exp from raft/core/math.hpp instead")]] HDI float myExp(float x)
{
  return expf(x);
}
template <>
[[deprecated("use raft::exp from raft/core/math.hpp instead")]] HDI double myExp(double x)
{
  return ::exp(x);
}
/** @} */

/**
 * @defgroup Cuda infinity values
 * @{
 */
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename std::enable_if_t<std::is_same_v<T, float>, float> myInf()
{
  return CUDART_INF_F;
}
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename std::enable_if_t<std::is_same_v<T, double>, double> myInf()
{
  return CUDART_INF;
}
// Half/Bfloat constants only defined after CUDA 12.2
#if __CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 2)
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename std::enable_if_t<std::is_same_v<T, __half>, __half> myInf()
{
#if (__CUDA_ARCH__ >= 530)
  return __ushort_as_half((unsigned short)0x7C00U);
#else
  // Fail during template instantiation if the compute capability doesn't support this operation
  static_assert(sizeof(T) != sizeof(T), "__half is only supported on __CUDA_ARCH__ >= 530");
  return T{};
#endif
}
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename std::enable_if_t<std::is_same_v<T, nv_bfloat16>, nv_bfloat16>
myInf()
{
#if (__CUDA_ARCH__ >= 800)
  return __ushort_as_bfloat16((unsigned short)0x7F80U);
#else
  // Fail during template instantiation if the compute capability doesn't support this operation
  static_assert(sizeof(T) != sizeof(T), "nv_bfloat16 is only supported on __CUDA_ARCH__ >= 800");
  return T{};
#endif
}
#else
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename std::enable_if_t<std::is_same_v<T, __half>, __half> myInf()
{
  return CUDART_INF_FP16;
}
template <typename T>
RAFT_DEVICE_INLINE_FUNCTION typename std::enable_if_t<std::is_same_v<T, nv_bfloat16>, nv_bfloat16>
myInf()
{
  return CUDART_INF_BF16;
}
#endif
/** @} */

/**
 * @defgroup Log Natural logarithm
 * @{
 */
template <typename T>
HDI T myLog(T x);
template <>
[[deprecated("use raft::log from raft/core/math.hpp instead")]] HDI float myLog(float x)
{
  return logf(x);
}
template <>
[[deprecated("use raft::log from raft/core/math.hpp instead")]] HDI double myLog(double x)
{
  return ::log(x);
}
/** @} */

/**
 * @defgroup Sqrt Square root
 * @{
 */
template <typename T>
HDI T mySqrt(T x);
template <>
[[deprecated("use raft::sqrt from raft/core/math.hpp instead")]] HDI float mySqrt(float x)
{
  return sqrtf(x);
}
template <>
[[deprecated("use raft::sqrt from raft/core/math.hpp instead")]] HDI double mySqrt(double x)
{
  return ::sqrt(x);
}
/** @} */

/**
 * @defgroup SineCosine Sine and cosine calculation
 * @{
 */
template <typename T>
DI void mySinCos(T x, T& s, T& c);
template <>
[[deprecated("use raft::sincos from raft/core/math.hpp instead")]] DI void mySinCos(float x,
                                                                                    float& s,
                                                                                    float& c)
{
  sincosf(x, &s, &c);
}
template <>
[[deprecated("use raft::sincos from raft/core/math.hpp instead")]] DI void mySinCos(double x,
                                                                                    double& s,
                                                                                    double& c)
{
  ::sincos(x, &s, &c);
}
/** @} */

/**
 * @defgroup Sine Sine calculation
 * @{
 */
template <typename T>
DI T mySin(T x);
template <>
[[deprecated("use raft::sin from raft/core/math.hpp instead")]] DI float mySin(float x)
{
  return sinf(x);
}
template <>
[[deprecated("use raft::sin from raft/core/math.hpp instead")]] DI double mySin(double x)
{
  return ::sin(x);
}
/** @} */

/**
 * @defgroup Abs Absolute value
 * @{
 */
template <typename T>
DI T myAbs(T x)
{
  return x < 0 ? -x : x;
}
template <>
[[deprecated("use raft::abs from raft/core/math.hpp instead")]] DI float myAbs(float x)
{
  return fabsf(x);
}
template <>
[[deprecated("use raft::abs from raft/core/math.hpp instead")]] DI double myAbs(double x)
{
  return fabs(x);
}
/** @} */

/**
 * @defgroup Pow Power function
 * @{
 */
template <typename T>
HDI T myPow(T x, T power);
template <>
[[deprecated("use raft::pow from raft/core/math.hpp instead")]] HDI float myPow(float x,
                                                                                float power)
{
  return powf(x, power);
}
template <>
[[deprecated("use raft::pow from raft/core/math.hpp instead")]] HDI double myPow(double x,
                                                                                 double power)
{
  return ::pow(x, power);
}
/** @} */

/**
 * @defgroup myTanh tanh function
 * @{
 */
template <typename T>
HDI T myTanh(T x);
template <>
[[deprecated("use raft::tanh from raft/core/math.hpp instead")]] HDI float myTanh(float x)
{
  return tanhf(x);
}
template <>
[[deprecated("use raft::tanh from raft/core/math.hpp instead")]] HDI double myTanh(double x)
{
  return ::tanh(x);
}
/** @} */

/**
 * @defgroup myATanh arctanh function
 * @{
 */
template <typename T>
HDI T myATanh(T x);
template <>
[[deprecated("use raft::atanh from raft/core/math.hpp instead")]] HDI float myATanh(float x)
{
  return atanhf(x);
}
template <>
[[deprecated("use raft::atanh from raft/core/math.hpp instead")]] HDI double myATanh(double x)
{
  return ::atanh(x);
}
/** @} */

/**
 * @defgroup LambdaOps Legacy lambda operations, to be deprecated
 * @{
 */
template <typename Type, typename IdxType = int>
struct Nop {
  [[deprecated("Nop is deprecated. Use identity_op instead.")]] HDI Type
  operator()(Type in, IdxType i = 0) const
  {
    return in;
  }
};

template <typename Type, typename IdxType = int>
struct SqrtOp {
  [[deprecated("SqrtOp is deprecated. Use sqrt_op instead.")]] HDI Type
  operator()(Type in, IdxType i = 0) const
  {
    return raft::sqrt(in);
  }
};

template <typename Type, typename IdxType = int>
struct L0Op {
  [[deprecated("L0Op is deprecated. Use nz_op instead.")]] HDI Type operator()(Type in,
                                                                               IdxType i = 0) const
  {
    return in != Type(0) ? Type(1) : Type(0);
  }
};

template <typename Type, typename IdxType = int>
struct L1Op {
  [[deprecated("L1Op is deprecated. Use abs_op instead.")]] HDI Type operator()(Type in,
                                                                                IdxType i = 0) const
  {
    return raft::abs(in);
  }
};

template <typename Type, typename IdxType = int>
struct L2Op {
  [[deprecated("L2Op is deprecated. Use sq_op instead.")]] HDI Type operator()(Type in,
                                                                               IdxType i = 0) const
  {
    return in * in;
  }
};

template <typename InT, typename OutT = InT>
struct Sum {
  [[deprecated("Sum is deprecated. Use add_op instead.")]] HDI OutT operator()(InT a, InT b) const
  {
    return a + b;
  }
};

template <typename Type>
struct Max {
  [[deprecated("Max is deprecated. Use max_op instead.")]] HDI Type operator()(Type a, Type b) const
  {
    if (b > a) { return b; }
    return a;
  }
};
/** @} */

/**
 * @defgroup Sign Obtain sign value
 * @brief Obtain sign of x
 * @param x input
 * @return +1 if x >= 0 and -1 otherwise
 * @{
 */
template <typename T>
DI T signPrim(T x)
{
  return x < 0 ? -1 : +1;
}
template <>
DI float signPrim(float x)
{
  return signbit(x) == true ? -1.0f : +1.0f;
}
template <>
DI double signPrim(double x)
{
  return signbit(x) == true ? -1.0 : +1.0;
}
/** @} */

/**
 * @defgroup Max maximum of two numbers
 * @brief Obtain maximum of two values
 * @param x one item
 * @param y second item
 * @return maximum of two items
 * @{
 */
template <typename T>
DI T maxPrim(T x, T y)
{
  return x > y ? x : y;
}
template <>
DI float maxPrim(float x, float y)
{
  return fmaxf(x, y);
}
template <>
DI double maxPrim(double x, double y)
{
  return fmax(x, y);
}
/** @} */

/**
 * @brief Four-way byte dot product-accumulate.
 * @tparam T Four-byte integer: int or unsigned int
 * @tparam S Either same as T or a 4-byte vector of the same signedness.
 *
 * @param a
 * @param b
 * @param c
 * @return dot(a, b) + c
 */
template <typename T, typename S = T>
DI auto dp4a(S a, S b, T c) -> T;

template <>
DI auto dp4a(char4 a, char4 b, int c) -> int
{
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  c += static_cast<int>(a.x) * static_cast<int>(b.x);
  c += static_cast<int>(a.y) * static_cast<int>(b.y);
  c += static_cast<int>(a.z) * static_cast<int>(b.z);
  c += static_cast<int>(a.w) * static_cast<int>(b.w);
  return c;
#endif
}

template <>
DI auto dp4a(uchar4 a, uchar4 b, unsigned int c) -> unsigned int
{
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  c += static_cast<unsigned int>(a.x) * static_cast<unsigned int>(b.x);
  c += static_cast<unsigned int>(a.y) * static_cast<unsigned int>(b.y);
  c += static_cast<unsigned int>(a.z) * static_cast<unsigned int>(b.z);
  c += static_cast<unsigned int>(a.w) * static_cast<unsigned int>(b.w);
  return c;
#endif
}

template <>
DI auto dp4a(int a, int b, int c) -> int
{
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  return dp4a(*reinterpret_cast<char4*>(&a), *reinterpret_cast<char4*>(&b), c);
#endif
}

template <>
DI auto dp4a(unsigned int a, unsigned int b, unsigned int c) -> unsigned int
{
#if __CUDA_ARCH__ >= 610
  return __dp4a(a, b, c);
#else
  return dp4a(*reinterpret_cast<uchar4*>(&a), *reinterpret_cast<uchar4*>(&b), c);
#endif
}

/**
 * @brief Simple utility function to determine whether user_stream or one of the
 * internal streams should be used.
 * @param user_stream main user stream
 * @param int_streams array of internal streams
 * @param n_int_streams number of internal streams
 * @param idx the index for which to query the stream
 */
inline cudaStream_t select_stream(cudaStream_t user_stream,
                                  cudaStream_t* int_streams,
                                  int n_int_streams,
                                  int idx)
{
  return n_int_streams > 0 ? int_streams[idx % n_int_streams] : user_stream;
}

}  // namespace raft
