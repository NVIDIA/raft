/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_span.hpp>
#include <raft/util/cuda_dev_essentials.cuh>  // DI

#include <cuda_fp16.h>

#include <cstdint>  // uintX_t
#include <type_traits>

namespace raft {

/**
 * @defgroup SmemStores Shared memory store operations
 * @{
 * @brief Stores to shared memory (both vectorized and non-vectorized forms)
 *        requires the given shmem pointer to be aligned by the vector
          length, like for float4 lds/sts shmem pointer should be aligned
          by 16 bytes else it might silently fail or can also give
          runtime error.
 * @param[out] addr shared memory address (should be aligned to vector size)
 * @param[in]  x    data to be stored at this address
 */
DI void sts(uint8_t* addr, const uint8_t& x)
{
  uint32_t x_int;
  x_int   = x;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.u8 [%0], {%1};" : : "l"(s1), "r"(x_int));
}
DI void sts(uint8_t* addr, const uint8_t (&x)[1])
{
  uint32_t x_int[1];
  x_int[0] = x[0];
  auto s1  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.u8 [%0], {%1};" : : "l"(s1), "r"(x_int[0]));
}
DI void sts(uint8_t* addr, const uint8_t (&x)[2])
{
  uint32_t x_int[2];
  x_int[0] = x[0];
  x_int[1] = x[1];
  auto s2  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.v2.u8 [%0], {%1, %2};" : : "l"(s2), "r"(x_int[0]), "r"(x_int[1]));
}
DI void sts(uint8_t* addr, const uint8_t (&x)[4])
{
  uint32_t x_int[4];
  x_int[0] = x[0];
  x_int[1] = x[1];
  x_int[2] = x[2];
  x_int[3] = x[3];
  auto s4  = __cvta_generic_to_shared(reinterpret_cast<uint8_t*>(addr));
  asm volatile("st.shared.v4.u8 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x_int[0]), "r"(x_int[1]), "r"(x_int[2]), "r"(x_int[3]));
}

DI void sts(int8_t* addr, const int8_t& x)
{
  int32_t x_int = x;
  auto s1       = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.s8 [%0], {%1};" : : "l"(s1), "r"(x_int));
}
DI void sts(int8_t* addr, const int8_t (&x)[1])
{
  int32_t x_int[1];
  x_int[0] = x[0];
  auto s1  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.s8 [%0], {%1};" : : "l"(s1), "r"(x_int[0]));
}
DI void sts(int8_t* addr, const int8_t (&x)[2])
{
  int32_t x_int[2];
  x_int[0] = x[0];
  x_int[1] = x[1];
  auto s2  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.v2.s8 [%0], {%1, %2};" : : "l"(s2), "r"(x_int[0]), "r"(x_int[1]));
}
DI void sts(int8_t* addr, const int8_t (&x)[4])
{
  int32_t x_int[4];
  x_int[0] = x[0];
  x_int[1] = x[1];
  x_int[2] = x[2];
  x_int[3] = x[3];
  auto s4  = __cvta_generic_to_shared(reinterpret_cast<int8_t*>(addr));
  asm volatile("st.shared.v4.s8 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x_int[0]), "r"(x_int[1]), "r"(x_int[2]), "r"(x_int[3]));
}

DI void sts(uint32_t* addr, const uint32_t& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x));
}
DI void sts(uint32_t* addr, const uint32_t (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x[0]));
}
DI void sts(uint32_t* addr, const uint32_t (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<uint2*>(addr));
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};" : : "l"(s2), "r"(x[0]), "r"(x[1]));
}
DI void sts(uint32_t* addr, const uint32_t (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<uint4*>(addr));
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));
}

DI void sts(int32_t* addr, const int32_t& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<int32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x));
}
DI void sts(int32_t* addr, const int32_t (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<int32_t*>(addr));
  asm volatile("st.shared.u32 [%0], {%1};" : : "l"(s1), "r"(x[0]));
}
DI void sts(int32_t* addr, const int32_t (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<int2*>(addr));
  asm volatile("st.shared.v2.u32 [%0], {%1, %2};" : : "l"(s2), "r"(x[0]), "r"(x[1]));
}
DI void sts(int32_t* addr, const int32_t (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<int4*>(addr));
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "r"(x[0]), "r"(x[1]), "r"(x[2]), "r"(x[3]));
}

DI void sts(half* addr, const half& x)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint16_t*>(addr));
  asm volatile("st.shared.u16 [%0], {%1};" : : "l"(s), "h"(*reinterpret_cast<const uint16_t*>(&x)));
}
DI void sts(half* addr, const half (&x)[1])
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint16_t*>(addr));
  asm volatile("st.shared.u16 [%0], {%1};" : : "l"(s), "h"(*reinterpret_cast<const uint16_t*>(x)));
}
DI void sts(half* addr, const half (&x)[2])
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint16_t*>(addr));
  asm volatile("st.shared.v2.u16 [%0], {%1, %2};"
               :
               : "l"(s),
                 "h"(*reinterpret_cast<const uint16_t*>(x)),
                 "h"(*reinterpret_cast<const uint16_t*>(x + 1)));
}
DI void sts(half* addr, const half (&x)[4])
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint16_t*>(addr));
  asm volatile("st.shared.v4.u16 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s),
                 "h"(*reinterpret_cast<const uint16_t*>(x)),
                 "h"(*reinterpret_cast<const uint16_t*>(x + 1)),
                 "h"(*reinterpret_cast<const uint16_t*>(x + 2)),
                 "h"(*reinterpret_cast<const uint16_t*>(x + 3)));
}
DI void sts(half* addr, const half (&x)[8])
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<uint32_t*>(addr));
  half2 y[4];
  y[0].x = x[0];
  y[0].y = x[1];
  y[1].x = x[2];
  y[1].y = x[3];
  y[2].x = x[4];
  y[2].y = x[5];
  y[3].x = x[6];
  y[3].y = x[7];
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s),
                 "r"(*reinterpret_cast<uint32_t*>(y)),
                 "r"(*reinterpret_cast<uint32_t*>(y + 1)),
                 "r"(*reinterpret_cast<uint32_t*>(y + 2)),
                 "r"(*reinterpret_cast<uint32_t*>(y + 3)));
}

DI void sts(float* addr, const float& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("st.shared.f32 [%0], {%1};" : : "l"(s1), "f"(x));
}
DI void sts(float* addr, const float (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("st.shared.f32 [%0], {%1};" : : "l"(s1), "f"(x[0]));
}
DI void sts(float* addr, const float (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<float2*>(addr));
  asm volatile("st.shared.v2.f32 [%0], {%1, %2};" : : "l"(s2), "f"(x[0]), "f"(x[1]));
}
DI void sts(float* addr, const float (&x)[4])
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<float4*>(addr));
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};"
               :
               : "l"(s4), "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]));
}

DI void sts(double* addr, const double& x)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("st.shared.f64 [%0], {%1};" : : "l"(s1), "d"(x));
}
DI void sts(double* addr, const double (&x)[1])
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("st.shared.f64 [%0], {%1};" : : "l"(s1), "d"(x[0]));
}
DI void sts(double* addr, const double (&x)[2])
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<double2*>(addr));
  asm volatile("st.shared.v2.f64 [%0], {%1, %2};" : : "l"(s2), "d"(x[0]), "d"(x[1]));
}
/** @} */

/**
 * @defgroup SmemLoads Shared memory load operations
 * @{
 * @brief Loads from shared memory (both vectorized and non-vectorized forms)
          requires the given shmem pointer to be aligned by the vector
          length, like for float4 lds/sts shmem pointer should be aligned
          by 16 bytes else it might silently fail or can also give
          runtime error.
 * @param[out] x    the data to be loaded
 * @param[in]  addr shared memory address from where to load
 *                  (should be aligned to vector size)
 */

DI void lds(uint8_t& x, const uint8_t* addr)
{
  uint32_t x_int;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(x_int) : "l"(s1));
  x = x_int;
}
DI void lds(uint8_t (&x)[1], const uint8_t* addr)
{
  uint32_t x_int[1];
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.u8 {%0}, [%1];" : "=r"(x_int[0]) : "l"(s1));
  x[0] = x_int[0];
}
DI void lds(uint8_t (&x)[2], const uint8_t* addr)
{
  uint32_t x_int[2];
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.v2.u8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(s2));
  x[0] = x_int[0];
  x[1] = x_int[1];
}
DI void lds(uint8_t (&x)[4], const uint8_t* addr)
{
  uint32_t x_int[4];
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const uint8_t*>(addr));
  asm volatile("ld.shared.v4.u8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(s4));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

DI void lds(int8_t& x, const int8_t* addr)
{
  int32_t x_int;
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.s8 {%0}, [%1];" : "=r"(x_int) : "l"(s1));
  x = x_int;
}
DI void lds(int8_t (&x)[1], const int8_t* addr)
{
  int32_t x_int[1];
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.s8 {%0}, [%1];" : "=r"(x_int[0]) : "l"(s1));
  x[0] = x_int[0];
}
DI void lds(int8_t (&x)[2], const int8_t* addr)
{
  int32_t x_int[2];
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.v2.s8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(s2));
  x[0] = x_int[0];
  x[1] = x_int[1];
}
DI void lds(int8_t (&x)[4], const int8_t* addr)
{
  int32_t x_int[4];
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const int8_t*>(addr));
  asm volatile("ld.shared.v4.s8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(s4));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

DI void lds(uint32_t (&x)[4], const uint32_t* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(s4));
}

DI void lds(uint32_t (&x)[2], const uint32_t* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(s2));
}

DI void lds(uint32_t (&x)[1], const uint32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x[0]) : "l"(s1));
}

DI void lds(uint32_t& x, const uint32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "l"(s1));
}

DI void lds(int32_t (&x)[4], const int32_t* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(s4));
}

DI void lds(int32_t (&x)[2], const int32_t* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(s2));
}

DI void lds(int32_t (&x)[1], const int32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x[0]) : "l"(s1));
}

DI void lds(int32_t& x, const int32_t* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const int32_t*>(addr));
  asm volatile("ld.shared.u32 {%0}, [%1];" : "=r"(x) : "l"(s1));
}

DI void lds(half& x, const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint16_t*>(addr));
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(*reinterpret_cast<uint16_t*>(&x)) : "l"(s));
}
DI void lds(half (&x)[1], const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint16_t*>(addr));
  asm volatile("ld.shared.u16 {%0}, [%1];" : "=h"(*reinterpret_cast<uint16_t*>(x)) : "l"(s));
}
DI void lds(half (&x)[2], const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint16_t*>(addr));
  asm volatile("ld.shared.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "l"(s));
}
DI void lds(half (&x)[4], const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint16_t*>(addr));
  asm volatile("ld.shared.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "l"(s));
}
DI void lds(half (&x)[8], const half* addr)
{
  auto s = __cvta_generic_to_shared(reinterpret_cast<const uint32_t*>(addr));
  half2 y[4];
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(y)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 1)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 2)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 3))
               : "l"(s));
  x[0] = y[0].x;
  x[1] = y[0].y;
  x[2] = y[1].x;
  x[3] = y[1].y;
  x[4] = y[2].x;
  x[5] = y[2].y;
  x[6] = y[3].x;
  x[7] = y[3].y;
}
DI void lds(float& x, const float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "l"(s1));
}
DI void lds(float (&x)[1], const float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<const float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x[0]) : "l"(s1));
}
DI void lds(float (&x)[2], const float* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<const float2*>(addr));
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(s2));
}
DI void lds(float (&x)[4], const float* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<const float4*>(addr));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(s4));
}

DI void lds(float& x, float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x) : "l"(s1));
}
DI void lds(float (&x)[1], float* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<float*>(addr));
  asm volatile("ld.shared.f32 {%0}, [%1];" : "=f"(x[0]) : "l"(s1));
}
DI void lds(float (&x)[2], float* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<float2*>(addr));
  asm volatile("ld.shared.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(s2));
}
DI void lds(float (&x)[4], float* addr)
{
  auto s4 = __cvta_generic_to_shared(reinterpret_cast<float4*>(addr));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(s4));
}
DI void lds(double& x, double* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("ld.shared.f64 {%0}, [%1];" : "=d"(x) : "l"(s1));
}
DI void lds(double (&x)[1], double* addr)
{
  auto s1 = __cvta_generic_to_shared(reinterpret_cast<double*>(addr));
  asm volatile("ld.shared.f64 {%0}, [%1];" : "=d"(x[0]) : "l"(s1));
}
DI void lds(double (&x)[2], double* addr)
{
  auto s2 = __cvta_generic_to_shared(reinterpret_cast<double2*>(addr));
  asm volatile("ld.shared.v2.f64 {%0, %1}, [%2];" : "=d"(x[0]), "=d"(x[1]) : "l"(s2));
}
/** @} */

/**
 * @defgroup GlobalLoads Global cached load operations
 * @{
 * @brief Load from global memory with caching at L1 level
 * @param[out] x    data to be loaded from global memory
 * @param[in]  addr address in global memory from where to load
 */
DI void ldg(float& x, const float* addr)
{
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(addr));
}
DI void ldg(float (&x)[1], const float* addr)
{
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x[0]) : "l"(addr));
}
DI void ldg(float (&x)[2], const float* addr)
{
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];" : "=f"(x[0]), "=f"(x[1]) : "l"(addr));
}
DI void ldg(float (&x)[4], const float* addr)
{
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(addr));
}
DI void ldg(half& x, const half* addr)
{
  asm volatile("ld.global.cg.u16 {%0}, [%1];"
               : "=h"(*reinterpret_cast<uint16_t*>(&x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
DI void ldg(half (&x)[1], const half* addr)
{
  asm volatile("ld.global.cg.u16 {%0}, [%1];"
               : "=h"(*reinterpret_cast<uint16_t*>(x))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
DI void ldg(half (&x)[2], const half* addr)
{
  asm volatile("ld.global.cg.v2.u16 {%0, %1}, [%2];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)), "=h"(*reinterpret_cast<uint16_t*>(x + 1))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}
DI void ldg(half (&x)[4], const half* addr)
{
  asm volatile("ld.global.cg.v4.u16 {%0, %1, %2, %3}, [%4];"
               : "=h"(*reinterpret_cast<uint16_t*>(x)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 1)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 2)),
                 "=h"(*reinterpret_cast<uint16_t*>(x + 3))
               : "l"(reinterpret_cast<const uint16_t*>(addr)));
}

DI void ldg(half (&x)[8], const half* addr)
{
  half2 y[4];
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(*reinterpret_cast<uint32_t*>(y)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 1)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 2)),
                 "=r"(*reinterpret_cast<uint32_t*>(y + 3))
               : "l"(reinterpret_cast<const uint32_t*>(addr)));
  x[0] = y[0].x;
  x[1] = y[0].y;
  x[2] = y[1].x;
  x[3] = y[1].y;
  x[4] = y[2].x;
  x[5] = y[2].y;
  x[6] = y[3].x;
  x[7] = y[3].y;
}
DI void ldg(double& x, const double* addr)
{
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x) : "l"(addr));
}
DI void ldg(double (&x)[1], const double* addr)
{
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x[0]) : "l"(addr));
}
DI void ldg(double (&x)[2], const double* addr)
{
  asm volatile("ld.global.cg.v2.f64 {%0, %1}, [%2];" : "=d"(x[0]), "=d"(x[1]) : "l"(addr));
}

DI void ldg(uint32_t (&x)[4], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(addr));
}

DI void ldg(uint32_t (&x)[2], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(addr));
}

DI void ldg(uint32_t (&x)[1], const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x[0]) : "l"(addr));
}

DI void ldg(uint32_t& x, const uint32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

DI void ldg(int32_t (&x)[4], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x[0]), "=r"(x[1]), "=r"(x[2]), "=r"(x[3])
               : "l"(addr));
}

DI void ldg(int32_t (&x)[2], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.v2.u32 {%0, %1}, [%2];" : "=r"(x[0]), "=r"(x[1]) : "l"(addr));
}

DI void ldg(int32_t (&x)[1], const int32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x[0]) : "l"(addr));
}

DI void ldg(int32_t& x, const int32_t* const& addr)
{
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(x) : "l"(addr));
}

DI void ldg(uint8_t (&x)[4], const uint8_t* const& addr)
{
  uint32_t x_int[4];
  asm volatile("ld.global.cg.v4.u8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

DI void ldg(uint8_t (&x)[2], const uint8_t* const& addr)
{
  uint32_t x_int[2];
  asm volatile("ld.global.cg.v2.u8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
}

DI void ldg(uint8_t (&x)[1], const uint8_t* const& addr)
{
  uint32_t x_int;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x[0] = x_int;
}

DI void ldg(uint8_t& x, const uint8_t* const& addr)
{
  uint32_t x_int;
  asm volatile("ld.global.cg.u8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x = x_int;
}

DI void ldg(int8_t (&x)[4], const int8_t* const& addr)
{
  int x_int[4];
  asm volatile("ld.global.cg.v4.s8 {%0, %1, %2, %3}, [%4];"
               : "=r"(x_int[0]), "=r"(x_int[1]), "=r"(x_int[2]), "=r"(x_int[3])
               : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
  x[2] = x_int[2];
  x[3] = x_int[3];
}

DI void ldg(int8_t (&x)[2], const int8_t* const& addr)
{
  int x_int[2];
  asm volatile("ld.global.cg.v2.s8 {%0, %1}, [%2];" : "=r"(x_int[0]), "=r"(x_int[1]) : "l"(addr));
  x[0] = x_int[0];
  x[1] = x_int[1];
}

DI void ldg(int8_t& x, const int8_t* const& addr)
{
  int x_int;
  asm volatile("ld.global.cg.s8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x = x_int;
}

DI void ldg(int8_t (&x)[1], const int8_t* const& addr)
{
  int x_int;
  asm volatile("ld.global.cg.s8 %0, [%1];" : "=r"(x_int) : "l"(addr));
  x[0] = x_int;
}

/**
 * @brief Executes a 1D block strided copy
 * @param dst destination pointer
 * @param src source pointer
 * @param size number of items to copy
 */
template <typename T>
DI void block_copy(T* dst, const T* src, const size_t size)
{
  for (auto i = threadIdx.x; i < size; i += blockDim.x) {
    dst[i] = src[i];
  }
}

/**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 * @param size number of items to copy
 */
template <typename T>
DI void block_copy(raft::device_span<T> dst,
                   const raft::device_span<const T> src,
                   const size_t size)
{
  assert(src.size() >= size);
  assert(dst.size() >= size);
  block_copy(dst.data(), src.data(), size);
}

/**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 * @param size number of items to copy
 */
template <typename T>
DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src, const size_t size)
{
  assert(src.size() >= size);
  assert(dst.size() >= size);
  block_copy(dst.data(), src.data(), size);
}

/**
 * @brief Executes a 1D block strided copy
 * @param dst span of destination pointer
 * @param src span of source pointer
 */
template <typename T>
DI void block_copy(raft::device_span<T> dst, const raft::device_span<T> src)
{
  assert(dst.size() >= src.size());
  block_copy(dst, src, src.size());
}

/** @} */

/**
 * @defgroup GlobalStores Global Store Operations
 * @{
 * @brief Perform conditional stores to global memory.
 *
 * These functions store data to a specified global memory address,
 * controlled by a guard flag to enable conditional execution.
 *
 * @param[in] reg   The data to store in global memory.
 *                  The type of `reg` determines the size of the store.
 * @param[in] addr  The global memory address where the data will be stored.
 * @param[in] guard A flag to conditionally enable the store operation.
 *                  If `true`, the store is performed; otherwise, it is skipped
 */
DI void stg(const int& reg, void* addr, bool guard)
{
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %2, 0;\n"
    "@p st.global.b32 [%0], %1;\n"
    "}\n"
    :
    : "l"(addr), "r"(reg), "r"((int)guard)
    : "memory");
}

DI void stg(const int64_t& reg, void* addr, bool guard)
{
  asm volatile(
    "{\n"
    ".reg .pred p;\n"
    "setp.ne.b32 p, %2, 0;\n"
    "@p st.global.b64 [%0], %1;\n"
    "}\n"
    :
    : "l"(addr), "l"(reg), "r"((int)guard)
    : "memory");
}

/** @} */

/**
 * @defgroup PlainWordAccess Single-instruction word loads/stores
 * @{
 * @brief Pin a 4- or 8-byte access to one load/store instruction with the
 *        default cache policy (.ca loads / .wb stores), keeping L1.
 *
 * For intentionally racy algorithms (e.g. concurrent union-find): a single
 * instruction cannot be torn or fused by the compiler, and uniform-size data
 * races are defined under the PTX memory model (ISA 8.7.2: a racing scalar
 * read returns the value committed by some write). Unlike `ldg`
 * (ld.global.cg) or atomics, these do not bypass the non-coherent L1.
 */
template <typename T, typename = std::enable_if_t<sizeof(T) == 4 || sizeof(T) == 8>>
DI T ldg_ca(const T* addr)
{
  if constexpr (sizeof(T) == 4) {
    uint32_t r;
    asm volatile("ld.b32 %0, [%1];" : "=r"(r) : "l"(addr) : "memory");
    return static_cast<T>(r);
  } else {
    uint64_t r;
    asm volatile("ld.b64 %0, [%1];" : "=l"(r) : "l"(addr) : "memory");
    return static_cast<T>(r);
  }
}

template <typename T, typename = std::enable_if_t<sizeof(T) == 4 || sizeof(T) == 8>>
DI void stg_wb(T* addr, T val)
{
  if constexpr (sizeof(T) == 4) {
    asm volatile("st.b32 [%0], %1;" ::"l"(addr), "r"(static_cast<uint32_t>(val)) : "memory");
  } else {
    asm volatile("st.b64 [%0], %1;" ::"l"(addr), "l"(static_cast<uint64_t>(val)) : "memory");
  }
}

/** @} */

}  // namespace raft
