/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_dev_essentials.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/warp_primitives.cuh>

#include <stdint.h>

namespace raft {

/**
 * @brief Logical-warp-level reduction
 * @tparam logicalWarpSize Logical warp size (2, 4, 8, 16 or 32)
 * @tparam T Value type to be reduced
 * @tparam ReduceLambda Reduction operation type
 * @param val input value
 * @param reduce_op Reduction operation
 * @return Reduction result. All lanes will have the valid result.
 */
template <int logicalWarpSize, typename T, typename ReduceLambda>
DI T logicalWarpReduce(T val, ReduceLambda reduce_op)
{
#pragma unroll
  for (int i = logicalWarpSize / 2; i > 0; i >>= 1) {
    const T tmp = shfl_xor(val, i, logicalWarpSize);
    val         = reduce_op(val, tmp);
  }
  return val;
}

/**
 * @brief Warp-level reduction
 * @tparam T Value type to be reduced
 * @tparam ReduceLambda Reduction operation type
 * @param val input value
 * @param reduce_op Reduction operation
 * @return Reduction result. All lanes will have the valid result.
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block. All threads in the warp must enter this
 *       function together
 */
template <typename T, typename ReduceLambda>
DI T warpReduce(T val, ReduceLambda reduce_op)
{
  return logicalWarpReduce<WarpSize>(val, reduce_op);
}

/**
 * @brief Warp-level reduction with raft::add_op
 * @tparam T Value type to be reduced
 * @param val input value
 * @param reduce_op raft::add_op instance
 * @return Reduction result. All lanes will have the valid result.
 * @note Explicit overload to disambiguate from cub::detail::scan::warpReduce when
 *       called with raft::add_op; avoids ambiguous template instantiation on CUDA 13.2+
 */
template <typename T>
DI T warpReduce(T val, raft::add_op reduce_op)
{
  return logicalWarpReduce<WarpSize>(val, reduce_op);
}

/**
 * @brief Warp-level reduction
 * @tparam T Value type to be reduced
 * @param val input value
 * @return Reduction result. All lanes will have the valid result.
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block. All threads in the warp must enter this
 *       function together
 */
template <typename T>
DI T warpReduce(T val)
{
  return warpReduce(val, raft::add_op{});
}

/**
 * @brief 1-D block-level reduction
 * @param val input value
 * @param smem shared memory region needed for storing intermediate results. It
 *             must alteast be of size: `sizeof(T) * nWarps`
 * @param reduce_op a binary reduction operation.
 * @return only the thread0 will contain valid reduced result
 * @note Why not cub? Because cub doesn't seem to allow working with arbitrary
 *       number of warps in a block. All threads in the block must enter this
 *       function together. cub also uses too many registers
 */
template <typename T, typename ReduceLambda = raft::add_op>
DI T blockReduce(T val, char* smem, ReduceLambda reduce_op = raft::add_op{})
{
  auto* sTemp = reinterpret_cast<T*>(smem);
  int nWarps  = (blockDim.x + WarpSize - 1) / WarpSize;
  int lid     = laneId();
  int wid     = threadIdx.x / WarpSize;
  val         = warpReduce(val, reduce_op);
  if (lid == 0) sTemp[wid] = val;
  __syncthreads();
  val = lid < nWarps ? sTemp[lid] : T();
  return warpReduce(val, reduce_op);
}

/**
 * @brief 1-D warp-level ranked reduction which returns the value and rank.
 * thread 0 will have valid result and rank(idx).
 * @param val input value
 * @param idx index to be used as rank
 * @param reduce_op a binary reduction operation.
 */
template <typename T, typename ReduceLambda, typename i_t = int>
DI void warpRankedReduce(T& val, i_t& idx, ReduceLambda reduce_op = raft::min_op{})
{
#pragma unroll
  for (i_t offset = WarpSize / 2; offset > 0; offset /= 2) {
    T tmpVal   = shfl(val, laneId() + offset);
    i_t tmpIdx = shfl(idx, laneId() + offset);
    if (reduce_op(tmpVal, val) == tmpVal) {
      val = tmpVal;
      idx = tmpIdx;
    }
  }
}

/**
 * @brief 1-D block-level ranked reduction which returns the value and rank.
 * thread 0 will have valid result and rank(idx).
 * @param val input value
 * @param shbuf shared memory region needed for storing intermediate results. It
 *             must alteast be of size: `(sizeof(T) + sizeof(i_t)) * WarpSize`
 * @param idx index to be used as rank
 * @param reduce_op binary min or max operation.
 * @return only the thread0 will contain valid reduced result
 */
template <typename T, typename ReduceLambda, typename i_t = int>
DI std::pair<T, i_t> blockRankedReduce(T val,
                                       T* shbuf,
                                       i_t idx                = threadIdx.x,
                                       ReduceLambda reduce_op = raft::min_op{})
{
  T* values    = shbuf;
  i_t* indices = (i_t*)&shbuf[WarpSize];
  i_t wid      = threadIdx.x / WarpSize;
  i_t nWarps   = (blockDim.x + WarpSize - 1) / WarpSize;
  warpRankedReduce(val, idx, reduce_op);  // Each warp performs partial reduction
  i_t lane = laneId();
  if (lane == 0) {
    values[wid]  = val;  // Write reduced value to shared memory
    indices[wid] = idx;  // Write reduced value to shared memory
  }

  __syncthreads();  // Wait for all partial reductions

  // read from shared memory only if that warp existed
  if (lane < nWarps) {
    val = values[lane];
    idx = indices[lane];
  } else {
    // get the lower_bound of the type if it is a max op,
    // get the upper bound of the type if it is a min op
    val = reduce_op(lower_bound<T>(), upper_bound<T>()) == lower_bound<T>() ? upper_bound<T>()
                                                                            : lower_bound<T>();
    idx = -1;
  }
  __syncthreads();
  if (wid == 0) warpRankedReduce(val, idx, reduce_op);
  return std::pair<T, i_t>{val, idx};
}

/**
 * @brief Executes a 1d binary block reduce
 * @param val binary value to be reduced across the thread block
 * @param shmem memory needed for the reduction. It should be at least of size blockDim.x/WarpSize
 * @return only the thread0 will contain valid reduced result
 */
template <int BLOCK_SIZE, typename i_t>
DI i_t binaryBlockReduce(i_t val, i_t* shmem)
{
  static_assert(BLOCK_SIZE <= 1024);
  assert(val == 0 || val == 1);
  const uint32_t mask    = __ballot_sync(~0, val);
  const uint32_t n_items = __popc(mask);

  // Each first thread of the warp
  if (threadIdx.x % WarpSize == 0) { shmem[threadIdx.x / WarpSize] = n_items; }
  __syncthreads();

  val = (threadIdx.x < BLOCK_SIZE / WarpSize) ? shmem[threadIdx.x] : 0;

  if (threadIdx.x < WarpSize) {
    return warpReduce(val);
  }
  // Only first warp gets the results
  else {
    return -1;
  }
}

/**
 * @brief Executes a collaborative vector reduction per sub-warp
 *
 * This uses fewer shuffles than naively reducing each element independently.
 * Better performance is achieved with a larger vector width, up to vecWidth == warpSize/2.
 * For example, for logicalWarpSize == 32 and vecWidth == 16, the naive method requires 80
 * shuffles, this one only 31, 2.58x fewer.
 *
 * However, the output of the reduction is not broadcasted. The vector is modified in place and
 * each thread holds a part of the output vector. The outputs are distributed in a round-robin
 * pattern between the threads to facilitate coalesced IO. There are 2 possible layouts based on
 * which of logicalWarpSize and vecWidth is larger:
 * - If vecWidth >= logicalWarpSize, each thread has vecWidth/logicalWarpSize outputs.
 * - If logicalWarpSize > vecWidth, logicalWarpSize/vecWidth threads have a copy of the same output.
 *
 * Example 1: logicalWarpSize == 4, vecWidth == 8, v = a+b+c+d
 *           IN                        OUT
 *  lane 0 | a0 a1 a2 a3 a4 a5 a6 a7 | v0 v4 - - - - - -
 *  lane 1 | b0 b1 b2 b3 b4 b5 b6 b7 | v1 v5 - - - - - -
 *  lane 2 | c0 c1 c2 c3 c4 c5 c6 c7 | v2 v6 - - - - - -
 *  lane 3 | d0 d1 d2 d3 d4 d5 d6 d7 | v3 v7 - - - - - -
 *
 * Example 2: logicalWarpSize == 8, vecWidth == 4, v = a+b+c+d+e+f+g+h
 *           IN            OUT
 *  lane 0 | a0 a1 a2 a3 | v0 - - -
 *  lane 1 | b0 b1 b2 b3 | v0 - - -
 *  lane 2 | c0 c1 c2 c3 | v1 - - -
 *  lane 3 | d0 d1 d2 d3 | v1 - - -
 *  lane 4 | e0 e1 e2 e3 | v2 - - -
 *  lane 5 | f0 f1 f2 f3 | v2 - - -
 *  lane 6 | g0 g1 g2 g3 | v3 - - -
 *  lane 7 | h0 h1 h2 h3 | v3 - - -
 *
 * @tparam logicalWarpSize Sub-warp size. Must be 2, 4, 8, 16 or 32.
 * @tparam vecWidth Vector width. Must be a power of two.
 * @tparam T Vector element type.
 * @tparam ReduceLambda Reduction operator type.
 * @param[in,out] acc Pointer to a vector of size vecWidth or more in registers
 * @param[in] lane_id Lane id between 0 and logicalWarpSize-1
 * @param[in] reduce_op Reduction operator, assumed to be commutative and associative.
 */
template <int logicalWarpSize, int vecWidth, typename T, typename ReduceLambda>
DI void logicalWarpReduceVector(T* acc, int lane_id, ReduceLambda reduce_op)
{
  static_assert(vecWidth > 0, "Vec width must be strictly positive.");
  static_assert(!(vecWidth & (vecWidth - 1)), "Vec width must be a power of two.");
  static_assert(logicalWarpSize >= 2 && logicalWarpSize <= 32,
                "Logical warp size must be between 2 and 32");
  static_assert(!(logicalWarpSize & (logicalWarpSize - 1)),
                "Logical warp size must be a power of two.");

  constexpr int shflStride   = logicalWarpSize / 2;
  constexpr int nextWarpSize = logicalWarpSize / 2;

  // One step of the butterfly reduction, applied to each element of the vector.
#pragma unroll
  for (int k = 0; k < vecWidth; k++) {
    const T tmp = shfl_xor(acc[k], shflStride, logicalWarpSize);
    acc[k]      = reduce_op(acc[k], tmp);
  }

  constexpr int nextVecWidth = std::max(1, vecWidth / 2);

  /* Split into 2 smaller logical warps and distribute half of the data to each for the next step.
   * The distribution pattern is designed so that at the end the outputs are coalesced/round-robin.
   * The idea is to distribute contiguous "chunks" of the vectors based on the new warp size. These
   * chunks will be halved in the next step and so on.
   *
   * Example for logicalWarpSize == 4, vecWidth == 8:
   *  lane 0 | 0 1 2 3 4 5 6 7 | [0 1] [4 5] - - - - | [0] [4] - - - - - -
   *  lane 1 | 0 1 2 3 4 5 6 7 | [0 1] [4 5] - - - - | [1] [5] - - - - - -
   *  lane 2 | 0 1 2 3 4 5 6 7 | [2 3] [6 7] - - - - | [2] [6] - - - - - -
   *  lane 3 | 0 1 2 3 4 5 6 7 | [2 3] [6 7] - - - - | [3] [7] - - - - - -
   *                      chunkSize=2           chunkSize=1
   */
  if constexpr (nextVecWidth < vecWidth) {
    T tmp[nextVecWidth];
    const bool firstHalf    = (lane_id % logicalWarpSize) < nextWarpSize;
    constexpr int chunkSize = std::min(nextVecWidth, nextWarpSize);
    constexpr int numChunks = nextVecWidth / chunkSize;
#pragma unroll
    for (int c = 0; c < numChunks; c++) {
#pragma unroll
      for (int i = 0; i < chunkSize; i++) {
        const int k = c * chunkSize + i;
        tmp[k]      = firstHalf ? acc[2 * c * chunkSize + i] : acc[(2 * c + 1) * chunkSize + i];
      }
    }
#pragma unroll
    for (int k = 0; k < nextVecWidth; k++) {
      acc[k] = tmp[k];
    }
  }

  // Recursively call with smaller sub-warps and possibly smaller vector width.
  if constexpr (nextWarpSize > 1) {
    logicalWarpReduceVector<nextWarpSize, nextVecWidth>(acc, lane_id % nextWarpSize, reduce_op);
  }
}

}  // namespace raft
