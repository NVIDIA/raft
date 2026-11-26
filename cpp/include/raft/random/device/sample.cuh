/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
namespace random::device {

/**
 * @brief warp-level random sampling of an index.
 *
 * It selects an index with the given discrete probability
 * distribution(represented by weights of each index).
 * Only thread 0 will contain the valid reduced result.
 *
 * @param rng random number generator, must have next_u32() function
 * @param weight weight of the rank/index.
 * @param idx index to be used as rank
 */
template <typename T, typename rng_t, typename i_t = int>
DI void warp_random_sample(rng_t& rng, T& weight, i_t& idx)
{
  // Todo(#1491): benchmark whether a scan and then selecting within the ranges is more efficient.
  static_assert(std::is_integral<T>::value, "The type T must be an integral type.");
#pragma unroll
  for (i_t offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
    T tmp_weight = shfl(weight, laneId() + offset);
    i_t tmp_idx  = shfl(idx, laneId() + offset);
    T sum        = (tmp_weight + weight);
    weight       = sum;
    if (sum != 0) {
      i_t rnd_number = (rng.next_u32() % sum);
      if (rnd_number < tmp_weight) { idx = tmp_idx; }
    }
  }
}

/**
 * @brief 1-D block-level random sampling of an index.
 * It selects an index with the given discrete probability
 * distribution(represented by weights of each index)
 *
 * Let w_i be the weight stored on thread i.  We calculate the cumulative distribution function
 * F_i = sum_{k=0..i} weight_i.
 * Sequentially, we could select one of the elements with with the desired probability using the
 * following method. We can consider that each element has a subinterval assigned: [F_{i-1}, F_i).
 * We generate a uniform random number in the [0, F_i) range, and check which subinterval it falls.
 * We return idx corresponding to the selected subinterval.
 * In parallel, we do a tree reduction and make a selection at every step when we combine two
 * values.
 * @param rng random number generator, must have next_u32() function
 * @param shbuf shared memory region needed for storing intermediate results. It
 *             must at least be of size: `(sizeof(T) + sizeof(i_t)) * WarpSize`
 * @param weight weight of the rank/index.
 * @param idx index to be used as rank
 * @return only the thread0 will contain valid reduced result
 */
template <typename T, typename rng_t, typename i_t = int>
DI i_t block_random_sample(rng_t rng, T* shbuf, T weight = 1, i_t idx = threadIdx.x)
{
  T* values    = shbuf;
  i_t* indices = (i_t*)&shbuf[WarpSize];
  i_t wid      = threadIdx.x / WarpSize;
  i_t nWarps   = (blockDim.x + WarpSize - 1) / WarpSize;
  warp_random_sample(rng, weight, idx);  // Each warp performs partial reduction
  i_t lane = laneId();
  if (lane == 0) {
    values[wid]  = weight;  // Write reduced value to shared memory
    indices[wid] = idx;     // Write reduced value to shared memory
  }

  __syncthreads();  // Wait for all partial reductions

  // read from shared memory only if that warp existed
  if (lane < nWarps) {
    weight = values[lane];
    idx    = indices[lane];
  } else {
    weight = 0;
    idx    = -1;
  }
  __syncthreads();
  if (wid == 0) warp_random_sample(rng, weight, idx);
  return idx;
}

}  // namespace random::device
}  // namespace raft
