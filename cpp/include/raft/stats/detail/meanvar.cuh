/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace stats::detail {

template <typename T>
class mean_var {
 private:
  T w;
  T m;
  T s;

 public:
  /** Monoidal neutral. */
  HDI mean_var() : w(0.0), m(0.0), s(0.0) {}
  /** Lift a single value. */
  HDI explicit mean_var(T x) : w(1.0), m(x), s(0.0) {}

  /**
   * Monoidal binary op: combine means and vars of two sets.
   * (associative and commutative)
   */
  friend HDI auto operator+(mean_var<T> a, mean_var<T> const& b) -> mean_var<T>
  {
    a += b;
    return a;
  }

  /**
   * Combine means and vars of two sets.
   *
   * Similar to:
   * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
   */
  HDI auto operator+=(mean_var<T> const& b) & -> mean_var<T>&
  {
    mean_var<T>& a(*this);
    T cw = a.w + b.w;
    if (cw == 0) return a;
    T aw_frac = a.w / cw;
    T bw_frac = b.w / cw;
    a.w       = cw;
    T d       = a.m - b.m;
    a.s += b.s + cw * (d * aw_frac) * (d * bw_frac);
    a.m = a.m * aw_frac + b.m * bw_frac;
    return a;
  }

  /** Get the computed mean. */
  HDI auto mean() const -> T { return m; }

  /**
   * @brief Get the computed variance.
   *
   * @param [in] sample whether to produce sample variance (divide by `N - 1` instead of `N`).
   * @return variance
   */
  HDI auto var(bool sample) const -> T { return s / max(T(1.0), sample ? w - T(1.0) : w); }

  HDI void load(volatile mean_var<T>* address)
  {
    this->m = address->m;
    this->s = address->s;
    this->w = address->w;
  }

  HDI void store(volatile mean_var<T>* address)
  {
    address->m = this->m;
    address->s = this->s;
    address->w = this->w;
  }
};

/*
NB: current implementation here is not optimal, especially the rowmajor version;
    leaving this for further work (perhaps, as a more generic "linewiseReduce").
    Vectorized loads/stores could speed things up a lot.
 */
/**
 * meanvar kernel - row-major version
 *
 * Assumptions:
 *
 *  1. blockDim.x == WarpSize
 *  2. Dimension X goes along columns (D)
 *  3. Dimension Y goes along rows (N)
 *
 *
 * @tparam T element type
 * @tparam I indexing type
 * @tparam BlockSize must be equal to blockDim.x * blockDim.y * blockDim.z
 * @param data input data
 * @param mvs meanvars -- output
 * @param locks guards for updating meanvars
 * @param len total length of input data (N * D)
 * @param D number of columns in the input data.
 */
template <typename T, typename I, int BlockSize>
RAFT_KERNEL __launch_bounds__(BlockSize)
  meanvar_kernel_rowmajor(const T* data, volatile mean_var<T>* mvs, int* locks, I len, I D)
{
  // read the data
  const I col = threadIdx.x + blockDim.x * blockIdx.x;
  mean_var<T> thread_data;
  if (col < D) {
    const I step = D * blockDim.y * gridDim.y;
    for (I i = col + D * (threadIdx.y + blockDim.y * blockIdx.y); i < len; i += step) {
      thread_data += mean_var<T>(data[i]);
    }
  }

  // aggregate within block
  if (blockDim.y > 1) {
    __shared__ uint8_t shm_bytes[BlockSize * sizeof(mean_var<T>)];
    auto shm = (mean_var<T>*)shm_bytes;
    int tid  = threadIdx.x + threadIdx.y * blockDim.x;
    shm[tid] = thread_data;
    for (int bs = BlockSize >> 1; bs >= blockDim.x; bs = bs >> 1) {
      __syncthreads();
      if (tid < bs) { shm[tid] += shm[tid + bs]; }
    }
    thread_data = shm[tid];
  }

  // aggregate across blocks
  if (threadIdx.y == 0) {
    int* lock = locks + blockIdx.x;
    if (threadIdx.x == 0 && col < D) {
      while (atomicCAS(lock, 0, 1) == 1) {
        __threadfence();
      }
    }
    __syncthreads();
    if (col < D) {
      __threadfence();
      mean_var<T> global_data;
      global_data.load(mvs + col);
      global_data += thread_data;
      global_data.store(mvs + col);
      __threadfence();
    }
    __syncthreads();
    if (threadIdx.x == 0 && col < D) { __stwt(lock, 0); }
  }
}

template <typename T, typename I, int BlockSize>
RAFT_KERNEL __launch_bounds__(BlockSize)
  meanvar_kernel_colmajor(T* mean, T* var, const T* data, I D, I N, bool sample)
{
  using BlockReduce = cub::BlockReduce<mean_var<T>, BlockSize>;
  __shared__ typename BlockReduce::TempStorage shm;

  const T* block_data = data + N * blockIdx.x;
  mean_var<T> thread_data;
  for (I i = threadIdx.x; i < N; i += BlockSize) {
    thread_data += mean_var<T>(block_data[i]);
  }
  mean_var<T> acc = BlockReduce(shm).Sum(thread_data);
  if (threadIdx.x == 0) {
    mean[blockIdx.x] = acc.mean();
    var[blockIdx.x]  = acc.var(sample);
  }
}

template <typename T, typename I>
RAFT_KERNEL meanvar_kernel_fill(T* mean, T* var, const mean_var<T>* aggr, I D, bool sample)
{
  I i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= D) return;
  auto x  = aggr[i];
  mean[i] = x.mean();
  var[i]  = x.var(sample);
}

template <typename T, typename I = int, int BlockSize = 256>
void meanvar(bool dry_run,
             T* mean,
             T* var,
             const T* data,
             I D,
             I N,
             bool sample,
             bool rowMajor,
             cudaStream_t stream)
{
  if (rowMajor) {
    static_assert(BlockSize >= WarpSize, "Block size must be not smaller than the warp size.");
    const dim3 bs(WarpSize, BlockSize / WarpSize, 1);
    dim3 gs(raft::ceildiv<decltype(bs.x)>(D, bs.x), raft::ceildiv<decltype(bs.y)>(N, bs.y), 1);

    // Don't create more blocks than necessary to occupy the GPU
    int occupancy;
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, meanvar_kernel_rowmajor<T, I, BlockSize>, BlockSize, 0));
    gs.y =
      std::min(gs.y, raft::ceildiv<decltype(gs.y)>(occupancy * getMultiProcessorCount(), gs.x));

    // Global memory: one mean_var<T> for each column
    //                one lock per all blocks working on the same set of columns
    rmm::device_buffer buf(sizeof(mean_var<T>) * D + sizeof(int) * gs.x, stream);
    if (!dry_run) {
      RAFT_CUDA_TRY(cudaMemsetAsync(buf.data(), 0, buf.size(), stream));
      mean_var<T>* mvs = static_cast<mean_var<T>*>(buf.data());
      int* locks       = static_cast<int*>(static_cast<void*>(mvs + D));

      const uint64_t len = uint64_t(D) * uint64_t(N);
      ASSERT(len <= uint64_t(std::numeric_limits<I>::max()),
             "N * D does not fit the indexing type");
      meanvar_kernel_rowmajor<T, I, BlockSize><<<gs, bs, 0, stream>>>(data, mvs, locks, len, D);
      meanvar_kernel_fill<T, I>
        <<<raft::ceildiv<I>(D, BlockSize), BlockSize, 0, stream>>>(mean, var, mvs, D, sample);
    }
  } else {
    if (!dry_run) {
      meanvar_kernel_colmajor<T, I, BlockSize>
        <<<D, BlockSize, 0, stream>>>(mean, var, data, D, N, sample);
    }
  }
  if (!dry_run) { RAFT_CHECK_CUDA(stream); }
}

};  // namespace stats::detail
}  // namespace raft
