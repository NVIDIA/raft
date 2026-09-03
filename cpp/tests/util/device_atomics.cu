/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/detail/macros.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/device_atomics.cuh>
#include <raft/util/kernel_launch.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <memory>
#include <numeric>

namespace raft {

RAFT_KERNEL test_atomic_inc_warp_kernel(int* counter, int* out_array)
{
  int global_tid                    = blockDim.x * blockIdx.x + threadIdx.x;
  out_array[atomicIncWarp(counter)] = global_tid;
}

TEST(Raft, AtomicIncWarp)
{
  const int num_blocks        = 1024;
  const int threads_per_block = 1024;
  const int num_elts          = num_blocks * threads_per_block;

  rmm::cuda_stream_pool pool{1};
  auto s = pool.get_stream();

  int zero = 0;
  rmm::device_scalar<int> counter{zero, s};
  rmm::device_uvector<int> out_device{num_elts, s};
  std::array<int, num_elts> out_host{0};

  // Write all 1M thread indices to a unique location in `out_device`
  raft::launch_kernel(s,
                      num_blocks,
                      threads_per_block,
                      test_atomic_inc_warp_kernel,
                      counter.data(),
                      out_device.data());
  // Copy data to host
  RAFT_CUDA_TRY(cudaMemcpyAsync(out_host.data(),
                                (const void*)out_device.data(),
                                num_elts * sizeof(int),
                                cudaMemcpyDeviceToHost,
                                s));

  // Check that count is correct and that each thread index is contained in the
  // array exactly once.
  ASSERT_EQ(num_elts, counter.value(s));  // NB: accessing the counter synchronizes `s`
  std::sort(out_host.begin(), out_host.end());
  for (int i = 0; i < num_elts; ++i) {
    ASSERT_EQ(i, out_host[i]);
  }
}

}  // namespace raft
