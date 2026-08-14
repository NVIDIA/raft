/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/detail/macros.hpp>
#include <raft/core/interruptible.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/cuda_stream.hpp>

#include <gtest/gtest.h>
#include <omp.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

namespace raft {

RAFT_KERNEL gpu_wait(int millis)
{
  for (auto i = millis; i > 0; i--) {
#if __CUDA_ARCH__ >= 700
    __nanosleep(1000000);
#else
    // For older CUDA devices:
    // just do some random work that takes more or less the same time from run to run.
    volatile double x = 0;
    for (int i = 0; i < 10000; i++) {
      x = x + double(i);
      x = x / 2.0;
      __syncthreads();
    }
#endif
  }
}

TEST(Raft, InterruptibleBasic)
{
  ASSERT_TRUE(interruptible::yield_no_throw());

  // Cancel using the token
  interruptible::get_token()->cancel();
  ASSERT_FALSE(interruptible::yield_no_throw());
  ASSERT_TRUE(interruptible::yield_no_throw());

  // Cancel by thread id
  interruptible::cancel(std::this_thread::get_id());
  ASSERT_FALSE(interruptible::yield_no_throw());
  ASSERT_TRUE(interruptible::yield_no_throw());
}

TEST(Raft, InterruptibleRepeatedGetToken)
{
  auto i     = std::this_thread::get_id();
  auto a1    = interruptible::get_token();
  auto count = a1.use_count();
  auto a2    = interruptible::get_token();
  ASSERT_LT(count, a1.use_count());
  count   = a1.use_count();
  auto b1 = interruptible::get_token(i);
  ASSERT_LT(count, a1.use_count());
  count   = a1.use_count();
  auto b2 = interruptible::get_token(i);
  ASSERT_LT(count, a1.use_count());

  ASSERT_EQ(a1, a2);
  ASSERT_EQ(a1, b2);
  ASSERT_EQ(b1, b2);
}

TEST(Raft, InterruptibleDelayedInit)
{
  std::thread([&]() {
    auto a = interruptible::get_token(std::this_thread::get_id());
    ASSERT_EQ(a.use_count(), 1);  // the only pointer here is [a]
    auto b = interruptible::get_token();
    ASSERT_EQ(a.use_count(), 3);  // [a, b, thread_local]
    auto c = interruptible::get_token();
    ASSERT_EQ(a.use_count(), 4);  // [a, b, c, thread_local]
    ASSERT_EQ(a.get(), b.get());
    ASSERT_EQ(a.get(), c.get());
  }).join();
}

TEST(Raft, InterruptibleOpenMP)
{
  // number of threads must be smaller than max number of resident grids for GPU
  const int n_threads = 10;
  // 1 <= n_expected_succeed <= n_threads
  const int n_expected_succeed = 5;
  // How many milliseconds passes between a thread i and i+1 finishes.
  // i.e. thread i executes (C + i*n_expected_succeed) milliseconds in total.
  const int thread_delay_millis = 20;
  common::nvtx::range fun_scope("interruptible");

  std::vector<std::shared_ptr<interruptible>> thread_tokens(n_threads);
  int n_finished  = 0;
  int n_cancelled = 0;

  omp_set_dynamic(0);
  omp_set_num_threads(n_threads);
#pragma omp parallel reduction(+ : n_finished) reduction(+ : n_cancelled) num_threads(n_threads)
  {
    auto i = omp_get_thread_num();
    common::nvtx::range omp_scope("interruptible::thread-%d", i);
    rmm::cuda_stream stream;
    raft::launch_kernel(stream.value(), 1, 1, gpu_wait, 1);
    interruptible::synchronize(stream);
    thread_tokens[i] = interruptible::get_token();

#pragma omp barrier
    try {
      common::nvtx::range wait_scope("interruptible::wait-%d", i);
      raft::launch_kernel(stream.value(), 1, 1, gpu_wait, (1 + i) * thread_delay_millis);
      interruptible::synchronize(stream);
      n_finished = 1;
    } catch (interrupted_exception&) {
      n_cancelled = 1;
    }
    if (i == n_expected_succeed - 1) {
      common::nvtx::range cancel_scope("interruptible::cancel-%d", i);
      for (auto token : thread_tokens)
        token->cancel();
    }

#pragma omp barrier
    // clear the cancellation state to not disrupt other tests
    interruptible::yield_no_throw();
  }
  ASSERT_EQ(n_finished, n_expected_succeed);
  ASSERT_EQ(n_cancelled, n_threads - n_expected_succeed);
}
}  // namespace raft
