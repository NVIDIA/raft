/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/cuda_support.hpp>
#include <raft/core/stream_view.hpp>

#include <gtest/gtest.h>
#ifndef RAFT_DISABLE_CUDA
#include <cuda/stream>
#endif
namespace raft {
TEST(StreamView, Default)
{
  auto stream = stream_view_per_thread;
  ASSERT_EQ(stream.is_per_thread_default(), raft::CUDA_ENABLED);
  ASSERT_FALSE(stream.is_default());
  if (raft::CUDA_ENABLED) {
    EXPECT_NO_THROW(stream.synchronize());
    EXPECT_NO_THROW(stream.interruptible_synchronize());
  } else {
    EXPECT_THROW(stream.synchronize(), raft::non_cuda_build_error);
    EXPECT_THROW(stream.interruptible_synchronize(), raft::non_cuda_build_error);
  }
  EXPECT_NO_THROW(stream.synchronize_no_throw());
  EXPECT_NO_THROW(stream.synchronize_if_cuda_enabled());
#ifndef RAFT_DISABLE_CUDA
  static_assert(std::is_same_v<decltype(stream.underlying()), cuda::stream_ref>,
                "underlying should return cuda::stream_ref");
#endif
}
}  // namespace raft
