/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <regex>
#include <string>

namespace raft {

namespace {

RAFT_KERNEL noop_kernel() {}

RAFT_KERNEL write_one_kernel(int* out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { *out = 1; }
}

}  // namespace

TEST(KernelLaunch, SuccessfulLaunch)
{
  raft::resources res;
  rmm::device_uvector<int> out(1, resource::get_cuda_stream(res));
  RAFT_CUDA_TRY(cudaMemsetAsync(out.data(), 0, sizeof(int), resource::get_cuda_stream(res)));

  raft::launch_kernel(res, 1, 32)(write_one_kernel, out.data());
  resource::sync_stream(res);

  int host_out = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&host_out, out.data(), sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(host_out, 1);
}

TEST(KernelLaunch, StreamOverload)
{
  raft::resources res;
  auto stream = resource::get_cuda_stream(res);
  EXPECT_NO_THROW(raft::launch_kernel(stream, 1, 1)(noop_kernel));
  resource::sync_stream(res);
}

TEST(KernelLaunch, ErrorReportsCallSite)
{
  raft::resources res;

  // Intentionally invalid configuration: block size exceeds hardware limit.
  constexpr int k_bad_block = 2048;
  std::string caught;
  int launch_line = 0;
  try {
    launch_line = __LINE__ + 1;
    raft::launch_kernel(res, 1, k_bad_block)(noop_kernel);
    FAIL() << "Expected cuda_error from invalid launch configuration";
  } catch (raft::cuda_error const& e) {
    caught = e.what();
  }

  // Must blame this test translation unit, not the launcher header.
  EXPECT_EQ(caught.find("kernel_launch.hpp"), std::string::npos) << caught;
  EXPECT_NE(caught.find("kernel_launch.cu"), std::string::npos) << caught;

  std::string re_exp{R"(CUDA error encountered at: file=.*kernel_launch\.cu line=)"};
  re_exp += std::to_string(launch_line);
  re_exp += R"( function=.*ErrorReportsCallSite.*: call='cudaLaunchKernel', Reason=.*)";
  EXPECT_TRUE(std::regex_search(caught, std::regex(re_exp)))
    << "message:'" << caught << "'\nexpected regex:'" << re_exp << "'";
}

}  // namespace raft
