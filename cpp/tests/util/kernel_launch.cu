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
#include <utility>

namespace raft {

namespace {

RAFT_KERNEL noop_kernel() {}

RAFT_KERNEL write_one_kernel(int* out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { *out = 1; }
}

RAFT_KERNEL copy_restricted_kernel(int const* __restrict__ in, int* out)
{
  if (threadIdx.x == 0 && blockIdx.x == 0) { *out = *in; }
}

void launch_write_one_with_restricted_pointer(raft::resources const& res, int* __restrict__ out)
{
  raft::launch_kernel(res, 1, 32, write_one_kernel, out);
}

RAFT_KERNEL smem_kernel(int* out)
{
  extern __shared__ int shared[];  // NOLINT(modernize-avoid-c-arrays)
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    shared[0] = 1;
    __threadfence_block();
    *out = shared[0];
  }
}

/** Whether `w` can be launched as it is named, i.e. as an lvalue. */
template <typename W>
concept launchable_as_named = requires(W w)
{
  raft::launch_kernel(w, dim3{}, dim3{}, noop_kernel);
};

/** Whether `w` can be launched after being moved from. */
template <typename W>
concept launchable_when_moved = requires(W w)
{
  raft::launch_kernel(std::move(w), dim3{}, dim3{}, noop_kernel);
};

// Only a prvalue built inside the launch expression may be launched, so that the reported location
// is always the one of the launch. Everything else must fail to compile.
static_assert(launchable_as_named<raft::resources&>,
              "resources must convert to a launch_on prvalue");
static_assert(launchable_as_named<rmm::cuda_stream_view>,
              "a stream view must convert to a launch_on prvalue");
static_assert(launchable_as_named<cudaStream_t>,
              "a raw stream handle must convert to a launch_on prvalue");
static_assert(!launchable_as_named<raft::launch_on>, "a stored launch_on must not be launchable");
static_assert(!launchable_as_named<raft::launch_on&>, "an lvalue launch_on must not be launchable");
static_assert(!launchable_when_moved<raft::launch_on>,
              "a moved-from launch_on must not be launchable");

}  // namespace

TEST(KernelLaunch, SuccessfulLaunch)
{
  raft::resources res;
  rmm::device_uvector<int> out(1, resource::get_cuda_stream(res));
  RAFT_CUDA_TRY(cudaMemsetAsync(out.data(), 0, sizeof(int), resource::get_cuda_stream(res)));

  raft::launch_kernel(res, 1, 32, write_one_kernel, out.data());
  resource::sync_stream(res);

  int host_out = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&host_out, out.data(), sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(host_out, 1);
}

TEST(KernelLaunch, RestrictedPointerArgument)
{
  raft::resources res;
  rmm::device_uvector<int> out(1, resource::get_cuda_stream(res));
  RAFT_CUDA_TRY(cudaMemsetAsync(out.data(), 0, sizeof(int), resource::get_cuda_stream(res)));

  launch_write_one_with_restricted_pointer(res, out.data());
  resource::sync_stream(res);

  int host_out = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&host_out, out.data(), sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(host_out, 1);
}

TEST(KernelLaunch, ConvertedRestrictedPointerArgument)
{
  raft::resources res;
  auto stream = resource::get_cuda_stream(res);
  rmm::device_uvector<int> in(1, stream);
  rmm::device_uvector<int> out(1, stream);
  int host_in = 1;
  RAFT_CUDA_TRY(cudaMemcpyAsync(in.data(), &host_in, sizeof(int), cudaMemcpyHostToDevice, stream));

  raft::launch_kernel(res, 1, 32, copy_restricted_kernel, in.data(), out.data());
  resource::sync_stream(res);

  int host_out = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&host_out, out.data(), sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(host_out, 1);
}

TEST(KernelLaunch, StreamOverload)
{
  raft::resources res;
  auto stream = resource::get_cuda_stream(res);
  EXPECT_NO_THROW(raft::launch_kernel(stream, 1, 1, noop_kernel));
  resource::sync_stream(res);
}

TEST(KernelLaunch, RawStreamHandleOverload)
{
  raft::resources res;
  cudaStream_t stream = resource::get_cuda_stream(res).value();
  EXPECT_NO_THROW(raft::launch_kernel(stream, 1, 1, noop_kernel));
  resource::sync_stream(res);
}

TEST(KernelLaunch, SharedMemory)
{
  raft::resources res;
  auto stream = resource::get_cuda_stream(res);
  rmm::device_uvector<int> out(1, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(out.data(), 0, sizeof(int), stream));

  raft::launch_kernel({stream, sizeof(int)}, 1, 32, smem_kernel, out.data());
  resource::sync_stream(res);

  int host_out = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&host_out, out.data(), sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(host_out, 1);
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
    raft::launch_kernel(res, 1, k_bad_block, noop_kernel);
    FAIL() << "Expected cuda_error from invalid launch configuration";
  } catch (raft::cuda_error const& e) {
    caught = e.what();
  }

  // Must blame this test translation unit, not the launcher header.
  EXPECT_EQ(caught.find("kernel_launch.hpp"), std::string::npos) << caught;
  EXPECT_NE(caught.find("kernel_launch.cu"), std::string::npos) << caught;

  std::string re_exp{R"(CUDA error encountered at: file=.*kernel_launch\.cu line=)"};
  re_exp += std::to_string(launch_line);
  re_exp += R"( function=.*ErrorReportsCallSite.*: call='cudaLaunchKernelExC', Reason=.*)";
  EXPECT_TRUE(std::regex_search(caught, std::regex(re_exp)))
    << "message:'" << caught << "'\nexpected regex:'" << re_exp << "'";
}

}  // namespace raft
