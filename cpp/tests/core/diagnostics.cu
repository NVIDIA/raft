/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/cusparse_macros.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/stream_view.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>

#include <source_location>
#include <string>
#include <type_traits>

namespace {
using location = raft::source_location;
#if defined(RAFT_COMPACT_DIAGNOSTICS) && RAFT_COMPACT_DIAGNOSTICS
static_assert(!std::is_same_v<location, std::source_location>);
static_assert(location{}.function_name()[0] == '\0');
static_assert(location{}.column() == 0);
#else
static_assert(std::is_same_v<location, std::source_location>);
#endif
std::string first_line(std::string msg) { return msg.substr(0, msg.find('\n')); }
void check_message(std::string msg, unsigned line, std::string const& reason)
{
  msg = first_line(msg);
  EXPECT_NE(msg.find(std::string("file=") + __FILE__ + " line=" + std::to_string(line)),
            std::string::npos)
    << msg;
  EXPECT_NE(msg.find(reason), std::string::npos) << msg;
#if defined(RAFT_COMPACT_DIAGNOSTICS) && RAFT_COMPACT_DIAGNOSTICS
  EXPECT_EQ(msg.find(" function="), std::string::npos) << msg;
#else
  EXPECT_NE(msg.find(" function="), std::string::npos) << msg;
#endif
}
std::string inner(location loc)
{
  return raft::format_error_message(loc, "forward: ", "value=%d %s %.2f", 42, "answer", 1.25);
}
std::string outer(location loc = location::current()) { return inner(loc); }
void cuda_inner(location loc)
{
  raft::check_cuda_error(cudaErrorInvalidValue, "forwarded_cuda_call", loc);
}
void cuda_outer(location loc = location::current()) { cuda_inner(loc); }

TEST(Diagnostics, CaptureAndNestedForwarding)
{
  auto line = __LINE__ + 1;
  auto loc  = location::current();
  EXPECT_EQ(loc.line(), line);
  EXPECT_STREQ(loc.file_name(), __FILE__);
  line     = __LINE__ + 1;
  auto msg = outer();
  check_message(msg, line, "value=42 answer 1.25");
}
TEST(Diagnostics, DirectMacrosAndSuccess)
{
  int calls = 0;
  EXPECT_NO_THROW(RAFT_EXPECTS(++calls == 1, "must not throw"));
  EXPECT_EQ(calls, 1);
  auto line = __LINE__ + 2;
  try {
    RAFT_EXPECTS(false, "value=%d %s", -7, "bad");
    FAIL();
  } catch (raft::logic_error const& e) {
    check_message(e.what(), line, "value=-7 bad");
  }
  line = __LINE__ + 2;
  try {
    RAFT_FAIL("failure %d", 9);
    FAIL();
  } catch (raft::logic_error const& e) {
    check_message(e.what(), line, "failure 9");
  }
}
TEST(Diagnostics, ExplicitStdLocation)
{
  auto line          = __LINE__ + 1;
  auto loc           = std::source_location::current();
  location converted = loc;
  EXPECT_EQ(converted.line(), line);
  check_message(inner(loc), line, "value=42 answer 1.25");
  try {
    cuda_outer(loc);
    FAIL();
  } catch (raft::cuda_error const& e) {
    check_message(e.what(), line, "cudaErrorInvalidValue");
  }
}
TEST(Diagnostics, CudaWrapperAndNestedForwarding)
{
  EXPECT_NO_THROW(RAFT_CUDA_TRY(cudaSuccess));
  auto line = __LINE__ + 2;
  try {
    RAFT_CUDA_TRY(cudaErrorInvalidValue);
    FAIL();
  } catch (raft::cuda_error const& e) {
    check_message(e.what(), line, "call='cudaErrorInvalidValue', Reason=cudaErrorInvalidValue");
  }
  line = __LINE__ + 2;
  try {
    cuda_outer();
    FAIL();
  } catch (raft::cuda_error const& e) {
    check_message(e.what(), line, "call='forwarded_cuda_call'");
  }
}
TEST(Diagnostics, LibraryWrappers)
{
  EXPECT_NO_THROW(RAFT_CUBLAS_TRY(CUBLAS_STATUS_SUCCESS));
  EXPECT_NO_THROW(RAFT_CUSOLVER_TRY(CUSOLVER_STATUS_SUCCESS));
  EXPECT_NO_THROW(RAFT_CUSPARSE_TRY(CUSPARSE_STATUS_SUCCESS));
  auto line = __LINE__ + 2;
  try {
    RAFT_CUBLAS_TRY(CUBLAS_STATUS_INVALID_VALUE);
    FAIL();
  } catch (raft::cublas_error const& e) {
    check_message(e.what(), line, "Reason=7:CUBLAS_STATUS_INVALID_VALUE");
  }
  line = __LINE__ + 2;
  try {
    RAFT_CUSOLVER_TRY(CUSOLVER_STATUS_INVALID_VALUE);
    FAIL();
  } catch (raft::cusolver_error const& e) {
    check_message(e.what(), line, "Reason=3:CUSOLVER_STATUS_INVALID_VALUE");
  }
  line = __LINE__ + 2;
  try {
    RAFT_CUSPARSE_TRY(CUSPARSE_STATUS_INVALID_VALUE);
    FAIL();
  } catch (raft::cusparse_error const& e) {
    check_message(e.what(), line, "Reason=3:invalid value");
  }
}
TEST(Diagnostics, ResourceHelpersAcceptExplicitLocations)
{
  raft::device_resources res;
  auto loc = std::source_location::current();
  EXPECT_NO_THROW(raft::resource::sync_stream(res, loc));
  EXPECT_NO_THROW(raft::resource::sync_stream(res, raft::resource::get_cuda_stream(res), loc));
  EXPECT_NO_THROW(res.sync_stream(loc));
  EXPECT_NO_THROW(raft::interruptible::yield(loc));
}
TEST(Diagnostics, StreamAndResourceCancellationBlamesCaller)
{
  raft::device_resources res;
  raft::interruptible::get_token()->cancel();
  auto line = __LINE__ + 2;
  try {
    raft::resource::sync_stream(res);
    FAIL();
  } catch (raft::interrupted_exception const& e) {
    check_message(e.what(), line, "The work in this thread was cancelled.");
  }
  raft::interruptible::get_token()->cancel();
  line = __LINE__ + 2;
  try {
    res.sync_stream();
    FAIL();
  } catch (raft::interrupted_exception const& e) {
    check_message(e.what(), line, "The work in this thread was cancelled.");
  }
  raft::interruptible::get_token()->cancel();
  line = __LINE__ + 2;
  try {
    raft::resource::sync_stream_view(res);
    FAIL();
  } catch (raft::interrupted_exception const& e) {
    check_message(e.what(), line, "The work in this thread was cancelled.");
  }
  auto explicit_loc = std::source_location::current();
  raft::interruptible::get_token()->cancel();
  try {
    raft::resource::sync_stream(res, explicit_loc);
    FAIL();
  } catch (raft::interrupted_exception const& e) {
    check_message(e.what(), explicit_loc.line(), "The work in this thread was cancelled.");
  }
}
}  // namespace
