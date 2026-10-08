/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * RAFT_LOG_TRACE_VEC expands to void(0) unless RAFT_LOG_ACTIVE_LEVEL is TRACE,
 * so a build at the default logging level never compiles its body. Raise the
 * level for this translation unit only, so the macro is actually instantiated.
 */
#include <rapids_logger/logger.hpp>

#undef RAFT_LOG_ACTIVE_LEVEL
#define RAFT_LOG_ACTIVE_LEVEL RAPIDS_LOGGER_LOG_LEVEL_TRACE

#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace raft {

// Compiling these at all is the point: the macro previously referenced
// raft::detail::format, which no longer exists, an unqualified print_vector,
// and a log level constant where the enumerator is expected.

TEST(Logger, TraceVecHostPointer)
{
  const std::vector<int> host_data{1, 2, 3, 4};
  ASSERT_NO_THROW(RAFT_LOG_TRACE_VEC(host_data.data(), host_data.size()));
}

TEST(Logger, TraceVecDevicePointer)
{
  // print_vector inspects the pointer with cudaPointerGetAttributes and copies
  // device memory back to the host, so cover that branch as well.
  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);

  const std::vector<int> host_data{5, 6, 7, 8};
  rmm::device_uvector<int> device_data(host_data.size(), stream);
  raft::update_device(device_data.data(), host_data.data(), host_data.size(), stream);
  resource::sync_stream(handle, stream);

  ASSERT_NO_THROW(RAFT_LOG_TRACE_VEC(device_data.data(), device_data.size()));
}

}  // namespace raft
