/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_resources.hpp>

namespace RAFT_EXPORT raft {

/**
 * raft::handle_t is being kept around for backwards
 * compatibility and will be removed in a future version.
 *
 * Extending the `raft::handle_t` instead of `using` to
 * minimize needed changes downstream
 * (e.g. existing forward declarations, etc...)
 *
 * Use of `raft::resources` or `raft::handle_t` is preferred.
 */
class handle_t : public raft::device_resources {
 public:
  handle_t(const handle_t& handle, raft::mr::device_resource workspace_resource)
    : device_resources(handle, std::move(workspace_resource))
  {
  }

  handle_t(const handle_t&)            = default;
  handle_t(handle_t&&)                 = default;
  handle_t& operator=(const handle_t&) = default;
  handle_t& operator=(handle_t&&)      = default;

  /**
   * @brief Construct a resources instance with a stream view and stream pool
   *
   * @param[in] stream_view the default stream (which has the default per-thread stream if
   * unspecified)
   * @param[in] stream_pool the stream pool used (which has default of nullptr if unspecified)
   * @param[in] workspace_resource an optional resource used by some functions for allocating
   *            temporary workspaces.
   */
  handle_t(rmm::cuda_stream_view stream_view                  = rmm::cuda_stream_per_thread,
           std::shared_ptr<rmm::cuda_stream_pool> stream_pool = {nullptr},
           std::optional<raft::mr::device_resource> workspace_resource = std::nullopt)
    : device_resources{stream_view, stream_pool, std::move(workspace_resource)}
  {
  }

  /** Destroys all held-up resources */
  ~handle_t() override {}
};

}  // namespace RAFT_EXPORT raft
