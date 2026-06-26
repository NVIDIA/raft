/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_event.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace raft {
namespace resource::detail {

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the res_t.
 */
class cuda_stream_sync_event_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::CUDA_STREAM_SYNC_EVENT; }
  resource* make_resource() override { return new cuda_event_resource(); }
};

/**
 * Load a cudaEvent from a resources instance (and populate it on the resources instance)
 * if needed) for syncing the main cuda stream.
 * @param res raft resources instance for managing resources
 * @return
 */
inline cudaEvent_t& get_cuda_stream_sync_event(resources const& res)
{
  res.ensure_default_factory(std::make_shared<cuda_stream_sync_event_resource_factory>());
  return *res.get_resource<cudaEvent_t>(resource_type::CUDA_STREAM_SYNC_EVENT);
};

}  // namespace resource::detail
}  // namespace raft
