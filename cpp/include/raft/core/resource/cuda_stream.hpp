/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/interruptible.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

namespace RAFT_EXPORT raft {
namespace resource {
class cuda_stream_resource : public resource {
 public:
  cuda_stream_resource(rmm::cuda_stream_view stream_view = rmm::cuda_stream_per_thread)
    : stream(stream_view)
  {
  }
  void* get_resource() override { return &stream; }

  ~cuda_stream_resource() override {}

 private:
  rmm::cuda_stream_view stream;
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
class cuda_stream_resource_factory : public resource_factory {
 public:
  cuda_stream_resource_factory(rmm::cuda_stream_view stream_view = rmm::cuda_stream_per_thread)
    : stream(stream_view)
  {
  }
  resource_type get_resource_type() override { return resource_type::CUDA_STREAM_VIEW; }
  resource* make_resource() override { return new cuda_stream_resource(stream); }

 private:
  rmm::cuda_stream_view stream;
};

/**
 * @defgroup resource_cuda_stream CUDA stream resource functions
 * @{
 */
/**
 * Load a rmm::cuda_stream_view from a resources instance (and populate it on the res
 * if needed).
 * @param res raft res object for managing resources
 * @return
 */
inline rmm::cuda_stream_view get_cuda_stream(resources const& res)
{
  res.ensure_default_factory(std::make_shared<cuda_stream_resource_factory>());
  return *res.get_resource<rmm::cuda_stream_view>(resource_type::CUDA_STREAM_VIEW);
};

/**
 * Load a rmm::cuda_stream_view from a resources instance (and populate it on the res
 * if needed).
 * @param[in] res raft resources object for managing resources
 * @param[in] stream_view cuda stream view
 */
inline void set_cuda_stream(resources& res, rmm::cuda_stream_view stream_view)
{
  res.add_resource_factory(std::make_shared<cuda_stream_resource_factory>(stream_view));
};

/**
 * @brief synchronize a specific stream
 *
 * @param[in] res the raft resources object
 * @param[in] stream stream to synchronize
 */
inline void sync_stream(const resources& res, rmm::cuda_stream_view stream)
{
  interruptible::synchronize(stream);
}

/**
 * @brief synchronize main stream on the resources instance
 */
inline void sync_stream(const resources& res) { sync_stream(res, get_cuda_stream(res)); }

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
