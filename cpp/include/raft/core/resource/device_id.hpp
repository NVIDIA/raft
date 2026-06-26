/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace RAFT_EXPORT raft {
namespace resource {

class device_id_resource : public resource {
 public:
  device_id_resource()
    : dev_id_([]() -> int {
        int cur_dev = -1;
        RAFT_CUDA_TRY_NO_THROW(cudaGetDevice(&cur_dev));
        return cur_dev;
      }())
  {
  }
  void* get_resource() override { return &dev_id_; }

  ~device_id_resource() override {}

 private:
  int dev_id_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class device_id_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::DEVICE_ID; }
  resource* make_resource() override { return new device_id_resource(); }
};

/**
 * @defgroup resource_device_id Device ID resource functions
 * @{
 */

/**
 * Load a device id from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return device id
 */
inline int get_device_id(resources const& res)
{
  res.ensure_default_factory(std::make_shared<device_id_resource_factory>());
  return *res.get_resource<int>(resource_type::DEVICE_ID);
};

/**
 * @}
 */
}  // namespace resource
}  // namespace RAFT_EXPORT raft
