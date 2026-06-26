/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>

namespace RAFT_EXPORT raft {
namespace resource {

class device_properties_resource : public resource {
 public:
  device_properties_resource(int dev_id)
  {
    RAFT_CUDA_TRY_NO_THROW(cudaGetDeviceProperties(&prop_, dev_id));
  }
  void* get_resource() override { return &prop_; }

  ~device_properties_resource() override {}

 private:
  cudaDeviceProp prop_;
};

/**
 * @defgroup resource_device_props Device properties resource functions
 * @{
 */

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class device_properties_resource_factory : public resource_factory {
 public:
  device_properties_resource_factory(int dev_id) : dev_id_(dev_id) {}
  resource_type get_resource_type() override { return resource_type::DEVICE_PROPERTIES; }
  resource* make_resource() override { return new device_properties_resource(dev_id_); }

 private:
  int dev_id_;
};

/**
 * Load a cudaDeviceProp from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return populated cuda device properties instance
 */
inline cudaDeviceProp& get_device_properties(resources const& res)
{
  res.ensure_default_factory(
    std::make_shared<device_properties_resource_factory>(get_device_id(res)));
  return *res.get_resource<cudaDeviceProp>(resource_type::DEVICE_PROPERTIES);
};

/**
 * @}
 */
}  // namespace resource
}  // namespace RAFT_EXPORT raft
