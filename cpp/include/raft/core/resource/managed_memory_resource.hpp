/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/host_device_resource.hpp>

#include <cuda/memory_resource>

namespace RAFT_EXPORT raft {
namespace resource {

/**
 * @defgroup managed_memory_resource Managed memory resource
 * @{
 */

class managed_memory_resource : public resource {
 public:
  explicit managed_memory_resource(raft::mr::host_device_resource mr) : mr_(std::move(mr)) {}
  ~managed_memory_resource() override = default;
  auto get_resource() -> void* override { return &mr_; }

 private:
  raft::mr::host_device_resource mr_;
};

class managed_memory_resource_factory : public resource_factory {
 public:
  managed_memory_resource_factory() : mr_(cuda::mr::legacy_managed_memory_resource{}) {}

  explicit managed_memory_resource_factory(raft::mr::host_device_resource mr) : mr_(std::move(mr))
  {
  }

  auto get_resource_type() -> resource_type override
  {
    return resource_type::MANAGED_MEMORY_RESOURCE;
  }
  auto make_resource() -> resource* override { return new managed_memory_resource(mr_); }

 private:
  raft::mr::host_device_resource mr_;
};

/**
 * @brief Get the managed memory resource as a non-owning host_device_resource_ref.
 *
 * Default: cuda::mr::legacy_managed_memory_resource.
 *
 * @param res raft resources object for managing resources
 * @return non-owning reference to the managed memory resource
 */
inline auto get_managed_memory_resource_ref(resources const& res)
  -> raft::mr::host_device_resource_ref
{
  res.ensure_default_factory(std::make_shared<managed_memory_resource_factory>());
  auto& mr =
    *res.get_resource<raft::mr::host_device_resource>(resource_type::MANAGED_MEMORY_RESOURCE);
  return raft::mr::host_device_resource_ref{mr};
}

/**
 * @brief Set the managed memory resource.
 *
 * @param res raft resources object for managing resources
 * @param mr  host+device accessible memory resource
 */
inline void set_managed_memory_resource(resources& res, raft::mr::host_device_resource mr)
{
  res.add_resource_factory(std::make_shared<managed_memory_resource_factory>(std::move(mr)));
}

/** @} */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
