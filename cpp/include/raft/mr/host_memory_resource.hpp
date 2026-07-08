/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/pmr/resource_adaptor.hpp>

#include <memory_resource>
#include <mutex>
#include <utility>

namespace raft {
namespace mr {

/**
 * @brief Get a reference to a stateless new/delete host memory resource.
 *
 * Analogous to std::pmr::new_delete_resource(), but returns raft::mr::host_resource_ref.
 */
inline auto new_delete_resource() -> raft::mr::host_resource_ref
{
  static raft::pmr::resource_adaptor instance{std::pmr::new_delete_resource()};
  return raft::mr::host_resource_ref{instance};
}

namespace detail {

struct default_host_resource_holder {
 private:
  std::mutex lock_;
  raft::mr::host_resource res_{raft::mr::new_delete_resource()};

 public:
  inline auto set(raft::mr::host_resource res) -> raft::mr::host_resource
  {
    std::unique_lock<std::mutex> guard(lock_);
    return std::exchange(res_, res);
  }
  inline auto get() -> raft::mr::host_resource_ref
  {
    std::unique_lock<std::mutex> guard(lock_);
    return raft::mr::host_resource_ref{res_};
  }
};

inline default_host_resource_holder default_host_resource_holder_{};

}  // namespace detail

/**
 * @brief Get the current default host memory resource.
 *
 * Returns raft::mr::host_resource_ref pointing to the resource installed
 * via set_default_host_resource(), or new_delete_resource() if none was set.
 */
inline auto get_default_host_resource() -> raft::mr::host_resource_ref
{
  return detail::default_host_resource_holder_.get();
}

/**
 * @brief Set the default host memory resource.
 *
 * (same contract as rmm::mr::set_current_device_resource).
 *
 * @param res The resource to install.
 * @return The previous default host resource.
 */
inline auto set_default_host_resource(raft::mr::host_resource res) -> raft::mr::host_resource
{
  return detail::default_host_resource_holder_.set(res);
}

}  // namespace mr
}  // namespace raft
