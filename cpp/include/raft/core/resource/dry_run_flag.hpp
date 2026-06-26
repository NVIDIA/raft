/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <memory>

namespace raft::resource {

/**
 * @defgroup dry_run_flag Dry-run flag resource
 * @{
 */

/**
 * @brief Resource that holds a boolean dry-run flag.
 *
 * When the dry-run flag is set, algorithms should skip kernel execution
 * and only perform allocations to measure memory usage.
 */
class dry_run_flag_resource : public resource {
 public:
  dry_run_flag_resource() = default;
  explicit dry_run_flag_resource(bool value) : flag_(value) {}
  ~dry_run_flag_resource() override = default;

  auto get_resource() -> void* override { return &flag_; }

  void set(bool value) { flag_ = value; }
  [[nodiscard]] auto get() const -> bool { return flag_; }

 private:
  bool flag_{false};
};

/**
 * @brief Factory that creates a dry_run_flag_resource.
 */
class dry_run_flag_resource_factory : public resource_factory {
 public:
  explicit dry_run_flag_resource_factory(bool initial_value = false) : initial_value_(initial_value)
  {
  }

  auto get_resource_type() -> resource_type override { return resource_type::DRY_RUN_FLAG; }
  auto make_resource() -> resource* override { return new dry_run_flag_resource(initial_value_); }

 private:
  bool initial_value_;
};

/**
 * @brief Get the dry-run flag from a resources handle.
 *
 * @param res raft resources object
 * @return true if dry-run mode is active
 */
inline auto get_dry_run_flag(resources const& res) -> bool
{
  if (!res.has_resource_factory(resource_type::DRY_RUN_FLAG)) {
    res.add_resource_factory(std::make_shared<dry_run_flag_resource_factory>());
  }
  return *res.get_resource<bool>(resource_type::DRY_RUN_FLAG);
}

/**
 * @brief Set the dry-run flag on a resources handle.
 *
 * @param res raft resources object
 * @param value true to enable dry-run mode, false to disable
 */
inline void set_dry_run_flag(resources const& res, bool value)
{
  if (!res.has_resource_factory(resource_type::DRY_RUN_FLAG)) {
    res.add_resource_factory(std::make_shared<dry_run_flag_resource_factory>(value));
  } else {
    // The resource may already be instantiated; update it directly
    auto* flag = res.get_resource<bool>(resource_type::DRY_RUN_FLAG);
    *flag      = value;
  }
}

/** @} */

}  // namespace raft::resource
