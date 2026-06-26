/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/memory_stats_resources.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/dry_run_resource.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/mr/host_memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace raft {

/**
 * @defgroup dry_run_memory Dry-run memory resources
 * @{
 */

/**
 * @brief Resources handle that wraps all reachable memory resources with
 *        dry-run adaptors and tracks peak allocation usage.
 *
 * Inherits from raft::resources, so it can be passed anywhere a
 * raft::resources& is expected.  On construction the handle:
 *   - If dry-run mode is already active, does nothing (no-op).
 *   - Materializes all tracked resource types (host, device, pinned,
 *     managed, workspace, large_workspace).
 *   - Takes a snapshot of the original resources to keep them alive.
 *   - Wraps each with dry_run_resource.
 *   - Replaces global host and device resources with dry-run versions.
 *   - Sets the dry-run flag.
 *
 * On destruction the handle resets the flag and restores global resources.
 * Composable with memory_tracking_resources in either order.
 */
class dry_run_resources : public resources {
 public:
  explicit dry_run_resources(const resources& existing)
    : resources(existing),
      active_(!resource::get_dry_run_flag(existing)),
      old_host_(raft::mr::get_default_host_resource()),
      old_device_(rmm::mr::get_current_device_resource_ref())
  {
    if (active_) init();
  }

  ~dry_run_resources() override
  {
    if (!active_) return;
    resource::set_dry_run_flag(*this, false);
    raft::mr::set_default_host_resource(old_host_);
    rmm::mr::set_current_device_resource(old_device_);

    // Drop all base-class entries so that probe container RAII cleanup runs
    // while old_device_ and snapshot_ are still alive
    resources_.clear();
    factories_.clear();
  }

  dry_run_resources(dry_run_resources const&)            = delete;
  dry_run_resources& operator=(dry_run_resources const&) = delete;
  dry_run_resources(dry_run_resources&&)                 = delete;
  dry_run_resources& operator=(dry_run_resources&&)      = delete;

  [[nodiscard]] auto get_bytes_peak() const -> memory_stats
  {
    if (!active_) return {};
    return {
      .device_workspace       = ws_stats_->get_peak_bytes(),
      .device_large_workspace = lws_stats_->get_peak_bytes(),
      .device_global          = device_stats_->get_peak_bytes(),
      .device_managed         = managed_stats_->get_peak_bytes(),
      .host                   = host_stats_->get_peak_bytes(),
      .host_pinned            = pinned_stats_->get_peak_bytes(),
    };
  }

  [[nodiscard]] auto get_bytes_current() const -> memory_stats
  {
    if (!active_) return {};
    return {
      .device_workspace       = ws_stats_->get_allocated_bytes(),
      .device_large_workspace = lws_stats_->get_allocated_bytes(),
      .device_global          = device_stats_->get_allocated_bytes(),
      .device_managed         = managed_stats_->get_allocated_bytes(),
      .host                   = host_stats_->get_allocated_bytes(),
      .host_pinned            = pinned_stats_->get_allocated_bytes(),
    };
  }

 private:
  // Declaration order determines destruction order.
  // snapshot_ is destroyed last (keeps original resource shared_ptrs alive
  // while dry-run adaptors hold non-owning refs into them).
  // old_device_ is destroyed after device_adaptor_ so the probe can
  // deallocate through it during device_adaptor_ destruction.
  std::vector<pair_resource> snapshot_;

  bool active_;
  raft::mr::host_resource old_host_;
  raft::mr::device_resource old_device_;

  using host_dry_run_t   = raft::mr::dry_run_resource<raft::mr::host_resource_ref>;
  using device_dry_run_t = raft::mr::dry_run_resource<rmm::device_async_resource_ref>;
  std::unique_ptr<host_dry_run_t> host_adaptor_;
  std::unique_ptr<device_dry_run_t> device_adaptor_;

  using counter_t = raft::mr::detail::dry_run_memory_counter;
  std::shared_ptr<counter_t> host_stats_;
  std::shared_ptr<counter_t> pinned_stats_;
  std::shared_ptr<counter_t> managed_stats_;
  std::shared_ptr<counter_t> ws_stats_;
  std::shared_ptr<counter_t> lws_stats_;
  std::shared_ptr<counter_t> device_stats_;

  void init()
  {
    // Independent-counting invariant
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 1. Force-initialize all lazily-created resources (workspace, large workspace,
    //    pinned, managed) so that their factories resolve against the *original*
    //    global device MR, not a tracking wrapper we install later.
    // 2. Capture every upstream ref while it still points to the original resource.
    // 3. Snapshot the resource map to keep the originals alive.
    // 4. Only *then* replace the global device resource with the tracking bridge.
    // 5. Wrap each captured upstream with a separate dry_run_resource adaptor.
    //
    // Because step 2 happens before step 4, workspace/lws allocations flow through
    // their own adaptor directly to the original device MR, bypassing the device adaptor.
    // Each allocation is therefore counted in exactly one category, and
    // memory_stats::total() returns an accurate, non-overlapping sum.
    auto* ws         = resource::get_workspace_resource(*this);
    auto ws_free     = resource::get_workspace_free_bytes(*this);
    auto ws_upstream = ws->get_upstream_resource();
    auto lws_ref     = resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref  = resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref = resource::get_managed_memory_resource_ref(*this);

    // Snapshot keeps original resource objects alive while dry-run
    // adaptors hold non-owning refs into them.
    snapshot_ = resources_;

    // --- Host (global) ---
    {
      host_adaptor_ = std::make_unique<host_dry_run_t>(raft::mr::host_resource_ref{old_host_});
      host_stats_   = host_adaptor_->get_counter();
      mr::set_default_host_resource(mr::host_resource_ref{*host_adaptor_});
    }

    // --- Pinned ---
    {
      mr::dry_run_resource<mr::host_device_resource_ref> dr{pinned_ref};
      pinned_stats_ = dr.get_counter();
      resource::set_pinned_memory_resource(*this, std::move(dr));
    }

    // --- Managed ---
    {
      mr::dry_run_resource<mr::host_device_resource_ref> dr{managed_ref};
      managed_stats_ = dr.get_counter();
      resource::set_managed_memory_resource(*this, std::move(dr));
    }

    // --- Device (global) ---
    // Invalidate the cached thrust policy (the resource_ref it captured
    // will be stale once we replace the global device resource).
    factories_.at(resource::resource_type::THRUST_POLICY) = std::make_pair(
      resource::resource_type::LAST_KEY, std::make_shared<resource::empty_resource_factory>());
    resources_.at(resource::resource_type::THRUST_POLICY) = std::make_pair(
      resource::resource_type::LAST_KEY, std::make_shared<resource::empty_resource>());
    {
      device_dry_run_t dr{rmm::device_async_resource_ref{old_device_}};
      device_stats_   = dr.get_counter();
      device_adaptor_ = std::make_unique<device_dry_run_t>(std::move(dr));
      rmm::mr::set_current_device_resource(*device_adaptor_);
    }

    // --- Workspace ---
    {
      mr::dry_run_resource<rmm::device_async_resource_ref> dr{ws_upstream};
      ws_stats_ = dr.get_counter();
      resource::set_workspace_resource(*this, std::move(dr), ws_free);
    }

    // --- Large workspace ---
    {
      mr::dry_run_resource<rmm::device_async_resource_ref> dr{lws_ref};
      lws_stats_ = dr.get_counter();
      resource::set_large_workspace_resource(*this, std::move(dr));
    }

    resource::set_dry_run_flag(*this, true);
  }
};

/** @} */

}  // namespace raft

namespace raft::util {

/**
 * @brief Execute an action in dry-run mode and return peak memory usage.
 *
 * Creates an independent copy of the resources handle with all memory resources
 * replaced by dry-run versions, executes the action, and returns peak usage stats.
 *
 * The action receives the dry-run resources handle (as const raft::resources&)
 * and can check the dry-run flag via raft::resource::get_dry_run_flag(res) to
 * skip kernel execution.
 *
 * @tparam Action A callable with signature void(const raft::resources&, Args...).
 * @tparam Args Additional argument types to forward to the action.
 * @param res The raft resources handle.
 * @param action The action to execute in dry-run mode.
 * @param args Additional arguments to forward to the action.
 * @return memory_stats with peak memory usage from the dry run.
 *
 * @code{.cpp}
 * raft::resources res;
 * auto stats = raft::util::dry_run_execute(res, [](const raft::resources& r) {
 *   my_algorithm(r);
 * });
 * std::cout << "Peak workspace: " << stats.device_workspace << " bytes\n";
 * @endcode
 */
template <typename Action, typename... Args>
auto dry_run_execute(const raft::resources& res, Action&& action, Args&&... args)
  -> raft::memory_stats
{
  raft::dry_run_resources dry_res(res);
  std::forward<Action>(action)(static_cast<const raft::resources&>(dry_res),
                               std::forward<Args>(args)...);
  return dry_res.get_bytes_peak();
}

}  // namespace raft::util
