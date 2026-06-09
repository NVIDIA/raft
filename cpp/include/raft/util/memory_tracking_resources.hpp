/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/allocation_event_monitor.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/mr/host_memory_resource.hpp>
#include <raft/mr/recording_adaptor.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <chrono>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>

namespace raft {

/**
 * @brief A resources handle that wraps all reachable memory resources with
 *        allocation-recording adaptors and logs CSV statistics from a
 *        background thread.
 *
 * Inherits from raft::resources, so it can be passed anywhere a
 * raft::resources& is expected.  On construction the handle:
 *   - Materializes all tracked resource types (host, device, pinned,
 *     managed, workspace, large_workspace).
 *   - Takes a snapshot of the original resources to keep them alive.
 *   - Wraps each with a recording_adaptor that pushes an allocation_event
 *     (carrying the NVTX range captured at allocation time) onto a shared queue.
 *   - Replaces global host and device resources with tracked versions.
 *   - Starts a background CSV writer that drains the queue.
 *
 * On destruction the handle stops the writer (draining all pending events) and
 * restores the global host and device resources.
 *
 * Unlike a sampling monitor, the NVTX range is captured on the allocating
 * thread at event time, so range attribution in the CSV is always correct.
 */
class memory_tracking_resources : public resources {
 public:
  using duration = std::chrono::steady_clock::duration;

  /**
   * @brief Construct from an existing resources handle, logging to an ostream.
   *
   * @param existing  Resources to shallow-copy and wrap with tracking.
   * @param out       Output stream for CSV rows (must outlive this object).
   * @param sample_interval  Accepted for API compatibility; unused by the
   *                         event-driven monitor (every event is recorded).
   */
  memory_tracking_resources(const resources& existing,
                            std::ostream& out,
                            duration sample_interval = std::chrono::milliseconds{10})
    : memory_tracking_resources(&existing, nullptr, &out, sample_interval)
  {
  }

  /**
   * @brief Construct from an existing resources handle, logging to a file.
   *
   * @param existing  Resources to shallow-copy and wrap with tracking.
   * @param file_path Path to the output CSV file (created/truncated).
   * @param sample_interval  Accepted for API compatibility; unused.
   */
  memory_tracking_resources(const resources& existing,
                            const std::string& file_path,
                            duration sample_interval = std::chrono::milliseconds{10})
    : memory_tracking_resources(
        &existing, std::make_unique<std::ofstream>(file_path), nullptr, sample_interval)
  {
  }

  /**
   * @brief Construct from scratch (default resources), logging to an ostream.
   *
   * @param out       Output stream for CSV rows (must outlive this object).
   * @param sample_interval  Accepted for API compatibility; unused.
   */
  explicit memory_tracking_resources(std::ostream& out,
                                     duration sample_interval = std::chrono::milliseconds{10})
    : memory_tracking_resources(nullptr, nullptr, &out, sample_interval)
  {
  }

  /**
   * @brief Construct from scratch (default resources), logging to a file.
   *
   * @param file_path Path to the output CSV file (created/truncated).
   * @param sample_interval  Accepted for API compatibility; unused.
   */
  explicit memory_tracking_resources(const std::string& file_path,
                                     duration sample_interval = std::chrono::milliseconds{10})
    : memory_tracking_resources(
        nullptr, std::make_unique<std::ofstream>(file_path), nullptr, sample_interval)
  {
  }

  ~memory_tracking_resources() override
  {
    report_.stop();
    raft::mr::set_default_host_resource(old_host_ref_);
    rmm::mr::set_current_device_resource(old_device_ref_);
  }

  memory_tracking_resources(memory_tracking_resources const&)            = delete;
  memory_tracking_resources(memory_tracking_resources&&)                 = delete;
  memory_tracking_resources& operator=(memory_tracking_resources const&) = delete;
  memory_tracking_resources& operator=(memory_tracking_resources&&)      = delete;

  /** @brief Access the underlying CSV writer. */
  [[nodiscard]] auto report() noexcept -> raft::mr::allocation_event_monitor& { return report_; }

 private:
  memory_tracking_resources(const resources* existing,
                            std::unique_ptr<std::ofstream> owned_stream,
                            std::ostream* out_override,
                            [[maybe_unused]] duration sample_interval)
    : resources(existing ? *existing : resources{}),
      owned_stream_(std::move(owned_stream)),
      report_(out_override ? *out_override : *owned_stream_),
      old_host_ref_(raft::mr::get_default_host_resource()),
      old_device_ref_(rmm::mr::get_current_device_resource_ref())
  {
    init();
  }

  // Declaration order determines initialization and destruction order.
  // snapshot_ is destroyed last  (keeps original resource shared_ptrs alive).
  // owned_stream_ outlives report_ (report_ writes to it).
  // report_ is destroyed first of the three (stops the background thread).
  std::vector<pair_resource> snapshot_;
  std::unique_ptr<std::ofstream> owned_stream_;
  raft::mr::allocation_event_monitor report_;

  raft::mr::host_resource_ref old_host_ref_;
  rmm::device_async_resource_ref old_device_ref_;
  std::size_t saved_ws_limit_{};

  // Host and device adaptors are installed as the *global* resources, which
  // hold them by reference, so they must outlive this object's use -> owned here.
  using host_adaptor_t   = raft::mr::recording_adaptor<raft::mr::host_resource_ref>;
  using device_adaptor_t = raft::mr::recording_adaptor<rmm::device_async_resource_ref>;
  std::unique_ptr<host_adaptor_t> host_adaptor_;
  std::unique_ptr<device_adaptor_t> device_adaptor_;

  void init()
  {
    auto* ws          = raft::resource::get_workspace_resource(*this);
    saved_ws_limit_   = ws->get_allocation_limit();
    auto upstream_ref = ws->get_upstream_resource();
    auto lws_ref      = raft::resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref   = raft::resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref  = raft::resource::get_managed_memory_resource_ref(*this);

    // Keeps original resource objects alive while tracking refs point into them.
    snapshot_ = resources_;

    auto queue = report_.get_queue();

    // Source ids are assigned in registration order, which must match the CSV
    // column-group order below.

    // --- Host (global) ---
    {
      int id        = report_.register_source("host");
      host_adaptor_ = std::make_unique<host_adaptor_t>(old_host_ref_, queue, id);
      raft::mr::set_default_host_resource(*host_adaptor_);
    }

    // --- Pinned ---
    {
      int id = report_.register_source("pinned");
      raft::resource::set_pinned_memory_resource(
        *this, raft::mr::recording_adaptor<raft::mr::host_device_resource_ref>{pinned_ref, queue, id});
    }

    // --- Managed ---
    {
      int id = report_.register_source("managed");
      raft::resource::set_managed_memory_resource(
        *this,
        raft::mr::recording_adaptor<raft::mr::host_device_resource_ref>{managed_ref, queue, id});
    }

    // --- Device (global) ---
    {
      int id          = report_.register_source("device");
      device_adaptor_ = std::make_unique<device_adaptor_t>(old_device_ref_, queue, id);
      rmm::mr::set_current_device_resource(*device_adaptor_);
    }

    // --- Workspace (track upstream to preserve limiting_resource_adaptor) ---
    {
      int id = report_.register_source("workspace");
      raft::resource::set_workspace_resource(
        *this,
        raft::mr::recording_adaptor<rmm::device_async_resource_ref>{upstream_ref, queue, id},
        saved_ws_limit_);
    }

    // --- Large workspace ---
    {
      int id = report_.register_source("large_workspace");
      raft::resource::set_large_workspace_resource(
        *this, raft::mr::recording_adaptor<rmm::device_async_resource_ref>{lws_ref, queue, id});
    }

    report_.start();
  }
};

}  // namespace raft
