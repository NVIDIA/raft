/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/mr/host_memory_resource.hpp>
#include <raft/mr/recording_adaptor.hpp>
#include <raft/mr/recording_monitor.hpp>

#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <fstream>
#include <memory>
#include <ostream>
#include <string>

namespace raft {

/**
 * @brief A resources handle that wraps all reachable memory resources with
 *        recording_adaptor and logs every allocation/deallocation as a CSV row
 *        from a background thread.
 *
 * Inherits from raft::resources, so it can be passed anywhere a
 * raft::resources& is expected.
 *
 * Every allocation and deallocation is pushed as an event onto a thread-safe
 * queue.  The active NVTX range path is captured from the allocating/deallocating
 * thread at the moment of the event — no mutex is taken on the NVTX stack because
 * the read is always on the owning thread.  The per-pointer alloc_map that maps
 * addresses to their allocation-time NVTX path still uses a mutex because
 * deallocation can legitimately occur on a different thread than allocation.
 *
 * On construction the handle:
 *   - Materializes all tracked resource types (host, device, pinned,
 *     managed, workspace, large_workspace).
 *   - Takes a snapshot of the original resources to keep them alive.
 *   - Wraps each with a recording_adaptor.
 *   - Replaces global host and device resources with tracked versions.
 *   - Starts a background CSV writer (recording_monitor).
 *
 * On destruction the handle stops the monitor and restores the original
 * global host and device resources.
 */
class memory_logging_resources : public resources {
 public:
  /**
   * @brief Construct from an existing resources handle, logging to an ostream.
   *        Every allocation and deallocation produces one CSV row.
   */
  memory_logging_resources(const resources& existing, std::ostream& out)
    : memory_logging_resources(&existing, nullptr, &out)
  {
  }

  /**
   * @brief Construct from an existing resources handle, logging to a file.
   *        Every allocation and deallocation produces one CSV row.
   */
  memory_logging_resources(const resources& existing, const std::string& file_path)
    : memory_logging_resources(&existing, std::make_unique<std::ofstream>(file_path), nullptr)
  {
  }

  ~memory_logging_resources() override
  {
    if (recorder_) recorder_->stop();
    raft::mr::set_default_host_resource(old_host_);
    rmm::mr::set_current_device_resource(old_device_);
  }

  memory_logging_resources(memory_logging_resources const&)            = delete;
  memory_logging_resources(memory_logging_resources&&)                 = delete;
  memory_logging_resources& operator=(memory_logging_resources const&) = delete;
  memory_logging_resources& operator=(memory_logging_resources&&)      = delete;

  /** @brief Access the recording monitor (always non-null after construction). */
  [[nodiscard]] auto get_recorder() noexcept -> raft::mr::recording_monitor*
  {
    return recorder_.get();
  }

 private:
  memory_logging_resources(const resources* existing,
                            std::unique_ptr<std::ofstream> owned_stream,
                            std::ostream* out_override)
    : resources(existing ? *existing : resources{}),
      owned_stream_(std::move(owned_stream)),
      old_host_(raft::mr::get_default_host_resource()),
      old_device_(rmm::mr::get_current_device_resource_ref())
  {
    std::ostream* outp = out_override;
    if (!outp) { outp = static_cast<std::ostream*>(owned_stream_.get()); }
    RAFT_LOG_INFO("memory_logging_resources: queue-based recording (every event captured)");
    recorder_ = std::make_unique<raft::mr::recording_monitor>(*outp);
    init_recording();
  }

  // Declaration order matters: snapshot_ is destroyed last (keeps original resource
  // shared_ptrs alive); owned_stream_ outlives recorder_ (it writes to it);
  // recorder_ is stopped in the destructor body before member destruction.
  std::vector<pair_resource> snapshot_;
  std::unique_ptr<std::ofstream> owned_stream_;
  std::unique_ptr<raft::mr::recording_monitor> recorder_;

  raft::mr::host_resource old_host_;
  raft::mr::device_resource old_device_;

  using host_record_t   = raft::mr::recording_adaptor<raft::mr::host_resource_ref>;
  using device_record_t = raft::mr::recording_adaptor<rmm::device_async_resource_ref>;
  std::unique_ptr<host_record_t> host_record_adaptor_;
  std::unique_ptr<device_record_t> device_record_adaptor_;

  void init_recording()
  {
    // Force-initialize lazily-created resources before we replace the global device MR,
    // so their upstreams resolve against the original resource.
    auto* ws          = raft::resource::get_workspace_resource(*this);
    auto ws_free      = raft::resource::get_workspace_free_bytes(*this);
    auto upstream_ref = ws->get_upstream_resource();
    auto lws_ref      = raft::resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref   = raft::resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref  = raft::resource::get_managed_memory_resource_ref(*this);

    snapshot_ = resources_;

    auto queue = recorder_->get_queue();

    // Source ids are assigned in registration order and must match the CSV column-group order.

    // --- Host (global) ---
    {
      int id               = recorder_->register_source("host");
      host_record_adaptor_ = std::make_unique<host_record_t>(old_host_, queue, id);
      raft::mr::set_default_host_resource(*host_record_adaptor_);
    }

    // --- Pinned ---
    {
      int id = recorder_->register_source("pinned");
      raft::resource::set_pinned_memory_resource(
        *this,
        raft::mr::recording_adaptor<raft::mr::host_device_resource_ref>{pinned_ref, queue, id});
    }

    // --- Managed ---
    {
      int id = recorder_->register_source("managed");
      raft::resource::set_managed_memory_resource(
        *this,
        raft::mr::recording_adaptor<raft::mr::host_device_resource_ref>{managed_ref, queue, id});
    }

    // --- Device (global) ---
    {
      // Invalidate the cached thrust policy — its resource_ref will be stale
      // once we replace the global device resource.
      factories_.at(resource::resource_type::THRUST_POLICY) = std::make_pair(
        resource::resource_type::LAST_KEY, std::make_shared<resource::empty_resource_factory>());
      resources_.at(resource::resource_type::THRUST_POLICY) = std::make_pair(
        resource::resource_type::LAST_KEY, std::make_shared<resource::empty_resource>());
      int id                 = recorder_->register_source("device");
      device_record_adaptor_ = std::make_unique<device_record_t>(old_device_, queue, id);
      rmm::mr::set_current_device_resource(*device_record_adaptor_);
    }

    // --- Workspace (track upstream to preserve limiting_resource_adaptor) ---
    {
      int id = recorder_->register_source("workspace");
      raft::resource::set_workspace_resource(
        *this,
        raft::mr::recording_adaptor<rmm::device_async_resource_ref>{upstream_ref, queue, id},
        ws_free);
    }

    // --- Large workspace ---
    {
      int id = recorder_->register_source("large_workspace");
      raft::resource::set_large_workspace_resource(
        *this,
        raft::mr::recording_adaptor<rmm::device_async_resource_ref>{lws_ref, queue, id});
    }

    recorder_->start();
  }
};

}  // namespace raft
