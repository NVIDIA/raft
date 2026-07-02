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
#include <raft/mr/notifying_adaptor.hpp>
#include <raft/mr/recording_adaptor.hpp>
#include <raft/mr/recording_monitor.hpp>
#include <raft/mr/sampling_monitor.hpp>
#include <raft/mr/statistics_adaptor.hpp>

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
 *        allocation-tracking adaptors and logs CSV statistics from a
 *        background thread.
 *
 * Inherits from raft::resources, so it can be passed anywhere a
 * raft::resources& is expected.
 *
 * Two underlying approaches are supported, selected at construction time:
 *  - **Recording approach** (default, no sample_interval): every allocation
 *    and deallocation is pushed as an event onto a thread-safe queue.  The
 *    active NVTX range is captured on the allocating thread at event time, so
 *    range attribution in the CSV is always accurate.  Use when you need
 *    per-event labels.
 *  - **Sampling approach** (pass a sample_interval): a background thread
 *    periodically samples aggregate statistics counters and writes a CSV row.
 *    Lower per-allocation overhead than the recording approach, but NVTX range
 *    association is approximate (whatever range is active when the sampler
 *    is writing).  Use when you want lower invasion and don't need exact
 *    per-event labels.
 *
 * On construction the handle:
 *   - Materializes all tracked resource types (host, device, pinned,
 *     managed, workspace, large_workspace).
 *   - Takes a snapshot of the original resources to keep them alive.
 *   - Wraps each with the chosen adaptor type.
 *      - statistics_adaptor + notifying_adaptor for the sampling approach
 *      - recording_adaptor for the recording approach
 *   - Replaces global host and device resources with tracked versions.
 *   - Starts a background CSV writer.
 *
 * On destruction the handle stops the reporter and restores the
 * global host and device resources.
 */
class memory_tracking_resources : public resources {
 public:
  using duration = std::chrono::steady_clock::duration;

  // -------------------------------------------------------------------------
  // Recording approach (no sample_interval): every event is captured.
  // -------------------------------------------------------------------------

  /**
   * @brief Construct from an existing resources handle, logging to an ostream.
   *        Uses the queue-based recording approach (every allocation captured).
   */
  memory_tracking_resources(const resources& existing, std::ostream& out)
    : memory_tracking_resources(&existing, nullptr, &out)
  {
  }

  /**
   * @brief Construct from an existing resources handle, logging to a file.
   *        Uses the queue-based recording approach (every allocation captured).
   */
  memory_tracking_resources(const resources& existing, const std::string& file_path)
    : memory_tracking_resources(&existing, std::make_unique<std::ofstream>(file_path), nullptr)
  {
  }

  /**
   * @brief Construct from scratch (default resources), logging to an ostream.
   *        Uses the queue-based recording approach (every allocation captured).
   */
  explicit memory_tracking_resources(std::ostream& out) = delete;

  /**
   * @brief Construct from scratch (default resources), logging to a file.
   *        Uses the queue-based recording approach (every allocation captured).
   */
  explicit memory_tracking_resources(const std::string& file_path) = delete;

  // -------------------------------------------------------------------------
  // Sampling approach (with sample_interval): stats sampled periodically.
  // -------------------------------------------------------------------------

  /**
   * @brief Construct from an existing resources handle, logging to an ostream.
   *        Uses the notification/sampling approach.
   *
   * @param sample_interval  Minimum time between successive CSV samples.
   */
  memory_tracking_resources(const resources& existing, std::ostream& out, duration sample_interval)
    : memory_tracking_resources(&existing, nullptr, &out, sample_interval)
  {
  }

  /**
   * @brief Construct from an existing resources handle, logging to a file.
   *        Uses the notification/sampling approach.
   *
   * @param sample_interval  Minimum time between successive CSV samples.
   */
  memory_tracking_resources(const resources& existing,
                            const std::string& file_path,
                            duration sample_interval)
    : memory_tracking_resources(
        &existing, std::make_unique<std::ofstream>(file_path), nullptr, sample_interval)
  {
  }

  /**
   * @brief Construct from scratch (default resources), logging to an ostream.
   *        Uses the notification/sampling approach.
   *
   * @param sample_interval  Minimum time between successive CSV samples.
   */
  memory_tracking_resources(std::ostream& out, duration sample_interval)
    : memory_tracking_resources(nullptr, nullptr, &out, sample_interval)
  {
  }

  /**
   * @brief Construct from scratch (default resources), logging to a file.
   *        Uses the notification/sampling approach.
   *
   * @param sample_interval  Minimum time between successive CSV samples.
   */
  memory_tracking_resources(const std::string& file_path, duration sample_interval)
    : memory_tracking_resources(
        nullptr, std::make_unique<std::ofstream>(file_path), nullptr, sample_interval)
  {
  }

  ~memory_tracking_resources() override
  {
    if (recorder_) recorder_->stop();
    if (sampler_) sampler_->stop();
    raft::mr::set_default_host_resource(old_host_);
    rmm::mr::set_current_device_resource(old_device_);
  }

  memory_tracking_resources(memory_tracking_resources const&)            = delete;
  memory_tracking_resources(memory_tracking_resources&&)                 = delete;
  memory_tracking_resources& operator=(memory_tracking_resources const&) = delete;
  memory_tracking_resources& operator=(memory_tracking_resources&&)      = delete;

  /** @brief Access the recording monitor (non-null for recording approach only). */
  [[nodiscard]] auto get_recorder() noexcept -> raft::mr::recording_monitor*
  {
    return recorder_.get();
  }

  /** @brief Access the sampling monitor (non-null for sampling approach only). */
  [[nodiscard]] auto get_sampler() noexcept -> raft::mr::sampling_monitor*
  {
    return sampler_.get();
  }

 private:
  // Constructor for the recording approach (no sample_interval).
  memory_tracking_resources(const resources* existing,
                            std::unique_ptr<std::ofstream> owned_stream,
                            std::ostream* out_override)
    : resources(existing ? *existing : resources{}),
      owned_stream_(std::move(owned_stream)),
      old_host_(raft::mr::get_default_host_resource()),
      old_device_(rmm::mr::get_current_device_resource_ref())
  {
    std::ostream& out =
      *(out_override ? out_override : static_cast<std::ostream*>(owned_stream_.get()));
    RAFT_LOG_INFO("memory_tracking_resources: using queue-based recording approach");
    recorder_ = std::make_unique<raft::mr::recording_monitor>(out);
    init_recording();
  }

  // Constructor for the sampling approach (with sample_interval).
  memory_tracking_resources(const resources* existing,
                            std::unique_ptr<std::ofstream> owned_stream,
                            std::ostream* out_override,
                            duration sample_interval)
    : resources(existing ? *existing : resources{}),
      owned_stream_(std::move(owned_stream)),
      old_host_(raft::mr::get_default_host_resource()),
      old_device_(rmm::mr::get_current_device_resource_ref())
  {
    std::ostream& out =
      *(out_override ? out_override : static_cast<std::ostream*>(owned_stream_.get()));
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(sample_interval).count();
    RAFT_LOG_INFO("memory_tracking_resources: using sampling approach with interval=%lld us",
                  (long long)us);
    sampler_ = std::make_unique<raft::mr::sampling_monitor>(out, sample_interval);
    init_sampling();
  }

  // Declaration order determines initialization and destruction order.
  // snapshot_ is destroyed last  (keeps original resource shared_ptrs alive).
  // owned_stream_ outlives recorder_/sampler_ (they write to it).
  // recorder_/sampler_ are stopped in the destructor body before member destruction.
  std::vector<pair_resource> snapshot_;
  std::unique_ptr<std::ofstream> owned_stream_;
  std::unique_ptr<raft::mr::recording_monitor> recorder_;
  std::unique_ptr<raft::mr::sampling_monitor> sampler_;

  raft::mr::host_resource old_host_;
  raft::mr::device_resource old_device_;

  // --- Recording approach adaptors (owned because installed as global resources) ---
  using host_record_t   = raft::mr::recording_adaptor<raft::mr::host_resource_ref>;
  using device_record_t = raft::mr::recording_adaptor<rmm::device_async_resource_ref>;
  std::unique_ptr<host_record_t> host_record_adaptor_;
  std::unique_ptr<device_record_t> device_record_adaptor_;

  // --- Sampling approach adaptors (owned because installed as global resources) ---
  using host_stats_t    = raft::mr::statistics_adaptor<raft::mr::host_resource_ref>;
  using host_notify_t   = raft::mr::notifying_adaptor<host_stats_t>;
  using device_stats_t  = raft::mr::statistics_adaptor<rmm::device_async_resource_ref>;
  using device_notify_t = raft::mr::notifying_adaptor<device_stats_t>;
  std::unique_ptr<host_notify_t> host_notify_adaptor_;
  std::unique_ptr<device_notify_t> device_notify_adaptor_;

  void init_recording()
  {
    // Independent-counting invariant: see comment in init_sampling() below.
    auto* ws          = raft::resource::get_workspace_resource(*this);
    auto ws_free      = raft::resource::get_workspace_free_bytes(*this);
    auto upstream_ref = ws->get_upstream_resource();
    auto lws_ref      = raft::resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref   = raft::resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref  = raft::resource::get_managed_memory_resource_ref(*this);

    snapshot_ = resources_;

    auto queue = recorder_->get_queue();

    // Source ids are assigned in registration order, which must match the CSV
    // column-group order below.

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
      // Invalidate the cached thrust policy (the resource_ref it captured
      // will be stale once we replace the global device resource).
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
        *this, raft::mr::recording_adaptor<rmm::device_async_resource_ref>{lws_ref, queue, id});
    }

    recorder_->start();
  }

  void init_sampling()
  {
    // Independent-counting invariant
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // 1. Force-initialize all lazily-created resources (workspace, large workspace,
    //    pinned, managed) so that their factories resolve against the *original*
    //    global device MR, not a tracking wrapper we install later.
    // 2. Capture every upstream ref while it still points to the original resource.
    // 3. Snapshot the resource map to keep the originals alive.
    // 4. Only *then* replace the global device resource with the tracking bridge.
    // 5. Wrap each captured upstream with a separate statistics/notifying adaptor.
    //
    // Because step 2 happens before step 4, workspace/lws allocations flow through
    // their own adaptor directly to the original device MR, bypassing the device adaptor.
    // Each allocation is therefore counted in exactly one category, and
    // memory_stats::total() returns an accurate, non-overlapping sum.
    auto* ws          = raft::resource::get_workspace_resource(*this);
    auto ws_free      = raft::resource::get_workspace_free_bytes(*this);
    auto upstream_ref = ws->get_upstream_resource();
    auto lws_ref      = raft::resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref   = raft::resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref  = raft::resource::get_managed_memory_resource_ref(*this);

    snapshot_ = resources_;

    // --- Host (global) ---
    {
      host_stats_t sa{raft::mr::host_resource_ref{old_host_}};
      sampler_->register_source("host", sa.get_stats());
      host_notify_adaptor_ =
        std::make_unique<host_notify_t>(std::move(sa), sampler_->get_notifier());
      raft::mr::set_default_host_resource(*host_notify_adaptor_);
    }

    // --- Pinned ---
    {
      using stats_t  = raft::mr::statistics_adaptor<raft::mr::host_device_resource_ref>;
      using notify_t = raft::mr::notifying_adaptor<stats_t>;
      stats_t sa{pinned_ref};
      sampler_->register_source("pinned", sa.get_stats());
      raft::resource::set_pinned_memory_resource(*this,
                                                 notify_t{std::move(sa), sampler_->get_notifier()});
    }

    // --- Managed ---
    {
      using stats_t  = raft::mr::statistics_adaptor<raft::mr::host_device_resource_ref>;
      using notify_t = raft::mr::notifying_adaptor<stats_t>;
      stats_t sa{managed_ref};
      sampler_->register_source("managed", sa.get_stats());
      raft::resource::set_managed_memory_resource(
        *this, notify_t{std::move(sa), sampler_->get_notifier()});
    }

    // --- Device (global) ---
    {
      // Invalidate the cached thrust policy (the resource_ref it captured
      // will be stale once we replace the global device resource).
      factories_.at(resource::resource_type::THRUST_POLICY) = std::make_pair(
        resource::resource_type::LAST_KEY, std::make_shared<resource::empty_resource_factory>());
      resources_.at(resource::resource_type::THRUST_POLICY) = std::make_pair(
        resource::resource_type::LAST_KEY, std::make_shared<resource::empty_resource>());
      device_stats_t sa{rmm::device_async_resource_ref{old_device_}};
      sampler_->register_source("device", sa.get_stats());
      device_notify_adaptor_ =
        std::make_unique<device_notify_t>(std::move(sa), sampler_->get_notifier());
      rmm::mr::set_current_device_resource(*device_notify_adaptor_);
    }

    // --- Workspace (track upstream to preserve limiting_resource_adaptor) ---
    {
      using ws_stats_t  = raft::mr::statistics_adaptor<rmm::device_async_resource_ref>;
      using ws_notify_t = raft::mr::notifying_adaptor<ws_stats_t>;
      ws_stats_t sa{upstream_ref};
      sampler_->register_source("workspace", sa.get_stats());
      raft::resource::set_workspace_resource(
        *this, ws_notify_t{std::move(sa), sampler_->get_notifier()}, ws_free);
    }

    // --- Large workspace ---
    {
      using lws_stats_t  = raft::mr::statistics_adaptor<rmm::device_async_resource_ref>;
      using lws_notify_t = raft::mr::notifying_adaptor<lws_stats_t>;
      lws_stats_t sa{lws_ref};
      sampler_->register_source("large_workspace", sa.get_stats());
      raft::resource::set_large_workspace_resource(
        *this, lws_notify_t{std::move(sa), sampler_->get_notifier()});
    }

    sampler_->start();
  }
};

}  // namespace raft
