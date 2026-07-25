/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __RAFT_DEVICE_RESOURCES
#define __RAFT_DEVICE_RESOURCES

#pragma once

#include <raft/core/comms.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_event.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resource/cusolver_sp_handle.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resource/sub_comms.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime.h>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace RAFT_EXPORT raft {

/**
 * @brief Main resource container object that stores all necessary resources
 * used for calling necessary device functions, cuda kernels and/or libraries
 */
class device_resources : public resources {
 public:
  device_resources(const device_resources& handle,
                   raft::mr::device_resource workspace_resource,
                   std::optional<std::size_t> allocation_limit = std::nullopt)
    : resources{handle}
  {
    resource::set_workspace_resource(*this, std::move(workspace_resource), allocation_limit);
  }

  device_resources(const device_resources&)            = default;
  device_resources(device_resources&&)                 = default;
  device_resources& operator=(const device_resources&) = default;
  device_resources& operator=(device_resources&&)      = default;

  /**
   * @brief Construct a resources instance with a stream view and stream pool
   *
   * @param[in] stream_view the default stream (which has the default per-thread stream if
   * unspecified)
   * @param[in] stream_pool the stream pool used (which has default of nullptr if unspecified)
   * @param[in] workspace_resource an optional resource used by some functions for allocating
   *            temporary workspaces.
   * @param[in] allocation_limit the total amount of memory in bytes available to the temporary
   *            workspace resources.
   */
  device_resources(rmm::cuda_stream_view stream_view                  = rmm::cuda_stream_per_thread,
                   std::shared_ptr<rmm::cuda_stream_pool> stream_pool = {nullptr},
                   std::optional<raft::mr::device_resource> workspace_resource = std::nullopt,
                   std::optional<std::size_t> allocation_limit                 = std::nullopt)
    : resources{}
  {
    resources::add_resource_factory(std::make_shared<resource::device_id_resource_factory>());
    resources::add_resource_factory(
      std::make_shared<resource::cuda_stream_resource_factory>(stream_view));
    resources::add_resource_factory(
      std::make_shared<resource::cuda_stream_pool_resource_factory>(stream_pool));
    if (workspace_resource) {
      resource::set_workspace_resource(*this, std::move(*workspace_resource), allocation_limit);
    }
  }

  /** Destroys all held-up resources */
  virtual ~device_resources() {}

  int get_device() const { return resource::get_device_id(*this); }

  cublasHandle_t get_cublas_handle() const { return resource::get_cublas_handle(*this); }

  cusolverDnHandle_t get_cusolver_dn_handle() const
  {
    return resource::get_cusolver_dn_handle(*this);
  }

  cusolverSpHandle_t get_cusolver_sp_handle() const
  {
    return resource::get_cusolver_sp_handle(*this);
  }

  cusparseHandle_t get_cusparse_handle() const { return resource::get_cusparse_handle(*this); }

  rmm::exec_policy_nosync& get_thrust_policy() const { return resource::get_thrust_policy(*this); }

  /**
   * @brief synchronize a stream on the current container
   */
  void sync_stream(rmm::cuda_stream_view stream) const { resource::sync_stream(*this, stream); }

  /**
   * @brief synchronize main stream on the current container
   */
  void sync_stream() const { resource::sync_stream(*this); }

  /**
   * @brief returns main stream on the current container
   */
  rmm::cuda_stream_view get_stream() const { return resource::get_cuda_stream(*this); }

  /**
   * @brief returns whether stream pool was initialized on the current container
   */

  bool is_stream_pool_initialized() const { return resource::is_stream_pool_initialized(*this); }

  /**
   * @brief returns stream pool on the current container
   */
  const rmm::cuda_stream_pool& get_stream_pool() const
  {
    return resource::get_cuda_stream_pool(*this);
  }

  std::size_t get_stream_pool_size() const { return resource::get_stream_pool_size(*this); }

  /**
   * @brief return stream from pool
   */
  rmm::cuda_stream_view get_stream_from_stream_pool() const
  {
    return resource::get_stream_from_stream_pool(*this);
  }

  /**
   * @brief return stream from pool at index
   */
  rmm::cuda_stream_view get_stream_from_stream_pool(std::size_t stream_idx) const
  {
    return resource::get_stream_from_stream_pool(*this, stream_idx);
  }

  /**
   * @brief return stream from pool if size > 0, else main stream on current container
   */
  rmm::cuda_stream_view get_next_usable_stream() const
  {
    return resource::get_next_usable_stream(*this);
  }

  /**
   * @brief return stream from pool at index if size > 0, else main stream on current container
   *
   * @param[in] stream_idx the required index of the stream in the stream pool if available
   */
  rmm::cuda_stream_view get_next_usable_stream(std::size_t stream_idx) const
  {
    return resource::get_next_usable_stream(*this, stream_idx);
  }

  /**
   * @brief synchronize the stream pool on the current container
   */
  void sync_stream_pool() const { return resource::sync_stream_pool(*this); }

  /**
   * @brief synchronize subset of stream pool
   *
   * @param[in] stream_indices the indices of the streams in the stream pool to synchronize
   */
  void sync_stream_pool(const std::vector<std::size_t> stream_indices) const
  {
    return resource::sync_stream_pool(*this, stream_indices);
  }

  /**
   * @brief ask stream pool to wait on last event in main stream
   */
  void wait_stream_pool_on_stream() const { return resource::wait_stream_pool_on_stream(*this); }

  void set_comms(std::shared_ptr<comms::comms_t> communicator)
  {
    resource::set_comms(*this, communicator);
  }

  const comms::comms_t& get_comms() const { return resource::get_comms(*this); }

  void set_subcomm(std::string key, std::shared_ptr<comms::comms_t> subcomm)
  {
    resource::set_subcomm(*this, key, subcomm);
  }

  const comms::comms_t& get_subcomm(std::string key) const
  {
    return resource::get_subcomm(*this, key);
  }

  rmm::mr::limiting_resource_adaptor* get_workspace_resource() const
  {
    return resource::get_workspace_resource(*this);
  }

  bool comms_initialized() const { return resource::comms_initialized(*this); }

  const cudaDeviceProp& get_device_properties() const
  {
    return resource::get_device_properties(*this);
  }
};  // class device_resources

/**
 * @brief RAII approach to synchronizing across all streams in the current container
 */
class stream_syncer {
 public:
  explicit stream_syncer(const device_resources& handle) : handle_(handle)
  {
    resource::sync_stream(handle_);
  }
  ~stream_syncer()
  {
    handle_.wait_stream_pool_on_stream();
    handle_.sync_stream_pool();
  }

  stream_syncer(const stream_syncer& other)            = delete;
  stream_syncer& operator=(const stream_syncer& other) = delete;

 private:
  const device_resources& handle_;
};  // class stream_syncer

}  // namespace RAFT_EXPORT raft

#endif
