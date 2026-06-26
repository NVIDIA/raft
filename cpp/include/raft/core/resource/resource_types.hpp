/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
namespace RAFT_EXPORT raft {
namespace resource {

/**
 * @defgroup resource_types Core resource vocabulary types
 * @{
 */

/**
 * @brief Resource types can apply to any resource and don't have to be host- or device-specific.
 */
enum resource_type {
  // device-specific resource types
  CUBLAS_HANDLE = 0,         // cublas handle
  CUSOLVER_DN_HANDLE,        // cusolver dn handle
  CUSOLVER_SP_HANDLE,        // cusolver sp handle
  CUSPARSE_HANDLE,           // cusparse handle
  CUDA_STREAM_VIEW,          // view of a cuda stream
  CUDA_STREAM_POOL,          // cuda stream pool
  CUDA_STREAM_SYNC_EVENT,    // cuda event for syncing streams
  COMMUNICATOR,              // raft communicator
  SUB_COMMUNICATOR,          // raft sub communicator
  DEVICE_PROPERTIES,         // cuda device properties
  DEVICE_ID,                 // cuda device id
  STREAM_VIEW,               // view of a cuda stream or a placeholder in
                             // CUDA-free builds
  THRUST_POLICY,             // thrust execution policy
  WORKSPACE_RESOURCE,        // memory resource for small temporary allocations (bounded)
  CUBLASLT_HANDLE,           // cublasLt handle
  CUSTOM,                    // runtime-shared default-constructible resource
  LARGE_WORKSPACE_RESOURCE,  // memory resource for somewhat large temporary allocations (unbounded)
  NCCL_COMM,                 // nccl comm
  ROOT_RANK,                 // root rank in multi-gpu world
  MULTI_GPU,                 // resource that tracks resource of each device in multi-gpu world
  PINNED_MEMORY_RESOURCE,    // memory resource for pinned (page-locked) host allocations
  MANAGED_MEMORY_RESOURCE,   // resource for managed (unified) allocations
  DRY_RUN_FLAG,              // dry-run mode flag for allocation profiling

  LAST_KEY  // reserved for the last key
};

/**
 * @brief A resource constructs and contains an instance of
 * some pre-determined object type and facades that object
 * behind a common API.
 */
class resource {
 public:
  virtual void* get_resource() = 0;

  virtual ~resource() {}
};

class empty_resource : public resource {
 public:
  empty_resource() : resource() {}

  void* get_resource() override { return nullptr; }

  ~empty_resource() override {}
};

/**
 * @brief A resource factory knows how to construct an instance of
 * a specific raft::resource::resource.
 */
class resource_factory {
 public:
  /**
   * @brief Return the resource_type associated with the current factory
   * @return resource_type corresponding to the current factory
   */
  virtual resource_type get_resource_type() = 0;

  /**
   * @brief Construct an instance of the factory's underlying resource.
   * @return resource instance
   */
  virtual resource* make_resource() = 0;

  virtual ~resource_factory() {}
};

/**
 * @brief A resource factory knows how to construct an instance of
 * a specific raft::resource::resource.
 */
class empty_resource_factory : public resource_factory {
 public:
  empty_resource_factory() : resource_factory() {}
  /**
   * @brief Return the resource_type associated with the current factory
   * @return resource_type corresponding to the current factory
   */
  resource_type get_resource_type() override { return resource_type::LAST_KEY; }

  /**
   * @brief Construct an instance of the factory's underlying resource.
   * @return resource instance
   */
  resource* make_resource() override { return &res; }

 private:
  empty_resource res;
};

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
