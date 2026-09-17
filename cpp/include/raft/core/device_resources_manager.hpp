/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_setter.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <algorithm>
#include <memory>
#include <optional>

namespace RAFT_EXPORT raft {

/**
 * @brief A singleton used to easily generate a raft::device_resources object
 *
 * Many calls to RAFT functions require a `raft::device_resources` object
 * to provide CUDA resources like streams and stream pools. The
 * `raft::device_resources_manager` singleton provides a straightforward method to create those
 * objects in a way that allows consumers of RAFT to limit total consumption of device resources
 * without actively managing streams or other CUDA-specific objects.
 *
 * To control the resources a consuming application will use, the
 * resource manager provides setters for a variety of values. For
 * instance, to ensure that no more than `N` CUDA streams are used per
 * device, a consumer might call
 * `raft::device_resources_manager::set_streams_per_device(N)`. Note that all of these
 * setters must be used prior to retrieving the first `device_resources` from
 * the manager. Setters invoked after this will log a warning but have no
 * effect.
 *
 * After calling all desired setters, consumers can simply call
 * `auto res = raft::device_resources_manager::get_device_resources();` to get a valid
 * device_resources object for the current device based on previously-set
 * parameters. Importantly, calling `get_device_resources()` again from the same
 * thread is guaranteed to return a `device_resources` object with the same
 * underlying CUDA stream and (if a non-zero number of stream pools has been
 * requested) stream pool.
 *
 * Typical usage might look something like the following:
 * @code
 * void initialize_application() {
 *   raft::device_resources_manager::set_streams_per_device(16);
 * }
 *
 * void foo_called_from_multiple_threads() {
 *   auto res = raft::device_resources_manager::get_device_resources();
 *   // Call RAFT function using res
 *   res.sync_stream() // Ensure work completes before returning
 * }
 * @endcode
 *
 * Note that all public methods of the `device_resources_manager` are thread-safe,
 * but the manager is designed to minimize locking required for
 * retrieving `device_resources` objects. Each thread must acquire a lock
 * exactly once per device when calling `get_device_resources`. Subsequent calls
 * will still be thread-safe but will not require a lock.
 *
 * All public methods of the `device_resources_manager` are static. Please see
 * documentation of those methods for additional usage information.
 *
 */
struct device_resources_manager {
  device_resources_manager(device_resources_manager const&) = delete;
  void operator=(device_resources_manager const&)           = delete;

 private:
  device_resources_manager() {}
  ~device_resources_manager()
  {
    // Ensure that we destroy any pool memory resources before CUDA context is
    // lost
    per_device_components_.clear();
  }

  // Get an id used to identify this thread for the purposes of assigning
  // (in round-robin fashion) the same resources to the thread on subsequent calls to
  // `get_device_resources`
  static auto get_thread_id()
  {
    static std::atomic<std::size_t> thread_counter{};
    thread_local std::size_t id = ++thread_counter;
    return id;
  }

  // This struct holds the various parameters used to control
  // construction of the underlying resources shared by all
  // `device_resources` objects returned by `get_device_resources`
  struct resource_params {
    // The total number of primary streams to be used by the
    // application. If no value is provided, the default stream per thread
    // is used.
    std::optional<std::size_t> stream_count{std::nullopt};
    // The total number of stream pools to be used by the application
    std::size_t pool_count{};
    // How many streams to assign to each pool
    std::size_t pool_size{rmm::cuda_stream_pool::default_size};
    // If a memory pool is requested (max_mem_pool_size is non-zero), use
    // this initial size for the pool in bytes. Must be a multiple of 256.
    // If nullopt, use half of the available memory on the current
    // device.
    std::optional<std::size_t> init_mem_pool_size{std::nullopt};
    // If set to any non-zero value, create a memory pool with this
    // maximum size. If nullopt, use up to the entire available memory of the
    // device
    std::optional<std::size_t> max_mem_pool_size{std::size_t{}};
    // Limit on workspace memory for the returned device_resources object
    std::optional<std::size_t> workspace_allocation_limit{std::nullopt};
    // Optional specification of separate workspace memory resources for each
    // device. The integer in each pair indicates the device for this memory
    // resource.
    std::vector<std::pair<raft::mr::device_resource, int>> workspace_mrs{};
  } params_;

  // This struct stores the underlying resources to be shared among
  // `device_resources` objects returned by this manager.
  struct resource_components {
    // Construct all underlying resources indicated by `params` for the
    // indicated device. This includes primary streams, stream pools, and
    // a memory pool if requested.
    resource_components(int device_id, resource_params const& params)
      : device_id_{device_id},
        streams_{[&params, this]() {
          auto scoped_device = device_setter{device_id_};
          auto result        = std::unique_ptr<rmm::cuda_stream_pool>{nullptr};
          if (params.stream_count) {
            result = std::make_unique<rmm::cuda_stream_pool>(*params.stream_count);
          }
          return result;
        }()},
        pools_{[&params, this]() {
          auto scoped_device = device_setter{device_id_};
          auto result        = std::vector<std::shared_ptr<rmm::cuda_stream_pool>>{};
          if (params.pool_size != 0) {
            for (auto i = std::size_t{}; i < params.pool_count; ++i) {
              result.push_back(std::make_shared<rmm::cuda_stream_pool>(params.pool_size));
            }
          } else if (params.pool_count != 0) {
            RAFT_LOG_WARN("Stream pools of size 0 requested; no pools will be created");
          }
          return result;
        }()},
        pool_mr_{[&params, this]() {
          auto scoped_device = device_setter{device_id_};
          auto result        = std::optional<rmm::mr::pool_memory_resource>{};
          // If max_mem_pool_size is nullopt or non-zero, create a pool memory
          // resource
          if (params.max_mem_pool_size.value_or(1) != 0) {
            auto upstream = rmm::mr::get_current_device_resource_ref();
            if (cuda::mr::resource_cast<rmm::mr::cuda_memory_resource>(&upstream) != nullptr) {
              result.emplace(
                upstream,
                params.init_mem_pool_size.value_or(rmm::percent_of_free_device_memory(50)),
                params.max_mem_pool_size);
              rmm::mr::set_current_device_resource(*result);
            } else {
              RAFT_LOG_WARN(
                "Pool allocation requested, but other memory resource has already been set and "
                "will not be overwritten");
            }
          }
          return result;
        }()},
        workspace_mr_{[&params, this]() {
          auto result = std::optional<raft::mr::device_resource>{};
          auto iter   = std::find_if(std::begin(params.workspace_mrs),
                                   std::end(params.workspace_mrs),
                                   [this](auto&& pair) { return pair.second == device_id_; });
          if (iter != std::end(params.workspace_mrs)) { result = iter->first; }
          return result;
        }()},
        workspace_allocation_limit_{params.workspace_allocation_limit}
    {
    }

    // Get the id of the device associated with the constructed resource
    // components
    [[nodiscard]] auto get_device_id() const { return device_id_; }
    // Get the total number of streams available for this application
    [[nodiscard]] auto stream_count() const
    {
      auto result = std::size_t{};
      if (streams_) { result = streams_->get_pool_size(); }
      return result;
    }
    // Get the stream assigned to this host thread. Note that the same stream
    // may be used by multiple threads, but any given thread will always use
    // the same stream
    [[nodiscard]] auto get_stream() const
    {
      auto result = cuda::stream_ref{cudaStreamPerThread};
      if (stream_count() != 0) { result = streams_->get_stream(get_thread_id() % stream_count()); }
      return result;
    }
    // Get the total number of stream pools available for this
    // application
    [[nodiscard]] auto pool_count() const { return pools_.size(); }
    // Get the stream pool assigned to this host thread. Note that the same stream pool
    // may be used by multiple threads, but any given thread will always use
    // the same stream pool
    [[nodiscard]] auto get_pool() const
    {
      auto result = std::shared_ptr<rmm::cuda_stream_pool>{nullptr};
      if (pool_count() != 0) { result = pools_[get_thread_id() % pool_count()]; }
      return result;
    }
    // Return the pool memory resource created for this device by the manager (if any)
    [[nodiscard]] auto& get_pool_memory_resource() { return pool_mr_; }
    // Return the RAFT workspace allocation limit that will be used by
    // `device_resources` returned from this manager
    [[nodiscard]] auto get_workspace_allocation_limit() const
    {
      return workspace_allocation_limit_;
    }
    // Return the workspace memory resource for this device (if any)
    [[nodiscard]] auto& get_workspace_memory_resource() { return workspace_mr_; }

   private:
    int device_id_;
    std::unique_ptr<rmm::cuda_stream_pool> streams_;
    std::vector<std::shared_ptr<rmm::cuda_stream_pool>> pools_;
    std::optional<rmm::mr::pool_memory_resource> pool_mr_;
    std::optional<raft::mr::device_resource> workspace_mr_;
    std::optional<std::size_t> workspace_allocation_limit_{std::nullopt};
  };

  // Mutex used to lock access to shared data until after the first
  // `get_device_resources` call in each thread
  mutable std::mutex manager_mutex_{};
  // Indicates whether or not `get_device_resources` has been called by any
  // host thread
  bool params_finalized_{};
  // Container for underlying device resources to be re-used across host
  // threads for each device
  std::vector<resource_components> per_device_components_;

  // Return a lock for accessing shared data
  [[nodiscard]] auto get_lock() const { return std::unique_lock{manager_mutex_}; }

  // Retrieve the underlying resources to be shared across the
  // application for the indicated device. This method acquires a lock the
  // first time it is called in each thread for a specific device to ensure that the
  // underlying resources have been correctly initialized exactly once across
  // all host threads.
  auto const& get_device_resources_(int device_id)
  {
    thread_local auto thread_resources = std::vector<std::optional<raft::device_resources>>([]() {
      auto result = 0;
      RAFT_CUDA_TRY(cudaGetDeviceCount(&result));
      RAFT_EXPECTS(result != 0, "No CUDA devices found");
      return result;
    }());
    if (!thread_resources[device_id]) {
      // Only lock if we have not previously accessed this device on this
      // thread
      auto lock = get_lock();
      // If we are building components, do not allow any further changes to
      // resource parameters.
      params_finalized_ = true;

      // Even if we have not yet built device_resources for the current
      // device, we may have already built the underlying components, since
      // multiple device_resources may point to the same components.
      auto component_iter = std::find_if(
        std::begin(per_device_components_),
        std::end(per_device_components_),
        [device_id](auto&& components) { return components.get_device_id() == device_id; });

      if (component_iter == std::end(per_device_components_)) {
        // Build components for this device if we have not yet done so on
        // another thread
        per_device_components_.emplace_back(device_id, params_);
        component_iter = std::prev(std::end(per_device_components_));
      }
      auto scoped_device = device_setter(device_id);
      // Build the device_resources object for this thread out of shared
      // components
      thread_resources[device_id].emplace(component_iter->get_stream(),
                                          component_iter->get_pool(),
                                          component_iter->get_workspace_memory_resource(),
                                          component_iter->get_workspace_allocation_limit());
    }

    return thread_resources[device_id].value();
  }

  // Thread-safe setter for the number of streams
  void set_streams_per_device_(std::optional<std::size_t> num_streams)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set device_resources_manager properties after resources have already been "
        "retrieved");
    } else {
      params_.stream_count = num_streams;
    }
  }

  // Thread-safe setter for the number and size of stream pools
  void set_stream_pools_per_device_(std::size_t num_pools, std::size_t num_streams)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set device_resources_manager properties after resources have already been "
        "retrieved");
    } else {
      params_.pool_count = num_pools;
      params_.pool_size  = num_streams;
    }
  }

  // Thread-safe setter for the RAFT workspace allocation limit
  void set_workspace_allocation_limit_(std::size_t memory_limit)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set device_resources_manager properties after resources have already been "
        "retrieved");
    } else {
      params_.workspace_allocation_limit.emplace(memory_limit);
    }
  }

  // Thread-safe setter for the maximum memory pool size
  void set_max_mem_pool_size_(std::optional<std::size_t> memory_limit)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set device_resources_manager properties after resources have already been "
        "retrieved");
    } else {
      if (memory_limit) {
        params_.max_mem_pool_size.emplace(*memory_limit);
      } else {
        params_.max_mem_pool_size = std::nullopt;
      }
    }
  }

  // Thread-safe setter for the initial memory pool size
  void set_init_mem_pool_size_(std::optional<std::size_t> init_memory)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set device_resources_manager properties after resources have already been "
        "retrieved");
    } else {
      if (init_memory) {
        params_.init_mem_pool_size.emplace(*init_memory);
      } else {
        params_.init_mem_pool_size = std::nullopt;
      }
    }
  }

  // Thread-safe setter for workspace memory resources
  void set_workspace_memory_resource_(raft::mr::device_resource mr, int device_id)
  {
    auto lock = get_lock();
    if (params_finalized_) {
      RAFT_LOG_WARN(
        "Attempted to set device_resources_manager properties after resources have already been "
        "retrieved");
    } else {
      auto iter = std::find_if(std::begin(params_.workspace_mrs),
                               std::end(params_.workspace_mrs),
                               [device_id](auto&& pair) { return pair.second == device_id; });
      if (iter != std::end(params_.workspace_mrs)) {
        iter->first = std::move(mr);
      } else {
        params_.workspace_mrs.emplace_back(std::move(mr), device_id);
      }
    }
  }

  // Retrieve the instance of this singleton
  static auto& get_manager()
  {
    static auto manager = device_resources_manager{};
    return manager;
  }

 public:
  /**
   * @brief Retrieve device_resources to be used with the RAFT API
   *
   * This thread-safe method ensures that a `device_resources` object with
   * the same underlying stream and stream pool is returned every time it is
   * called by the same host thread. This means that if `get_device_resources` is
   * used to provide all `device_resources` in an application, then
   * `raft::get_device_resources().sync_stream()` and (if a stream pool is used)
   * raft::get_device_resources().sync_stream_pool() are guaranteed to synchronize all
   * work previously submitted to the device by this host thread.
   *
   * If the max memory pool size set with `set_max_mem_pool_size` is non-zero,
   * the first call of this method will also create a memory pool to be used
   * for all RMM-based allocations on device.
   *
   * @param device_id int If provided, the device for which resources should
   * be returned. Defaults to active CUDA device.
   */
  static auto const& get_device_resources(int device_id = device_setter::get_current_device())
  {
    return get_manager().get_device_resources_(device_id);
  }

  /**
   * @brief Set the total number of CUDA streams to be used per device
   *
   * If nullopt, the default stream per thread will be used
   * (essentially allowing as many streams as there are host threads).
   * Otherwise, all returned `device_resources` will draw their streams from this
   * limited pool.
   *
   * Limiting the total number of streams can be desirable for a number of
   * reasons, but it is most often used in consuming applications to
   * prevent a large number of host threads from flooding the device with
   * simultaneous requests that may exhaust device memory or other
   * resources.
   *
   * If called after the first call to
   * `raft::device_resources_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_streams_per_device(std::optional<std::size_t> num_streams)
  {
    get_manager().set_streams_per_device_(num_streams);
  }

  /**
   * @brief Set the total number and size of CUDA stream pools to be used per device
   *
   * Setting the number of stream pools to a non-zero value will provide a
   * pool of stream pools that can be shared among host threads. This can be
   * useful for the same reason it is useful to limit the total number of
   * primary streams assigned to `device_resoures` for each host thread.
   * Repeated calls to `get_device_resources` on a given host thread are
   * guaranteed to return `device_resources` with the same underlying stream
   * pool.
   *
   * If called after the first call to
   * `raft::device_resources_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_stream_pools_per_device(
    std::size_t num_pools, std::size_t num_streams = rmm::cuda_stream_pool::default_size)
  {
    get_manager().set_stream_pools_per_device_(num_pools, num_streams);
  }
  /**
   * @brief Set the maximum size of temporary RAFT workspaces
   *
   * Note that this limits only the size of temporary workspace
   * allocations. To cap the device memory generally available for all device
   * allocations made with RMM, use
   * `raft::device_manager::set_max_mem_pool_size`
   *
   * If called after the first call to
   * `raft::device_resources_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_workspace_allocation_limit(std::size_t memory_limit)
  {
    get_manager().set_workspace_allocation_limit_(memory_limit);
  }

  /**
   * @brief Set the maximum size of the device memory pool
   *
   * If set to 0, no memory pool will be used. If set to nullopt, the memory
   * pool is allowed to grow to the size of available device memory.
   *
   * Note that the pool will not actually be created until the first call
   * to `raft::device_manager::get_device_resources(device_id)`, after which it will become
   * the current RMM device memory resource for the indicated device. If the
   * current RMM device memory resource has already been set to some
   * non-default resource, no pool resource will be created and a warning will be emitted. It is
   * assumed that applications which have set a memory resource already wish to manage RMM
   * themselves.
   *
   * If called after the first call to
   * `raft::device_resources_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_max_mem_pool_size(std::optional<std::size_t> max_mem)
  {
    get_manager().set_max_mem_pool_size_(max_mem);
  }

  /**
   * @brief Set the initial size of the device memory pool
   *
   * If set to nullopt, the memory pool starts with half of the available
   * device memory.
   *
   * If called after the first call to
   * `raft::device_resources_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_init_mem_pool_size(std::optional<std::size_t> init_mem)
  {
    get_manager().set_init_mem_pool_size_(init_mem);
  }
  /**
   * @brief Request a device memory pool with specified parameters
   *
   * This convenience method essentially combines
   * `set_init_mem_pool_size` and `set_max_mem_pool_size`. It is provided
   * primarily to allow users who want a memory pool but do not want to choose
   * specific pool sizes to simply call
   * `raft::device_manager::set_memory_pool()` and enable a memory pool using
   * RMM defaults (initialize with half of available memory, allow to grow
   * to all available memory).
   *
   * If called after the first call to
   * `raft::device_resources_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_mem_pool(std::optional<std::size_t> init_mem = std::nullopt,
                           std::optional<std::size_t> max_mem  = std::nullopt)
  {
    set_init_mem_pool_size(init_mem);
    set_max_mem_pool_size(max_mem);
  }

  /**
   * @brief Set the workspace memory resource to be used on a specific device
   *
   * RAFT device_resources objects can be built with a separate memory
   * resource for allocating temporary workspaces. If a memory
   * resource is provided by this setter, it will be used as the
   * workspace memory resource for all `device_resources` returned for the
   * indicated device.
   *
   * If called after the first call to
   * `raft::device_resources_manager::get_device_resources`, no change will be made,
   * and a warning will be emitted.
   */
  static void set_workspace_memory_resource(raft::mr::device_resource mr,
                                            int device_id = device_setter::get_current_device())
  {
    get_manager().set_workspace_memory_resource_(std::move(mr), device_id);
  }
};
}  // namespace RAFT_EXPORT raft
