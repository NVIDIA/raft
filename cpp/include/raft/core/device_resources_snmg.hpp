/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream_pool.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resource/resource_types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream_pool.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstddef>
#include <memory>
#include <unordered_set>
#include <vector>

namespace RAFT_EXPORT raft {

const std::unordered_set<resource::resource_type> snmg_related_resources = {
  raft::resource::MULTI_GPU, raft::resource::NCCL_COMM, raft::resource::ROOT_RANK};

/**
 * @brief SNMG (single-node multi-GPU) resource container object that tracks resources for each GPU.
 * Note the `device_resources_snmg` object can also be used as a classic
 * `device_resources` object. The associated resources will be the ones of the GPU used during
 * object instantiation and a GPU switch operation will be ordered during the retrieval of said
 * resources.
 *
 * The `device_resources_snmg` class is intended to be used in a single process to manage several
 * GPUs.
 */
class device_resources_snmg : public device_resources {
 public:
  /**
   * @brief Construct a SNMG resources instance with all available GPUs
   */
  device_resources_snmg() : device_resources()
  {
    RAFT_CUDA_TRY(cudaGetDevice(&main_gpu_id_));
    raft::resource::set_root_rank(*this, 0);

    // initialize all resources
    std::vector<raft::resources>& world_resources = raft::resource::get_multi_gpu_resource(*this);
    _init_world(world_resources);

    RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
  }

  /**
   * @brief Construct a SNMG resources instance with a subset of available GPUs
   *
   * @param[in] device_ids List of device IDs to be used
   */
  device_resources_snmg(const std::vector<int>& device_ids) : device_resources()
  {
    RAFT_CUDA_TRY(cudaGetDevice(&main_gpu_id_));
    raft::resource::set_root_rank(*this, 0);

    // initialize resources for the given device ids
    std::vector<raft::resources>& world_resources = raft::resource::get_multi_gpu_resource(*this);
    _init_world(world_resources, device_ids);
    RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
  }

  /**
   * @brief SNMG resources instance copy constructor
   *
   * @param[in] world A SNMG resources instance
   */
  device_resources_snmg(const device_resources_snmg& world)
    : device_resources(world), main_gpu_id_(world.main_gpu_id_)
  {
  }

  device_resources_snmg(device_resources_snmg&&)            = delete;
  device_resources_snmg& operator=(device_resources_snmg&&) = delete;
  ~device_resources_snmg()
  {
    for (int device_id : pool_device_ids_) {
      rmm::mr::reset_per_device_resource(rmm::cuda_device_id{device_id});
    }
  }

  /**
   * @brief Set a memory pool on all GPUs of the multi-gpu world
   */
  void set_memory_pool(int percent_of_free_memory)
  {
    if (!per_device_pools_.empty()) {
      for (int device_id : pool_device_ids_) {
        rmm::mr::reset_per_device_resource(rmm::cuda_device_id{device_id});
      }
      per_device_pools_.clear();
      pool_device_ids_.clear();
    }

    int world_size = raft::resource::get_num_ranks(*this);
    for (int rank = 0; rank < world_size; rank++) {
      const raft::resources& dev_res = raft::resource::set_current_device_to_rank(*this, rank);

      int device_id = raft::resource::get_device_id(dev_res);
      pool_device_ids_.push_back(device_id);

      per_device_pools_.push_back(std::make_unique<rmm::mr::pool_memory_resource>(
        rmm::mr::get_current_device_resource_ref(),
        rmm::percent_of_free_device_memory(percent_of_free_memory)));
      rmm::mr::set_per_device_resource(rmm::cuda_device_id{device_id}, *per_device_pools_.back());
    }
    RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
  }

  /**
   * @brief Set a CUDA stream pool on all GPUs of the multi-GPU world
   *
   * @param[in] num_streams Number of CUDA streams in each device's pool
   */
  void set_stream_pool(std::size_t num_streams)
  {
    RAFT_EXPECTS(num_streams > 0, "num_streams must be greater than zero");

    auto& device_resources = raft::resource::get_multi_gpu_resource(*this);
    for (auto& device_resource : device_resources) {
      rmm::cuda_set_device_raii device_guard{
        rmm::cuda_device_id{raft::resource::get_device_id(device_resource)}};
      raft::resource::set_cuda_stream_pool(device_resource,
                                           std::make_shared<rmm::cuda_stream_pool>(num_streams));
    }
  }

  bool has_resource_factory(resource::resource_type resource_type) const override
  {
    if (snmg_related_resources.find(resource_type) == snmg_related_resources.end()) {
      // for resources unrelated to SNMG switch current GPU to main GPU ID
      RAFT_CUDA_TRY(cudaSetDevice(main_gpu_id_));
    }
    return raft::resources::has_resource_factory(resource_type);
  }

 private:
  inline void _init_world(std::vector<raft::resources>& world_resources)
  {
    int world_size;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&world_size));

    for (int rank = 0; rank < world_size; rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(rank));
      world_resources.emplace_back();

      // initialize the device ID
      raft::resource::get_device_id(world_resources.back());
    }
  }

  inline void _init_world(std::vector<raft::resources>& world_resources,
                          const std::vector<int>& device_ids)
  {
    for (size_t rank = 0; rank < device_ids.size(); rank++) {
      RAFT_CUDA_TRY(cudaSetDevice(device_ids[rank]));
      world_resources.emplace_back();

      // initialize the device ID
      raft::resource::get_device_id(world_resources.back());
    }
  }
  int main_gpu_id_;
  std::vector<std::unique_ptr<rmm::mr::pool_memory_resource>> per_device_pools_;
  std::vector<int> pool_device_ids_;
};  // class device_resources_snmg

}  // namespace RAFT_EXPORT raft
