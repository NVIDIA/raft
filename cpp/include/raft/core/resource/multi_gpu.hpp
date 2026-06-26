/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <memory>

namespace RAFT_EXPORT raft {
namespace resource {

class multi_gpu_resource : public resource {
 public:
  multi_gpu_resource() : world_resources_(std::make_unique<std::vector<raft::resources>>(0)) {}
  ~multi_gpu_resource() override {}
  void* get_resource() override { return world_resources_.get(); }

 private:
  std::unique_ptr<std::vector<raft::resources>> world_resources_;
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class multi_gpu_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::MULTI_GPU; }
  resource* make_resource() override { return new multi_gpu_resource(); }
};

class root_rank_resource : public resource {
 public:
  explicit root_rank_resource(int root_rank = 0) : root_rank_(root_rank) {}
  void* get_resource() override { return &root_rank_; }

  ~root_rank_resource() override {}

 private:
  int root_rank_;
};

class root_rank_resource_factory : public resource_factory {
 public:
  explicit root_rank_resource_factory(int root_rank = 0) : root_rank_(root_rank) {}
  resource_type get_resource_type() override { return resource_type::ROOT_RANK; }
  resource* make_resource() override { return new root_rank_resource(root_rank_); }

 private:
  int root_rank_;
};

inline int get_root_rank(resources const& res)
{
  res.ensure_default_factory(std::make_shared<root_rank_resource_factory>());
  return *res.get_resource<int>(resource_type::ROOT_RANK);
};

/**
 * Retrieves a multi gpu resource from raft res if it exists, otherwise initializes it and returns
 * it.
 *
 * @param[in] res the raft resources object
 * @return resource for each device in multi-gpu world
 */
inline std::vector<raft::resources>& get_multi_gpu_resource(resources const& res)
{
  res.ensure_default_factory(std::make_shared<multi_gpu_resource_factory>());
  return *res.get_resource<std::vector<raft::resources>>(resource_type::MULTI_GPU);
};

/**
 * @brief Returns true if res has a multi GPU resource type
 */
inline bool is_multi_gpu(resources const& res)
{
  return res.has_resource_factory(resource_type::MULTI_GPU);
};

/**
 * @brief Get number of gpus in multi-gpu world
 */
inline int get_num_ranks(resources const& res)
{
  return raft::resource::get_multi_gpu_resource(res).size();
}

/**
 * @brief Get specific rank's raft::resources object
 */
inline const raft::resources& get_device_resources_for_rank(resources const& res, int rank)
{
  std::vector<raft::resources>& world_resources = raft::resource::get_multi_gpu_resource(res);
  return world_resources[rank];
}

/**
 * @brief Switch device to rank and return its raft::resources object
 */
inline const raft::resources& set_current_device_to_rank(resources const& res, int rank)
{
  const raft::resources& dev_res = raft::resource::get_device_resources_for_rank(res, rank);
  RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(dev_res)));
  return dev_res;
}

/**
 * @brief Switch to root rank and return its raft::resources object
 */
inline const raft::resources& set_current_device_to_root_rank(resources const& res)
{
  int root_rank                  = get_root_rank(res);
  const raft::resources& dev_res = raft::resource::get_device_resources_for_rank(res, root_rank);
  RAFT_CUDA_TRY(cudaSetDevice(raft::resource::get_device_id(dev_res)));
  return dev_res;
}

/**
 * @brief Set the root rank to given rank
 */
inline void set_root_rank(resources& res, int root_rank)
{
  res.add_resource_factory(std::make_shared<root_rank_resource_factory>(root_rank));
};

}  // namespace resource
}  // namespace RAFT_EXPORT raft
