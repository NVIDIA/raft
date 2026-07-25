/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/common/nccl_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <nccl.h>

#include <algorithm>
#include <memory>
#include <vector>

namespace RAFT_EXPORT raft {
namespace resource {

class nccl_comm_resource : public resource {
 public:
  explicit nccl_comm_resource(const std::vector<int>& device_ids)
    : nccl_comms_(std::make_unique<std::vector<ncclComm_t>>(device_ids.size()))
  {
    RAFT_NCCL_TRY(
      ncclCommInitAll(nccl_comms_->data(), static_cast<int>(device_ids.size()), device_ids.data()));
  }
  ~nccl_comm_resource() override
  {
    int num_ranks = nccl_comms_->size();

    ncclGroupStart();
    for (int rank = 0; rank < num_ranks; rank++) {
      RAFT_NCCL_TRY_NO_THROW(ncclCommDestroy((*nccl_comms_)[rank]));
    }
    ncclGroupEnd();
  }
  void* get_resource() override { return nccl_comms_.get(); }

 private:
  std::unique_ptr<std::vector<ncclComm_t>> nccl_comms_;
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class nccl_comm_resource_factory : public resource_factory {
 public:
  explicit nccl_comm_resource_factory(std::vector<int> device_ids)
    : device_ids_(std::move(device_ids))
  {
  }
  resource_type get_resource_type() override { return resource_type::NCCL_COMM; }
  resource* make_resource() override { return new nccl_comm_resource(device_ids_); }

 private:
  std::vector<int> device_ids_;
};

/**
 * @defgroup ncclComm_t NCCL comm resource functions
 * @{
 */

/**
 * Load a NCCL comms for all gpus from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return NCCL comm for all gpus
 */
inline std::vector<ncclComm_t>& get_nccl_comms(const resources& res)
{
  if (!res.has_resource_factory(resource_type::NCCL_COMM)) {
    auto& world = raft::resource::get_multi_gpu_resource(res);
    std::vector<int> device_ids(world.size());
    std::ranges::transform(world, device_ids.begin(), [](const raft::resources& dev_res) {
      return raft::resource::get_device_id(dev_res);
    });

    res.ensure_default_factory(std::make_shared<nccl_comm_resource_factory>(std::move(device_ids)));
  }
  return *res.get_resource<std::vector<ncclComm_t>>(resource_type::NCCL_COMM);
};

/**
 * Load a NCCL comm from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @param rank rank number
 * @return NCCL comm
 */
inline ncclComm_t& get_nccl_comm_for_rank(const resources& res, int rank)
{
  auto& nccl_comms = raft::resource::get_nccl_comms(res);
  return nccl_comms[rank];
};

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
