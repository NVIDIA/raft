/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <raft/common/nccl_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/multi_gpu.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <nccl.h>

#include <memory>

namespace RAFT_EXPORT raft {
namespace resource {

class nccl_comm_resource : public resource {
 public:
  nccl_comm_resource() : nccl_comms_(std::make_unique<std::vector<ncclComm_t>>(0)) {}
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
  resource_type get_resource_type() override { return resource_type::NCCL_COMM; }
  resource* make_resource() override { return new nccl_comm_resource(); }
};

inline void _init_nccl_comms(const resources& res)
{
  int curr_gpu_id;
  RAFT_CUDA_TRY(cudaGetDevice(&curr_gpu_id));

  int num_ranks    = raft::resource::get_num_ranks(res);
  auto& nccl_comms = *res.get_resource<std::vector<ncclComm_t>>(resource_type::NCCL_COMM);
  nccl_comms.resize(num_ranks);

  ncclUniqueId id;
  ncclGetUniqueId(&id);

  ncclGroupStart();
  for (int rank = 0; rank < num_ranks; rank++) {
    raft::resource::set_current_device_to_rank(res, rank);
    RAFT_NCCL_TRY(ncclCommInitRank(&(nccl_comms[rank]), num_ranks, id, rank));
  }
  ncclGroupEnd();
  RAFT_CUDA_TRY(cudaSetDevice(curr_gpu_id));
}

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
    res.ensure_default_factory(std::make_shared<nccl_comm_resource_factory>());
    _init_nccl_comms(res);
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
