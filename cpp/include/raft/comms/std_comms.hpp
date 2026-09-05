/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/comms/detail/std_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <nccl.h>
#include <ucp/api/ucp.h>
#include <ucxx/api.h>

#include <iostream>

namespace raft {
namespace comms {

using std_comms = detail::std_comms;

/**
 * @defgroup std_comms_factory std_comms Factory functions
 * @{
 */

/**
 * Factory function to construct a RAFT NCCL communicator and inject it into a
 * RAFT handle.
 *
 * @param handle raft::resources for injecting the comms
 * @param nccl_comm initialized NCCL communicator to use for collectives
 * @param num_ranks number of ranks in communicator clique
 * @param rank rank of local instance
 *
 * @code{.cpp}
 * #include <raft/comms/std_comms.hpp>
 * #include <raft/core/device_mdarray.hpp>
 *
 * ncclComm_t nccl_comm;
 * raft::resources handle;
 *
 * build_comms_nccl_only(&handle, nccl_comm, 5, 0);
 * ...
 * const auto& comm = resource::get_comms(handle);
 * auto gather_data = raft::make_device_vector<float>(handle, comm.get_size());
 * ...
 * comm.allgather((gather_data.data_handle())[comm.get_rank()],
 *                gather_data.data_handle(),
 *                1,
 *                resource::get_cuda_stream(handle));
 *
 * comm.sync_stream(resource::get_cuda_stream(handle));
 * @endcode
 */
void build_comms_nccl_only(resources* handle, ncclComm_t nccl_comm, int num_ranks, int rank)
{
  cudaStream_t stream = resource::get_cuda_stream(*handle).get();

  auto communicator = std::make_shared<comms_t>(
    std::unique_ptr<comms_iface>(new raft::comms::std_comms(nccl_comm, num_ranks, rank, stream)));
  resource::set_comms(*handle, communicator);
}

/**
 * Factory function to construct a RAFT NCCL+UCX and inject it into a RAFT
 * handle.
 *
 * @param handle raft::resources for injecting the comms
 * @param nccl_comm initialized NCCL communicator to use for collectives
 * @param is_ucxx whether `ucp_worker` and `eps` objects are UCXX (true) or
 *                pure UCX (false).
 * @param ucp_worker of local process
 *        Note: This is purposefully left as void* so that the ucp_worker_h
 *        doesn't need to be exposed through the cython layer
 * @param eps array of ucp_ep_h instances.
 *        Note: This is purposefully left as void* so that
 *        the ucp_ep_h doesn't need to be exposed through the cython layer.
 * @param num_ranks number of ranks in communicator clique
 * @param rank rank of local instance
 *
 * @code{.cpp}
 * #include <raft/comms/std_comms.hpp>
 * #include <raft/core/device_mdarray.hpp>
 *
 * ncclComm_t nccl_comm;
 * raft::resources handle;
 * ucp_worker_h ucp_worker;
 * ucp_ep_h *ucp_endpoints_arr;
 *
 * build_comms_nccl_ucx(&handle, nccl_comm, &ucp_worker, ucp_endpoints_arr, 5, 0);
 * ...
 * const auto& comm = resource::get_comms(handle);
 * auto gather_data = raft::make_device_vector<float>(handle, comm.get_size());
 * ...
 * comm.allgather((gather_data.data_handle())[comm.get_rank()],
 *                gather_data.data_handle(),
 *                1,
 *                resource::get_cuda_stream(handle));
 *
 * comm.sync_stream(resource::get_cuda_stream(handle));
 * @endcode
 */
void build_comms_nccl_ucx(resources* handle,
                          ncclComm_t nccl_comm,
                          bool is_ucxx,
                          void* ucp_worker,
                          void* eps,
                          int num_ranks,
                          int rank)
{
  detail::ucx_objects_t ucx_objects;
  if (is_ucxx) {
    ucx_objects.endpoints = std::make_shared<ucxx::Endpoint**>(new ucxx::Endpoint*[num_ranks]);
    ucx_objects.worker    = static_cast<ucxx::Worker*>(ucp_worker);
  } else {
    ucx_objects.endpoints = std::make_shared<ucp_ep_h*>(new ucp_ep_h[num_ranks]);
    ucx_objects.worker    = static_cast<ucp_worker_h>(ucp_worker);
  }

  auto size_t_ep_arr = reinterpret_cast<size_t*>(eps);

  for (int i = 0; i < num_ranks; i++) {
    size_t ptr = size_t_ep_arr[i];

    if (is_ucxx) {
      auto ucp_ep_v = reinterpret_cast<ucxx::Endpoint**>(
        *std::get<detail::ucxx_endpoint_array_t>(ucx_objects.endpoints));

      if (ptr != 0) {
        auto eps_ptr = reinterpret_cast<ucxx::Endpoint*>(size_t_ep_arr[i]);
        ucp_ep_v[i]  = eps_ptr;
      } else {
        ucp_ep_v[i] = nullptr;
      }
    } else {
      auto ucp_ep_v =
        reinterpret_cast<ucp_ep_h*>(*std::get<detail::ucp_endpoint_array_t>(ucx_objects.endpoints));

      if (ptr != 0) {
        auto eps_ptr = reinterpret_cast<ucp_ep_h>(size_t_ep_arr[i]);
        ucp_ep_v[i]  = eps_ptr;
      } else {
        ucp_ep_v[i] = nullptr;
      }
    }
  }

  cudaStream_t stream = resource::get_cuda_stream(*handle).get();

  auto communicator = std::make_shared<comms_t>(std::unique_ptr<comms_iface>(
    new raft::comms::std_comms(nccl_comm, ucx_objects, num_ranks, rank, stream)));
  resource::set_comms(*handle, communicator);
}

/**
 * @}
 */

inline void nccl_unique_id_from_char(ncclUniqueId* id, char* uniqueId)
{
  memcpy(id->internal, uniqueId, NCCL_UNIQUE_ID_BYTES);
}

inline void get_nccl_unique_id(char* uid)
{
  ncclUniqueId id;
  ncclGetUniqueId(&id);

  memcpy(uid, id.internal, NCCL_UNIQUE_ID_BYTES);
}
};  // namespace comms
};  // namespace raft
