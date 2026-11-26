/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/comms/detail/mpi_comms.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/comms.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft {
namespace comms {

using mpi_comms = detail::mpi_comms;

/**
 * @defgroup mpi_comms_factory MPI Comms Factory Functions
 * @{
 */

/**
 * Given a properly initialized MPI_Comm, construct an instance of RAFT's
 * MPI Communicator and inject it into the given RAFT handle instance
 * @param handle raft handle for managing expensive resources
 * @param comm an initialized MPI communicator
 *
 * @code{.cpp}
 * #include <raft/comms/mpi_comms.hpp>
 * #include <raft/core/device_mdarray.hpp>
 *
 * MPI_Comm mpi_comm;
 * raft::resources handle;
 *
 * initialize_mpi_comms(&handle, mpi_comm);
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
inline void initialize_mpi_comms(resources* handle, MPI_Comm comm)
{
  auto communicator = std::make_shared<comms_t>(
    std::unique_ptr<comms_iface>(new mpi_comms(comm, false, resource::get_cuda_stream(*handle))));
  resource::set_comms(*handle, communicator);
};

/**
 * @}
 */

};  // namespace comms
};  // namespace raft
