/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/cusolver_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <cusolverSp.h>

namespace RAFT_EXPORT raft {
namespace resource {

/**
 *
 */
class cusolver_sp_resource : public resource {
 public:
  cusolver_sp_resource(rmm::cuda_stream_view stream)
  {
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpCreate(&cusolver_res));
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpSetStream(cusolver_res, stream));
  }

  void* get_resource() override { return &cusolver_res; }

  ~cusolver_sp_resource() override { RAFT_CUSOLVER_TRY_NO_THROW(cusolverSpDestroy(cusolver_res)); }

 private:
  cusolverSpHandle_t cusolver_res;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class cusolver_sp_resource_factory : public resource_factory {
 public:
  cusolver_sp_resource_factory(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type get_resource_type() override { return resource_type::CUSOLVER_SP_HANDLE; }
  resource* make_resource() override { return new cusolver_sp_resource(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * @defgroup resource_cusolver_sp cuSolver SP handle resource functions
 * @{
 */

/**
 * Load a cusolverSpres_t from raft res if it exists, otherwise
 * add it and return it.
 * @param[in] res the raft resources object
 * @return cusolver sp handle
 */
inline cusolverSpHandle_t get_cusolver_sp_handle(resources const& res)
{
  res.ensure_default_factory(std::make_shared<cusolver_sp_resource_factory>(get_cuda_stream(res)));
  return *res.get_resource<cusolverSpHandle_t>(resource_type::CUSOLVER_SP_HANDLE);
};

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
