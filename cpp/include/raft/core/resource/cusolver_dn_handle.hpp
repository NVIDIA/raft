/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuda_stream.hpp"

#include <raft/core/cusolver_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cusolverDn.h>

namespace RAFT_EXPORT raft {
namespace resource {

/**
 *
 */
class cusolver_dn_resource : public resource {
 public:
  cusolver_dn_resource(rmm::cuda_stream_view stream)
  {
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnCreate(&cusolver_res));
    RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnSetStream(cusolver_res, stream));
  }

  void* get_resource() override { return &cusolver_res; }

  ~cusolver_dn_resource() override { RAFT_CUSOLVER_TRY_NO_THROW(cusolverDnDestroy(cusolver_res)); }

 private:
  cusolverDnHandle_t cusolver_res;
};

/**
 * @defgroup resource_cusolver_dn cuSolver DN handle resource functions
 * @{
 */

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class cusolver_dn_resource_factory : public resource_factory {
 public:
  cusolver_dn_resource_factory(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type get_resource_type() override { return resource_type::CUSOLVER_DN_HANDLE; }
  resource* make_resource() override { return new cusolver_dn_resource(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * Load a cusolverSpres_t from raft res if it exists, otherwise
 * add it and return it.
 * @param[in] res the raft resources object
 * @return cusolver dn handle
 */
inline cusolverDnHandle_t get_cusolver_dn_handle(resources const& res)
{
  res.ensure_default_factory(std::make_shared<cusolver_dn_resource_factory>(get_cuda_stream(res)));
  return *res.get_resource<cusolverDnHandle_t>(resource_type::CUSOLVER_DN_HANDLE);
};

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
