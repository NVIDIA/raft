/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/cublas_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <cublas_v2.h>

namespace RAFT_EXPORT raft {
namespace resource {

class cublas_resource : public resource {
 public:
  cublas_resource(rmm::cuda_stream_view stream)
  {
    RAFT_CUBLAS_TRY_NO_THROW(cublasCreate(&cublas_res));
    RAFT_CUBLAS_TRY_NO_THROW(cublasSetStream(cublas_res, stream));
  }

  ~cublas_resource() override { RAFT_CUBLAS_TRY_NO_THROW(cublasDestroy(cublas_res)); }

  void* get_resource() override { return &cublas_res; }

 private:
  cublasHandle_t cublas_res;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class cublas_resource_factory : public resource_factory {
 public:
  cublas_resource_factory(rmm::cuda_stream_view stream) : stream_(stream) {}
  resource_type get_resource_type() override { return resource_type::CUBLAS_HANDLE; }
  resource* make_resource() override { return new cublas_resource(stream_); }

 private:
  rmm::cuda_stream_view stream_;
};

/**
 * @defgroup resource_cublas cuBLAS handle resource functions
 * @{
 */

/**
 * Load a `cublasHandle_t` from raft res if it exists, otherwise add it and return it.
 *
 * @param[in] res the raft resources object
 * @return cublas handle
 */
inline cublasHandle_t get_cublas_handle(resources const& res)
{
  res.ensure_default_factory(std::make_shared<cublas_resource_factory>(get_cuda_stream(res)));
  auto ret = *res.get_resource<cublasHandle_t>(resource_type::CUBLAS_HANDLE);
  RAFT_CUBLAS_TRY(cublasSetStream(ret, get_cuda_stream(res)));
  return ret;
};

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
