/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/cublas_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <cublasLt.h>

#include <memory>

namespace RAFT_EXPORT raft {
namespace resource {

class cublaslt_resource : public resource {
 public:
  cublaslt_resource() { RAFT_CUBLAS_TRY(cublasLtCreate(&handle_)); }
  ~cublaslt_resource() noexcept override { RAFT_CUBLAS_TRY_NO_THROW(cublasLtDestroy(handle_)); }
  auto get_resource() -> void* override { return &handle_; }

 private:
  cublasLtHandle_t handle_;
};

/** Factory that knows how to construct a specific raft::resource to populate the res_t. */
class cublaslt_resource_factory : public resource_factory {
 public:
  auto get_resource_type() -> resource_type override { return resource_type::CUBLASLT_HANDLE; }
  auto make_resource() -> resource* override { return new cublaslt_resource(); }
};

/**
 * @defgroup resource_cublaslt cuBLASLt handle resource functions
 * @{
 */

/**
 * Load a `cublasLtHandle_t` from raft res if it exists, otherwise add it and return it.
 *
 * @param[in] res the raft resources object
 * @return cublasLt handle
 */
inline auto get_cublaslt_handle(resources const& res) -> cublasLtHandle_t
{
  res.ensure_default_factory(std::make_shared<cublaslt_resource_factory>());
  return *res.get_resource<cublasLtHandle_t>(resource_type::CUBLASLT_HANDLE);
};

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
