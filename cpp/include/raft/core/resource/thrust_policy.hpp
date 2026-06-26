/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

#include <rmm/exec_policy.hpp>
namespace RAFT_EXPORT raft {
namespace resource {
class thrust_policy_resource : public resource {
 public:
  thrust_policy_resource(rmm::cuda_stream_view stream_view)
    : thrust_policy_(std::make_unique<rmm::exec_policy_nosync>(stream_view))
  {
  }
  void* get_resource() override { return thrust_policy_.get(); }

  ~thrust_policy_resource() override {}

 private:
  std::unique_ptr<rmm::exec_policy_nosync> thrust_policy_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class thrust_policy_resource_factory : public resource_factory {
 public:
  thrust_policy_resource_factory(rmm::cuda_stream_view stream_view) : stream_view_(stream_view) {}
  resource_type get_resource_type() override { return resource_type::THRUST_POLICY; }
  resource* make_resource() override { return new thrust_policy_resource(stream_view_); }

 private:
  rmm::cuda_stream_view stream_view_;
};

/**
 * @defgroup resource_thrust_policy Thrust policy resource functions
 * @{
 */

/**
 * Load a thrust policy from a res (and populate it on the res if needed).
 * @param res raft res object for managing resources
 * @return thrust execution policy
 */
inline rmm::exec_policy_nosync& get_thrust_policy(resources const& res)
{
  res.ensure_default_factory(
    std::make_shared<thrust_policy_resource_factory>(get_cuda_stream(res)));
  return *res.get_resource<rmm::exec_policy_nosync>(resource_type::THRUST_POLICY);
};

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
