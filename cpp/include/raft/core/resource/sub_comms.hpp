/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/comms.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>

namespace RAFT_EXPORT raft {
namespace resource {
class sub_comms_resource : public resource {
 public:
  sub_comms_resource() : communicators_() {}
  void* get_resource() override { return &communicators_; }

  ~sub_comms_resource() override {}

 private:
  std::unordered_map<std::string, std::shared_ptr<comms::comms_t>> communicators_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class sub_comms_resource_factory : public resource_factory {
 public:
  resource_type get_resource_type() override { return resource_type::SUB_COMMUNICATOR; }
  resource* make_resource() override { return new sub_comms_resource(); }
};

/**
 * @defgroup resource_sub_comms Subcommunicator resource functions
 * @{
 */

inline const comms::comms_t& get_subcomm(const resources& res, std::string key)
{
  res.ensure_default_factory(std::make_shared<sub_comms_resource_factory>());

  auto sub_comms =
    res.get_resource<std::unordered_map<std::string, std::shared_ptr<comms::comms_t>>>(
      resource_type::SUB_COMMUNICATOR);
  auto sub_comm = sub_comms->at(key);
  RAFT_EXPECTS(nullptr != sub_comm.get(), "ERROR: Subcommunicator was not initialized");

  return *sub_comm;
}

// In-place mutation: sub-communicators are typically set once during initialization
// and should be visible to all copies of the resources handle (Goal 1 / lazy-init
// propagation semantics).
// Note: we don't replace the _resource_ here (sub_comms_resource), so `res` is passed
//       by const reference.
inline void set_subcomm(resources const& res,
                        std::string key,
                        std::shared_ptr<comms::comms_t> subcomm)
{
  res.ensure_default_factory(std::make_shared<sub_comms_resource_factory>());
  auto sub_comms =
    res.get_resource<std::unordered_map<std::string, std::shared_ptr<comms::comms_t>>>(
      resource_type::SUB_COMMUNICATOR);
  sub_comms->insert(std::make_pair(key, subcomm));
}

/**
 * @}
 */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
