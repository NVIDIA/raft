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
class comms_resource : public resource {
 public:
  comms_resource(std::shared_ptr<comms::comms_t> comnumicator) : communicator_(comnumicator) {}

  void* get_resource() override { return &communicator_; }

  ~comms_resource() override {}

 private:
  std::shared_ptr<comms::comms_t> communicator_;
};

/**
 * Factory that knows how to construct a
 * specific raft::resource to populate
 * the res_t.
 */
class comms_resource_factory : public resource_factory {
 public:
  comms_resource_factory(std::shared_ptr<comms::comms_t> communicator) : communicator_(communicator)
  {
  }

  resource_type get_resource_type() override { return resource_type::COMMUNICATOR; }

  resource* make_resource() override { return new comms_resource(communicator_); }

 private:
  std::shared_ptr<comms::comms_t> communicator_;
};

/**
 * @defgroup resource_comms Comms resource functions
 * @{
 */

inline bool comms_initialized(resources const& res)
{
  return res.has_resource_factory(resource_type::COMMUNICATOR);
}

inline comms::comms_t const& get_comms(resources const& res)
{
  RAFT_EXPECTS(comms_initialized(res), "ERROR: Communicator was not initialized\n");
  return *(*res.get_resource<std::shared_ptr<comms::comms_t>>(resource_type::COMMUNICATOR));
}

inline void set_comms(resources& res, std::shared_ptr<comms::comms_t> communicator)
{
  res.add_resource_factory(std::make_shared<comms_resource_factory>(communicator));
}

/**
 * @}
 */
}  // namespace resource
}  // namespace RAFT_EXPORT raft
