/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "resource/resource_types.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>  // RAFT_EXPECTS
#include <raft/core/logger.hpp>

#include <memory>
#include <vector>

namespace RAFT_EXPORT raft {

/**
 * @brief Resource container which allows lazy-loading and registration
 * of resource_factory implementations, which in turn generate resource instances.
 *
 * This class is intended to be agnostic of the resources it contains and
 * does not, itself, differentiate between host and device resources. Downstream
 * accessor functions can then register and load resources as needed in order
 * to keep its usage somewhat opaque to end-users.
 *
 * Copies of a resources handle share the underlying resource_cell objects.
 * Lazy initialization (via get_resource / ensure_default_factory) stores into the
 * shared cell's atomics, so all copies see the update.  Explicit modification
 * (via add_resource_factory) replaces the shared_ptr<resource_cell> in the local
 * vector, isolating the change from other copies.
 *
 * Thread safety: concurrent const operations on the same handle are safe
 * (inner-cell atomics).  Concurrent const + non-const on the same handle
 * requires external synchronization (standard C++ rules).
 *
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/resource/cuda_stream.hpp>
 * #include <raft/core/resource/cublas_handle.hpp>
 *
 * raft::resources res;
 * auto stream = raft::resource::get_cuda_stream(res);
 * auto cublas_handle = raft::resource::get_cublas_handle(res);
 * @endcode
 */
class resources {
 public:
  resources() : cells_(resource::resource_type::LAST_KEY)
  {
    for (auto& c : cells_) {
      c = std::make_shared<resource::resource_cell>();
    }
  }

  resources(const resources&)            = default;
  resources(resources&&)                 = default;
  resources& operator=(const resources&) = default;
  resources& operator=(resources&&)      = default;
  virtual ~resources() {}

  /**
   * @brief Returns true if a resource_factory has been registered for the
   * given resource_type, false otherwise.
   * @param resource_type resource type to check
   * @return true if resource_factory is registered for the given resource_type
   */
  virtual bool has_resource_factory(resource::resource_type resource_type) const
  {
    return cells_[resource_type]->factory.load() != nullptr;
  }

  /**
   * @brief Register a resource_factory with the current instance (explicit set).
   *
   * Creates a new resource_cell with the given factory.  Other copies of this
   * handle continue to point at the old cell, so the change does not propagate.
   *
   * @param factory resource factory to register on the current instance
   */
  void add_resource_factory(std::shared_ptr<resource::resource_factory> factory)
  {
    resource::resource_type rtype = factory->get_resource_type();
    RAFT_EXPECTS(rtype != resource::resource_type::LAST_KEY,
                 "LAST_KEY is a placeholder and not a valid resource factory type.");
    auto new_cell = std::make_shared<resource::resource_cell>();
    new_cell->factory.store(std::move(factory));
    cells_[rtype] = std::move(new_cell);
  }

  /**
   * @brief Register a default factory if none has been set yet (lazy default).
   *
   * CAS's the factory into the existing shared cell.  If another thread or copy
   * already set a factory, this is a no-op.  Because the cell is shared, all
   * copies see the registered default.
   *
   * @param factory default resource factory
   */
  void ensure_default_factory(std::shared_ptr<resource::resource_factory> factory) const
  {
    resource::resource_type rtype = factory->get_resource_type();
    std::shared_ptr<resource::resource_factory> expected{};
    cells_[rtype]->factory.compare_exchange_strong(expected, std::move(factory));
  }

  /**
   * @brief Retrieve a resource for the given resource_type and cast to given pointer type.
   *
   * Resources are created lazily on first access using the registered factory.
   * The created resource is stored atomically in the shared cell, so all copies
   * of this handle that share the same cell see the resource.
   *
   * @tparam res_t pointer type for which retrieved resource will be casted
   * @param resource_type resource type to retrieve
   * @return the given resource, if it exists.
   */
  template <typename res_t>
  res_t* get_resource(resource::resource_type resource_type) const
  {
    auto& cell = cells_[resource_type];
    auto res   = cell->res.load();
    if (!res) {
      auto factory = cell->factory.load();
      RAFT_EXPECTS(factory != nullptr,
                   "No resource factory has been registered for the given resource %d.",
                   resource_type);
      auto new_res = std::shared_ptr<resource::resource>(factory->make_resource());
      std::shared_ptr<resource::resource> expected{};
      if (cell->res.compare_exchange_strong(expected, new_res)) {
        res = new_res;
      } else {
        res = expected;
      }
    }
    return reinterpret_cast<res_t*>(res->get_resource());
  }

 protected:
  std::vector<std::shared_ptr<resource::resource_cell>> cells_;
};
}  // namespace RAFT_EXPORT raft
