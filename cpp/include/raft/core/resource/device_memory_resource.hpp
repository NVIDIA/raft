/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/limiting_resource_adaptor.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cstddef>
#include <memory>
#include <optional>

namespace RAFT_EXPORT raft {
namespace resource {

/**
 * \defgroup device_memory_resource Device memory resources
 * @{
 */

class device_memory_resource : public resource {
 public:
  explicit device_memory_resource(raft::mr::device_resource ar) : any_mr_(std::move(ar)) {}
  ~device_memory_resource() override = default;
  auto get_resource() -> void* override { return &any_mr_; }

 private:
  raft::mr::device_resource any_mr_;
};

class limiting_memory_resource : public resource {
 public:
  limiting_memory_resource(raft::mr::device_resource ar,
                           std::size_t allocation_limit,
                           std::optional<std::size_t> alignment)
    : any_upstream_(std::move(ar)),
      mr_(make_adaptor(rmm::device_async_resource_ref{any_upstream_}, allocation_limit, alignment))
  {
  }

  auto get_resource() -> void* override { return &mr_; }

  ~limiting_memory_resource() override = default;

 private:
  raft::mr::device_resource any_upstream_;
  rmm::mr::limiting_resource_adaptor mr_;

  static inline auto make_adaptor(rmm::device_async_resource_ref upstream,
                                  std::size_t limit,
                                  std::optional<std::size_t> alignment)
    -> rmm::mr::limiting_resource_adaptor
  {
    if (alignment.has_value()) {
      return rmm::mr::limiting_resource_adaptor(upstream, limit, alignment.value());
    } else {
      return rmm::mr::limiting_resource_adaptor(upstream, limit);
    }
  }
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
class large_workspace_resource_factory : public resource_factory {
 public:
  large_workspace_resource_factory()
    : any_mr_(raft::mr::device_resource{rmm::mr::get_current_device_resource_ref()})
  {
  }

  explicit large_workspace_resource_factory(raft::mr::device_resource mr) : any_mr_(std::move(mr))
  {
  }

  auto get_resource_type() -> resource_type override
  {
    return resource_type::LARGE_WORKSPACE_RESOURCE;
  }
  auto make_resource() -> resource* override { return new device_memory_resource(any_mr_); }

 private:
  raft::mr::device_resource any_mr_;
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
class workspace_resource_factory : public resource_factory {
 public:
  explicit workspace_resource_factory(raft::mr::device_resource mr =
                                        raft::mr::device_resource{
                                          rmm::mr::get_current_device_resource_ref()},
                                      std::optional<std::size_t> allocation_limit = std::nullopt,
                                      std::optional<std::size_t> alignment        = std::nullopt)
    : allocation_limit_(allocation_limit.has_value() ? allocation_limit.value()
                                                     : default_allocation_limit()),
      alignment_(alignment),
      any_mr_(std::move(mr))
  {
  }

  auto get_resource_type() -> resource_type override { return resource_type::WORKSPACE_RESOURCE; }
  auto make_resource() -> resource* override
  {
    return new limiting_memory_resource(any_mr_, allocation_limit_, alignment_);
  }

  /** Construct a sensible default pool memory resource. */
  static inline auto default_pool_resource(std::size_t limit) -> raft::mr::device_resource
  {
    // Set the default granularity to 1 GiB
    constexpr std::size_t kOneGb = 1024lu * 1024lu * 1024lu;
    // The initial size of the pool. The choice of this value only affects the performance a little
    // bit. Heuristics:
    //   1) the pool shouldn't be too big from the beginning independently of the limit;
    //   2) otherwise, set it to half the max size to avoid too many resize calls.
    auto min_size = std::min<std::size_t>(kOneGb, limit / 2lu);
    // The pool is going to be place behind the limiting resource adaptor. This means the user won't
    // be able to allocate more than 'limit' bytes of memory anyway. At the same time, the pool
    // itself may consume a little bit more memory than the 'limit' due to memory fragmentation.
    // Therefore, we look for a compromise, such that:
    //   1) 'limit' is accurate - the user should be more likely to run into the limiting
    //      resource adaptor bad_alloc error than into the pool bad_alloc error.
    //   2) The pool doesn't grab too much memory on top of the 'limit'.
    auto max_size = std::min<std::size_t>(limit + kOneGb / 2lu, limit * 3lu / 2lu);
    auto upstream = rmm::mr::get_current_device_resource_ref();
    RAFT_LOG_DEBUG(
      "Setting the workspace pool resource; memory limit = %zu, initial pool size = %zu, max pool "
      "size = %zu.",
      limit,
      min_size,
      max_size);
    return raft::mr::device_resource{rmm::mr::pool_memory_resource(upstream, min_size, max_size)};
  }

 private:
  std::size_t allocation_limit_;
  std::optional<std::size_t> alignment_;
  raft::mr::device_resource any_mr_;

  static inline auto default_allocation_limit() -> std::size_t
  {
    std::size_t free_size{};
    std::size_t total_size{};
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_size, &total_size));
    // Note, the workspace does not claim all this memory from the start, so it's still usable by
    // the main resource as well.
    // This limit is merely an order for algorithm internals to plan the batching accordingly.
    return total_size / 4;
  }
};

}  // namespace resource
}  // namespace RAFT_EXPORT raft

namespace raft {
namespace resource {
namespace detail {

inline auto get_workspace_adaptor(resources const& res) -> rmm::mr::limiting_resource_adaptor*
{
  res.ensure_default_factory(std::make_shared<workspace_resource_factory>());
  return res.get_resource<rmm::mr::limiting_resource_adaptor>(resource_type::WORKSPACE_RESOURCE);
}

}  // namespace detail
}  // namespace resource
}  // namespace raft

namespace RAFT_EXPORT raft {
namespace resource {
/**
 * @brief Load a temp workspace resource from a resources instance (and populate it on the res if
 * needed).
 *
 * Prefer get_workspace_resource_ref() for allocations and
 * get_workspace_{total,used,free}_bytes() for accounting queries.
 *
 * @param res raft resources object for managing resources
 * @return pointer to the workspace limiting_resource_adaptor
 */
inline auto get_workspace_resource(resources const& res) -> rmm::mr::limiting_resource_adaptor*
{
  return detail::get_workspace_adaptor(res);
}

/**
 * @brief Get the workspace as a non-owning device_async_resource_ref.
 *
 * @param res raft resources object for managing resources
 * @return non-owning reference to the workspace device memory resource
 */
inline auto get_workspace_resource_ref(resources const& res) -> rmm::device_async_resource_ref
{
  return rmm::device_async_resource_ref{*detail::get_workspace_adaptor(res)};
}

/**
 * @brief Get the total size of the workspace resource.
 *
 * @param res raft resources object for managing resources
 * @return total allocation limit in bytes
 */
inline auto get_workspace_total_bytes(resources const& res) -> size_t
{
  return detail::get_workspace_adaptor(res)->get_allocation_limit();
}

/**
 * @brief Get the already allocated size of the workspace resource.
 *
 * @param res raft resources object for managing resources
 * @return currently allocated bytes
 */
inline auto get_workspace_used_bytes(resources const& res) -> size_t
{
  return detail::get_workspace_adaptor(res)->get_allocated_bytes();
}

/**
 * @brief Get the available size of the workspace resource.
 *
 * @param res raft resources object for managing resources
 * @return free bytes (total limit minus allocated)
 */
inline auto get_workspace_free_bytes(resources const& res) -> size_t
{
  const auto* p = detail::get_workspace_adaptor(res);
  return p->get_allocation_limit() - p->get_allocated_bytes();
}

/**
 * @brief Set the workspace resource.
 *
 * @param res raft resources object for managing resources
 * @param mr device memory resource
 * @param allocation_limit
 *   the total amount of memory in bytes available to the temporary workspace resources.
 * @param alignment optional alignment requirements passed to allocations
 */
inline void set_workspace_resource(resources& res,
                                   raft::mr::device_resource mr,
                                   std::optional<std::size_t> allocation_limit = std::nullopt,
                                   std::optional<std::size_t> alignment        = std::nullopt)
{
  res.add_resource_factory(
    std::make_shared<workspace_resource_factory>(std::move(mr), allocation_limit, alignment));
}

/**
 * Set the temporary workspace resource to a pool on top of the global memory resource
 * (`rmm::mr::get_current_device_resource_ref()`).
 *
 * @param res raft resources object for managing resources
 * @param allocation_limit
 *   the total amount of memory in bytes available to the temporary workspace resources;
 *   if not provided, a last used or default limit is used.
 *
 */
inline void set_workspace_to_pool_resource(
  resources& res, std::optional<std::size_t> allocation_limit = std::nullopt)
{
  if (!allocation_limit.has_value()) { allocation_limit = get_workspace_total_bytes(res); }
  res.add_resource_factory(std::make_shared<workspace_resource_factory>(
    workspace_resource_factory::default_pool_resource(*allocation_limit),
    allocation_limit,
    std::nullopt));
};

/**
 * Set the temporary workspace resource the same as the global memory resource
 * (`rmm::mr::get_current_device_resource_ref()`).
 *
 * Note, the workspace resource is always limited; the limit here defines how much of the global
 * memory resource can be consumed by the workspace allocations.
 *
 * @param res raft resources object for managing resources
 * @param allocation_limit
 *   the total amount of memory in bytes available to the temporary workspace resources.
 */
inline void set_workspace_to_global_resource(
  resources& res, std::optional<std::size_t> allocation_limit = std::nullopt)
{
  res.add_resource_factory(std::make_shared<workspace_resource_factory>(
    raft::mr::device_resource{rmm::mr::get_current_device_resource_ref()},
    allocation_limit,
    std::nullopt));
};

/**
 * @brief Get the large workspace as a non-owning device_async_resource_ref.
 *
 * @param res raft resources object for managing resources
 * @return non-owning reference to the large workspace device memory resource
 */
inline auto get_large_workspace_resource_ref(resources const& res) -> rmm::device_async_resource_ref
{
  res.ensure_default_factory(std::make_shared<large_workspace_resource_factory>());
  return rmm::device_async_resource_ref{
    *res.get_resource<raft::mr::device_resource>(resource_type::LARGE_WORKSPACE_RESOURCE)};
}

/**
 * @brief Set the large workspace resource.
 *
 * @param res raft resources object for managing resources
 * @param mr device memory resource
 */
inline void set_large_workspace_resource(resources& res, raft::mr::device_resource mr)
{
  res.add_resource_factory(std::make_shared<large_workspace_resource_factory>(std::move(mr)));
}

/** @} */

}  // namespace resource
}  // namespace RAFT_EXPORT raft
