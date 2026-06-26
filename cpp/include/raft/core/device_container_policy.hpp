/*
 * SPDX-FileCopyrightText: Copyright (2019) Sandia Corporation
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 */
/*
 * Copyright (2019) Sandia Corporation
 *
 * The source code is licensed under the 3-clause BSD license found in the LICENSE file
 * thirdparty/LICENSES/mdarray.license
 */

#pragma once
#include <raft/core/detail/macros.hpp>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/detail/span.hpp>  // dynamic_extent
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/device_ptr.h>

namespace RAFT_EXPORT raft {
/**
 * @brief A simplified version of thrust::device_reference with support for CUDA stream.
 */
template <typename T>
class device_reference {
 public:
  using value_type    = typename std::remove_cv_t<T>;
  using pointer       = thrust::device_ptr<T>;
  using const_pointer = thrust::device_ptr<T const>;

 private:
  std::conditional_t<std::is_const<T>::value, const_pointer, pointer> ptr_;
  rmm::cuda_stream_view stream_;

 public:
  device_reference(thrust::device_ptr<T> ptr, rmm::cuda_stream_view stream)
    : ptr_{ptr}, stream_{stream}
  {
  }

  operator value_type() const  // NOLINT
  {
    auto* raw = ptr_.get();
    value_type v{};
    update_host(&v, raw, 1, stream_);
    return v;
  }
  auto operator=(T const& other) -> device_reference&
  {
    auto* raw = ptr_.get();
    update_device(raw, &other, 1, stream_);
    return *this;
  }
};

/**
 * @brief A thin wrapper over rmm::device_uvector for implementing the mdarray container policy.
 *
 */
template <typename T>
class device_uvector {
  rmm::device_uvector<T> data_;

 public:
  using value_type = T;
  using size_type  = std::size_t;

  using reference       = device_reference<T>;
  using const_reference = device_reference<T const>;

  using pointer       = value_type*;
  using const_pointer = value_type const*;

  using iterator       = pointer;
  using const_iterator = const_pointer;

 public:
  ~device_uvector()                                                       = default;
  device_uvector(device_uvector&&) noexcept                               = default;
  device_uvector(device_uvector const& that)                              = delete;
  auto operator=(device_uvector<T> const& that) -> device_uvector<T>&     = delete;
  auto operator=(device_uvector<T>&& that) noexcept -> device_uvector<T>& = default;

  /**
   * @brief Default ctor is deleted as it doesn't accept stream.
   */
  device_uvector() = delete;
  /**
   * @brief Ctor that accepts a size, stream and an optional mr.
   */
  explicit device_uvector(std::size_t size, rmm::cuda_stream_view stream) : data_{size, stream} {}
  /**
   * @brief Ctor that accepts a size, stream and a memory resource.
   */
  explicit device_uvector(std::size_t size,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
    : data_{size, stream, mr}
  {
  }
  /**
   * @brief Index operator that returns a proxy to the actual data.
   */
  template <typename Index>
  auto operator[](Index i) noexcept -> reference
  {
    return device_reference<T>{thrust::device_ptr<T>{data_.data() + i}, data_.stream()};
  }
  /**
   * @brief Index operator that returns a proxy to the actual data.
   */
  template <typename Index>
  auto operator[](Index i) const noexcept
  {
    return device_reference<T const>{thrust::device_ptr<T const>{data_.data() + i}, data_.stream()};
  }

  void resize(size_type size) { data_.resize(size, data_.stream()); }

  /**
   * @brief Resize the internal buffer without copying old data.
   *
   * Unlike resize(), this never copies old data.
   * Thus, unlike in resize(), there's no point in time where the old and the new buffers are both
   * alive, and the peak memory usage is lower.
   *
   * Unlike resize(), this deallocates the old buffer even if the new size is smaller.
   * This ensures the memory is released promptly.
   */
  void reallocate(size_type size)
  {
    if (size != data_.size()) {
      auto stream = data_.stream();
      auto mr     = data_.memory_resource();
      // Resize and shrink rmm::device_uvector: force deallocation without copying old data
      data_.resize(0, data_.stream());
      data_.shrink_to_fit(data_.stream());
      // Assign a new value after the old one is deallocated
      data_ = rmm::device_uvector<T>(size, stream, mr);
    }
  }

  [[nodiscard]] auto data() noexcept -> pointer { return data_.data(); }
  [[nodiscard]] auto data() const noexcept -> const_pointer { return data_.data(); }
};

/**
 * @brief A container policy for device mdarray.
 */
template <typename ElementType>
class device_container_policy {
 public:
  using element_type   = ElementType;
  using container_type = device_uvector<element_type>;
  // FIXME(jiamingy): allocator type is not supported by rmm::device_uvector
  using pointer         = typename container_type::pointer;
  using const_pointer   = typename container_type::const_pointer;
  using reference       = device_reference<element_type>;
  using const_reference = device_reference<element_type const>;

  using accessor_policy       = cuda::std::default_accessor<element_type>;
  using const_accessor_policy = cuda::std::default_accessor<element_type const>;

 public:
  auto create(raft::resources const& res, size_t n) -> container_type
  {
    return container_type(n, resource::get_cuda_stream(res), mr_);
  }

  constexpr device_container_policy() = default;
  explicit device_container_policy(rmm::device_async_resource_ref mr) noexcept : mr_(mr) {}

  [[nodiscard]] constexpr auto access(container_type& c, size_t n) const noexcept -> reference
  {
    return c[n];
  }
  [[nodiscard]] constexpr auto access(container_type const& c, size_t n) const noexcept
    -> const_reference
  {
    return c[n];
  }

  [[nodiscard]] auto make_accessor_policy() noexcept { return accessor_policy{}; }
  [[nodiscard]] auto make_accessor_policy() const noexcept { return const_accessor_policy{}; }

 private:
  rmm::device_async_resource_ref mr_{rmm::mr::get_current_device_resource_ref()};
};

}  // namespace RAFT_EXPORT raft
#else
#include <raft/core/detail/fail_container_policy.hpp>
namespace RAFT_EXPORT raft {

// Provide placeholders that will allow CPU-GPU interoperable codebases to
// compile in non-CUDA mode but which will throw exceptions at runtime on any
// attempt to touch device data

template <typename T>
using device_reference = detail::fail_reference<T>;

template <typename T>
using device_uvector = detail::fail_container<T>;

template <typename ElementType>
using device_container_policy = detail::fail_container_policy<ElementType>;

}  // namespace RAFT_EXPORT raft
#endif
