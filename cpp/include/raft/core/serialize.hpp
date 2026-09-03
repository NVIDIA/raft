/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/detail/mdspan_numpy_serializer.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/managed_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <iostream>
#include <vector>

/**
 * Collection of serialization functions for RAFT data types
 */

namespace RAFT_EXPORT raft {

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void serialize_mdspan(
  const raft::resources&,
  std::ostream& os,
  const raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  detail::numpy_serializer::serialize_host_mdspan(os, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void serialize_mdspan(
  const raft::resources& handle,
  std::ostream& os,
  const raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");
  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  // Copy to host before serializing
  // For contiguous layouts, size() == product of dimensions
  std::vector<typename obj_t::value_type> tmp(obj.size());
  cudaStream_t stream = resource::get_cuda_stream(handle).get();
  raft::update_host(tmp.data(), obj.data_handle(), obj.size(), stream);
  resource::sync_stream(handle);
  using inner_accessor_type = typename obj_t::accessor_type::accessor_type;
  auto tmp_mdspan =
    raft::host_mdspan<ElementType, Extents, LayoutPolicy, raft::host_accessor<inner_accessor_type>>(
      tmp.data(), obj.extents());
  detail::numpy_serializer::serialize_host_mdspan(os, tmp_mdspan);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void serialize_mdspan(
  const raft::resources&,
  std::ostream& os,
  const raft::managed_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  using obj_t = raft::managed_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  using inner_accessor_type = typename obj_t::accessor_type::accessor_type;
  auto tmp_mdspan =
    raft::host_mdspan<ElementType, Extents, LayoutPolicy, raft::host_accessor<inner_accessor_type>>(
      obj.data_handle(), obj.extents());
  detail::numpy_serializer::serialize_host_mdspan(os, tmp_mdspan);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::resources&,
  std::istream& is,
  raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  detail::numpy_serializer::deserialize_host_mdspan(is, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::resources& handle,
  std::istream& is,
  raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  static_assert(std::is_same_v<LayoutPolicy, raft::layout_c_contiguous> ||
                  std::is_same_v<LayoutPolicy, raft::layout_f_contiguous>,
                "The serializer only supports row-major and column-major layouts");
  using obj_t = raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

  // Copy to device after serializing
  // For contiguous layouts, size() == product of dimensions
  std::vector<typename obj_t::value_type> tmp(obj.size());
  using inner_accessor_type = typename obj_t::accessor_type::accessor_type;
  auto tmp_mdspan =
    raft::host_mdspan<ElementType, Extents, LayoutPolicy, raft::host_accessor<inner_accessor_type>>(
      tmp.data(), obj.extents());
  detail::numpy_serializer::deserialize_host_mdspan(is, tmp_mdspan);

  cudaStream_t stream = resource::get_cuda_stream(handle).get();
  raft::update_device(obj.data_handle(), tmp.data(), obj.size(), stream);
  resource::sync_stream(handle);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::resources& handle,
  std::istream& is,
  raft::host_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>&& obj)
{
  deserialize_mdspan(handle, is, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::resources& handle,
  std::istream& is,
  raft::managed_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>& obj)
{
  using obj_t = raft::managed_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;
  using inner_accessor_type = typename obj_t::accessor_type::accessor_type;
  auto tmp_mdspan =
    raft::host_mdspan<ElementType, Extents, LayoutPolicy, raft::host_accessor<inner_accessor_type>>(
      obj.data_handle(), obj.extents());
  detail::numpy_serializer::deserialize_host_mdspan(is, tmp_mdspan);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::resources& handle,
  std::istream& is,
  raft::managed_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>&& obj)
{
  deserialize_mdspan(handle, is, obj);
}

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
inline void deserialize_mdspan(
  const raft::resources& handle,
  std::istream& is,
  raft::device_mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>&& obj)
{
  deserialize_mdspan(handle, is, obj);
}

template <typename T>
inline void serialize_scalar(const raft::resources&, std::ostream& os, const T& value)
{
  detail::numpy_serializer::serialize_scalar(os, value);
}

template <typename T>
inline T deserialize_scalar(const raft::resources&, std::istream& is)
{
  return detail::numpy_serializer::deserialize_scalar<T>(is);
}

}  // namespace RAFT_EXPORT raft
