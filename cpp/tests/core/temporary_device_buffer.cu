/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/temporary_device_buffer.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/fill.h>

#include <gtest/gtest.h>

namespace raft {

TEST(TemporaryDeviceBuffer, DevicePointer)
{
  {
    raft::resources handle;
    auto exts  = raft::make_extents<int>(5);
    auto array = raft::make_device_mdarray<int, int>(handle, exts);

    auto d_buf = raft::make_temporary_device_buffer(handle, array.data_handle(), exts);

    ASSERT_EQ(array.data_handle(), d_buf.view().data_handle());
    static_assert(!std::is_const_v<typename decltype(d_buf.view())::element_type>,
                  "element_type should not be const");
  }

  {
    raft::resources handle;
    auto exts  = raft::make_extents<int>(5);
    auto array = raft::make_device_mdarray<int, int>(handle, exts);

    auto d_buf = raft::make_readonly_temporary_device_buffer(handle, array.data_handle(), exts);

    ASSERT_EQ(array.data_handle(), d_buf.view().data_handle());
    static_assert(std::is_const_v<typename decltype(d_buf.view())::element_type>,
                  "element_type should be const");
  }
}

TEST(TemporaryDeviceBuffer, HostPointerWithWriteBack)
{
  raft::resources handle;
  auto exts  = raft::make_extents<int>(5);
  auto array = raft::make_host_mdarray<int, int>(exts);
  thrust::fill(array.data_handle(), array.data_handle() + array.extent(0), 1);
  rmm::device_uvector<int> result(5, resource::get_cuda_stream(handle));

  {
    execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        auto d_buf  = raft::make_writeback_temporary_device_buffer(h, array.data_handle(), exts);
        auto d_view = d_buf.view();

        thrust::fill(rmm::exec_policy(resource::get_cuda_stream(h)),
                     d_view.data_handle(),
                     d_view.data_handle() + d_view.extent(0),
                     10);
        raft::copy(
          result.data(), d_view.data_handle(), d_view.extent(0), resource::get_cuda_stream(h));
      },
      alloc_behavior::ARGUMENT_DRIVEN,
      5 * sizeof(int));
  }

  ASSERT_TRUE(raft::devArrMatchHost(array.data_handle(),
                                    result.data(),
                                    array.extent(0),
                                    raft::Compare<int>(),
                                    resource::get_cuda_stream(handle)));
}

}  // namespace raft
