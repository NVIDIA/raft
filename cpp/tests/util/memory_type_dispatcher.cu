/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/managed_mdarray.hpp>
#include <raft/core/managed_mdspan.hpp>
#include <raft/core/mdbuffer.cuh>
#include <raft/core/pinned_mdarray.hpp>
#include <raft/core/pinned_mdspan.hpp>
#include <raft/util/memory_type_dispatcher.cuh>

#include <gtest/gtest.h>

#include <cstdint>
#include <utility>
#include <variant>

namespace raft {

namespace dispatch_test {
struct functor_h {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    return memory_type::host;
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::host; }
};
struct functor_d {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    return memory_type::device;
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::device; }
};
struct functor_m {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    return memory_type::managed;
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::managed; }
};
struct functor_p {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    return memory_type::pinned;
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::pinned; }
};

struct functor_hd {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::host) {
      return memory_type::host;
    } else {
      return memory_type::device;
    }
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::host; }
  auto operator()(device_matrix_view<double> input) { return memory_type::device; }
};
struct functor_hm {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::managed) {
      return memory_type::managed;
    } else {
      return memory_type::host;
    }
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::host; }
  auto operator()(managed_matrix_view<double> input) { return memory_type::managed; }
};
struct functor_hp {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::pinned) {
      return memory_type::pinned;
    } else {
      return memory_type::host;
    }
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::host; }
  auto operator()(pinned_matrix_view<double> input) { return memory_type::pinned; }
};
struct functor_dm {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::managed) {
      return memory_type::managed;
    } else {
      return memory_type::device;
    }
  }
  auto operator()(device_matrix_view<double> input) { return memory_type::device; }
  auto operator()(managed_matrix_view<double> input) { return memory_type::managed; }
};
struct functor_dp {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::pinned) {
      return memory_type::pinned;
    } else {
      return memory_type::device;
    }
  }
  auto operator()(device_matrix_view<double> input) { return memory_type::device; }
  auto operator()(pinned_matrix_view<double> input) { return memory_type::pinned; }
};
struct functor_mp {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::pinned) {
      return memory_type::pinned;
    } else {
      return memory_type::managed;
    }
  }
  auto operator()(managed_matrix_view<double> input) { return memory_type::managed; }
  auto operator()(pinned_matrix_view<double> input) { return memory_type::pinned; }
};

struct functor_hdm {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::host) {
      return memory_type::host;
    } else if constexpr (input_memory_type == memory_type::managed) {
      return memory_type::managed;
    } else {
      return memory_type::device;
    }
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::host; }
  auto operator()(device_matrix_view<double> input) { return memory_type::device; }
  auto operator()(managed_matrix_view<double> input) { return memory_type::managed; }
};
struct functor_hdp {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::host) {
      return memory_type::host;
    } else if constexpr (input_memory_type == memory_type::pinned) {
      return memory_type::pinned;
    } else {
      return memory_type::device;
    }
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::host; }
  auto operator()(device_matrix_view<double> input) { return memory_type::device; }
  auto operator()(pinned_matrix_view<double> input) { return memory_type::pinned; }
};
struct functor_dmp {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    if constexpr (input_memory_type == memory_type::managed) {
      return memory_type::managed;
    } else if constexpr (input_memory_type == memory_type::pinned) {
      return memory_type::pinned;
    } else {
      return memory_type::device;
    }
  }
  auto operator()(device_matrix_view<double> input) { return memory_type::device; }
  auto operator()(managed_matrix_view<double> input) { return memory_type::managed; }
  auto operator()(pinned_matrix_view<double> input) { return memory_type::pinned; }
};

struct functor_hdmp {
  template <memory_type input_memory_type>
  auto static constexpr expected_output()
  {
    return input_memory_type;
  }
  auto operator()(host_matrix_view<double> input) { return memory_type::host; }
  auto operator()(device_matrix_view<double> input) { return memory_type::device; }
  auto operator()(managed_matrix_view<double> input) { return memory_type::managed; }
  auto operator()(pinned_matrix_view<double> input) { return memory_type::pinned; }
};

template <raft::memory_type input_memory_type,
          typename T             = double,
          typename layout_policy = layout_c_contiguous>
auto generate_input(raft::resources const& res)
{
  auto constexpr rows = std::uint32_t{3};
  auto constexpr cols = std::uint32_t{5};
  if constexpr (input_memory_type == raft::memory_type::host) {
    return raft::make_host_matrix<T, std::uint32_t, layout_policy>(rows, cols);
  } else if constexpr (input_memory_type == raft::memory_type::device) {
    return raft::make_device_matrix<T, std::uint32_t, layout_policy>(res, rows, cols);
  } else if constexpr (input_memory_type == raft::memory_type::managed) {
    return raft::make_managed_matrix<T, std::uint32_t, layout_policy>(res, rows, cols);
  } else if constexpr (input_memory_type == raft::memory_type::pinned) {
    return raft::make_pinned_matrix<T, std::uint32_t, layout_policy>(res, rows, cols);
  }
}

template <memory_type input_memory_type>
void test_memory_type_dispatcher()
{
  execute_with_dry_run_check(
    raft::device_resources{},
    [&](raft::resources const& res) {
      auto data         = generate_input<input_memory_type>(res);
      auto data_float   = generate_input<input_memory_type, float>(res);
      auto data_f       = generate_input<input_memory_type, double, layout_f_contiguous>(res);
      auto data_f_float = generate_input<input_memory_type, float, layout_f_contiguous>(res);

      EXPECT_EQ(memory_type_dispatcher(res, functor_h{}, data.view()),
                functor_h::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_d{}, data.view()),
                functor_d::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_m{}, data.view()),
                functor_m::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_p{}, data.view()),
                functor_p::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_hd{}, data.view()),
                functor_hd::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_hm{}, data.view()),
                functor_hm::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_hp{}, data.view()),
                functor_hp::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_dm{}, data.view()),
                functor_dm::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_dp{}, data.view()),
                functor_dp::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_mp{}, data.view()),
                functor_mp::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_hdm{}, data.view()),
                functor_hdm::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_hdp{}, data.view()),
                functor_hdp::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_dmp{}, data.view()),
                functor_dmp::expected_output<input_memory_type>());
      EXPECT_EQ(memory_type_dispatcher(res, functor_hdmp{}, data.view()),
                functor_hdmp::expected_output<input_memory_type>());

      // Functor expects double; input is float
      auto out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_h{}, data_float.view());
      EXPECT_EQ(out, functor_h::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_d{}, data_float.view());
      EXPECT_EQ(out, functor_d::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_m{}, data_float.view());
      EXPECT_EQ(out, functor_m::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_p{}, data_float.view());
      EXPECT_EQ(out, functor_p::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hd{}, data_float.view());
      EXPECT_EQ(out, functor_hd::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hm{}, data_float.view());
      EXPECT_EQ(out, functor_hm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hp{}, data_float.view());
      EXPECT_EQ(out, functor_hp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dm{}, data_float.view());
      EXPECT_EQ(out, functor_dm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dp{}, data_float.view());
      EXPECT_EQ(out, functor_dp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_mp{}, data_float.view());
      EXPECT_EQ(out, functor_mp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdm{}, data_float.view());
      EXPECT_EQ(out, functor_hdm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdp{}, data_float.view());
      EXPECT_EQ(out, functor_hdp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dmp{}, data_float.view());
      EXPECT_EQ(out, functor_dmp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdmp{}, data_float.view());
      EXPECT_EQ(out, functor_hdmp::expected_output<input_memory_type>());

      // Functor expects C-contiguous; input is F-contiguous
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_h{}, data_f.view());
      EXPECT_EQ(out, functor_h::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_d{}, data_f.view());
      EXPECT_EQ(out, functor_d::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_m{}, data_f.view());
      EXPECT_EQ(out, functor_m::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_p{}, data_f.view());
      EXPECT_EQ(out, functor_p::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hd{}, data_f.view());
      EXPECT_EQ(out, functor_hd::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hm{}, data_f.view());
      EXPECT_EQ(out, functor_hm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hp{}, data_f.view());
      EXPECT_EQ(out, functor_hp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dm{}, data_f.view());
      EXPECT_EQ(out, functor_dm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dp{}, data_f.view());
      EXPECT_EQ(out, functor_dp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_mp{}, data_f.view());
      EXPECT_EQ(out, functor_mp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdm{}, data_f.view());
      EXPECT_EQ(out, functor_hdm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdp{}, data_f.view());
      EXPECT_EQ(out, functor_hdp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dmp{}, data_f.view());
      EXPECT_EQ(out, functor_dmp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdmp{}, data_f.view());
      EXPECT_EQ(out, functor_hdmp::expected_output<input_memory_type>());

      // Functor expects C-contiguous double; input is F-contiguous float
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_h{}, data_f_float.view());
      EXPECT_EQ(out, functor_h::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_d{}, data_f_float.view());
      EXPECT_EQ(out, functor_d::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_m{}, data_f_float.view());
      EXPECT_EQ(out, functor_m::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_p{}, data_f_float.view());
      EXPECT_EQ(out, functor_p::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hd{}, data_f_float.view());
      EXPECT_EQ(out, functor_hd::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hm{}, data_f_float.view());
      EXPECT_EQ(out, functor_hm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hp{}, data_f_float.view());
      EXPECT_EQ(out, functor_hp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dm{}, data_f_float.view());
      EXPECT_EQ(out, functor_dm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dp{}, data_f_float.view());
      EXPECT_EQ(out, functor_dp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_mp{}, data_f_float.view());
      EXPECT_EQ(out, functor_mp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdm{}, data_f_float.view());
      EXPECT_EQ(out, functor_hdm::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdp{}, data_f_float.view());
      EXPECT_EQ(out, functor_hdp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_dmp{}, data_f_float.view());
      EXPECT_EQ(out, functor_dmp::expected_output<input_memory_type>());
      out = memory_type_dispatcher<mdbuffer<double, matrix_extent<std::uint32_t>>>(
        res, functor_hdmp{}, data_f_float.view());
      EXPECT_EQ(out, functor_hdmp::expected_output<input_memory_type>());
    },
    alloc_behavior::ARGUMENT_DRIVEN);
}

}  // namespace dispatch_test

TEST(MemoryTypeDispatcher, FromHost)
{
  dispatch_test::test_memory_type_dispatcher<memory_type::host>();
}

TEST(MemoryTypeDispatcher, FromDevice)
{
  dispatch_test::test_memory_type_dispatcher<memory_type::device>();
}

TEST(MemoryTypeDispatcher, FromManaged)
{
  dispatch_test::test_memory_type_dispatcher<memory_type::managed>();
}

TEST(MemoryTypeDispatcher, FromPinned)
{
  dispatch_test::test_memory_type_dispatcher<memory_type::pinned>();
}

}  // namespace raft
