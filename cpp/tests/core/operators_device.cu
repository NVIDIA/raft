/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/kvp.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/kernel_launch.hpp>

#include <rmm/cuda_stream.hpp>
#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

template <typename OutT, typename OpT, typename... Args>
RAFT_KERNEL eval_op_on_device_kernel(OutT* out, OpT op, Args... args)
{
  out[0] = op(std::forward<Args>(args)...);
}

template <typename OpT, typename... Args>
auto eval_op_on_device(OpT op, Args&&... args)
{
  typedef decltype(op(args...)) OutT;
  auto stream = rmm::cuda_stream_default;
  rmm::device_scalar<OutT> result(stream);
  raft::launch_kernel(
    stream, 1, 1, eval_op_on_device_kernel, result.data(), op, std::forward<Args>(args)...);
  return result.value(stream);
}

TEST(OperatorsDevice, IdentityOp)
{
  raft::identity_op op;
  ASSERT_TRUE(raft::match(12.34f, eval_op_on_device(op, 12.34f, 0), raft::Compare<float>()));
}

TEST(OperatorsDevice, CastOp)
{
  raft::cast_op<float> op;
  ASSERT_TRUE(
    raft::match(1234.0f, eval_op_on_device(op, 1234, 0), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsDevice, KeyOp)
{
  raft::key_op op;
  raft::KeyValuePair<int, float> kvp(12, 3.4f);
  ASSERT_TRUE(raft::match(12, eval_op_on_device(op, kvp, 0), raft::Compare<int>()));
}

TEST(OperatorsDevice, ValueOp)
{
  raft::value_op op;
  raft::KeyValuePair<int, float> kvp(12, 3.4f);
  ASSERT_TRUE(
    raft::match(3.4f, eval_op_on_device(op, kvp, 0), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsDevice, SqrtOpF)
{
  raft::sqrt_op op;
  ASSERT_TRUE(raft::match(
    std::sqrt(12.34f), eval_op_on_device(op, 12.34f, 0), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(raft::match(
    std::sqrt(12.34), eval_op_on_device(op, 12.34, 0), raft::CompareApprox<double>(0.000001)));
}

TEST(OperatorsDevice, NZOp)
{
  raft::nz_op op;
  ASSERT_TRUE(
    raft::match(0.0f, eval_op_on_device(op, 0.0f, 0), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(
    raft::match(1.0f, eval_op_on_device(op, 12.34f, 0), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsDevice, AbsOp)
{
  raft::abs_op op;
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, -12.34f, 0), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(
    raft::match(12.34, eval_op_on_device(op, -12.34, 0), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(1234, eval_op_on_device(op, -1234, 0), raft::Compare<int>()));
}

TEST(OperatorsDevice, SqOp)
{
  raft::sq_op op;
  ASSERT_TRUE(
    raft::match(152.2756f, eval_op_on_device(op, 12.34f, 0), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(289, eval_op_on_device(op, -17, 0), raft::Compare<int>()));
}

TEST(OperatorsDevice, AddOp)
{
  raft::add_op op;
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 12.0f, 0.34f), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(1234, eval_op_on_device(op, 1200, 34), raft::Compare<int>()));
}

TEST(OperatorsDevice, SubOp)
{
  raft::sub_op op;
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 13.0f, 0.66f), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(raft::match(1234, eval_op_on_device(op, 1300, 66), raft::Compare<int>()));
}

TEST(OperatorsDevice, MulOp)
{
  raft::mul_op op;
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 2.0f, 6.17f), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsDevice, DivOp)
{
  raft::div_op op;
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 37.02f, 3.0f), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsDevice, DivCheckZeroOp)
{
  raft::div_checkzero_op op;
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 37.02f, 3.0f), raft::CompareApprox<float>(0.00001f)));
  ASSERT_TRUE(
    raft::match(0.0f, eval_op_on_device(op, 37.02f, 0.0f), raft::CompareApprox<float>(0.00001f)));
}

TEST(OperatorsDevice, PowOp)
{
  raft::pow_op op;
  ASSERT_TRUE(
    raft::match(1000.0f, eval_op_on_device(op, 10.0f, 3.0f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(1000.0, eval_op_on_device(op, 10.0, 3.0), raft::CompareApprox<double>(0.000001)));
}

TEST(OperatorsDevice, MinOp)
{
  raft::min_op op;
  ASSERT_TRUE(
    raft::match(3.0f, eval_op_on_device(op, 3.0f, 5.0f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(3.0, eval_op_on_device(op, 5.0, 3.0), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(3, eval_op_on_device(op, 3, 5), raft::Compare<int>()));
}

TEST(OperatorsDevice, MaxOp)
{
  raft::max_op op;
  ASSERT_TRUE(
    raft::match(5.0f, eval_op_on_device(op, 3.0f, 5.0f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(5.0, eval_op_on_device(op, 5.0, 3.0), raft::CompareApprox<double>(0.000001)));
  ASSERT_TRUE(raft::match(5, eval_op_on_device(op, 3, 5), raft::Compare<int>()));
}

TEST(OperatorsDevice, SqDiffOp)
{
  raft::sqdiff_op op;
  ASSERT_TRUE(
    raft::match(4.0f, eval_op_on_device(op, 3.0f, 5.0f), raft::CompareApprox<float>(0.0001f)));
  ASSERT_TRUE(
    raft::match(4.0, eval_op_on_device(op, 5.0, 3.0), raft::CompareApprox<double>(0.000001)));
}

TEST(OperatorsDevice, ArgminOp)
{
  raft::argmin_op op;
  raft::KeyValuePair<int, float> kvp_a(0, 1.2f);
  raft::KeyValuePair<int, float> kvp_b(0, 3.4f);
  raft::KeyValuePair<int, float> kvp_c(1, 1.2f);
  ASSERT_TRUE(raft::match(
    kvp_a, eval_op_on_device(op, kvp_a, kvp_b), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_a, eval_op_on_device(op, kvp_b, kvp_a), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_a, eval_op_on_device(op, kvp_a, kvp_c), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_a, eval_op_on_device(op, kvp_c, kvp_a), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_c, eval_op_on_device(op, kvp_b, kvp_c), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_c, eval_op_on_device(op, kvp_c, kvp_b), raft::Compare<raft::KeyValuePair<int, float>>()));
}

TEST(OperatorsDevice, ArgmaxOp)
{
  raft::argmax_op op;
  raft::KeyValuePair<int, float> kvp_a(0, 1.2f);
  raft::KeyValuePair<int, float> kvp_b(0, 3.4f);
  raft::KeyValuePair<int, float> kvp_c(1, 1.2f);
  ASSERT_TRUE(raft::match(
    kvp_b, eval_op_on_device(op, kvp_a, kvp_b), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_b, eval_op_on_device(op, kvp_b, kvp_a), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_a, eval_op_on_device(op, kvp_a, kvp_c), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_a, eval_op_on_device(op, kvp_c, kvp_a), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_b, eval_op_on_device(op, kvp_b, kvp_c), raft::Compare<raft::KeyValuePair<int, float>>()));
  ASSERT_TRUE(raft::match(
    kvp_b, eval_op_on_device(op, kvp_c, kvp_b), raft::Compare<raft::KeyValuePair<int, float>>()));
}

TEST(OperatorsDevice, ConstOp)
{
  raft::const_op op(12.34f);
  ASSERT_TRUE(raft::match(12.34f, eval_op_on_device(op), raft::Compare<float>()));
  ASSERT_TRUE(raft::match(12.34f, eval_op_on_device(op, 42), raft::Compare<float>()));
  ASSERT_TRUE(raft::match(12.34f, eval_op_on_device(op, 13, 37.0f), raft::Compare<float>()));
}

template <typename T>
struct trinary_add {
  const T c;
  constexpr explicit trinary_add(const T& c_) : c{c_} {}
  constexpr RAFT_INLINE_FUNCTION auto operator()(T a, T b) const { return a + b + c; }
};

TEST(OperatorsDevice, PlugConstOp)
{
  // First, wrap around a default-constructible op
  {
    raft::plug_const_op<float, raft::add_op> op(0.34f);
    ASSERT_TRUE(
      raft::match(12.34f, eval_op_on_device(op, 12.0f), raft::CompareApprox<float>(0.0001f)));
  }

  // Second, wrap around a non-default-constructible op
  {
    auto op = raft::plug_const_op(10.0f, trinary_add<float>(2.0f));
    ASSERT_TRUE(
      raft::match(12.34f, eval_op_on_device(op, 0.34f), raft::CompareApprox<float>(0.0001f)));
  }
}

TEST(OperatorsDevice, AddConstOp)
{
  raft::add_const_op<float> op(0.34f);
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 12.0f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsDevice, SubConstOp)
{
  raft::sub_const_op<float> op(0.66f);
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 13.0f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsDevice, MulConstOp)
{
  raft::mul_const_op<float> op(2.0f);
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 6.17f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsDevice, DivConstOp)
{
  raft::div_const_op<float> op(3.0f);
  ASSERT_TRUE(
    raft::match(12.34f, eval_op_on_device(op, 37.02f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsDevice, DivCheckZeroConstOp)
{
  // Non-zero denominator
  {
    raft::div_checkzero_const_op<float> op(3.0f);
    ASSERT_TRUE(
      raft::match(12.34f, eval_op_on_device(op, 37.02f), raft::CompareApprox<float>(0.0001f)));
  }
  // Zero denominator
  {
    raft::div_checkzero_const_op<float> op(0.0f);
    ASSERT_TRUE(
      raft::match(0.0f, eval_op_on_device(op, 37.02f), raft::CompareApprox<float>(0.0001f)));
  }
}

TEST(OperatorsDevice, PowConstOp)
{
  raft::pow_const_op<float> op(3.0f);
  ASSERT_TRUE(
    raft::match(1000.0f, eval_op_on_device(op, 10.0f), raft::CompareApprox<float>(0.0001f)));
}

TEST(OperatorsDevice, ComposeOp)
{
  // All ops are default-constructible
  {
    raft::compose_op<raft::sqrt_op, raft::abs_op, raft::cast_op<float>> op;
    ASSERT_TRUE(raft::match(
      std::sqrt(42.0f), eval_op_on_device(op, -42, 0), raft::CompareApprox<float>(0.0001f)));
  }
  // Some ops are not default-constructible
  {
    auto op = raft::compose_op(
      raft::sqrt_op(), raft::abs_op(), raft::add_const_op<float>(8.0f), raft::cast_op<float>());
    ASSERT_TRUE(raft::match(
      std::sqrt(42.0f), eval_op_on_device(op, -50, 0), raft::CompareApprox<float>(0.0001f)));
  }
}

TEST(OperatorsDevice, MapArgsOp)
{
  // All ops are default-constructible
  {
    raft::map_args_op<raft::add_op, raft::sq_op, raft::abs_op> op;
    ASSERT_TRUE(
      raft::match(42.0f, eval_op_on_device(op, 5.0f, -17.0f), raft::CompareApprox<float>(0.0001f)));
  }
  // Some ops are not default-constructible
  {
    auto op = raft::map_args_op(
      raft::add_op(), raft::pow_const_op<float>(2.0f), raft::mul_const_op<float>(-1.0f));
    ASSERT_TRUE(
      raft::match(42.0f, eval_op_on_device(op, 5.0f, -17.0f), raft::CompareApprox<float>(0.0001f)));
  }
}
