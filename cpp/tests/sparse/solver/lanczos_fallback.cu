/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/detail/lanczos.cuh>

#include <gtest/gtest.h>

#include <array>
#include <cmath>

namespace raft::sparse::solver {
TEST(LanczosEigensolverFallback, Float32Jacobi)
{
  // Reduced cuSOLVER reproducer. Symmetry also makes this column-major input.
  // clang-format off
  constexpr std::array<float, 25> input = {
     4.52996005e-8f,   6.31930868e-13f,  0.0f,             0.0f,             0.0f,
     6.31930868e-13f, -2.11824158e-9f,  -2.23728129e-8f,  0.0f,             0.0f,
     0.0f,           -2.23728129e-8f,  -2.36300252e-7f,  4.03643658e-14f,  0.0f,
     0.0f,            0.0f,            4.03643658e-14f, -2.38418579e-7f,   0.0f,
     0.0f,            0.0f,            0.0f,             0.0f,             1.0f,
  };
  // clang-format on
  constexpr std::array<float, 5> expected = {
    -2.38418596e-7f, -2.38418484e-7f, 0.0f, 4.52996005e-8f, 1.0f};

  raft::resources handle;
  auto stream = raft::resource::get_cuda_stream(handle).get();
  rmm::device_uvector<float> d_input(25, stream), d_vectors(25, stream), d_values(5, stream);
  raft::update_device(d_input.data(), input.data(), input.size(), stream);

  try {
    raft::linalg::eigDC(handle, d_input.data(), 5, 5, d_vectors.data(), d_values.data(), stream);
    GTEST_SKIP() << "This cuSOLVER version solves the reproducer without a fallback.";
  } catch (raft::linalg::detail::eigensolver_convergence_error const&) {
  }
  ASSERT_NO_THROW(detail::lanczos_eig_dc_with_fallback(
    handle, d_input.data(), 5, d_vectors.data(), d_values.data()));

  std::array<float, 25> vectors, unchanged;
  std::array<float, 5> values;
  raft::update_host(vectors.data(), d_vectors.data(), vectors.size(), stream);
  raft::update_host(values.data(), d_values.data(), values.size(), stream);
  raft::update_host(unchanged.data(), d_input.data(), unchanged.size(), stream);
  raft::resource::sync_stream(handle);

  EXPECT_EQ(input, unchanged);
  for (int i = 0; i < 5; ++i) {
    EXPECT_NEAR(values[i], expected[i], 1e-7f);
  }
  for (float value : vectors) {
    EXPECT_TRUE(std::isfinite(value));
  }
}
}  // namespace raft::sparse::solver
