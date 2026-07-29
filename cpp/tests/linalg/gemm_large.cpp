/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace raft::linalg {

TEST(Raft, GemmCublasLt136LargeSpan)
{
  if (cublasLtGetVersion() != 130600) {
    GTEST_SKIP() << "This regression test targets cuBLASLt 13.6";
  }

  constexpr std::int64_t lda         = 16;
  constexpr std::int64_t informative = 2;
  constexpr std::int64_t targets     = 1;
  constexpr std::array<std::int64_t, 3> sample_counts{134217727, 134217728, 134217729};
  constexpr auto max_samples = sample_counts.back();
  constexpr auto a_elements  = static_cast<std::size_t>(max_samples) * lda;
  constexpr auto required_bytes =
    (a_elements + informative + static_cast<std::size_t>(max_samples)) * sizeof(float);
  constexpr std::size_t allocation_headroom = std::size_t{1} << 30;

  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);

  std::size_t free_bytes{};
  std::size_t total_bytes{};
  RAFT_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (free_bytes < required_bytes + allocation_headroom) {
    GTEST_SKIP() << "This regression test needs " << required_bytes + allocation_headroom
                 << " free device bytes, but only " << free_bytes << " are available";
  }

  rmm::device_uvector<float> a(a_elements, stream);
  rmm::device_uvector<float> b(informative, stream);
  rmm::device_uvector<float> c(max_samples, stream);

  RAFT_CUDA_TRY(cudaMemsetAsync(a.data(), 0, a.size() * sizeof(float), stream.value()));
  RAFT_CUDA_TRY(cudaMemsetAsync(b.data(), 0, b.size() * sizeof(float), stream.value()));

  const float alpha = 1.0f;
  const float beta  = 0.0f;
  for (const auto samples : sample_counts) {
    SCOPED_TRACE(samples);
    RAFT_CUDA_TRY(cudaMemsetAsync(c.data(), 0xff, samples * sizeof(float), stream.value()));
    raft::linalg::gemm(resources,
                       true,
                       true,
                       static_cast<int>(samples),
                       targets,
                       informative,
                       &alpha,
                       a.data(),
                       lda,
                       b.data(),
                       targets,
                       &beta,
                       c.data(),
                       static_cast<int>(samples),
                       stream.value());

    std::array<float, 2> output_edges{};
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      &output_edges[0], c.data(), sizeof(float), cudaMemcpyDeviceToHost, stream.value()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(&output_edges[1],
                                  c.data() + samples - 1,
                                  sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    EXPECT_FLOAT_EQ(output_edges[0], 0.0f);
    EXPECT_FLOAT_EQ(output_edges[1], 0.0f);
  }
}

}  // namespace raft::linalg
