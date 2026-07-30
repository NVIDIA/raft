/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/gemm.cuh>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

namespace raft::linalg {

TEST(Raft, GemmLargeSpan)
{
  constexpr std::int64_t lda         = 16;
  constexpr std::int64_t informative = 2;
  constexpr std::int64_t targets     = 1;
  constexpr std::array<std::int64_t, 3> sample_counts{134217727, 134217728, 134217729};
  constexpr auto max_samples = sample_counts.back();
  constexpr auto a_elements  = max_samples * lda;

  raft::resources resources;
  const auto stream = raft::resource::get_cuda_stream(resources);

  auto a = raft::make_device_vector<float, std::int64_t>(resources, a_elements);
  auto b = raft::make_device_vector<float, std::int64_t>(resources, informative);
  auto c = raft::make_device_vector<float, std::int64_t>(resources, max_samples);

  RAFT_CUDA_TRY(cudaMemsetAsync(a.data_handle(), 0, a.size() * sizeof(float), stream.value()));
  RAFT_CUDA_TRY(cudaMemsetAsync(b.data_handle(), 0, b.size() * sizeof(float), stream.value()));

  const float alpha = 1.0f;
  const float beta  = 0.0f;
  for (const auto samples : sample_counts) {
    SCOPED_TRACE(samples);
    RAFT_CUDA_TRY(cudaMemsetAsync(c.data_handle(), 0xff, samples * sizeof(float), stream.value()));
    raft::linalg::gemm(resources,
                       true,
                       true,
                       static_cast<int>(samples),
                       targets,
                       informative,
                       &alpha,
                       a.data_handle(),
                       lda,
                       b.data_handle(),
                       targets,
                       &beta,
                       c.data_handle(),
                       static_cast<int>(samples),
                       stream.value());

    std::array<float, 3> output_samples{};
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      &output_samples[0], c.data_handle(), sizeof(float), cudaMemcpyDeviceToHost, stream.value()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(&output_samples[1],
                                  c.data_handle() + samples / 2,
                                  sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    RAFT_CUDA_TRY(cudaMemcpyAsync(&output_samples[2],
                                  c.data_handle() + samples - 1,
                                  sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  stream.value()));
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream.value()));
    EXPECT_FLOAT_EQ(output_samples[0], 0.0f);
    EXPECT_FLOAT_EQ(output_samples[1], 0.0f);
    EXPECT_FLOAT_EQ(output_samples[2], 0.0f);
  }
}

}  // namespace raft::linalg
