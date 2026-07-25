/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <sys/timeb.h>

#include <vector>

namespace raft {
namespace random {

// Courtesy: cuGraph unit-tests

struct RmatInputs {
  size_t r_scale;
  size_t c_scale;
  size_t n_edges;
  bool theta_array;
  uint64_t seed;
  float eps;
};

template <typename OutT, typename InT>
RAFT_KERNEL normalize_kernel(
  OutT* theta, const InT* in_vals, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = threadIdx.x;
  if (idx < max_scale) {
    auto a   = OutT(in_vals[4 * idx]);
    auto b   = OutT(in_vals[4 * idx + 1]);
    auto c   = OutT(in_vals[4 * idx + 2]);
    auto d   = OutT(in_vals[4 * idx + 3]);
    auto sum = a + b + c + d;
    a /= sum;
    b /= sum;
    c /= sum;
    d /= sum;
    theta[4 * idx]     = a;
    theta[4 * idx + 1] = b;
    theta[4 * idx + 2] = c;
    theta[4 * idx + 3] = d;
  }
}

// handle rectangular cases correctly
template <typename OutT>
RAFT_KERNEL handle_rect_kernel(OutT* theta, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = threadIdx.x;
  if (idx < max_scale) {
    auto a = theta[4 * idx];
    auto b = theta[4 * idx + 1];
    auto c = theta[4 * idx + 2];
    auto d = theta[4 * idx + 3];
    if (idx >= r_scale) {
      a += c;
      c = OutT(0);
      b += d;
      d = OutT(0);
    }
    if (idx >= c_scale) {
      a += b;
      b = OutT(0);
      c += d;
      d = OutT(0);
    }
    theta[4 * idx]     = a;
    theta[4 * idx + 1] = b;
    theta[4 * idx + 2] = c;
    theta[4 * idx + 3] = d;
  }
}

// for a single probability distribution across depths, just replicate the theta's!
// this will keep the test code simpler
template <typename OutT>
RAFT_KERNEL theta_kernel(OutT* theta, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = threadIdx.x;
  if (idx != 0 && idx < max_scale) {
    auto a = theta[0];
    auto b = theta[1];
    auto c = theta[2];
    auto d = theta[3];
    if (idx >= r_scale) {
      a += c;
      c = OutT(0);
      b += d;
      d = OutT(0);
    }
    if (idx >= c_scale) {
      a += b;
      b = OutT(0);
      c += d;
      d = OutT(0);
    }
    theta[4 * idx]     = a;
    theta[4 * idx + 1] = b;
    theta[4 * idx + 2] = c;
    theta[4 * idx + 3] = d;
  }
}

template <typename OutT, typename InT>
void normalize(OutT* theta,
               const InT* in_vals,
               size_t max_scale,
               size_t r_scale,
               size_t c_scale,
               bool handle_rect,
               bool theta_array,
               cudaStream_t stream)
{
  // one threadblock with 256 threads is more than enough as the 'scale' parameters
  // won't be that large!
  normalize_kernel<OutT, InT><<<1, 256, 0, stream>>>(theta, in_vals, max_scale, r_scale, c_scale);
  RAFT_CUDA_TRY(cudaGetLastError());
  if (handle_rect) {
    handle_rect_kernel<<<1, 256, 0, stream>>>(theta, max_scale, r_scale, c_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  if (!theta_array) {
    theta_kernel<<<1, 256, 0, stream>>>(theta, max_scale, r_scale, c_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
}

RAFT_KERNEL compute_hist(
  int* hist, const size_t* out, size_t len, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  if (idx + 1 < len) {
    auto src = out[idx], dst = out[idx + 1];
    for (size_t bit_pos = 0; bit_pos < max_scale; ++bit_pos) {
      bool src_bit = bit_pos < r_scale ? src & (1 << bit_pos) : 0;
      bool dst_bit = bit_pos < c_scale ? dst & (1 << bit_pos) : 0;
      auto idx     = bit_pos * 4 + src_bit * 2 + dst_bit;
      atomicAdd(hist + idx, 1);
    }
  }
}

class RmatGenTest : public ::testing::TestWithParam<RmatInputs> {
 public:
  RmatGenTest()
    : handle{},
      stream{resource::get_cuda_stream(handle)},
      params{::testing::TestWithParam<RmatInputs>::GetParam()},
      out{params.n_edges * 2, stream},
      out_src{params.n_edges, stream},
      out_dst{params.n_edges, stream},
      theta{0, stream},
      h_theta{},
      state{params.seed, GeneratorType::GenPC},
      max_scale{std::max(params.r_scale, params.c_scale)}
  {
    theta.resize(4 * max_scale, stream);
    uniform<float>(handle, state, theta.data(), theta.size(), 0.0f, 1.0f);
    normalize<float, float>(theta.data(),
                            theta.data(),
                            max_scale,
                            params.r_scale,
                            params.c_scale,
                            params.r_scale != params.c_scale,
                            params.theta_array,
                            stream);
    h_theta.resize(theta.size());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    raft::update_host(h_theta.data(), theta.data(), theta.size(), stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  void SetUp() override
  {
    if (params.theta_array) {
      rmat_rectangular_gen(out.data(),
                           out_src.data(),
                           out_dst.data(),
                           theta.data(),
                           params.r_scale,
                           params.c_scale,
                           params.n_edges,
                           stream,
                           state);
    } else {
      rmat_rectangular_gen(out.data(),
                           out_src.data(),
                           out_dst.data(),
                           h_theta[0],
                           h_theta[1],
                           h_theta[2],
                           params.r_scale,
                           params.c_scale,
                           params.n_edges,
                           stream,
                           state);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void validate()
  {
    rmm::device_uvector<int> hist{theta.size(), stream};
    RAFT_CUDA_TRY(cudaMemsetAsync(hist.data(), 0, hist.size() * sizeof(int), stream));
    compute_hist<<<raft::ceildiv<size_t>(out.size() / 2, 256), 256, 0, stream>>>(
      hist.data(), out.data(), out.size(), max_scale, params.r_scale, params.c_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
    rmm::device_uvector<float> computed_theta{theta.size(), stream};
    normalize<float, int>(computed_theta.data(),
                          hist.data(),
                          max_scale,
                          params.r_scale,
                          params.c_scale,
                          false,
                          true,
                          stream);
    RAFT_CUDA_TRY(cudaGetLastError());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ASSERT_TRUE(devArrMatchHost(
      h_theta.data(), computed_theta.data(), theta.size(), CompareApprox<float>(params.eps)));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RmatInputs params;
  rmm::device_uvector<size_t> out, out_src, out_dst;
  rmm::device_uvector<float> theta;
  std::vector<float> h_theta;
  RngState state;
  size_t max_scale;
};

class RmatGenMdspanTest : public ::testing::TestWithParam<RmatInputs> {
 public:
  RmatGenMdspanTest()
    : handle{},
      stream{resource::get_cuda_stream(handle)},
      params{::testing::TestWithParam<RmatInputs>::GetParam()},
      out{params.n_edges * 2, stream},
      out_src{params.n_edges, stream},
      out_dst{params.n_edges, stream},
      theta{0, stream},
      h_theta{},
      state{params.seed, GeneratorType::GenPC},
      max_scale{std::max(params.r_scale, params.c_scale)}
  {
    theta.resize(4 * max_scale, stream);
    uniform<float>(handle, state, theta.data(), theta.size(), 0.0f, 1.0f);
    normalize<float, float>(theta.data(),
                            theta.data(),
                            max_scale,
                            params.r_scale,
                            params.c_scale,
                            params.r_scale != params.c_scale,
                            params.theta_array,
                            stream);
    h_theta.resize(theta.size());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    raft::update_host(h_theta.data(), theta.data(), theta.size(), stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  void SetUp() override
  {
    using index_type = size_t;

    using out_view_type = raft::device_mdspan<index_type,
                                              raft::extents<index_type, raft::dynamic_extent, 2>,
                                              raft::row_major>;
    out_view_type out_view(out.data(), params.n_edges);

    using out_src_view_type = raft::device_vector_view<index_type, index_type>;
    out_src_view_type out_src_view(out_src.data(), out_src.size());

    using out_dst_view_type = raft::device_vector_view<index_type, index_type>;
    out_dst_view_type out_dst_view(out_dst.data(), out_dst.size());

    if (params.theta_array) {
      raft::device_vector_view<const float, index_type> theta_view(theta.data(), theta.size());
      rmat_rectangular_gen(handle,
                           state,
                           theta_view,
                           out_view,
                           out_src_view,
                           out_dst_view,
                           params.r_scale,
                           params.c_scale);
    } else {
      rmat_rectangular_gen(handle,
                           state,
                           out_view,
                           out_src_view,
                           out_dst_view,
                           h_theta[0],
                           h_theta[1],
                           h_theta[2],
                           params.r_scale,
                           params.c_scale);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void validate()
  {
    rmm::device_uvector<int> hist{theta.size(), stream};
    RAFT_CUDA_TRY(cudaMemsetAsync(hist.data(), 0, hist.size() * sizeof(int), stream));
    compute_hist<<<raft::ceildiv<size_t>(out.size() / 2, 256), 256, 0, stream>>>(
      hist.data(), out.data(), out.size(), max_scale, params.r_scale, params.c_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
    rmm::device_uvector<float> computed_theta{theta.size(), stream};
    normalize<float, int>(computed_theta.data(),
                          hist.data(),
                          max_scale,
                          params.r_scale,
                          params.c_scale,
                          false,
                          true,
                          stream);
    RAFT_CUDA_TRY(cudaGetLastError());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ASSERT_TRUE(devArrMatchHost(
      h_theta.data(), computed_theta.data(), theta.size(), CompareApprox<float>(params.eps)));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RmatInputs params;
  rmm::device_uvector<size_t> out, out_src, out_dst;
  rmm::device_uvector<float> theta;
  std::vector<float> h_theta;
  RngState state;
  size_t max_scale;
};

static const float TOLERANCE = 0.01f;

const std::vector<RmatInputs> inputs = {
  // square adjacency
  {16, 16, 100000, false, 123456ULL, TOLERANCE},
  {16, 16, 100000, true, 123456ULL, TOLERANCE},
  {16, 16, 200000, false, 123456ULL, TOLERANCE},
  {16, 16, 200000, true, 123456ULL, TOLERANCE},
  {18, 18, 100000, false, 123456ULL, TOLERANCE},
  {18, 18, 100000, true, 123456ULL, TOLERANCE},
  {18, 18, 200000, false, 123456ULL, TOLERANCE},
  {18, 18, 200000, true, 123456ULL, TOLERANCE},
  {16, 16, 100000, false, 456789ULL, TOLERANCE},
  {16, 16, 100000, true, 456789ULL, TOLERANCE},
  {16, 16, 200000, false, 456789ULL, TOLERANCE},
  {16, 16, 200000, true, 456789ULL, TOLERANCE},
  {18, 18, 100000, false, 456789ULL, TOLERANCE},
  {18, 18, 100000, true, 456789ULL, TOLERANCE},
  {18, 18, 200000, false, 456789ULL, TOLERANCE},
  {18, 18, 200000, true, 456789ULL, TOLERANCE},

  // rectangular adjacency
  {16, 18, 200000, false, 123456ULL, TOLERANCE},
  {16, 18, 200000, true, 123456ULL, TOLERANCE},
  {18, 16, 200000, false, 123456ULL, TOLERANCE},
  {18, 16, 200000, true, 123456ULL, TOLERANCE},
  {16, 18, 200000, false, 456789ULL, TOLERANCE},
  {16, 18, 200000, true, 456789ULL, TOLERANCE},
  {18, 16, 200000, false, 456789ULL, TOLERANCE},
  {18, 16, 200000, true, 456789ULL, TOLERANCE}};

struct RmatForcedOutputs {
  size_t r_scale;
  size_t c_scale;
  size_t r_node_id;
  size_t c_node_id;
};

class RmatGenForceTest : public ::testing::TestWithParam<RmatForcedOutputs> {
 public:
  RmatGenForceTest()
    : handle{},
      stream{resource::get_cuda_stream(handle)},
      params{::testing::TestWithParam<RmatForcedOutputs>::GetParam()},
      out{2, stream},
      out_src{1, stream},
      out_dst{1, stream},
      theta{0, stream},
      h_theta{},
      state{0, GeneratorType::GenPC},
      max_scale(std::max(params.r_scale, params.c_scale))
  {
    theta.resize(4 * max_scale, stream);
    h_theta.resize(theta.size(), 0.f);
    for (size_t bit_pos = 0; bit_pos < max_scale; ++bit_pos) {
      size_t row_bit = ((params.r_node_id & (1 << bit_pos)) != 0);
      size_t col_bit = ((params.c_node_id & (1 << bit_pos)) != 0);

      // now force theta for bit -- 2x2 matrix row major
      h_theta[4 * bit_pos + row_bit * 2 + col_bit] = 1.f;
    }

    raft::update_device(theta.data(), h_theta.data(), max_scale * 4, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  void SetUp() override
  {
    rmat_rectangular_gen(out.data(),
                         out_src.data(),
                         out_dst.data(),
                         theta.data(),
                         params.r_scale,
                         params.c_scale,
                         size_t(1),
                         stream,
                         state);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void validate()
  {
    std::vector<size_t> h_out(2, size_t(0));
    raft::update_host(h_out.data(), out.data(), 2, stream);
    RAFT_CUDA_TRY(cudaGetLastError());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    std::vector<size_t> h_out_expect;
    h_out_expect.push_back(params.r_node_id);
    h_out_expect.push_back(params.c_node_id);

    ASSERT_TRUE(hostVecMatch(h_out_expect, h_out, raft::Compare<size_t>()));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RmatForcedOutputs params;
  size_t max_scale;
  std::vector<float> h_theta;
  rmm::device_uvector<size_t> out, out_src, out_dst;
  rmm::device_uvector<float> theta;
  RngState state;
};

const std::vector<RmatForcedOutputs> forcedInputs = {{16, 16, 12425, 1233},
                                                     {16, 16, 12, 424},
                                                     {5, 5, 15, 15},
                                                     {5, 6, 15, 15},
                                                     {5, 15, 15, 15},
                                                     {6, 5, 15, 15},
                                                     {15, 5, 15, 15},
                                                     {32, 16, 1253163, 60000},
                                                     {16, 16, 12, 0},
                                                     {16, 16, 0, 1255}};

TEST_P(RmatGenTest, Result) { validate(); }
INSTANTIATE_TEST_SUITE_P(RmatGenTests, RmatGenTest, ::testing::ValuesIn(inputs));

TEST_P(RmatGenMdspanTest, Result) { validate(); }
INSTANTIATE_TEST_SUITE_P(RmatGenMdspanTests, RmatGenMdspanTest, ::testing::ValuesIn(inputs));

TEST_P(RmatGenForceTest, Result) { validate(); }
INSTANTIATE_TEST_SUITE_P(RmatGenForceTests, RmatGenForceTest, ::testing::ValuesIn(forcedInputs));

}  // namespace random
}  // namespace raft
