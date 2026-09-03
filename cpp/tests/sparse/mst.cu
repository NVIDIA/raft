/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/mst.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/memory.h>
#include <thrust/reduce.h>

#include <bits/stdc++.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <optional>
#include <vector>

template <typename vertex_t, typename edge_t, typename weight_t>
struct CSRHost {
  std::vector<edge_t> offsets;
  std::vector<vertex_t> indices;
  std::vector<weight_t> weights;
};

template <typename vertex_t, typename edge_t, typename weight_t>
struct MSTTestInput {
  struct CSRHost<vertex_t, edge_t, weight_t> csr_h;
  int iterations;
};

template <typename vertex_t, typename edge_t, typename weight_t>
struct CSRDevice {
  rmm::device_buffer offsets;
  rmm::device_buffer indices;
  rmm::device_buffer weights;
};

namespace raft {
namespace mst {

// Sequential prims function
// Returns total weight of MST
template <typename vertex_t, typename edge_t, typename weight_t>
weight_t prims(CSRHost<vertex_t, edge_t, weight_t>& csr_h)
{
  std::size_t n_vertices = csr_h.offsets.size() - 1;

  bool active_vertex[n_vertices];
  //  bool mst_set[csr_h.n_edges];
  weight_t curr_edge[n_vertices];

  for (std::size_t i = 0; i < n_vertices; i++) {
    active_vertex[i] = false;
    curr_edge[i]     = static_cast<weight_t>(std::numeric_limits<int>::max());
  }
  curr_edge[0] = 0;

  // function to pick next min vertex-edge
  auto min_vertex_edge = [](auto* curr_edge, auto* active_vertex, auto n_vertices) {
    auto min = static_cast<weight_t>(std::numeric_limits<int>::max());
    vertex_t min_vertex{};

    for (std::size_t v = 0; v < n_vertices; v++) {
      if (!active_vertex[v] && curr_edge[v] < min) {
        min        = curr_edge[v];
        min_vertex = v;
      }
    }

    return min_vertex;
  };

  // iterate over n vertices
  for (std::size_t v = 0; v < n_vertices - 1; v++) {
    // pick min vertex-edge
    auto curr_v = min_vertex_edge(curr_edge, active_vertex, n_vertices);

    active_vertex[curr_v] = true;  // set to active

    // iterate through edges of current active vertex
    auto edge_st  = csr_h.offsets[curr_v];
    auto edge_end = csr_h.offsets[curr_v + 1];

    for (auto e = edge_st; e < edge_end; e++) {
      // put edges to be considered for next iteration
      auto neighbor_idx = csr_h.indices[e];
      if (!active_vertex[neighbor_idx] && csr_h.weights[e] < curr_edge[neighbor_idx]) {
        curr_edge[neighbor_idx] = csr_h.weights[e];
      }
    }
  }

  // find sum of MST
  weight_t total_weight = 0;
  for (std::size_t v = 1; v < n_vertices; v++) {
    total_weight += curr_edge[v];
  }

  return total_weight;
}

template <typename vertex_t, typename edge_t, typename weight_t>
class MSTTest : public ::testing::TestWithParam<MSTTestInput<vertex_t, edge_t, weight_t>> {
 protected:
  std::pair<raft::sparse::solver::Graph_COO<vertex_t, edge_t, weight_t>,
            raft::sparse::solver::Graph_COO<vertex_t, edge_t, weight_t>>
  mst_gpu()
  {
    edge_t* offsets   = static_cast<edge_t*>(csr_d.offsets.data());
    vertex_t* indices = static_cast<vertex_t*>(csr_d.indices.data());
    weight_t* weights = static_cast<weight_t*>(csr_d.weights.data());

    v = static_cast<vertex_t>((csr_d.offsets.size() / sizeof(vertex_t)) - 1);
    e = static_cast<edge_t>(csr_d.indices.size() / sizeof(edge_t));

    rmm::device_uvector<vertex_t> mst_src(2 * v - 2, resource::get_cuda_stream(handle));
    rmm::device_uvector<vertex_t> mst_dst(2 * v - 2, resource::get_cuda_stream(handle));
    rmm::device_uvector<vertex_t> color(v, resource::get_cuda_stream(handle));

    RAFT_CUDA_TRY(cudaMemsetAsync(mst_src.data(),
                                  std::numeric_limits<vertex_t>::max(),
                                  mst_src.size() * sizeof(vertex_t),
                                  resource::get_cuda_stream(handle)));
    RAFT_CUDA_TRY(cudaMemsetAsync(mst_dst.data(),
                                  std::numeric_limits<vertex_t>::max(),
                                  mst_dst.size() * sizeof(vertex_t),
                                  resource::get_cuda_stream(handle)));
    RAFT_CUDA_TRY(cudaMemsetAsync(
      color.data(), 0, color.size() * sizeof(vertex_t), resource::get_cuda_stream(handle)));

    vertex_t* color_ptr = thrust::raw_pointer_cast(color.data());

    auto run_mst = [&](bool symmetrize, bool init_colors, int iters) {
      using solver_t = raft::sparse::solver::MST_solver<vertex_t, edge_t, weight_t, float>;
      using COO      = raft::sparse::solver::Graph_COO<vertex_t, edge_t, weight_t>;
      std::optional<COO> out;
      raft::execute_with_dry_run_check(
        handle,
        [&](raft::resources const& h) {
          solver_t solver(h,
                          offsets,
                          indices,
                          weights,
                          v,
                          e,
                          color_ptr,
                          resource::get_cuda_stream(h),
                          symmetrize,
                          init_colors,
                          iters);
          auto result = solver.solve();
          if (!resource::get_dry_run_flag(h)) { out.emplace(std::move(result)); }
        },
        raft::alloc_behavior::DATA_DRIVEN);
      return std::move(*out);
    };

    if (iterations == 0) {
      auto symmetric_result     = run_mst(true, true, 0);
      auto non_symmetric_result = run_mst(false, true, 0);

      EXPECT_LE(symmetric_result.n_edges, 2 * v - 2);
      EXPECT_LE(non_symmetric_result.n_edges, v - 1);

      return std::make_pair(std::move(symmetric_result), std::move(non_symmetric_result));
    } else {
      auto intermediate_result = run_mst(true, true, iterations);
      auto symmetric_result    = run_mst(true, false, 0);

      // symmetric_result.n_edges += intermediate_result.n_edges;
      auto total_edge_size = symmetric_result.n_edges + intermediate_result.n_edges;
      symmetric_result.src.resize(total_edge_size, resource::get_cuda_stream(handle));
      symmetric_result.dst.resize(total_edge_size, resource::get_cuda_stream(handle));
      symmetric_result.weights.resize(total_edge_size, resource::get_cuda_stream(handle));

      raft::copy(symmetric_result.src.data() + symmetric_result.n_edges,
                 intermediate_result.src.data(),
                 intermediate_result.n_edges,
                 resource::get_cuda_stream(handle));
      raft::copy(symmetric_result.dst.data() + symmetric_result.n_edges,
                 intermediate_result.dst.data(),
                 intermediate_result.n_edges,
                 resource::get_cuda_stream(handle));
      raft::copy(symmetric_result.weights.data() + symmetric_result.n_edges,
                 intermediate_result.weights.data(),
                 intermediate_result.n_edges,
                 resource::get_cuda_stream(handle));
      symmetric_result.n_edges = total_edge_size;

      auto non_symmetric_result = run_mst(false, true, 0);

      EXPECT_LE(symmetric_result.n_edges, 2 * v - 2);
      EXPECT_LE(non_symmetric_result.n_edges, v - 1);

      return std::make_pair(std::move(symmetric_result), std::move(non_symmetric_result));
    }
  }

  void SetUp() override
  {
    mst_input  = ::testing::TestWithParam<MSTTestInput<vertex_t, edge_t, weight_t>>::GetParam();
    iterations = mst_input.iterations;

    csr_d.offsets = rmm::device_buffer(mst_input.csr_h.offsets.data(),
                                       mst_input.csr_h.offsets.size() * sizeof(edge_t),
                                       resource::get_cuda_stream(handle));
    csr_d.indices = rmm::device_buffer(mst_input.csr_h.indices.data(),
                                       mst_input.csr_h.indices.size() * sizeof(vertex_t),
                                       resource::get_cuda_stream(handle));
    csr_d.weights = rmm::device_buffer(mst_input.csr_h.weights.data(),
                                       mst_input.csr_h.weights.size() * sizeof(weight_t),
                                       resource::get_cuda_stream(handle));
  }

  void TearDown() override {}

 protected:
  MSTTestInput<vertex_t, edge_t, weight_t> mst_input;
  CSRDevice<vertex_t, edge_t, weight_t> csr_d;
  vertex_t v;
  edge_t e;
  int iterations;

  raft::resources handle;
};

// connected components tests
// a full MST is produced
const std::vector<MSTTestInput<int, int, float>> csr_in_h = {
  // single iteration
  {{{0, 3, 5, 7, 8}, {1, 2, 3, 0, 3, 0, 0, 1}, {2, 3, 4, 2, 1, 3, 4, 1}}, 0},

  //  multiple iterations and cycles
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {5.0f, 9.0f,  1.0f, 4.0f, 8.0f, 7.0f, 5.0f, 2.0f, 6.0f, 8.0f,
     1.0f, 10.0f, 9.0f, 2.0f, 1.0f, 1.0f, 6.0f, 4.0f, 7.0f, 10.0f}},
   1},
  // negative weights
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {-5.0f, -9.0f,  -1.0f, 4.0f,  -8.0f, -7.0f, -5.0f, -2.0f, -6.0f, -8.0f,
     -1.0f, -10.0f, -9.0f, -2.0f, -1.0f, -1.0f, -6.0f, 4.0f,  -7.0f, -10.0f}},
   0},

  // // equal weights
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2,
     0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1}},
   0},

  // // 1 - all equal weights
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}},
   0},

  // // 2 - all equal weights
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {2, 4, 5, 6, 3, 6, 0, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
     0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1}},
   0},

  // //self loop
  {{{0, 4, 6, 9, 12, 15, 17, 20},
    {0, 4, 5, 6, 3, 6, 2, 4, 5, 1, 4, 6, 0, 2, 3, 0, 2, 0, 1, 3},
    {0.5f, 9.0f,  1.0f, 4.0f, 8.0f, 7.0f, 0.5f, 2.0f, 6.0f, 8.0f,
     1.0f, 10.0f, 9.0f, 2.0f, 1.0f, 1.0f, 6.0f, 4.0f, 7.0f, 10.0f}},
   0}};

//  disconnected
const std::vector<CSRHost<int, int, float>> csr_in4_h = {
  {{0, 3, 5, 8, 10, 12, 14, 16},
   {2, 4, 5, 3, 6, 0, 4, 5, 1, 6, 0, 2, 0, 2, 1, 3},
   {5.0f,
    9.0f,
    1.0f,
    8.0f,
    7.0f,
    5.0f,
    2.0f,
    6.0f,
    8.0f,
    10.0f,
    9.0f,
    2.0f,
    1.0f,
    6.0f,
    7.0f,
    10.0f}}};

//  singletons
const std::vector<CSRHost<int, int, float>> csr_in5_h = {
  {{0, 3, 5, 8, 10, 10, 10, 12, 14, 16, 16},
   {2, 8, 7, 3, 8, 0, 8, 7, 1, 8, 0, 2, 0, 2, 1, 3},
   {5.0f,
    9.0f,
    1.0f,
    8.0f,
    7.0f,
    5.0f,
    2.0f,
    6.0f,
    8.0f,
    10.0f,
    9.0f,
    2.0f,
    1.0f,
    6.0f,
    7.0f,
    10.0f}}};

typedef MSTTest<int, int, float> MSTTestSequential;
TEST_P(MSTTestSequential, Sequential)
{
  auto results_pair          = mst_gpu();
  auto& symmetric_result     = results_pair.first;
  auto& non_symmetric_result = results_pair.second;

  // do assertions here
  // in this case, running sequential MST
  auto prims_result = prims(mst_input.csr_h);

  auto symmetric_sum = thrust::reduce(thrust::device,
                                      symmetric_result.weights.data(),
                                      symmetric_result.weights.data() + symmetric_result.n_edges);
  auto non_symmetric_sum =
    thrust::reduce(thrust::device,
                   non_symmetric_result.weights.data(),
                   non_symmetric_result.weights.data() + non_symmetric_result.n_edges);

  ASSERT_TRUE(raft::match(2 * prims_result, symmetric_sum, raft::CompareApprox<float>(0.1)));
  ASSERT_TRUE(raft::match(prims_result, non_symmetric_sum, raft::CompareApprox<float>(0.1)));
}

INSTANTIATE_TEST_SUITE_P(MSTTests, MSTTestSequential, ::testing::ValuesIn(csr_in_h));

}  // namespace mst
}  // namespace raft
