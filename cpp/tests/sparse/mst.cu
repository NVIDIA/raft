/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/mst.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/memory.h>
#include <thrust/reduce.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <tuple>
#include <utility>
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

// Build a symmetric CSR from an undirected edge list
template <typename vertex_t, typename edge_t, typename weight_t>
CSRHost<vertex_t, edge_t, weight_t> csr_from_undirected_edges(
  vertex_t v, const std::vector<std::tuple<vertex_t, vertex_t, weight_t>>& edges)
{
  std::vector<std::vector<std::pair<vertex_t, weight_t>>> adj(v);
  for (auto& [s, d, w] : edges) {
    adj[s].push_back({d, w});
    adj[d].push_back({s, w});
  }
  CSRHost<vertex_t, edge_t, weight_t> csr;
  csr.offsets.push_back(0);
  for (vertex_t i = 0; i < v; i++) {
    std::sort(adj[i].begin(), adj[i].end());
    for (auto& [d, w] : adj[i]) {
      csr.indices.push_back(d);
      csr.weights.push_back(w);
    }
    csr.offsets.push_back(static_cast<edge_t>(csr.indices.size()));
  }
  return csr;
}

// Kruskal oracle with the solver's tie-break (weight, then CSR edge index);
// also produces the expected MSF colors (min vertex id per component).
// Weights must not contain NaN (the sort comparator requires a total order).
struct KruskalResult {
  double weight;
  int n_edges;
  std::vector<int> colors;
};

template <typename vertex_t, typename edge_t, typename weight_t>
KruskalResult kruskal_mst(const CSRHost<vertex_t, edge_t, weight_t>& csr_h)
{
  const vertex_t v = static_cast<vertex_t>(csr_h.offsets.size() - 1);
  const edge_t e   = static_cast<edge_t>(csr_h.indices.size());

  std::vector<vertex_t> row_of(e);
  std::vector<edge_t> order;
  for (vertex_t r = 0; r < v; r++) {
    for (edge_t j = csr_h.offsets[r]; j < csr_h.offsets[r + 1]; j++) {
      row_of[j] = r;
      if (csr_h.indices[j] > r) order.push_back(j);  // canonical direction, skips self-loops
    }
  }
  std::sort(order.begin(), order.end(), [&](edge_t a, edge_t b) {
    if (csr_h.weights[a] != csr_h.weights[b]) return csr_h.weights[a] < csr_h.weights[b];
    return a < b;
  });

  std::vector<vertex_t> parent(v);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](vertex_t x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x         = parent[x];
    }
    return x;
  };

  KruskalResult res{0.0, 0, {}};
  for (edge_t j : order) {
    vertex_t a = find(row_of[j]);
    vertex_t b = find(csr_h.indices[j]);
    if (a != b) {
      parent[std::max(a, b)] = std::min(a, b);  // root of every set is its min vertex id
      res.weight += static_cast<double>(csr_h.weights[j]);
      res.n_edges++;
    }
  }
  res.colors.resize(v);
  for (vertex_t i = 0; i < v; i++) {
    res.colors[i] = find(i);
  }
  return res;
}

template <typename weight_t>
struct MSTResultHost {
  std::vector<int> src, dst;
  std::vector<weight_t> weights;
  std::vector<int> colors;
  int n_edges;

  double total_weight() const
  {
    double sum = 0.0;
    for (int i = 0; i < n_edges; i++) {
      sum += static_cast<double>(weights[i]);
    }
    return sum;
  }

  std::vector<std::tuple<int, int, weight_t>> sorted_edges() const
  {
    std::vector<std::tuple<int, int, weight_t>> out;
    for (int i = 0; i < n_edges; i++) {
      out.push_back({src[i], dst[i], weights[i]});
    }
    std::sort(out.begin(), out.end());
    return out;
  }
};

// colors_in seeds resume runs (initialize_colors = false)
template <typename weight_t>
MSTResultHost<weight_t> run_mst_gpu(raft::resources const& handle,
                                    const CSRHost<int, int, weight_t>& csr_h,
                                    bool symmetrize_output,
                                    bool initialize_colors,
                                    int iterations,
                                    const std::vector<int>* colors_in = nullptr)
{
  auto stream = resource::get_cuda_stream(handle);
  const int v = static_cast<int>(csr_h.offsets.size() - 1);
  const int e = static_cast<int>(csr_h.indices.size());

  rmm::device_uvector<int> offsets_d(v + 1, stream);
  rmm::device_uvector<int> indices_d(e, stream);
  rmm::device_uvector<weight_t> weights_d(e, stream);
  rmm::device_uvector<int> colors_d(v, stream);
  raft::update_device(offsets_d.data(), csr_h.offsets.data(), v + 1, stream);
  raft::update_device(indices_d.data(), csr_h.indices.data(), e, stream);
  raft::update_device(weights_d.data(), csr_h.weights.data(), e, stream);
  if (colors_in != nullptr) { raft::update_device(colors_d.data(), colors_in->data(), v, stream); }

  raft::sparse::solver::MST_solver<int, int, weight_t, weight_t> solver(handle,
                                                                        offsets_d.data(),
                                                                        indices_d.data(),
                                                                        weights_d.data(),
                                                                        v,
                                                                        e,
                                                                        colors_d.data(),
                                                                        stream,
                                                                        symmetrize_output,
                                                                        initialize_colors,
                                                                        iterations);
  auto result = solver.solve();

  MSTResultHost<weight_t> out;
  out.n_edges = result.n_edges;
  out.src.resize(result.n_edges);
  out.dst.resize(result.n_edges);
  out.weights.resize(result.n_edges);
  out.colors.resize(v);
  raft::update_host(out.src.data(), result.src.data(), result.n_edges, stream);
  raft::update_host(out.dst.data(), result.dst.data(), result.n_edges, stream);
  raft::update_host(out.weights.data(), result.weights.data(), result.n_edges, stream);
  raft::update_host(out.colors.data(), colors_d.data(), v, stream);
  resource::sync_stream(handle, stream);
  return out;
}

// Connected random graph: ring + deterministic pseudo-random chords
template <typename weight_t, typename weight_fn_t>
CSRHost<int, int, weight_t> make_ring_plus_chords(int v,
                                                  int chords_per_vertex,
                                                  weight_fn_t weight_of)
{
  std::vector<std::tuple<int, int, weight_t>> edges;
  std::set<std::pair<int, int>> seen;
  std::mt19937 rng(42);
  auto add = [&](int a, int b) {
    if (a == b) return;
    auto p = std::minmax(a, b);
    if (seen.insert({p.first, p.second}).second) {
      edges.push_back({p.first, p.second, weight_of(static_cast<int>(edges.size()))});
    }
  };
  for (int i = 0; i < v; i++) {
    add(i, (i + 1) % v);
  }
  std::uniform_int_distribution<int> pick(0, v - 1);
  for (int i = 0; i < v; i++) {
    for (int c = 0; c < chords_per_vertex; c++) {
      add(i, pick(rng));
    }
  }
  return csr_from_undirected_edges<int, int, weight_t>(v, edges);
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

    if (iterations == 0) {
      raft::sparse::solver::MST_solver<vertex_t, edge_t, weight_t, float> symmetric_solver(
        handle,
        offsets,
        indices,
        weights,
        v,
        e,
        color_ptr,
        resource::get_cuda_stream(handle),
        true,
        true,
        0);
      auto symmetric_result = symmetric_solver.solve();

      raft::sparse::solver::MST_solver<vertex_t, edge_t, weight_t, float> non_symmetric_solver(
        handle,
        offsets,
        indices,
        weights,
        v,
        e,
        color_ptr,
        resource::get_cuda_stream(handle),
        false,
        true,
        0);
      auto non_symmetric_result = non_symmetric_solver.solve();

      EXPECT_LE(symmetric_result.n_edges, 2 * v - 2);
      EXPECT_LE(non_symmetric_result.n_edges, v - 1);

      return std::make_pair(std::move(symmetric_result), std::move(non_symmetric_result));
    } else {
      raft::sparse::solver::MST_solver<vertex_t, edge_t, weight_t, float> intermediate_solver(
        handle,
        offsets,
        indices,
        weights,
        v,
        e,
        color_ptr,
        resource::get_cuda_stream(handle),
        true,
        true,
        iterations);
      auto intermediate_result = intermediate_solver.solve();

      raft::sparse::solver::MST_solver<vertex_t, edge_t, weight_t, float> symmetric_solver(
        handle,
        offsets,
        indices,
        weights,
        v,
        e,
        color_ptr,
        resource::get_cuda_stream(handle),
        true,
        false,
        0);
      auto symmetric_result = symmetric_solver.solve();

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

      raft::sparse::solver::MST_solver<vertex_t, edge_t, weight_t, float> non_symmetric_solver(
        handle,
        offsets,
        indices,
        weights,
        v,
        e,
        color_ptr,
        resource::get_cuda_stream(handle),
        false,
        true,
        0);
      auto non_symmetric_result = non_symmetric_solver.solve();

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

void expect_forest(int v, const std::vector<int>& src, const std::vector<int>& dst, int n_edges)
{
  std::vector<int> parent(v);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x         = parent[x];
    }
    return x;
  };
  for (int i = 0; i < n_edges; i++) {
    int a = find(src[i]);
    int b = find(dst[i]);
    ASSERT_NE(a, b) << "cycle introduced by edge " << src[i] << " -> " << dst[i];
    parent[std::max(a, b)] = std::min(a, b);
  }
}

// Regression: heavily-tied integer-valued float weights at ~1e7, where the
// previous solver's alteration rounded away (float ulp there is 1-2) and
// produced cycles or non-minimal forests past its own guard.
TEST(MST, FloatMagnitudeTies)
{
  raft::resources handle;
  const int v = 4096;
  auto csr_h  = make_ring_plus_chords<float>(
    v, 4, [](int idx) { return 1.0e7f + static_cast<float>(idx % 2); });
  auto truth = kruskal_mst(csr_h);

  auto result = run_mst_gpu<float>(handle, csr_h, false, true, 0);

  ASSERT_EQ(truth.n_edges, result.n_edges);
  ASSERT_EQ(v - 1, result.n_edges);  // ring keeps it connected
  // integer-valued weights, sum < 2^53: exact compare
  ASSERT_EQ(truth.weight, result.total_weight());
  expect_forest(v, result.src, result.dst, result.n_edges);
  for (int i = 0; i < v; i++) {
    ASSERT_EQ(0, result.colors[i]);  // connected: min vertex id everywhere
  }
}

// symmetrize_output must emit every edge in both directions and agree with
// the non-symmetrized solve
TEST(MST, SymmetrizeOutput)
{
  raft::resources handle;
  const int v = 1024;
  auto csr_h  = make_ring_plus_chords<float>(
    v, 4, [v](int idx) { return static_cast<float>(1 + (idx * 7919) % v); });
  auto truth = kruskal_mst(csr_h);

  auto sym     = run_mst_gpu<float>(handle, csr_h, true, true, 0);
  auto non_sym = run_mst_gpu<float>(handle, csr_h, false, true, 0);

  ASSERT_EQ(non_sym.n_edges, truth.n_edges);
  ASSERT_EQ(sym.n_edges, 2 * truth.n_edges);
  ASSERT_EQ(truth.weight, non_sym.total_weight());
  ASSERT_EQ(2 * truth.weight, sym.total_weight());

  std::vector<std::tuple<int, int, float>> expected;
  for (int i = 0; i < non_sym.n_edges; i++) {
    expected.push_back({non_sym.src[i], non_sym.dst[i], non_sym.weights[i]});
    expected.push_back({non_sym.dst[i], non_sym.src[i], non_sym.weights[i]});
  }
  std::sort(expected.begin(), expected.end());
  ASSERT_EQ(expected, sym.sorted_edges());
}

// bounded solve + resume must reproduce the one-shot result exactly
TEST(MST, ResumeMatchesOneShot)
{
  raft::resources handle;
  const int v = 2048;
  auto csr_h  = make_ring_plus_chords<float>(
    v, 4, [v](int idx) { return static_cast<float>(1 + (idx * 7919) % v); });

  auto one_shot = run_mst_gpu<float>(handle, csr_h, false, true, 0);

  auto partial = run_mst_gpu<float>(handle, csr_h, false, true, 2);
  ASSERT_LT(partial.n_edges, one_shot.n_edges);  // 2 rounds must not finish this graph
  auto resumed = run_mst_gpu<float>(handle, csr_h, false, false, 0, &partial.colors);

  ASSERT_EQ(one_shot.n_edges, partial.n_edges + resumed.n_edges);
  auto combined = partial.sorted_edges();
  auto rest     = resumed.sorted_edges();
  combined.insert(combined.end(), rest.begin(), rest.end());
  std::sort(combined.begin(), combined.end());
  ASSERT_EQ(one_shot.sorted_edges(), combined);
  ASSERT_EQ(one_shot.colors, resumed.colors);
}

// disconnected MSF: n_edges = v - #components, colors = min id per component
TEST(MST, DisconnectedColors)
{
  raft::resources handle;
  for (const auto& csr_h : {csr_in4_h[0], csr_in5_h[0]}) {
    auto truth  = kruskal_mst(csr_h);
    auto result = run_mst_gpu<float>(handle, csr_h, false, true, 0);

    ASSERT_EQ(truth.n_edges, result.n_edges);
    ASSERT_EQ(truth.weight, result.total_weight());
    ASSERT_EQ(truth.colors, result.colors);
    expect_forest(
      static_cast<int>(csr_h.offsets.size() - 1), result.src, result.dst, result.n_edges);
  }
}

// Regression: an equal-weight path graph chain-joins into an O(v)-deep
// parent chain; quadratic (minutes at v=1M) without path halving.
TEST(MST, UniformWeightPathGraph)
{
  raft::resources handle;
  const int v = 1000000;
  std::vector<std::tuple<int, int, float>> edges;
  for (int i = 0; i + 1 < v; i++) {
    edges.push_back({i, i + 1, 1.0f});
  }
  auto csr_h = csr_from_undirected_edges<int, int, float>(v, edges);

  auto result = run_mst_gpu<float>(handle, csr_h, false, true, 0);

  ASSERT_EQ(v - 1, result.n_edges);
  ASSERT_EQ(static_cast<double>(v - 1), result.total_weight());
  for (int i = 0; i < v; i++) {
    ASSERT_EQ(0, result.colors[i]);
  }
}

// double weights exercise the wide (8-byte key) kernel path
TEST(MST, DoubleWeights)
{
  raft::resources handle;
  const int v = 2048;
  auto csr_h  = make_ring_plus_chords<double>(
    v, 4, [v](int idx) { return static_cast<double>(1 + (idx * 7919) % v); });
  auto truth = kruskal_mst(csr_h);

  auto result = run_mst_gpu<double>(handle, csr_h, false, true, 0);

  ASSERT_EQ(truth.n_edges, result.n_edges);
  ASSERT_EQ(truth.weight, result.total_weight());
  expect_forest(v, result.src, result.dst, result.n_edges);
}

// int32 weights exercise the integer order-key overload
TEST(MST, IntegerWeights)
{
  raft::resources handle;
  const int v = 1024;
  auto csr_f  = make_ring_plus_chords<float>(
    v, 4, [v](int idx) { return static_cast<float>(1 + (idx * 7919) % v); });
  auto truth = kruskal_mst(csr_f);

  CSRHost<int, int, int32_t> csr_i;
  csr_i.offsets = csr_f.offsets;
  csr_i.indices = csr_f.indices;
  csr_i.weights.assign(csr_f.weights.begin(), csr_f.weights.end());
  auto result = run_mst_gpu<int32_t>(handle, csr_i, false, true, 0);

  ASSERT_EQ(truth.n_edges, result.n_edges);
  ASSERT_EQ(truth.weight, result.total_weight());
  ASSERT_EQ(truth.colors, result.colors);
}

// wide, mixed, and unsigned instantiations must match the 32-bit forest exactly
TEST(MST, Int64Indices)
{
  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);
  const int v = 4096;
  auto csr32  = make_ring_plus_chords<float>(
    v, 4, [v](int idx) { return static_cast<float>(1 + (idx * 7919) % v); });
  auto truth  = kruskal_mst(csr32);
  const int e = static_cast<int>(csr32.indices.size());

  auto run_typed = [&](auto vertex_tag, auto edge_tag, auto weight_tag) {
    using vertex2_t = decltype(vertex_tag);
    using edge2_t   = decltype(edge_tag);
    using weight_t  = decltype(weight_tag);
    std::vector<edge2_t> off_h(csr32.offsets.begin(), csr32.offsets.end());
    std::vector<vertex2_t> ind_h(csr32.indices.begin(), csr32.indices.end());
    std::vector<weight_t> w_h(csr32.weights.begin(), csr32.weights.end());

    rmm::device_uvector<edge2_t> off_d(v + 1, stream);
    rmm::device_uvector<vertex2_t> ind_d(e, stream);
    rmm::device_uvector<weight_t> w_d(e, stream);
    rmm::device_uvector<vertex2_t> col_d(v, stream);
    raft::update_device(off_d.data(), off_h.data(), v + 1, stream);
    raft::update_device(ind_d.data(), ind_h.data(), e, stream);
    raft::update_device(w_d.data(), w_h.data(), e, stream);

    auto res =
      raft::sparse::solver::mst<vertex2_t, edge2_t, weight_t, weight_t>(handle,
                                                                        off_d.data(),
                                                                        ind_d.data(),
                                                                        w_d.data(),
                                                                        static_cast<vertex2_t>(v),
                                                                        static_cast<edge2_t>(e),
                                                                        col_d.data(),
                                                                        stream,
                                                                        false,
                                                                        true,
                                                                        0);

    std::vector<weight_t> w_out(res.n_edges);
    std::vector<vertex2_t> col_out(v);
    raft::update_host(w_out.data(), res.weights.data(), res.n_edges, stream);
    raft::update_host(col_out.data(), col_d.data(), v, stream);
    resource::sync_stream(handle, stream);

    ASSERT_EQ(truth.n_edges, static_cast<int>(res.n_edges));
    double sum = 0.0;
    for (auto w : w_out)
      sum += static_cast<double>(w);
    ASSERT_EQ(truth.weight, sum);
    for (int i = 0; i < v; i++) {
      ASSERT_EQ(static_cast<vertex2_t>(truth.colors[i]), col_out[i]);
    }
  };

  run_typed(int64_t{}, int64_t{}, float{});    // uniform 64-bit, widened-key wide path
  run_typed(int64_t{}, int64_t{}, double{});   // uniform 64-bit, full 128-bit key
  run_typed(int32_t{}, int64_t{}, float{});    // mixed 32-bit vertex_t, 64-bit edge_t
  run_typed(uint32_t{}, int64_t{}, float{});   // unsigned vertex_t on the wide path
  run_typed(uint32_t{}, uint32_t{}, float{});  // unsigned narrow path (below INT_MAX)
}

}  // namespace mst
}  // namespace raft
