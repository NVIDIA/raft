
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

namespace raft {
namespace sparse::solver {

template <typename vertex_t, typename edge_t, typename weight_t>
struct Graph_COO {
  rmm::device_uvector<vertex_t> src;
  rmm::device_uvector<vertex_t> dst;
  rmm::device_uvector<weight_t> weights;
  edge_t n_edges;

  Graph_COO(vertex_t size, cudaStream_t stream)
    : src(size, stream), dst(size, stream), weights(size, stream)
  {
  }
};

/**
 * @brief MST solver with deterministic tie-breaking.
 *
 * @tparam vertex_t integral type for vertex indexing (32- or 64-bit)
 * @tparam edge_t integral type for edge indexing (32- or 64-bit, at least as
 * wide as vertex_t)
 * @tparam weight_t type of the weights array
 * @tparam alteration_t unused; retained for source compatibility with
 * existing callers (the solver no longer alters weights to break ties)
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
class MST_solver {
 public:
  MST_solver(raft::resources const& handle_,
             const edge_t* offsets_,
             const vertex_t* indices_,
             const weight_t* weights_,
             const vertex_t v_,
             const edge_t e_,
             vertex_t* color_,
             cudaStream_t stream_,
             bool symmetrize_output_,
             bool initialize_colors_,
             int iterations_);

  Graph_COO<vertex_t, edge_t, weight_t> solve();

  ~MST_solver() {}

 private:
  raft::resources const& handle;
  cudaStream_t stream;
  bool symmetrize_output, initialize_colors;
  int iterations;

  // CSR
  const edge_t* offsets;
  const vertex_t* indices;
  const weight_t* weights;
  const vertex_t v;
  const edge_t e;

  vertex_t* color_index;  // user-provided output colors array (one color per vertex)
};

}  // namespace sparse::solver
}  // namespace raft

#include <raft/sparse/solver/detail/mst_solver_inl.cuh>
