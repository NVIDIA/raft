
/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/sparse/solver/mst_solver.cuh>

namespace raft {
namespace sparse::solver {

/**
 * Compute the minimum spanning tree (MST) or minimum spanning forest (MSF) depending on
 * the connected components of the given graph.
 * Algorithm based on ECL-MST (Fallin, Gonzalez, Seo, Burtscher, SC'23).
 *
 * Usage example:
 * @code{.cpp}
 * #include <raft/core/resources.hpp>
 * #include <raft/core/resource/cuda_stream.hpp>
 * #include <raft/sparse/solver/mst.cuh>
 *
 * raft::resources res;
 * auto stream = raft::resource::get_cuda_stream(res);
 * // device CSR of a symmetric graph: offsets (size v+1), indices and weights (size e)
 * rmm::device_uvector<int> colors(v, stream);
 * auto forest = raft::sparse::solver::mst<int, int, float>(
 *   res, offsets, indices, weights, v, e, colors.data(), stream);
 * // forest.src/dst/weights hold forest.n_edges edges (both directions when
 * // symmetrize_output); colors[i] = component id of vertex i
 * @endcode
 *
 * @tparam vertex_t integral type for precision of vertex indexing
 * @tparam edge_t integral type for precision of edge indexing
 * @tparam weight_t type of weights array
 * @tparam alteration_t unused; retained for source compatibility (the solver
 * breaks weight ties deterministically by edge index and no longer alters
 * weights)
 *
 * @param handle
 * @param offsets csr indptr array of row offsets (size v+1, symmetric input required: each
 * undirected edge must be stored in both directions with equal weights)
 * @param indices csr array of column indices (size e, each in [0, v))
 * @param weights csr array of weights (size e)
 * @param v number of vertices in graph
 * @param e number of edges in graph
 * @param color array to store resulting colors for MSF; when initialize_colors is false it is
 * also the input seeding and must hold a valid component labeling from a previous solve of the same
 * graph
 * @param stream cuda stream for ordering operations
 * @param symmetrize_output should the resulting output edge list be symmetrized?
 * @param initialize_colors should the colors array be initialized inside the MST?
 * @param iterations maximum number of Boruvka rounds to perform (values <= 0 solve to
 * completion).
 * Bounded solves run textbook Boruvka rounds (the internal two-phase edge filter is disabled),
 * so partial results are deterministic and can be resumed exactly via initialize_colors=false
 * @return a list of edges containing the mst (or a subset of the edges guaranteed to be in the mst
 * when an msf is encountered)
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t = weight_t>
Graph_COO<vertex_t, edge_t, weight_t> mst(raft::resources const& handle,
                                          edge_t const* offsets,
                                          vertex_t const* indices,
                                          weight_t const* weights,
                                          vertex_t const v,
                                          edge_t const e,
                                          vertex_t* color,
                                          cudaStream_t stream,
                                          bool symmetrize_output = true,
                                          bool initialize_colors = true,
                                          int iterations         = 0)
{
  MST_solver<vertex_t, edge_t, weight_t, alteration_t> mst_solver(handle,
                                                                  offsets,
                                                                  indices,
                                                                  weights,
                                                                  v,
                                                                  e,
                                                                  color,
                                                                  stream,
                                                                  symmetrize_output,
                                                                  initialize_colors,
                                                                  iterations);
  return mst_solver.solve();
}

}  // end namespace sparse::solver
}  // end namespace raft
