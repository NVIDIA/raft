/*
 * SPDX-FileCopyrightText: Copyright 2020 KETAN DATE & RAKESH NAGI
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright 2020 KETAN DATE & RAKESH NAGI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *      CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
 *      Authors: Ketan Date and Rakesh Nagi
 *
 *      Article reference:
 *          Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms
 *          for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
 *
 */
#pragma once

#include "../linear_assignment_types.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda/std/limits>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

#include <cstddef>
namespace raft {
namespace solver::detail {
const int DORMANT{0};
const int ACTIVE{1};
const int VISITED{2};
const int REVERSE{3};
const int AUGMENT{4};
const int MODIFIED{5};

template <typename weight_t>
bool __device__ near_zero(weight_t w, weight_t epsilon)
{
  return ((w > -epsilon) && (w < epsilon));
}

template <>
bool __device__ near_zero<int32_t>(int32_t w, int32_t epsilon)
{
  return (w == 0);
}

template <>
bool __device__ near_zero<int64_t>(int64_t w, int64_t epsilon)
{
  return (w == 0);
}

// Device function for traversing the neighbors from start pointer to end pointer and updating the
// covers. The function sets d_next to 4 if there are uncovered zeros, indicating the requirement of
// Step 4 execution.
template <typename vertex_t, typename weight_t>
__device__ void cover_and_expand_row(weight_t const* d_elements,
                                     weight_t const* d_row_duals,
                                     weight_t const* d_col_duals,
                                     weight_t* d_col_slacks,
                                     int* d_row_covers,
                                     int* d_col_covers,
                                     vertex_t const* d_col_assignments,
                                     bool* d_flag,
                                     vertex_t* d_row_parents,
                                     vertex_t* d_col_parents,
                                     int* d_row_visited,
                                     int* d_col_visited,
                                     vertex_t rowid,
                                     int spid,
                                     int colid,
                                     vertex_t N,
                                     weight_t epsilon)
{
  int ROWID = spid * N + rowid;
  int COLID = spid * N + colid;

  weight_t slack =
    d_elements[spid * N * N + rowid * N + colid] - d_row_duals[ROWID] - d_col_duals[COLID];

  int nxt_rowid = d_col_assignments[COLID];
  int NXT_ROWID = spid * N + nxt_rowid;

  if (rowid != nxt_rowid && d_col_covers[COLID] == 0) {
    if (slack < d_col_slacks[COLID]) {
      d_col_slacks[COLID]  = slack;
      d_col_parents[COLID] = ROWID;
    }

    if (near_zero(d_col_slacks[COLID], epsilon)) {
      if (nxt_rowid != -1) {
        d_row_parents[NXT_ROWID] = COLID;  // update parent info

        d_row_covers[NXT_ROWID] = 0;
        d_col_covers[COLID]     = 1;

        if (d_row_visited[NXT_ROWID] != VISITED) d_row_visited[NXT_ROWID] = ACTIVE;
      } else {
        d_col_visited[COLID] = REVERSE;
        *d_flag              = true;
      }
    }
  }
  d_row_visited[ROWID] = VISITED;
}

// Device function for traversing an alternating path from unassigned row to unassigned column.
template <typename vertex_t>
__device__ void __reverse_traversal(int* d_row_visited,
                                    vertex_t* d_row_children,
                                    vertex_t* d_col_children,
                                    vertex_t const* d_row_parents,
                                    vertex_t const* d_col_parents,
                                    int cur_colid)
{
  int cur_rowid = -1;

  while (cur_colid != -1) {
    d_col_children[cur_colid] = cur_rowid;
    cur_rowid                 = d_col_parents[cur_colid];

    d_row_children[cur_rowid] = cur_colid;
    cur_colid                 = d_row_parents[cur_rowid];
  }
  d_row_visited[cur_rowid] = AUGMENT;
}

// Device function for augmenting the alternating path from unassigned column to unassigned row.
template <typename vertex_t>
__device__ void __augment(vertex_t* d_row_assignments,
                          vertex_t* d_col_assignments,
                          vertex_t const* d_row_children,
                          vertex_t const* d_col_children,
                          vertex_t cur_rowid,
                          vertex_t N)
{
  int cur_colid = -1;

  while (cur_rowid != -1) {
    cur_colid = d_row_children[cur_rowid];

    d_row_assignments[cur_rowid] = cur_colid % N;
    d_col_assignments[cur_colid] = cur_rowid % N;

    cur_rowid = d_col_children[cur_colid];
  }
}

// Kernel for reducing the rows by subtracting row minimum from each row element.
template <typename vertex_t, typename weight_t>
RAFT_KERNEL kernel_rowReduction(weight_t const* d_costs, weight_t* d_row_duals, int SP, vertex_t N)
{
  int spid     = blockIdx.y * blockDim.y + threadIdx.y;
  int rowid    = blockIdx.x * blockDim.x + threadIdx.x;
  weight_t min = cuda::std::numeric_limits<weight_t>::max();

  if (spid < SP && rowid < N) {
    for (int colid = 0; colid < N; colid++) {
      weight_t slack = d_costs[spid * N * N + rowid * N + colid];

      if (slack < min) { min = slack; }
    }

    d_row_duals[spid * N + rowid] = min;
  }
}

// Kernel for reducing the column by subtracting column minimum from each column element.
template <typename vertex_t, typename weight_t>
RAFT_KERNEL kernel_columnReduction(
  weight_t const* d_costs, weight_t const* d_row_duals, weight_t* d_col_duals, int SP, vertex_t N)
{
  int spid  = blockIdx.y * blockDim.y + threadIdx.y;
  int colid = blockIdx.x * blockDim.x + threadIdx.x;

  weight_t min = cuda::std::numeric_limits<weight_t>::max();

  if (spid < SP && colid < N) {
    for (int rowid = 0; rowid < N; rowid++) {
      weight_t cost     = d_costs[spid * N * N + rowid * N + colid];
      weight_t row_dual = d_row_duals[spid * N + rowid];

      weight_t slack = cost - row_dual;

      if (slack < min) { min = slack; }
    }

    d_col_duals[spid * N + colid] = min;
  }
}

// Kernel for calculating initial assignments.
template <typename vertex_t, typename weight_t>
RAFT_KERNEL kernel_computeInitialAssignments(weight_t const* d_costs,
                                             weight_t const* d_row_duals,
                                             weight_t const* d_col_duals,
                                             vertex_t* d_row_assignments,
                                             vertex_t* d_col_assignments,
                                             int* d_row_lock,
                                             int* d_col_lock,
                                             int SP,
                                             vertex_t N,
                                             weight_t epsilon)
{
  int spid  = blockIdx.y * blockDim.y + threadIdx.y;
  int colid = blockIdx.x * blockDim.x + threadIdx.x;

  if (spid < SP && colid < N) {
    int overall_colid = spid * N + colid;
    weight_t col_dual = d_col_duals[overall_colid];

    for (vertex_t rowid = 0; rowid < N; rowid++) {
      int overall_rowid = spid * N + rowid;

      if (d_col_lock[overall_colid] == 1) break;

      weight_t cost     = d_costs[spid * N * N + rowid * N + colid];
      weight_t row_dual = d_row_duals[overall_rowid];
      weight_t slack    = cost - row_dual - col_dual;

      if (near_zero(slack, epsilon)) {
        if (atomicCAS(&d_row_lock[overall_rowid], 0, 1) == 0) {
          d_row_assignments[overall_rowid] = colid;
          d_col_assignments[overall_colid] = rowid;
          d_col_lock[overall_colid]        = 1;
        }
      }
    }
  }
}

// Kernel for populating the cover arrays and initializing alternating tree.
template <typename vertex_t>
RAFT_KERNEL kernel_computeRowCovers(
  vertex_t* d_row_assignments, int* d_row_covers, int* d_row_visited, int SP, vertex_t N)
{
  int spid  = blockIdx.y * blockDim.y + threadIdx.y;
  int rowid = blockIdx.x * blockDim.x + threadIdx.x;

  if (spid < SP && rowid < N) {
    int index = spid * N + rowid;

    if (d_row_assignments[index] != -1) {
      d_row_covers[index] = 1;
    } else {
      d_row_visited[index] = ACTIVE;
    }
  }
}

// Kernel for populating the predicate matrix for edges in row major format.
template <typename vertex_t>
RAFT_KERNEL kernel_rowPredicateConstructionCSR(
  bool* d_predicates, vertex_t* d_addresses, int* d_row_visited, int SP, vertex_t N)
{
  int spid  = blockIdx.y * blockDim.y + threadIdx.y;
  int rowid = blockIdx.x * blockDim.x + threadIdx.x;

  if (spid < SP && rowid < N) {
    int index = spid * N + rowid;

    if (d_row_visited[index] == ACTIVE) {
      d_predicates[index] = true;
      d_addresses[index]  = 1;
    } else {
      d_predicates[index] = false;
      d_addresses[index]  = 0;
    }
  }
}

// Kernel for scattering the edges based on the scatter addresses.
template <typename vertex_t>
RAFT_KERNEL kernel_rowScatterCSR(bool const* d_predicates,
                                 vertex_t const* d_addresses,
                                 vertex_t* d_neighbors,
                                 vertex_t* d_ptrs,
                                 vertex_t M,
                                 int SP,
                                 vertex_t N)
{
  int spid  = blockIdx.y * blockDim.y + threadIdx.y;
  int rowid = blockIdx.x * blockDim.x + threadIdx.x;

  if (spid < SP && rowid < N) {
    int index = spid * N + rowid;

    bool predicate  = d_predicates[index];
    vertex_t compid = d_addresses[index];

    if (predicate) { d_neighbors[compid] = rowid; }
    if (rowid == 0) {
      d_ptrs[spid] = compid;
      d_ptrs[SP]   = M;
    }
  }
}

// Kernel for finding the minimum zero cover.
template <typename vertex_t, typename weight_t>
RAFT_KERNEL kernel_coverAndExpand(bool* d_flag,
                                  vertex_t const* d_ptrs,
                                  vertex_t const* d_neighbors,
                                  weight_t const* d_elements,
                                  Vertices<vertex_t, weight_t> d_vertices,
                                  VertexData<vertex_t> d_row_data,
                                  VertexData<vertex_t> d_col_data,
                                  int SP,
                                  vertex_t N,
                                  weight_t epsilon)
{
  int spid  = blockIdx.y * blockDim.y + threadIdx.y;
  int colid = blockIdx.x * blockDim.x + threadIdx.x;

  // Load values into local memory

  if (spid < SP && colid < N) {
    thrust::for_each(
      thrust::seq,
      d_neighbors + d_ptrs[spid],
      d_neighbors + d_ptrs[spid + 1],
      [d_elements, d_vertices, d_flag, d_row_data, d_col_data, spid, colid, N, epsilon] __device__(
        vertex_t rowid) {
        cover_and_expand_row(d_elements,
                             d_vertices.row_duals,
                             d_vertices.col_duals,
                             d_vertices.col_slacks,
                             d_vertices.row_covers,
                             d_vertices.col_covers,
                             d_vertices.col_assignments,
                             d_flag,
                             d_row_data.parents,
                             d_col_data.parents,
                             d_row_data.is_visited,
                             d_col_data.is_visited,
                             rowid,
                             spid,
                             colid,
                             N,
                             epsilon);
      });
  }
}

// Kernel for constructing the predicates for reverse pass or augmentation candidates.
template <typename vertex_t>
RAFT_KERNEL kernel_augmentPredicateConstruction(bool* d_predicates,
                                                vertex_t* d_addresses,
                                                int* d_visited,
                                                int size)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < size) {
    int visited = d_visited[id];
    if ((visited == REVERSE) || (visited == AUGMENT)) {
      d_predicates[id] = true;
      d_addresses[id]  = 1;
    } else {
      d_predicates[id] = false;
      d_addresses[id]  = 0;
    }
  }
}

// Kernel for scattering the vertices based on the scatter addresses.
template <typename vertex_t>
RAFT_KERNEL kernel_augmentScatter(vertex_t* d_elements,
                                  bool const* d_predicates,
                                  vertex_t const* d_addresses,
                                  std::size_t size)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < size) {
    if (d_predicates[id]) { d_elements[d_addresses[id]] = id; }
  }
}

// Kernel for executing the reverse pass of the maximum matching algorithm.
template <typename vertex_t>
RAFT_KERNEL kernel_reverseTraversal(vertex_t* d_elements,
                                    VertexData<vertex_t> d_row_data,
                                    VertexData<vertex_t> d_col_data,
                                    int size)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < size) {
    __reverse_traversal(d_row_data.is_visited,
                        d_row_data.children,
                        d_col_data.children,
                        d_row_data.parents,
                        d_col_data.parents,
                        d_elements[id]);
  }
}

// Kernel for executing the augmentation pass of the maximum matching algorithm.
template <typename vertex_t>
RAFT_KERNEL kernel_augmentation(vertex_t* d_row_assignments,
                                vertex_t* d_col_assignments,
                                vertex_t const* d_row_elements,
                                VertexData<vertex_t> d_row_data,
                                VertexData<vertex_t> d_col_data,
                                vertex_t N,
                                vertex_t size)
{
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < size) {
    __augment(d_row_assignments,
              d_col_assignments,
              d_row_data.children,
              d_col_data.children,
              d_row_elements[id],
              N);
  }
}

// Kernel for updating the dual values in Step 5.
template <typename vertex_t, typename weight_t>
RAFT_KERNEL kernel_dualUpdate_1(
  weight_t* d_sp_min, weight_t const* d_col_slacks, int const* d_col_covers, int SP, vertex_t N)
{
  int spid = blockIdx.x * blockDim.x + threadIdx.x;

  if (spid < SP) {
    weight_t min = cuda::std::numeric_limits<weight_t>::max();
    for (int colid = 0; colid < N; colid++) {
      int index      = spid * N + colid;
      weight_t slack = d_col_slacks[index];
      int col_cover  = d_col_covers[index];

      if (col_cover == 0)
        if (slack < min) min = slack;
    }

    d_sp_min[spid] = min;
  }
}

// Kernel for updating the dual values in Step 5.
template <typename vertex_t, typename weight_t>
RAFT_KERNEL kernel_dualUpdate_2(weight_t const* d_sp_min,
                                weight_t* d_row_duals,
                                weight_t* d_col_duals,
                                weight_t* d_col_slacks,
                                int const* d_row_covers,
                                int const* d_col_covers,
                                int* d_row_visited,
                                vertex_t* d_col_parents,
                                int SP,
                                vertex_t N,
                                weight_t epsilon)
{
  int spid = blockIdx.y * blockDim.y + threadIdx.y;
  int id   = blockIdx.x * blockDim.x + threadIdx.x;

  if (spid < SP && id < N) {
    int index = spid * N + id;

    if (d_sp_min[spid] < cuda::std::numeric_limits<weight_t>::max()) {
      weight_t theta = d_sp_min[spid];
      int row_cover  = d_row_covers[index];
      int col_cover  = d_col_covers[index];

      if (row_cover == 0)  // Row vertex is reachable from source.
        d_row_duals[index] += theta;

      if (col_cover == 1)  // Col vertex is reachable from source.
        d_col_duals[index] -= theta;
      else {
        // Col vertex is unreachable from source.
        d_col_slacks[index] -= d_sp_min[spid];

        if (near_zero(d_col_slacks[index], epsilon)) {
          int par_rowid = d_col_parents[index];
          if (par_rowid != -1) d_row_visited[par_rowid] = ACTIVE;
        }
      }
    }
  }
}

// Kernel for calculating optimal objective function value using dual variables.
template <typename vertex_t, typename weight_t>
RAFT_KERNEL kernel_calcObjValDual(weight_t* d_obj_val_dual,
                                  weight_t const* d_row_duals,
                                  weight_t const* d_col_duals,
                                  int SP,
                                  vertex_t N)
{
  int spid = blockIdx.x * blockDim.x + threadIdx.x;

  if (spid < SP) {
    float val = 0;

    for (int i = 0; i < N; i++)
      val += (d_row_duals[spid * N + i] + d_col_duals[spid * N + i]);

    d_obj_val_dual[spid] = val;
  }
}

// Kernel for calculating optimal objective function value using dual variables.
template <typename vertex_t, typename weight_t>
RAFT_KERNEL kernel_calcObjValPrimal(weight_t* d_obj_val_primal,
                                    weight_t const* d_costs,
                                    vertex_t const* d_row_assignments,
                                    int SP,
                                    vertex_t N)
{
  int spid = blockIdx.x * blockDim.x + threadIdx.x;

  if (spid < SP) {
    weight_t val = 0;

    for (int i = 0; i < N; i++) {
      vertex_t j = d_row_assignments[spid * N + i];
      val += d_costs[spid * N * N + i * N + j];
    }

    d_obj_val_primal[spid] = val;
  }
}

}  // namespace solver::detail
}  // namespace raft
