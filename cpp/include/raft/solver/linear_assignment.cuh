/*
 * SPDX-FileCopyrightText: Copyright 2020 KETAN DATE & RAKESH NAGI
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
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

#ifndef __LAP_H
#define __LAP_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/solver/detail/lap_functions.cuh>
#include <raft/solver/linear_assignment_types.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>

namespace raft {
namespace solver {

/**
 * @brief CUDA Implementation of O(n^3) alternating tree Hungarian Algorithm
 * @note This is a port to RAFT from original authors Ketan Date and Rakesh Nagi
 *
 * @see Date, Ketan, and Rakesh Nagi. "GPU-accelerated Hungarian algorithms
 *          for the Linear Assignment Problem." Parallel Computing 57 (2016): 52-72.
 *
 * @tparam vertex_t
 * @tparam weight_t
 */
template <typename vertex_t, typename weight_t>
class LinearAssignmentProblem {
 private:
  vertex_t size_;
  vertex_t batchsize_;
  weight_t epsilon_;

  weight_t const* d_costs_;

  Vertices<vertex_t, weight_t> d_vertices_dev;
  VertexData<vertex_t> d_row_data_dev, d_col_data_dev;

  raft::resources const& handle_;
  rmm::device_uvector<int> row_covers_v;
  rmm::device_uvector<int> col_covers_v;
  rmm::device_uvector<weight_t> row_duals_v;
  rmm::device_uvector<weight_t> col_duals_v;
  rmm::device_uvector<weight_t> col_slacks_v;
  rmm::device_uvector<int> row_is_visited_v;
  rmm::device_uvector<int> col_is_visited_v;
  rmm::device_uvector<vertex_t> row_parents_v;
  rmm::device_uvector<vertex_t> col_parents_v;
  rmm::device_uvector<vertex_t> row_children_v;
  rmm::device_uvector<vertex_t> col_children_v;
  rmm::device_uvector<weight_t> obj_val_primal_v;
  rmm::device_uvector<weight_t> obj_val_dual_v;

 public:
  /**
   * @brief Constructor
   * @param handle raft handle for managing resources
   * @param size size of square matrix
   * @param batchsize
   * @param epsilon
   */
  LinearAssignmentProblem(raft::resources const& handle,
                          vertex_t size,
                          vertex_t batchsize,
                          weight_t epsilon)
    : handle_(handle),
      size_(size),
      batchsize_(batchsize),
      epsilon_(epsilon),
      d_costs_(nullptr),
      row_covers_v(0, resource::get_cuda_stream(handle_)),
      col_covers_v(0, resource::get_cuda_stream(handle_)),
      row_duals_v(0, resource::get_cuda_stream(handle_)),
      col_duals_v(0, resource::get_cuda_stream(handle_)),
      col_slacks_v(0, resource::get_cuda_stream(handle_)),
      row_is_visited_v(0, resource::get_cuda_stream(handle_)),
      col_is_visited_v(0, resource::get_cuda_stream(handle_)),
      row_parents_v(0, resource::get_cuda_stream(handle_)),
      col_parents_v(0, resource::get_cuda_stream(handle_)),
      row_children_v(0, resource::get_cuda_stream(handle_)),
      col_children_v(0, resource::get_cuda_stream(handle_)),
      obj_val_primal_v(0, resource::get_cuda_stream(handle_)),
      obj_val_dual_v(0, resource::get_cuda_stream(handle_))
  {
  }

  /**
   * Executes Hungarian algorithm on the input cost matrix.
   * @param d_cost_matrix
   * @param d_row_assignment
   * @param d_col_assignment
   */
  void solve(weight_t const* d_cost_matrix, vertex_t* d_row_assignment, vertex_t* d_col_assignment)
  {
    initializeDevice();

    d_vertices_dev.row_assignments = d_row_assignment;
    d_vertices_dev.col_assignments = d_col_assignment;

    d_costs_ = d_cost_matrix;

    int step = 0;

    while (step != 100) {
      switch (step) {
        case 0: step = hungarianStep0(); break;
        case 1: step = hungarianStep1(); break;
        case 2: step = hungarianStep2(); break;
        case 3: step = hungarianStep3(); break;
        case 4: step = hungarianStep4(); break;
        case 5: step = hungarianStep5(); break;
        case 6: step = hungarianStep6(); break;
      }
    }

    d_costs_ = nullptr;
  }

  /**
   * Function for getting optimal row dual vector for subproblem spId.
   * @param spId
   * @return
   */
  std::pair<const weight_t*, vertex_t> getRowDualVector(int spId) const
  {
    return std::make_pair(row_duals_v.data() + spId * size_, size_);
  }

  /**
   * Function for getting optimal col dual vector for subproblem spId.
   * @param spId
   * @return
   */
  std::pair<const weight_t*, vertex_t> getColDualVector(int spId)
  {
    return std::make_pair(col_duals_v.data() + spId * size_, size_);
  }

  /**
   * Function for getting optimal primal objective value for subproblem spId.
   * @param spId
   * @return
   */
  weight_t getPrimalObjectiveValue(int spId)
  {
    weight_t result;
    raft::update_host(
      &result, obj_val_primal_v.data() + spId, 1, resource::get_cuda_stream(handle_));
    RAFT_CHECK_CUDA(resource::get_cuda_stream(handle_));
    return result;
  }

  /**
   * Function for getting optimal dual objective value for subproblem spId.
   * @param spId
   * @return
   */
  weight_t getDualObjectiveValue(int spId)
  {
    weight_t result;
    raft::update_host(&result, obj_val_dual_v.data() + spId, 1, resource::get_cuda_stream(handle_));
    RAFT_CHECK_CUDA(resource::get_cuda_stream(handle_).get());
    return result;
  }

 private:
  // Helper function for initializing global variables and arrays on a single host.
  void initializeDevice()
  {
    cudaStream_t stream = resource::get_cuda_stream(handle_).get();
    row_covers_v.resize(batchsize_ * size_, stream);
    col_covers_v.resize(batchsize_ * size_, stream);
    row_duals_v.resize(batchsize_ * size_, stream);
    col_duals_v.resize(batchsize_ * size_, stream);
    col_slacks_v.resize(batchsize_ * size_, stream);
    row_is_visited_v.resize(batchsize_ * size_, stream);
    col_is_visited_v.resize(batchsize_ * size_, stream);
    row_parents_v.resize(batchsize_ * size_, stream);
    col_parents_v.resize(batchsize_ * size_, stream);
    row_children_v.resize(batchsize_ * size_, stream);
    col_children_v.resize(batchsize_ * size_, stream);
    obj_val_primal_v.resize(batchsize_, stream);
    obj_val_dual_v.resize(batchsize_, stream);

    d_vertices_dev.row_covers = row_covers_v.data();
    d_vertices_dev.col_covers = col_covers_v.data();

    d_vertices_dev.row_duals  = row_duals_v.data();
    d_vertices_dev.col_duals  = col_duals_v.data();
    d_vertices_dev.col_slacks = col_slacks_v.data();

    d_row_data_dev.is_visited = row_is_visited_v.data();
    d_col_data_dev.is_visited = col_is_visited_v.data();
    d_row_data_dev.parents    = row_parents_v.data();
    d_row_data_dev.children   = row_children_v.data();
    d_col_data_dev.parents    = col_parents_v.data();
    d_col_data_dev.children   = col_children_v.data();

    thrust::fill(thrust::device, row_covers_v.begin(), row_covers_v.end(), int{0});
    thrust::fill(thrust::device, col_covers_v.begin(), col_covers_v.end(), int{0});
    thrust::fill(thrust::device, row_duals_v.begin(), row_duals_v.end(), weight_t{0});
    thrust::fill(thrust::device, col_duals_v.begin(), col_duals_v.end(), weight_t{0});
  }

  // Function for calculating initial zeros by subtracting row and column minima from each element.
  int hungarianStep0()
  {
    detail::initialReduction(handle_, d_costs_, d_vertices_dev, batchsize_, size_);

    return 1;
  }

  // Function for calculating initial zeros by subtracting row and column minima from each element.
  int hungarianStep1()
  {
    detail::computeInitialAssignments(
      handle_, d_costs_, d_vertices_dev, batchsize_, size_, epsilon_);

    int next = 2;

    while (true) {
      if ((next = hungarianStep2()) == 6) break;

      if ((next = hungarianStep3()) == 5) break;

      hungarianStep4();
    }

    return next;
  }

  // Function for checking optimality and constructing predicates and covers.
  int hungarianStep2()
  {
    int cover_count = detail::computeRowCovers(
      handle_, d_vertices_dev, d_row_data_dev, d_col_data_dev, batchsize_, size_);

    int next = (cover_count == batchsize_ * size_) ? 6 : 3;

    return next;
  }

  // Function for building alternating tree rooted at unassigned rows.
  int hungarianStep3()
  {
    int next;

    rmm::device_scalar<bool> flag_v(resource::get_cuda_stream(handle_));

    bool h_flag = false;
    flag_v.set_value_async(h_flag, resource::get_cuda_stream(handle_));

    detail::executeZeroCover(handle_,
                             d_costs_,
                             d_vertices_dev,
                             d_row_data_dev,
                             d_col_data_dev,
                             flag_v.data(),
                             batchsize_,
                             size_,
                             epsilon_);

    h_flag = flag_v.value(resource::get_cuda_stream(handle_));

    next = h_flag ? 4 : 5;

    return next;
  }

  // Function for augmenting the solution along multiple node-disjoint alternating trees.
  int hungarianStep4()
  {
    detail::reversePass(handle_, d_row_data_dev, d_col_data_dev, batchsize_, size_);

    detail::augmentationPass(
      handle_, d_vertices_dev, d_row_data_dev, d_col_data_dev, batchsize_, size_);

    return 2;
  }

  // Function for updating dual solution to introduce new zero-cost arcs.
  int hungarianStep5()
  {
    detail::dualUpdate(
      handle_, d_vertices_dev, d_row_data_dev, d_col_data_dev, batchsize_, size_, epsilon_);

    return 3;
  }

  // Function for calculating primal and dual objective values at optimality.
  int hungarianStep6()
  {
    detail::calcObjValPrimal(handle_,
                             obj_val_primal_v.data(),
                             d_costs_,
                             d_vertices_dev.row_assignments,
                             batchsize_,
                             size_);

    detail::calcObjValDual(handle_, obj_val_dual_v.data(), d_vertices_dev, batchsize_, size_);

    return 100;
  }
};

}  // namespace solver
}  // namespace raft

#endif
