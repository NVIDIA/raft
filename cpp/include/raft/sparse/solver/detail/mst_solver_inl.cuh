/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/sparse/solver/detail/mst_kernels.cuh>
#include <raft/sparse/solver/detail/mst_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <curand.h>

#include <iostream>

namespace raft {
namespace sparse::solver {

// curand generator uniform
inline curandStatus_t curand_generate_uniformX(curandGenerator_t generator,
                                               float* outputPtr,
                                               size_t n)
{
  return curandGenerateUniform(generator, outputPtr, n);
}
inline curandStatus_t curand_generate_uniformX(curandGenerator_t generator,
                                               double* outputPtr,
                                               size_t n)
{
  return curandGenerateUniformDouble(generator, outputPtr, n);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
MST_solver<vertex_t, edge_t, weight_t, alteration_t>::MST_solver(raft::resources const& handle_,
                                                                 const edge_t* offsets_,
                                                                 const vertex_t* indices_,
                                                                 const weight_t* weights_,
                                                                 const vertex_t v_,
                                                                 const edge_t e_,
                                                                 vertex_t* color_,
                                                                 cudaStream_t stream_,
                                                                 bool symmetrize_output_,
                                                                 bool initialize_colors_,
                                                                 int iterations_)
  : handle(handle_),
    offsets(offsets_),
    indices(indices_),
    weights(weights_),
    altered_weights(e_, stream_),
    v(v_),
    e(e_),
    color_index(color_),
    color(v_, stream_),
    next_color(v_, stream_),
    min_edge_color(v_, stream_),
    new_mst_edge(v_, stream_),
    mst_edge(e_, stream_),
    temp_src(2 * v_, stream_),
    temp_dst(2 * v_, stream_),
    temp_weights(2 * v_, stream_),
    mst_edge_count(1, stream_),
    prev_mst_edge_count(1, stream_),
    stream(stream_),
    symmetrize_output(symmetrize_output_),
    initialize_colors(initialize_colors_),
    iterations(iterations_)
{
  max_blocks  = resource::get_device_properties(handle_).maxGridSize[0];
  max_threads = resource::get_device_properties(handle_).maxThreadsPerBlock;
  sm_count    = resource::get_device_properties(handle_).multiProcessorCount;

  mst_edge_count.set_value_to_zero_async(stream);
  prev_mst_edge_count.set_value_to_zero_async(stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(mst_edge.data(), 0, mst_edge.size() * sizeof(bool), stream));

  // Initially, color holds the vertex id as color
  auto policy = resource::get_thrust_policy(handle);
  if (initialize_colors_) {
    thrust::sequence(policy, color.begin(), color.end(), 0);
    thrust::sequence(policy, color_index, color_index + v, 0);
  } else {
    raft::copy(color.data(), color_index, v, stream);
  }
  thrust::sequence(policy, next_color.begin(), next_color.end(), 0);
}

template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
Graph_COO<vertex_t, edge_t, weight_t> MST_solver<vertex_t, edge_t, weight_t, alteration_t>::solve()
{
  RAFT_EXPECTS(v > 0, "0 vertices");
  RAFT_EXPECTS(e > 0, "0 edges");
  RAFT_EXPECTS(offsets != nullptr, "Null offsets.");
  RAFT_EXPECTS(indices != nullptr, "Null indices.");
  RAFT_EXPECTS(weights != nullptr, "Null weights.");

  // Alterating the weights
  // this is done by identifying the lowest cost edge weight gap that is not 0, call this theta.
  // For each edge, add noise that is less than theta. That is, generate a random number in the
  // range [0.0, theta) and add it to each edge weight.
  if (e > 1) alteration();

  auto max_mst_edges = symmetrize_output ? 2 * v - 2 : v - 1;

  Graph_COO<vertex_t, edge_t, weight_t> mst_result(max_mst_edges, stream);

  // Boruvka original formulation says "while more than 1 supervertex remains"
  // Here we adjust it to support disconnected components (spanning forest)
  // track completion with mst_edge_found status and v as upper bound
  auto mst_iterations = iterations > 0 ? iterations : v;
  for (auto i = 0; i < mst_iterations; i++) {
    // Finds the minimum edge from each vertex to the lowest color
    // by working at each vertex of the supervertex
    min_edge_per_vertex();

    // Finds the minimum edge from each supervertex to the lowest color
    min_edge_per_supervertex();

    // check if msf/mst done, count new edges added
    check_termination();

    auto curr_mst_edge_count = mst_edge_count.value(stream);
    RAFT_EXPECTS(curr_mst_edge_count <= max_mst_edges,
                 "Number of edges found by MST is invalid. This may be due to "
                 "loss in precision. Try increasing precision of weights.");

    if (curr_mst_edge_count == prev_mst_edge_count.value(stream)) {
      // exit here when reaching steady state
      break;
    }

    // append the newly found MST edges to the final output
    append_src_dst_pair(mst_result.src.data(), mst_result.dst.data(), mst_result.weights.data());

    // updates colors of vertices by propagating the lower color to the higher
    label_prop(mst_result.src.data(), mst_result.dst.data());

    // copy this iteration's results and store
    prev_mst_edge_count.set_value_async(curr_mst_edge_count, stream);
  }

  // result packaging
  mst_result.n_edges = mst_edge_count.value(stream);
  mst_result.src.resize(mst_result.n_edges, stream);
  mst_result.dst.resize(mst_result.n_edges, stream);
  mst_result.weights.resize(mst_result.n_edges, stream);

  return mst_result;
}

// ||y|-|x||
template <typename weight_t>
struct alteration_functor {
  __host__ __device__ weight_t operator()(const cuda::std::tuple<weight_t, weight_t>& t)
  {
    auto x = cuda::std::get<0>(t);
    auto y = cuda::std::get<1>(t);
    x      = x < 0 ? -x : x;
    y      = y < 0 ? -y : y;
    return x < y ? y - x : x - y;
  }
};

// Compute the uper bound for the alteration
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
alteration_t MST_solver<vertex_t, edge_t, weight_t, alteration_t>::alteration_max()
{
  auto policy = resource::get_thrust_policy(handle);
  rmm::device_uvector<weight_t> tmp(e, stream);
  thrust::device_ptr<const weight_t> weights_ptr(weights);
  thrust::copy(policy, weights_ptr, weights_ptr + e, tmp.begin());
  // sort tmp weights
  thrust::sort(policy, tmp.begin(), tmp.end());

  // remove duplicates
  auto new_end = thrust::unique(policy, tmp.begin(), tmp.end());

  // min(a[i+1]-a[i])/2
  auto begin = cuda::make_zip_iterator(cuda::std::make_tuple(tmp.begin(), tmp.begin() + 1));
  auto end   = cuda::make_zip_iterator(cuda::std::make_tuple(new_end - 1, new_end));
  auto init  = tmp.element(1, stream) - tmp.element(0, stream);
  auto max   = thrust::transform_reduce(
    policy, begin, end, alteration_functor<weight_t>(), init, cuda::minimum<weight_t>());
  // Enforce distinct weights if initial edge weights are identical by returning
  // a value of 1
  return max > 0 ? max / static_cast<alteration_t>(2) : 1;
}

// Compute the alteration to make all undirected edge weight unique
// Preserves weights order
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
void MST_solver<vertex_t, edge_t, weight_t, alteration_t>::alteration()
{
  auto nthreads = std::min(v, max_threads);
  auto nblocks  = std::min((v + nthreads - 1) / nthreads, max_blocks);

  // maximum alteration that does not change relative weights order
  // Note: The relative weights order will be altered if initial edge weights are identical
  alteration_t max = alteration_max();

  // pool of rand values
  rmm::device_uvector<alteration_t> rand_values(v, stream);

  // Random number generator
  curandGenerator_t randGen;
  curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(randGen, 1234567);

  // Initialize rand values
  auto curand_status = curand_generate_uniformX(randGen, rand_values.data(), v);
  RAFT_EXPECTS(curand_status == CURAND_STATUS_SUCCESS, "MST: CURAND failed");
  curand_status = curandDestroyGenerator(randGen);
  RAFT_EXPECTS(curand_status == CURAND_STATUS_SUCCESS, "MST: CURAND cleanup failed");

  // Alterate the weights, make all undirected edge weight unique while keeping Wuv == Wvu
  detail::alteration_kernel<<<nblocks, nthreads, 0, stream>>>(
    v, e, offsets, indices, weights, max, rand_values.data(), altered_weights.data());
}

// updates colors of vertices by propagating the lower color to the higher
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
void MST_solver<vertex_t, edge_t, weight_t, alteration_t>::label_prop(vertex_t* mst_src,
                                                                      vertex_t* mst_dst)
{
  // update the colors of both ends its until there is no change in colors
  edge_t curr_mst_edge_count = mst_edge_count.value(stream);

  auto min_pair_nthreads = std::min(v, (vertex_t)max_threads);
  auto min_pair_nblocks =
    std::min((v + min_pair_nthreads - 1) / min_pair_nthreads, (vertex_t)max_blocks);

  edge_t* new_mst_edge_ptr = new_mst_edge.data();
  vertex_t* color_ptr      = color.data();
  vertex_t* next_color_ptr = next_color.data();

  rmm::device_scalar<bool> done(stream);
  done.set_value_to_zero_async(stream);
  bool* done_ptr      = done.data();
  const bool true_val = true;

  auto i = 0;
  while (!done.value(stream)) {
    done.set_value_async(true_val, stream);

    detail::min_pair_colors<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
      v, indices, new_mst_edge_ptr, color_ptr, color_index, next_color_ptr);

    detail::update_colors<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
      v, color_ptr, color_index, next_color_ptr, done_ptr);
    i++;
  }

  detail::final_color_indices<<<min_pair_nblocks, min_pair_nthreads, 0, stream>>>(
    v, color_ptr, color_index);
}

// Finds the minimum edge from each vertex to the lowest color
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
void MST_solver<vertex_t, edge_t, weight_t, alteration_t>::min_edge_per_vertex()
{
  auto policy = resource::get_thrust_policy(handle);
  thrust::fill(
    policy, min_edge_color.begin(), min_edge_color.end(), std::numeric_limits<alteration_t>::max());
  thrust::fill(
    policy, new_mst_edge.begin(), new_mst_edge.end(), std::numeric_limits<weight_t>::max());

  int n_threads = 32;

  vertex_t* color_ptr               = color.data();
  edge_t* new_mst_edge_ptr          = new_mst_edge.data();
  bool* mst_edge_ptr                = mst_edge.data();
  alteration_t* min_edge_color_ptr  = min_edge_color.data();
  alteration_t* altered_weights_ptr = altered_weights.data();

  detail::kernel_min_edge_per_vertex<<<v, n_threads, 0, stream>>>(offsets,
                                                                  indices,
                                                                  altered_weights_ptr,
                                                                  color_ptr,
                                                                  color_index,
                                                                  new_mst_edge_ptr,
                                                                  mst_edge_ptr,
                                                                  min_edge_color_ptr,
                                                                  v);
}

// Finds the minimum edge from each supervertex to the lowest color
template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
void MST_solver<vertex_t, edge_t, weight_t, alteration_t>::min_edge_per_supervertex()
{
  auto nthreads = std::min(v, max_threads);
  auto nblocks  = std::min((v + nthreads - 1) / nthreads, max_blocks);

  auto policy = resource::get_thrust_policy(handle);
  thrust::fill(policy, temp_src.begin(), temp_src.end(), std::numeric_limits<vertex_t>::max());

  vertex_t* color_ptr               = color.data();
  edge_t* new_mst_edge_ptr          = new_mst_edge.data();
  bool* mst_edge_ptr                = mst_edge.data();
  alteration_t* min_edge_color_ptr  = min_edge_color.data();
  alteration_t* altered_weights_ptr = altered_weights.data();
  vertex_t* temp_src_ptr            = temp_src.data();
  vertex_t* temp_dst_ptr            = temp_dst.data();
  weight_t* temp_weights_ptr        = temp_weights.data();

  detail::min_edge_per_supervertex<<<nblocks, nthreads, 0, stream>>>(color_ptr,
                                                                     color_index,
                                                                     new_mst_edge_ptr,
                                                                     mst_edge_ptr,
                                                                     indices,
                                                                     weights,
                                                                     altered_weights_ptr,
                                                                     temp_src_ptr,
                                                                     temp_dst_ptr,
                                                                     temp_weights_ptr,
                                                                     min_edge_color_ptr,
                                                                     v,
                                                                     symmetrize_output);

  // the above kernel only adds directed mst edges in the case where
  // a pair of vertices don't pick the same min edge between them
  // so, now we add the reverse edge to make it undirected
  if (symmetrize_output) {
    detail::add_reverse_edge<<<nblocks, nthreads, 0, stream>>>(new_mst_edge_ptr,
                                                               indices,
                                                               weights,
                                                               temp_src_ptr,
                                                               temp_dst_ptr,
                                                               temp_weights_ptr,
                                                               v,
                                                               symmetrize_output);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
void MST_solver<vertex_t, edge_t, weight_t, alteration_t>::check_termination()
{
  vertex_t nthreads = std::min(2 * v, (vertex_t)max_threads);
  vertex_t nblocks  = std::min((2 * v + nthreads - 1) / nthreads, (vertex_t)max_blocks);

  // count number of new mst edges
  edge_t* mst_edge_count_ptr = mst_edge_count.data();
  vertex_t* temp_src_ptr     = temp_src.data();

  detail::kernel_count_new_mst_edges<<<nblocks, nthreads, 0, stream>>>(
    temp_src_ptr, mst_edge_count_ptr, 2 * v);
}

template <typename vertex_t, typename weight_t>
struct new_edges_functor {
  __host__ __device__ bool operator()(const cuda::std::tuple<vertex_t, vertex_t, weight_t>& t)
  {
    auto src = cuda::std::get<0>(t);

    return src != std::numeric_limits<vertex_t>::max() ? true : false;
  }
};

template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
void MST_solver<vertex_t, edge_t, weight_t, alteration_t>::append_src_dst_pair(
  vertex_t* mst_src, vertex_t* mst_dst, weight_t* mst_weights)
{
  auto policy = resource::get_thrust_policy(handle);

  edge_t curr_mst_edge_count = prev_mst_edge_count.value(stream);

  // iterator to end of mst edges added to final output in previous iteration
  auto src_dst_zip_end =
    cuda::make_zip_iterator(cuda::std::make_tuple(mst_src + curr_mst_edge_count,
                                                  mst_dst + curr_mst_edge_count,
                                                  mst_weights + curr_mst_edge_count));

  // iterator to new mst edges found
  auto temp_src_dst_zip_begin = cuda::make_zip_iterator(
    cuda::std::make_tuple(temp_src.begin(), temp_dst.begin(), temp_weights.begin()));
  auto temp_src_dst_zip_end = cuda::make_zip_iterator(
    cuda::std::make_tuple(temp_src.end(), temp_dst.end(), temp_weights.end()));

  // copy new mst edges to final output
  thrust::copy_if(policy,
                  temp_src_dst_zip_begin,
                  temp_src_dst_zip_end,
                  src_dst_zip_end,
                  new_edges_functor<vertex_t, weight_t>());
}
}  // namespace sparse::solver
}  // namespace raft
