/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/sparse/solver/detail/mst_kernels.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace raft {
namespace sparse::solver {

namespace detail {

/*
 * MST solver (algorithm: mst_kernels.cuh; user contract: mst.cuh).
 * Key order (from float): -0.0 < +0.0; NaNs order by bit pattern.
 * Colors = min vertex id per component.
 * iterations != 0 disables the edge filter; > 0 bounds the rounds.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
Graph_COO<vertex_t, edge_t, weight_t> mst_solve(raft::resources const& handle,
                                                edge_t const* offsets,
                                                vertex_t const* indices,
                                                weight_t const* weights,
                                                vertex_t const v,
                                                edge_t const e,
                                                vertex_t* color,
                                                cudaStream_t stream,
                                                bool symmetrize_output,
                                                bool initialize_colors,
                                                int iterations)
{
  static_assert((sizeof(vertex_t) == 4 || sizeof(vertex_t) == 8) &&
                  (sizeof(edge_t) == 4 || sizeof(edge_t) == 8) &&
                  sizeof(vertex_t) <= sizeof(edge_t),
                "raft::sparse::solver::mst supports 32- and 64-bit vertex_t/edge_t, with "
                "edge_t at least as wide as vertex_t (worklist entries store vertex ids in "
                "fields sized by the edge width)");
  constexpr bool narrow = mst_traits<weight_t, edge_t>::narrow;
  using key_t           = typename mst_traits<weight_t, edge_t>::key_t;
  using entry_t         = typename mst_traits<weight_t, edge_t>::entry_t;
  using wl_size_t       = typename mst_traits<weight_t, edge_t>::wl_size_t;

  RAFT_EXPECTS(v > 0, "0 vertices");
  RAFT_EXPECTS(e > 0, "0 edges");
  RAFT_EXPECTS(offsets != nullptr, "Null offsets.");
  RAFT_EXPECTS(indices != nullptr, "Null indices.");
  RAFT_EXPECTS(weights != nullptr, "Null weights.");
  // narrow packing uses signed 32-bit int4 fields; unsigned 32-bit types
  // can exceed them (use a 64-bit edge_t for edge counts above INT_MAX)
  if constexpr (narrow && std::is_unsigned_v<vertex_t>) {
    RAFT_EXPECTS(v <= static_cast<vertex_t>(std::numeric_limits<int>::max()),
                 "unsigned 32-bit vertex ids above INT_MAX are not supported");
  }
  if constexpr (narrow && std::is_unsigned_v<edge_t>) {
    RAFT_EXPECTS(e <= static_cast<edge_t>(std::numeric_limits<int>::max()),
                 "unsigned 32-bit edge counts above INT_MAX are not supported");
  }

  // one worklist entry per undirected edge
  const wl_size_t wl_capacity = static_cast<wl_size_t>(e / 2 + 1);

  rmm::device_uvector<vertex_t> parent(v, stream);
  rmm::device_uvector<bool> in_mst(e, stream);
  rmm::device_uvector<entry_t> wl1(static_cast<size_t>(wl_capacity), stream);
  rmm::device_uvector<entry_t> wl2(static_cast<size_t>(wl_capacity), stream);
  rmm::device_scalar<wl_size_t> wl_size_d(stream);

#if RAFT_MST_HAS_CAS128
  const size_t minv_bytes = narrow ? 2 * static_cast<size_t>(v) * sizeof(mst_ull)
                                   : 2 * static_cast<size_t>(v) * sizeof(mst_u128);
#else
  const size_t minv_bytes = narrow ? 2 * static_cast<size_t>(v) * sizeof(mst_ull)
                                   : 4 * static_cast<size_t>(v) * sizeof(mst_ull);
#endif
  rmm::device_uvector<char> minv_raw(minv_bytes, stream);

  const int vblocks =
    static_cast<int>((static_cast<size_t>(v) + mst_block_size - 1) / mst_block_size);
  if (initialize_colors) {
    thrust::sequence(rmm::exec_policy(stream), parent.begin(), parent.end());
  } else {
    mst_init_parent_kernel<<<vblocks, mst_block_size, 0, stream>>>(v, color, parent.data());
  }
  RAFT_CUDA_TRY(cudaMemsetAsync(minv_raw.data(), 0xFF, minv_bytes, stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(in_mst.data(), 0, e * sizeof(bool), stream));

  // Two-phase filter: solve a sampled light-edge prefix first, then the
  // rest find-filtered. Disabled for bounded solves. Constants are empirical.
  constexpr int filter_min_avg_degree            = 4;
  constexpr double filter_light_edges_per_vertex = 3.0;
  constexpr int max_samples                      = 20;

  key_t thr_key = std::numeric_limits<key_t>::max();
  bool filtered = false;
  if (iterations == 0 && e / v >= filter_min_avg_degree) {
    const int ns = static_cast<int>(std::min<edge_t>(e, max_samples));
    rmm::device_uvector<key_t> keys_d(ns, stream);
    mst_sample_keys_kernel<<<1, 32, 0, stream>>>(ns, e, weights, keys_d.data());
    key_t keys[max_samples];
    raft::update_host(keys, keys_d.data(), ns, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    std::sort(keys, keys + ns);
    thr_key =
      keys[std::min(max_samples - 1, static_cast<int>(filter_light_edges_per_vertex * v * ns / e))];
    filtered = true;
  }

  int round    = 0;
  auto boruvka = [&](wl_size_t wl_size) {
    entry_t* d1 = wl1.data();
    entry_t* d2 = wl2.data();
    while (wl_size > 0) {
      if (iterations > 0 && round >= iterations) break;
      wl_size_d.set_value_to_zero_async(stream);
      const int wblocks =
        static_cast<int>((static_cast<long long>(wl_size) + mst_block_size - 1) / mst_block_size);
      if constexpr (narrow) {
        mst_ull* const base = reinterpret_cast<mst_ull*>(minv_raw.data());
        mst_ull* const cur  = base + (round % 2) * static_cast<size_t>(v);
        mst_ull* const prev = base + ((round + 1) % 2) * static_cast<size_t>(v);
        mst_filter_min_kernel<<<wblocks, mst_block_size, 0, stream>>>(
          d1, wl_size, d2, wl_size_d.data(), parent.data(), cur, prev);
        std::swap(d1, d2);
        wl_size = wl_size_d.value(stream);
        if (wl_size > 0) {
          const int nblocks = static_cast<int>(
            (static_cast<long long>(wl_size) + mst_block_size - 1) / mst_block_size);
          mst_select_join_kernel<<<nblocks, mst_block_size, 0, stream>>>(
            d1, wl_size, parent.data(), cur, in_mst.data());
        }
      } else {
#if RAFT_MST_HAS_CAS128
        mst_u128* const base = reinterpret_cast<mst_u128*>(minv_raw.data());
        mst_u128* const cur  = base + (round % 2) * static_cast<size_t>(v);
        mst_u128* const prev = base + ((round + 1) % 2) * static_cast<size_t>(v);
        mst_filter_min_kernel<<<wblocks, mst_block_size, 0, stream>>>(
          d1, wl_size, d2, wl_size_d.data(), parent.data(), cur, prev);
        std::swap(d1, d2);
        wl_size = wl_size_d.value(stream);
        if (wl_size > 0) {
          const int nblocks = static_cast<int>(
            (static_cast<long long>(wl_size) + mst_block_size - 1) / mst_block_size);
          mst_select_join_kernel<<<nblocks, mst_block_size, 0, stream>>>(
            d1, wl_size, parent.data(), cur, in_mst.data());
        }
#else
        mst_ull* const base      = reinterpret_cast<mst_ull*>(minv_raw.data());
        mst_ull* const minw_cur  = base + (round % 2) * static_cast<size_t>(v);
        mst_ull* const minw_prev = base + ((round + 1) % 2) * static_cast<size_t>(v);
        mst_ull* const mine_cur =
          base + 2 * static_cast<size_t>(v) + (round % 2) * static_cast<size_t>(v);
        mst_ull* const mine_prev =
          base + 2 * static_cast<size_t>(v) + ((round + 1) % 2) * static_cast<size_t>(v);
        mst_filter_min_kernel<<<wblocks, mst_block_size, 0, stream>>>(
          d1, wl_size, d2, wl_size_d.data(), parent.data(), minw_cur, minw_prev, mine_prev);
        std::swap(d1, d2);
        wl_size = wl_size_d.value(stream);
        if (wl_size > 0) {
          const int nblocks = static_cast<int>(
            (static_cast<long long>(wl_size) + mst_block_size - 1) / mst_block_size);
          mst_min_index_kernel<<<nblocks, mst_block_size, 0, stream>>>(
            d1, wl_size, minw_cur, mine_cur);
          mst_select_join_kernel<<<nblocks, mst_block_size, 0, stream>>>(
            d1, wl_size, parent.data(), mine_cur, in_mst.data());
        }
#endif
      }
      round++;
    }
  };

  const int eblocks =
    static_cast<int>((static_cast<size_t>(e) + mst_block_size - 1) / mst_block_size);
  auto launch_init = [&](bool first) {
    wl_size_d.set_value_to_zero_async(stream);
    if (first) {
      mst_init_worklist_kernel<true><<<eblocks, mst_block_size, 0, stream>>>(wl1.data(),
                                                                             wl_size_d.data(),
                                                                             wl_capacity,
                                                                             v,
                                                                             e,
                                                                             offsets,
                                                                             indices,
                                                                             weights,
                                                                             parent.data(),
                                                                             thr_key);
    } else {
      mst_init_worklist_kernel<false><<<eblocks, mst_block_size, 0, stream>>>(wl1.data(),
                                                                              wl_size_d.data(),
                                                                              wl_capacity,
                                                                              v,
                                                                              e,
                                                                              offsets,
                                                                              indices,
                                                                              weights,
                                                                              parent.data(),
                                                                              thr_key);
    }
    const wl_size_t wl_size = wl_size_d.value(stream);
    // wl_size < 0 means the admission counter wrapped (malformed input)
    RAFT_EXPECTS(wl_size >= 0 && wl_size <= wl_capacity,
                 "MST worklist overflow: the input CSR must be symmetric (each "
                 "undirected edge stored in both directions).");
    return wl_size;
  };

  boruvka(launch_init(true));

  if (filtered) {
    // clear straggler minima before admitting the remaining edges
    RAFT_CUDA_TRY(cudaMemsetAsync(minv_raw.data(), 0xFF, minv_bytes, stream));
    boruvka(launch_init(false));
  }

  mst_flatten_colors_kernel<<<vblocks, mst_block_size, 0, stream>>>(v, parent.data(), color);

  // symmetrized count can exceed 32-bit edge_t/vertex_t for v > 2^30:
  // fail loudly rather than under-allocate
  const int64_t max_out_wide =
    symmetrize_output ? 2 * (static_cast<int64_t>(v) - 1) : (static_cast<int64_t>(v) - 1);
  RAFT_EXPECTS(max_out_wide <= std::numeric_limits<edge_t>::max() &&
                 max_out_wide <= std::numeric_limits<vertex_t>::max(),
               "MST output edge count can exceed edge_t/vertex_t; use fewer vertices or "
               "symmetrize_output=false");
  const edge_t max_out = static_cast<edge_t>(max_out_wide);
  Graph_COO<vertex_t, edge_t, weight_t> mst_result(std::max<edge_t>(max_out, 1), stream);
  rmm::device_scalar<edge_t> out_count(stream);
  out_count.set_value_to_zero_async(stream);
  mst_extract_coo_kernel<<<eblocks, mst_block_size, 0, stream>>>(v,
                                                                 e,
                                                                 offsets,
                                                                 indices,
                                                                 weights,
                                                                 in_mst.data(),
                                                                 symmetrize_output,
                                                                 mst_result.src.data(),
                                                                 mst_result.dst.data(),
                                                                 mst_result.weights.data(),
                                                                 out_count.data());
  mst_result.n_edges = out_count.value(stream);
  mst_result.src.resize(mst_result.n_edges, stream);
  mst_result.dst.resize(mst_result.n_edges, stream);
  mst_result.weights.resize(mst_result.n_edges, stream);

  return mst_result;
}

}  // namespace detail

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
    stream(stream_),
    symmetrize_output(symmetrize_output_),
    initialize_colors(initialize_colors_),
    iterations(iterations_),
    offsets(offsets_),
    indices(indices_),
    weights(weights_),
    v(v_),
    e(e_),
    color_index(color_)
{
}

template <typename vertex_t, typename edge_t, typename weight_t, typename alteration_t>
Graph_COO<vertex_t, edge_t, weight_t> MST_solver<vertex_t, edge_t, weight_t, alteration_t>::solve()
{
  return detail::mst_solve(handle,
                           offsets,
                           indices,
                           weights,
                           v,
                           e,
                           color_index,
                           stream,
                           symmetrize_output,
                           initialize_colors,
                           iterations);
}

}  // namespace sparse::solver
}  // namespace raft
