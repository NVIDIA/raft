/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rmat_rectangular_generator.cuh>

#include <raft_runtime/random/rmat_rectangular_generator.hpp>

#define FUNC_DEF(IdxT, ProbT)                                                                \
  void rmat_rectangular_gen(raft::resources const& handle,                                   \
                            IdxT* out,                                                       \
                            IdxT* out_src,                                                   \
                            IdxT* out_dst,                                                   \
                            const ProbT* theta,                                              \
                            IdxT r_scale,                                                    \
                            IdxT c_scale,                                                    \
                            IdxT n_edges,                                                    \
                            raft::random::RngState& r)                                       \
  {                                                                                          \
    raft::random::rmat_rectangular_gen<IdxT, ProbT>(out,                                     \
                                                    out_src,                                 \
                                                    out_dst,                                 \
                                                    theta,                                   \
                                                    r_scale,                                 \
                                                    c_scale,                                 \
                                                    n_edges,                                 \
                                                    resource::get_cuda_stream(handle).get(), \
                                                    r);                                      \
  }
