/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "rmat_rectangular_generator_types.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng_device.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace random {
namespace detail {

template <typename IdxT, typename ProbT>
DI void gen_and_update_bits(IdxT& src_id,
                            IdxT& dst_id,
                            ProbT a,
                            ProbT ab,
                            ProbT abc,
                            IdxT r_scale,
                            IdxT c_scale,
                            IdxT curr_depth,
                            raft::random::PCGenerator& gen)
{
  bool src_bit, dst_bit;
  ProbT val;
  gen.next(val);
  if (val <= a) {
    src_bit = dst_bit = false;
  } else if (val <= ab) {
    src_bit = false;
    dst_bit = true;
  } else if (val <= abc) {
    src_bit = true;
    dst_bit = false;
  } else {
    src_bit = dst_bit = true;
  }
  if (curr_depth < r_scale) { src_id |= (IdxT(src_bit) << (curr_depth)); }
  if (curr_depth < c_scale) { dst_id |= (IdxT(dst_bit) << (curr_depth)); }
}

template <typename IdxT>
DI void store_ids(
  IdxT* out, IdxT* out_src, IdxT* out_dst, IdxT src_id, IdxT dst_id, IdxT idx, IdxT n_edges)
{
  if (idx < n_edges) {
    if (out != nullptr) {
      // uncoalesced gmem accesses!
      out[idx * 2]     = src_id;
      out[idx * 2 + 1] = dst_id;
    }
    if (out_src != nullptr) { out_src[idx] = src_id; }
    if (out_dst != nullptr) { out_dst[idx] = dst_id; }
  }
}

template <typename IdxT, typename ProbT>
RAFT_KERNEL rmat_gen_kernel(IdxT* out,
                            IdxT* out_src,
                            IdxT* out_dst,
                            const ProbT* theta,
                            IdxT r_scale,
                            IdxT c_scale,
                            IdxT n_edges,
                            IdxT max_scale,
                            raft::random::RngState r)
{
  IdxT idx = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  extern __shared__ ProbT s_theta[];
  auto theta_len = max_scale * 2 * 2;
  // load the probabilities into shared memory and then convert them into cdf's
  // currently there are smem bank conflicts due to the way these are accessed
  for (int i = threadIdx.x; i < theta_len; i += blockDim.x) {
    s_theta[i] = theta[i];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < max_scale; i += blockDim.x) {
    auto a             = s_theta[4 * i];
    auto b             = s_theta[4 * i + 1];
    auto c             = s_theta[4 * i + 2];
    s_theta[4 * i + 1] = a + b;
    s_theta[4 * i + 2] = a + b + c;
    s_theta[4 * i + 3] += a + b + c;
  }
  __syncthreads();
  IdxT src_id{0}, dst_id{0};
  raft::random::PCGenerator gen{r.seed, r.base_subsequence + idx, 0};
  for (IdxT i = 0; i < max_scale; ++i) {
    auto a = s_theta[i * 4], ab = s_theta[i * 4 + 1], abc = s_theta[i * 4 + 2];
    gen_and_update_bits(src_id, dst_id, a, ab, abc, r_scale, c_scale, i, gen);
  }
  store_ids(out, out_src, out_dst, src_id, dst_id, idx, n_edges);
}

template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_caller(IdxT* out,
                                 IdxT* out_src,
                                 IdxT* out_dst,
                                 const ProbT* theta,
                                 IdxT r_scale,
                                 IdxT c_scale,
                                 IdxT n_edges,
                                 cudaStream_t stream,
                                 raft::random::RngState& r)
{
  if (n_edges <= 0) return;
  static constexpr int N_THREADS = 512;
  auto max_scale                 = max(r_scale, c_scale);
  size_t smem_size               = sizeof(ProbT) * max_scale * 2 * 2;
  auto n_blks                    = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  rmat_gen_kernel<<<n_blks, N_THREADS, smem_size, stream>>>(
    out, out_src, out_dst, theta, r_scale, c_scale, n_edges, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

template <typename IdxT, typename ProbT>
RAFT_KERNEL rmat_gen_kernel(IdxT* out,
                            IdxT* out_src,
                            IdxT* out_dst,
                            ProbT a,
                            ProbT b,
                            ProbT c,
                            IdxT r_scale,
                            IdxT c_scale,
                            IdxT n_edges,
                            IdxT max_scale,
                            raft::random::RngState r)
{
  IdxT idx = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  IdxT src_id{0}, dst_id{0};
  raft::random::PCGenerator gen{r.seed, r.base_subsequence + idx, 0};
  auto min_scale = min(r_scale, c_scale);
  IdxT i         = 0;
  // Whether we have more rows than columns.
  const bool more_rows = r_scale > c_scale;

  for (; i < max_scale; ++i) {
    ProbT A   = (i < min_scale) ? a : (more_rows ? a + b : a + c);
    ProbT AB  = (i < min_scale) ? a + b : (more_rows ? a + b : ProbT(1));
    ProbT ABC = (i < min_scale) ? a + b + c : ProbT(1);
    gen_and_update_bits(src_id, dst_id, A, AB, ABC, r_scale, c_scale, i, gen);
  }

  store_ids(out, out_src, out_dst, src_id, dst_id, idx, n_edges);
}

template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_caller(IdxT* out,
                                 IdxT* out_src,
                                 IdxT* out_dst,
                                 ProbT a,
                                 ProbT b,
                                 ProbT c,
                                 IdxT r_scale,
                                 IdxT c_scale,
                                 IdxT n_edges,
                                 cudaStream_t stream,
                                 raft::random::RngState& r)
{
  if (n_edges <= 0) return;
  static constexpr int N_THREADS = 512;
  auto max_scale                 = max(r_scale, c_scale);
  auto n_blks                    = raft::ceildiv<IdxT>(n_edges, N_THREADS);
  rmat_gen_kernel<<<n_blks, N_THREADS, 0, stream>>>(
    out, out_src, out_dst, a, b, c, r_scale, c_scale, n_edges, max_scale, r);
  RAFT_CUDA_TRY(cudaGetLastError());
  r.advance(n_edges, max_scale);
}

/**
 * @brief Implementation of `raft::random::rmat_rectangular_gen_impl`.
 *
 * @tparam IdxT  type of each node index
 * @tparam ProbT data type used for probability distributions (either fp32 or fp64)
 * @param[in]  handle  RAFT handle, containing the CUDA stream on which to schedule work
 * @param[in]  r       underlying state of the random generator. Especially useful when
 *                     one wants to call this API for multiple times in order to generate
 *                     a larger graph. For that case, just create this object with the
 *                     initial seed once and after every call continue to pass the same
 *                     object for the successive calls.
 * @param[out] output  Encapsulation of one, two, or three output vectors.
 * @param[in]  theta   distribution of each quadrant at each level of resolution.
 *                     Since these are probabilities, each of the 2x2 matrices for
 *                     each level of the RMAT must sum to one. [on device]
 *                     [dim = max(r_scale, c_scale) x 2 x 2]. Of course, it is assumed
 *                     that each of the group of 2 x 2 numbers all sum up to 1.
 * @param[in]  r_scale 2^r_scale represents the number of source nodes
 * @param[in]  c_scale 2^c_scale represents the number of destination nodes
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_impl(raft::resources const& handle,
                               raft::random::RngState& r,
                               raft::device_vector_view<const ProbT, IdxT> theta,
                               raft::random::detail::rmat_rectangular_gen_output<IdxT> output,
                               IdxT r_scale,
                               IdxT c_scale)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  static_assert(std::is_integral_v<IdxT>,
                "rmat_rectangular_gen: "
                "Template parameter IdxT must be an integral type");
  if (output.empty()) {
    return;  // nothing to do; not an error
  }

  const IdxT expected_theta_len = IdxT(4) * (r_scale >= c_scale ? r_scale : c_scale);
  RAFT_EXPECTS(theta.extent(0) == expected_theta_len,
               "rmat_rectangular_gen: "
               "theta.extent(0) = %zu != 2 * 2 * max(r_scale = %zu, c_scale = %zu) = %zu",
               static_cast<std::size_t>(theta.extent(0)),
               static_cast<std::size_t>(r_scale),
               static_cast<std::size_t>(c_scale),
               static_cast<std::size_t>(expected_theta_len));

  auto out                     = output.out_view();
  auto out_src                 = output.out_src_view();
  auto out_dst                 = output.out_dst_view();
  const bool out_has_value     = out.has_value();
  const bool out_src_has_value = out_src.has_value();
  const bool out_dst_has_value = out_dst.has_value();
  IdxT* out_ptr                = out_has_value ? (*out).data_handle() : nullptr;
  IdxT* out_src_ptr            = out_src_has_value ? (*out_src).data_handle() : nullptr;
  IdxT* out_dst_ptr            = out_dst_has_value ? (*out_dst).data_handle() : nullptr;
  const IdxT n_edges           = output.number_of_edges();

  rmat_rectangular_gen_caller(out_ptr,
                              out_src_ptr,
                              out_dst_ptr,
                              theta.data_handle(),
                              r_scale,
                              c_scale,
                              n_edges,
                              resource::get_cuda_stream(handle),
                              r);
}

/**
 * @brief Overload of `rmat_rectangular_gen` that assumes the same
 *   a, b, c, d probability distributions across all the scales.
 *
 * `a`, `b, and `c` effectively replace the above overload's
 * `theta` parameter.
 */
template <typename IdxT, typename ProbT>
void rmat_rectangular_gen_impl(raft::resources const& handle,
                               raft::random::RngState& r,
                               raft::random::detail::rmat_rectangular_gen_output<IdxT> output,
                               ProbT a,
                               ProbT b,
                               ProbT c,
                               IdxT r_scale,
                               IdxT c_scale)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  static_assert(std::is_integral_v<IdxT>,
                "rmat_rectangular_gen: "
                "Template parameter IdxT must be an integral type");
  if (output.empty()) {
    return;  // nothing to do; not an error
  }

  auto out                     = output.out_view();
  auto out_src                 = output.out_src_view();
  auto out_dst                 = output.out_dst_view();
  const bool out_has_value     = out.has_value();
  const bool out_src_has_value = out_src.has_value();
  const bool out_dst_has_value = out_dst.has_value();
  IdxT* out_ptr                = out_has_value ? (*out).data_handle() : nullptr;
  IdxT* out_src_ptr            = out_src_has_value ? (*out_src).data_handle() : nullptr;
  IdxT* out_dst_ptr            = out_dst_has_value ? (*out_dst).data_handle() : nullptr;
  const IdxT n_edges           = output.number_of_edges();

  detail::rmat_rectangular_gen_caller(out_ptr,
                                      out_src_ptr,
                                      out_dst_ptr,
                                      a,
                                      b,
                                      c,
                                      r_scale,
                                      c_scale,
                                      n_edges,
                                      resource::get_cuda_stream(handle),
                                      r);
}

}  // end namespace detail
}  // end namespace random
}  // namespace raft
