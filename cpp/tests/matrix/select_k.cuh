/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/sparse/detail/utils.h>
#include <raft/util/cudart_utils.hpp>

#include <raft_internal/matrix/select_k.cuh>

#include <rmm/device_uvector.hpp>

#include <cuda_bf16.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <ostream>

// ostream overload for __nv_bfloat16 so GTest can print values
inline std::ostream& operator<<(std::ostream& os, const __nv_bfloat16& val)
{
  os << __bfloat162float(val);
  return os;
}

namespace raft::matrix {

template <typename IdxT>
auto gen_simple_ids(uint32_t batch_size, uint32_t len) -> std::vector<IdxT>
{
  std::vector<IdxT> out(batch_size * len);
  auto s = rmm::cuda_stream_default;
  rmm::device_uvector<IdxT> out_d(out.size(), s);
  sparse::iota_fill(out_d.data(), IdxT(batch_size), IdxT(len), s.get());
  update_host(out.data(), out_d.data(), out.size(), s);
  s.sync();
  return out;
}

template <typename KeyT, typename IdxT>
struct io_simple {
 public:
  bool not_supported             = false;
  std::optional<SelectAlgo> algo = std::nullopt;

  io_simple(const select::params& spec,
            const std::vector<KeyT>& in_dists,
            const std::optional<std::vector<IdxT>>& in_ids,
            const std::vector<KeyT>& out_dists,
            const std::vector<IdxT>& out_ids)
    : in_dists_(in_dists),
      in_ids_(in_ids.value_or(gen_simple_ids<IdxT>(spec.batch_size, spec.len))),
      out_dists_(out_dists),
      out_ids_(out_ids)
  {
  }

  auto get_in_dists() -> std::vector<KeyT>& { return in_dists_; }
  auto get_in_ids() -> std::vector<IdxT>& { return in_ids_; }
  auto get_out_dists() -> std::vector<KeyT>& { return out_dists_; }
  auto get_out_ids() -> std::vector<IdxT>& { return out_ids_; }

 private:
  std::vector<KeyT> in_dists_;
  std::vector<IdxT> in_ids_;
  std::vector<KeyT> out_dists_;
  std::vector<IdxT> out_ids_;
};

template <typename KeyT, typename IdxT>
struct io_computed {
 public:
  bool not_supported = false;
  SelectAlgo algo;

  io_computed(const select::params& spec,
              const SelectAlgo& algo,
              const std::vector<KeyT>& in_dists,
              const std::optional<std::vector<IdxT>>& in_ids = std::nullopt)
    : algo(algo),
      in_dists_(in_dists),
      in_ids_(in_ids.value_or(gen_simple_ids<IdxT>(spec.batch_size, spec.len))),
      out_dists_(spec.batch_size * spec.k),
      out_ids_(spec.batch_size * spec.k)
  {
    // check if the size is supported by the algorithm
    switch (algo) {
      case SelectAlgo::kWarpAuto:
      case SelectAlgo::kWarpImmediate:
      case SelectAlgo::kWarpFiltered:
      case SelectAlgo::kWarpDistributed:
      case SelectAlgo::kWarpDistributedShm: {
        if (spec.k > raft::matrix::detail::select::warpsort::kMaxCapacity) {
          not_supported = true;
          return;
        }
      } break;
      default: break;
    }

    resources handle{};
    auto stream = resource::get_cuda_stream(handle);

    rmm::device_uvector<KeyT> in_dists_d(in_dists_.size(), stream);
    rmm::device_uvector<IdxT> in_ids_d(in_ids_.size(), stream);
    rmm::device_uvector<KeyT> out_dists_d(out_dists_.size(), stream);
    rmm::device_uvector<IdxT> out_ids_d(out_ids_.size(), stream);

    update_device(in_dists_d.data(), in_dists_.data(), in_dists_.size(), stream);
    update_device(in_ids_d.data(), in_ids_.data(), in_ids_.size(), stream);

    std::optional<raft::device_matrix_view<const IdxT, int64_t, row_major>> in_ids_view;
    if (spec.use_index_input) {
      in_ids_view = raft::make_device_matrix_view<const IdxT, int64_t>(
        in_ids_d.data(), spec.batch_size, spec.len);
    }

    matrix::select_k<KeyT, IdxT>(
      handle,
      raft::make_device_matrix_view<const KeyT, int64_t>(
        in_dists_d.data(), spec.batch_size, spec.len),
      in_ids_view,
      raft::make_device_matrix_view<KeyT, int64_t>(out_dists_d.data(), spec.batch_size, spec.k),
      raft::make_device_matrix_view<IdxT, int64_t>(out_ids_d.data(), spec.batch_size, spec.k),
      spec.select_min,
      false,
      algo);

    update_host(out_dists_.data(), out_dists_d.data(), out_dists_.size(), stream);
    update_host(out_ids_.data(), out_ids_d.data(), out_ids_.size(), stream);

    interruptible::synchronize(stream);

    auto p = topk_sort_permutation(out_dists_, out_ids_, spec.k, spec.select_min);
    apply_permutation(out_dists_, p);
    apply_permutation(out_ids_, p);
  }

  auto get_in_dists() -> std::vector<KeyT>& { return in_dists_; }
  auto get_in_ids() -> std::vector<IdxT>& { return in_ids_; }
  auto get_out_dists() -> std::vector<KeyT>& { return out_dists_; }
  auto get_out_ids() -> std::vector<IdxT>& { return out_ids_; }

 private:
  std::vector<KeyT> in_dists_;
  std::vector<IdxT> in_ids_;
  std::vector<KeyT> out_dists_;
  std::vector<IdxT> out_ids_;

  auto topk_sort_permutation(const std::vector<KeyT>& vec,
                             const std::vector<IdxT>& inds,
                             uint32_t k,
                             bool select_min) -> std::vector<IdxT>
  {
    std::vector<IdxT> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    if (select_min) {
      std::sort(p.begin(), p.end(), [&vec, &inds, k](IdxT i, IdxT j) {
        const IdxT ik = i / k;
        const IdxT jk = j / k;
        if (ik == jk) {
          if (vec[i] == vec[j]) { return inds[i] < inds[j]; }
          return vec[i] < vec[j];
        }
        return ik < jk;
      });
    } else {
      std::sort(p.begin(), p.end(), [&vec, &inds, k](IdxT i, IdxT j) {
        const IdxT ik = i / k;
        const IdxT jk = j / k;
        if (ik == jk) {
          if (vec[i] == vec[j]) { return inds[i] < inds[j]; }
          return vec[i] > vec[j];
        }
        return ik < jk;
      });
    }
    return p;
  }

  template <typename T>
  void apply_permutation(std::vector<T>& vec, const std::vector<IdxT>& p)  // NOLINT
  {
    for (auto i = IdxT(vec.size()) - 1; i > 0; i--) {
      auto j = p[i];
      while (j > i)
        j = p[j];
      std::swap(vec[j], vec[i]);
    }
  }
};

template <typename InOut>
using Params = std::tuple<select::params, SelectAlgo, InOut>;

template <typename KeyT, typename IdxT, template <typename, typename> typename ParamsReader>
struct SelectK  // NOLINT
  : public testing::TestWithParam<typename ParamsReader<KeyT, IdxT>::params_t> {
  const select::params spec;
  const SelectAlgo algo;
  typename ParamsReader<KeyT, IdxT>::io_t ref;
  io_computed<KeyT, IdxT> res;

  explicit SelectK(Params<typename ParamsReader<KeyT, IdxT>::io_t> ps)
    : spec(std::get<0>(ps)),
      algo(std::get<1>(ps)),                                 // NOLINT
      ref(std::get<2>(ps)),                                  // NOLINT
      res(spec, algo, ref.get_in_dists(), ref.get_in_ids())  // NOLINT
  {
  }

  explicit SelectK(typename ParamsReader<KeyT, IdxT>::params_t ps)
    : SelectK(ParamsReader<KeyT, IdxT>::read(ps))
  {
  }

  SelectK()
    : SelectK(testing::TestWithParam<typename ParamsReader<KeyT, IdxT>::params_t>::GetParam())
  {
  }

  void run()
  {
    if (ref.not_supported || res.not_supported) { GTEST_SKIP(); }
    ASSERT_TRUE(hostVecMatch(ref.get_out_dists(), res.get_out_dists(), Compare<KeyT>()));

    // If the dists (keys) are the same, different corresponding ids may end up in the selection
    // due to non-deterministic nature of some implementations.
    auto compare_ids = [this](const IdxT& i, const IdxT& j, const size_t pos) {
      if (i == j) return true;
      auto& in_ids   = ref.get_in_ids();
      auto& in_dists = ref.get_in_dists();
      auto batch     = pos / spec.k;
      auto start_pos = in_ids.begin() + batch * spec.len;
      // Get the indices of found neighbor ids in the source (input ids)
      auto ix_i =
        static_cast<int64_t>(std::find(start_pos, start_pos + spec.len, i) - in_ids.begin());
      auto ix_j =
        static_cast<int64_t>(std::find(start_pos, start_pos + spec.len, j) - in_ids.begin());
      auto forgive_i = forgive_algo(ref.algo, i);
      auto forgive_j = forgive_algo(res.algo, j);
      // Some algorithms return invalid indices in special cases.
      // TODO: https://github.com/rapidsai/raft/issues/1822
      if (static_cast<size_t>(ix_i) >= in_ids.size()) return forgive_i;
      if (static_cast<size_t>(ix_j) >= in_ids.size()) return forgive_j;
      auto dist_i = in_dists[ix_i];
      auto dist_j = in_dists[ix_j];
      if (dist_i == dist_j) return true;
      const auto bound = spec.select_min ? raft::upper_bound<KeyT>() : raft::lower_bound<KeyT>();
      if (forgive_i && dist_i == bound) return true;
      if (forgive_j && dist_j == bound) return true;
      // Otherwise really fail
      std::cout << "ERROR: ref[" << ix_i << "] = " << dist_i << " != "
                << "res[" << ix_j << "] = " << dist_j << std::endl;
      return false;
    };
    ASSERT_TRUE(hostVecMatch(ref.get_out_ids(), res.get_out_ids(), compare_ids));
  }

  auto forgive_algo(const std::optional<SelectAlgo>& algo, IdxT ix) const -> bool
  {
    if (!algo.has_value()) { return false; }
    switch (algo.value()) {
      // not sure which algo this is.
      case SelectAlgo::kAuto: return true;
      // warp-sort-based algos currently return zero index for inf distances.
      case SelectAlgo::kWarpAuto:
      case SelectAlgo::kWarpImmediate:
      case SelectAlgo::kWarpFiltered:
      case SelectAlgo::kWarpDistributed:
      case SelectAlgo::kWarpDistributedShm: return ix == 0;
      // Do not forgive by default
      default: return false;
    }
  }
};

template <typename KeyT, typename IdxT>
struct params_simple {
  using io_t     = io_simple<KeyT, IdxT>;
  using input_t  = std::tuple<select::params,
                              std::vector<KeyT>,
                              std::optional<std::vector<IdxT>>,
                              std::vector<KeyT>,
                              std::vector<IdxT>>;
  using params_t = std::tuple<input_t, SelectAlgo>;

  static auto read(params_t ps) -> Params<io_t>
  {
    auto ins  = std::get<0>(ps);
    auto algo = std::get<1>(ps);
    return std::make_tuple(
      std::get<0>(ins),
      algo,
      io_simple<KeyT, IdxT>(
        std::get<0>(ins), std::get<1>(ins), std::get<2>(ins), std::get<3>(ins), std::get<4>(ins)));
  }
};

auto inf_f           = std::numeric_limits<float>::max();
auto inputs_simple_f = testing::Values(
  params_simple<float, uint32_t>::input_t(
    {5, 5, 5, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    std::nullopt,
    {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
     4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3}),
  params_simple<float, uint32_t>::input_t(
    {5, 5, 3, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    std::nullopt,
    {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
    {4, 3, 2, 0, 1, 2, 3, 0, 1, 4, 2, 1, 0, 2, 1}),
  params_simple<float, uint32_t>::input_t(
    {5, 5, 5, true, false},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    std::nullopt,
    {1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0,
     4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0},
    {4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 3, 0, 1, 4, 2, 4, 2, 1, 3, 0, 0, 2, 1, 4, 3}),
  params_simple<float, uint32_t>::input_t(
    {5, 5, 3, true, false},
    {5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 3.0, 5.0,
     1.0, 4.0, 5.0, 3.0, 2.0, 4.0, 1.0, 1.0, 3.0, 2.0, 5.0, 4.0},
    std::nullopt,
    {1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0},
    {4, 3, 2, 0, 1, 2, 3, 0, 1, 4, 2, 1, 0, 2, 1}),
  params_simple<float, uint32_t>::input_t(
    {5, 7, 3, true, true},
    {5.0, 4.0, 3.0, 2.0, 1.3, 7.5, 19.0, 9.0, 2.0, 3.0, 3.0, 5.0, 6.0, 4.0, 2.0, 3.0, 5.0, 1.0,
     4.0, 1.0, 1.0, 5.0, 7.0, 2.5, 4.0,  7.0, 8.0, 8.0, 1.0, 3.0, 2.0, 5.0, 4.0, 1.1, 1.2},
    std::nullopt,
    {1.3, 2.0, 3.0, 2.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.5, 4.0, 5.0, 1.0, 1.1, 1.2},
    {4, 3, 2, 1, 2, 3, 3, 5, 6, 2, 3, 0, 0, 5, 6}),
  params_simple<float, uint32_t>::input_t({1, 7, 3, true, true},
                                          {2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 1.0},
                                          std::nullopt,
                                          {1.0, 1.0, 1.0},
                                          {3, 5, 6}),
  params_simple<float, uint32_t>::input_t({1, 7, 3, false, false},
                                          {2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 1.0},
                                          std::nullopt,
                                          {5.0, 4.0, 3.0},
                                          {2, 4, 1}),
  params_simple<float, uint32_t>::input_t({1, 7, 3, false, true},
                                          {2.0, 3.0, 5.0, 9.0, 4.0, 9.0, 9.0},
                                          std::nullopt,
                                          {9.0, 9.0, 9.0},
                                          {3, 5, 6}),
  params_simple<float, uint32_t>::input_t(
    {1, 130, 5, false, true},
    {19, 1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
     1,  2, 1, 2, 1, 2,  1,  2,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4,
     5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 4, 2, 3, 2, 3, 2, 3, 2, 3, 2, 20},
    std::nullopt,
    {20, 19, 18, 17, 16},
    {129, 0, 117, 116, 115}),
  params_simple<float, uint32_t>::input_t(
    {1, 130, 15, false, true},
    {19, 1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  0,  1,  0,  1,  0,  1,  0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
     0,  1, 0, 1, 0, 1,  0,  1,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
     1,  2, 1, 2, 1, 2,  1,  2,  1,  2,  1,  2,  1,  2,  1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 4,
     5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 4, 4, 2, 3, 2, 3, 2, 3, 2, 3, 2, 20},
    std::nullopt,
    {20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6},
    {129, 0, 117, 116, 115, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105}),
  params_simple<float, uint32_t>::input_t(
    select::params{1, 32, 31, true, true},
    {0,  1,  2,  3,  inf_f, inf_f, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20,    21,    22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    std::optional{std::vector<uint32_t>{31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
                                        20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
                                        9,  8,  7,  6,  75, 74, 3,  2,  1,  0}},
    {0,  1,  2,  3,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,   17,
     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, inf_f},
    {31, 30, 29, 28, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
     13, 12, 11, 10, 9,  8,  7,  6,  75, 74, 3,  2,  1,  0,  27}));

using SimpleFloatInt = SelectK<float, uint32_t, params_simple>;
TEST_P(SimpleFloatInt, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                // NOLINT
  SelectK,
  SimpleFloatInt,
  testing::Combine(inputs_simple_f,
                   testing::Values(SelectAlgo::kAuto,
                                   SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass,
                                   SelectAlgo::kWarpImmediate,
                                   SelectAlgo::kWarpFiltered,
                                   SelectAlgo::kWarpDistributed)));

// Utility to convert float vector to nv_bfloat16
inline std::vector<__nv_bfloat16> make_bf16_vec(const std::vector<float>& v)
{
  std::vector<__nv_bfloat16> out;
  out.reserve(v.size());
  for (float f : v) {
    out.push_back(__float2bfloat16(f));
  }
  return out;
}

// Prepare simple test data including negative datasets (batch, N, K, select_min?, sorted?)
auto inputs_simple_bfloat16 = testing::Values(
  params_simple<__nv_bfloat16, uint32_t>::input_t({1, 5, 3, false, true},
                                                  make_bf16_vec({3.5, 1.5, 7.5, -0.5, 2.5}),
                                                  std::nullopt,
                                                  make_bf16_vec({7.5, 3.5, 2.5}),
                                                  {2, 0, 4}),
  params_simple<__nv_bfloat16, uint32_t>::input_t({1, 5, 3, true, true},
                                                  make_bf16_vec({3.5, 1.5, 7.5, -0.5, 2.5}),
                                                  std::nullopt,
                                                  make_bf16_vec({-0.5, 1.5, 2.5}),
                                                  {3, 1, 4}),
  params_simple<__nv_bfloat16, uint32_t>::input_t({1, 5, 3, false, true},
                                                  make_bf16_vec({-3.5, -1.5, -7.5, 0.5, -2.5}),
                                                  std::nullopt,
                                                  make_bf16_vec({0.5, -1.5, -2.5}),
                                                  {3, 1, 4}),
  params_simple<__nv_bfloat16, uint32_t>::input_t({1, 5, 3, true, true},
                                                  make_bf16_vec({-3.5, -1.5, -7.5, 0.5, -2.5}),
                                                  std::nullopt,
                                                  make_bf16_vec({-7.5, -3.5, -2.5}),
                                                  {2, 0, 4}));

using SimpleBFloat16Int = SelectK<__nv_bfloat16, uint32_t, params_simple>;
TEST_P(SimpleBFloat16Int, Run) { run(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(                   // NOLINT
  SelectK,
  SimpleBFloat16Int,
  testing::Combine(inputs_simple_bfloat16,
                   testing::Values(SelectAlgo::kAuto,
                                   SelectAlgo::kRadix8bits,
                                   SelectAlgo::kRadix11bits,
                                   SelectAlgo::kRadix11bitsExtraPass,
                                   SelectAlgo::kWarpImmediate,
                                   SelectAlgo::kWarpFiltered,
                                   SelectAlgo::kWarpDistributed)));

template <typename KeyT>
struct replace_with_mask {
  KeyT replacement;
  constexpr auto inline operator()(KeyT x, uint8_t mask) -> KeyT { return mask ? replacement : x; }
};

template <SelectAlgo RefAlgo>
struct with_ref {
  template <typename KeyT, typename IdxT>
  struct params_random {
    using io_t     = io_computed<KeyT, IdxT>;
    using params_t = std::tuple<select::params, SelectAlgo>;

    static auto read(params_t ps) -> Params<io_t>
    {
      auto spec = std::get<0>(ps);
      auto algo = std::get<1>(ps);
      std::vector<KeyT> dists(spec.len * spec.batch_size);

      raft::resources handle;
      {
        auto s = resource::get_cuda_stream(handle);
        rmm::device_uvector<KeyT> dists_d(spec.len * spec.batch_size, s);
        raft::random::RngState r(42);
        normal(handle, r, dists_d.data(), dists_d.size(), KeyT(10.0), KeyT(100.0));

        if (spec.frac_infinities > 0.0) {
          rmm::device_uvector<uint8_t> mask_buf(dists_d.size(), s);
          auto mask = make_device_vector_view<uint8_t, size_t>(mask_buf.data(), mask_buf.size());
          raft::random::bernoulli(handle, r, mask, spec.frac_infinities);
          KeyT bound = spec.select_min ? raft::upper_bound<KeyT>() : raft::lower_bound<KeyT>();
          auto mask_in =
            make_device_vector_view<const uint8_t, size_t>(mask_buf.data(), mask_buf.size());
          auto dists_in  = make_device_vector_view<const KeyT>(dists_d.data(), dists_d.size());
          auto dists_out = make_device_vector_view<KeyT>(dists_d.data(), dists_d.size());
          raft::linalg::map(handle, dists_out, replace_with_mask<KeyT>{bound}, dists_in, mask_in);
        }

        update_host(dists.data(), dists_d.data(), dists_d.size(), s);
        s.sync();
      }

      return std::make_tuple(spec, algo, io_computed<KeyT, IdxT>(spec, RefAlgo, dists));
    }
  };
};

}  // namespace raft::matrix
