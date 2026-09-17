/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

template <typename IdxT>
struct rcbk_params {
  IdxT rows, cols;
  IdxT keys;
};

template <typename IdxT>
inline auto operator<<(std::ostream& os, const rcbk_params<IdxT>& p) -> std::ostream&
{
  os << p.rows << "#" << p.cols << "#" << p.keys;
  return os;
}

template <typename T, typename KeyT, typename IdxT>
struct reduce_cols_by_key : public fixture {
  reduce_cols_by_key(const rcbk_params<IdxT>& p)
    : params(p), in(p.rows * p.cols, stream), out(p.rows * p.keys, stream), keys(p.cols, stream)
  {
    raft::random::RngState rng{42};
    raft::random::uniformInt(handle, rng, keys.data(), p.cols, (KeyT)0, (KeyT)p.keys);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    loop_on_state(state, [this]() {
      raft::linalg::reduce_cols_by_key(in.data(),
                                       keys.data(),
                                       out.data(),
                                       params.rows,
                                       params.cols,
                                       params.keys,
                                       stream.get(),
                                       false);
    });
  }

 protected:
  rcbk_params<IdxT> params;
  rmm::device_uvector<T> in, out;
  rmm::device_uvector<KeyT> keys;
};  // struct reduce_cols_by_key

const std::vector<rcbk_params<int>> rcbk_inputs_i32 =
  raft::util::itertools::product<rcbk_params<int>>(
    {1, 10, 100, 1000}, {1000, 10000, 100000}, {8, 32, 128, 512, 2048});
const std::vector<rcbk_params<int64_t>> rcbk_inputs_i64 =
  raft::util::itertools::product<rcbk_params<int64_t>>(
    {1, 10, 100, 1000}, {1000, 10000, 100000}, {8, 32, 128, 512, 2048});

RAFT_BENCH_REGISTER((reduce_cols_by_key<float, uint32_t, int>), "", rcbk_inputs_i32);
RAFT_BENCH_REGISTER((reduce_cols_by_key<double, uint32_t, int>), "", rcbk_inputs_i32);
RAFT_BENCH_REGISTER((reduce_cols_by_key<float, uint32_t, int64_t>), "", rcbk_inputs_i64);
RAFT_BENCH_REGISTER((reduce_cols_by_key<double, uint32_t, int64_t>), "", rcbk_inputs_i64);

}  // namespace raft::bench::linalg
