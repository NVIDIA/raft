/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/linalg/reduce_rows_by_key.cuh>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct rrbk_params {
  int64_t rows, cols;
  int64_t keys;
};

template <typename T, typename KeyT>
struct reduce_rows_by_key : public fixture {
  reduce_rows_by_key(const rrbk_params& p)
    : params(p),
      in(p.rows * p.cols, stream),
      out(p.keys * p.cols, stream),
      keys(p.rows, stream),
      workspace(p.rows, stream)
  {
    raft::random::RngState rng{42};
    raft::random::uniformInt(handle, rng, keys.data(), p.rows, (KeyT)0, (KeyT)p.keys);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::linalg::reduce_rows_by_key(in.data(),
                                       params.cols,
                                       keys.data(),
                                       workspace.data(),
                                       params.rows,
                                       params.cols,
                                       params.keys,
                                       out.data(),
                                       stream.get(),
                                       false);
    });
  }

 protected:
  rrbk_params params;
  rmm::device_uvector<T> in, out;
  rmm::device_uvector<KeyT> keys;
  rmm::device_uvector<char> workspace;
};  // struct reduce_rows_by_key

const std::vector<rrbk_params> kInputSizes{
  {10000, 128, 64},
  {100000, 128, 64},
  {1000000, 128, 64},
  {10000000, 128, 64},
  {10000, 128, 256},
  {100000, 128, 256},
  {1000000, 128, 256},
  {10000000, 128, 256},
  {10000, 128, 1024},
  {100000, 128, 1024},
  {1000000, 128, 1024},
  {10000000, 128, 1024},
  {10000, 128, 4096},
  {100000, 128, 4096},
  {1000000, 128, 4096},
  {10000000, 128, 4096},
};

RAFT_BENCH_REGISTER((reduce_rows_by_key<float, uint32_t>), "", kInputSizes);
RAFT_BENCH_REGISTER((reduce_rows_by_key<double, uint32_t>), "", kInputSizes);

}  // namespace raft::bench::linalg
