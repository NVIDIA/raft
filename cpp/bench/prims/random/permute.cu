/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace raft::bench::random {

struct permute_inputs {
  int rows, cols;
  bool needPerms, needShuffle, rowMajor;
};  // struct permute_inputs

template <typename T>
struct permute : public fixture {
  permute(const permute_inputs& p)
    : params(p),
      perms(p.needPerms ? p.rows : 0, stream),
      out(p.rows * p.cols, stream),
      in(p.rows * p.cols, stream)
  {
    raft::random::RngState r(123456ULL);
    uniform(handle, r, in.data(), p.rows, T(-1.0), T(1.0));
  }

  void run_benchmark(::benchmark::State& state) override
  {
    raft::random::RngState r(123456ULL);
    loop_on_state(state, [this, &r]() {
      raft::random::permute(perms.data(),
                            out.data(),
                            in.data(),
                            params.cols,
                            params.rows,
                            params.rowMajor,
                            stream,
                            123456ULL);
    });
  }

 private:
  raft::device_resources handle;
  permute_inputs params;
  rmm::device_uvector<T> out, in;
  rmm::device_uvector<int> perms;
};  // struct permute

const std::vector<permute_inputs> permute_input_vecs = {
  {32 * 1024, 128, true, true, true},
  {1024 * 1024, 128, true, true, true},
  {32 * 1024, 128 + 2, true, true, true},
  {1024 * 1024, 128 + 2, true, true, true},
  {32 * 1024, 128 + 1, true, true, true},
  {1024 * 1024, 128 + 1, true, true, true},

  {32 * 1024, 128, true, true, false},
  {1024 * 1024, 128, true, true, false},
  {32 * 1024, 128 + 2, true, true, false},
  {1024 * 1024, 128 + 2, true, true, false},
  {32 * 1024, 128 + 1, true, true, false},
  {1024 * 1024, 128 + 1, true, true, false},

};

RAFT_BENCH_REGISTER(permute<float>, "", permute_input_vecs);
RAFT_BENCH_REGISTER(permute<double>, "", permute_input_vecs);

}  // namespace raft::bench::random
