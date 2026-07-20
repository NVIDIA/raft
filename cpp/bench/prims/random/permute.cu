/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

template <typename IntType>
struct permute_perms_only : public fixture {
  permute_perms_only(int rows) : n_rows(rows), perms(rows, stream) {}

  void run_benchmark(::benchmark::State& state) override
  {
    size_t bytes_processed = 0;
    loop_on_state(state, [this, &bytes_processed]() {
      raft::random::permute(perms.data(),
                            (float*)nullptr,
                            (const float*)nullptr,
                            IntType(0),
                            IntType(n_rows),
                            true,
                            stream,
                            123456ULL);
      bytes_processed += size_t(n_rows) * sizeof(IntType);
    });
    state.SetBytesProcessed(bytes_processed);
  }

 private:
  raft::device_resources handle;
  int n_rows;
  rmm::device_uvector<IntType> perms;
};

RAFT_BENCH_REGISTER((permute_perms_only<int>), "", std::vector<int>({32 * 1024, 1024 * 1024, 32 * 1024 * 1024}));
RAFT_BENCH_REGISTER((permute_perms_only<uint32_t>), "", std::vector<int>({1024 * 1024 * 1024}));

}  // namespace raft::bench::random
