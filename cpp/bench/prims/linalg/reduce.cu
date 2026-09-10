/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/linalg/reduce.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct input_size {
  int rows, cols;
  bool along_rows;
};

template <typename T>
struct reduce : public fixture {
  reduce(bool along_rows, const input_size& p)
    : input_size(p), along_rows(along_rows), in(p.rows * p.cols, stream), out(p.rows, stream)
  {
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      if (along_rows) {
        raft::linalg::reduce<true, true>(
          out.data(), in.data(), input_size.cols, input_size.rows, T(0.f), stream.get());
      } else {
        raft::linalg::reduce<true, false>(
          out.data(), in.data(), input_size.cols, input_size.rows, T(0.f), stream.get());
      }
    });
  }

 private:
  bool along_rows;
  input_size input_size;
  rmm::device_uvector<T> in, out;
};  // struct reduce

const std::vector<input_size> kInputSizes{{8 * 1024, 1024},
                                          {1024, 8 * 1024},
                                          {8 * 1024, 8 * 1024},
                                          {32 * 1024, 1024},
                                          {1024, 32 * 1024},
                                          {32 * 1024, 32 * 1024}};

const std::vector<bool> kAlongRows{false, true};

RAFT_BENCH_REGISTER(reduce<float>, "", kAlongRows, kInputSizes);
RAFT_BENCH_REGISTER(reduce<double>, "", kAlongRows, kInputSizes);

}  // namespace raft::bench::linalg
