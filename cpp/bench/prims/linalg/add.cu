/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/linalg/add.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

struct add_inputs {
  int len;
};  // struct add_inputs

template <typename T>
struct add : public fixture {
  add(const add_inputs& p) : params(p), ptr0(p.len, stream), ptr1(p.len, stream) {}

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::linalg::add(ptr0.data(), ptr0.data(), ptr1.data(), params.len, stream.get());
    });
  }

 private:
  add_inputs params;
  rmm::device_uvector<T> ptr0, ptr1;
};  // struct add

const std::vector<add_inputs> add_input_vecs{
  {256 * 1024 * 1024}, {256 * 1024 * 1024 + 2}, {256 * 1024 * 1024 + 1}

};

RAFT_BENCH_REGISTER(add<float>, "", add_input_vecs);
RAFT_BENCH_REGISTER(add<double>, "", add_input_vecs);

}  // namespace raft::bench::linalg
