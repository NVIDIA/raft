/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/random/make_blobs.cuh>

#include <rmm/device_uvector.hpp>

#include <vector>

namespace raft::bench::random {
struct make_blobs_inputs {
  int rows, cols, clusters;
  bool row_major;
};  // struct make_blobs_inputs

inline auto operator<<(std::ostream& os, const make_blobs_inputs& p) -> std::ostream&
{
  os << p.rows << "#" << p.cols << "#" << p.clusters << "#" << p.row_major;
  return os;
}

template <typename T>
struct make_blobs : public fixture {
  make_blobs(const make_blobs_inputs& p)
    : params(p), data(p.rows * p.cols, stream), labels(p.rows, stream)
  {
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    loop_on_state(state, [this]() {
      raft::random::make_blobs(data.data(),
                               labels.data(),
                               params.rows,
                               params.cols,
                               params.clusters,
                               this->stream.get(),
                               params.row_major);
    });
  }

 private:
  make_blobs_inputs params;
  rmm::device_uvector<T> data;
  rmm::device_uvector<int> labels;
};  // struct MakeBlobs

static std::vector<make_blobs_inputs> get_make_blobs_input_vecs()
{
  std::vector<make_blobs_inputs> out;
  make_blobs_inputs p;
  for (auto rows : std::vector<int>{100000, 1000000}) {
    for (auto cols : std::vector<int>{10, 100}) {
      for (auto clusters : std::vector<int>{2, 10, 100}) {
        p.rows      = rows;
        p.cols      = cols;
        p.clusters  = clusters;
        p.row_major = true;
        out.push_back(p);
        p.row_major = false;
        out.push_back(p);
      }
    }
  }
  return out;
}

RAFT_BENCH_REGISTER(make_blobs<float>, "", get_make_blobs_input_vecs());
RAFT_BENCH_REGISTER(make_blobs<double>, "", get_make_blobs_input_vecs());

}  // namespace raft::bench::random
