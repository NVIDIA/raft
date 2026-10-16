/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/random/permute.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/sample_without_replacement.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/memory_resource>

namespace raft::bench::random {

struct sample_inputs {
  int n_samples;
  int n_train;
  int method;
};  // struct sample_inputs

inline auto operator<<(std::ostream& os, const sample_inputs& p) -> std::ostream&
{
  os << p.n_samples << "#" << p.n_train << "#" << p.method;
  return os;
}

// Sample with replacement. We use this as a baseline.
template <typename IdxT>
auto bernoulli_subsample(raft::resources const& res, IdxT n_samples, IdxT n_subsamples, int seed)
  -> raft::device_vector<IdxT, IdxT>
{
  RAFT_EXPECTS(n_subsamples <= n_samples, "Cannot have more training samples than dataset vectors");

  auto indices = raft::make_device_vector<IdxT, IdxT>(res, n_subsamples);
  raft::random::RngState state(123456ULL);
  raft::random::uniformInt(
    res, state, indices.data_handle(), n_subsamples, IdxT(0), IdxT(n_samples));
  return indices;
}

template <typename T>
struct sample : public fixture {
  sample(const sample_inputs& p)
    : params(p),
      pool_mr(rmm::mr::get_current_device_resource_ref(), 2 * GiB),
      prev_mr(rmm::mr::set_current_device_resource(pool_mr)),
      in(make_device_vector<T, int64_t>(res, p.n_samples)),
      out(make_device_vector<T, int64_t>(res, p.n_train))
  {
    raft::random::RngState r(123456ULL);
  }

  ~sample() { rmm::mr::set_current_device_resource(prev_mr); }
  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    raft::random::RngState r(123456ULL);
    loop_on_state(state, [this, &r]() {
      if (params.method == 1) {
        this->out =
          bernoulli_subsample<T>(this->res, this->params.n_samples, this->params.n_train, 137);
      } else if (params.method == 2) {
        this->out = raft::random::excess_subsample<T, int64_t>(
          this->res, r, this->params.n_samples, this->params.n_train);
      }
    });
  }

 private:
  float GiB = 1073741824.0f;
  raft::device_resources res;
  rmm::mr::pool_memory_resource pool_mr;
  cuda::mr::any_device_resource prev_mr;
  sample_inputs params;
  raft::device_vector<T, int64_t> out, in;
};  // struct sample

const std::vector<sample_inputs> input_vecs = {{100000000, 10000000, 1},
                                               {100000000, 50000000, 1},
                                               {100000000, 100000000, 1},
                                               {100000000, 10000000, 2},
                                               {100000000, 50000000, 2},
                                               {100000000, 100000000, 2}};

RAFT_BENCH_REGISTER(sample<int64_t>, "", input_vecs);

}  // namespace raft::bench::random
