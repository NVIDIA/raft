/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/sparse/convert/csr.cuh>

#include <rmm/device_uvector.hpp>

#include <stdio.h>

namespace raft::bench::sparse {

template <typename index_t>
struct bench_param {
  index_t num_cols;
  index_t num_rows;
  index_t divisor;
};

template <typename index_t>
RAFT_KERNEL init_adj_kernel(bool* adj, index_t num_rows, index_t num_cols, index_t divisor)
{
  index_t r = blockDim.y * blockIdx.y + threadIdx.y;
  index_t c = blockDim.x * blockIdx.x + threadIdx.x;

  for (; r < num_rows; r += gridDim.y * blockDim.y) {
    for (; c < num_cols; c += gridDim.x * blockDim.x) {
      adj[r * num_cols + c] = c % divisor == 0;
    }
  }
}

template <typename index_t>
void init_adj(bool* adj, index_t num_rows, index_t num_cols, index_t divisor, cudaStream_t stream)
{
  // adj matrix: element a_ij is set to one if j is divisible by divisor.
  dim3 block(32, 32);
  const index_t max_y_grid_dim = 65535;
  dim3 grid(num_cols / 32 + 1, (int)min(num_rows / 32 + 1, max_y_grid_dim));
  init_adj_kernel<index_t><<<grid, block, 0, stream>>>(adj, num_rows, num_cols, divisor);
  RAFT_CHECK_CUDA(stream);
}

template <typename index_t>
struct bench_base : public fixture {
  bench_base(const bench_param<index_t>& p)
    : params(p),
      handle(stream),
      adj(p.num_rows * p.num_cols, stream),
      row_ind(p.num_rows, stream),
      row_ind_host(p.num_rows),
      row_counters(p.num_rows, stream),
      // col_ind is over-dimensioned because nnz is unknown at this point
      col_ind(p.num_rows * p.num_cols, stream)
  {
    init_adj(adj.data(), p.num_rows, p.num_cols, p.divisor, stream.get());

    std::vector<index_t> row_ind_host(p.num_rows);
    for (size_t i = 0; i < row_ind_host.size(); ++i) {
      size_t nnz_per_row = raft::ceildiv(p.num_cols, p.divisor);
      row_ind_host[i]    = nnz_per_row * i;
    }
    raft::update_device(row_ind.data(), row_ind_host.data(), row_ind.size(), stream);
  }

  void run_benchmark(::benchmark::State& state) override
  {
    loop_on_state(state, [this]() {
      raft::sparse::convert::adj_to_csr<index_t>(handle,
                                                 adj.data(),
                                                 row_ind.data(),
                                                 params.num_rows,
                                                 params.num_cols,
                                                 row_counters.data(),
                                                 col_ind.data());
    });

    // Estimate bandwidth:
    index_t num_entries = params.num_rows * params.num_cols;
    index_t bytes_read  = num_entries * sizeof(bool);
    index_t bytes_write = num_entries / params.divisor * sizeof(index_t);

    state.counters["BW"]      = benchmark::Counter(bytes_read + bytes_write,
                                              benchmark::Counter::kIsIterationInvariantRate,
                                              benchmark::Counter::OneK::kIs1024);
    state.counters["BW read"] = benchmark::Counter(
      bytes_read, benchmark::Counter::kIsIterationInvariantRate, benchmark::Counter::OneK::kIs1024);
    state.counters["BW write"] = benchmark::Counter(bytes_write,
                                                    benchmark::Counter::kIsIterationInvariantRate,
                                                    benchmark::Counter::OneK::kIs1024);

    state.counters["Fraction nz"] = benchmark::Counter(100.0 / ((double)params.divisor));
    state.counters["Columns"]     = benchmark::Counter(params.num_cols);
    state.counters["Rows"]        = benchmark::Counter(params.num_rows);
  }

 protected:
  raft::device_resources handle;
  bench_param<index_t> params;
  rmm::device_uvector<bool> adj;
  rmm::device_uvector<index_t> row_ind;
  std::vector<index_t> row_ind_host;
  rmm::device_uvector<index_t> row_counters;
  rmm::device_uvector<index_t> col_ind;
};  // struct bench_base

const int64_t num_cols = 1 << 30;

const std::vector<bench_param<int64_t>> bench_params = {
  {num_cols, 1, 8},
  {num_cols >> 3, 1 << 3, 8},
  {num_cols >> 6, 1 << 6, 8},

  {num_cols, 1, 64},
  {num_cols >> 3, 1 << 3, 64},
  {num_cols >> 6, 1 << 6, 64},

  {num_cols, 1, 2048},
  {num_cols >> 3, 1 << 3, 2048},
  {num_cols >> 6, 1 << 6, 2048},
};

RAFT_BENCH_REGISTER(bench_base<int64_t>, "", bench_params);

}  // namespace raft::bench::sparse
