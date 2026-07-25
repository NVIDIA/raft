/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/core/memory_tracking_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resources.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <unistd.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

namespace raft::bench::core {

struct tracking_inputs {
  int num_allocs;
  size_t alloc_size;
  int64_t sample_rate_us;
  bool batch;
};

struct tracking_overhead : public fixture {
  tracking_overhead(const tracking_inputs& p) : fixture(true), params(p)
  {
    if (p.sample_rate_us >= 0) {
      std::string tpl = (std::filesystem::temp_directory_path() / "raft_bench_XXXXXX").string();
      int fd          = mkstemp(tpl.data());
      if (fd != -1) close(fd);
      tmp_path_ = std::move(tpl);
      tracked_res_.emplace(handle, tmp_path_, std::chrono::microseconds{p.sample_rate_us});
    }
  }

  ~tracking_overhead()
  {
    tracked_res_.reset();
    if (!tmp_path_.empty()) { std::remove(tmp_path_.c_str()); }
  }

  void run_benchmark(::benchmark::State& state) override
  {
    state.counters["alloc_size"]     = params.alloc_size;
    state.counters["sample_rate_us"] = params.sample_rate_us;
    state.counters["batch"]          = params.batch;

    run_allocs(state, tracked_res_ ? reinterpret_cast<raft::resources&>(*tracked_res_) : handle);

    state.SetItemsProcessed(state.iterations() * params.num_allocs * 2);
  }

 private:
  void run_allocs(::benchmark::State& state, raft::resources& res)
  {
    auto mr = raft::resource::get_workspace_resource_ref(res);
    auto sv = raft::resource::get_cuda_stream(res);

    if (params.batch) {
      std::vector<void*> ptrs(params.num_allocs);
      for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < params.num_allocs; i++)
          ptrs[i] = mr.allocate(sv, params.alloc_size);
        for (int i = params.num_allocs - 1; i >= 0; i--)
          mr.deallocate(sv, ptrs[i], params.alloc_size);
        state.SetIterationTime(
          std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
      }
    } else {
      for (auto _ : state) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < params.num_allocs; i++) {
          void* p = mr.allocate(sv, params.alloc_size);
          mr.deallocate(sv, p, params.alloc_size);
        }
        state.SetIterationTime(
          std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count());
      }
    }
  }

  tracking_inputs params;
  std::string tmp_path_;
  std::optional<raft::memory_tracking_resources> tracked_res_ = std::nullopt;
};

const std::vector<tracking_inputs> inputs{
  // ping-pong (isolates per-call overhead, pool recycles same block)
  {10000, 256, -1, false},
  {10000, 256, 0, false},
  {10000, 256, 1, false},
  {10000, 256, 10, false},
  {10000, 256, 100, false},
  {10000, 1 << 20, -1, false},
  {10000, 1 << 20, 0, false},
  {10000, 1 << 20, 1, false},
  {10000, 1 << 20, 10, false},
  {10000, 1 << 20, 100, false},
  {1000, 1 << 26, -1, false},
  {1000, 1 << 26, 0, false},
  {1000, 1 << 26, 1, false},
  {1000, 1 << 26, 10, false},
  {1000, 1 << 26, 100, false},
  // batch (allocate all, then deallocate all)
  {10000, 256, -1, true},
  {10000, 256, 0, true},
  {10000, 256, 1, true},
  {10000, 256, 10, true},
  {10000, 256, 100, true},
  {1000, 1 << 20, -1, true},
  {1000, 1 << 20, 0, true},
  {1000, 1 << 20, 1, true},
  {1000, 1 << 20, 10, true},
  {1000, 1 << 20, 100, true},
};

RAFT_BENCH_REGISTER(tracking_overhead, "", inputs);

}  // namespace raft::bench::core
