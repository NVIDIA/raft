/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cuda/stream>

#include <benchmark/benchmark.h>

#include <memory>

namespace raft::bench {

/**
 * RAII way to temporary set the pooling memory allocator in rmm.
 * This may be useful for benchmarking functions that do some memory allocations.
 */
struct using_pool_memory_res {
 private:
  rmm::mr::pool_memory_resource pool_res_;
  cuda::mr::any_resource<cuda::mr::device_accessible> prev_res_;

 public:
  using_pool_memory_res(size_t initial_size, size_t max_size)
    : pool_res_(rmm::mr::cuda_memory_resource{}, initial_size, max_size),
      prev_res_(rmm::mr::set_current_device_resource(pool_res_))
  {
  }

  using_pool_memory_res()
    : pool_res_(rmm::mr::cuda_memory_resource{}, rmm::percent_of_free_device_memory(50)),
      prev_res_(rmm::mr::set_current_device_resource(pool_res_))
  {
  }

  ~using_pool_memory_res() { rmm::mr::set_current_device_resource(prev_res_); }
};

/**
 * RAII way of timing cuda calls. This has been shamelessly copied from the
 * cudf codebase via cuml codebase. So, credits for this class goes to cudf developers.
 */
struct cuda_event_timer {
 private:
  ::benchmark::State* state_;
  cuda::stream_ref stream_;
  cudaEvent_t start_;
  cudaEvent_t stop_;

 public:
  /**
   * @param state  the benchmark::State whose timer we are going to update.
   * @param stream CUDA stream we are measuring time on.
   */
  cuda_event_timer(::benchmark::State& state, cuda::stream_ref stream)
    : state_(&state), stream_(stream)
  {
    RAFT_CUDA_TRY(cudaEventCreate(&start_));
    RAFT_CUDA_TRY(cudaEventCreate(&stop_));
    raft::interruptible::synchronize(stream_);
    RAFT_CUDA_TRY(cudaEventRecord(start_, stream_));
  }
  cuda_event_timer() = delete;

  /**
   * @brief The dtor stops the timer and performs a synchroniazation. Time of
   *       the benchmark::State object provided to the ctor will be set to the
   *       value given by `cudaEventElapsedTime()`.
   */
  ~cuda_event_timer()
  {
    RAFT_CUDA_TRY_NO_THROW(cudaEventRecord(stop_, stream_));
    raft::interruptible::synchronize(stop_);
    float milliseconds = 0.0f;
    RAFT_CUDA_TRY_NO_THROW(cudaEventElapsedTime(&milliseconds, start_, stop_));
    state_->SetIterationTime(milliseconds / 1000.f);
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(start_));
    RAFT_CUDA_TRY_NO_THROW(cudaEventDestroy(stop_));
  }
};

/** Main fixture to be inherited and used by all other c++ benchmarks */
class fixture {
 private:
  rmm::device_buffer scratch_buf_;

 public:
  raft::device_resources handle;
  cuda::stream_ref stream;

  explicit fixture(bool use_pool_memory_resource = false)
    : stream{resource::get_cuda_stream(handle)}
  {
    // Cache memory pool between test runs, since it is expensive to create.
    // This speeds up the time required to run the select_k bench by over 3x.
    // This is part of the fixture class here so that the pool will get cleaned
    // up, rather than outliving the benchmarks that require it.
    static std::unique_ptr<using_pool_memory_res> memory_pool;
    if (use_pool_memory_resource) {
      if (!memory_pool) { memory_pool.reset(new using_pool_memory_res()); }
    } else if (memory_pool) {
      memory_pool.reset();
    }

    int l2_cache_size = 0;
    int device_id     = 0;
    RAFT_CUDA_TRY(cudaGetDevice(&device_id));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device_id));
    scratch_buf_ = rmm::device_buffer(l2_cache_size * 3, stream);
  }

  // every benchmark should be overriding this
  virtual void run_benchmark(::benchmark::State& state) = 0;
  virtual void generate_metrics(::benchmark::State& state) {}
  virtual void allocate_data(const ::benchmark::State& state) {}
  virtual void deallocate_data(const ::benchmark::State& state) {}
  virtual void allocate_temp_buffers(const ::benchmark::State& state) {}
  virtual void deallocate_temp_buffers(const ::benchmark::State& state) {}

 protected:
  /** The helper that writes zeroes to some buffer in GPU memory to flush the L2 cache.  */
  void flush_L2_cache()
  {
    RAFT_CUDA_TRY(cudaMemsetAsync(scratch_buf_.data(), 0, scratch_buf_.size(), stream));
  }

  /**
   * The helper to be used inside `run_benchmark`, to loop over the state and record time using the
   * cuda_event_timer.
   */
  template <typename Lambda>
  void loop_on_state(::benchmark::State& state, Lambda benchmark_func, bool flush_L2 = true)
  {
    for (auto _ : state) {
      if (flush_L2) { flush_L2_cache(); }
      cuda_event_timer timer(state, stream);
      benchmark_func();
    }
  }
};

/** Indicates the dataset size. */
struct DatasetParams {
  size_t rows;
  size_t cols;
  bool row_major;
};

/** Holds params needed to generate blobs dataset */
struct BlobsParams {
  int n_clusters;
  double cluster_std;
  bool shuffle;
  double center_box_min, center_box_max;
  uint64_t seed;
};

/** Fixture for cluster benchmarks using make_blobs */
template <typename T, typename IndexT = int>
class BlobsFixture : public fixture {
 public:
  BlobsFixture(const DatasetParams dp, const BlobsParams bp)
    : data_params(dp), blobs_params(bp), X(this->handle)
  {
  }

  virtual void run_benchmark(::benchmark::State& state) = 0;

  void allocate_data(const ::benchmark::State& state) override
  {
    auto labels_ref = raft::make_device_vector<IndexT, IndexT>(this->handle, data_params.rows);
    X = raft::make_device_matrix<T, IndexT>(this->handle, data_params.rows, data_params.cols);

    raft::random::make_blobs<T, IndexT>(X.data_handle(),
                                        labels_ref.data_handle(),
                                        (IndexT)data_params.rows,
                                        (IndexT)data_params.cols,
                                        (IndexT)blobs_params.n_clusters,
                                        stream,
                                        data_params.row_major,
                                        nullptr,
                                        nullptr,
                                        (T)blobs_params.cluster_std,
                                        blobs_params.shuffle,
                                        (T)blobs_params.center_box_min,
                                        (T)blobs_params.center_box_max,
                                        blobs_params.seed);
    resource::sync_stream(this->handle, stream);
  }

 protected:
  DatasetParams data_params;
  BlobsParams blobs_params;
  raft::device_matrix<T, IndexT> X;
};

namespace internal {

template <typename Class, typename... Params>
class Fixture : public ::benchmark::Fixture {
  using State = ::benchmark::State;

 public:
  explicit Fixture(const std::string name, const Params&... params)
    : ::benchmark::Fixture(), params_(params...), name_(name)
  {
    SetName(name_.c_str());
  }
  Fixture() = delete;

  void SetUp(const State& state) override
  {
    fixture_ =
      std::apply([](const Params&... ps) { return std::make_unique<Class>(ps...); }, params_);
    fixture_->allocate_data(state);
    fixture_->allocate_temp_buffers(state);
  }

  void TearDown(const State& state) override
  {
    fixture_->deallocate_temp_buffers(state);
    fixture_->deallocate_data(state);
    fixture_.reset();
  }

  void SetUp(State& st) override { SetUp(const_cast<const State&>(st)); }
  void TearDown(State& st) override { TearDown(const_cast<const State&>(st)); }

 private:
  std::unique_ptr<Class> fixture_;
  std::tuple<Params...> params_;
  const std::string name_;

 protected:
  void BenchmarkCase(State& state) override
  {
    fixture_->run_benchmark(state);
    fixture_->generate_metrics(state);
  }
};  // class Fixture

/**
 * A helper struct to create a fixture for every combination of input vectors.
 * Use with care, this can blow up quickly!
 */
template <typename Class, typename... Params>
struct cartesian_registrar {
  template <typename... Fixed>
  static void run(const std::string case_name,
                  const std::vector<Params>&... params,
                  const Fixed&... fixed);
};

template <typename Class>
struct cartesian_registrar<Class> {
  template <typename... Fixed>
  static void run(const std::string case_name, const Fixed&... fixed)
  {
    auto* b = ::benchmark::internal::RegisterBenchmarkInternal(
      std::unique_ptr<::benchmark::Fixture>(new Fixture<Class, Fixed...>(case_name, fixed...)));
    b->UseManualTime();
    b->Unit(benchmark::kMillisecond);
  }
};

template <typename Class, typename Param, typename... Params>
struct cartesian_registrar<Class, Param, Params...> {
  template <typename... Fixed>
  static void run(const std::string case_name,
                  const std::vector<Param>& param,
                  const std::vector<Params>&... params,
                  const Fixed&... fixed)
  {
    int param_len = param.size();
    for (int i = 0; i < param_len; i++) {
      cartesian_registrar<Class, Params...>::run(
        case_name + "/" + std::to_string(i), params..., fixed..., param[i]);
    }
  }
};

template <typename Class>
struct registrar {
  /**
   * Register a fixture `Class` named `testClass` for every combination of input `params`.
   *
   * @param test_class
   *     A string representation of the `Class` name.
   * @param test_name
   *     Optional test name. Leave empty, if you don't need it.
   * @param params
   *     Zero or more vectors of parameters.
   *     The generated test cases are a cartesian product of these vectors.
   *     Use with care, this can blow up quickly!
   */
  template <typename... Params>
  registrar(const std::string& test_class,
            const std::string& test_name,
            const std::vector<Params>&... params)
  {
    std::stringstream name_stream;
    name_stream << test_class;
    if (!test_name.empty()) { name_stream << "/" << test_name; }
    cartesian_registrar<Class, Params...>::run(name_stream.str(), params...);
  }
};

};  // namespace internal

#define RAFT_BENCH_REGISTER_INTERNAL(TestClass, ...)                                    \
  static raft::bench::internal::registrar<TestClass> BENCHMARK_PRIVATE_NAME(registrar)( \
    RAFT_STRINGIFY(TestClass), __VA_ARGS__)

/**
 * This is the entry point macro for all benchmarks. This needs to be called
 * for the set of benchmarks to be registered so that the main harness inside
 * google bench can find these benchmarks and run them.
 *
 * @param TestClass   child class of `raft::bench::Fixture` which contains
 *                    the logic to generate the dataset and run training on it
 *                    for a given algo. Ideally, once such struct is needed for
 *                    every algo to be benchmarked
 * @param test_name   a unique string to identify these tests at the end of run
 *                    This is optional and if choose not to use this, pass an
 *                    empty string
 * @param params...   zero or more lists of params upon which to benchmark.
 */
#define RAFT_BENCH_REGISTER(TestClass, ...) \
  RAFT_BENCH_REGISTER_INTERNAL(RAFT_DEPAREN(TestClass), __VA_ARGS__)

}  // namespace raft::bench
