/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/util/fast_int_div.cuh>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <random>
#include <type_traits>
#include <vector>

namespace raft::bench::util {

constexpr size_t kNumNumerators = 1000 * 1000;
constexpr size_t kNumDivisors   = 100;

/**
 * A single kernel serves both variants: `DivisorT` is either `raft::util::FastIntDiv<IntT>`
 * or a plain `IntT` (native division), and both support `operator/` and `operator%` against
 * an `IntT` numerator.
 */
template <typename IntT, typename DivisorT>
RAFT_KERNEL divmod_kernel(const IntT* numerators,
                          const int64_t n_numerators,
                          const DivisorT* divisors,
                          const int64_t n_divisors,
                          IntT* out)
{
  int64_t tid    = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  int64_t stride = int64_t(gridDim.x) * int64_t(blockDim.x);
  IntT acc       = 0;  // to prevent compiler from optimizing away the division ops
  for (int64_t j = 0; j < n_divisors; ++j) {
    DivisorT divisor = divisors[j];
    for (int64_t i = tid; i < n_numerators; i += stride) {
      IntT n = numerators[i];
      acc ^= n / divisor;
      acc ^= n % divisor;
    }
  }
  out[tid] = acc;
}

template <typename IntT, bool UseFastIntDiv>
struct fast_int_div_bench : public fixture {
  using divisor_t = std::conditional_t<UseFastIntDiv, raft::util::FastIntDiv<IntT>, IntT>;

  explicit fast_int_div_bench()
    : d_numerators(kNumNumerators, stream),
      d_divisors(size_t(kNumDivisors) * sizeof(divisor_t), stream),
      out_d(size_t(kBlocks) * size_t(kThreads), stream)
  {
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<IntT> numerator_dist(std::numeric_limits<IntT>::min(),
                                                       std::numeric_limits<IntT>::max());
    // non-zero, non-neg divisors
    std::uniform_int_distribution<IntT> divisor_dist(1, std::numeric_limits<IntT>::max());

    std::vector<IntT> h_numerators(kNumNumerators);
    for (auto& n : h_numerators) {
      n = numerator_dist(rng);
    }

    std::vector<divisor_t> h_divisors;
    h_divisors.reserve(kNumDivisors);
    for (size_t i = 0; i < kNumDivisors; ++i) {
      h_divisors.push_back(divisor_t(divisor_dist(rng)));
    }

    RAFT_CUDA_TRY(cudaMemcpyAsync(d_numerators.data(),
                                  h_numerators.data(),
                                  h_numerators.size() * sizeof(IntT),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(d_divisors.data(),
                                  h_divisors.data(),
                                  h_divisors.size() * sizeof(divisor_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    stream.synchronize();
  }

  void run_benchmark(::benchmark::State& state) override
  {
    const auto* divisors = static_cast<const divisor_t*>(d_divisors.data());
    loop_on_state(state, [this, divisors]() {
      divmod_kernel<IntT, divisor_t><<<kBlocks, kThreads, 0, stream>>>(
        d_numerators.data(), kNumNumerators, divisors, kNumDivisors, out_d.data());
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    });
  }

 private:
  static constexpr int kThreads = 256;
  static constexpr int kBlocks  = 1024;

  rmm::device_uvector<IntT> d_numerators;
  rmm::device_buffer d_divisors;
  rmm::device_uvector<IntT> out_d;
};

using fast_int_div_i32 = fast_int_div_bench<int32_t, true>;
using native_div_i32   = fast_int_div_bench<int32_t, false>;
using fast_int_div_i64 = fast_int_div_bench<int64_t, true>;
using native_div_i64   = fast_int_div_bench<int64_t, false>;

RAFT_BENCH_REGISTER(fast_int_div_i32, "FastIntDiv/int32");
RAFT_BENCH_REGISTER(native_div_i32, "NativeIntDiv/int32");
RAFT_BENCH_REGISTER(fast_int_div_i64, "FastIntDiv/int64");
RAFT_BENCH_REGISTER(native_div_i64, "NativeIntDiv/int64");

}  // namespace raft::bench::util
