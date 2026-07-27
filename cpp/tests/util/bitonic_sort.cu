/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/random/rng.cuh>
#include <raft/util/bitonic_sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace raft::util {

constexpr int kMaxBlockSize = 512;
constexpr int kMaxCapacity  = 128;

struct test_spec {
  int n_inputs;
  int warp_width;
  int capacity;
  bool ascending;

  [[nodiscard]] auto len() const -> int { return n_inputs * warp_width * capacity; }
};

auto operator<<(std::ostream& os, const test_spec& ss) -> std::ostream&
{
  os << "spec{n_inputs: " << ss.n_inputs << ", input_len: " << (ss.warp_width * ss.capacity) << " ("
     << ss.warp_width << " * " << ss.capacity << ")";
  os << (ss.ascending ? "; asc}" : "; dsc}");
  return os;
}

template <int Capacity, typename T>
RAFT_KERNEL bitonic_kernel(T* arr, bool ascending, int warp_width, int n_inputs)
{
  const int tid          = blockDim.x * blockIdx.x + threadIdx.x;
  const int subwarp_id   = tid / warp_width;
  const int subwarp_lane = tid % warp_width;
  T local_arr[Capacity];  // NOLINT
  // Split the data into chunks of size `warp_width * Capacity`, each thread pointing
  // to the beginning of its stride within the chunk.
  T* per_thread_arr = arr + subwarp_id * warp_width * Capacity + subwarp_lane;

  if (subwarp_id < n_inputs) {
#pragma unroll
    for (int i = 0; i < Capacity; i++) {
      local_arr[i] = per_thread_arr[i * warp_width];
    }
  }

  bitonic<Capacity>(ascending, warp_width).sort(local_arr);

  if (subwarp_id < n_inputs) {
#pragma unroll
    for (int i = 0; i < Capacity; i++) {
      per_thread_arr[i * warp_width] = local_arr[i];
    }
  }
}

template <int Capacity>
struct bitonic_launch {
  template <typename T>
  static void run(const test_spec& spec, T* arr, rmm::cuda_stream_view stream)
  {
    ASSERT(spec.capacity <= Capacity, "Invalid input: the requested capacity is too high.");
    ASSERT(spec.warp_width <= WarpSize,
           "Invalid input: the requested warp_width must be not larger than the WarpSize.");
    if constexpr (Capacity > 1) {
      if (spec.capacity < Capacity) {
        return bitonic_launch<std::max(1, Capacity / 2)>::run(spec, arr, stream);
      }
    }
    int max_block_size, min_grid_size;
    RAFT_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
      &min_grid_size, &max_block_size, bitonic_kernel<Capacity, T>, 0, kMaxBlockSize));
    const int n_warps =
      ceildiv(std::min(spec.n_inputs * spec.warp_width, max_block_size), WarpSize);
    const int block_dim  = n_warps * WarpSize;
    const int n_subwarps = block_dim / spec.warp_width;
    const int grid_dim   = ceildiv(spec.n_inputs, n_subwarps);
    bitonic_kernel<Capacity, T>
      <<<grid_dim, block_dim, 0, stream>>>(arr, spec.ascending, spec.warp_width, spec.n_inputs);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
};

template <typename T>
class BitonicTest : public testing::TestWithParam<test_spec> {  // NOLINT
 protected:
  const test_spec spec;  // NOLINT
  std::vector<T> in;     // NOLINT
  std::vector<T> out;    // NOLINT
  std::vector<T> ref;    // NOLINT
  raft::resources handle_;

  void segmented_sort(std::vector<T>& vec, int k, bool ascending)  // NOLINT
  {
    std::vector<int> p(vec.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), [&vec, k, ascending](int i, int j) {
      const int ik = i / k;
      const int jk = j / k;
      if (ik == jk) { return ascending ? vec[i] < vec[j] : vec[i] > vec[j]; }
      return ik < jk;
    });
    for (auto i = int(vec.size()) - 1; i > 0; i--) {
      auto j = p[i];
      while (j > i)
        j = p[j];
      std::swap(vec[j], vec[i]);
    }
  }

  void fill_random(rmm::device_uvector<T>& arr)
  {
    raft::random::RngState rng(42);
    if constexpr (std::is_floating_point_v<T>) {
      return raft::random::normal(handle_, rng, arr.data(), arr.size(), T(10), T(100));
    }
    if constexpr (std::is_integral_v<T>) {
      return raft::random::normalInt(handle_, rng, arr.data(), arr.size(), T(10), T(100));
    }
  }

 public:
  explicit BitonicTest()
    : spec(testing::TestWithParam<test_spec>::GetParam()),
      in(spec.len()),
      out(spec.len()),
      ref(spec.len())
  {
    auto stream = resource::get_cuda_stream(handle_);

    // generate input
    rmm::device_uvector<T> arr_d(spec.len(), stream);
    fill_random(arr_d);
    update_host(in.data(), arr_d.data(), arr_d.size(), stream);

    // calculate the results
    bitonic_launch<kMaxCapacity>::run(spec, arr_d.data(), stream);
    update_host(out.data(), arr_d.data(), arr_d.size(), stream);

    // make sure the results are available on host
    stream.synchronize();

    // calculate the reference
    std::copy(in.begin(), in.end(), ref.begin());
    segmented_sort(ref, spec.warp_width * spec.capacity, spec.ascending);
  }

  void run() { ASSERT_TRUE(hostVecMatch(ref, out, Compare<T>())); }
};

auto inputs = ::testing::Values(test_spec{1, 1, 1, true},
                                test_spec{1, 2, 1, true},
                                test_spec{1, 4, 1, true},
                                test_spec{1, 8, 1, true},
                                test_spec{1, 16, 1, false},
                                test_spec{1, 32, 1, false},
                                test_spec{1, 32, 2, false},
                                test_spec{1, 32, 4, true},
                                test_spec{1, 32, 8, true},
                                test_spec{5, 32, 2, true},
                                test_spec{7, 16, 4, true},
                                test_spec{7, 8, 2, true},
                                test_spec{70, 4, 32, true},
                                test_spec{70, 1, 64, true},
                                test_spec{70, 2, 128, false});

using Floats = BitonicTest<float>;                     // NOLINT
TEST_P(Floats, Run) { run(); }                         // NOLINT
INSTANTIATE_TEST_CASE_P(BitonicTest, Floats, inputs);  // NOLINT

using Ints = BitonicTest<int>;                       // NOLINT
TEST_P(Ints, Run) { run(); }                         // NOLINT
INSTANTIATE_TEST_CASE_P(BitonicTest, Ints, inputs);  // NOLINT

using Doubles = BitonicTest<double>;                    // NOLINT
TEST_P(Doubles, Run) { run(); }                         // NOLINT
INSTANTIATE_TEST_CASE_P(BitonicTest, Doubles, inputs);  // NOLINT

}  // namespace raft::util
