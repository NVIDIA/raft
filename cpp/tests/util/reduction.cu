/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/random/device/sample.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/reduction.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace raft::util {

constexpr int max_warps_per_block = 32;

template <typename ReduceLambda>
RAFT_KERNEL test_reduction_kernel(const int* input, int* reduction_res, ReduceLambda reduce_op)
{
  assert(gridDim.x == 1);
  __shared__ int red_buf[max_warps_per_block];
  int th_val = input[threadIdx.x];
  th_val     = raft::blockReduce(th_val, (char*)red_buf, reduce_op);
  if (threadIdx.x == 0) { reduction_res[0] = th_val; }
}

// regression: warpReduce(val, raft::add_op{}) must not cause ambiguity with CUB on CUDA 13.2+
RAFT_KERNEL test_warp_reduce_with_add_op_kernel(const int* input, int* reduction_res)
{
  assert(gridDim.x == 1);
  int th_val = input[threadIdx.x];
  th_val     = raft::warpReduce(th_val, raft::add_op{});
  if (threadIdx.x % 32 == 0) { atomicAdd(reduction_res, th_val); }
}

template <typename ReduceLambda>
RAFT_KERNEL test_ranked_reduction_kernel(const int* input,
                                         int* reduction_res,
                                         int* out_rank,
                                         ReduceLambda reduce_op)
{
  assert(gridDim.x == 1);
  __shared__ int red_buf[2 * max_warps_per_block];
  int th_val  = input[threadIdx.x];
  int th_rank = threadIdx.x;
  auto result = raft::blockRankedReduce(th_val, red_buf, th_rank, reduce_op);
  if (threadIdx.x == 0) {
    reduction_res[0] = result.first;
    out_rank[0]      = result.second;
  }
}

RAFT_KERNEL test_block_random_sample_kernel(const int* input, int* reduction_res)
{
  assert(gridDim.x == 1);
  __shared__ int red_buf[2 * max_warps_per_block];
  raft::random::PCGenerator thread_rng(1234, threadIdx.x, 0);
  int th_val  = input[threadIdx.x];
  int th_rank = threadIdx.x;
  int result  = raft::random::device::block_random_sample(thread_rng, red_buf, th_val, th_rank);
  if (threadIdx.x == 0) { reduction_res[0] = result; }
}

template <int TPB>
RAFT_KERNEL test_binary_reduction_kernel(const int* input, int* reduction_res)
{
  assert(gridDim.x == 1);
  __shared__ int shared[TPB / WarpSize];
  int th_val = input[threadIdx.x];
  int result = raft::binaryBlockReduce<TPB>(th_val, shared);
  if (threadIdx.x == 0) { reduction_res[0] = result; }
}

struct reduction_launch {
  template <typename ReduceLambda>
  static void run(const rmm::device_uvector<int>& arr_d,
                  int ref_val,
                  ReduceLambda reduce_op,
                  rmm::cuda_stream_view stream)
  {
    rmm::device_scalar<int> ref_d(stream);
    const int block_dim = 64;
    const int grid_dim  = 1;
    test_reduction_kernel<<<grid_dim, block_dim, 0, stream>>>(
      arr_d.data(), ref_d.data(), reduce_op);
    stream.synchronize();
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    ASSERT_EQ(ref_d.value(stream), ref_val);
  }

  template <typename ReduceLambda>
  static void run_ranked(const rmm::device_uvector<int>& arr_d,
                         int ref_val,
                         int rank_ref_val,
                         ReduceLambda reduce_op,
                         rmm::cuda_stream_view stream)
  {
    rmm::device_scalar<int> ref_d(stream);
    rmm::device_scalar<int> rank_d(stream);
    const int block_dim = 64;
    const int grid_dim  = 1;
    test_ranked_reduction_kernel<<<grid_dim, block_dim, 0, stream>>>(
      arr_d.data(), ref_d.data(), rank_d.data(), reduce_op);
    stream.synchronize();
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    ASSERT_EQ(ref_d.value(stream), ref_val);
    ASSERT_EQ(rank_d.value(stream), rank_ref_val);
  }

  static void run_random_sample(const rmm::device_uvector<int>& arr_d,
                                int ref_val,
                                rmm::cuda_stream_view stream)
  {
    rmm::device_scalar<int> ref_d(stream);
    const int block_dim = 64;
    const int grid_dim  = 1;
    test_block_random_sample_kernel<<<grid_dim, block_dim, 0, stream>>>(arr_d.data(), ref_d.data());
    stream.synchronize();
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    ASSERT_EQ(ref_d.value(stream), ref_val);
  }

  static void run_binary(const rmm::device_uvector<int>& arr_d,
                         int ref_val,
                         rmm::cuda_stream_view stream)
  {
    rmm::device_scalar<int> ref_d(stream);
    constexpr int block_dim = 64;
    const int grid_dim      = 1;
    test_binary_reduction_kernel<block_dim>
      <<<grid_dim, block_dim, 0, stream>>>(arr_d.data(), ref_d.data());
    stream.synchronize();
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    ASSERT_EQ(ref_d.value(stream), ref_val);
  }

  static void run_warp_reduce_with_add_op(const rmm::device_uvector<int>& arr_d,
                                          int ref_val,
                                          rmm::cuda_stream_view stream)
  {
    rmm::device_scalar<int> ref_d(0, stream);
    const int block_dim = 64;
    const int grid_dim  = 1;
    test_warp_reduce_with_add_op_kernel<<<grid_dim, block_dim, 0, stream>>>(arr_d.data(),
                                                                             ref_d.data());
    stream.synchronize();
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    ASSERT_EQ(ref_d.value(stream), ref_val);
  }
};

template <typename T>
class ReductionTest : public testing::TestWithParam<std::vector<int>> {  // NOLINT
 protected:
  const std::vector<int> input;    // NOLINT
  rmm::cuda_stream_view stream;    // NOLINT
  rmm::device_uvector<int> arr_d;  // NOLINT

 public:
  explicit ReductionTest()
    : input(testing::TestWithParam<std::vector<int>>::GetParam()),
      stream(rmm::cuda_stream_default),
      arr_d(input.size(), stream)
  {
    update_device(arr_d.data(), input.data(), input.size(), stream);
  }

  void run_reduction()
  {
    // calculate the results
    reduction_launch::run(arr_d, 0, raft::min_op{}, stream);
    reduction_launch::run(arr_d, 5, raft::max_op{}, stream);
    reduction_launch::run(arr_d, 158, raft::add_op{}, stream);
    reduction_launch::run_ranked(arr_d, 5, 15, raft::max_op{}, stream);
    reduction_launch::run_ranked(arr_d, 0, 26, raft::min_op{}, stream);
    // value 15 is for the current state of PCgenerator. adjust this if rng changes
    reduction_launch::run_random_sample(arr_d, 15, stream);
  }

  void run_binary_reduction() { reduction_launch::run_binary(arr_d, 24, stream); }

  void run_warp_reduce_with_add_op()
  {
    // two warps of 32 threads; warp 0 sums elements 0-31 (=78), warp 1 sums 32-63 (=80)
    reduction_launch::run_warp_reduce_with_add_op(arr_d, 158, stream);
  }
};

const std::vector<int> test_vector{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 3, 4, 1, 2,
                                   3, 4, 1, 2, 0, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
                                   1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4};
const std::vector<int> binary_test_vector{
  1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0,
  1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0};
auto reduction_input        = ::testing::Values(test_vector);
auto binary_reduction_input = ::testing::Values(binary_test_vector);

using ReductionTestInt              = ReductionTest<int>;  // NOLINT
using BinaryReductionTestInt        = ReductionTest<int>;  // NOLINT
using WarpReduceAddOpTestInt        = ReductionTest<int>;  // NOLINT
TEST_P(ReductionTestInt, REDUCTIONS) { run_reduction(); }
INSTANTIATE_TEST_CASE_P(ReductionTest, ReductionTestInt, reduction_input);    // NOLINT
TEST_P(BinaryReductionTestInt, BINARY_REDUCTION) { run_binary_reduction(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(BinaryReductionTest,
                        BinaryReductionTestInt,
                        binary_reduction_input);  // NOLINT
TEST_P(WarpReduceAddOpTestInt, WARP_REDUCE_WITH_ADD_OP) { run_warp_reduce_with_add_op(); }  // NOLINT
INSTANTIATE_TEST_CASE_P(WarpReduceAddOpTest,
                        WarpReduceAddOpTestInt,
                        reduction_input);  // NOLINT

}  // namespace raft::util
