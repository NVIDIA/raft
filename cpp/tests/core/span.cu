/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "test_span.hpp"

#include <raft/core/device_span.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>

#include <gtest/gtest.h>

#include <numeric>  // iota

namespace raft {
struct TestStatus {
 private:
  int* status_;

 public:
  TestStatus()
  {
    RAFT_CUDA_TRY(cudaMalloc(&status_, sizeof(int)));
    int h_status = 1;
    RAFT_CUDA_TRY(cudaMemcpy(status_, &h_status, sizeof(int), cudaMemcpyHostToDevice));
  }
  ~TestStatus() noexcept(false) { RAFT_CUDA_TRY(cudaFree(status_)); }

  int Get()
  {
    int h_status;
    RAFT_CUDA_TRY(cudaMemcpy(&h_status, status_, sizeof(int), cudaMemcpyDeviceToHost));
    return h_status;
  }

  int* Data() { return status_; }
};

RAFT_KERNEL TestFromOtherKernel(device_span<float> span)
{
  // don't get optimized out
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= span.size()) { return; }
}
// Test converting different T
RAFT_KERNEL TestFromOtherKernelConst(device_span<float const, 16> span)
{
  // don't get optimized out
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= span.size()) { return; }
}

/*!
 * \brief Here we just test whether the code compiles.
 */
TEST(GPUSpan, FromOther)
{
  thrust::host_vector<float> h_vec(16);
  std::iota(h_vec.begin(), h_vec.end(), 0);

  thrust::device_vector<float> d_vec(h_vec.size());
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());
  // dynamic extent
  {
    device_span<float> span(d_vec.data().get(), d_vec.size());
    raft::launch_kernel(cudaStream_t{0}, 1, 16, TestFromOtherKernel, span);
  }
  {
    device_span<float> span(d_vec.data().get(), d_vec.size());
    raft::launch_kernel(cudaStream_t{0}, 1, 16, TestFromOtherKernelConst, span);
  }
  // static extent
  {
    device_span<float, 16> span(d_vec.data().get(), d_vec.data().get() + 16);
    raft::launch_kernel(cudaStream_t{0}, 1, 16, TestFromOtherKernel, span);
  }
  {
    device_span<float, 16> span(d_vec.data().get(), d_vec.data().get() + 16);
    raft::launch_kernel(cudaStream_t{0}, 1, 16, TestFromOtherKernelConst, span);
  }
}

TEST(GPUSpan, Assignment)
{
  TestStatus status;
  thrust::for_each_n(cuda::make_counting_iterator(0ul), 16, test_assignment_t<true>{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpan, TestStatus)
{
  TestStatus status;
  thrust::for_each_n(cuda::make_counting_iterator(0ul), 16, test_test_status_t{status.Data()});
  ASSERT_EQ(status.Get(), -1);
}

template <typename T>
struct TestEqual {
 private:
  T *lhs_, *rhs_;
  int* status_;

 public:
  TestEqual(T* _lhs, T* _rhs, int* _status) : lhs_(_lhs), rhs_(_rhs), status_(_status) {}

  HD void operator()(size_t _idx)
  {
    bool res = lhs_[_idx] == rhs_[_idx];
    SPAN_ASSERT_TRUE(res, status_);
  }
};

TEST(GPUSpan, WithTrust)
{
  // Not advised to initialize span with host_vector, since h_vec.data() is
  // a host function.
  thrust::host_vector<float> h_vec(16);
  std::iota(h_vec.begin(), h_vec.end(), 0);

  thrust::device_vector<float> d_vec(h_vec.size());
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

  // Can't initialize span with device_vector, since d_vec.data() is not raw
  // pointer
  {
    device_span<float> s(d_vec.data().get(), d_vec.size());

    ASSERT_EQ(d_vec.size(), s.size());
    ASSERT_EQ(d_vec.data().get(), s.data());
  }

  {
    TestStatus status;
    thrust::device_vector<float> d_vec1(d_vec.size());
    thrust::copy(thrust::device, d_vec.begin(), d_vec.end(), d_vec1.begin());
    device_span<float> s(d_vec1.data().get(), d_vec.size());

    thrust::for_each_n(
      cuda::make_counting_iterator(0ul),
      16,
      TestEqual<float>{thrust::raw_pointer_cast(d_vec1.data()), s.data(), status.Data()});
    ASSERT_EQ(status.Get(), 1);
  }
}

TEST(GPUSpan, BeginEnd)
{
  TestStatus status;
  thrust::for_each_n(cuda::make_counting_iterator(0ul), 16, test_beginend_t<true>{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpan, RBeginREnd)
{
  TestStatus status;
  thrust::for_each_n(cuda::make_counting_iterator(0ul), 16, test_rbeginrend_t<true>{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

RAFT_KERNEL TestModifyKernel(device_span<float> span)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

  if (idx >= span.size()) { return; }
  span[idx] = span.size() - idx;
}

TEST(GPUSpan, Modify)
{
  thrust::host_vector<float> h_vec(16);
  std::iota(h_vec.begin(), h_vec.end(), 0);

  thrust::device_vector<float> d_vec(h_vec.size());
  thrust::copy(h_vec.begin(), h_vec.end(), d_vec.begin());

  device_span<float> span(d_vec.data().get(), d_vec.size());

  raft::launch_kernel(cudaStream_t{0}, 1, 16, TestModifyKernel, span);

  for (size_t i = 0; i < d_vec.size(); ++i) {
    ASSERT_EQ(d_vec[i], d_vec.size() - i);
  }
}

TEST(GPUSpan, Observers)
{
  TestStatus status;
  thrust::for_each_n(cuda::make_counting_iterator(0ul), 16, test_observers_t<true>{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}

TEST(GPUSpan, Compare)
{
  TestStatus status;
  thrust::for_each_n(cuda::make_counting_iterator(0), 1, test_compare_t<false>{status.Data()});
  ASSERT_EQ(status.Get(), 1);
}
}  // namespace raft
