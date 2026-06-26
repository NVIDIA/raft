/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/axpy.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {
// Reference axpy implementation.
template <typename T>
RAFT_KERNEL naiveAxpy(const int n, const T alpha, const T* x, T* y, int incx, int incy)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) { y[idx * incy] += alpha * x[idx * incx]; }
}

template <typename InType, typename IndexType = int, typename OutType = InType>
struct AxpyInputs {
  OutType tolerance;
  IndexType len;
  InType alpha;
  IndexType incx;
  IndexType incy;
  unsigned long long int seed;
};

template <typename T, typename IndexType = int>
class AxpyTest : public ::testing::TestWithParam<AxpyInputs<T>> {
 protected:
  raft::resources handle;
  AxpyInputs<T, IndexType> params;
  rmm::device_uvector<T> refy;
  rmm::device_uvector<T> y_device_alpha;
  rmm::device_uvector<T> y_host_alpha;

 public:
  AxpyTest()
    : testing::TestWithParam<AxpyInputs<T>>(),
      refy(0, resource::get_cuda_stream(handle)),
      y_host_alpha(0, resource::get_cuda_stream(handle)),
      y_device_alpha(0, resource::get_cuda_stream(handle))
  {
    resource::sync_stream(handle);
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<AxpyInputs<T>>::GetParam();

    cudaStream_t stream = resource::get_cuda_stream(handle);

    raft::random::RngState r(params.seed);

    IndexType x_len = params.len * params.incx;
    IndexType y_len = params.len * params.incy;
    rmm::device_uvector<T> x(x_len, stream);
    y_host_alpha.resize(y_len, stream);
    y_device_alpha.resize(y_len, stream);
    refy.resize(y_len, stream);

    uniform(handle, r, x.data(), x_len, T(-1.0), T(1.0));
    uniform(handle, r, refy.data(), y_len, T(-1.0), T(1.0));

    // Take a copy of the random generated values in refy
    // this is necessary since axpy uses y for both input and output
    raft::copy(y_host_alpha.data(), refy.data(), y_len, stream);
    raft::copy(y_device_alpha.data(), refy.data(), y_len, stream);

    int threads = 64;
    int blocks  = raft::ceildiv<int>(params.len, threads);

    naiveAxpy<T><<<blocks, threads, 0, stream>>>(
      params.len, params.alpha, x.data(), refy.data(), params.incx, params.incy);

    auto host_alpha_view = make_host_scalar_view<const T>(&params.alpha);

    // test out both axpy overloads - taking either a host scalar or device scalar view
    rmm::device_scalar<T> device_alpha(params.alpha, stream);
    auto device_alpha_view = make_device_scalar_view<const T>(device_alpha.data());

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if ((params.incx > 1) && (params.incy > 1)) {
          auto x_view = make_device_vector_view<const T, IndexType, layout_stride>(
            x.data(), make_vector_strided_layout<IndexType>(params.len, params.incx));
          axpy(h,
               host_alpha_view,
               x_view,
               make_device_vector_view<T, IndexType, layout_stride>(
                 y_host_alpha.data(), make_vector_strided_layout(params.len, params.incy)));
          axpy(h,
               device_alpha_view,
               x_view,
               make_device_vector_view<T, IndexType, layout_stride>(
                 y_device_alpha.data(), make_vector_strided_layout(params.len, params.incy)));
        } else if (params.incx > 1) {
          auto x_view = make_device_vector_view<const T, IndexType, layout_stride>(
            x.data(), make_vector_strided_layout<IndexType>(params.len, params.incx));
          axpy(h,
               host_alpha_view,
               x_view,
               make_device_vector_view<T>(y_host_alpha.data(), params.len));
          axpy(h,
               device_alpha_view,
               x_view,
               make_device_vector_view<T>(y_device_alpha.data(), params.len));
        } else if (params.incy > 1) {
          auto x_view = make_device_vector_view<const T>(x.data(), params.len);
          axpy(h,
               host_alpha_view,
               x_view,
               make_device_vector_view<T, IndexType, layout_stride>(
                 y_host_alpha.data(), make_vector_strided_layout(params.len, params.incy)));
          axpy(h,
               device_alpha_view,
               x_view,
               make_device_vector_view<T, IndexType, layout_stride>(
                 y_device_alpha.data(), make_vector_strided_layout(params.len, params.incy)));
        } else {
          auto x_view = make_device_vector_view<const T>(x.data(), params.len);
          axpy(h,
               host_alpha_view,
               x_view,
               make_device_vector_view<T>(y_host_alpha.data(), params.len));
          axpy(h,
               device_alpha_view,
               x_view,
               make_device_vector_view<T>(y_device_alpha.data(), params.len));
        }
      },
      raft::alloc_behavior::NO_ALLOCATIONS);

    resource::sync_stream(handle);
  }

  void TearDown() override {}
};

const std::vector<AxpyInputs<float>> inputsf = {
  {0.000001f, 1024 * 1024, 2.f, 1, 1, 1234ULL},
  {0.000001f, 16 * 1024 * 1024, 128.f, 1, 1, 1234ULL},
  {0.000001f, 98689, 4.f, 1, 1, 1234ULL},
  {0.000001f, 4 * 1024 * 1024, -1, 1, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 6, 4, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 7, 1, 3, 1234ULL},
  {0.000001f, 1024 * 1024, 8, 4, 3, 1234ULL},
};

const std::vector<AxpyInputs<double>> inputsd = {
  {0.000001f, 1024 * 1024, 2.f, 1, 1, 1234ULL},
  {0.000001f, 16 * 1024 * 1024, 128.f, 1, 1, 1234ULL},
  {0.000001f, 98689, 4.f, 1, 1, 1234ULL},
  {0.000001f, 4 * 1024 * 1024, -1, 1, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 6, 4, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 7, 1, 3, 1234ULL},
  {0.000001f, 1024 * 1024, 8, 4, 3, 1234ULL},
};

typedef AxpyTest<float> AxpyTestF;
TEST_P(AxpyTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(refy.data(),
                                y_host_alpha.data(),
                                params.len * params.incy,
                                raft::CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(raft::devArrMatch(refy.data(),
                                y_device_alpha.data(),
                                params.len * params.incy,
                                raft::CompareApprox<float>(params.tolerance)));
}

typedef AxpyTest<double> AxpyTestD;
TEST_P(AxpyTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(refy.data(),
                                y_host_alpha.data(),
                                params.len * params.incy,
                                raft::CompareApprox<double>(params.tolerance)));
  ASSERT_TRUE(raft::devArrMatch(refy.data(),
                                y_device_alpha.data(),
                                params.len * params.incy,
                                raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(AxpyTests, AxpyTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(AxpyTests, AxpyTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
