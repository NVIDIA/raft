/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/dot.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {
// Reference dot implementation.
template <typename T>
RAFT_KERNEL naiveDot(const int n, const T* x, int incx, const T* y, int incy, T* out)
{
  T sum = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    sum += x[i * incx] * y[i * incy];
  }
  atomicAdd(out, sum);
}

template <typename InType, typename IndexType = int, typename OutType = InType>
struct DotInputs {
  OutType tolerance;
  IndexType len;
  IndexType incx;
  IndexType incy;
  unsigned long long int seed;
};

template <typename T, typename IndexType = int>
class DotTest : public ::testing::TestWithParam<DotInputs<T>> {
 protected:
  DotInputs<T, IndexType> params;
  T host_output, device_output, ref_output;

 public:
  DotTest() : testing::TestWithParam<DotInputs<T>>() {}

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<DotInputs<T>>::GetParam();

    raft::resources handle;
    cudaStream_t stream = resource::get_cuda_stream(handle);

    raft::random::RngState r(params.seed);

    IndexType x_len = params.len * params.incx;
    IndexType y_len = params.len * params.incy;

    rmm::device_uvector<T> x(x_len, stream);
    rmm::device_uvector<T> y(y_len, stream);
    uniform(handle, r, x.data(), x_len, T(-1.0), T(1.0));
    uniform(handle, r, y.data(), y_len, T(-1.0), T(1.0));

    rmm::device_scalar<T> ref(0, resource::get_cuda_stream(handle));
    naiveDot<<<256, 256, 0, stream>>>(
      params.len, x.data(), params.incx, y.data(), params.incy, ref.data());
    raft::update_host(&ref_output, ref.data(), 1, stream);

    // Test out both the device and host api's
    rmm::device_scalar<T> out(0, resource::get_cuda_stream(handle));
    auto device_out_view = make_device_scalar_view<T, IndexType>(out.data());
    auto host_out_view   = make_host_scalar_view<T, IndexType>(&host_output);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if ((params.incx > 1) && (params.incy > 1)) {
          auto x_view = make_device_vector_view<const T, IndexType, layout_stride>(
            x.data(), make_vector_strided_layout(params.len, params.incx));
          auto y_view = make_device_vector_view<const T, IndexType, layout_stride>(
            y.data(), make_vector_strided_layout(params.len, params.incy));
          dot(h, x_view, y_view, device_out_view);
          dot(h, x_view, y_view, host_out_view);
        } else if (params.incx > 1) {
          auto x_view = make_device_vector_view<const T, IndexType, layout_stride>(
            x.data(), make_vector_strided_layout(params.len, params.incx));
          auto y_view = make_device_vector_view<const T>(y.data(), params.len);
          dot(h, x_view, y_view, device_out_view);
          dot(h, x_view, y_view, host_out_view);
        } else if (params.incy > 1) {
          auto x_view = make_device_vector_view<const T>(x.data(), params.len);
          auto y_view = make_device_vector_view<const T, IndexType, layout_stride>(
            y.data(), make_vector_strided_layout(params.len, params.incy));
          dot(h, x_view, y_view, device_out_view);
          dot(h, x_view, y_view, host_out_view);
        } else {
          auto x_view = make_device_vector_view<const T>(x.data(), params.len);
          auto y_view = make_device_vector_view<const T>(y.data(), params.len);
          dot(h, x_view, y_view, device_out_view);
          dot(h, x_view, y_view, host_out_view);
        }
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    raft::update_host(&device_output, out.data(), 1, stream);
    resource::sync_stream(handle);
  }

  void TearDown() override {}
};

const std::vector<DotInputs<float>> inputsf = {
  {0.0001f, 1024 * 1024, 1, 1, 1234ULL},
  {0.0001f, 16 * 1024 * 1024, 1, 1, 1234ULL},
  {0.0001f, 98689, 1, 1, 1234ULL},
  {0.0001f, 4 * 1024 * 1024, 1, 1, 1234ULL},
  {0.0001f, 1024 * 1024, 4, 1, 1234ULL},
  {0.0001f, 1024 * 1024, 1, 3, 1234ULL},
  {0.0001f, 1024 * 1024, 4, 3, 1234ULL},
};

const std::vector<DotInputs<double>> inputsd = {
  {0.000001f, 1024 * 1024, 1, 1, 1234ULL},
  {0.000001f, 16 * 1024 * 1024, 1, 1, 1234ULL},
  {0.000001f, 98689, 1, 1, 1234ULL},
  {0.000001f, 4 * 1024 * 1024, 1, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 4, 1, 1234ULL},
  {0.000001f, 1024 * 1024, 1, 3, 1234ULL},
  {0.000001f, 1024 * 1024, 4, 3, 1234ULL},
};

typedef DotTest<float> DotTestF;
TEST_P(DotTestF, Result)
{
  auto compare = raft::CompareApprox<float>(params.tolerance);
  ASSERT_TRUE(compare(ref_output, host_output));
  ASSERT_TRUE(compare(ref_output, device_output));
}

typedef DotTest<double> DotTestD;
TEST_P(DotTestD, Result)
{
  auto compare = raft::CompareApprox<float>(params.tolerance);
  ASSERT_TRUE(compare(ref_output, host_output));
  ASSERT_TRUE(compare(ref_output, device_output));
}

INSTANTIATE_TEST_SUITE_P(DotTests, DotTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(DotTests, DotTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
