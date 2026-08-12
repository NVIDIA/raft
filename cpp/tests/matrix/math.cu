/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/power.cuh>
#include <raft/matrix/ratio.cuh>
#include <raft/matrix/reciprocal.cuh>
#include <raft/matrix/sign_flip.cuh>
#include <raft/matrix/sqrt.cuh>
#include <raft/matrix/threshold.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

template <typename Type>
RAFT_KERNEL naivePowerKernel(Type* in, Type* out, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = in[idx] * in[idx]; }
}

template <typename Type>
void naivePower(Type* in, Type* out, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(stream, nblks, TPB, naivePowerKernel<Type>, in, out, len);
}

template <typename Type>
RAFT_KERNEL naiveSqrtKernel(Type* in, Type* out, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = raft::sqrt(in[idx]); }
}

template <typename Type>
void naiveSqrt(Type* in, Type* out, int len, cudaStream_t stream)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  raft::launch_kernel(stream, nblks, TPB, naiveSqrtKernel<Type>, in, out, len);
}

template <typename Type>
RAFT_KERNEL naiveSignFlipKernel(Type* in, Type* out, int rowCount, int colCount)
{
  int d_i = blockIdx.x * rowCount;
  int end = d_i + rowCount;

  if (blockIdx.x < colCount) {
    Type max      = 0.0;
    int max_index = 0;
    for (int i = d_i; i < end; i++) {
      Type val = in[i];
      if (val < 0.0) { val = -val; }
      if (val > max) {
        max       = val;
        max_index = i;
      }
    }

    for (int i = d_i; i < end; i++) {
      if (in[max_index] < 0.0) {
        out[i] = -in[i];
      } else {
        out[i] = in[i];
      }
    }
  }

  __syncthreads();
}

template <typename Type>
void naiveSignFlip(Type* in, Type* out, int rowCount, int colCount, cudaStream_t stream)
{
  raft::launch_kernel(stream, colCount, 1, naiveSignFlipKernel<Type>, in, out, rowCount, colCount);
}

template <typename T>
struct MathInputs {
  T tolerance;
  int n_row;
  int n_col;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MathInputs<T>& dims)
{
  return os;
}

template <typename T>
class MathTest : public ::testing::TestWithParam<MathInputs<T>> {
 public:
  MathTest()
    : params(::testing::TestWithParam<MathInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      in_power(params.len, stream),
      out_power_ref(params.len, stream),
      in_sqrt(params.len, stream),
      out_sqrt_ref(params.len, stream),
      in_sign_flip(params.len, stream),
      out_sign_flip_ref(params.len, stream),
      in_ratio(4, stream),
      out_ratio_ref(4, stream),
      in_recip(4, stream),
      in_recip_ref(4, stream),
      out_recip(4, stream),
      in_smallzero(4, stream),
      out_smallzero(4, stream),
      out_smallzero_ref(4, stream)
  {
  }

 protected:
  void SetUp() override
  {
    random::RngState r(params.seed);
    int len         = params.len;
    T in_ratio_h[4] = {1.0, 2.0, 2.0, 3.0};
    update_device(in_ratio.data(), in_ratio_h, 4, stream);

    T out_ratio_ref_h[4] = {0.125, 0.25, 0.25, 0.375};
    update_device(out_ratio_ref.data(), out_ratio_ref_h, 4, stream);

    uniform(handle, r, in_power.data(), len, T(-1.0), T(1.0));
    uniform(handle, r, in_sqrt.data(), len, T(0.0), T(1.0));
    // uniform(r, in_ratio, len, T(0.0), T(1.0));
    uniform(handle, r, in_sign_flip.data(), len, T(-100.0), T(100.0));

    naivePower(in_power.data(), out_power_ref.data(), len, stream);

    auto in_power_view = raft::make_device_matrix_view<T>(in_power.data(), len, 1);
    power<T>(handle, in_power_view);

    naiveSqrt(in_sqrt.data(), out_sqrt_ref.data(), len, stream);

    auto in_sqrt_view = raft::make_device_matrix_view(in_sqrt.data(), len, 1);
    sqrt<T>(handle, in_sqrt_view);

    auto in_ratio_view = raft::make_device_matrix_view<T>(in_ratio.data(), 4, 1);
    ratio<T>(handle, in_ratio_view);

    naiveSignFlip(
      in_sign_flip.data(), out_sign_flip_ref.data(), params.n_row, params.n_col, stream);

    auto in_sign_flip_view = raft::make_device_matrix_view<T, int, col_major>(
      in_sign_flip.data(), params.n_row, params.n_col);
    sign_flip<T>(handle, in_sign_flip_view);

    // default threshold is 1e-15
    std::vector<T> in_recip_h     = {0.1, 0.01, -0.01, 0.1e-16};
    std::vector<T> in_recip_ref_h = {10.0, 100.0, -100.0, 0.0};
    update_device(in_recip.data(), in_recip_h.data(), 4, stream);
    update_device(in_recip_ref.data(), in_recip_ref_h.data(), 4, stream);
    T recip_scalar = T(1.0);

    auto in_recip_view  = raft::make_device_matrix_view<const T>(in_recip.data(), 4, 1);
    auto out_recip_view = raft::make_device_matrix_view<T>(out_recip.data(), 4, 1);

    // this `reciprocal()` has to go first bc next one modifies its input
    reciprocal<T>(
      handle, in_recip_view, out_recip_view, raft::make_host_scalar_view(&recip_scalar));

    auto inout_recip_view = raft::make_device_matrix_view<T>(in_recip.data(), 4, 1);

    reciprocal<T>(handle, inout_recip_view, raft::make_host_scalar_view(&recip_scalar), true);

    std::vector<T> in_small_val_zero_h     = {0.1, 1e-16, -1e-16, -0.1};
    std::vector<T> in_small_val_zero_ref_h = {0.1, 0.0, 0.0, -0.1};

    auto in_smallzero_view    = raft::make_device_matrix_view<const T>(in_smallzero.data(), 4, 1);
    auto inout_smallzero_view = raft::make_device_matrix_view<T>(in_smallzero.data(), 4, 1);
    auto out_smallzero_view   = raft::make_device_matrix_view<T>(out_smallzero.data(), 4, 1);

    update_device(in_smallzero.data(), in_small_val_zero_h.data(), 4, stream);
    update_device(out_smallzero_ref.data(), in_small_val_zero_ref_h.data(), 4, stream);
    zero_small_values<T>(handle, in_smallzero_view, out_smallzero_view);
    zero_small_values<T>(handle, inout_smallzero_view);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MathInputs<T> params;
  rmm::device_uvector<T> in_power, out_power_ref, in_sqrt, out_sqrt_ref, in_ratio, out_ratio_ref,
    in_sign_flip, out_sign_flip_ref, in_recip, in_recip_ref, out_recip, in_smallzero, out_smallzero,
    out_smallzero_ref;
};

const std::vector<MathInputs<float>> inputsf = {{0.00001f, 1024, 1024, 1024 * 1024, 1234ULL}};

const std::vector<MathInputs<double>> inputsd = {{0.00001, 1024, 1024, 1024 * 1024, 1234ULL}};

typedef MathTest<float> MathPowerTestF;
TEST_P(MathPowerTestF, Result)
{
  ASSERT_TRUE(devArrMatch(in_power.data(),
                          out_power_ref.data(),
                          params.len,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef MathTest<double> MathPowerTestD;
TEST_P(MathPowerTestD, Result)
{
  ASSERT_TRUE(devArrMatch(in_power.data(),
                          out_power_ref.data(),
                          params.len,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

typedef MathTest<float> MathSqrtTestF;
TEST_P(MathSqrtTestF, Result)
{
  ASSERT_TRUE(devArrMatch(in_sqrt.data(),
                          out_sqrt_ref.data(),
                          params.len,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef MathTest<double> MathSqrtTestD;
TEST_P(MathSqrtTestD, Result)
{
  ASSERT_TRUE(devArrMatch(in_sqrt.data(),
                          out_sqrt_ref.data(),
                          params.len,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

typedef MathTest<float> MathRatioTestF;
TEST_P(MathRatioTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    in_ratio.data(), out_ratio_ref.data(), 4, CompareApprox<float>(params.tolerance), stream));
}

typedef MathTest<double> MathRatioTestD;
TEST_P(MathRatioTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    in_ratio.data(), out_ratio_ref.data(), 4, CompareApprox<double>(params.tolerance), stream));
}

typedef MathTest<float> MathSignFlipTestF;
TEST_P(MathSignFlipTestF, Result)
{
  ASSERT_TRUE(devArrMatch(in_sign_flip.data(),
                          out_sign_flip_ref.data(),
                          params.len,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef MathTest<double> MathSignFlipTestD;
TEST_P(MathSignFlipTestD, Result)
{
  ASSERT_TRUE(devArrMatch(in_sign_flip.data(),
                          out_sign_flip_ref.data(),
                          params.len,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

typedef MathTest<float> MathReciprocalTestF;
TEST_P(MathReciprocalTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    in_recip.data(), in_recip_ref.data(), 4, CompareApprox<float>(params.tolerance), stream));

  // 4-th term tests `setzero=true` functionality, not present in this version of `reciprocal`.
  ASSERT_TRUE(devArrMatch(
    out_recip.data(), in_recip_ref.data(), 3, CompareApprox<float>(params.tolerance), stream));
}

typedef MathTest<double> MathReciprocalTestD;
TEST_P(MathReciprocalTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    in_recip.data(), in_recip_ref.data(), 4, CompareApprox<double>(params.tolerance), stream));

  // 4-th term tests `setzero=true` functionality, not present in this version of `reciprocal`.
  ASSERT_TRUE(devArrMatch(
    out_recip.data(), in_recip_ref.data(), 3, CompareApprox<double>(params.tolerance), stream));
}

typedef MathTest<float> MathSetSmallZeroTestF;
TEST_P(MathSetSmallZeroTestF, Result)
{
  ASSERT_TRUE(devArrMatch(in_smallzero.data(),
                          out_smallzero_ref.data(),
                          4,
                          CompareApprox<float>(params.tolerance),
                          stream));

  ASSERT_TRUE(devArrMatch(out_smallzero.data(),
                          out_smallzero_ref.data(),
                          4,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef MathTest<double> MathSetSmallZeroTestD;
TEST_P(MathSetSmallZeroTestD, Result)
{
  ASSERT_TRUE(devArrMatch(in_smallzero.data(),
                          out_smallzero_ref.data(),
                          4,
                          CompareApprox<double>(params.tolerance),
                          stream));

  ASSERT_TRUE(devArrMatch(out_smallzero.data(),
                          out_smallzero_ref.data(),
                          4,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

INSTANTIATE_TEST_SUITE_P(MathTests, MathPowerTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathPowerTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSqrtTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSqrtTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathRatioTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathRatioTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSignFlipTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSignFlipTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathReciprocalTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MathTests, MathReciprocalTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_SUITE_P(MathTests, MathSetSmallZeroTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_SUITE_P(MathTests, MathSetSmallZeroTestD, ::testing::ValuesIn(inputsd));

}  // namespace matrix
}  // namespace raft
