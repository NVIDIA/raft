/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <cuda_fp16.h>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T, typename IdxT>
struct NormInputs {
  T tolerance;
  IdxT rows, cols;
  NormType type;
  bool do_sqrt;
  bool rowMajor;
  unsigned long long int seed;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const NormInputs<T, IdxT>& I)
{
  os << "{ " << I.tolerance << ", " << I.rows << ", " << I.cols << ", " << I.type << ", "
     << I.do_sqrt << ", " << I.seed << '}' << std::endl;
  return os;
}

///// Row-wise norm test definitions
template <typename Type, typename IdxT, typename OutType = Type>
RAFT_KERNEL naiveRowNormKernel(
  OutType* dots, const Type* data, IdxT D, IdxT N, NormType type, bool do_sqrt)
{
  OutType acc   = (OutType)0;
  IdxT rowStart = threadIdx.x + static_cast<IdxT>(blockIdx.x) * blockDim.x;
  if (rowStart < N) {
    for (IdxT i = 0; i < D; ++i) {
      if constexpr (std::is_same_v<Type, half>) {
        if (type == L2Norm) {
          acc += __half2float(data[rowStart * D + i]) * __half2float(data[rowStart * D + i]);
        } else {
          acc += raft::abs(__half2float(data[rowStart * D + i]));
        }
      } else {
        if (type == L2Norm) {
          acc += data[rowStart * D + i] * data[rowStart * D + i];
        } else {
          acc += raft::abs(data[rowStart * D + i]);
        }
      }
    }
    dots[rowStart] = do_sqrt ? raft::sqrt(acc) : acc;
  }
}

template <typename Type, typename IdxT, typename OutType = Type>
void naiveRowNorm(
  OutType* dots, const Type* data, IdxT D, IdxT N, NormType type, bool do_sqrt, cudaStream_t stream)
{
  static const IdxT TPB = 64;
  IdxT nblks            = raft::ceildiv(N, TPB);
  naiveRowNormKernel<Type, IdxT, OutType>
    <<<nblks, TPB, 0, stream>>>(dots, data, D, N, type, do_sqrt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T, typename IdxT, typename OutT = T>
class RowNormTest : public ::testing::TestWithParam<NormInputs<OutT, IdxT>> {
 public:
  RowNormTest()
    : params(::testing::TestWithParam<NormInputs<OutT, IdxT>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      dots_exp(params.rows, stream),
      dots_act(params.rows, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    IdxT rows = params.rows, cols = params.cols, len = rows * cols;
    uniform(handle, r, data.data(), len, T(-1.0), T(1.0));
    naiveRowNorm(dots_exp.data(), data.data(), cols, rows, params.type, params.do_sqrt, stream);
    auto output_view     = raft::make_device_vector_view<OutT, IdxT>(dots_act.data(), params.rows);
    auto input_row_major = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      data.data(), params.rows, params.cols);
    auto input_col_major = raft::make_device_matrix_view<const T, IdxT, raft::col_major>(
      data.data(), params.rows, params.cols);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.do_sqrt) {
          if (params.rowMajor) {
            if (params.type == L2Norm) {
              norm<L2Norm, Apply::ALONG_ROWS>(h, input_row_major, output_view, raft::sqrt_op{});
            } else {
              norm<L1Norm, Apply::ALONG_ROWS>(h, input_row_major, output_view, raft::sqrt_op{});
            }
          } else {
            if (params.type == L2Norm) {
              norm<L2Norm, Apply::ALONG_ROWS>(h, input_col_major, output_view, raft::sqrt_op{});
            } else {
              norm<L1Norm, Apply::ALONG_ROWS>(h, input_col_major, output_view, raft::sqrt_op{});
            }
          }
        } else {
          if (params.rowMajor) {
            if (params.type == L2Norm) {
              norm<L2Norm, Apply::ALONG_ROWS>(h, input_row_major, output_view);
            } else {
              norm<L1Norm, Apply::ALONG_ROWS>(h, input_row_major, output_view);
            }
          } else {
            if (params.type == L2Norm) {
              norm<L2Norm, Apply::ALONG_ROWS>(h, input_col_major, output_view);
            } else {
              norm<L1Norm, Apply::ALONG_ROWS>(h, input_col_major, output_view);
            }
          }
        }
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  NormInputs<OutT, IdxT> params;
  rmm::device_uvector<T> data;
  rmm::device_uvector<OutT> dots_exp, dots_act;
};

///// Column-wise norm test definitisons
template <typename Type, typename IdxT, typename OutType = Type>
RAFT_KERNEL naiveColNormKernel(
  OutType* dots, const Type* data, IdxT D, IdxT N, NormType type, bool do_sqrt)
{
  IdxT colID = threadIdx.x + static_cast<IdxT>(blockIdx.x) * blockDim.x;
  if (colID >= D) return;  // avoid out-of-bounds thread

  OutType acc = 0;
  for (IdxT i = 0; i < N; i++) {
    OutType v = data[colID + i * D];
    acc += type == L2Norm ? v * v : raft::abs(v);
  }

  dots[colID] = do_sqrt ? raft::sqrt(acc) : acc;
}

template <typename Type, typename IdxT, typename OutType = Type>
void naiveColNorm(
  OutType* dots, const Type* data, IdxT D, IdxT N, NormType type, bool do_sqrt, cudaStream_t stream)
{
  static const IdxT TPB = 64;
  IdxT nblks            = raft::ceildiv(D, TPB);
  naiveColNormKernel<Type, IdxT, OutType>
    <<<nblks, TPB, 0, stream>>>(dots, data, D, N, type, do_sqrt);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T, typename IdxT, typename OutT = T>
class ColNormTest : public ::testing::TestWithParam<NormInputs<OutT, IdxT>> {
 public:
  ColNormTest()
    : params(::testing::TestWithParam<NormInputs<OutT, IdxT>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      dots_exp(params.cols, stream),
      dots_act(params.cols, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    IdxT rows = params.rows, cols = params.cols, len = rows * cols;
    uniform(handle, r, data.data(), len, T(-1.0), T(1.0));

    naiveColNorm(dots_exp.data(), data.data(), cols, rows, params.type, params.do_sqrt, stream);
    auto output_view     = raft::make_device_vector_view<OutT, IdxT>(dots_act.data(), params.cols);
    auto input_row_major = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      data.data(), params.rows, params.cols);
    auto input_col_major = raft::make_device_matrix_view<const T, IdxT, raft::col_major>(
      data.data(), params.rows, params.cols);
    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.do_sqrt) {
          if (params.rowMajor) {
            if (params.type == L2Norm) {
              norm<L2Norm, Apply::ALONG_COLUMNS>(h, input_row_major, output_view, raft::sqrt_op{});
            } else {
              norm<L1Norm, Apply::ALONG_COLUMNS>(h, input_row_major, output_view, raft::sqrt_op{});
            }
          } else {
            if (params.type == L2Norm) {
              norm<L2Norm, Apply::ALONG_COLUMNS>(h, input_col_major, output_view, raft::sqrt_op{});
            } else {
              norm<L1Norm, Apply::ALONG_COLUMNS>(h, input_col_major, output_view, raft::sqrt_op{});
            }
          }
        } else {
          if (params.rowMajor) {
            if (params.type == L2Norm) {
              norm<L2Norm, Apply::ALONG_COLUMNS>(h, input_row_major, output_view);
            } else {
              norm<L1Norm, Apply::ALONG_COLUMNS>(h, input_row_major, output_view);
            }
          } else {
            if (params.type == L2Norm) {
              norm<L2Norm, Apply::ALONG_COLUMNS>(h, input_col_major, output_view);
            } else {
              norm<L1Norm, Apply::ALONG_COLUMNS>(h, input_col_major, output_view);
            }
          }
        }
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  NormInputs<OutT, IdxT> params;
  rmm::device_uvector<T> data;
  rmm::device_uvector<OutT> dots_exp, dots_act;
};

///// Row- and column-wise tests
const std::vector<NormInputs<float, int>> inputsf_i32 =
  raft::util::itertools::product<NormInputs<float, int>>(
    {0.00001f}, {11, 1234}, {7, 33, 128, 500}, {L1Norm, L2Norm}, {false, true}, {true}, {1234ULL});
const std::vector<NormInputs<double, int>> inputsd_i32 =
  raft::util::itertools::product<NormInputs<double, int>>({0.00000001},
                                                          {11, 1234},
                                                          {7, 33, 128, 500},
                                                          {L1Norm, L2Norm},
                                                          {false, true},
                                                          {true},
                                                          {1234ULL});
const std::vector<NormInputs<float, int64_t>> inputsf_i64 =
  raft::util::itertools::product<NormInputs<float, int64_t>>(
    {0.00001f}, {11, 1234}, {7, 33, 128, 500}, {L1Norm, L2Norm}, {false, true}, {true}, {1234ULL});
const std::vector<NormInputs<double, int64_t>> inputsd_i64 =
  raft::util::itertools::product<NormInputs<double, int64_t>>({0.00000001},
                                                              {11, 1234},
                                                              {7, 33, 128, 500},
                                                              {L1Norm, L2Norm},
                                                              {false, true},
                                                              {true},
                                                              {1234ULL});
const std::vector<NormInputs<float, int>> inputscf_i32 =
  raft::util::itertools::product<NormInputs<float, int>>(
    {0.00001f}, {7, 33, 128, 500}, {11, 1234}, {L1Norm, L2Norm}, {false, true}, {true}, {1234ULL});
const std::vector<NormInputs<double, int>> inputscd_i32 =
  raft::util::itertools::product<NormInputs<double, int>>({0.00000001},
                                                          {7, 33, 128, 500},
                                                          {11, 1234},
                                                          {L1Norm, L2Norm},
                                                          {false, true},
                                                          {true},
                                                          {1234ULL});
const std::vector<NormInputs<float, int64_t>> inputscf_i64 =
  raft::util::itertools::product<NormInputs<float, int64_t>>(
    {0.00001f}, {7, 33, 128, 500}, {11, 1234}, {L1Norm, L2Norm}, {false, true}, {true}, {1234ULL});
const std::vector<NormInputs<double, int64_t>> inputscd_i64 =
  raft::util::itertools::product<NormInputs<double, int64_t>>({0.00000001},
                                                              {7, 33, 128, 500},
                                                              {11, 1234},
                                                              {L1Norm, L2Norm},
                                                              {false, true},
                                                              {true},
                                                              {1234ULL});

const std::vector<NormInputs<float, int>> inputsh_i32 =
  raft::util::itertools::product<NormInputs<float, int>>(
    {0.00001f}, {11, 1234}, {7, 33, 128, 500}, {L1Norm, L2Norm}, {false, true}, {true}, {1234ULL});
const std::vector<NormInputs<float, int64_t>> inputsh_i64 =
  raft::util::itertools::product<NormInputs<float, int64_t>>(
    {0.00001f}, {11, 1234}, {7, 33, 128, 500}, {L1Norm, L2Norm}, {false, true}, {true}, {1234ULL});
const std::vector<NormInputs<float, int>> inputsch_i32 =
  raft::util::itertools::product<NormInputs<float, int>>(
    {0.00001f}, {7, 33, 128, 500}, {11, 1234}, {L1Norm, L2Norm}, {false, true}, {true}, {1234ULL});
const std::vector<NormInputs<float, int64_t>> inputsch_i64 =
  raft::util::itertools::product<NormInputs<float, int64_t>>(
    {0.00001f}, {7, 33, 128, 500}, {11, 1234}, {L1Norm, L2Norm}, {false, true}, {true}, {1234ULL});

typedef RowNormTest<float, int> RowNormTestF_i32;
typedef RowNormTest<double, int> RowNormTestD_i32;
typedef RowNormTest<float, int64_t> RowNormTestF_i64;
typedef RowNormTest<double, int64_t> RowNormTestD_i64;
typedef ColNormTest<float, int> ColNormTestF_i32;
typedef ColNormTest<double, int> ColNormTestD_i32;
typedef ColNormTest<float, int64_t> ColNormTestF_i64;
typedef ColNormTest<double, int64_t> ColNormTestD_i64;

typedef RowNormTest<half, int, float> RowNormTestH_i32;
typedef RowNormTest<half, int64_t, float> RowNormTestH_i64;
typedef ColNormTest<half, int, float> ColNormTestH_i32;
typedef ColNormTest<half, int64_t, float> ColNormTestH_i64;

#define ROWNORM_TEST(test_type, test_inputs)                                                      \
  TEST_P(test_type, Result)                                                                       \
  {                                                                                               \
    ASSERT_TRUE(raft::devArrMatch(                                                                \
      dots_exp.data(), dots_act.data(), dots_exp.size(), raft::CompareApprox(params.tolerance))); \
  }                                                                                               \
  INSTANTIATE_TEST_CASE_P(RowNormTests, test_type, ::testing::ValuesIn(test_inputs))

ROWNORM_TEST(RowNormTestF_i32, inputsf_i32);
ROWNORM_TEST(RowNormTestD_i32, inputsd_i32);
ROWNORM_TEST(RowNormTestF_i64, inputsf_i64);
ROWNORM_TEST(RowNormTestD_i64, inputsd_i64);
ROWNORM_TEST(ColNormTestF_i32, inputscf_i32);
ROWNORM_TEST(ColNormTestD_i32, inputscd_i32);
ROWNORM_TEST(ColNormTestF_i64, inputscf_i64);
ROWNORM_TEST(ColNormTestD_i64, inputscd_i64);

ROWNORM_TEST(RowNormTestH_i32, inputsh_i32);
ROWNORM_TEST(RowNormTestH_i64, inputsh_i64);
ROWNORM_TEST(ColNormTestH_i32, inputsch_i32);
ROWNORM_TEST(ColNormTestH_i64, inputsch_i64);

}  // end namespace linalg
}  // end namespace raft
