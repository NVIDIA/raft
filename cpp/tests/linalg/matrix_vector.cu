/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "matrix_vector_op.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T, typename IdxType = int>
struct MatrixVectorInputs {
  T tolerance;
  IdxType rows, cols;
  int operation_type;
  bool row_major, bcast_along_rows;
  unsigned long long int seed;
};

template <typename T, typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const MatrixVectorInputs<T, IdxType>& dims)
{
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T, typename IdxType>
void matrix_vector_op_launch(const raft::resources& handle,
                             T* in,
                             const T* vec1,
                             IdxType D,
                             IdxType N,
                             bool row_major,
                             bool bcast_along_rows,
                             int operation_type)
{
  auto in_row_major = raft::make_device_matrix_view<T, IdxType, raft::row_major>(in, N, D);
  auto in_col_major = raft::make_device_matrix_view<T, IdxType, raft::col_major>(in, N, D);

  auto apply     = bcast_along_rows ? Apply::ALONG_ROWS : Apply::ALONG_COLUMNS;
  auto len       = bcast_along_rows ? D : N;
  auto vec1_view = raft::make_device_vector_view<const T, IdxType>(vec1, len);

  if (operation_type == 0) {
    if (row_major) {
      if (apply == Apply::ALONG_ROWS) {
        binary_mult_skip_zero<Apply::ALONG_ROWS>(handle, in_row_major, vec1_view);
      } else {
        binary_mult_skip_zero<Apply::ALONG_COLUMNS>(handle, in_row_major, vec1_view);
      }
    } else {
      if (apply == Apply::ALONG_ROWS) {
        binary_mult_skip_zero<Apply::ALONG_ROWS>(handle, in_col_major, vec1_view);
      } else {
        binary_mult_skip_zero<Apply::ALONG_COLUMNS>(handle, in_col_major, vec1_view);
      }
    }
  } else if (operation_type == 1) {
    if (row_major) {
      if (apply == Apply::ALONG_ROWS) {
        binary_div<Apply::ALONG_ROWS>(handle, in_row_major, vec1_view);
      } else {
        binary_div<Apply::ALONG_COLUMNS>(handle, in_row_major, vec1_view);
      }
    } else {
      if (apply == Apply::ALONG_ROWS) {
        binary_div<Apply::ALONG_ROWS>(handle, in_col_major, vec1_view);
      } else {
        binary_div<Apply::ALONG_COLUMNS>(handle, in_col_major, vec1_view);
      }
    }
  } else if (operation_type == 2) {
    if (row_major) {
      if (apply == Apply::ALONG_ROWS) {
        binary_div_skip_zero<Apply::ALONG_ROWS>(handle, in_row_major, vec1_view, false);
      } else {
        binary_div_skip_zero<Apply::ALONG_COLUMNS>(handle, in_row_major, vec1_view, false);
      }
    } else {
      if (apply == Apply::ALONG_ROWS) {
        binary_div_skip_zero<Apply::ALONG_ROWS>(handle, in_col_major, vec1_view, false);
      } else {
        binary_div_skip_zero<Apply::ALONG_COLUMNS>(handle, in_col_major, vec1_view, false);
      }
    }
  } else if (operation_type == 3) {
    if (row_major) {
      if (apply == Apply::ALONG_ROWS) {
        binary_add<Apply::ALONG_ROWS>(handle, in_row_major, vec1_view);
      } else {
        binary_add<Apply::ALONG_COLUMNS>(handle, in_row_major, vec1_view);
      }
    } else {
      if (apply == Apply::ALONG_ROWS) {
        binary_add<Apply::ALONG_ROWS>(handle, in_col_major, vec1_view);
      } else {
        binary_add<Apply::ALONG_COLUMNS>(handle, in_col_major, vec1_view);
      }
    }
  } else if (operation_type == 4) {
    if (row_major) {
      if (apply == Apply::ALONG_ROWS) {
        binary_sub<Apply::ALONG_ROWS>(handle, in_row_major, vec1_view);
      } else {
        binary_sub<Apply::ALONG_COLUMNS>(handle, in_row_major, vec1_view);
      }
    } else {
      if (apply == Apply::ALONG_ROWS) {
        binary_sub<Apply::ALONG_ROWS>(handle, in_col_major, vec1_view);
      } else {
        binary_sub<Apply::ALONG_COLUMNS>(handle, in_col_major, vec1_view);
      }
    }
  } else if (operation_type == 5) {
    if (row_major) {
      if (apply == Apply::ALONG_ROWS) {
        binary_mult<Apply::ALONG_ROWS>(handle, in_row_major, vec1_view);
      } else {
        binary_mult<Apply::ALONG_COLUMNS>(handle, in_row_major, vec1_view);
      }
    } else {
      if (apply == Apply::ALONG_ROWS) {
        binary_mult<Apply::ALONG_ROWS>(handle, in_col_major, vec1_view);
      } else {
        binary_mult<Apply::ALONG_COLUMNS>(handle, in_col_major, vec1_view);
      }
    }
  } else {
    THROW("Unknown operation type '%d'!", (int)operation_type);
  }
}

template <typename T, typename IdxType>
void naive_matrix_vector_op_launch(const raft::resources& handle,
                                   T* in,
                                   const T* vec1,
                                   IdxType D,
                                   IdxType N,
                                   bool row_major,
                                   bool bcast_along_rows,
                                   int operation_type)
{
  auto stream                       = resource::get_cuda_stream(handle).get();
  auto operation_bin_mult_skip_zero = [] __device__(T mat_element, T vec_element) {
    if (vec_element != T(0)) {
      return mat_element * vec_element;
    } else {
      return mat_element;
    }
  };
  auto operation_bin_div_skip_zero = [] __device__(T mat_element, T vec_element) {
    if (raft::abs(vec_element) < T(1e-10))
      return T(0);
    else
      return mat_element / vec_element;
  };

  if (operation_type == 0) {
    naiveMatVec(
      in, in, vec1, D, N, row_major, bcast_along_rows, operation_bin_mult_skip_zero, stream);
  } else if (operation_type == 1) {
    naiveMatVec(in, in, vec1, D, N, row_major, bcast_along_rows, raft::div_op{}, stream);
  } else if (operation_type == 2) {
    naiveMatVec(
      in, in, vec1, D, N, row_major, bcast_along_rows, operation_bin_div_skip_zero, stream);
  } else if (operation_type == 3) {
    naiveMatVec(in, in, vec1, D, N, row_major, bcast_along_rows, raft::add_op{}, stream);
  } else if (operation_type == 4) {
    naiveMatVec(in, in, vec1, D, N, row_major, bcast_along_rows, raft::sub_op{}, stream);
  } else if (operation_type == 5) {
    naiveMatVec(in, in, vec1, D, N, row_major, bcast_along_rows, raft::mul_op{}, stream);
  } else {
    THROW("Unknown operation type '%d'!", (int)operation_type);
  }
}

template <typename T, typename IdxType>
class MatrixVectorTest : public ::testing::TestWithParam<MatrixVectorInputs<T, IdxType>> {
 public:
  MatrixVectorTest()
    : params(::testing::TestWithParam<MatrixVectorInputs<T, IdxType>>::GetParam()),
      stream(resource::get_cuda_stream(handle).get()),
      in(params.rows * params.cols, stream),
      out_ref(params.rows * params.cols, stream),
      out(params.rows * params.cols, stream),
      vec1(params.bcast_along_rows ? params.cols : params.rows, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    IdxType N = params.rows, D = params.cols;
    IdxType len    = N * D;
    IdxType vecLen = params.bcast_along_rows ? D : N;
    uniform(handle, r, in.data(), len, (T)-1.0, (T)1.0);
    uniform(handle, r, vec1.data(), vecLen, (T)-1.0, (T)1.0);
    raft::copy(out_ref.data(), in.data(), len, resource::get_cuda_stream(handle));
    raft::copy(out.data(), in.data(), len, resource::get_cuda_stream(handle));
    naive_matrix_vector_op_launch(handle,
                                  out_ref.data(),
                                  vec1.data(),
                                  D,
                                  N,
                                  params.row_major,
                                  params.bcast_along_rows,
                                  params.operation_type);
    matrix_vector_op_launch(handle,
                            out.data(),
                            vec1.data(),
                            D,
                            N,
                            params.row_major,
                            params.bcast_along_rows,
                            params.operation_type);
    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MatrixVectorInputs<T, IdxType> params;
  rmm::device_uvector<T> in, out, out_ref, vec1;
};

const std::vector<MatrixVectorInputs<float, int>> inputsf_i32 = {
  {0.00001f, 1024, 32, 0, true, true, 1234ULL},   {0.00001f, 1024, 64, 1, true, true, 1234ULL},
  {0.00001f, 1024, 32, 2, true, false, 1234ULL},  {0.00001f, 1024, 64, 3, true, false, 1234ULL},
  {0.00001f, 1024, 32, 4, false, true, 1234ULL},  {0.00001f, 1024, 64, 0, false, true, 1234ULL},
  {0.00001f, 1024, 32, 1, false, false, 1234ULL}, {0.00001f, 1024, 64, 2, false, false, 1234ULL},

  {0.00001f, 1024, 32, 3, true, true, 1234ULL},   {0.00001f, 1024, 64, 4, true, true, 1234ULL},
  {0.00001f, 1024, 32, 0, true, false, 1234ULL},  {0.00001f, 1024, 64, 1, true, false, 1234ULL},
  {0.00001f, 1024, 32, 2, false, true, 1234ULL},  {0.00001f, 1024, 64, 3, false, true, 1234ULL},
  {0.00001f, 1024, 32, 4, false, false, 1234ULL}, {0.00001f, 1024, 64, 0, false, false, 1234ULL},

  {0.00001f, 1024, 32, 5, true, true, 1234ULL},   {0.00001f, 1024, 64, 5, true, false, 1234ULL},
  {0.00001f, 1024, 32, 5, false, true, 1234ULL},  {0.00001f, 1024, 64, 5, false, false, 1234ULL}};
typedef MatrixVectorTest<float, int> MatrixVectorTestF_i32;
TEST_P(MatrixVectorTestF_i32, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.rows * params.cols, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MatrixVectorTests,
                         MatrixVectorTestF_i32,
                         ::testing::ValuesIn(inputsf_i32));

const std::vector<MatrixVectorInputs<float, size_t>> inputsf_i64 = {
  {0.00001f, 2500, 250, 0, false, false, 1234ULL},
  {0.00001f, 2500, 250, 1, false, false, 1234ULL},
  {0.00001f, 2500, 250, 5, false, false, 1234ULL}};
typedef MatrixVectorTest<float, size_t> MatrixVectorTestF_i64;
TEST_P(MatrixVectorTestF_i64, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.rows * params.cols, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MatrixVectorTests,
                         MatrixVectorTestF_i64,
                         ::testing::ValuesIn(inputsf_i64));

const std::vector<MatrixVectorInputs<double, int>> inputsd_i32 = {
  {0.0000001, 1024, 32, 0, true, true, 1234ULL},   {0.0000001, 1024, 64, 1, true, true, 1234ULL},
  {0.0000001, 1024, 32, 2, true, false, 1234ULL},  {0.0000001, 1024, 64, 3, true, false, 1234ULL},
  {0.0000001, 1024, 32, 4, false, true, 1234ULL},  {0.0000001, 1024, 64, 0, false, true, 1234ULL},
  {0.0000001, 1024, 32, 1, false, false, 1234ULL}, {0.0000001, 1024, 64, 2, false, false, 1234ULL},

  {0.0000001, 1024, 32, 3, true, true, 1234ULL},   {0.0000001, 1024, 64, 4, true, true, 1234ULL},
  {0.0000001, 1024, 32, 0, true, false, 1234ULL},  {0.0000001, 1024, 64, 1, true, false, 1234ULL},
  {0.0000001, 1024, 32, 2, false, true, 1234ULL},  {0.0000001, 1024, 64, 3, false, true, 1234ULL},
  {0.0000001, 1024, 32, 4, false, false, 1234ULL}, {0.0000001, 1024, 64, 0, false, false, 1234ULL},

  {0.0000001, 1024, 32, 5, true, true, 1234ULL},   {0.0000001, 1024, 64, 5, true, false, 1234ULL},
  {0.0000001, 1024, 32, 5, false, true, 1234ULL},  {0.0000001, 1024, 64, 5, false, false, 1234ULL}};
typedef MatrixVectorTest<double, int> MatrixVectorTestD_i32;
TEST_P(MatrixVectorTestD_i32, Result)
{
  ASSERT_TRUE(devArrMatch(out_ref.data(),
                          out.data(),
                          params.rows * params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MatrixVectorTests,
                         MatrixVectorTestD_i32,
                         ::testing::ValuesIn(inputsd_i32));

const std::vector<MatrixVectorInputs<double, size_t>> inputsd_i64 = {
  {0.0000001, 2500, 250, 0, false, false, 1234ULL},
  {0.0000001, 2500, 250, 1, false, false, 1234ULL},
  {0.0000001, 2500, 250, 5, false, false, 1234ULL}};
typedef MatrixVectorTest<double, size_t> MatrixVectorTestD_i64;
TEST_P(MatrixVectorTestD_i64, Result)
{
  ASSERT_TRUE(devArrMatch(out_ref.data(),
                          out.data(),
                          params.rows * params.cols,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MatrixVectorTests,
                         MatrixVectorTestD_i64,
                         ::testing::ValuesIn(inputsd_i64));

}  // end namespace linalg
}  // end namespace raft
