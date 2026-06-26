/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "reduce.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename IdxType>
struct ReduceInputs {
  OutType tolerance;
  IdxType rows, cols;
  bool rowMajor, alongRows;
  OutType init;
  unsigned long long int seed;
};

template <typename InType, typename OutType, typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const ReduceInputs<InType, OutType, IdxType>& dims)
{
  os << "{ " << dims.tolerance << ", " << dims.rows << ", " << dims.cols << ", " << dims.rowMajor
     << ", " << dims.alongRows << ", " << dims.init << " " << dims.seed << '}';
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
void reduceLaunch(const raft::resources& handle,
                  OutType* dots,
                  const InType* data,
                  IdxType cols,
                  IdxType rows,
                  bool rowMajor,
                  bool alongRows,
                  OutType init,
                  bool inplace,
                  MainLambda main_op,
                  ReduceLambda reduce_op,
                  FinalLambda final_op)
{
  IdxType output_size = alongRows ? rows : cols;

  auto output_view          = raft::make_device_vector_view(dots, output_size);
  auto input_view_row_major = raft::make_device_matrix_view(data, rows, cols);
  auto input_view_col_major =
    raft::make_device_matrix_view<const InType, IdxType, raft::col_major>(data, rows, cols);

  if (rowMajor and alongRows) {
    reduce<Apply::ALONG_ROWS>(
      handle, input_view_row_major, output_view, init, inplace, main_op, reduce_op, final_op);
  } else if (rowMajor and !alongRows) {
    reduce<Apply::ALONG_COLUMNS>(
      handle, input_view_row_major, output_view, init, inplace, main_op, reduce_op, final_op);
  } else if (!rowMajor and alongRows) {
    reduce<Apply::ALONG_ROWS>(
      handle, input_view_col_major, output_view, init, inplace, main_op, reduce_op, final_op);
  } else if (!rowMajor and !alongRows) {
    reduce<Apply::ALONG_COLUMNS>(
      handle, input_view_col_major, output_view, init, inplace, main_op, reduce_op, final_op);
  } else {
    RAFT_FAIL("Invalid combination of rowMajor and alongRows");
  }
}

template <typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda   = raft::sq_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::sqrt_op>
class ReduceTest : public ::testing::TestWithParam<ReduceInputs<InType, OutType, IdxType>> {
 public:
  ReduceTest()
    : params(::testing::TestWithParam<ReduceInputs<InType, OutType, IdxType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      dots_exp(params.alongRows ? params.rows : params.cols, stream),
      dots_act(params.alongRows ? params.rows : params.cols, stream)
  {
  }

 protected:
  void simpleTest()
  {
    IdxType rows           = 3;
    IdxType cols           = 4;
    auto data_h            = std::vector<InType>{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto expected_output_h = std::vector<OutType>{0.0f, 0.0f, 0.0f, 0.0f};
    auto output =
      raft::make_device_vector<OutType, IdxType>(handle, params.alongRows ? rows : cols);

    // Row Major
    {
      auto data = raft::make_device_matrix<InType, IdxType, raft::row_major>(handle, rows, cols);
      raft::copy(data.data_handle(), data_h.data(), data.size(), stream);

      if (params.alongRows) {
        reduce<Apply::ALONG_ROWS>(handle,
                                  raft::make_const_mdspan(data.view()),
                                  output.view(),
                                  OutType(0.0f),
                                  false,
                                  raft::identity_op(),
                                  raft::max_op(),
                                  raft::identity_op());
        expected_output_h = std::vector<OutType>{4, 8, 12};
      } else {
        reduce<Apply::ALONG_COLUMNS>(handle,
                                     raft::make_const_mdspan(data.view()),
                                     output.view(),
                                     OutType(0.0f),
                                     false,
                                     raft::identity_op(),
                                     raft::max_op(),
                                     raft::identity_op());
        expected_output_h = std::vector<OutType>{9, 10, 11, 12};
      }
      ASSERT_TRUE(raft::devArrMatchHost(expected_output_h.data(),
                                        output.data_handle(),
                                        output.size(),
                                        raft::CompareApprox(0.0f),
                                        stream));
    }
    // Col Major
    {
      auto data = raft::make_device_matrix<InType, IdxType, raft::col_major>(handle, rows, cols);
      raft::copy(data.data_handle(), data_h.data(), data.size(), stream);
      if (params.alongRows) {
        reduce<Apply::ALONG_ROWS>(handle,
                                  raft::make_const_mdspan(data.view()),
                                  output.view(),
                                  OutType(0.0f),
                                  false,
                                  raft::identity_op(),
                                  raft::max_op(),
                                  raft::identity_op());
        expected_output_h = std::vector<OutType>{10, 11, 12};
      } else {
        reduce<Apply::ALONG_COLUMNS>(handle,
                                     raft::make_const_mdspan(data.view()),
                                     output.view(),
                                     OutType(0.0f),
                                     false,
                                     raft::identity_op(),
                                     raft::max_op(),
                                     raft::identity_op());
        expected_output_h = std::vector<OutType>{3, 6, 9, 12};
      }
      ASSERT_TRUE(raft::devArrMatchHost(expected_output_h.data(),
                                        output.data_handle(),
                                        output.size(),
                                        raft::CompareApprox(0.0f),
                                        stream));
    }
  }

  void fullTest()
  {
    raft::random::RngState r(params.seed);
    IdxType rows = params.rows, cols = params.cols;
    IdxType len = rows * cols;
    gen_uniform(handle, data.data(), r, len);

    MainLambda main_op;
    ReduceLambda reduce_op;
    FinalLambda fin_op;

    // For both the naive and the actual implementation, execute first with inplace=false then true

    naiveReduction(dots_exp.data(),
                   data.data(),
                   cols,
                   rows,
                   params.rowMajor,
                   params.alongRows,
                   stream,
                   params.init,
                   false,
                   main_op,
                   reduce_op,
                   fin_op);
    naiveReduction(dots_exp.data(),
                   data.data(),
                   cols,
                   rows,
                   params.rowMajor,
                   params.alongRows,
                   stream,
                   params.init,
                   true,
                   main_op,
                   reduce_op,
                   fin_op);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        reduceLaunch(h,
                     dots_act.data(),
                     data.data(),
                     cols,
                     rows,
                     params.rowMajor,
                     params.alongRows,
                     params.init,
                     false,
                     main_op,
                     reduce_op,
                     fin_op);
        reduceLaunch(h,
                     dots_act.data(),
                     data.data(),
                     cols,
                     rows,
                     params.rowMajor,
                     params.alongRows,
                     params.init,
                     true,
                     main_op,
                     reduce_op,
                     fin_op);
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN);

    resource::sync_stream(handle, stream);
  }
  void SetUp() override {}

 protected:
  raft::resources handle;
  cudaStream_t stream;

  ReduceInputs<InType, OutType, IdxType> params;
  rmm::device_uvector<InType> data;
  rmm::device_uvector<OutType> dots_exp, dots_act;
};

#define REDUCE_TEST(test_type, test_name, test_inputs)                                            \
  typedef RAFT_DEPAREN(test_type) test_name;                                                      \
  TEST_P(test_name, Result)                                                                       \
  {                                                                                               \
    fullTest();                                                                                   \
    ASSERT_TRUE(raft::devArrMatch(                                                                \
      dots_exp.data(), dots_act.data(), dots_exp.size(), raft::CompareApprox(params.tolerance))); \
  }                                                                                               \
  INSTANTIATE_TEST_CASE_P(ReduceTests, test_name, ::testing::ValuesIn(test_inputs))

#define SIMPLE_REDUCE_TEST(test_type, test_name, test_inputs) \
  typedef RAFT_DEPAREN(test_type) test_name;                  \
  TEST_P(test_name, Result) { simpleTest(); }                 \
  INSTANTIATE_TEST_CASE_P(SimpleReduceTests, test_name, ::testing::ValuesIn(test_inputs))

const std::vector<ReduceInputs<float, float, int>> inputs_simple_ff_i32 =
  raft::util::itertools::product<ReduceInputs<float, float, int>>(
    {0.000002f}, {3}, {4}, {true, false}, {true, false}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<double, double, int>> inputs_simple_dd_i32 =
  raft::util::itertools::product<ReduceInputs<double, double, int>>(
    {0.000002}, {3}, {4}, {true, false}, {true, false}, {0.0}, {1234ULL});
const std::vector<ReduceInputs<float, float, uint32_t>> inputs_simple_ff_u32 =
  raft::util::itertools::product<ReduceInputs<float, float, uint32_t>>(
    {0.000002f}, {3u}, {4u}, {true, false}, {true, false}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<double, double, uint64_t>> inputs_simple_dd_u64 =
  raft::util::itertools::product<ReduceInputs<double, double, uint64_t>>(
    {0.000002}, {3ULL}, {4ULL}, {true, false}, {true, false}, {0.0}, {1234ULL});
SIMPLE_REDUCE_TEST((ReduceTest<float, float, int>), SimpleReduceTestFFI32, inputs_simple_ff_i32);
SIMPLE_REDUCE_TEST((ReduceTest<double, double, int>), SimpleReduceTestDDI32, inputs_simple_dd_i32);
SIMPLE_REDUCE_TEST((ReduceTest<float, float, uint32_t>),
                   SimpleReduceTestFFU32,
                   inputs_simple_ff_u32);
SIMPLE_REDUCE_TEST((ReduceTest<double, double, uint64_t>),
                   SimpleReduceTestDDU64,
                   inputs_simple_dd_u64);

const std::vector<ReduceInputs<float, float, int>> inputsff_i32 =
  raft::util::itertools::product<ReduceInputs<float, float, int>>(
    {0.000002f}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<double, double, int>> inputsdd_i32 =
  raft::util::itertools::product<ReduceInputs<double, double, int>>(
    {0.000000001}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {0.0}, {1234ULL});
const std::vector<ReduceInputs<float, double, int>> inputsfd_i32 =
  raft::util::itertools::product<ReduceInputs<float, double, int>>(
    {0.000000001}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<float, float, uint32_t>> inputsff_u32 =
  raft::util::itertools::product<ReduceInputs<float, float, uint32_t>>({0.000002f},
                                                                       {11u, 1234u},
                                                                       {7u, 33u, 128u, 500u},
                                                                       {true, false},
                                                                       {true, false},
                                                                       {0.0f},
                                                                       {1234ULL});
const std::vector<ReduceInputs<float, float, int64_t>> inputsff_i64 =
  raft::util::itertools::product<ReduceInputs<float, float, int64_t>>(
    {0.000002f}, {11, 1234}, {7, 33, 128, 500}, {true, false}, {true, false}, {0.0f}, {1234ULL});

REDUCE_TEST((ReduceTest<float, float, int>), ReduceTestFFI32, inputsff_i32);
REDUCE_TEST((ReduceTest<double, double, int>), ReduceTestDDI32, inputsdd_i32);
REDUCE_TEST((ReduceTest<float, double, int>), ReduceTestFDI32, inputsfd_i32);
REDUCE_TEST((ReduceTest<float, float, uint32_t>), ReduceTestFFU32, inputsff_u32);
REDUCE_TEST((ReduceTest<float, float, int64_t>), ReduceTestFFI64, inputsff_i64);

// The following test cases are for "thick" coalesced reductions

const std::vector<ReduceInputs<float, float, int>> inputsff_thick_i32 =
  raft::util::itertools::product<ReduceInputs<float, float, int>>(
    {0.0001f}, {3, 9}, {17771, 33333, 200000}, {true}, {true}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<double, double, int>> inputsdd_thick_i32 =
  raft::util::itertools::product<ReduceInputs<double, double, int>>(
    {0.000001}, {3, 9}, {17771, 33333, 200000}, {true}, {true}, {0.0}, {1234ULL});
const std::vector<ReduceInputs<float, double, int>> inputsfd_thick_i32 =
  raft::util::itertools::product<ReduceInputs<float, double, int>>(
    {0.000001}, {3, 9}, {17771, 33333, 200000}, {true}, {true}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<float, float, uint32_t>> inputsff_thick_u32 =
  raft::util::itertools::product<ReduceInputs<float, float, uint32_t>>(
    {0.0001f}, {3u, 9u}, {17771u, 33333u, 200000u}, {true}, {true}, {0.0f}, {1234ULL});
const std::vector<ReduceInputs<float, float, int64_t>> inputsff_thick_i64 =
  raft::util::itertools::product<ReduceInputs<float, float, int64_t>>(
    {0.0001f}, {3, 9}, {17771, 33333, 200000}, {true}, {true}, {0.0f}, {1234ULL});

REDUCE_TEST((ReduceTest<float, float, int>), ReduceTestFFI32Thick, inputsff_thick_i32);
REDUCE_TEST((ReduceTest<double, double, int>), ReduceTestDDI32Thick, inputsdd_thick_i32);
REDUCE_TEST((ReduceTest<float, double, int>), ReduceTestFDI32Thick, inputsfd_thick_i32);
REDUCE_TEST((ReduceTest<float, float, uint32_t>), ReduceTestFFU32Thick, inputsff_thick_u32);
REDUCE_TEST((ReduceTest<float, float, int64_t>), ReduceTestFFI64Thick, inputsff_thick_i64);

// Test key-value-pair reductions. This is important because shuffle intrinsics can't be used
// directly with those types.

template <typename T, typename IdxT = int>
struct ValueToKVP {
  HDI raft::KeyValuePair<IdxT, T> operator()(T value, IdxT idx) { return {idx, value}; }
};

template <typename T1, typename T2>
struct ArgMaxOp {
  HDI raft::KeyValuePair<T1, T2> operator()(raft::KeyValuePair<T1, T2> a,
                                            raft::KeyValuePair<T1, T2> b)
  {
    return (a.value > b.value || (a.value == b.value && a.key <= b.key)) ? a : b;
  }
};

const std::vector<ReduceInputs<short, raft::KeyValuePair<int, short>, int>> inputs_kvpis_i32 =
  raft::util::itertools::product<ReduceInputs<short, raft::KeyValuePair<int, short>, int>>(
    {raft::KeyValuePair{0, short(0)}},
    {11, 1234},
    {7, 33, 128, 500},
    {true},
    {true},
    {raft::KeyValuePair{0, short(0)}},
    {1234ULL});
const std::vector<ReduceInputs<float, raft::KeyValuePair<int, float>, int>> inputs_kvpif_i32 =
  raft::util::itertools::product<ReduceInputs<float, raft::KeyValuePair<int, float>, int>>(
    {raft::KeyValuePair{0, 0.0001f}},
    {11, 1234},
    {7, 33, 128, 500},
    {true},
    {true},
    {raft::KeyValuePair{0, 0.0f}},
    {1234ULL});
const std::vector<ReduceInputs<double, raft::KeyValuePair<int, double>, int>> inputs_kvpid_i32 =
  raft::util::itertools::product<ReduceInputs<double, raft::KeyValuePair<int, double>, int>>(
    {raft::KeyValuePair{0, 0.000001}},
    {11, 1234},
    {7, 33, 128, 500},
    {true},
    {true},
    {raft::KeyValuePair{0, 0.0}},
    {1234ULL});

REDUCE_TEST((ReduceTest<short,
                        raft::KeyValuePair<int, short>,
                        int,
                        ValueToKVP<short, int>,
                        ArgMaxOp<int, short>,
                        raft::identity_op>),
            ReduceTestKVPISI32,
            inputs_kvpis_i32);
REDUCE_TEST((ReduceTest<float,
                        raft::KeyValuePair<int, float>,
                        int,
                        ValueToKVP<float, int>,
                        ArgMaxOp<int, float>,
                        raft::identity_op>),
            ReduceTestKVPIFI32,
            inputs_kvpif_i32);
REDUCE_TEST((ReduceTest<double,
                        raft::KeyValuePair<int, double>,
                        int,
                        ValueToKVP<double, int>,
                        ArgMaxOp<int, double>,
                        raft::identity_op>),
            ReduceTestKVPIDI32,
            inputs_kvpid_i32);

}  // end namespace linalg
}  // end namespace raft
