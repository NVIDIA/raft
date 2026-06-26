/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda_fp16.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace raft {
namespace linalg {

template <typename T>
void initialize_array(T* data_h, size_t size)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0.0, 1.0);

  for (size_t i = 0; i < size; ++i) {
    if constexpr (std::is_same_v<T, half>) {
      data_h[i] = __float2half(static_cast<float>(dis(gen)));
    } else {
      data_h[i] = static_cast<T>(dis(gen));
    }
  }
}

template <typename T>
void cpu_transpose_row_major(
  const T* input, T* output, int rows, int cols, int stride_in = -1, int stride_out = -1)
{
  stride_in  = stride_in == -1 ? cols : stride_in;
  stride_out = stride_out == -1 ? rows : stride_out;
  if (stride_in)
    for (int i = 0; i < rows; ++i) {
      for (int j = 0; j < cols; ++j) {
        output[j * stride_out + i] = input[i * stride_in + j];
      }
    }
}

template <typename T>
void cpu_transpose_col_major(
  const T* input, T* output, int rows, int cols, int stride_in = -1, int stride_out = -1)
{
  cpu_transpose_row_major(input, output, cols, rows, stride_in, stride_out);
}

bool validate_half(const half* h_ref, const half* h_result, half tolerance, int len)
{
  bool success = true;
  for (int i = 0; i < len; ++i) {
    if (raft::abs(__half2float(h_result[i]) - __half2float(h_ref[i])) >= __half2float(tolerance)) {
      success = false;
      break;
    }
    if (!success) break;
  }
  return success;
}

namespace transpose_regular_test {

template <typename T>
struct TransposeInputs {
  T tolerance;
  int n_row;
  int n_col;
  unsigned long long int seed;
};

template <typename T>
class TransposeTest : public ::testing::TestWithParam<TransposeInputs<T>> {
 public:
  TransposeTest()
    : params(::testing::TestWithParam<TransposeInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.n_row * params.n_col, stream),
      data_trans_ref(params.n_row * params.n_col, stream),
      data_trans(params.n_row * params.n_col, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len = params.n_row * params.n_col;
    std::vector<T> data_h(len);
    std::vector<T> data_ref_h(len);

    initialize_array(data_h.data(), len);

    cpu_transpose_col_major(data_h.data(), data_ref_h.data(), params.n_row, params.n_col);

    raft::update_device(data.data(), data_h.data(), len, stream);
    raft::update_device(data_trans_ref.data(), data_ref_h.data(), len, stream);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        transpose(h,
                  data.data(),
                  data_trans.data(),
                  params.n_row,
                  params.n_col,
                  resource::get_cuda_stream(h));
      },
      raft::alloc_behavior::NO_ALLOCATIONS);
    if (params.n_row == params.n_col) { transpose(data.data(), params.n_col, stream); }
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  TransposeInputs<T> params;
  rmm::device_uvector<T> data, data_trans, data_trans_ref;
};

const std::vector<TransposeInputs<float>> inputsf2 = {{0.1f, 3, 3, 1234ULL},
                                                      {0.1f, 3, 4, 1234ULL},
                                                      {0.1f, 300, 300, 1234ULL},
                                                      {0.1f, 300, 4100, 1234ULL},
                                                      {0.1f, 1, 13000, 1234ULL},
                                                      {0.1f, 3, 130001, 1234ULL}};

const std::vector<TransposeInputs<double>> inputsd2 = {{0.1f, 3, 3, 1234ULL},
                                                       {0.1f, 3, 4, 1234ULL},
                                                       {0.1f, 300, 300, 1234ULL},
                                                       {0.1f, 300, 4100, 1234ULL},
                                                       {0.1f, 1, 13000, 1234ULL},
                                                       {0.1f, 3, 130001, 1234ULL}};

const std::vector<TransposeInputs<half>> inputsh2 = {{0.1f, 3, 3, 1234ULL},
                                                     {0.1f, 3, 4, 1234ULL},
                                                     {0.1f, 300, 300, 1234ULL},
                                                     {0.1f, 300, 4100, 1234ULL},
                                                     {0.1f, 1, 13000, 1234ULL},
                                                     {0.1f, 3, 130001, 1234ULL}};

typedef TransposeTest<float> TransposeTestValF;
TEST_P(TransposeTestValF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data_trans.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));

  if (params.n_row == params.n_col) {
    ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                  data.data(),
                                  params.n_row * params.n_col,
                                  raft::CompareApproxAbs<float>(params.tolerance)));
  }
}

typedef TransposeTest<double> TransposeTestValD;
TEST_P(TransposeTestValD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data_trans.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
  if (params.n_row == params.n_col) {
    ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                  data.data(),
                                  params.n_row * params.n_col,
                                  raft::CompareApproxAbs<double>(params.tolerance)));
  }
}

typedef TransposeTest<half> TransposeTestValH;
TEST_P(TransposeTestValH, Result)
{
  auto len = params.n_row * params.n_col;

  std::vector<half> data_trans_ref_h(len);
  std::vector<half> data_trans_h(len);
  std::vector<half> data_h(len);

  raft::copy(
    data_trans_ref_h.data(), data_trans_ref.data(), len, resource::get_cuda_stream(handle));
  raft::copy(data_trans_h.data(), data_trans.data(), len, resource::get_cuda_stream(handle));
  raft::copy(data_h.data(), data.data(), len, resource::get_cuda_stream(handle));
  resource::sync_stream(handle, stream);

  ASSERT_TRUE(validate_half(
    data_trans_ref_h.data(), data_trans_h.data(), params.tolerance, params.n_row * params.n_col));

  if (params.n_row == params.n_col) {
    ASSERT_TRUE(validate_half(
      data_trans_ref_h.data(), data_h.data(), params.tolerance, params.n_row * params.n_col));
  }
}

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValF, ::testing::ValuesIn(inputsf2));
INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValD, ::testing::ValuesIn(inputsd2));
INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValH, ::testing::ValuesIn(inputsh2));
}  // namespace transpose_regular_test

namespace transpose_extra_test {

/**
 * We hide these functions in tests for now until we have a heterogeneous mdarray
 * implementation.
 */

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix elements.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in] handle raft handle for managing expensive cuda resources.
 * @param[in] in     Input matrix.
 *
 * @return The transposed matrix.
 */
template <typename T, typename IndexType, typename LayoutPolicy>
[[nodiscard]] auto transpose(raft::resources const& handle,
                             device_matrix_view<T, IndexType, LayoutPolicy> in)
  -> std::enable_if_t<(raft::is_floating_point_v<T>) &&
                        (std::is_same_v<LayoutPolicy, layout_c_contiguous> ||
                         std::is_same_v<LayoutPolicy, layout_f_contiguous>),
                      device_matrix<T, IndexType, LayoutPolicy>>
{
  auto out = make_device_matrix<T, IndexType, LayoutPolicy>(handle, in.extent(1), in.extent(0));
  ::raft::linalg::transpose(handle, in, out.view());
  return out;
}

/**
 * @brief Transpose a matrix. The output has same layout policy as the input.
 *
 * @tparam T Data type of input matrix elements.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 *
 * @param[in] handle raft handle for managing expensive cuda resources.
 * @param[in] in     Input matrix.
 *
 * @return The transposed matrix.
 */
template <typename T, typename IndexType>
[[nodiscard]] auto transpose(raft::resources const& handle,
                             device_matrix_view<T, IndexType, layout_stride> in)
  -> std::enable_if_t<raft::is_floating_point_v<T>, device_matrix<T, IndexType, layout_stride>>
{
  matrix_extent<size_t> exts{in.extent(1), in.extent(0)};
  using policy_type =
    typename raft::device_matrix<T, IndexType, layout_stride>::container_policy_type;
  policy_type policy{};

  RAFT_EXPECTS(in.stride(0) == 1 || in.stride(1) == 1, "Unsupported matrix layout.");
  if (in.stride(1) == 1) {
    // row-major submatrix
    cuda::std::array<size_t, 2> strides{in.extent(0), 1};
    auto layout = layout_stride::mapping<matrix_extent<size_t>>{exts, strides};
    raft::device_matrix<T, IndexType, layout_stride> out{handle, layout, policy};
    ::raft::linalg::transpose(handle, in, out.view());
    return out;
  } else {
    // col-major submatrix
    cuda::std::array<size_t, 2> strides{1, in.extent(1)};
    auto layout = layout_stride::mapping<matrix_extent<size_t>>{exts, strides};
    raft::device_matrix<T, IndexType, layout_stride> out{handle, layout, policy};
    ::raft::linalg::transpose(handle, in, out.view());
    return out;
  }
}

template <typename T>
struct TransposeMdspanInputs {
  int n_row;
  int n_col;
  T tolerance = T{0.01};
};

template <typename T, typename LayoutPolicy>
void test_transpose_with_mdspan(const TransposeMdspanInputs<T>& param)
{
  auto len = param.n_row * param.n_col;
  std::vector<T> in_h(len);
  std::vector<T> out_ref_h(len);

  initialize_array(in_h.data(), len);

  raft::resources handle;
  auto stream  = resource::get_cuda_stream(handle);
  auto in      = make_device_matrix<T, size_t, LayoutPolicy>(handle, param.n_row, param.n_col);
  auto out_ref = make_device_matrix<T, size_t, LayoutPolicy>(handle, param.n_row, param.n_col);
  resource::sync_stream(handle, stream);
  if constexpr (std::is_same_v<LayoutPolicy, layout_c_contiguous>) {
    cpu_transpose_row_major(in_h.data(), out_ref_h.data(), param.n_row, param.n_col);
  } else {
    cpu_transpose_col_major(in_h.data(), out_ref_h.data(), param.n_row, param.n_col);
  }
  raft::copy(in.data_handle(), in_h.data(), len, resource::get_cuda_stream(handle));
  raft::copy(out_ref.data_handle(), out_ref_h.data(), len, resource::get_cuda_stream(handle));

  auto out = transpose(handle, in.view());
  static_assert(std::is_same_v<LayoutPolicy, typename decltype(out)::layout_type>);
  ASSERT_EQ(out.extent(0), in.extent(1));
  ASSERT_EQ(out.extent(1), in.extent(0));
  if constexpr (std::is_same_v<T, half>) {
    std::vector<half> out_h(len);
    raft::copy(out_h.data(), out.data_handle(), len, resource::get_cuda_stream(handle));
    ASSERT_TRUE(validate_half(out_ref_h.data(), out_h.data(), param.tolerance, len));
  } else {
    ASSERT_TRUE(raft::devArrMatch(
      out_ref.data_handle(), out.data_handle(), len, raft::CompareApproxAbs<T>(param.tolerance)));
  }
}

const std::vector<TransposeMdspanInputs<float>> inputs_mdspan_f  = {{3, 3},
                                                                    {3, 4},
                                                                    {300, 300},
                                                                    {300, 4100},
                                                                    {1, 13000},
                                                                    {3, 130001},
                                                                    {4100, 300},
                                                                    {13000, 1},
                                                                    {130001, 3}};
const std::vector<TransposeMdspanInputs<double>> inputs_mdspan_d = {{3, 3},
                                                                    {3, 4},
                                                                    {300, 300},
                                                                    {300, 4100},
                                                                    {1, 13000},
                                                                    {3, 130001},
                                                                    {4100, 300},
                                                                    {13000, 1},
                                                                    {130001, 3}};
const std::vector<TransposeMdspanInputs<half>> inputs_mdspan_h   = {{3, 3},
                                                                    {3, 4},
                                                                    {300, 300},
                                                                    {300, 4100},
                                                                    {1, 13000},
                                                                    {3, 130001},
                                                                    {4100, 300},
                                                                    {13000, 1},
                                                                    {130001, 3}};

TEST(TransposeTest, MDSpanFloat)
{
  for (const auto& p : inputs_mdspan_f) {
    test_transpose_with_mdspan<float, layout_c_contiguous>(p);
    test_transpose_with_mdspan<float, layout_f_contiguous>(p);
  }
}
TEST(TransposeTest, MDSpanDouble)
{
  for (const auto& p : inputs_mdspan_d) {
    test_transpose_with_mdspan<double, layout_c_contiguous>(p);
    test_transpose_with_mdspan<double, layout_f_contiguous>(p);
  }
}
TEST(TransposeTest, MDSpanHalf)
{
  for (const auto& p : inputs_mdspan_h) {
    test_transpose_with_mdspan<half, layout_c_contiguous>(p);
    test_transpose_with_mdspan<half, layout_f_contiguous>(p);
  }
}

template <typename T>
struct TransposeSubmatrixInputs {
  int n_row;
  int n_col;
  int row_beg;
  int row_end;
  int col_beg;
  int col_end;
  T tolerance = T{0.01};
};

template <typename T, typename LayoutPolicy>
void test_transpose_submatrix(const TransposeSubmatrixInputs<T>& param)
{
  auto len     = param.n_row * param.n_col;
  auto sub_len = (param.row_end - param.row_beg) * (param.col_end - param.col_beg);

  std::vector<T> in_h(len);
  std::vector<T> out_ref_h(sub_len);

  initialize_array(in_h.data(), len);

  raft::resources handle;
  auto stream = resource::get_cuda_stream(handle);

  auto in      = make_device_matrix<T, size_t, LayoutPolicy>(handle, param.n_row, param.n_col);
  auto out_ref = make_device_matrix<T, size_t, LayoutPolicy>(
    handle, (param.row_end - param.row_beg), (param.col_end - param.col_beg));

  if constexpr (std::is_same_v<LayoutPolicy, layout_c_contiguous>) {
    auto offset = param.row_beg * param.n_col + param.col_beg;
    cpu_transpose_row_major(in_h.data() + offset,
                            out_ref_h.data(),
                            (param.row_end - param.row_beg),
                            (param.col_end - param.col_beg),
                            in.extent(1),
                            (param.row_end - param.row_beg));
  } else {
    auto offset = param.col_beg * param.n_row + param.row_beg;
    cpu_transpose_col_major(in_h.data() + offset,
                            out_ref_h.data(),
                            (param.row_end - param.row_beg),
                            (param.col_end - param.col_beg),
                            in.extent(0),
                            (param.col_end - param.col_beg));
  }

  raft::copy(in.data_handle(), in_h.data(), len, resource::get_cuda_stream(handle));
  raft::copy(out_ref.data_handle(), out_ref_h.data(), sub_len, resource::get_cuda_stream(handle));
  resource::sync_stream(handle, stream);

  auto in_submat = cuda::std::submdspan(in.view(),
                                        cuda::std::tuple{param.row_beg, param.row_end},
                                        cuda::std::tuple{param.col_beg, param.col_end});

  static_assert(std::is_same_v<typename decltype(in_submat)::layout_type, layout_stride>);
  auto out = transpose(handle, in_submat);

  ASSERT_EQ(out.extent(0), in_submat.extent(1));
  ASSERT_EQ(out.extent(1), in_submat.extent(0));

  if constexpr (std::is_same_v<T, half>) {
    std::vector<half> out_h(sub_len);

    raft::copy(out_h.data(), out.data_handle(), sub_len, resource::get_cuda_stream(handle));
    ASSERT_TRUE(validate_half(out_ref_h.data(), out_h.data(), param.tolerance, sub_len));
  } else {
    ASSERT_TRUE(raft::devArrMatch(out_ref.data_handle(),
                                  out.data_handle(),
                                  sub_len,
                                  raft::CompareApproxAbs<T>(param.tolerance)));
  }
}
const std::vector<TransposeSubmatrixInputs<float>> inputs_submatrix_f = {
  {3, 3, 1, 2, 0, 2},
  {3, 4, 1, 3, 2, 3},
  {300, 300, 1, 299, 2, 239},
  {300, 4100, 3, 299, 101, 4001},
  {2, 13000, 0, 1, 3, 13000},
  {3, 130001, 0, 3, 3999, 129999},
  {4100, 300, 159, 4001, 125, 300},
  {13000, 5, 0, 11111, 0, 3},
  {130001, 3, 19, 130000, 2, 3}};
const std::vector<TransposeSubmatrixInputs<double>> inputs_submatrix_d = {
  {3, 3, 1, 2, 0, 2},
  {3, 4, 1, 3, 2, 3},
  {300, 300, 1, 299, 2, 239},
  {300, 4100, 3, 299, 101, 4001},
  {2, 13000, 0, 1, 3, 13000},
  {3, 130001, 0, 3, 3999, 129999},
  {4100, 300, 159, 4001, 125, 300},
  {13000, 5, 0, 11111, 0, 3},
  {130001, 3, 19, 130000, 2, 3}};
const std::vector<TransposeSubmatrixInputs<half>> inputs_submatrix_h = {
  {3, 3, 1, 2, 0, 2},
  {3, 4, 1, 3, 2, 3},
  {300, 300, 1, 299, 2, 239},
  {300, 4100, 3, 299, 101, 4001},
  {2, 13000, 0, 1, 3, 13000},
  {3, 130001, 0, 3, 3999, 129999},
  {4100, 300, 159, 4001, 125, 300},
  {13000, 5, 0, 11111, 0, 3},
  {130001, 3, 19, 130000, 2, 3}};

TEST(TransposeTest, SubMatrixFloat)
{
  for (const auto& p : inputs_submatrix_f) {
    test_transpose_submatrix<float, layout_c_contiguous>(p);
    test_transpose_submatrix<float, layout_f_contiguous>(p);
  }
}
TEST(TransposeTest, SubMatrixDouble)
{
  for (const auto& p : inputs_submatrix_d) {
    test_transpose_submatrix<double, layout_c_contiguous>(p);
    test_transpose_submatrix<double, layout_f_contiguous>(p);
  }
}
TEST(TransposeTest, SubMatrixHalf)
{
  for (const auto& p : inputs_submatrix_h) {
    test_transpose_submatrix<half, layout_c_contiguous>(p);
    test_transpose_submatrix<half, layout_f_contiguous>(p);
  }
}

}  // namespace transpose_extra_test
}  // end namespace linalg
}  // end namespace raft
