/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"
#include "../util/preprocess_utils.cu"

#include <raft/sparse/matrix/preprocessing.cuh>
#include <raft/sparse/op/reduce.cuh>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename T2, typename T1>
void get_clean_coo(raft::resources& handle,
                   raft::device_vector_view<T1, int64_t> rows,
                   raft::device_vector_view<T1, int64_t> columns,
                   raft::device_vector_view<T2, int64_t> values,
                   int nnz,
                   int num_rows,
                   int num_cols,
                   raft::sparse::COO<T2, T1, T1>& coo)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::sparse::op::coo_sort(int(rows.size()),
                             int(columns.size()),
                             int(values.size()),
                             rows.data_handle(),
                             columns.data_handle(),
                             values.data_handle(),
                             stream);

  raft::sparse::op::max_duplicates<T1, T2, T1>(handle,
                                               coo,
                                               rows.data_handle(),
                                               columns.data_handle(),
                                               values.data_handle(),
                                               nnz,
                                               num_rows,
                                               num_cols);
}
template <typename T2, typename T1>
raft::device_coo_matrix<T2, T1, T1, T1, raft::device_container_policy, raft::PRESERVING>
create_coo_matrix(raft::resources& handle,
                  raft::device_vector_view<T1> rows,
                  raft::device_vector_view<T1> columns,
                  raft::device_vector_view<T2> values,
                  int num_rows,
                  int num_cols)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto coo_struct_view = raft::make_device_coordinate_structure_view(
    rows.data_handle(), columns.data_handle(), num_rows, num_cols, int(rows.size()));
  auto c_matrix = raft::make_device_coo_matrix<T2, T1, T1, T1>(handle, coo_struct_view);
  raft::update_device<T2>(
    c_matrix.view().get_elements().data(), values.data_handle(), int(values.size()), stream);
  return c_matrix;
}

template <typename T2, typename T1>
raft::device_coo_matrix<T2, T1, T1, T1, raft::device_container_policy, raft::PRESERVING>
create_coo_matrix(raft::resources& handle, raft::sparse::COO<T2, T1, T1>& coo)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto coo_struct_view = raft::make_device_coordinate_structure_view(
    coo.rows(), coo.cols(), coo.n_rows, coo.n_cols, int(coo.nnz));
  auto c_matrix = raft::make_device_coo_matrix<T2, T1, T1, T1>(handle, coo_struct_view);
  raft::update_device<T2>(c_matrix.view().get_elements().data(), coo.vals(), coo.nnz, stream);
  return c_matrix;
}

template <typename Type_f, typename Index_>
struct SparsePreprocessInputs {
  int n_rows;
  int n_cols;
  int nnz_edges;
};

template <typename Type_f, typename Index_>
class SparsePreprocessCSR
  : public ::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>> {
 public:
  SparsePreprocessCSR()
    : params(::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle))
  {
  }

 protected:
  void SetUp() override {}

  void Run(bool bm25_on, bool coo_on)
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    int num_rows        = pow(2, params.n_rows);
    int num_cols        = pow(2, params.n_cols);
    int nnz             = params.nnz_edges;
    auto a_rows         = raft::make_device_vector<Index_, int64_t>(handle, nnz);
    auto a_columns      = raft::make_device_vector<Index_, int64_t>(handle, nnz);
    auto a_values       = raft::make_device_vector<Type_f, int64_t>(handle, nnz);

    int random_seed = rand();
    raft::util::create_dataset<Index_, Type_f>(handle,
                                               a_rows.view(),
                                               a_columns.view(),
                                               a_values.view(),
                                               5,
                                               params.n_rows,
                                               params.n_cols,
                                               random_seed);

    raft::sparse::COO<Type_f, Index_, Index_> coo_a(stream);
    get_clean_coo<Type_f, Index_>(
      handle, a_rows.view(), a_columns.view(), a_values.view(), nnz, num_rows, num_cols, coo_a);

    auto rows_csr = raft::make_device_vector<Index_, int64_t>(handle, num_rows + 1);
    raft::sparse::convert::sorted_coo_to_csr(
      coo_a.rows(), coo_a.nnz, rows_csr.data_handle(), rows_csr.size(), stream);
    auto csr_struct_view = raft::make_device_compressed_structure_view(
      rows_csr.data_handle(), coo_a.cols(), num_rows, num_cols, int(coo_a.nnz));
    auto csr_matrix =
      raft::make_device_csr_matrix<Type_f, Index_, Index_, Index_>(handle, csr_struct_view);
    raft::update_device<Type_f>(
      csr_matrix.view().get_elements().data(), coo_a.vals(), int(coo_a.nnz), stream);

    auto coo_a_matrix = create_coo_matrix<Type_f, Index_>(handle, coo_a);

    auto result = raft::make_device_vector<Type_f, int64_t>(handle, int(coo_a.nnz));

    if (bm25_on) {
      auto bm25_vals = raft::make_device_vector<Type_f, int64_t>(handle, int(coo_a.nnz));
      raft::util::calc_tfidf_bm25<Index_, Type_f>(handle, csr_matrix.view(), bm25_vals.view());
      if (coo_on) {
        raft::execute_with_dry_run_check(
          handle,
          [&](raft::resources const& h) {
            raft::sparse::matrix::encode_bm25<float, int>(h, coo_a_matrix, result.view());
          },
          raft::alloc_behavior::DATA_DRIVEN,
          sizeof(float) * coo_a.nnz);
      } else {
        raft::execute_with_dry_run_check(
          handle,
          [&](raft::resources const& h) {
            raft::sparse::matrix::encode_bm25<float, int>(h, csr_matrix, result.view());
          },
          raft::alloc_behavior::DATA_DRIVEN,
          sizeof(float) * coo_a.nnz);
      }
      ASSERT_TRUE(raft::devArrMatch<Type_f>(bm25_vals.data_handle(),
                                            result.data_handle(),
                                            result.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    } else {
      auto tfidf_vals = raft::make_device_vector<Type_f, int64_t>(handle, int(coo_a.nnz));
      raft::util::calc_tfidf_bm25<Index_, Type_f>(
        handle, csr_matrix.view(), tfidf_vals.view(), true);
      if (coo_on) {
        raft::execute_with_dry_run_check(
          handle,
          [&](raft::resources const& h) {
            raft::sparse::matrix::encode_tfidf<float, int>(h, coo_a_matrix, result.view());
          },
          raft::alloc_behavior::ARGUMENT_DRIVEN,
          1);
      } else {
        raft::execute_with_dry_run_check(
          handle,
          [&](raft::resources const& h) {
            raft::sparse::matrix::encode_tfidf<float, int>(h, csr_matrix, result.view());
          },
          raft::alloc_behavior::ARGUMENT_DRIVEN,
          1);
      }
      ASSERT_TRUE(raft::devArrMatch<Type_f>(tfidf_vals.data_handle(),
                                            result.data_handle(),
                                            result.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    }

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SparsePreprocessInputs<Type_f, Index_> params;
};

using SparsePreprocessFF = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessFF, Result) { Run(false, false); }

using SparsePreprocessTT = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessTT, Result) { Run(true, true); }

using SparsePreprocessFT = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessFT, Result) { Run(false, true); }

using SparsePreprocessTF = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessTF, Result) { Run(true, false); }

using SparsePreprocessBigFF = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBigFF, Result) { Run(false, false); }

using SparsePreprocessBigTT = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBigTT, Result) { Run(true, true); }

using SparsePreprocessBigFT = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBigFT, Result) { Run(false, true); }

using SparsePreprocessBigTF = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBigTF, Result) { Run(true, false); }

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs = {
  {
    6,  // n_rows_factor
    6,  // n_cols_factor
    25  // num nnz values
  },
};

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs_big = {
  {
    14,     // n_rows_factor
    14,     // n_cols_factor
    500000  // nnz_edges
  },
};

INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessFF,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessTT,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessFT,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessTF,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBigTT,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBigFF,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBigTF,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBigFT,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));

}  // namespace sparse
}  // namespace raft
