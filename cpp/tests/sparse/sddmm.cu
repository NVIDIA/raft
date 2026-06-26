/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/make_blobs.cuh>
#include <raft/sparse/linalg/sddmm.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <cusparse.h>
#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename ValueType, typename IndexType, typename OutputType = ValueType>
struct SDDMMInputs {
  OutputType tolerance;

  IndexType m;
  IndexType k;
  IndexType n;

  OutputType alpha;
  OutputType beta;

  bool transpose_a;
  bool transpose_b;

  ValueType sparsity;

  unsigned long long int seed;
};

template <typename ValueType>
struct sum_abs_op {
  __host__ __device__ ValueType operator()(const ValueType& x, const ValueType& y) const
  {
    return y >= ValueType(0.0) ? (x + y) : (x - y);
  }
};

struct float_to_half {
  __host__ __device__ __half operator()(const float x) const { return __float2half(x); }
};

template <typename ValueType, typename IndexType>
::std::ostream& operator<<(::std::ostream& os, const SDDMMInputs<ValueType, IndexType>& params)
{
  os << " m: " << params.m << "\tk: " << params.k << "\tn: " << params.n
     << "\talpha: " << params.alpha << "\tbeta: " << params.beta
     << "\tsparsity: " << params.sparsity;

  return os;
}

template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyA = raft::layout_c_contiguous,
          typename LayoutPolicyB = raft::layout_c_contiguous,
          typename OutputType    = ValueType>
class SDDMMTest : public ::testing::TestWithParam<SDDMMInputs<ValueType, IndexType, OutputType>> {
 public:
  SDDMMTest()
    : params(::testing::TestWithParam<SDDMMInputs<ValueType, IndexType, OutputType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      a_data_d(0, resource::get_cuda_stream(handle)),
      b_data_d(0, resource::get_cuda_stream(handle)),
      c_indptr_d(0, resource::get_cuda_stream(handle)),
      c_indices_d(0, resource::get_cuda_stream(handle)),
      c_data_d(0, resource::get_cuda_stream(handle)),
      c_expected_data_d(0, resource::get_cuda_stream(handle))
  {
  }

 protected:
  bool isCuSparseVersionGreaterThan_12_0_1()
  {
    int version;
    cusparseHandle_t handle;
    cusparseCreate(&handle);
    cusparseGetVersion(handle, &version);

    int major = version / 1000;
    int minor = (version % 1000) / 100;
    int patch = version % 100;

    cusparseDestroy(handle);

    return (major > 12) || (major == 12 && minor > 0) || (major == 12 && minor == 0 && patch >= 2);
  }

  IndexType create_sparse_matrix(IndexType m,
                                 IndexType n,
                                 OutputType sparsity,
                                 std::vector<bool>& matrix)
  {
    IndexType total_elements = static_cast<IndexType>(m * n);
    IndexType num_ones       = static_cast<IndexType>((total_elements * 1.0f) * sparsity);
    IndexType res            = num_ones;

    for (IndexType i = 0; i < total_elements; ++i) {
      matrix[i] = false;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, total_elements - 1);

    while (num_ones > 0) {
      size_t index = dis(gen);

      if (matrix[index] == false) {
        matrix[index] = true;
        num_ones--;
      }
    }
    return res;
  }

  void convert_to_csr(std::vector<bool>& matrix,
                      IndexType rows,
                      IndexType cols,
                      std::vector<OutputType>& values,
                      std::vector<IndexType>& indices,
                      std::vector<IndexType>& indptr)
  {
    IndexType offset_indptr = 0;
    IndexType offset_values = 0;
    indptr[offset_indptr++] = 0;

    for (IndexType i = 0; i < rows; ++i) {
      for (IndexType j = 0; j < cols; ++j) {
        if (matrix[i * cols + j]) {
          values[offset_values]  = static_cast<OutputType>(1.0);
          indices[offset_values] = static_cast<IndexType>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<IndexType>(offset_values);
    }
  }

  void cpu_sddmm(const std::vector<ValueType>& A,
                 const std::vector<ValueType>& B,
                 std::vector<OutputType>& vals,
                 const std::vector<IndexType>& cols,
                 const std::vector<IndexType>& row_ptrs,
                 bool is_row_major_A,
                 bool is_row_major_B)
  {
    if (params.m * params.k != static_cast<IndexType>(A.size()) ||
        params.k * params.n != static_cast<IndexType>(B.size())) {
      std::cerr << "Matrix dimensions and vector size do not match!" << std::endl;
      return;
    }

    bool trans_a = params.transpose_a ? !is_row_major_A : is_row_major_A;
    bool trans_b = params.transpose_b ? !is_row_major_B : is_row_major_B;

    for (IndexType i = 0; i < params.m; ++i) {
      for (IndexType j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
        OutputType sum = 0;
        for (IndexType l = 0; l < params.k; ++l) {
          IndexType a_index = trans_a ? i * params.k + l : l * params.m + i;
          IndexType b_index = trans_b ? l * params.n + cols[j] : cols[j] * params.k + l;
          if constexpr ((std::is_same_v<OutputType, float> && std::is_same_v<ValueType, half>)) {
            sum += __half2float(A[a_index]) * __half2float(B[b_index]);
          } else {
            sum += A[a_index] * B[b_index];
          }
        }
        vals[j] = params.alpha * sum + params.beta * vals[j];
      }
    }
  }

  void make_data()
  {
    IndexType a_size = params.m * params.k;
    IndexType b_size = params.k * params.n;
    IndexType c_size = params.m * params.n;

    std::vector<ValueType> a_data_h(a_size);
    std::vector<ValueType> b_data_h(b_size);

    a_data_d.resize(a_size, stream);
    b_data_d.resize(b_size, stream);

    auto blobs_a_b = raft::make_device_matrix<OutputType, IndexType>(handle, 1, a_size + b_size);
    auto labels    = raft::make_device_vector<IndexType, IndexType>(handle, 1);

    raft::random::make_blobs<OutputType, IndexType>(blobs_a_b.data_handle(),
                                                    labels.data_handle(),
                                                    1,
                                                    a_size + b_size,
                                                    1,
                                                    stream,
                                                    false,
                                                    nullptr,
                                                    nullptr,
                                                    OutputType(1.0),
                                                    false,
                                                    OutputType(-1.0f),
                                                    OutputType(1.0f),
                                                    uint64_t(2024));
    if constexpr ((std::is_same_v<OutputType, float> && std::is_same_v<ValueType, half>)) {
      {
        thrust::device_ptr<OutputType> d_output_ptr =
          thrust::device_pointer_cast(blobs_a_b.data_handle());
        thrust::device_ptr<ValueType> d_value_ptr = thrust::device_pointer_cast(a_data_d.data());
        thrust::transform(thrust::cuda::par.on(stream),
                          d_output_ptr,
                          d_output_ptr + a_size,
                          d_value_ptr,
                          float_to_half());
      }
      {
        thrust::device_ptr<OutputType> d_output_ptr =
          thrust::device_pointer_cast(blobs_a_b.data_handle() + a_size);
        thrust::device_ptr<ValueType> d_value_ptr = thrust::device_pointer_cast(b_data_d.data());
        thrust::transform(thrust::cuda::par.on(stream),
                          d_output_ptr,
                          d_output_ptr + b_size,
                          d_value_ptr,
                          float_to_half());
      }
      raft::copy(a_data_h.data(), a_data_d.data(), a_size, stream);
      raft::copy(b_data_h.data(), b_data_d.data(), b_size, stream);
    } else {
      raft::copy(a_data_h.data(), blobs_a_b.data_handle(), a_size, stream);
      raft::copy(b_data_h.data(), blobs_a_b.data_handle() + a_size, b_size, stream);

      raft::copy(a_data_d.data(), blobs_a_b.data_handle(), a_size, stream);
      raft::copy(b_data_d.data(), blobs_a_b.data_handle() + a_size, b_size, stream);
    }

    resource::sync_stream(handle);

    std::vector<bool> c_dense_data_h(c_size);
    IndexType c_true_nnz =
      create_sparse_matrix(params.m, params.n, params.sparsity, c_dense_data_h);

    std::vector<IndexType> c_indptr_h(params.m + 1);
    std::vector<IndexType> c_indices_h(c_true_nnz);
    std::vector<OutputType> c_data_h(c_true_nnz);

    convert_to_csr(c_dense_data_h, params.m, params.n, c_data_h, c_indices_h, c_indptr_h);

    bool is_row_major_A = (std::is_same_v<LayoutPolicyA, raft::row_major>);
    bool is_row_major_B = (std::is_same_v<LayoutPolicyB, raft::row_major>);

    c_data_d.resize(c_data_h.size(), stream);
    update_device(c_data_d.data(), c_data_h.data(), c_data_h.size(), stream);
    resource::sync_stream(handle);

    cpu_sddmm(
      a_data_h, b_data_h, c_data_h, c_indices_h, c_indptr_h, is_row_major_A, is_row_major_B);

    c_indptr_d.resize(c_indptr_h.size(), stream);
    c_indices_d.resize(c_indices_h.size(), stream);
    c_expected_data_d.resize(c_data_h.size(), stream);

    update_device(c_indptr_d.data(), c_indptr_h.data(), c_indptr_h.size(), stream);
    update_device(c_indices_d.data(), c_indices_h.data(), c_indices_h.size(), stream);
    update_device(c_expected_data_d.data(), c_data_h.data(), c_data_h.size(), stream);

    resource::sync_stream(handle);
  }

  void SetUp() override
  {
    if (std::is_same_v<ValueType, half> && !isCuSparseVersionGreaterThan_12_0_1()) {
      GTEST_SKIP() << "Skipping all tests for half-float as cuSparse doesn't support it.";
    }
    make_data();
  }

  void Run()
  {
    auto a = raft::make_device_matrix_view<const ValueType, IndexType, LayoutPolicyA>(
      a_data_d.data(),
      (!params.transpose_a ? params.m : params.k),
      (!params.transpose_a ? params.k : params.m));
    auto b = raft::make_device_matrix_view<const ValueType, IndexType, LayoutPolicyB>(
      b_data_d.data(),
      (!params.transpose_b ? params.k : params.n),
      (!params.transpose_b ? params.n : params.k));

    auto c_structure = raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(
      c_indptr_d.data(),
      c_indices_d.data(),
      params.m,
      params.n,
      static_cast<IndexType>(c_indices_d.size()));

    auto c = raft::make_device_csr_matrix_view<OutputType>(c_data_d.data(), c_structure);

    auto op_a = params.transpose_a ? raft::linalg::Operation::TRANSPOSE
                                   : raft::linalg::Operation::NON_TRANSPOSE;
    auto op_b = params.transpose_b ? raft::linalg::Operation::TRANSPOSE
                                   : raft::linalg::Operation::NON_TRANSPOSE;

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        raft::sparse::linalg::sddmm(h,
                                    a,
                                    b,
                                    c,
                                    op_a,
                                    op_b,
                                    raft::make_host_scalar_view<OutputType>(&params.alpha),
                                    raft::make_host_scalar_view<OutputType>(&params.beta));
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN,
      1);

    resource::sync_stream(handle);

    ASSERT_TRUE(raft::devArrMatch<OutputType>(c_expected_data_d.data(),
                                              c.get_elements().data(),
                                              c_expected_data_d.size(),
                                              raft::CompareApprox<OutputType>(params.tolerance),
                                              stream));

    thrust::device_ptr<OutputType> expected_data_ptr =
      thrust::device_pointer_cast(c_expected_data_d.data());
    OutputType sum_abs = thrust::reduce(thrust::cuda::par.on(stream),
                                        expected_data_ptr,
                                        expected_data_ptr + c_expected_data_d.size(),
                                        OutputType(0.0f),
                                        sum_abs_op<OutputType>());
    OutputType avg     = sum_abs / (1.0f * c_expected_data_d.size());

    ASSERT_GE(avg, (params.tolerance * static_cast<OutputType>(0.001f)));
  }

  raft::resources handle;
  cudaStream_t stream;
  SDDMMInputs<ValueType, IndexType, OutputType> params;

  rmm::device_uvector<ValueType> a_data_d;
  rmm::device_uvector<ValueType> b_data_d;

  rmm::device_uvector<IndexType> c_indptr_d;
  rmm::device_uvector<IndexType> c_indices_d;
  rmm::device_uvector<OutputType> c_data_d;

  rmm::device_uvector<OutputType> c_expected_data_d;
};

using SDDMMTestF_Row_Col = SDDMMTest<float, int, raft::row_major, raft::col_major>;
TEST_P(SDDMMTestF_Row_Col, Result) { Run(); }

using SDDMMTestF_Col_Row = SDDMMTest<float, int, raft::col_major, raft::row_major>;
TEST_P(SDDMMTestF_Col_Row, Result) { Run(); }

using SDDMMTestF_Row_Row = SDDMMTest<float, int, raft::row_major, raft::row_major>;
TEST_P(SDDMMTestF_Row_Row, Result) { Run(); }

using SDDMMTestF_Col_Col = SDDMMTest<float, int, raft::col_major, raft::col_major>;
TEST_P(SDDMMTestF_Col_Col, Result) { Run(); }

using SDDMMTestD_Row_Col = SDDMMTest<double, int, raft::row_major, raft::col_major>;
TEST_P(SDDMMTestD_Row_Col, Result) { Run(); }

using SDDMMTestD_Col_Row = SDDMMTest<double, int, raft::col_major, raft::row_major>;
TEST_P(SDDMMTestD_Col_Row, Result) { Run(); }

using SDDMMTestD_Row_Row = SDDMMTest<double, int, raft::row_major, raft::row_major>;
TEST_P(SDDMMTestD_Row_Row, Result) { Run(); }

using SDDMMTestD_Col_Col = SDDMMTest<double, int, raft::col_major, raft::col_major>;
TEST_P(SDDMMTestD_Col_Col, Result) { Run(); }

using SDDMMTestHF_Row_Col = SDDMMTest<half, int, raft::row_major, raft::col_major, float>;
TEST_P(SDDMMTestHF_Row_Col, Result) { Run(); }

using SDDMMTestHF_Col_Row = SDDMMTest<half, int, raft::col_major, raft::row_major, float>;
TEST_P(SDDMMTestHF_Col_Row, Result) { Run(); }

using SDDMMTestHF_Row_Row = SDDMMTest<half, int, raft::row_major, raft::row_major, float>;
TEST_P(SDDMMTestHF_Row_Row, Result) { Run(); }

using SDDMMTestHF_Col_Col = SDDMMTest<half, int, raft::col_major, raft::col_major, float>;
TEST_P(SDDMMTestHF_Col_Col, Result) { Run(); }

const std::vector<SDDMMInputs<float, int>> sddmm_inputs_f = {
  {0.0001f, 10, 5, 32, 1.0, 0.0, false, false, 0.01, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.3, 0.0, true, false, 0.1, 1234ULL},
  {0.0003f, 32, 1024, 1024, 1.0, 0.3, false, true, 0.2, 1234ULL},
  {0.001f, 1024, 1024, 1024, 0.2, 0.2, true, true, 0.19, 1234ULL},
  {0.0001f, 1024, 1024, 32, 0.1, 0.2, false, false, 0.3, 1234ULL},
  {0.0001f, 1024, 32, 1024, 1.0, 0.3, true, false, 0.4, 1234ULL},
  {0.0003f, 32, 1024, 1024, 2.0, 0.2, false, true, 0.19, 1234ULL},
  {0.001f, 1024, 1024, 1024, 0.0, 1.2, true, true, 0.1, 1234ULL}};

const std::vector<SDDMMInputs<double, int>> sddmm_inputs_d = {
  {0.0001f, 10, 5, 32, 1.0, 0.0, false, false, 0.01, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.3, 0.0, true, false, 0.1, 1234ULL},
  {0.0001f, 32, 1024, 1024, 1.0, 0.3, false, true, 0.2, 1234ULL},
  {0.0001f, 1024, 1024, 1024, 0.2, 0.2, true, true, 0.19, 1234ULL},
  {0.0001f, 1024, 1024, 32, 0.1, 0.2, false, false, 0.3, 1234ULL},
  {0.0001f, 1024, 32, 1024, 1.0, 0.3, true, false, 0.4, 1234ULL},
  {0.0001f, 32, 1024, 1024, 2.0, 0.2, false, true, 0.19, 1234ULL},
  {0.0001f, 1024, 1024, 1024, 0.0, 1.2, true, true, 0.1, 1234ULL}};

const std::vector<SDDMMInputs<half, int, float>> sddmm_inputs_h_f = {
  {0.0001f, 10, 5, 32, 1.0, 0.0, false, false, 0.01, 1234ULL},
  {0.0001f, 1024, 32, 1024, 0.3, 0.0, true, false, 0.1, 1234ULL},
  {0.0003f, 32, 1024, 1024, 1.0, 0.3, false, true, 0.2, 1234ULL},
  {0.001f, 1024, 1024, 1024, 0.2, 0.2, true, true, 0.19, 1234ULL},
  {0.0001f, 1024, 1024, 32, 0.1, 0.2, false, false, 0.3, 1234ULL},
  {0.0001f, 1024, 32, 1024, 1.0, 0.3, true, false, 0.4, 1234ULL},
  {0.0003f, 32, 1024, 1024, 2.0, 0.2, false, true, 0.19, 1234ULL},
  {0.001f, 1024, 1024, 1024, 0.0, 1.2, true, true, 0.1, 1234ULL}};

INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestF_Row_Col, ::testing::ValuesIn(sddmm_inputs_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestF_Col_Row, ::testing::ValuesIn(sddmm_inputs_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestF_Row_Row, ::testing::ValuesIn(sddmm_inputs_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestF_Col_Col, ::testing::ValuesIn(sddmm_inputs_f));

INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestD_Row_Col, ::testing::ValuesIn(sddmm_inputs_d));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestD_Col_Row, ::testing::ValuesIn(sddmm_inputs_d));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestD_Row_Row, ::testing::ValuesIn(sddmm_inputs_d));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestD_Col_Col, ::testing::ValuesIn(sddmm_inputs_d));

INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestHF_Row_Col, ::testing::ValuesIn(sddmm_inputs_h_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestHF_Col_Row, ::testing::ValuesIn(sddmm_inputs_h_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestHF_Row_Row, ::testing::ValuesIn(sddmm_inputs_h_f));
INSTANTIATE_TEST_CASE_P(SDDMMTest, SDDMMTestHF_Col_Col, ::testing::ValuesIn(sddmm_inputs_h_f));

}  // namespace sparse
}  // namespace raft
