/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace raft::sparse::linalg {

// Test case structure for Laplacian tests
struct LaplacianTestCase {
  std::string name;
  std::vector<float> data;
  std::vector<int> indices;
  std::vector<int> indptr;
  std::vector<float> expected_data;
  std::vector<int> expected_indices;
  std::vector<int> expected_indptr;
};

class ComputeGraphLaplacianTest : public ::testing::TestWithParam<LaplacianTestCase> {};

TEST_P(ComputeGraphLaplacianTest, ComputeLaplacian)
{
  const auto& test_case = GetParam();

  auto res              = raft::resources{};
  auto adjacency_matrix = make_device_csr_matrix<float>(
    res, int(test_case.indptr.size() - 1), int(test_case.indptr.size() - 1), test_case.data.size());
  auto adjacency_structure = adjacency_matrix.structure_view();

  raft::copy(adjacency_matrix.get_elements().data(),
             test_case.data.data(),
             test_case.data.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_structure.get_indices().data(),
             test_case.indices.data(),
             test_case.indices.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_structure.get_indptr().data(),
             test_case.indptr.data(),
             test_case.indptr.size(),
             raft::resource::get_cuda_stream(res));

  auto laplacian           = compute_graph_laplacian(res, adjacency_matrix.view());
  auto laplacian_structure = laplacian.structure_view();
  auto laplacian_data      = std::vector<float>(laplacian_structure.get_nnz());
  auto laplacian_indices   = std::vector<int>(laplacian_structure.get_nnz());
  auto laplacian_indptr    = std::vector<int>(laplacian_structure.get_n_rows() + 1);

  raft::copy(laplacian_data.data(),
             laplacian.get_elements().data(),
             laplacian_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(laplacian_indices.data(),
             laplacian_structure.get_indices().data(),
             laplacian_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(laplacian_indptr.data(),
             laplacian_structure.get_indptr().data(),
             laplacian_structure.get_n_rows() + 1,
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  EXPECT_EQ(test_case.expected_data, laplacian_data) << "Failed for test case: " << test_case.name;
  EXPECT_EQ(test_case.expected_indices, laplacian_indices)
    << "Failed for test case: " << test_case.name;
  EXPECT_EQ(test_case.expected_indptr, laplacian_indptr)
    << "Failed for test case: " << test_case.name;
}

// Define test cases
const std::vector<LaplacianTestCase> laplacian_test_cases = {
  {
    "GraphWithoutSelfLoop",
    // Adjacency matrix:
    // [[0 1 1 1]
    //  [1 0 0 1]
    //  [1 0 0 0]
    //  [1 1 0 0]]
    {1, 1, 1, 1, 1, 1, 1, 1},  // data
    {1, 2, 3, 0, 3, 0, 0, 1},  // indices
    {0, 3, 5, 6, 8},           // indptr
    // Expected Laplacian L = D - A:
    // [[ 3 -1 -1 -1]
    //  [-1  2  0 -1]
    //  [-1  0  1  0]
    //  [-1 -1  0  2]]
    {3, -1, -1, -1, -1, 2, -1, -1, 1, -1, -1, 2},  // expected_data
    {0, 1, 2, 3, 0, 1, 3, 0, 2, 0, 1, 3},          // expected_indices
    {0, 4, 7, 9, 12},                              // expected_indptr
  },
  {
    "GraphWithSelfLoop",
    // Adjacency matrix:
    // [[1 1 1 1]   (node 0 has a self-loop)
    //  [1 0 0 1]
    //  [1 0 0 0]
    //  [1 1 0 0]]
    {1, 1, 1, 1, 1, 1, 1, 1, 1},  // data
    {0, 1, 2, 3, 0, 3, 0, 0, 1},  // indices
    {0, 4, 6, 7, 9},              // indptr
    // Expected Laplacian L = D - A:
    // [[ 3 -1 -1 -1]   (4 - 1 = 3 on diagonal despite self-loop)
    //  [-1  2  0 -1]
    //  [-1  0  1  0]
    //  [-1 -1  0  2]]
    {3, -0, -1, -1, -1, -1, 2, -1, -1, 1, -1, -1, 2},  // expected_data
    {0, 0, 1, 2, 3, 0, 1, 3, 0, 2, 0, 1, 3},           // expected_indices
    {0, 5, 8, 10, 13},                                 // expected_indptr
  }};

INSTANTIATE_TEST_SUITE_P(LaplacianTests,
                         ComputeGraphLaplacianTest,
                         ::testing::ValuesIn(laplacian_test_cases),
                         [](const ::testing::TestParamInfo<LaplacianTestCase>& info) {
                           return info.param.name;
                         });

TEST(Raft, ComputeGraphLaplacianNormalizedCSR)
{
  // Using the same adjacency matrix as in the ComputeGraphLaplacian test:
  // [[0 1 1 1]
  //  [1 0 0 1]
  //  [1 0 0 0]
  //  [1 1 0 0]]

  auto data    = std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1};
  auto indices = std::vector<int>{1, 2, 3, 0, 3, 0, 0, 1};
  auto indptr  = std::vector<int>{0, 3, 5, 6, 8};

  auto res              = raft::resources{};
  auto adjacency_matrix = make_device_csr_matrix<float, int, int, int>(
    res, (indptr.size() - 1), (indptr.size() - 1), data.size());
  auto adjacency_structure = adjacency_matrix.structure_view();
  raft::copy(adjacency_matrix.get_elements().data(),
             &(data[0]),
             data.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_structure.get_indices().data(),
             &(indices[0]),
             indices.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_structure.get_indptr().data(),
             &(indptr[0]),
             indptr.size(),
             raft::resource::get_cuda_stream(res));

  // Create diagonal output vector
  auto diagonal_out = raft::make_device_vector<float, int>(res, adjacency_structure.get_n_rows());

  // Compute normalized Laplacian
  auto normalized_laplacian =
    laplacian_normalized(res, adjacency_matrix.view(), diagonal_out.view());
  auto normalized_laplacian_structure = normalized_laplacian.structure_view();

  // Copy results back to host
  auto normalized_laplacian_data    = std::vector<float>(normalized_laplacian_structure.get_nnz());
  auto normalized_laplacian_indices = std::vector<int>(normalized_laplacian_structure.get_nnz());
  auto normalized_laplacian_indptr =
    std::vector<int>(normalized_laplacian_structure.get_n_rows() + 1);
  auto diagonal_data = std::vector<float>(adjacency_structure.get_n_rows());

  raft::copy(&(normalized_laplacian_data[0]),
             normalized_laplacian.get_elements().data(),
             normalized_laplacian_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(normalized_laplacian_indices[0]),
             normalized_laplacian_structure.get_indices().data(),
             normalized_laplacian_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(normalized_laplacian_indptr[0]),
             normalized_laplacian_structure.get_indptr().data(),
             normalized_laplacian_structure.get_n_rows() + 1,
             raft::resource::get_cuda_stream(res));
  raft::copy(&(diagonal_data[0]),
             diagonal_out.data_handle(),
             diagonal_out.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // For the given adjacency matrix, the diagonal degree matrix D has values [3, 2, 1, 2]
  // The square root of these values is [√3, √2, 1, √2]
  // The normalized Laplacian should have values close to:
  // [1, -1/√(3*2), -1/√(3*1), -1/√(3*2),
  //  -1/√(2*3), 1, -1/√(2*2),
  //  -1/√(1*3), 1,
  //  -1/√(2*3), -1/√(2*2), 1]

  // Expected diagonal values (sqrt of diagonal degree matrix)
  auto expected_diagonal =
    std::vector<float>{std::sqrt(3.0f), std::sqrt(2.0f), 1.0f, std::sqrt(2.0f)};

  // Expected normalized Laplacian values
  auto expected_data = std::vector<float>{1.0f,
                                          -1.0f / std::sqrt(3.0f * 2.0f),
                                          -1.0f / std::sqrt(3.0f * 1.0f),
                                          -1.0f / std::sqrt(3.0f * 2.0f),
                                          -1.0f / std::sqrt(2.0f * 3.0f),
                                          1.0f,
                                          -1.0f / std::sqrt(2.0f * 2.0f),
                                          -1.0f / std::sqrt(1.0f * 3.0f),
                                          1.0f,
                                          -1.0f / std::sqrt(2.0f * 3.0f),
                                          -1.0f / std::sqrt(2.0f * 2.0f),
                                          1.0f};

  // Same indices and indptr as non-normalized Laplacian
  auto expected_indices = std::vector<int>{0, 1, 2, 3, 0, 1, 3, 0, 2, 0, 1, 3};
  auto expected_indptr  = std::vector<int>{0, 4, 7, 9, 12};

  // Compare results with expected values with a small tolerance for floating point differences
  const float tol = 1e-6f;
  ASSERT_EQ(expected_diagonal.size(), diagonal_data.size());
  for (size_t i = 0; i < expected_diagonal.size(); ++i) {
    EXPECT_NEAR(expected_diagonal[i], diagonal_data[i], tol);
  }

  ASSERT_EQ(expected_data.size(), normalized_laplacian_data.size());
  for (size_t i = 0; i < expected_data.size(); ++i) {
    EXPECT_NEAR(expected_data[i], normalized_laplacian_data[i], tol);
  }

  EXPECT_EQ(expected_indices, normalized_laplacian_indices);
  EXPECT_EQ(expected_indptr, normalized_laplacian_indptr);
}

TEST(Raft, ComputeGraphLaplacianNormalizedCOO)
{
  // Using the same adjacency matrix as in the ComputeGraphLaplacian test:
  // [[0 1 1 1]
  //  [1 0 0 1]
  //  [1 0 0 0]
  //  [1 1 0 0]]

  auto data    = std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1};
  auto indices = std::vector<int>{1, 2, 3, 0, 3, 0, 0, 1};
  auto indptr  = std::vector<int>{0, 3, 5, 6, 8};

  auto res = raft::resources{};

  // First create a CSR matrix
  auto adjacency_matrix_csr = make_device_csr_matrix<float, int, int, int>(
    res, (indptr.size() - 1), (indptr.size() - 1), data.size());
  auto adjacency_structure_csr = adjacency_matrix_csr.structure_view();
  raft::copy(adjacency_matrix_csr.get_elements().data(),
             &(data[0]),
             data.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_structure_csr.get_indices().data(),
             &(indices[0]),
             indices.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_structure_csr.get_indptr().data(),
             &(indptr[0]),
             indptr.size(),
             raft::resource::get_cuda_stream(res));

  // Convert CSR to COO
  auto adjacency_matrix_coo =
    make_device_coo_matrix<float, int, int, int>(res,
                                                 adjacency_structure_csr.get_n_rows(),
                                                 adjacency_structure_csr.get_n_cols(),
                                                 adjacency_structure_csr.get_nnz());

  raft::sparse::convert::csr_to_coo<int>(adjacency_structure_csr.get_indptr().data(),
                                         adjacency_structure_csr.get_n_rows(),
                                         adjacency_matrix_coo.structure_view().get_rows().data(),
                                         adjacency_structure_csr.get_nnz(),
                                         raft::resource::get_cuda_stream(res));

  raft::copy(adjacency_matrix_coo.structure_view().get_cols().data(),
             adjacency_structure_csr.get_indices().data(),
             adjacency_structure_csr.get_nnz(),
             raft::resource::get_cuda_stream(res));

  raft::copy(adjacency_matrix_coo.get_elements().data(),
             adjacency_matrix_csr.get_elements().data(),
             adjacency_structure_csr.get_nnz(),
             raft::resource::get_cuda_stream(res));

  // Create diagonal output vector
  auto diagonal_out =
    raft::make_device_vector<float, int>(res, adjacency_structure_csr.get_n_rows());

  // Compute normalized Laplacian using COO matrix (result is also COO)
  auto normalized_laplacian_coo =
    laplacian_normalized(res, adjacency_matrix_coo.view(), diagonal_out.view());
  auto normalized_laplacian_coo_structure = normalized_laplacian_coo.structure_view();

  // Sort the COO matrix first
  raft::sparse::op::coo_sort<float, int, int>(normalized_laplacian_coo_structure.get_n_rows(),
                                              normalized_laplacian_coo_structure.get_n_cols(),
                                              normalized_laplacian_coo_structure.get_nnz(),
                                              normalized_laplacian_coo_structure.get_rows().data(),
                                              normalized_laplacian_coo_structure.get_cols().data(),
                                              normalized_laplacian_coo.get_elements().data(),
                                              raft::resource::get_cuda_stream(res));

  // Convert COO result to CSR for comparison
  auto normalized_laplacian_csr =
    make_device_csr_matrix<float, int, int, int>(res,
                                                 normalized_laplacian_coo_structure.get_n_rows(),
                                                 normalized_laplacian_coo_structure.get_n_cols(),
                                                 normalized_laplacian_coo_structure.get_nnz());
  auto normalized_laplacian_csr_structure = normalized_laplacian_csr.structure_view();

  // Initialize indptr to zeros
  raft::matrix::fill(
    res,
    raft::make_device_vector_view(normalized_laplacian_csr_structure.get_indptr().data(),
                                  normalized_laplacian_coo_structure.get_n_rows() + 1),
    int(0));

  // Convert sorted COO to CSR
  raft::sparse::convert::sorted_coo_to_csr<int, int, int>(
    normalized_laplacian_coo_structure.get_rows().data(),
    normalized_laplacian_coo_structure.get_nnz(),
    normalized_laplacian_csr_structure.get_indptr().data(),
    normalized_laplacian_coo_structure.get_n_rows(),
    raft::resource::get_cuda_stream(res));

  // Manually set the last element of indptr to nnz (workaround for potential bug)
  int nnz = normalized_laplacian_coo_structure.get_nnz();
  raft::copy(normalized_laplacian_csr_structure.get_indptr().data() +
               normalized_laplacian_coo_structure.get_n_rows(),
             &nnz,
             1,
             raft::resource::get_cuda_stream(res));

  raft::copy(normalized_laplacian_csr_structure.get_indices().data(),
             normalized_laplacian_coo_structure.get_cols().data(),
             normalized_laplacian_coo_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));

  raft::copy(normalized_laplacian_csr.get_elements().data(),
             normalized_laplacian_coo.get_elements().data(),
             normalized_laplacian_coo_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));

  // Copy results back to host
  auto normalized_laplacian_data = std::vector<float>(normalized_laplacian_csr_structure.get_nnz());
  auto normalized_laplacian_indices =
    std::vector<int>(normalized_laplacian_csr_structure.get_nnz());
  auto normalized_laplacian_indptr =
    std::vector<int>(normalized_laplacian_csr_structure.get_n_rows() + 1);
  auto diagonal_data = std::vector<float>(adjacency_structure_csr.get_n_rows());

  raft::copy(&(normalized_laplacian_data[0]),
             normalized_laplacian_csr.get_elements().data(),
             normalized_laplacian_csr_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(normalized_laplacian_indices[0]),
             normalized_laplacian_csr_structure.get_indices().data(),
             normalized_laplacian_csr_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(normalized_laplacian_indptr[0]),
             normalized_laplacian_csr_structure.get_indptr().data(),
             normalized_laplacian_csr_structure.get_n_rows() + 1,
             raft::resource::get_cuda_stream(res));
  raft::copy(&(diagonal_data[0]),
             diagonal_out.data_handle(),
             diagonal_out.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // For the given adjacency matrix, the diagonal degree matrix D has values [3, 2, 1, 2]
  // The square root of these values is [√3, √2, 1, √2]
  // The normalized Laplacian should have values close to:
  // [1, -1/√(3*2), -1/√(3*1), -1/√(3*2),
  //  -1/√(2*3), 1, -1/√(2*2),
  //  -1/√(1*3), 1,
  //  -1/√(2*3), -1/√(2*2), 1]

  // Expected diagonal values (sqrt of diagonal degree matrix)
  auto expected_diagonal =
    std::vector<float>{std::sqrt(3.0f), std::sqrt(2.0f), 1.0f, std::sqrt(2.0f)};

  // Expected normalized Laplacian values
  auto expected_data = std::vector<float>{1.0f,
                                          -1.0f / std::sqrt(3.0f * 2.0f),
                                          -1.0f / std::sqrt(3.0f * 1.0f),
                                          -1.0f / std::sqrt(3.0f * 2.0f),
                                          -1.0f / std::sqrt(2.0f * 3.0f),
                                          1.0f,
                                          -1.0f / std::sqrt(2.0f * 2.0f),
                                          -1.0f / std::sqrt(1.0f * 3.0f),
                                          1.0f,
                                          -1.0f / std::sqrt(2.0f * 3.0f),
                                          -1.0f / std::sqrt(2.0f * 2.0f),
                                          1.0f};

  // Same indices and indptr as non-normalized Laplacian
  auto expected_indices = std::vector<int>{0, 1, 2, 3, 0, 1, 3, 0, 2, 0, 1, 3};
  auto expected_indptr  = std::vector<int>{0, 4, 7, 9, 12};

  // Compare results with expected values with a small tolerance for floating point differences
  const float tol = 1e-6f;
  ASSERT_EQ(expected_diagonal.size(), diagonal_data.size());
  for (size_t i = 0; i < expected_diagonal.size(); ++i) {
    EXPECT_NEAR(expected_diagonal[i], diagonal_data[i], tol)
      << "COO: Failed at diagonal index " << i;
  }

  ASSERT_EQ(expected_data.size(), normalized_laplacian_data.size());
  for (size_t i = 0; i < expected_data.size(); ++i) {
    EXPECT_NEAR(expected_data[i], normalized_laplacian_data[i], tol)
      << "COO: Failed at data index " << i;
  }

  EXPECT_EQ(expected_indices, normalized_laplacian_indices);
  EXPECT_EQ(expected_indptr, normalized_laplacian_indptr);
}

/**
 * @brief Regression test for laplacian.cuh type mismatch fix (aarch64 + CCCL 3.4.0)
 *
 * Before the fix, `marked_diagonal` was `device_vector<int>` but `thrust::exclusive_scan`
 * expected `device_vector<NZType>`. With NZType=long (64-bit), the type mismatch caused
 * writes to incorrect memory addresses on SM_121a (Grace Hopper) with CCCL 3.4.0's
 * "warpspeed scan" optimization, corrupting the CUDA context.
 *
 * This test uses NZType=long to ensure the fix holds: marked_diagonal must be
 * device_vector<NZType> so the exclusive_scan output type is consistent.
 */
TEST(Raft, ComputeGraphLaplacianCOOLongNZType)
{
  // Adjacency matrix (COO format, NZType=long):
  // [[0 1 1 1]
  //  [1 0 0 1]
  //  [1 0 0 0]
  //  [1 1 0 0]]
  auto rows   = std::vector<int>{0, 0, 0, 1, 1, 2, 3, 3};
  auto cols   = std::vector<int>{1, 2, 3, 0, 3, 0, 0, 1};
  auto data   = std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1};
  int n_rows  = 4;
  int n_cols  = 4;
  long nnz    = static_cast<long>(data.size());

  auto res = raft::resources{};

  auto adjacency_matrix =
    make_device_coo_matrix<float, int, int, long>(res, n_rows, n_cols, nnz);

  raft::copy(adjacency_matrix.get_elements().data(),
             data.data(),
             data.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_matrix.structure_view().get_rows().data(),
             rows.data(),
             rows.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_matrix.structure_view().get_cols().data(),
             cols.data(),
             cols.size(),
             raft::resource::get_cuda_stream(res));

  // This call exercises the fixed code path: marked_diagonal must be NZType=long,
  // not int, to avoid type mismatch in thrust::exclusive_scan on aarch64 + CCCL 3.4.0.
  auto laplacian = compute_graph_laplacian(res, adjacency_matrix.view());
  raft::resource::sync_stream(res);

  // Sort COO output for deterministic comparison
  auto lap_struct = laplacian.structure_view();
  raft::sparse::op::coo_sort<float, int, int>(lap_struct.get_n_rows(),
                                              lap_struct.get_n_cols(),
                                              lap_struct.get_nnz(),
                                              lap_struct.get_rows().data(),
                                              lap_struct.get_cols().data(),
                                              laplacian.get_elements().data(),
                                              raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  auto out_rows   = std::vector<int>(lap_struct.get_nnz());
  auto out_cols   = std::vector<int>(lap_struct.get_nnz());
  auto out_data   = std::vector<float>(lap_struct.get_nnz());

  raft::copy(out_rows.data(), lap_struct.get_rows().data(), lap_struct.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(out_cols.data(), lap_struct.get_cols().data(), lap_struct.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(out_data.data(), laplacian.get_elements().data(), lap_struct.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Expected Laplacian L = D - A:
  // [[ 3 -1 -1 -1]
  //  [-1  2  0 -1]
  //  [-1  0  1  0]
  //  [-1 -1  0  2]]
  // Expected nnz = 8 (off-diag) + 4 (diag) = 12
  ASSERT_EQ(lap_struct.get_nnz(), 12);

  auto expected_rows = std::vector<int>{0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3};
  auto expected_cols = std::vector<int>{0, 1, 2, 3, 0, 1, 3, 0, 2, 0, 1, 3};
  auto expected_data = std::vector<float>{3, -1, -1, -1, -1, 2, -1, -1, 1, -1, -1, 2};

  EXPECT_EQ(expected_rows, out_rows);
  EXPECT_EQ(expected_cols, out_cols);
  EXPECT_EQ(expected_data, out_data);
}

}  // namespace raft::sparse::linalg
