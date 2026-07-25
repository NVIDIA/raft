/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../test_utils.cuh"

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/init.cuh>
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/degree.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/reduce.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/sparse/solver/lanczos.cuh>
#include <raft/sparse/solver/lanczos_types.hpp>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime.h>
#include <driver_types.h>

#include <gtest/gtest.h>
#include <sys/types.h>
#include <test_utils.h>

#include <algorithm>
#include <cstdint>

namespace raft::sparse::solver {

template <typename IndexType, typename ValueType>
struct lanczos_inputs {
  int n_components;
  int restartiter;
  int maxiter;
  int conv_n_iters;
  float conv_eps;
  float tol;
  solver::LANCZOS_WHICH which;
  uint64_t seed;
  std::vector<IndexType> rows;  // indptr
  std::vector<IndexType> cols;  // indices
  std::vector<ValueType> vals;  // data
  std::vector<ValueType> expected_eigenvalues;
};

template <typename IndexType, typename ValueType>
struct rmat_lanczos_inputs {
  int n_components;
  int restartiter;
  int maxiter;
  int conv_n_iters;
  float conv_eps;
  float tol;
  solver::LANCZOS_WHICH which;
  uint64_t seed;
  int r_scale;
  int c_scale;
  float sparsity;
  std::vector<ValueType> expected_eigenvalues;
};

template <typename IndexType, typename ValueType>
class rmat_lanczos_tests
  : public ::testing::TestWithParam<rmat_lanczos_inputs<IndexType, ValueType>> {
 public:
  rmat_lanczos_tests()
    : params(::testing::TestWithParam<rmat_lanczos_inputs<IndexType, ValueType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rng(params.seed),
      expected_eigenvalues(raft::make_device_vector<ValueType, uint32_t, raft::col_major>(
        handle, params.n_components)),
      r_scale(params.r_scale),
      c_scale(params.c_scale),
      sparsity(params.sparsity)
  {
  }

 protected:
  void SetUp() override
  {
    raft::copy(expected_eigenvalues.data_handle(),
               params.expected_eigenvalues.data(),
               params.n_components,
               stream);
  }

  void TearDown() override {}

  void Run()
  {
    uint64_t n_edges   = sparsity * ((long long)(1 << r_scale) * (long long)(1 << c_scale));
    uint64_t n_nodes   = 1 << std::max(r_scale, c_scale);
    uint64_t theta_len = std::max(r_scale, c_scale) * 4;

    auto theta = raft::make_device_vector<ValueType, IndexType, raft::row_major>(handle, theta_len);
    raft::random::uniform<ValueType>(handle, rng, theta.view(), 0, 1);

    auto out = raft::make_device_mdarray<IndexType, IndexType, raft::row_major>(
      handle, raft::extents<IndexType, raft::dynamic_extent, 2>(n_edges, 2));
    auto out_src = raft::make_device_vector<IndexType, IndexType, raft::row_major>(handle, n_edges);
    auto out_dst = raft::make_device_vector<IndexType, IndexType, raft::row_major>(handle, n_edges);

    raft::random::RngState rng1{params.seed};

    raft::random::rmat_rectangular_gen<IndexType, ValueType>(handle,
                                                             rng1,
                                                             make_const_mdspan(theta.view()),
                                                             out.view(),
                                                             out_src.view(),
                                                             out_dst.view(),
                                                             r_scale,
                                                             c_scale);

    raft::device_vector<ValueType, uint32_t, raft::row_major> out_data =
      raft::make_device_vector<ValueType, uint32_t, raft::row_major>(handle, n_edges);
    raft::matrix::fill<ValueType>(handle, out_data.view(), 1.0);
    raft::sparse::COO<ValueType, IndexType, IndexType> coo(stream);

    raft::sparse::op::coo_sort<ValueType, int>(n_nodes,
                                               n_nodes,
                                               n_edges,
                                               out_src.data_handle(),
                                               out_dst.data_handle(),
                                               out_data.data_handle(),
                                               stream);
    raft::sparse::op::max_duplicates<IndexType, ValueType>(handle,
                                                           coo,
                                                           out_src.data_handle(),
                                                           out_dst.data_handle(),
                                                           out_data.data_handle(),
                                                           (IndexType)n_edges,
                                                           (IndexType)n_nodes,
                                                           (IndexType)n_nodes);

    raft::sparse::COO<ValueType, IndexType, IndexType> symmetric_coo(stream);
    raft::sparse::linalg::symmetrize(
      handle, coo.rows(), coo.cols(), coo.vals(), coo.n_rows, coo.n_cols, coo.nnz, symmetric_coo);

    raft::device_vector<IndexType, uint32_t, raft::row_major> row_indices =
      raft::make_device_vector<IndexType, uint32_t, raft::row_major>(handle,
                                                                     symmetric_coo.n_rows + 1);
    raft::sparse::convert::sorted_coo_to_csr(symmetric_coo.rows(),
                                             symmetric_coo.nnz,
                                             row_indices.data_handle(),
                                             symmetric_coo.n_rows + 1,
                                             stream);

    int n_components = params.n_components;

    raft::device_vector<ValueType, uint32_t, raft::row_major> v0 =
      raft::make_device_vector<ValueType, uint32_t, raft::row_major>(handle, symmetric_coo.n_rows);

    raft::random::uniform<ValueType>(handle, rng, v0.view(), 0, 1);
    std::tuple<IndexType, ValueType, IndexType> stats;

    raft::device_vector<ValueType, uint32_t, raft::col_major> eigenvalues =
      raft::make_device_vector<ValueType, uint32_t, raft::col_major>(handle, n_components);
    raft::device_matrix<ValueType, uint32_t, raft::col_major> eigenvectors =
      raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(
        handle, symmetric_coo.n_rows, n_components);

    raft::spectral::matrix::sparse_matrix_t<IndexType, ValueType> const csr_m{
      handle,
      row_indices.data_handle(),
      symmetric_coo.cols(),
      symmetric_coo.vals(),
      symmetric_coo.n_rows,
      (uint64_t)symmetric_coo.nnz};
    raft::sparse::solver::lanczos_solver_config<ValueType> config{
      n_components, params.maxiter, params.restartiter, params.tol, params.which, rng.seed};

    auto csr_structure =
      raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(
        const_cast<IndexType*>(row_indices.data_handle()),
        const_cast<IndexType*>(symmetric_coo.cols()),
        symmetric_coo.n_rows,
        symmetric_coo.n_rows,
        symmetric_coo.nnz);

    auto csr_matrix = raft::make_device_csr_matrix_view<ValueType, IndexType, IndexType, IndexType>(
      const_cast<ValueType*>(symmetric_coo.vals()), csr_structure);

    std::get<0>(stats) = raft::sparse::solver::lanczos_compute_eigenpairs<IndexType, ValueType>(
      handle,
      config,
      csr_matrix,
      std::make_optional(v0.view()),
      eigenvalues.view(),
      eigenvectors.view());

    ASSERT_TRUE(raft::devArrMatch<ValueType>(eigenvalues.data_handle(),
                                             expected_eigenvalues.data_handle(),
                                             n_components,
                                             raft::CompareApprox<ValueType>(1e-5),
                                             stream));

    // Reproducibility test - run again with same seed and verify exact match
    raft::device_vector<ValueType, uint32_t, raft::col_major> eigenvalues2 =
      raft::make_device_vector<ValueType, uint32_t, raft::col_major>(handle, n_components);
    raft::device_matrix<ValueType, uint32_t, raft::col_major> eigenvectors2 =
      raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(
        handle, symmetric_coo.n_rows, n_components);

    raft::sparse::solver::lanczos_compute_eigenpairs<IndexType, ValueType>(
      handle,
      config,
      csr_matrix,
      std::make_optional(v0.view()),
      eigenvalues2.view(),
      eigenvectors2.view());

    ASSERT_TRUE(raft::devArrMatch<ValueType>(eigenvalues.data_handle(),
                                             eigenvalues2.data_handle(),
                                             n_components,
                                             raft::Compare<ValueType>(),
                                             stream));

    // Also test with COO matrix - use the existing symmetric_coo directly
    // Create COO matrix view from the existing COO data
    auto coo_structure =
      raft::make_device_coordinate_structure_view<IndexType, IndexType, IndexType>(
        symmetric_coo.rows(),
        symmetric_coo.cols(),
        symmetric_coo.n_rows,
        symmetric_coo.n_rows,
        symmetric_coo.nnz);

    auto coo_matrix = raft::make_device_coo_matrix_view<ValueType, IndexType, IndexType, IndexType>(
      symmetric_coo.vals(), coo_structure);

    raft::device_vector<ValueType, uint32_t, raft::col_major> eigenvalues_coo =
      raft::make_device_vector<ValueType, uint32_t, raft::col_major>(handle, n_components);
    raft::device_matrix<ValueType, uint32_t, raft::col_major> eigenvectors_coo =
      raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(
        handle, symmetric_coo.n_rows, n_components);

    std::get<0>(stats) = raft::sparse::solver::lanczos_compute_eigenpairs<IndexType, ValueType>(
      handle,
      config,
      coo_matrix,
      std::make_optional(v0.view()),
      eigenvalues_coo.view(),
      eigenvectors_coo.view());

    ASSERT_TRUE(raft::devArrMatch<ValueType>(eigenvalues_coo.data_handle(),
                                             expected_eigenvalues.data_handle(),
                                             n_components,
                                             raft::CompareApprox<ValueType>(1e-4),
                                             stream));
  }

 protected:
  rmat_lanczos_inputs<IndexType, ValueType> params;
  raft::resources handle;
  cudaStream_t stream;
  raft::random::RngState rng;
  int r_scale;
  int c_scale;
  float sparsity;
  raft::device_vector<ValueType, uint32_t, raft::col_major> expected_eigenvalues;
};

template <typename IndexType, typename ValueType>
class lanczos_tests : public ::testing::TestWithParam<lanczos_inputs<IndexType, ValueType>> {
 public:
  lanczos_tests()
    : params(::testing::TestWithParam<lanczos_inputs<IndexType, ValueType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      n(params.rows.size() - 1),
      nnz(params.vals.size()),
      rng(params.seed),
      rows(raft::make_device_vector<IndexType, uint32_t, raft::row_major>(handle, n + 1)),
      cols(raft::make_device_vector<IndexType, uint32_t, raft::row_major>(handle, nnz)),
      vals(raft::make_device_vector<ValueType, uint32_t, raft::row_major>(handle, nnz)),
      v0(raft::make_device_vector<ValueType, uint32_t, raft::row_major>(handle, n)),
      eigenvalues(raft::make_device_vector<ValueType, uint32_t, raft::col_major>(
        handle, params.n_components)),
      eigenvectors(raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(
        handle, n, params.n_components)),
      expected_eigenvalues(
        raft::make_device_vector<ValueType, uint32_t, raft::col_major>(handle, params.n_components))
  {
  }

 protected:
  void SetUp() override
  {
    raft::copy(rows.data_handle(), params.rows.data(), n + 1, stream);
    raft::copy(cols.data_handle(), params.cols.data(), nnz, stream);
    raft::copy(vals.data_handle(), params.vals.data(), nnz, stream);
    raft::copy(expected_eigenvalues.data_handle(),
               params.expected_eigenvalues.data(),
               params.n_components,
               stream);
  }

  void TearDown() override {}

  void Run()
  {
    int runtimeVersion;
    cudaError_t result = cudaRuntimeGetVersion(&runtimeVersion);

    if (result == cudaSuccess) {
      int major = runtimeVersion / 1000;
      int minor = (runtimeVersion % 1000) / 10;

      // Skip SM gtests for CUDA 13.1.x
      // See https://github.com/rapidsai/raft/issues/2705
      if (major == 13 && minor == 1 && params.which == raft::sparse::solver::LANCZOS_WHICH::SM) {
        GTEST_SKIP();
      }
    }

    raft::random::uniform<ValueType>(handle, rng, v0.view(), 0, 1);
    std::tuple<IndexType, ValueType, IndexType> stats;

    raft::sparse::solver::lanczos_solver_config<ValueType> config{
      params.n_components, params.maxiter, params.restartiter, params.tol, params.which, rng.seed};
    auto csr_structure =
      raft::make_device_compressed_structure_view<IndexType, IndexType, IndexType>(
        const_cast<IndexType*>(rows.data_handle()),
        const_cast<IndexType*>(cols.data_handle()),
        n,
        n,
        nnz);

    auto csr_matrix = raft::make_device_csr_matrix_view<ValueType, IndexType, IndexType, IndexType>(
      const_cast<ValueType*>(vals.data_handle()), csr_structure);

    std::get<0>(stats) = raft::sparse::solver::lanczos_compute_eigenpairs<IndexType, ValueType>(
      handle,
      config,
      csr_matrix,
      std::make_optional(v0.view()),
      eigenvalues.view(),
      eigenvectors.view());

    ASSERT_TRUE(raft::devArrMatch<ValueType>(
      eigenvalues.data_handle(),
      expected_eigenvalues.data_handle(),
      params.n_components,
      raft::CompareApprox<ValueType>(
        params.which == raft::sparse::solver::LANCZOS_WHICH::SM ? 5e-5 : 1e-5),
      stream));

    // Reproducibility test - run again with same seed and verify exact match
    raft::device_vector<ValueType, uint32_t, raft::col_major> eigenvalues2 =
      raft::make_device_vector<ValueType, uint32_t, raft::col_major>(handle, params.n_components);
    raft::device_matrix<ValueType, uint32_t, raft::col_major> eigenvectors2 =
      raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(
        handle, n, params.n_components);

    raft::sparse::solver::lanczos_compute_eigenpairs<IndexType, ValueType>(
      handle,
      config,
      csr_matrix,
      std::make_optional(v0.view()),
      eigenvalues2.view(),
      eigenvectors2.view());

    ASSERT_TRUE(raft::devArrMatch<ValueType>(eigenvalues.data_handle(),
                                             eigenvalues2.data_handle(),
                                             params.n_components,
                                             raft::Compare<ValueType>(),
                                             stream));

    // Also test with COO matrix (convert CSR to COO format)
    raft::device_vector<IndexType, uint32_t, raft::row_major> coo_rows =
      raft::make_device_vector<IndexType, uint32_t, raft::row_major>(handle, nnz);
    raft::sparse::convert::csr_to_coo(rows.data_handle(), n, coo_rows.data_handle(), nnz, stream);

    auto coo_structure =
      raft::make_device_coordinate_structure_view<IndexType, IndexType, IndexType>(
        coo_rows.data_handle(), cols.data_handle(), n, n, nnz);

    auto coo_matrix = raft::make_device_coo_matrix_view<ValueType, IndexType, IndexType, IndexType>(
      vals.data_handle(), coo_structure);

    raft::device_vector<ValueType, uint32_t, raft::col_major> eigenvalues_coo =
      raft::make_device_vector<ValueType, uint32_t, raft::col_major>(handle, params.n_components);
    raft::device_matrix<ValueType, uint32_t, raft::col_major> eigenvectors_coo =
      raft::make_device_matrix<ValueType, uint32_t, raft::col_major>(
        handle, n, params.n_components);

    std::get<0>(stats) = raft::sparse::solver::lanczos_compute_eigenpairs<IndexType, ValueType>(
      handle,
      config,
      coo_matrix,
      std::make_optional(v0.view()),
      eigenvalues_coo.view(),
      eigenvectors_coo.view());

    ASSERT_TRUE(raft::devArrMatch<ValueType>(
      eigenvalues_coo.data_handle(),
      expected_eigenvalues.data_handle(),
      params.n_components,
      raft::CompareApprox<ValueType>(
        params.which == raft::sparse::solver::LANCZOS_WHICH::SM ? 5e-5 : 1e-5),
      stream));
  }

 protected:
  lanczos_inputs<IndexType, ValueType> params;
  raft::resources handle;
  cudaStream_t stream;
  int n;
  int nnz;
  raft::random::RngState rng;
  raft::device_vector<IndexType, uint32_t, raft::row_major> rows;
  raft::device_vector<IndexType, uint32_t, raft::row_major> cols;
  raft::device_vector<ValueType, uint32_t, raft::row_major> vals;
  raft::device_vector<ValueType, uint32_t, raft::row_major> v0;
  raft::device_vector<ValueType, uint32_t, raft::col_major> eigenvalues;
  raft::device_matrix<ValueType, uint32_t, raft::col_major> eigenvectors;
  raft::device_vector<ValueType, uint32_t, raft::col_major> expected_eigenvalues;
};

template <typename IndexType>
std::vector<IndexType> rows()
{
  return {0,    19,   33,   44,   63,   75,   89,   101,  115,  122,  134, 146, 156,  169,  182,
          192,  205,  216,  231,  237,  244,  256,  269,  280,  291,  296, 303, 315,  325,  336,
          349,  368,  382,  390,  405,  418,  429,  432,  442,  455,  466, 481, 491,  505,  517,
          523,  534,  546,  562,  570,  590,  603,  614,  626,  638,  648, 662, 670,  678,  689,
          701,  714,  723,  736,  745,  756,  769,  782,  791,  805,  812, 826, 835,  851,  863,
          876,  889,  901,  911,  922,  931,  940,  957,  967,  981,  991, 999, 1006, 1019, 1034,
          1042, 1058, 1068, 1077, 1088, 1103, 1112, 1121, 1129, 1140, 1152};
}

template <typename IndexType>
std::vector<IndexType> cols()
{
  return {
    {0,  3,  5,  14, 29, 31, 33, 34, 35, 39, 51, 61, 76, 80, 85, 88, 91, 92, 96, 1,  6,  7,  8,
     19, 35, 44, 65, 72, 76, 78, 80, 90, 97, 28, 42, 43, 51, 52, 59, 63, 68, 90, 91, 92, 0,  3,
     4,  11, 21, 30, 33, 40, 46, 49, 54, 57, 59, 60, 66, 67, 72, 84, 88, 3,  16, 23, 32, 34, 39,
     45, 50, 60, 72, 83, 87, 0,  10, 18, 32, 33, 43, 52, 55, 60, 68, 72, 76, 82, 90, 1,  16, 21,
     23, 24, 29, 30, 47, 52, 68, 73, 87, 1,  11, 13, 30, 33, 35, 64, 65, 75, 78, 79, 94, 95, 99,
     1,  12, 14, 29, 49, 56, 59, 15, 26, 52, 53, 58, 64, 70, 71, 81, 87, 93, 99, 5,  26, 29, 34,
     42, 56, 59, 68, 71, 83, 84, 90, 3,  7,  12, 21, 23, 45, 47, 79, 88, 94, 8,  11, 15, 24, 33,
     34, 39, 41, 47, 48, 53, 61, 94, 7,  15, 22, 29, 45, 46, 47, 48, 55, 62, 73, 83, 84, 0,  8,
     45, 54, 72, 74, 82, 92, 94, 98, 9,  12, 13, 22, 26, 30, 58, 62, 75, 78, 80, 93, 97, 4,  6,
     17, 23, 31, 42, 45, 50, 70, 72, 91, 16, 30, 31, 33, 38, 42, 58, 60, 65, 74, 75, 85, 93, 94,
     95, 5,  20, 62, 64, 69, 76, 1,  48, 49, 50, 63, 88, 99, 18, 20, 25, 29, 45, 67, 74, 75, 80,
     87, 94, 99, 3,  6,  11, 24, 43, 45, 47, 49, 54, 65, 66, 78, 89, 13, 15, 25, 29, 31, 32, 33,
     48, 49, 54, 58, 4,  6,  11, 16, 37, 40, 44, 51, 53, 60, 91, 6,  12, 21, 58, 98, 20, 22, 45,
     46, 78, 86, 92, 9,  10, 15, 27, 29, 35, 50, 64, 68, 73, 78, 88, 26, 29, 37, 48, 50, 53, 55,
     67, 81, 88, 2,  55, 63, 64, 72, 76, 77, 83, 84, 94, 98, 0,  6,  8,  10, 13, 20, 22, 26, 27,
     49, 63, 75, 86, 3,  6,  7,  15, 17, 32, 33, 39, 41, 46, 48, 61, 66, 68, 70, 79, 90, 95, 99,
     0,  16, 17, 22, 32, 45, 46, 47, 54, 60, 62, 72, 87, 97, 4,  5,  22, 30, 31, 38, 70, 79, 0,
     3,  5,  7,  12, 17, 22, 30, 37, 50, 60, 89, 93, 97, 99, 0,  4,  10, 12, 34, 38, 52, 65, 68,
     73, 81, 82, 88, 0,  1,  7,  26, 41, 56, 70, 72, 93, 95, 97, 62, 74, 83, 23, 27, 33, 38, 43,
     55, 71, 85, 87, 90, 17, 32, 34, 37, 40, 57, 59, 72, 74, 79, 90, 92, 98, 0,  4,  12, 30, 41,
     60, 73, 75, 90, 92, 99, 3,  23, 38, 41, 46, 49, 54, 58, 59, 63, 71, 76, 80, 81, 84, 12, 30,
     35, 39, 40, 57, 78, 85, 90, 94, 2,  10, 16, 17, 44, 47, 49, 64, 69, 76, 77, 79, 81, 83, 2,
     5,  21, 37, 67, 68, 81, 82, 86, 87, 96, 98, 1,  23, 42, 71, 89, 96, 4,  11, 13, 14, 16, 20,
     21, 25, 31, 75, 94, 3,  13, 25, 30, 31, 40, 66, 71, 81, 84, 88, 95, 6,  11, 12, 13, 21, 31,
     42, 47, 49, 51, 55, 68, 73, 75, 84, 88, 12, 13, 19, 22, 27, 30, 51, 75, 3,  8,  19, 21, 22,
     29, 40, 42, 47, 49, 55, 57, 61, 63, 77, 83, 88, 90, 91, 93, 4,  16, 19, 26, 27, 33, 50, 52,
     53, 65, 66, 78, 82, 0,  2,  23, 47, 48, 56, 69, 70, 83, 96, 98, 2,  5,  6,  9,  34, 50, 62,
     65, 66, 69, 72, 79, 9,  12, 23, 27, 50, 56, 58, 59, 67, 77, 81, 97, 3,  14, 21, 22, 31, 40,
     59, 67, 81, 90, 5,  13, 27, 28, 37, 47, 49, 59, 71, 72, 83, 89, 91, 96, 8,  10, 35, 51, 53,
     62, 65, 74, 3,  38, 41, 49, 66, 74, 94, 95, 9,  15, 17, 22, 24, 40, 53, 61, 70, 81, 85, 2,
     3,  8,  10, 38, 40, 53, 54, 55, 70, 81, 98, 3,  4,  5,  17, 23, 31, 33, 39, 60, 70, 77, 82,
     98, 0,  12, 30, 49, 58, 62, 63, 76, 83, 13, 15, 18, 31, 36, 52, 56, 61, 66, 83, 89, 92, 93,
     2,  19, 28, 29, 40, 49, 61, 63, 86, 7,  9,  18, 26, 28, 42, 72, 74, 82, 88, 96, 1,  7,  17,
     21, 34, 50, 52, 56, 67, 89, 90, 95, 99, 3,  21, 30, 46, 50, 52, 57, 62, 79, 81, 87, 90, 99,
     3,  20, 27, 43, 53, 54, 65, 89, 90, 2,  5,  6,  10, 26, 30, 34, 43, 47, 69, 70, 79, 88, 97,
     18, 42, 51, 52, 68, 71, 77, 9,  16, 30, 32, 35, 51, 58, 59, 60, 68, 72, 77, 86, 93, 9,  10,
     37, 40, 44, 46, 55, 69, 98, 1,  3,  4,  5,  14, 16, 28, 31, 35, 38, 52, 55, 64, 70, 77, 99,
     6,  13, 26, 34, 39, 47, 76, 77, 83, 88, 96, 99, 14, 17, 20, 36, 38, 56, 57, 64, 80, 81, 85,
     87, 92, 7,  15, 17, 20, 29, 39, 45, 47, 48, 85, 90, 92, 94, 0,  1,  5,  18, 28, 40, 42, 61,
     73, 80, 84, 95, 28, 42, 49, 53, 60, 69, 70, 72, 73, 91, 1,  7,  15, 21, 25, 26, 41, 50, 81,
     82, 87, 7,  11, 30, 32, 38, 42, 52, 66, 68, 0,  1,  15, 20, 40, 74, 76, 80, 93, 9,  27, 34,
     40, 42, 43, 46, 53, 54, 58, 59, 66, 74, 78, 81, 84, 87, 5,  14, 34, 43, 50, 60, 64, 78, 83,
     84, 4,  10, 13, 28, 36, 42, 49, 51, 55, 61, 62, 73, 82, 97, 3,  10, 13, 28, 40, 46, 47, 76,
     81, 82, 0,  17, 37, 41, 58, 74, 75, 94, 25, 29, 43, 63, 70, 94, 99, 4,  6,  9,  20, 31, 37,
     43, 66, 74, 78, 81, 91, 94, 0,  3,  11, 19, 26, 27, 34, 46, 47, 49, 64, 68, 73, 94, 95, 21,
     33, 44, 55, 62, 65, 67, 91, 1,  2,  5,  10, 30, 37, 38, 39, 41, 49, 54, 65, 66, 67, 75, 91,
     0,  2,  16, 23, 49, 55, 77, 87, 89, 90, 0,  2,  14, 25, 38, 39, 62, 74, 75, 9,  15, 17, 33,
     35, 49, 62, 70, 80, 93, 98, 7,  11, 12, 14, 17, 20, 28, 41, 45, 57, 75, 85, 86, 87, 88, 7,
     17, 30, 35, 46, 57, 65, 76, 88, 0,  43, 44, 51, 55, 64, 73, 96, 98, 1,  15, 31, 33, 35, 53,
     68, 83, 14, 24, 28, 38, 43, 51, 59, 60, 71, 93, 96, 7,  9,  19, 20, 30, 33, 39, 65, 66, 72,
     73, 86}};
};

template <typename IndexType>
std::vector<IndexType> vals()
{
  return {0.8385492, 0.0887047, 0.4178915, 0.4469863, 0.3129920, 0.1031984, 0.2898832, 0.3742032,
          0.0522814, 0.4283062, 0.0722899, 0.2825163, 0.3556329, 0.4577443, 0.1463357, 0.3288944,
          0.4404407, 0.3466910, 0.1813836, 0.4619252, 0.2565850, 0.3349850, 0.4806673, 0.2509201,
          0.5977305, 0.1196116, 0.4918659, 0.4487811, 0.3017929, 0.1373989, 0.3152796, 0.0489285,
          0.4335400, 0.2137529, 0.0752723, 0.1238085, 0.3818015, 0.2676854, 0.3768051, 0.0550861,
          0.0404749, 0.2313439, 0.4663205, 0.3860448, 0.0887047, 0.2646495, 0.4319774, 0.0975544,
          0.0630765, 0.3240158, 0.3895027, 0.0810854, 0.2185860, 0.1749017, 0.4413908, 0.3265684,
          0.3630947, 0.3439550, 0.4709083, 0.4871525, 0.4059546, 0.1080630, 0.4250860, 0.4319774,
          0.1542346, 0.1705101, 0.3939655, 0.2228894, 0.1612249, 0.3348420, 0.2223779, 0.2221761,
          0.0100264, 0.4906516, 0.4211857, 0.4178915, 0.2112697, 0.0841055, 0.4681202, 0.3307955,
          0.2055745, 0.3013350, 0.0921099, 0.0400448, 0.3691756, 0.2582222, 0.2400539, 0.5111299,
          0.3747997, 0.2565850, 0.1914687, 0.3671139, 0.1276162, 0.3537677, 0.4076773, 0.1181683,
          0.2186738, 0.2885858, 0.0898227, 0.2496215, 0.0710704, 0.3349850, 0.1886938, 0.3514959,
          0.4223959, 0.3594954, 0.2522124, 0.2545409, 0.3583593, 0.2345200, 0.0271059, 0.3803033,
          0.2441548, 0.4808156, 0.3853460, 0.4806673, 0.2801995, 0.3858915, 0.4087827, 0.3671360,
          0.4641679, 0.4647529, 0.4388565, 0.2036911, 0.3038480, 0.0642898, 0.1680801, 0.4766186,
          0.0175873, 0.3034133, 0.4123132, 0.0379968, 0.2168891, 0.4039690, 0.2112697, 0.1334496,
          0.0195393, 0.3627390, 0.4262152, 0.0403542, 0.3081270, 0.0121030, 0.4202002, 0.0157230,
          0.3200969, 0.3018721, 0.0975544, 0.1886938, 0.0700827, 0.2364397, 0.4924574, 0.1961911,
          0.1126200, 0.2911978, 0.3152088, 0.2863414, 0.2801995, 0.0700827, 0.3417172, 0.3470362,
          0.1736438, 0.0880198, 0.4559108, 0.1121846, 0.4507234, 0.3779829, 0.3405251, 0.4304054,
          0.1778966, 0.3514959, 0.2997384, 0.0548181, 0.3352703, 0.0580903, 0.3582265, 0.1743012,
          0.2712877, 0.4273703, 0.0667344, 0.1248124, 0.2219280, 0.0032627, 0.4469863, 0.3858915,
          0.4647915, 0.4203163, 0.0276749, 0.3857849, 0.3978934, 0.3266448, 0.7978231, 0.4209127,
          0.4388565, 0.3417172, 0.2997384, 0.2899171, 0.1479114, 0.4418810, 0.0634220, 0.0929668,
          0.0947451, 0.2785034, 0.3162361, 0.3741804, 0.1492740, 0.1542346, 0.1914687, 0.3533593,
          0.3017906, 0.4373029, 0.3273854, 0.2602280, 0.4756982, 0.0231310, 0.1348172, 0.3698108,
          0.3533593, 0.2026076, 0.2999429, 0.2704931, 0.2946429, 0.3154255, 0.0627621, 0.2774129,
          0.4206699, 0.3919643, 0.2397596, 0.2703603, 0.4124186, 0.5805955, 0.1959839, 0.0841055,
          0.4032480, 0.3097841, 0.2355419, 0.0072671, 0.2746342, 0.2509201, 0.4638373, 0.4052183,
          0.4148478, 0.0242814, 0.0475468, 0.4618716, 0.4032480, 0.8676416, 0.3012557, 0.1490197,
          0.0643736, 0.2925389, 0.4948683, 0.4953563, 0.0355901, 0.1778870, 0.0851296, 0.0398270,
          0.0630765, 0.3671139, 0.2364397, 0.4431843, 0.4763364, 0.4127292, 0.2751093, 0.4222012,
          0.4474346, 0.0614609, 0.0204758, 0.4293712, 0.6431416, 0.0548181, 0.2899171, 0.3634890,
          0.4412351, 0.1490288, 0.0447601, 0.4229826, 0.4666647, 0.4085002, 0.1522877, 0.2532282,
          0.1705101, 0.1276162, 0.4924574, 0.3017906, 0.4809324, 0.2783410, 0.4891155, 0.4529251,
          0.2209245, 0.3961954, 0.3886663, 0.3537677, 0.3470362, 0.4431843, 0.3647585, 0.4439365,
          0.3012557, 0.3634890, 0.4831129, 0.0737263, 0.0975274, 0.1318027, 0.4008097, 0.2036911,
          0.1334496, 0.1479114, 0.1494316, 0.4203018, 0.2847606, 0.1855530, 0.1457941, 0.1710113,
          0.4871747, 0.0616420, 0.2720916, 0.1494316, 0.3342525, 0.4981484, 0.2835164, 0.2201109,
          0.3569505, 0.3228796, 0.2912845, 0.3213883, 0.1017065, 0.2137529, 0.0917027, 0.2898162,
          0.1962939, 0.4857582, 0.0757288, 0.4719980, 0.0358659, 0.3427203, 0.0026993, 0.3121863,
          0.3129920, 0.4076773, 0.4087827, 0.0195393, 0.3352703, 0.1490197, 0.4412351, 0.4203018,
          0.3342525, 0.3198876, 0.0273423, 0.2868139, 0.0167729, 0.3240158, 0.1181683, 0.4223959,
          0.4418810, 0.2026076, 0.2938424, 0.2349892, 0.3265936, 0.4161196, 0.4671349, 0.4333457,
          0.2783662, 0.3295173, 0.1414430, 0.4789151, 0.5482149, 0.2405941, 0.0873055, 0.4915100,
          0.1031984, 0.4373029, 0.2999429, 0.1490288, 0.4463057, 0.3714156, 0.1611302, 0.3379962,
          0.4963855, 0.0027219, 0.2011968, 0.0713272, 0.4434711, 0.4655451, 0.3939655, 0.4681202,
          0.0447601, 0.2938424, 0.4463057, 0.4189366, 0.4029289, 0.2293249, 0.2898832, 0.3895027,
          0.3307955, 0.3594954, 0.1736438, 0.2704931, 0.4229826, 0.2349892, 0.3299349, 0.3658398,
          0.1270155, 0.1812549, 0.2024299, 0.2777625, 0.4278218, 0.3742032, 0.2228894, 0.3627390,
          0.0880198, 0.6981739, 0.2518100, 0.4279115, 0.4749146, 0.1864383, 0.0079268, 0.2446150,
          0.4631400, 0.2501489, 0.0522814, 0.5977305, 0.2522124, 0.2847606, 0.0239194, 0.2294849,
          0.1320064, 0.0275548, 0.2787146, 0.2884416, 0.1149032, 0.2807390, 0.0142855, 0.0743166,
          0.4809324, 0.4981484, 0.3299349, 0.2030550, 0.3606436, 0.1871530, 0.1441479, 0.3848211,
          0.3917919, 0.4778197, 0.2946429, 0.4189366, 0.2518100, 0.2030550, 0.4912895, 0.1039219,
          0.0522639, 0.1156571, 0.2768389, 0.4478964, 0.3747582, 0.0281233, 0.0742416, 0.4283062,
          0.1612249, 0.4559108, 0.3265936, 0.3309392, 0.0915848, 0.4962989, 0.4802843, 0.3192275,
          0.4662680, 0.1974726, 0.0810854, 0.2783410, 0.4912895, 0.3111583, 0.2043965, 0.3393941,
          0.2771504, 0.4931641, 0.1747926, 0.0280012, 0.1170995, 0.0880413, 0.2797376, 0.2573348,
          0.2639170, 0.1121846, 0.4161196, 0.0239194, 0.3309392, 0.3111583, 0.0100230, 0.4227756,
          0.4669757, 0.1495063, 0.4342381, 0.0752723, 0.4262152, 0.3273854, 0.3154255, 0.1063635,
          0.3477053, 0.4418667, 0.4065750, 0.2158301, 0.0978730, 0.3474138, 0.0941925, 0.2502904,
          0.4167871, 0.1238085, 0.2055745, 0.4763364, 0.3606436, 0.4519934, 0.0640926, 0.3494168,
          0.2239768, 0.1506959, 0.3404586, 0.3356108, 0.0563368, 0.1196116, 0.4891155, 0.1063635,
          0.2269343, 0.1701081, 0.2478454, 0.3348420, 0.1961911, 0.0580903, 0.4647915, 0.2602280,
          0.0643736, 0.4127292, 0.4831129, 0.3714156, 0.1118892, 0.3946897, 0.2185860, 0.3582265,
          0.0737263, 0.4671349, 0.1611302, 0.2043965, 0.0178831, 0.3605796, 0.3192719, 0.2912215,
          0.2253530, 0.0770738, 0.2186738, 0.1126200, 0.4507234, 0.1743012, 0.2751093, 0.3379962,
          0.3477053, 0.7218004, 0.3724327, 0.2531753, 0.0252278, 0.1346799, 0.1110411, 0.2297511,
          0.4249687, 0.4478221, 0.3779829, 0.2712877, 0.4638373, 0.4666647, 0.2835164, 0.4333457,
          0.4028734, 0.3528297, 0.1749017, 0.3671360, 0.4052183, 0.4222012, 0.4085002, 0.3198876,
          0.3393941, 0.4418667, 0.3724327, 0.0297363, 0.1562718, 0.1693750, 0.3152241, 0.2076069,
          0.2933232, 0.0599440, 0.1064471, 0.2469246, 0.0528503, 0.0321264, 0.2223779, 0.4756982,
          0.4148478, 0.1855530, 0.2201109, 0.3658398, 0.1724464, 0.2184300, 0.4806912, 0.0183148,
          0.4068079, 0.3377662, 0.1434079, 0.0722899, 0.3818015, 0.4529251, 0.2531753, 0.4028734,
          0.3104094, 0.3041750, 0.7444220, 0.0469955, 0.4269330, 0.3876660, 0.2676854, 0.3013350,
          0.2885858, 0.3038480, 0.4279115, 0.2184300, 0.3897764, 0.1165223, 0.1623044, 0.3694641,
          0.4647038, 0.1950430, 0.0642898, 0.3405251, 0.2209245, 0.3569505, 0.4806912, 0.3142114,
          0.1688596, 0.2013356, 0.3616973, 0.3665765, 0.1457076, 0.2400093, 0.4413908, 0.4203163,
          0.4474346, 0.1522877, 0.4963855, 0.2771504, 0.3932629, 0.0499926, 0.1397347, 0.1162840,
          0.0921099, 0.4273703, 0.3228796, 0.0917027, 0.1871530, 0.0252278, 0.1562718, 0.0526326,
          0.2417610, 0.1361428, 0.4773201, 0.2157922, 0.2075818, 0.1745262, 0.4641679, 0.0403542,
          0.2294849, 0.3104094, 0.3142114, 0.1920804, 0.3407301, 0.3448193, 0.3265684, 0.1039219,
          0.0100230, 0.1693750, 0.0846982, 0.3384884, 0.1491689, 0.4107678, 0.1680801, 0.0634220,
          0.0627621, 0.2532282, 0.3647585, 0.4931641, 0.1688596, 0.4067034, 0.2580674, 0.3928346,
          0.2107771, 0.3768051, 0.3630947, 0.4647529, 0.3081270, 0.0522639, 0.1747926, 0.2013356,
          0.3932629, 0.0526326, 0.0972740, 0.3075456, 0.4610244, 0.3439550, 0.2221761, 0.0400448,
          0.2774129, 0.3961954, 0.0027219, 0.1270155, 0.0915848, 0.0703681, 0.2613653, 0.1867701,
          0.3600508, 0.2641133, 0.2825163, 0.4304054, 0.2783662, 0.3152241, 0.4067034, 0.3054797,
          0.4133712, 0.3381947, 0.2909326, 0.0667344, 0.0929668, 0.3097841, 0.2011968, 0.2807390,
          0.3897764, 0.1920804, 0.3054797, 0.0461345, 0.4405020, 0.0232168, 0.2595815, 0.1997993,
          0.0550861, 0.0242814, 0.2898162, 0.0273423, 0.0280012, 0.2076069, 0.4133712, 0.1691249,
          0.4277944, 0.2545409, 0.4766186, 0.2355419, 0.1457941, 0.1962939, 0.4065750, 0.0040041,
          0.3576335, 0.1925813, 0.0949263, 0.3370988, 0.4918659, 0.3583593, 0.4206699, 0.0614609,
          0.4749146, 0.0183148, 0.1165223, 0.3407301, 0.3476292, 0.0964920, 0.2228194, 0.2653476,
          0.1358538, 0.4709083, 0.0204758, 0.3295173, 0.0178831, 0.4068079, 0.1623044, 0.0846982,
          0.0461345, 0.1464626, 0.1254936, 0.0599609, 0.2089537, 0.0068666, 0.4871525, 0.2925389,
          0.2912845, 0.4519934, 0.3616973, 0.0499926, 0.3476292, 0.1524191, 0.4803574, 0.0404749,
          0.3691756, 0.0898227, 0.0121030, 0.1710113, 0.1414430, 0.1864383, 0.0640926, 0.1346799,
          0.0812578, 0.3378220, 0.2461788, 0.3674253, 0.4063063, 0.0072671, 0.2158301, 0.3041750,
          0.3694641, 0.0812578, 0.4179677, 0.0273935, 0.0175873, 0.0231310, 0.4789151, 0.4029289,
          0.1320064, 0.7444220, 0.2580674, 0.0972740, 0.2613653, 0.3378220, 0.0734471, 0.2880797,
          0.0932892, 0.4374486, 0.3034133, 0.4202002, 0.1441479, 0.1170995, 0.2269343, 0.3605796,
          0.2417610, 0.4179677, 0.1591760, 0.4487811, 0.4059546, 0.0100264, 0.2582222, 0.0276749,
          0.1348172, 0.4857582, 0.0713272, 0.0275548, 0.1156571, 0.4647038, 0.1361428, 0.0040041,
          0.0734471, 0.1602434, 0.4003320, 0.2496215, 0.1248124, 0.4871747, 0.0079268, 0.4962989,
          0.1110411, 0.2440522, 0.4165031, 0.0252984, 0.4441046, 0.4572209, 0.4238202, 0.3857849,
          0.3919643, 0.4948683, 0.0142855, 0.2768389, 0.3448193, 0.3384884, 0.3576335, 0.1268078,
          0.2563025, 0.1582990, 0.4166048, 0.2596988, 0.2345200, 0.0947451, 0.2397596, 0.4953563,
          0.2868139, 0.4802843, 0.1118892, 0.2297511, 0.3528297, 0.1603665, 0.3632484, 0.4808010,
          0.3155171, 0.3556329, 0.3017929, 0.2400539, 0.2746342, 0.0757288, 0.0880413, 0.0978730,
          0.3381947, 0.2440522, 0.1453612, 0.4289585, 0.1462075, 0.4719980, 0.3474138, 0.2933232,
          0.3665765, 0.1867701, 0.0273935, 0.2880797, 0.1602434, 0.4165031, 0.3889723, 0.1373989,
          0.0271059, 0.2785034, 0.4293712, 0.0975274, 0.0616420, 0.4227756, 0.3377662, 0.4863133,
          0.1468162, 0.3094676, 0.3803033, 0.2911978, 0.5482149, 0.2293249, 0.4478964, 0.0941925,
          0.1950430, 0.1464626, 0.2461788, 0.4577443, 0.3152796, 0.3162361, 0.0355901, 0.2797376,
          0.1268078, 0.1453612, 0.8429500, 0.2124143, 0.4123132, 0.3213883, 0.2446150, 0.2573348,
          0.2502904, 0.3494168, 0.3192719, 0.1457076, 0.1397347, 0.3928346, 0.3075456, 0.1254936,
          0.2563025, 0.4863133, 0.2630258, 0.1192136, 0.1283403, 0.5111299, 0.3978934, 0.4631400,
          0.2239768, 0.1434079, 0.3600508, 0.1925813, 0.1468162, 0.2636598, 0.2737032, 0.4906516,
          0.0157230, 0.2219280, 0.0358659, 0.0743166, 0.4167871, 0.0599440, 0.0469955, 0.4773201,
          0.2909326, 0.4405020, 0.0252984, 0.2636598, 0.2486204, 0.1080630, 0.3200969, 0.0032627,
          0.3427203, 0.2639170, 0.2912215, 0.4249687, 0.4289585, 0.1192136, 0.2737032, 0.1463357,
          0.2703603, 0.3848211, 0.4669757, 0.2107771, 0.1582990, 0.1603665, 0.1633865, 0.1318027,
          0.0167729, 0.1506959, 0.4277944, 0.0932892, 0.4795561, 0.3394504, 0.4211857, 0.0710704,
          0.0379968, 0.1778870, 0.4434711, 0.3917919, 0.3404586, 0.0599609, 0.4166048, 0.3094676,
          0.1283403, 0.4126990, 0.0727838, 0.3288944, 0.4250860, 0.3152088, 0.0475468, 0.2720916,
          0.1017065, 0.2501489, 0.2253530, 0.4478221, 0.1064471, 0.0949263, 0.3674253, 0.4441046,
          0.1663671, 0.0782135, 0.6431416, 0.1812549, 0.1701081, 0.2157922, 0.0232168, 0.0964920,
          0.1524191, 0.4609937, 0.0489285, 0.2313439, 0.3747997, 0.3018721, 0.2405941, 0.4778197,
          0.3747582, 0.3192275, 0.1495063, 0.2469246, 0.1162840, 0.2228194, 0.2089537, 0.4803574,
          0.3632484, 0.1381195, 0.4404407, 0.4663205, 0.3698108, 0.3886663, 0.0528503, 0.2075818,
          0.3889723, 0.4126990, 0.4609937, 0.1381195, 0.3466910, 0.3860448, 0.3266448, 0.4008097,
          0.0281233, 0.4662680, 0.2595815, 0.2596988, 0.4808010, 0.2168891, 0.3741804, 0.4124186,
          0.2024299, 0.2787146, 0.0321264, 0.1997993, 0.4374486, 0.2124143, 0.2074694, 0.1906150,
          0.2441548, 0.2863414, 0.1778966, 0.7978231, 0.5805955, 0.0851296, 0.0026993, 0.4342381,
          0.3946897, 0.1491689, 0.3155171, 0.1633865, 0.4795561, 0.0727838, 0.1663671, 0.4808156,
          0.1959839, 0.0873055, 0.2884416, 0.0770738, 0.4107678, 0.2653476, 0.1462075, 0.0782135,
          0.1813836, 0.3356108, 0.2478454, 0.4269330, 0.1745262, 0.3370988, 0.4572209, 0.3415696,
          0.1628898, 0.4335400, 0.1492740, 0.4655451, 0.2777625, 0.1149032, 0.2400093, 0.4063063,
          0.2486204, 0.4209127, 0.4439365, 0.3121863, 0.0742416, 0.0563368, 0.3876660, 0.4610244,
          0.2641133, 0.1591760, 0.1906150, 0.1628898, 0.3853460, 0.4039690, 0.4618716, 0.0398270,
          0.4915100, 0.4278218, 0.1974726, 0.1358538, 0.0068666, 0.4003320, 0.4238202, 0.3394504};
}

// TODO: Find a way to generate and validate test data without hardcoding them (issue #2485)
const std::vector<lanczos_inputs<int, float>> inputsf = {
  {2,
   34,
   10000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::SA,
   42,
   {0,   0,   0,   0,   3,   5,   6,   8,   9,   11,  16,  16,  18,  20,  23,  24,  27,
    30,  31,  33,  37,  37,  39,  41,  43,  44,  46,  46,  47,  49,  50,  50,  51,  53,
    57,  58,  59,  66,  67,  68,  69,  71,  72,  75,  78,  83,  86,  90,  93,  94,  96,
    98,  99,  101, 101, 104, 106, 108, 109, 109, 109, 109, 111, 113, 118, 120, 121, 123,
    124, 128, 132, 134, 136, 138, 139, 141, 145, 148, 151, 152, 154, 155, 157, 160, 164,
    167, 170, 170, 170, 173, 178, 179, 182, 184, 186, 191, 192, 196, 198, 198, 198},
   {44, 68, 74, 16, 36, 85, 34, 75, 61, 51, 83, 15, 33, 55, 69, 71, 18, 84, 70, 95, 71, 83,
    97, 83, 9,  36, 54, 4,  42, 46, 52, 11, 89, 31, 37, 74, 96, 36, 88, 56, 64, 68, 94, 82,
    35, 90, 50, 82, 85, 83, 19, 47, 94, 9,  44, 56, 79, 6,  25, 4,  15, 21, 52, 75, 79, 92,
    19, 72, 94, 94, 96, 80, 16, 54, 89, 46, 48, 63, 3,  33, 67, 73, 77, 46, 47, 75, 16, 43,
    45, 81, 32, 45, 68, 43, 55, 63, 27, 89, 8,  17, 36, 15, 42, 96, 9,  49, 22, 33, 77, 7,
    75, 78, 88, 43, 49, 66, 76, 91, 22, 82, 69, 63, 84, 44, 3,  23, 47, 81, 9,  65, 76, 92,
    12, 96, 9,  13, 38, 93, 44, 3,  19, 6,  36, 45, 61, 63, 69, 89, 44, 57, 94, 62, 33, 36,
    41, 46, 68, 24, 28, 64, 8,  13, 14, 29, 11, 66, 88, 5,  28, 93, 21, 62, 84, 18, 42, 50,
    76, 91, 25, 63, 89, 97, 36, 69, 72, 85, 23, 32, 39, 40, 77, 12, 19, 40, 54, 70, 13, 91},
   {0.4734894, 0.1402491, 0.7686475, 0.0416142, 0.2559651, 0.9360436, 0.7486080, 0.5206724,
    0.0374126, 0.8082515, 0.5993828, 0.4866583, 0.8907925, 0.9251201, 0.8566143, 0.9528994,
    0.4557763, 0.4907070, 0.4158074, 0.8311127, 0.9026024, 0.3103237, 0.5876446, 0.7585195,
    0.4866583, 0.4493615, 0.5909155, 0.0416142, 0.0963910, 0.6722401, 0.3468698, 0.4557763,
    0.1445242, 0.7720124, 0.9923756, 0.1227579, 0.7194629, 0.8916773, 0.4320931, 0.5840980,
    0.0216121, 0.3709223, 0.1705930, 0.8297898, 0.2409706, 0.9585592, 0.3171389, 0.0228039,
    0.4350971, 0.4939908, 0.7720124, 0.2722416, 0.1792683, 0.8907925, 0.1085757, 0.8745620,
    0.3298612, 0.7486080, 0.2409706, 0.2559651, 0.4493615, 0.8916773, 0.5540361, 0.5150571,
    0.9160119, 0.1767728, 0.9923756, 0.5717281, 0.1077409, 0.9368132, 0.6273088, 0.6616613,
    0.0963910, 0.9378265, 0.3059566, 0.3159291, 0.0449106, 0.9085807, 0.4734894, 0.1085757,
    0.2909013, 0.7787509, 0.7168902, 0.9691764, 0.2669757, 0.4389115, 0.6722401, 0.3159291,
    0.9691764, 0.7467896, 0.2722416, 0.2669757, 0.1532843, 0.0449106, 0.2023634, 0.8934466,
    0.3171389, 0.6594226, 0.8082515, 0.3468698, 0.5540361, 0.5909155, 0.9378265, 0.2909178,
    0.9251201, 0.2023634, 0.5840980, 0.8745620, 0.2624605, 0.0374126, 0.1034030, 0.3736577,
    0.3315690, 0.9085807, 0.8934466, 0.5548525, 0.2302140, 0.7827352, 0.0216121, 0.8262919,
    0.1646078, 0.5548525, 0.2658700, 0.2909013, 0.1402491, 0.3709223, 0.1532843, 0.5792196,
    0.8566143, 0.1646078, 0.0827300, 0.5810611, 0.4158074, 0.5188584, 0.9528994, 0.9026024,
    0.5717281, 0.7269946, 0.7787509, 0.7686475, 0.1227579, 0.5206724, 0.5150571, 0.4389115,
    0.1034030, 0.2302140, 0.0827300, 0.8961608, 0.7168902, 0.2624605, 0.4823034, 0.3736577,
    0.3298612, 0.9160119, 0.6616613, 0.7467896, 0.5792196, 0.8297898, 0.0228039, 0.8262919,
    0.5993828, 0.3103237, 0.7585195, 0.4939908, 0.4907070, 0.2658700, 0.0844443, 0.9360436,
    0.4350971, 0.6997072, 0.4320931, 0.3315690, 0.0844443, 0.1445242, 0.3059566, 0.6594226,
    0.8961608, 0.6498466, 0.9585592, 0.7827352, 0.6498466, 0.2812338, 0.1767728, 0.5810611,
    0.7269946, 0.6997072, 0.1705930, 0.1792683, 0.1077409, 0.9368132, 0.4823034, 0.8311127,
    0.7194629, 0.6273088, 0.2909178, 0.5188584, 0.5876446, 0.2812338},
   {-2.0369630, -1.7673520}}};

const std::vector<lanczos_inputs<int, double>> inputsd = {
  {2,
   34,
   10000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::SA,
   42,
   {0,   0,   0,   0,   3,   5,   6,   8,   9,   11,  16,  16,  18,  20,  23,  24,  27,
    30,  31,  33,  37,  37,  39,  41,  43,  44,  46,  46,  47,  49,  50,  50,  51,  53,
    57,  58,  59,  66,  67,  68,  69,  71,  72,  75,  78,  83,  86,  90,  93,  94,  96,
    98,  99,  101, 101, 104, 106, 108, 109, 109, 109, 109, 111, 113, 118, 120, 121, 123,
    124, 128, 132, 134, 136, 138, 139, 141, 145, 148, 151, 152, 154, 155, 157, 160, 164,
    167, 170, 170, 170, 173, 178, 179, 182, 184, 186, 191, 192, 196, 198, 198, 198},
   {44, 68, 74, 16, 36, 85, 34, 75, 61, 51, 83, 15, 33, 55, 69, 71, 18, 84, 70, 95, 71, 83,
    97, 83, 9,  36, 54, 4,  42, 46, 52, 11, 89, 31, 37, 74, 96, 36, 88, 56, 64, 68, 94, 82,
    35, 90, 50, 82, 85, 83, 19, 47, 94, 9,  44, 56, 79, 6,  25, 4,  15, 21, 52, 75, 79, 92,
    19, 72, 94, 94, 96, 80, 16, 54, 89, 46, 48, 63, 3,  33, 67, 73, 77, 46, 47, 75, 16, 43,
    45, 81, 32, 45, 68, 43, 55, 63, 27, 89, 8,  17, 36, 15, 42, 96, 9,  49, 22, 33, 77, 7,
    75, 78, 88, 43, 49, 66, 76, 91, 22, 82, 69, 63, 84, 44, 3,  23, 47, 81, 9,  65, 76, 92,
    12, 96, 9,  13, 38, 93, 44, 3,  19, 6,  36, 45, 61, 63, 69, 89, 44, 57, 94, 62, 33, 36,
    41, 46, 68, 24, 28, 64, 8,  13, 14, 29, 11, 66, 88, 5,  28, 93, 21, 62, 84, 18, 42, 50,
    76, 91, 25, 63, 89, 97, 36, 69, 72, 85, 23, 32, 39, 40, 77, 12, 19, 40, 54, 70, 13, 91},
   {0.4734894, 0.1402491, 0.7686475, 0.0416142, 0.2559651, 0.9360436, 0.7486080, 0.5206724,
    0.0374126, 0.8082515, 0.5993828, 0.4866583, 0.8907925, 0.9251201, 0.8566143, 0.9528994,
    0.4557763, 0.4907070, 0.4158074, 0.8311127, 0.9026024, 0.3103237, 0.5876446, 0.7585195,
    0.4866583, 0.4493615, 0.5909155, 0.0416142, 0.0963910, 0.6722401, 0.3468698, 0.4557763,
    0.1445242, 0.7720124, 0.9923756, 0.1227579, 0.7194629, 0.8916773, 0.4320931, 0.5840980,
    0.0216121, 0.3709223, 0.1705930, 0.8297898, 0.2409706, 0.9585592, 0.3171389, 0.0228039,
    0.4350971, 0.4939908, 0.7720124, 0.2722416, 0.1792683, 0.8907925, 0.1085757, 0.8745620,
    0.3298612, 0.7486080, 0.2409706, 0.2559651, 0.4493615, 0.8916773, 0.5540361, 0.5150571,
    0.9160119, 0.1767728, 0.9923756, 0.5717281, 0.1077409, 0.9368132, 0.6273088, 0.6616613,
    0.0963910, 0.9378265, 0.3059566, 0.3159291, 0.0449106, 0.9085807, 0.4734894, 0.1085757,
    0.2909013, 0.7787509, 0.7168902, 0.9691764, 0.2669757, 0.4389115, 0.6722401, 0.3159291,
    0.9691764, 0.7467896, 0.2722416, 0.2669757, 0.1532843, 0.0449106, 0.2023634, 0.8934466,
    0.3171389, 0.6594226, 0.8082515, 0.3468698, 0.5540361, 0.5909155, 0.9378265, 0.2909178,
    0.9251201, 0.2023634, 0.5840980, 0.8745620, 0.2624605, 0.0374126, 0.1034030, 0.3736577,
    0.3315690, 0.9085807, 0.8934466, 0.5548525, 0.2302140, 0.7827352, 0.0216121, 0.8262919,
    0.1646078, 0.5548525, 0.2658700, 0.2909013, 0.1402491, 0.3709223, 0.1532843, 0.5792196,
    0.8566143, 0.1646078, 0.0827300, 0.5810611, 0.4158074, 0.5188584, 0.9528994, 0.9026024,
    0.5717281, 0.7269946, 0.7787509, 0.7686475, 0.1227579, 0.5206724, 0.5150571, 0.4389115,
    0.1034030, 0.2302140, 0.0827300, 0.8961608, 0.7168902, 0.2624605, 0.4823034, 0.3736577,
    0.3298612, 0.9160119, 0.6616613, 0.7467896, 0.5792196, 0.8297898, 0.0228039, 0.8262919,
    0.5993828, 0.3103237, 0.7585195, 0.4939908, 0.4907070, 0.2658700, 0.0844443, 0.9360436,
    0.4350971, 0.6997072, 0.4320931, 0.3315690, 0.0844443, 0.1445242, 0.3059566, 0.6594226,
    0.8961608, 0.6498466, 0.9585592, 0.7827352, 0.6498466, 0.2812338, 0.1767728, 0.5810611,
    0.7269946, 0.6997072, 0.1705930, 0.1792683, 0.1077409, 0.9368132, 0.4823034, 0.8311127,
    0.7194629, 0.6273088, 0.2909178, 0.5188584, 0.5876446, 0.2812338},
   {-2.0369630, -1.7673520}}};

const std::vector<lanczos_inputs<int, double>> inputsd_SM = {
  {2,
   20,
   100000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::SM,
   42,
   rows<int>(),
   cols<int>(),
   vals<double>(),
   {-0.03944135, 0.01367824}}};

const std::vector<lanczos_inputs<int, double>> inputsd_LM = {
  {2,
   20,
   100000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::LM,
   42,
   rows<int>(),
   cols<int>(),
   vals<double>(),
   {-2.00968758, 3.45939575}}};

const std::vector<lanczos_inputs<int, double>> inputsd_LA = {
  {2,
   20,
   100000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::LA,
   42,
   rows<int>(),
   cols<int>(),
   vals<double>(),
   {1.99482678, 3.45939575}}};

const std::vector<lanczos_inputs<int, double>> inputsd_SA = {
  {2,
   20,
   100000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::SA,
   42,
   rows<int>(),
   cols<int>(),
   vals<double>(),
   {-2.00968758, -1.83487935}}};

const std::vector<lanczos_inputs<int, float>> inputsf_SM = {
  {2,
   20,
   100000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::SM,
   42,
   rows<int>(),
   cols<int>(),
   vals<float>(),
   {-0.03944135, 0.01367824}}};

const std::vector<lanczos_inputs<int, float>> inputsf_LM = {
  {2,
   20,
   100000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::LM,
   42,
   rows<int>(),
   cols<int>(),
   vals<float>(),
   {-2.00968758, 3.45939575}}};

const std::vector<lanczos_inputs<int, float>> inputsf_LA = {
  {2,
   20,
   100000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::LA,
   42,
   rows<int>(),
   cols<int>(),
   vals<float>(),
   {1.99482678, 3.45939575}}};

const std::vector<lanczos_inputs<int, float>> inputsf_SA = {
  {2,
   20,
   100000,
   0,
   0,
   1e-15,
   raft::sparse::solver::LANCZOS_WHICH::SA,
   42,
   rows<int>(),
   cols<int>(),
   vals<float>(),
   {-2.00968758, -1.83487935}}};

const std::vector<rmat_lanczos_inputs<int, float>> rmat_inputsf = {
  {50,
   100,
   10000,
   0,
   0,
   1e-9,
   raft::sparse::solver::LANCZOS_WHICH::SA,
   42,
   12,
   12,
   1,
   {-122.526794, -74.00686,   -59.698284,  -54.68617,   -49.686813, -34.02644,  -32.130703,
    -31.26906,   -30.32097,   -22.946098,  -20.497862,  -20.23817,  -19.269697, -18.42496,
    -17.675667,  -17.013401,  -16.734581,  -15.820215,  -15.73925,  -15.448187, -15.044634,
    -14.692028,  -14.127425,  -13.967386,  -13.6237755, -13.469393, -13.181225, -12.777589,
    -12.623185,  -12.55508,   -12.2874565, -12.053391,  -11.677346, -11.558279, -11.163732,
    -10.922034,  -10.7936945, -10.558049,  -10.205776,  -10.005316, -9.559181,  -9.491834,
    -9.242631,   -8.883637,   -8.765364,   -8.688508,   -8.458255,  -8.385196,  -8.217982,
    -8.0442095}}};

using LanczosTestF = lanczos_tests<int, float>;
TEST_P(LanczosTestF, Result) { Run(); }

using LanczosTestD = lanczos_tests<int, double>;
TEST_P(LanczosTestD, Result) { Run(); }

using LanczosTestD_SM = lanczos_tests<int, double>;
TEST_P(LanczosTestD_SM, Result) { Run(); }

using LanczosTestD_LA = lanczos_tests<int, double>;
TEST_P(LanczosTestD_LA, Result) { Run(); }

using LanczosTestD_LM = lanczos_tests<int, double>;
TEST_P(LanczosTestD_LM, Result) { Run(); }

using LanczosTestD_SA = lanczos_tests<int, double>;
TEST_P(LanczosTestD_SA, Result) { Run(); }

using LanczosTestF_SM = lanczos_tests<int, float>;
TEST_P(LanczosTestF_SM, Result) { Run(); }

using LanczosTestF_LA = lanczos_tests<int, float>;
TEST_P(LanczosTestF_LA, Result) { Run(); }

using LanczosTestF_LM = lanczos_tests<int, float>;
TEST_P(LanczosTestF_LM, Result) { Run(); }

using LanczosTestF_SA = lanczos_tests<int, float>;
TEST_P(LanczosTestF_SA, Result) { Run(); }

using RmatLanczosTestF = rmat_lanczos_tests<int, float>;
TEST_P(RmatLanczosTestF, Result) { Run(); }

INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestF, ::testing::ValuesIn(inputsf));
INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestD, ::testing::ValuesIn(inputsd));
INSTANTIATE_TEST_CASE_P(LanczosTests, RmatLanczosTestF, ::testing::ValuesIn(rmat_inputsf));

INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestD_SM, ::testing::ValuesIn(inputsd_SM));
INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestD_LA, ::testing::ValuesIn(inputsd_LA));
INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestD_LM, ::testing::ValuesIn(inputsd_LM));
INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestD_SA, ::testing::ValuesIn(inputsd_SA));

INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestF_SM, ::testing::ValuesIn(inputsf_SM));
INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestF_LA, ::testing::ValuesIn(inputsf_LA));
INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestF_LM, ::testing::ValuesIn(inputsf_LM));
INSTANTIATE_TEST_CASE_P(LanczosTests, LanczosTestF_SA, ::testing::ValuesIn(inputsf_SA));

}  // namespace raft::sparse::solver
