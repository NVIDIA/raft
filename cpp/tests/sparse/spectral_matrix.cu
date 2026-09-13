/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

namespace raft {
namespace spectral {
namespace matrix {
namespace {
template <typename index_type, typename value_type>
struct csr_view_t {
  index_type* offsets;
  index_type* indices;
  value_type* edge_data;
  index_type number_of_vertices;
  index_type number_of_edges;
};
}  // namespace
TEST(Raft, SpectralMatrices)
{
  using index_type = int;
  using value_type = double;
  using nnz_type   = uint64_t;

  raft::resources h;
  ASSERT_EQ(0, raft::resource::get_device_id(h));

  csr_view_t<index_type, value_type> csr_v{nullptr, nullptr, nullptr, 0, 0};

  int const sz = 10;
  vector_t<index_type> d_v{h, sz};

  index_type* ro{nullptr};
  index_type* ci{nullptr};
  value_type* vs{nullptr};
  nnz_type nnz     = 0;
  index_type nrows = 0;
  sparse_matrix_t<index_type, value_type, nnz_type> sm1{h, ro, ci, vs, nrows, nnz};
  sparse_matrix_t<index_type, value_type, nnz_type> sm2{h, csr_v};
  ASSERT_EQ(nullptr, sm1.row_offsets_);
  ASSERT_EQ(nullptr, sm2.row_offsets_);

  auto cnstr_lm1 = [&h, ro, ci, vs, nrows, nnz](void) {
    laplacian_matrix_t<index_type, value_type, nnz_type> lm1{h, ro, ci, vs, nrows, nnz};
  };
  EXPECT_ANY_THROW(cnstr_lm1());  // because of nullptr ptr args

  auto cnstr_lm2 = [&h, &sm2](void) {
    laplacian_matrix_t<index_type, value_type, nnz_type> lm2{h, sm2};
  };
  EXPECT_ANY_THROW(cnstr_lm2());  // because of nullptr ptr args

  auto cnstr_mm1 = [&h, ro, ci, vs, nrows, nnz](void) {
    modularity_matrix_t<index_type, value_type, nnz_type> mm1{h, ro, ci, vs, nrows, nnz};
  };
  EXPECT_ANY_THROW(cnstr_mm1());  // because of nullptr ptr args

  auto cnstr_mm2 = [&h, &sm2](void) {
    modularity_matrix_t<index_type, value_type, nnz_type> mm2{h, sm2};
  };
  EXPECT_ANY_THROW(cnstr_mm2());  // because of nullptr ptr args
}

}  // namespace matrix
}  // namespace spectral
}  // namespace raft
