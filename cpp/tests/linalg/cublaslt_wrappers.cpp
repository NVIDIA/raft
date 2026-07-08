/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/linalg/detail/cublaslt_wrappers.hpp>

#include <gtest/gtest.h>

#include <utility>

namespace raft::linalg::detail {

TEST(Raft, CublasLtMatrixLayoutMoveConstructorTransfersOwnership)
{
  auto layout = cublastlt_matrix_layout{CUDA_R_32F, 2, 3, 2};
  auto raw    = static_cast<cublasLtMatrixLayout_t>(layout);

  ASSERT_NE(raw, nullptr);

  auto moved = std::move(layout);

  EXPECT_EQ(static_cast<cublasLtMatrixLayout_t>(layout), nullptr);
  EXPECT_EQ(static_cast<cublasLtMatrixLayout_t>(moved), raw);
}

TEST(Raft, CublasLtMatrixLayoutMoveAssignmentTransfersOwnership)
{
  auto src     = cublastlt_matrix_layout{CUDA_R_32F, 2, 3, 2};
  auto src_raw = static_cast<cublasLtMatrixLayout_t>(src);
  auto dst     = cublastlt_matrix_layout{CUDA_R_16F, 4, 5, 4};
  auto dst_raw = static_cast<cublasLtMatrixLayout_t>(dst);

  ASSERT_NE(src_raw, nullptr);
  ASSERT_NE(dst_raw, nullptr);

  dst = std::move(src);

  EXPECT_EQ(static_cast<cublasLtMatrixLayout_t>(src), dst_raw);
  EXPECT_EQ(static_cast<cublasLtMatrixLayout_t>(dst), src_raw);
}

TEST(Raft, CublasLtMatmulDescMoveConstructorTransfersOwnership)
{
  auto desc = cublastlt_matmul_desc{CUBLAS_COMPUTE_32F, CUDA_R_32F};
  auto raw  = static_cast<cublasLtMatmulDesc_t>(desc);

  ASSERT_NE(raw, nullptr);

  auto moved = std::move(desc);

  EXPECT_EQ(static_cast<cublasLtMatmulDesc_t>(desc), nullptr);
  EXPECT_EQ(static_cast<cublasLtMatmulDesc_t>(moved), raw);
}

TEST(Raft, CublasLtMatmulDescMoveAssignmentTransfersOwnership)
{
  auto src     = cublastlt_matmul_desc{CUBLAS_COMPUTE_32F, CUDA_R_32F};
  auto src_raw = static_cast<cublasLtMatmulDesc_t>(src);
  auto dst     = cublastlt_matmul_desc{CUBLAS_COMPUTE_16F, CUDA_R_16F};
  auto dst_raw = static_cast<cublasLtMatmulDesc_t>(dst);

  ASSERT_NE(src_raw, nullptr);
  ASSERT_NE(dst_raw, nullptr);

  dst = std::move(src);

  EXPECT_EQ(static_cast<cublasLtMatmulDesc_t>(src), dst_raw);
  EXPECT_EQ(static_cast<cublasLtMatmulDesc_t>(dst), src_raw);
}

}  // namespace raft::linalg::detail
