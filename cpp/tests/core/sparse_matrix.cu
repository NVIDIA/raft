/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_utils.cuh"

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>

#include <type_traits>

namespace raft {

/**
 * Example of accepting a value-owning matrix type which doesn't need to adjust sparsity
 */
template <typename S, typename = std::enable_if_t<is_device_csr_matrix_v<S>>>
void test_device_csr_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a csr_matrix which also owns structure and can initialize sparsity
 */
template <typename S, typename = std::enable_if_t<is_device_csr_sparsity_owning_v<S>>>
void test_device_csr_sparsity_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a csr_matrix which does not own the structure
 */
template <typename S, typename = std::enable_if_t<is_device_csr_sparsity_preserving_v<S>>>
void test_device_csr_sparsity_preserving_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a value-owning matrix type which doesn't need to adjust sparsity
 */
template <typename S, typename = std::enable_if_t<is_device_coo_matrix_v<S>>>
void test_device_coo_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a coo_matrix which also owns structure and can initialize sparsity
 */
template <typename S, typename = std::enable_if_t<is_device_coo_sparsity_owning_v<S>>>
void test_device_coo_sparsity_owning_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

/**
 * Example of accepting a coo_matrix which does not own the structure
 */
template <typename S, typename = std::enable_if_t<is_device_coo_sparsity_preserving_v<S>>>
void test_device_coo_sparsity_preserving_ref(S& mat, void* d)
{
  ASSERT_EQ(static_cast<void*>(mat.get_elements().data()), d);
}

void test_device_coo_matrix()
{
  raft::resources handle;
  execute_with_dry_run_check(
    handle,
    [&](raft::resources const& h) {
      auto sparsity_owning = raft::make_device_coo_matrix<float, int, int, int>(h, 5, 5);

      auto structure_view = sparsity_owning.structure_view();

      ASSERT_EQ(structure_view.get_n_cols(), 5);
      ASSERT_EQ(structure_view.get_n_rows(), 5);
      ASSERT_EQ(structure_view.get_nnz(), 0);

      auto coord_struct = raft::make_device_coordinate_structure(h, 5, 5, 5);
      auto sparsity_preserving =
        raft::make_device_coo_matrix<float, int, int>(h, coord_struct.view());

      sparsity_owning.initialize_sparsity(5);

      auto structure_view2 = sparsity_owning.structure_view();

      ASSERT_EQ(structure_view2.get_n_cols(), 5);
      ASSERT_EQ(structure_view2.get_n_rows(), 5);
      ASSERT_EQ(structure_view2.get_nnz(), 5);

      void* d_owning     = static_cast<void*>(sparsity_owning.get_elements().data());
      void* d_preserving = static_cast<void*>(sparsity_preserving.get_elements().data());

      test_device_coo_owning_ref(sparsity_owning, d_owning);
      test_device_coo_owning_ref(sparsity_preserving, d_preserving);

      test_device_coo_sparsity_owning_ref(sparsity_owning, d_owning);
      test_device_coo_sparsity_preserving_ref(sparsity_preserving, d_preserving);
    },
    alloc_behavior::ARGUMENT_DRIVEN,
    4 * 5 * sizeof(int) + 2 * 5 * sizeof(float));
}

void test_device_csr_matrix()
{
  raft::resources handle;
  execute_with_dry_run_check(
    handle,
    [&](raft::resources const& h) {
      auto sparsity_owning = raft::make_device_csr_matrix<float, int, int, int>(h, 5, 5);

      auto comp_struct = raft::make_device_compressed_structure(h, 5, 5, 5);
      auto sparsity_preserving =
        raft::make_device_csr_matrix<float, int, int>(h, comp_struct.view());

      auto structure_view = sparsity_owning.structure_view();

      ASSERT_EQ(structure_view.get_n_cols(), 5);
      ASSERT_EQ(structure_view.get_n_rows(), 5);
      ASSERT_EQ(structure_view.get_nnz(), 0);

      sparsity_owning.initialize_sparsity(5);

      auto structure_view2 = sparsity_owning.structure_view();

      ASSERT_EQ(structure_view2.get_n_cols(), 5);
      ASSERT_EQ(structure_view2.get_n_rows(), 5);
      ASSERT_EQ(structure_view2.get_nnz(), 5);

      void* d_owning     = static_cast<void*>(sparsity_owning.get_elements().data());
      void* d_preserving = static_cast<void*>(sparsity_preserving.get_elements().data());

      test_device_csr_owning_ref(sparsity_owning, d_owning);
      test_device_csr_owning_ref(sparsity_preserving, d_preserving);

      test_device_csr_sparsity_owning_ref(sparsity_owning, d_owning);
      test_device_csr_sparsity_preserving_ref(sparsity_preserving, d_preserving);
    },
    alloc_behavior::ARGUMENT_DRIVEN,
    2 * (5 + 1) * sizeof(int) + 2 * 5 * sizeof(int) + 2 * 5 * sizeof(float));
}

TEST(DeviceCoordinateStructure, Initialization)
{
  raft::resources handle;
  execute_with_dry_run_check(
    handle,
    [&](raft::resources const& h) {
      auto uninitialized = raft::make_device_coordinate_structure(h, 5, 5, 0);
      EXPECT_EQ(uninitialized.view().get_rows().size(), 0);
      EXPECT_EQ(uninitialized.view().get_rows().data(), nullptr);

      auto initialized = raft::make_device_coordinate_structure(h, 5, 5, 5);
      EXPECT_EQ(initialized.view().get_rows().size(), 5);
      EXPECT_NE(initialized.view().get_rows().data(), nullptr);
    },
    alloc_behavior::ARGUMENT_DRIVEN,
    2 * 5 * sizeof(int));
}

TEST(DeviceSparseCOOMatrix, Basic) { test_device_coo_matrix(); }

TEST(DeviceSparseCSRMatrix, Basic) { test_device_csr_matrix(); }

}  // namespace raft
