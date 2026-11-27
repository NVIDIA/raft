/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
namespace raft {
namespace linalg {

/**
 * @brief Enum to select the upper or lower triangular part of a matrix
 *
 */
enum class FillMode { UPPER, LOWER };

/**
 * @brief Enum for this type indicates which operation is applied to the related input (e.g. sparse
 * matrix, or vector).
 *
 */
enum class Operation { NON_TRANSPOSE, TRANSPOSE };

}  // namespace linalg
}  // namespace raft
