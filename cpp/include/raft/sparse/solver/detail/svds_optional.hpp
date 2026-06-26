/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <optional>

namespace raft::sparse::solver::detail {

// Helper alias used to put optional U/Vt parameters in a non-deduced context
// so template argument deduction picks up the value type from earlier parameters and
// implicit conversion from `device_matrix_view` to `std::optional<...>` works.
template <typename T>
struct nondeduced {
  using type = T;
};

template <typename T>
using nondeduced_optional_matrix_view_t = std::optional<typename nondeduced<T>::type>;

}  // namespace raft::sparse::solver::detail
