# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(BUILD_TESTS OR BUILD_PRIMS_BENCH)
  add_library(raft_internal INTERFACE)
  target_include_directories(
    raft_internal INTERFACE "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/internal>"
  )
  target_compile_features(raft_internal INTERFACE cxx_std_20 $<BUILD_INTERFACE:cuda_std_20>)
endif()
