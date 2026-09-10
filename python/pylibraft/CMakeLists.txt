# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

# We always need CUDA for pylibraft because the raft dependency enables CUDA unconditionally.
include(rapids-cuda)
rapids_cuda_init_architectures(pylibraft)

project(
  pylibraft
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX CUDA
)

# Find C++ libraries from RAPIDS wheels in site-packages.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

# an installed version of raft contains the other necessary targets (like CCCL)
find_package(raft "${RAPIDS_VERSION}" REQUIRED COMPONENTS raft compiled)

include(rapids-cython-core)

rapids_cython_init()

add_subdirectory(pylibraft/common)
add_subdirectory(pylibraft/random)
add_subdirectory(pylibraft/sparse)
