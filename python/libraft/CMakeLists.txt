# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

cmake_minimum_required(VERSION 4.0 FATAL_ERROR)

include(../../cmake/rapids_config.cmake)

project(
  libraft-python
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES CXX
)

# Find C++ libraries from RAPIDS wheels in site-packages.
set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE)

# Check if raft is already available. If so, it is the user's responsibility to ensure that the
# CMake package is also available at build time of the Python raft package.
find_package(raft "${RAPIDS_VERSION}")

if(raft_FOUND)
  return()
endif()

unset(raft_FOUND)

# --- CUDA --- #
set(CUDA_STATIC_MATH_LIBRARIES OFF)

# --- RAFT ---#
set(BUILD_TESTS OFF)
set(BUILD_PRIMS_BENCH OFF)
set(RAFT_COMPILE_DYNAMIC_ONLY ON)
set(RAFT_COMPILE_LIBRARY ON)

add_subdirectory(../../cpp raft-cpp)

# assumes libraft.so is installed 2 levels deep, e.g. site-packages/libraft/lib64/libraft.so
set_property(
  TARGET raft_lib
  PROPERTY INSTALL_RPATH
           "$ORIGIN/../../nvidia/cublas/lib"
           "$ORIGIN/../../nvidia/curand/lib"
           "$ORIGIN/../../nvidia/cusolver/lib"
           "$ORIGIN/../../nvidia/cusparse/lib"
           "$ORIGIN/../../nvidia/nvjitlink/lib"
           # starting with CTK 13 wheels, all libraries are grouped in this 'nvidia/cu13/lib'
           # directory
           "$ORIGIN/../../nvidia/cu13/lib"
)
