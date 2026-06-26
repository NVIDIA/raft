# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
include(../cmake/rapids_config.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-export)
include(rapids-find)

# workaround for rapids_cuda_init_architectures not working for arch detection with
# enable_language(CUDA)
set(lang_list "CXX")

include(rapids-cuda)
rapids_cuda_init_architectures(RAFT)
list(APPEND lang_list "CUDA")

project(
  RAFT
  VERSION "${RAPIDS_VERSION}"
  LANGUAGES ${lang_list}
)

# Disable C++20 module scanning (not using modules, just C++20 features)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Write the version header
rapids_cmake_write_version_file(include/raft/version_config.hpp)

# ##################################################################################################
# * build type ---------------------------------------------------------------

# Set a default build type if none was specified
rapids_cmake_build_type(Release)

# this is needed for clang-tidy runs
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ##################################################################################################
# * User Options  ------------------------------------------------------------

option(BUILD_SHARED_LIBS "Build raft shared libraries" ON)
option(BUILD_TESTS "Build raft unit-tests" ON)
option(BUILD_PRIMS_BENCH "Build raft C++ benchmark tests" OFF)
option(CUDA_ENABLE_KERNELINFO "Enable kernel resource usage info" OFF)
option(CUDA_ENABLE_LINEINFO
       "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler)" OFF
)
option(CUDA_STATIC_MATH_LIBRARIES "Statically link the CUDA math libraries" OFF)
option(CUDA_LOG_COMPILE_TIME "Write a log of compilation times to nvcc_compile_log.csv" OFF)
option(DETECT_CONDA_ENV "Enable detection of conda environment for dependencies" ON)
option(DISABLE_DEPRECATION_WARNINGS "Disable deprecaction warnings " ON)
option(DISABLE_OPENMP "Disable OpenMP" OFF)
option(RAFT_NVTX "Enable nvtx markers" OFF)

set(RAFT_COMPILE_LIBRARY_DEFAULT OFF)
if(BUILD_TESTS OR BUILD_PRIMS_BENCH)
  set(RAFT_COMPILE_LIBRARY_DEFAULT ON)
endif()
option(RAFT_COMPILE_LIBRARY "Enable building raft library instantiations"
       ${RAFT_COMPILE_LIBRARY_DEFAULT}
)
option(RAFT_COMPILE_DYNAMIC_ONLY "Only build the shared library and skip the
static library. Has no effect if RAFT_COMPILE_LIBRARY is OFF" OFF
)

# Needed because GoogleBenchmark changes the state of FindThreads.cmake, causing subsequent runs to
# have different values for the `Threads::Threads` target. Setting this flag ensures
# `Threads::Threads` is the same value across all builds so that cache hits occur
set(THREADS_PREFER_PTHREAD_FLAG ON)

include(CMakeDependentOption)
# cmake_dependent_option( RAFT_USE_FAISS_STATIC "Build and statically link the FAISS library for
# nearest neighbors search on GPU" ON RAFT_COMPILE_LIBRARY OFF )

message(VERBOSE "RAFT: Building optional components: ${raft_FIND_COMPONENTS}")
message(VERBOSE "RAFT: Build RAFT unit-tests: ${BUILD_TESTS}")
message(VERBOSE "RAFT: Building raft C++ benchmarks: ${BUILD_PRIMS_BENCH}")
message(VERBOSE "RAFT: Enable detection of conda environment for dependencies: ${DETECT_CONDA_ENV}")
message(VERBOSE "RAFT: Disable depreaction warnings " ${DISABLE_DEPRECATION_WARNINGS})
message(VERBOSE "RAFT: Disable OpenMP: ${DISABLE_OPENMP}")
message(VERBOSE "RAFT: Enable kernel resource usage info: ${CUDA_ENABLE_KERNELINFO}")
message(VERBOSE "RAFT: Enable lineinfo in nvcc: ${CUDA_ENABLE_LINEINFO}")
message(VERBOSE "RAFT: Enable nvtx markers: ${RAFT_NVTX}")
message(VERBOSE "RAFT: Statically link the CUDA math libraries: ${CUDA_STATIC_MATH_LIBRARIES}")

# Set RMM logging level
set(RMM_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE RMM_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL" "OFF"
)
message(VERBOSE "RAFT: RMM_LOGGING_LEVEL = '${RMM_LOGGING_LEVEL}'.")

# Set logging level
set(LIBRAFT_LOGGING_LEVEL
    "INFO"
    CACHE STRING "Choose the logging level."
)
set_property(
  CACHE LIBRAFT_LOGGING_LEVEL PROPERTY STRINGS "TRACE" "DEBUG" "INFO" "WARN" "ERROR" "CRITICAL"
                                       "OFF"
)
message(VERBOSE "RAFT: LIBRAFT_LOGGING_LEVEL = '${LIBRAFT_LOGGING_LEVEL}'.")

# ##################################################################################################
# * Conda environment detection ----------------------------------------------

if(DETECT_CONDA_ENV)
  rapids_cmake_support_conda_env(conda_env MODIFY_PREFIX_PATH)
  if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX})
    message(
      STATUS "RAFT: No CMAKE_INSTALL_PREFIX argument detected, setting to: $ENV{CONDA_PREFIX}"
    )
    set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}")
  endif()
endif()

# ##################################################################################################
# * compiler options ----------------------------------------------------------

set(_ctk_static_suffix "")
if(CUDA_STATIC_MATH_LIBRARIES)
  set(_ctk_static_suffix "_static")
endif()

# CUDA runtime
rapids_cuda_init_runtime(USE_STATIC ON)
# * find CUDAToolkit package
# * determine GPU architectures
# * enable the CMake CUDA language
# * set other CUDA compilation flags
rapids_find_package(
  CUDAToolkit REQUIRED
  BUILD_EXPORT_SET raft-exports
  INSTALL_EXPORT_SET raft-exports
)

if(NOT DISABLE_OPENMP)
  rapids_find_package(
    OpenMP REQUIRED
    BUILD_EXPORT_SET raft-exports
    INSTALL_EXPORT_SET raft-exports
  )
  if(OPENMP_FOUND)
    message(VERBOSE "RAFT: OpenMP found in ${OpenMP_CXX_INCLUDE_DIRS}")
  endif()
endif()

include(cmake/modules/ConfigureCUDA.cmake)

# ##################################################################################################
# * Requirements -------------------------------------------------------------

# add third party dependencies using CPM
rapids_cpm_init()

include(${rapids-cmake-dir}/cpm/rapids_logger.cmake)
rapids_cpm_rapids_logger(BUILD_EXPORT_SET raft-exports INSTALL_EXPORT_SET raft-exports)
create_logger_macros(RAFT "raft::default_logger()" include/raft/core)

# CCCL before rmm so we get the right version of CCCL
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)

if(BUILD_TESTS)
  include(${rapids-cmake-dir}/cpm/gtest.cmake)
  rapids_cpm_gtest(BUILD_STATIC)
endif()

if(BUILD_PRIMS_BENCH)
  include(${rapids-cmake-dir}/cpm/gbench.cmake)
  rapids_cpm_gbench(BUILD_STATIC)
endif()

# ##################################################################################################
# * raft ---------------------------------------------------------------------
add_library(raft INTERFACE)
add_library(raft::raft ALIAS raft)

target_include_directories(
  raft INTERFACE "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/include>"
                 "$<BUILD_INTERFACE:${RAFT_BINARY_DIR}/include>" "$<INSTALL_INTERFACE:include>"
)

# Keep RAFT as lightweight as possible. Only CUDA libs and rmm should be used in global target.
target_link_libraries(raft INTERFACE rapids_logger::rapids_logger rmm::rmm CCCL::CCCL)

target_compile_features(raft INTERFACE cxx_std_20 $<BUILD_INTERFACE:cuda_std_20>)
target_compile_options(
  raft INTERFACE $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-extended-lambda
                 --expt-relaxed-constexpr>
)
target_compile_definitions(
  raft INTERFACE "RAFT_LOG_ACTIVE_LEVEL=RAPIDS_LOGGER_LOG_LEVEL_${LIBRAFT_LOGGING_LEVEL}"
)

set(RAFT_CUSOLVER_DEPENDENCY CUDA::cusolver${_ctk_static_suffix})
set(RAFT_CUBLAS_DEPENDENCY CUDA::cublas${_ctk_static_suffix})
set(RAFT_CURAND_DEPENDENCY CUDA::curand${_ctk_static_suffix})
set(RAFT_CUSPARSE_DEPENDENCY CUDA::cusparse${_ctk_static_suffix})

set(RAFT_CTK_MATH_DEPENDENCIES ${RAFT_CUBLAS_DEPENDENCY} ${RAFT_CUSOLVER_DEPENDENCY}
                               ${RAFT_CUSPARSE_DEPENDENCY} ${RAFT_CURAND_DEPENDENCY}
)

# Endian detection
include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
  target_compile_definitions(raft INTERFACE RAFT_SYSTEM_LITTLE_ENDIAN=0)
else()
  target_compile_definitions(raft INTERFACE RAFT_SYSTEM_LITTLE_ENDIAN=1)
endif()

if(RAFT_COMPILE_LIBRARY)
  file(
    WRITE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld"
    [=[
SECTIONS
{
.nvFatBinSegment : { *(.nvFatBinSegment) }
.nv_fatbin : { *(.nv_fatbin) }
}
]=]
  )
endif()

# ##################################################################################################
# * NVTX support in raft -----------------------------------------------------

if(RAFT_NVTX)
  # This enables NVTX within the project with no option to disable it downstream.
  target_link_libraries(raft INTERFACE CUDA::nvtx3)
  target_compile_definitions(raft INTERFACE NVTX_ENABLED)
else()
  # Allow enable NVTX downstream if not set here. This creates a new option at build/install time,
  # which is set by default to OFF, but can be enabled in the dependent project.
  get_property(
    nvtx_option_help_string
    CACHE RAFT_NVTX
    PROPERTY HELPSTRING
  )
  string(
    CONCAT
      nvtx_export_string
      "option(RAFT_NVTX \""
      ${nvtx_option_help_string}
      "\" OFF)"
      [=[

target_link_libraries(raft::raft INTERFACE $<$<BOOL:${RAFT_NVTX}>:CUDA::nvtx3>)
target_compile_definitions(raft::raft INTERFACE $<$<BOOL:${RAFT_NVTX}>:NVTX_ENABLED>)

  ]=]
  )
endif()

# ##################################################################################################
# * raft_compiled ------------------------------------------------------------
add_library(raft_compiled INTERFACE)

if(TARGET raft_compiled AND (NOT TARGET raft::compiled))
  add_library(raft::compiled ALIAS raft_compiled)
endif()

set_target_properties(raft_compiled PROPERTIES EXPORT_NAME compiled)

if(RAFT_COMPILE_LIBRARY)
  add_library(
    raft_objs OBJECT
    src/raft_runtime/random/rmat_rectangular_generator_int64_double.cu
    src/raft_runtime/random/rmat_rectangular_generator_int64_float.cu
    src/raft_runtime/random/rmat_rectangular_generator_int_double.cu
    src/raft_runtime/random/rmat_rectangular_generator_int_float.cu
    src/raft_runtime/solver/lanczos_solver_int64_double.cu
    src/raft_runtime/solver/lanczos_solver_int64_float.cu
    src/raft_runtime/solver/lanczos_solver_int_double.cu
    src/raft_runtime/solver/lanczos_solver_int_float.cu
    src/raft_runtime/solver/randomized_svds_float.cu
    src/raft_runtime/solver/randomized_svds_double.cu
  )
  set_target_properties(
    raft_objs
    PROPERTIES CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
               POSITION_INDEPENDENT_CODE ON
  )
  set_target_properties(
    raft_objs
    PROPERTIES CXX_VISIBILITY_PRESET hidden
               CUDA_VISIBILITY_PRESET hidden
               VISIBILITY_INLINES_HIDDEN ON
  )

  target_compile_definitions(raft_objs PRIVATE "RAFT_EXPLICIT_INSTANTIATE_ONLY")
  target_compile_options(
    raft_objs PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
                      "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )

  add_library(raft_lib SHARED $<TARGET_OBJECTS:raft_objs>)

  set(_raft_lib_targets raft_lib)
  if(NOT RAFT_COMPILE_DYNAMIC_ONLY)
    add_library(raft_lib_static STATIC $<TARGET_OBJECTS:raft_objs>)
    list(APPEND _raft_lib_targets raft_lib_static)
  endif()

  set_target_properties(
    ${_raft_lib_targets}
    PROPERTIES OUTPUT_NAME raft
               BUILD_RPATH "\$ORIGIN"
               INSTALL_RPATH "\$ORIGIN"
               INTERFACE_POSITION_INDEPENDENT_CODE ON
  )

  list(APPEND _raft_lib_targets raft_objs)
  foreach(target IN LISTS _raft_lib_targets)
    target_link_libraries(
      ${target}
      PUBLIC raft::raft
             ${RAFT_CTK_MATH_DEPENDENCIES} # TODO: Once `raft::resources` is used everywhere, this
                                           # will just be cublas
             $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
    )

    # So consumers know when using libraft.so/libraft.a
    target_compile_definitions(${target} PUBLIC "RAFT_COMPILED")
    # ensure CUDA symbols aren't relocated to the middle of the debug build binaries
    target_link_options(${target} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/fatbin.ld")
  endforeach()
endif()

if(TARGET raft_lib AND (NOT TARGET raft::raft_lib))
  add_library(raft::raft_lib ALIAS raft_lib)
endif()

target_link_libraries(raft_compiled INTERFACE raft::raft $<TARGET_NAME_IF_EXISTS:raft::raft_lib>)

# ##################################################################################################
# * raft_compiled_static----------------------------------------------------------------------------

if(NOT RAFT_COMPILE_DYNAMIC_ONLY)
  add_library(raft_compiled_static INTERFACE)

  if(TARGET raft_compiled_static AND (NOT TARGET raft::compiled_static))
    add_library(raft::compiled_static ALIAS raft_compiled_static)
  endif()
  set_target_properties(raft_compiled_static PROPERTIES EXPORT_NAME compiled_static)

  if(TARGET raft_lib_static AND (NOT TARGET raft::raft_lib_static))
    add_library(raft::raft_lib_static ALIAS raft_lib_static)
  endif()

  target_link_libraries(
    raft_compiled_static INTERFACE raft::raft $<TARGET_NAME_IF_EXISTS:raft::raft_lib_static>
  )
endif()

# ##################################################################################################
# * raft_distributed -------------------------------------------------------------------------------
add_library(raft_distributed INTERFACE)

if(TARGET raft_distributed AND (NOT TARGET raft::distributed))
  add_library(raft::distributed ALIAS raft_distributed)
endif()

set_target_properties(raft_distributed PROPERTIES EXPORT_NAME distributed)

rapids_find_generate_module(
  NCCL
  HEADER_NAMES nccl.h
  LIBRARY_NAMES nccl
  BUILD_EXPORT_SET raft-distributed-exports
  INSTALL_EXPORT_SET raft-distributed-exports
)

rapids_export_package(
  BUILD ucxx raft-distributed-exports COMPONENTS ucxx python GLOBAL_TARGETS ucxx::ucxx ucxx::python
)
rapids_export_package(
  INSTALL ucxx raft-distributed-exports COMPONENTS ucxx python GLOBAL_TARGETS ucxx::ucxx
                                                                              ucxx::python
)
rapids_export_package(BUILD NCCL raft-distributed-exports)
rapids_export_package(INSTALL NCCL raft-distributed-exports)

# ucx is a requirement for raft_distributed, but its config is not safe to be found multiple times,
# so rather than exporting a package dependency on it above we rely on consumers to find it
# themselves. Once https://github.com/rapidsai/ucxx/issues/173 is resolved we can export it above
# again.
target_link_libraries(raft_distributed INTERFACE ucx::ucp ucxx::ucxx NCCL::NCCL)

# ##################################################################################################
# * install targets-----------------------------------------------------------
rapids_cmake_install_lib_dir(lib_dir)
include(GNUInstallDirs)
include(CPack)

install(
  TARGETS raft
  DESTINATION ${lib_dir}
  COMPONENT raft
  EXPORT raft-exports
)

set(_raft_compiled_install_targets raft_compiled)
if(NOT RAFT_COMPILE_DYNAMIC_ONLY)
  list(APPEND _raft_compiled_install_targets raft_compiled_static)
endif()
install(
  TARGETS ${_raft_compiled_install_targets}
  DESTINATION ${lib_dir}
  COMPONENT raft
  EXPORT raft-compiled-exports
)

if(TARGET raft_lib)
  install(
    TARGETS raft_lib
    DESTINATION ${lib_dir}
    COMPONENT compiled
    EXPORT raft-compiled-lib-exports
  )
  if(NOT RAFT_COMPILE_DYNAMIC_ONLY)
    install(
      TARGETS raft_lib_static
      DESTINATION ${lib_dir}
      COMPONENT compiled-static
      EXPORT raft-compiled-static-lib-exports
    )
  endif()
  install(
    DIRECTORY include/raft_runtime
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    COMPONENT compiled
  )
endif()

install(
  TARGETS raft_distributed
  DESTINATION ${lib_dir}
  COMPONENT distributed
  EXPORT raft-distributed-exports
)

install(
  DIRECTORY include/raft
  COMPONENT raft
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/include/raft/version_config.hpp
  COMPONENT raft
  DESTINATION include/raft
)

# ##################################################################################################
# * install export -----------------------------------------------------------
set(doc_string
    [=[
Provide targets for the RAFT: Reusable Accelerated Functions and Tools

RAFT contains fundamental widely-used algorithms and primitives
for data science and machine learning.

Optional Components:
  - compiled
  - compiled_static
  - distributed

Imported Targets:
  - raft::raft
  - raft::compiled brought in by the `compiled` optional component
  - raft::compiled_static brought in by the `compiled_static` optional component
  - raft::distributed brought in by the `distributed` optional component

]=]
)

set(code_string ${nvtx_export_string})

string(
  APPEND
  code_string
  [=[
if(compiled IN_LIST raft_FIND_COMPONENTS)
  enable_language(CUDA)
endif()
]=]
)
set(raft_components compiled distributed)
set(raft_export_sets raft-compiled-exports raft-distributed-exports)
if(TARGET raft_lib)
  list(APPEND raft_components compiled)
  list(APPEND raft_export_sets raft-compiled-lib-exports)
  if(NOT RAFT_COMPILE_DYNAMIC_ONLY)
    list(APPEND raft_components compiled-static)
    list(APPEND raft_export_sets raft-compiled-static-lib-exports)
  endif()
endif()

string(
  APPEND
  code_string
  [=[
 option(RAFT_ENABLE_CUSOLVER_DEPENDENCY "Enable cusolver dependency" ON)
 option(RAFT_ENABLE_CUBLAS_DEPENDENCY "Enable cublas dependency" ON)
 option(RAFT_ENABLE_CURAND_DEPENDENCY "Enable curand dependency" ON)
 option(RAFT_ENABLE_CUSPARSE_DEPENDENCY "Enable cusparse dependency" ON)

mark_as_advanced(RAFT_ENABLE_CUSOLVER_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CUBLAS_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CURAND_DEPENDENCY)
mark_as_advanced(RAFT_ENABLE_CUSPARSE_DEPENDENCY)

target_link_libraries(raft::raft INTERFACE
  $<$<BOOL:${RAFT_ENABLE_CUSOLVER_DEPENDENCY}>:${RAFT_CUSOLVER_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CUBLAS_DEPENDENCY}>:${RAFT_CUBLAS_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CUSPARSE_DEPENDENCY}>:${RAFT_CUSPARSE_DEPENDENCY}>
  $<$<BOOL:${RAFT_ENABLE_CURAND_DEPENDENCY}>:${RAFT_CURAND_DEPENDENCY}>
)
]=]
)

# Use `rapids_export` for 22.04 as it will have COMPONENT support
rapids_export(
  INSTALL raft
  EXPORT_SET raft-exports
  COMPONENTS ${raft_components}
  COMPONENTS_EXPORT_SET ${raft_export_sets}
  GLOBAL_TARGETS raft compiled distributed
  NAMESPACE raft::
  DOCUMENTATION doc_string
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# * build export -------------------------------------------------------------
rapids_export(
  BUILD raft
  EXPORT_SET raft-exports
  COMPONENTS ${raft_components}
  COMPONENTS_EXPORT_SET ${raft_export_sets}
  GLOBAL_TARGETS raft compiled distributed
  DOCUMENTATION doc_string
  NAMESPACE raft::
  FINAL_CODE_BLOCK code_string
)

# ##################################################################################################
# * shared test/bench headers ------------------------------------------------

if(BUILD_TESTS OR BUILD_PRIMS_BENCH)
  add_subdirectory(internal)
endif()

# ##################################################################################################
# * build test executable ----------------------------------------------------

if(BUILD_TESTS)
  # include CTest module -- automatically calls enable_testing()
  include(CTest)
  add_subdirectory(tests)
endif()

# ##################################################################################################
# * build benchmark executable -----------------------------------------------

if(BUILD_PRIMS_BENCH)
  add_subdirectory(bench/prims/)
endif()
