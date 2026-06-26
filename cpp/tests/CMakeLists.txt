# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

# ##################################################################################################
# enable testing ################################################################################
# ##################################################################################################
enable_testing()
include(rapids-test)
rapids_test_init()

function(ConfigureTest)

  set(options OPTIONAL LIB EXPLICIT_INSTANTIATE_ONLY NOCUDA)
  set(oneValueArgs NAME GPUS PERCENT)
  set(multiValueArgs PATH TARGETS CONFIGURATIONS)

  cmake_parse_arguments(_RAFT_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(NOT DEFINED _RAFT_TEST_GPUS AND NOT DEFINED _RAFT_TEST_PERCENT)
    set(_RAFT_TEST_GPUS 1)
    set(_RAFT_TEST_PERCENT 30)
  endif()
  if(NOT DEFINED _RAFT_TEST_GPUS)
    set(_RAFT_TEST_GPUS 1)
  endif()
  if(NOT DEFINED _RAFT_TEST_PERCENT)
    set(_RAFT_TEST_PERCENT 100)
  endif()

  if(_RAFT_TEST_NOCUDA)
    set(TEST_NAME "${_RAFT_TEST_NAME}_NOCUDA")
  else()
    set(TEST_NAME ${_RAFT_TEST_NAME})
  endif()

  add_executable(${TEST_NAME} ${_RAFT_TEST_PATH})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE raft
            raft_internal
            $<$<BOOL:${_RAFT_TEST_LIB}>:raft::compiled>
            GTest::gtest
            GTest::gtest_main
            Threads::Threads
            ${RAFT_CTK_MATH_DEPENDENCIES}
            $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
            $<TARGET_NAME_IF_EXISTS:conda_env>
  )
  set_target_properties(
    ${TEST_NAME}
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY "$<BUILD_INTERFACE:${RAFT_BINARY_DIR}/gtests>"
               INSTALL_RPATH "\$ORIGIN/../../../lib"
               CXX_STANDARD 20
               CXX_STANDARD_REQUIRED ON
               CUDA_STANDARD 20
               CUDA_STANDARD_REQUIRED ON
  )
  target_compile_options(
    ${TEST_NAME} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${RAFT_CXX_FLAGS}>"
                         "$<$<COMPILE_LANGUAGE:CUDA>:${RAFT_CUDA_FLAGS}>"
  )
  if(_RAFT_TEST_EXPLICIT_INSTANTIATE_ONLY)
    target_compile_definitions(${TEST_NAME} PRIVATE "RAFT_EXPLICIT_INSTANTIATE_ONLY")
  endif()
  if(_RAFT_TEST_NOCUDA)
    target_compile_definitions(${TEST_NAME} PRIVATE "RAFT_DISABLE_CUDA")
  endif()

  target_include_directories(${TEST_NAME} PUBLIC "$<BUILD_INTERFACE:${RAFT_SOURCE_DIR}/tests>")

  rapids_test_add(
    NAME ${TEST_NAME}
    COMMAND ${TEST_NAME}
    GPUS ${_RAFT_TEST_GPUS}
    PERCENT ${_RAFT_TEST_PERCENT}
    INSTALL_COMPONENT_SET testing
  )
endfunction()

# ##################################################################################################
# test sources ##################################################################################
# ##################################################################################################
if(BUILD_TESTS)
  ConfigureTest(
    NAME
    CORE_TEST
    PATH
    core/allocation_tracking.cpp
    core/memory_stats_resources.cpp
    core/monitor_resources.cu
    core/bitmap.cu
    core/bitset.cu
    core/device_resources_manager.cpp
    core/device_setter.cpp
    core/math_device.cu
    core/math_host.cpp
    core/operators_device.cu
    core/operators_host.cpp
    core/handle.cpp
    core/interruptible.cu
    core/nvtx.cpp
    core/mdarray.cu
    core/mdbuffer.cu
    core/mdspan_copy.cpp
    core/mdspan_copy.cu
    core/mdspan_utils.cu
    core/numpy_serializer.cu
    core/memory_type.cpp
    core/sparse_matrix.cu
    core/sparse_matrix.cpp
    core/span.cpp
    core/span.cu
    core/stream_view.cpp
    core/temporary_device_buffer.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME CORE_TEST PATH core/stream_view.cpp core/mdspan_copy.cpp core/host_memory_resource.cpp
    core/allocation_tracking.cpp LIB EXPLICIT_INSTANTIATE_ONLY NOCUDA
  )

  list(APPEND EXT_HEADER_TEST_SOURCES ext_headers/raft_core_logger.cpp)

  # Test that the split headers compile in isolation with:
  #
  # * EXT_HEADERS_TEST_COMPILED_EXPLICIT: RAFT_COMPILED, RAFT_EXPLICIT_INSTANTIATE_ONLY defined
  # * EXT_HEADERS_TEST_COMPILED_IMPLICIT: RAFT_COMPILED defined
  # * EXT_HEADERS_TEST_IMPLICIT:          no macros defined.
  ConfigureTest(
    NAME EXT_HEADERS_TEST_COMPILED_EXPLICIT PATH ${EXT_HEADER_TEST_SOURCES} LIB
    EXPLICIT_INSTANTIATE_ONLY
  )
  ConfigureTest(NAME EXT_HEADERS_TEST_COMPILED_IMPLICIT PATH ${EXT_HEADER_TEST_SOURCES} LIB)
  ConfigureTest(NAME EXT_HEADERS_TEST_IMPLICIT PATH ${EXT_HEADER_TEST_SOURCES})

  ConfigureTest(NAME LABEL_TEST PATH label/label.cu label/merge_labels.cu)

  ConfigureTest(
    NAME
    LINALG_TEST
    PATH
    linalg/add.cu
    linalg/axpy.cu
    linalg/binary_op.cu
    linalg/cholesky_r1.cu
    linalg/coalesced_reduction.cu
    linalg/divide.cu
    linalg/dot.cu
    linalg/eig.cu
    linalg/eig_sel.cu
    linalg/gemm_basic.cpp
    linalg/gemm_layout.cu
    linalg/gemv.cu
    linalg/map.cu
    linalg/map_then_reduce.cu
    linalg/matrix_vector.cu
    linalg/matrix_vector_op.cu
    linalg/mean_squared_error.cu
    linalg/multiply.cu
    linalg/norm.cu
    linalg/normalize.cu
    linalg/pca.cu
    linalg/power.cu
    linalg/randomized_svd.cu
    linalg/reduce.cu
    linalg/reduce_cols_by_key.cu
    linalg/reduce_rows_by_key.cu
    linalg/rsvd.cu
    linalg/sqrt.cu
    linalg/strided_reduction.cu
    linalg/subtract.cu
    linalg/svd.cu
    linalg/ternary_op.cu
    linalg/transpose.cu
    linalg/tsvd.cu
    linalg/unary_op.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    MATRIX_TEST
    PATH
    matrix/argmax.cu
    matrix/argmin.cu
    matrix/columnSort.cu
    matrix/diagonal.cu
    matrix/gather.cu
    matrix/scatter.cu
    matrix/shift.cu
    matrix/eye.cu
    matrix/linewise_op.cu
    matrix/math.cu
    matrix/matrix.cu
    matrix/norm.cu
    matrix/reverse.cu
    matrix/sample_rows.cu
    matrix/slice.cu
    matrix/triangular.cu
    sparse/spectral_matrix.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME MATRIX_SELECT_TEST PATH matrix/select_k.cu matrix/select_k_edgecases.cu LIB
    EXPLICIT_INSTANTIATE_ONLY GPUS 1 PERCENT 100
  )

  ConfigureTest(
    NAME MATRIX_SELECT_LARGE_TEST PATH matrix/select_large_k.cu LIB EXPLICIT_INSTANTIATE_ONLY GPUS
    1 PERCENT 100
  )

  ConfigureTest(
    NAME
    RANDOM_TEST
    PATH
    random/make_blobs.cu
    random/make_regression.cu
    random/multi_variable_gaussian.cu
    random/rng_pcg_host_api.cu
    random/permute.cu
    random/rng.cu
    random/rng_discrete.cu
    random/rng_int.cu
    random/rmat_rectangular_generator.cu
    random/sample_without_replacement.cu
    random/excess_sampling.cu
  )

  ConfigureTest(
    NAME SOLVERS_TEST PATH linalg/eigen_solvers.cu lap/lap.cu sparse/mst.cu
    sparse/solver/lanczos.cu sparse/solver/randomized_svds.cu LIB EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    SPARSE_TEST
    PATH
    sparse/add.cu
    sparse/convert_coo.cu
    sparse/convert_csr.cu
    sparse/csr_row_slice.cu
    sparse/csr_to_dense.cu
    sparse/csr_transpose.cu
    sparse/degree.cu
    sparse/diagonal.cu
    sparse/filter.cu
    sparse/laplacian.cu
    sparse/masked_matmul.cu
    sparse/norm.cu
    sparse/normalize.cu
    sparse/preprocess.cu
    sparse/reduce.cu
    sparse/row_op.cu
    sparse/sddmm.cu
    sparse/select_k_csr.cu
    sparse/sort.cu
    sparse/spgemmi.cu
    sparse/spmm.cu
    sparse/symmetrize.cu
    GPUS
    1
    PERCENT
    100
  )

  ConfigureTest(
    NAME
    STATS_TEST
    PATH
    stats/accuracy.cu
    stats/adjusted_rand_index.cu
    stats/completeness_score.cu
    stats/contingencyMatrix.cu
    stats/cov.cu
    stats/dispersion.cu
    stats/entropy.cu
    stats/histogram.cu
    stats/homogeneity_score.cu
    stats/information_criterion.cu
    stats/kl_divergence.cu
    stats/mean.cu
    stats/meanvar.cu
    stats/mean_center.cu
    stats/minmax.cu
    stats/mutual_info_score.cu
    stats/r2_score.cu
    stats/rand_index.cu
    stats/regression_metrics.cu
    stats/stddev.cu
    stats/sum.cu
    stats/weighted_mean.cu
    stats/v_measure.cu
    LIB
    EXPLICIT_INSTANTIATE_ONLY
  )

  ConfigureTest(
    NAME
    UTILS_TEST
    PATH
    core/seive.cu
    util/bitonic_sort.cu
    util/cudart_utils.cpp
    util/device_atomics.cu
    util/integer_utils.cpp
    util/integer_utils.cu
    util/memory_type_dispatcher.cu
    util/popc.cu
    util/pow2_utils.cu
    util/reduction.cu
    util/dry_run_resources.cpp
    util/dry_run_guards.cu
  )
endif()

# ##################################################################################################
# Install tests ####################################################################################
# ##################################################################################################
rapids_test_install_relocatable(INSTALL_COMPONENT_SET testing DESTINATION bin/gtests/libraft)
