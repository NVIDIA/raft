/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/contingency_matrix.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <random>

namespace raft {
namespace stats {

struct ContingencyMatrixParam {
  int nElements;
  int minClass;
  int maxClass;
  bool calcCardinality;
  bool skipLabels;
  float tolerance;
};

template <typename T>
class ContingencyMatrixTest : public ::testing::TestWithParam<ContingencyMatrixParam> {
 protected:
  ContingencyMatrixTest()
    : stream(resource::get_cuda_stream(handle).get()),
      dY(0, stream),
      dYHat(0, stream),
      dComputedOutput(0, stream),
      dGoldenOutput(0, stream)
  {
  }

  void SetUp() override
  {
    params = ::testing::TestWithParam<ContingencyMatrixParam>::GetParam();

    int numElements     = params.nElements;
    int lowerLabelRange = params.minClass;
    int upperLabelRange = params.maxClass;

    std::vector<int> y(numElements, 0);
    std::vector<int> y_hat(numElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange, upperLabelRange);

    std::generate(y.begin(), y.end(), [&]() { return intGenerator(dre); });
    std::generate(y_hat.begin(), y_hat.end(), [&]() { return intGenerator(dre); });

    if (params.skipLabels) {
      // remove two label value from input arrays
      int y1 = (upperLabelRange - lowerLabelRange) / 2;
      int y2 = y1 + (upperLabelRange - lowerLabelRange) / 4;

      // replacement values
      int y1_R = y1 + 1;
      int y2_R = y2 + 1;

      std::replace(y.begin(), y.end(), y1, y1_R);
      std::replace(y.begin(), y.end(), y2, y2_R);
      std::replace(y_hat.begin(), y_hat.end(), y1, y1_R);
      std::replace(y_hat.begin(), y_hat.end(), y2, y2_R);
    }

    dY.resize(numElements, stream);
    dYHat.resize(numElements, stream);

    raft::update_device(dYHat.data(), &y_hat[0], numElements, stream);
    raft::update_device(dY.data(), &y[0], numElements, stream);

    if (params.calcCardinality) {
      raft::stats::get_input_class_cardinality(
        handle,
        raft::make_device_vector_view<const T>(dY.data(), numElements),
        raft::make_host_scalar_view(&minLabel),
        raft::make_host_scalar_view(&maxLabel));
    } else {
      minLabel = lowerLabelRange;
      maxLabel = upperLabelRange;
    }

    numUniqueClasses = maxLabel - minLabel + 1;

    dComputedOutput.resize(numUniqueClasses * numUniqueClasses, stream);
    dGoldenOutput.resize(numUniqueClasses * numUniqueClasses, stream);

    // generate golden output on CPU
    size_t sizeOfMat = numUniqueClasses * numUniqueClasses * sizeof(int);
    std::vector<int> hGoldenOutput(sizeOfMat, 0);

    for (int i = 0; i < numElements; i++) {
      auto row    = y[i] - minLabel;
      auto column = y_hat[i] - minLabel;
      hGoldenOutput[row * numUniqueClasses + column] += 1;
    }

    raft::update_device(
      dGoldenOutput.data(), hGoldenOutput.data(), numUniqueClasses * numUniqueClasses, stream);
    raft::interruptible::synchronize(stream);
  }

  void RunTest()
  {
    int numElements = params.nElements;
    raft::stats::contingency_matrix(
      handle,
      raft::make_device_vector_view<const T>(dY.data(), numElements),
      raft::make_device_vector_view<const T>(dYHat.data(), numElements),
      raft::make_device_matrix_view(dComputedOutput.data(), numUniqueClasses, numUniqueClasses),
      std::make_optional(minLabel),
      std::make_optional(maxLabel));

    raft::interruptible::synchronize(stream);
    ASSERT_TRUE(raft::devArrMatch(dComputedOutput.data(),
                                  dGoldenOutput.data(),
                                  numUniqueClasses * numUniqueClasses,
                                  raft::Compare<T>()));
  }

  raft::resources handle;
  ContingencyMatrixParam params;
  int numUniqueClasses = -1;
  T minLabel, maxLabel;
  cudaStream_t stream = 0;
  rmm::device_uvector<T> dY, dYHat;
  rmm::device_uvector<int> dComputedOutput, dGoldenOutput;
};

const std::vector<ContingencyMatrixParam> inputs = {
  {10000, 1, 10, true, false, 0.000001},
  {10000, 1, 5000, true, false, 0.000001},
  {10000, 1, 10000, true, false, 0.000001},
  {10000, 1, 20000, true, false, 0.000001},
  {10000, 1, 10, false, false, 0.000001},
  {10000, 1, 5000, false, false, 0.000001},
  {10000, 1, 10000, false, false, 0.000001},
  {10000, 1, 20000, false, false, 0.000001},
  {100000, 1, 100, false, false, 0.000001},
  {1000000, 1, 1200, true, false, 0.000001},
  {1000000, 1, 10000, false, false, 0.000001},
  {100000, 1, 100, false, true, 0.000001},
};

typedef ContingencyMatrixTest<int> ContingencyMatrixTestS;
TEST_P(ContingencyMatrixTestS, Result) { RunTest(); }
INSTANTIATE_TEST_CASE_P(ContingencyMatrix, ContingencyMatrixTestS, ::testing::ValuesIn(inputs));
}  // namespace stats
}  // namespace raft
