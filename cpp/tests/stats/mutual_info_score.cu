/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/mutual_info_score.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <random>

namespace raft {
namespace stats {

// parameter structure definition
struct mutualInfoParam {
  int nElements;
  int lowerLabelRange;
  int upperLabelRange;
  bool sameArrays;
  double tolerance;
};

// test fixture class
template <typename T>
class mutualInfoTest : public ::testing::TestWithParam<mutualInfoParam> {
 protected:
  // the constructor
  void SetUp() override
  {
    // getting the parameters
    params = ::testing::TestWithParam<mutualInfoParam>::GetParam();

    nElements       = params.nElements;
    lowerLabelRange = params.lowerLabelRange;
    upperLabelRange = params.upperLabelRange;

    // generating random value test input
    std::vector<int> arr1(nElements, 0);
    std::vector<int> arr2(nElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_int_distribution<int> intGenerator(lowerLabelRange, upperLabelRange);

    std::generate(arr1.begin(), arr1.end(), [&]() { return intGenerator(dre); });
    if (params.sameArrays) {
      arr2 = arr1;
    } else {
      std::generate(arr2.begin(), arr2.end(), [&]() { return intGenerator(dre); });
    }

    // generating the golden output
    // calculating the contingency matrix
    int numUniqueClasses     = upperLabelRange - lowerLabelRange + 1;
    const auto hGoldenOutput = std::make_unique<int[]>(numUniqueClasses * numUniqueClasses);
    int i, j;
    for (i = 0; i < nElements; i++) {
      int row    = arr1[i] - lowerLabelRange;
      int column = arr2[i] - lowerLabelRange;

      hGoldenOutput[row * numUniqueClasses + column] += 1;
    }

    const auto a = std::make_unique<int[]>(numUniqueClasses);
    const auto b = std::make_unique<int[]>(numUniqueClasses);

    // and also the reducing contingency matrix along row and column
    for (i = 0; i < numUniqueClasses; ++i) {
      for (j = 0; j < numUniqueClasses; ++j) {
        a[i] += hGoldenOutput[i * numUniqueClasses + j];
        b[i] += hGoldenOutput[j * numUniqueClasses + i];
      }
    }

    // calculating the truth mutual information
    for (int i = 0; i < numUniqueClasses; ++i) {
      for (int j = 0; j < numUniqueClasses; ++j) {
        if (a[i] * b[j] != 0 && hGoldenOutput[i * numUniqueClasses + j] != 0) {
          truthmutualInfo +=
            (double)(hGoldenOutput[i * numUniqueClasses + j]) *
            (log((double)(double(nElements) * hGoldenOutput[i * numUniqueClasses + j])) -
             log((double)(a[i] * b[j])));
        }
      }
    }

    truthmutualInfo /= nElements;

    // allocating and initializing memory to the GPU
    stream = resource::get_cuda_stream(handle).get();

    rmm::device_uvector<T> firstClusterArray(nElements, stream);
    rmm::device_uvector<T> secondClusterArray(nElements, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(firstClusterArray.data(), 0, firstClusterArray.size() * sizeof(T), stream));
    RAFT_CUDA_TRY(
      cudaMemsetAsync(secondClusterArray.data(), 0, secondClusterArray.size() * sizeof(T), stream));

    raft::update_device(firstClusterArray.data(), &arr1[0], (int)nElements, stream);
    raft::update_device(secondClusterArray.data(), &arr2[0], (int)nElements, stream);

    // calling the mutualInfo CUDA implementation
    computedmutualInfo = raft::stats::mutual_info_score(
      handle,
      raft::make_device_vector_view<const T>(firstClusterArray.data(), nElements),
      raft::make_device_vector_view<const T>(secondClusterArray.data(), nElements),
      lowerLabelRange,
      upperLabelRange);
  }

  // declaring the data values
  raft::resources handle;
  mutualInfoParam params;
  T lowerLabelRange, upperLabelRange;
  int nElements             = 0;
  double truthmutualInfo    = 0;
  double computedmutualInfo = 0;
  cudaStream_t stream       = 0;
};

// setting test parameter values
const std::vector<mutualInfoParam> inputs = {{199, 1, 10, false, 0.000001},
                                             {200, 15, 100, false, 0.000001},
                                             {100, 1, 20, false, 0.000001},
                                             {10, 1, 10, false, 0.000001},
                                             {198, 1, 100, false, 0.000001},
                                             {300, 3, 99, false, 0.000001},
                                             {199, 1, 10, true, 0.000001},
                                             {200, 15, 100, true, 0.000001},
                                             {100, 1, 20, true, 0.000001},
                                             {10, 1, 10, true, 0.000001},
                                             {198, 1, 100, true, 0.000001},
                                             {300, 3, 99, true, 0.000001}};

// writing the test suite
typedef mutualInfoTest<int> mutualInfoTestClass;
TEST_P(mutualInfoTestClass, Result)
{
  ASSERT_NEAR(computedmutualInfo, truthmutualInfo, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(mutualInfo, mutualInfoTestClass, ::testing::ValuesIn(inputs));

}  // end namespace stats
}  // end namespace raft
