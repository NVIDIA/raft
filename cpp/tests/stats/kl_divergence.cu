/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../test_utils.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/kl_divergence.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iostream>
#include <random>

namespace raft {
namespace stats {

// parameter structure definition
struct klDivergenceParam {
  int nElements;
  double tolerance;
};

// test fixture class
template <typename DataT>
class klDivergenceTest : public ::testing::TestWithParam<klDivergenceParam> {
 protected:
  // the constructor
  void SetUp() override
  {
    // getting the parameters
    params = ::testing::TestWithParam<klDivergenceParam>::GetParam();
    stream = resource::get_cuda_stream(handle).get();

    nElements = params.nElements;

    // generating random value test input
    std::vector<DataT> h_modelPDF(nElements, 0);
    std::vector<DataT> h_candidatePDF(nElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_real_distribution<DataT> realGenerator(0.0, 1.0);

    std::generate(h_modelPDF.begin(), h_modelPDF.end(), [&]() { return realGenerator(dre); });
    std::generate(
      h_candidatePDF.begin(), h_candidatePDF.end(), [&]() { return realGenerator(dre); });

    // allocating and initializing memory to the GPU
    rmm::device_uvector<DataT> d_modelPDF(nElements, stream);
    rmm::device_uvector<DataT> d_candidatePDF(nElements, stream);
    RAFT_CUDA_TRY(cudaMemset(d_modelPDF.data(), 0, d_modelPDF.size() * sizeof(DataT)));
    RAFT_CUDA_TRY(cudaMemset(d_candidatePDF.data(), 0, d_candidatePDF.size() * sizeof(DataT)));

    raft::update_device(d_modelPDF.data(), &h_modelPDF[0], (int)nElements, stream);
    raft::update_device(d_candidatePDF.data(), &h_candidatePDF[0], (int)nElements, stream);

    // generating the golden output
    for (int i = 0; i < nElements; ++i) {
      if (h_modelPDF[i] == 0.0)
        truthklDivergence += 0;

      else
        truthklDivergence += h_modelPDF[i] * log(h_modelPDF[i] / h_candidatePDF[i]);
    }

    // calling the kl_divergence CUDA implementation
    computedklDivergence = raft::stats::kl_divergence(
      handle,
      raft::make_device_vector_view<const DataT>(d_modelPDF.data(), nElements),
      raft::make_device_vector_view<const DataT>(d_candidatePDF.data(), nElements));
  }

  // declaring the data values
  raft::resources handle;
  klDivergenceParam params;
  int nElements              = 0;
  DataT truthklDivergence    = 0;
  DataT computedklDivergence = 0;
  cudaStream_t stream        = 0;
};

// setting test parameter values
const std::vector<klDivergenceParam> inputs = {
  {500, 0.000001}, {200, 0.001}, {5000, 0.000001}, {500000, 0.000001}

};

// writing the test suite
typedef klDivergenceTest<double> klDivergenceTestClass;
TEST_P(klDivergenceTestClass, Result)
{
  ASSERT_NEAR(computedklDivergence, truthklDivergence, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(klDivergence, klDivergenceTestClass, ::testing::ValuesIn(inputs));

}  // end namespace stats
}  // end namespace raft
