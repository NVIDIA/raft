/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/histogram.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/kernel_launch.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace stats {

// Note: this kernel also updates the input vector to take care of OOB bins!
RAFT_KERNEL naiveHistKernel(int* bins, int nbins, int* in, int nrows)
{
  int tid        = threadIdx.x + blockIdx.x * blockDim.x;
  int stride     = blockDim.x * gridDim.x;
  auto offset    = blockIdx.y * nrows;
  auto binOffset = blockIdx.y * nbins;
  for (; tid < nrows; tid += stride) {
    int id = in[offset + tid];
    if (id < 0)
      id = 0;
    else if (id >= nbins)
      id = nbins - 1;
    in[offset + tid] = id;
    raft::myAtomicAdd(bins + binOffset + id, 1);
  }
}

void naiveHist(int* bins, int nbins, int* in, int nrows, int ncols, cudaStream_t stream)
{
  const int TPB = 128;
  int nblksx    = raft::ceildiv(nrows, TPB);
  dim3 blks(nblksx, ncols);
  raft::launch_kernel(stream, blks, TPB, naiveHistKernel, bins, nbins, in, nrows);
}

struct HistInputs {
  int nrows, ncols, nbins;
  bool isNormal;
  HistType type;
  int start, end;
  unsigned long long int seed;
};

class HistTest : public ::testing::TestWithParam<HistInputs> {
 protected:
  HistTest()
    : in(0, resource::get_cuda_stream(handle)),
      bins(0, resource::get_cuda_stream(handle)),
      ref_bins(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    params = ::testing::TestWithParam<HistInputs>::GetParam();
    raft::random::RngState r(params.seed);
    auto stream = resource::get_cuda_stream(handle);
    int len     = params.nrows * params.ncols;
    in.resize(len, stream);
    if (params.isNormal) {
      normalInt(handle, r, in.data(), len, params.start, params.end);
    } else {
      uniformInt(handle, r, in.data(), len, params.start, params.end);
    }
    bins.resize(params.nbins * params.ncols, stream);
    ref_bins.resize(params.nbins * params.ncols, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(ref_bins.data(), 0, sizeof(int) * params.nbins * params.ncols, stream));
    naiveHist(ref_bins.data(), params.nbins, in.data(), params.nrows, params.ncols, stream);
    histogram(handle,
              params.type,
              raft::make_device_matrix_view<const int, int, raft::col_major>(
                in.data(), params.nrows, params.ncols),
              raft::make_device_matrix_view<int, int, raft::col_major>(
                bins.data(), params.nbins, params.ncols));
    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  HistInputs params;
  rmm::device_uvector<int> in, bins, ref_bins;
};

class HistMdspanTest : public ::testing::TestWithParam<HistInputs> {
 protected:
  HistMdspanTest()
    : in(0, resource::get_cuda_stream(handle)),
      bins(0, resource::get_cuda_stream(handle)),
      ref_bins(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    params = ::testing::TestWithParam<HistInputs>::GetParam();
    raft::random::RngState r(params.seed);
    auto stream = resource::get_cuda_stream(handle);
    int len     = params.nrows * params.ncols;
    in.resize(len, stream);

    raft::device_vector_view<int, int> in_view(in.data(), in.size());
    if (params.isNormal) {
      normalInt(handle, r, in_view, params.start, params.end);
    } else {
      uniformInt(handle, r, in_view, params.start, params.end);
    }
    bins.resize(params.nbins * params.ncols, stream);
    ref_bins.resize(params.nbins * params.ncols, stream);
    RAFT_CUDA_TRY(
      cudaMemsetAsync(ref_bins.data(), 0, sizeof(int) * params.nbins * params.ncols, stream));
    naiveHist(ref_bins.data(), params.nbins, in.data(), params.nrows, params.ncols, stream);
    histogram<int>(
      params.type, bins.data(), params.nbins, in.data(), params.nrows, params.ncols, stream);
    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  HistInputs params;
  rmm::device_uvector<int> in, bins, ref_bins;
};

static const int oneK                = 1024;
static const int oneM                = oneK * oneK;
const std::vector<HistInputs> inputs = {
  {oneM, 1, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM, 1, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 1, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 1, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM, 21, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 21, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneM, false, HistTypeGmem, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 21, 2 * oneM, true, HistTypeGmem, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmem, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmem, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemMatchAny, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemMatchAny, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits16, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits16, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits8, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits8, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits4, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits4, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits2, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits2, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemBits1, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemBits1, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM, 1, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 1, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 1, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM, 1, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM, 21, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 21, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneM, false, HistTypeSmemHash, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 21, 2 * oneM, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeSmemHash, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeSmemHash, 1000, 50, 1234ULL},

  {oneM, 1, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM, 1, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 1, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 1, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM, 1, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM, 1, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 1, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 1, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 1, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 1, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM, 21, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 1, 21, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneM, false, HistTypeAuto, 0, 2 * oneM, 1234ULL},
  {oneM + 2, 21, 2 * oneM, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM, 21, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM, 21, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 1, 21, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 1, 21, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
  {oneM + 2, 21, 2 * oneK, false, HistTypeAuto, 0, 2 * oneK, 1234ULL},
  {oneM + 2, 21, 2 * oneK, true, HistTypeAuto, 1000, 50, 1234ULL},
};

TEST_P(HistTest, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    ref_bins.data(), bins.data(), params.nbins * params.ncols, raft::Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(HistTests, HistTest, ::testing::ValuesIn(inputs));

TEST_P(HistMdspanTest, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    ref_bins.data(), bins.data(), params.nbins * params.ncols, raft::Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(HistMdspanTests, HistMdspanTest, ::testing::ValuesIn(inputs));

}  // end namespace stats
}  // end namespace raft
