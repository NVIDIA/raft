/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/scatter.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <gtest/gtest.h>

namespace raft {

template <typename InputIteratorT, typename MapIteratorT, typename OutputIteratorT, typename IdxT>
void naiveScatter(
  InputIteratorT in, IdxT D, IdxT N, MapIteratorT map, IdxT map_length, OutputIteratorT out)
{
  for (IdxT outRow = 0; outRow < map_length; ++outRow) {
    typename std::iterator_traits<MapIteratorT>::value_type map_val = map[outRow];
    IdxT outRowStart                                                = map_val * D;
    IdxT inRowStart                                                 = outRow * D;
    for (IdxT i = 0; i < D; ++i) {
      out[outRowStart + i] = in[inRowStart + i];
    }
  }
}

template <typename IdxT>
struct ScatterInputs {
  IdxT nrows;
  IdxT ncols;
  IdxT col_batch_size;
  unsigned long long int seed;
};

template <typename MatrixT, typename IdxT>
class ScatterTest : public ::testing::TestWithParam<ScatterInputs<IdxT>> {
 protected:
  ScatterTest()
    : stream(resource::get_cuda_stream(handle).get()),
      params(::testing::TestWithParam<ScatterInputs<IdxT>>::GetParam()),
      d_in(0, stream),
      d_out_exp(0, stream),
      d_map(0, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    raft::random::RngState r_int(params.seed);

    IdxT len = params.nrows * params.ncols;

    // input matrix setup
    d_in.resize(params.nrows * params.ncols, stream);
    h_in.resize(params.nrows * params.ncols);
    raft::random::uniform(handle, r, d_in.data(), len, MatrixT(-1.0), MatrixT(1.0));
    raft::update_host(h_in.data(), d_in.data(), len, stream);

    // map setup
    d_map.resize(params.nrows, stream);
    h_map.resize(params.nrows);

    auto exec_policy = raft::resource::get_thrust_policy(handle);

    cuda::counting_iterator<IdxT> permute_iter(0);
    thrust::copy(exec_policy, permute_iter, permute_iter + params.nrows, d_map.data());

    thrust::default_random_engine g;
    thrust::shuffle(exec_policy, d_map.data(), d_map.data() + params.nrows, g);

    raft::update_host(h_map.data(), d_map.data(), params.nrows, stream);
    resource::sync_stream(handle, stream);

    // expected and actual output matrix setup
    h_out.resize(params.nrows * params.ncols);
    d_out_exp.resize(params.nrows * params.ncols, stream);

    // launch scatter on the host and copy the results to device
    naiveScatter(h_in.data(), params.ncols, params.nrows, h_map.data(), params.nrows, h_out.data());
    raft::update_device(d_out_exp.data(), h_out.data(), params.nrows * params.ncols, stream);

    auto inout_view = raft::make_device_matrix_view<MatrixT, IdxT, row_major>(
      d_in.data(), params.nrows, params.ncols);
    auto map_view = raft::make_device_vector_view<const IdxT, IdxT>(d_map.data(), params.nrows);

    raft::matrix::scatter(handle, inout_view, map_view, params.col_batch_size);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream = 0;
  ScatterInputs<IdxT> params;
  std::vector<MatrixT> h_in, h_out;
  std::vector<IdxT> h_map;
  rmm::device_uvector<MatrixT> d_in, d_out_exp;
  rmm::device_uvector<IdxT> d_map;
};

#define SCATTER_TEST(test_type, test_name, test_inputs)                                      \
  typedef RAFT_DEPAREN(test_type) test_name;                                                 \
  TEST_P(test_name, Result)                                                                  \
  {                                                                                          \
    ASSERT_TRUE(                                                                             \
      devArrMatch(d_in.data(), d_out_exp.data(), d_out_exp.size(), raft::Compare<float>())); \
  }                                                                                          \
  INSTANTIATE_TEST_CASE_P(ScatterTests, test_name, ::testing::ValuesIn(test_inputs))

const std::vector<ScatterInputs<int>> inputs_i32 =
  raft::util::itertools::product<ScatterInputs<int>>(
    {25, 2000}, {6, 31, 129}, {0, 1, 2, 3, 6, 100}, {1234ULL});
const std::vector<ScatterInputs<int64_t>> inputs_i64 =
  raft::util::itertools::product<ScatterInputs<int64_t>>(
    {25, 2000}, {6, 31, 129}, {0, 1, 2, 3, 6, 100}, {1234ULL});

SCATTER_TEST((ScatterTest<float, int>), ScatterTestFI32, inputs_i32);
SCATTER_TEST((ScatterTest<float, int64_t>), ScatterTestFI64, inputs_i64);
}  // end namespace raft
