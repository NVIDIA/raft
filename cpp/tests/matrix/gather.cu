/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace raft {

template <bool Conditional,
          bool MapTransform,
          typename InputIteratorT,
          typename MapIteratorT,
          typename StencilIteratorT,
          typename UnaryPredicateOp,
          typename MapTransformOp,
          typename OutputIteratorT,
          typename IdxT>
void naiveGather(InputIteratorT in,
                 IdxT ld,
                 IdxT D,
                 IdxT N,
                 MapIteratorT map,
                 StencilIteratorT stencil,
                 IdxT map_length,
                 OutputIteratorT out,
                 UnaryPredicateOp pred_op,
                 MapTransformOp transform_op)
{
  for (IdxT outRow = 0; outRow < map_length; ++outRow) {
    if constexpr (Conditional) {
      auto stencil_val = stencil[outRow];
      if (!pred_op(stencil_val)) continue;
    }
    typename std::iterator_traits<MapIteratorT>::value_type map_val = map[outRow];
    IdxT transformed_val;
    if constexpr (MapTransform) {
      transformed_val = transform_op(map_val);
    } else {
      transformed_val = map_val;
    }
    IdxT inRowStart  = transformed_val * ld;
    IdxT outRowStart = outRow * D;
    for (IdxT i = 0; i < D; ++i) {
      out[outRowStart + i] = in[inRowStart + i];
    }
  }
}

template <typename IdxT>
struct GatherInputs {
  IdxT nrows;
  IdxT ncols;
  IdxT ncols_margin;  // For strided_view test
  IdxT map_length;
  IdxT col_batch_size;
  unsigned long long int seed;
};

template <bool Conditional,
          bool MapTransform,
          bool Inplace,
          typename MatrixT,
          typename MapT,
          typename IdxT>
class GatherTest : public ::testing::TestWithParam<GatherInputs<IdxT>> {
 protected:
  GatherTest()
    : stream(resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<GatherInputs<IdxT>>::GetParam()),
      d_in(0, stream),
      d_out_exp(0, stream),
      d_out_act(0, stream),
      d_stencil(0, stream),
      d_map(0, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    raft::random::RngState r_int(params.seed);

    IdxT map_length = params.map_length;
    IdxT len        = params.nrows * params.ncols;

    if (map_length > params.nrows) map_length = params.nrows;

    // input matrix setup
    const auto ld_in = params.ncols + params.ncols_margin;
    d_in.resize(params.nrows * ld_in, stream);
    h_in.resize(params.nrows * ld_in);
    raft::random::uniform(handle, r, d_in.data(), len, MatrixT(-1.0), MatrixT(1.0));
    raft::update_host(h_in.data(), d_in.data(), len, stream);

    // map setup
    d_map.resize(map_length, stream);
    h_map.resize(map_length);
    raft::random::uniformInt(handle, r_int, d_map.data(), map_length, (MapT)0, (MapT)params.nrows);
    raft::update_host(h_map.data(), d_map.data(), map_length, stream);

    // stencil setup
    if (Conditional) {
      d_stencil.resize(map_length, stream);
      h_stencil.resize(map_length);
      raft::random::uniform(handle, r, d_stencil.data(), map_length, MatrixT(-1.0), MatrixT(1.0));
      raft::update_host(h_stencil.data(), d_stencil.data(), map_length, stream);
    }

    // unary predicate op (used only when Conditional is true)
    auto pred_op = raft::plug_const_op(MatrixT(0.0), raft::greater_op());

    // map transform op (used only when MapTransform is true)
    auto transform_op =
      raft::compose_op(raft::mod_const_op<IdxT>(params.nrows), raft::add_const_op<IdxT>(10));

    // expected and actual output matrix setup
    h_out.resize(map_length * params.ncols);
    d_out_exp.resize(map_length * params.ncols, stream);
    d_out_act.resize(map_length * params.ncols, stream);

    // launch gather on the host and copy the results to device
    naiveGather<Conditional, MapTransform>(h_in.data(),
                                           ld_in,
                                           params.ncols,
                                           params.nrows,
                                           h_map.data(),
                                           h_stencil.data(),
                                           map_length,
                                           h_out.data(),
                                           pred_op,
                                           transform_op);
    raft::update_device(d_out_exp.data(), h_out.data(), map_length * params.ncols, stream);

    auto out_view = raft::make_device_matrix_view<MatrixT, IdxT, row_major>(
      d_out_act.data(), map_length, params.ncols);
    auto map_view = raft::make_device_vector_view<const MapT, IdxT>(d_map.data(), map_length);
    auto stencil_view =
      raft::make_device_vector_view<const MatrixT, IdxT>(d_stencil.data(), map_length);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        if (params.ncols_margin == 0) {
          auto in_view = raft::make_device_matrix_view<const MatrixT, IdxT, row_major>(
            d_in.data(), params.nrows, params.ncols);
          auto inout_view = raft::make_device_matrix_view<MatrixT, IdxT, row_major>(
            d_in.data(), params.nrows, params.ncols);
          if (Conditional && MapTransform) {
            raft::matrix::gather_if(
              h, in_view, out_view, map_view, stencil_view, pred_op, transform_op);
          } else if (Conditional) {
            raft::matrix::gather_if(h, in_view, out_view, map_view, stencil_view, pred_op);
          } else if (MapTransform && Inplace) {
            raft::matrix::gather(h, inout_view, map_view, params.col_batch_size, transform_op);
          } else if (MapTransform) {
            raft::matrix::gather(h, in_view, map_view, out_view, transform_op);
          } else if (Inplace) {
            raft::matrix::gather(h, inout_view, map_view, params.col_batch_size);
          } else {
            raft::matrix::gather(h, in_view, map_view, out_view);
          }
        } else {
          auto in_view = raft::make_device_strided_matrix_view<const MatrixT, IdxT, row_major>(
            d_in.data(), params.nrows, params.ncols, ld_in);
          auto inout_view = raft::make_device_strided_matrix_view<MatrixT, IdxT, row_major>(
            d_in.data(), params.nrows, params.ncols, ld_in);
          if (Conditional && MapTransform) {
            raft::matrix::gather_if(
              h, in_view, out_view, map_view, stencil_view, pred_op, transform_op);
          } else if (Conditional) {
            raft::matrix::gather_if(h, in_view, out_view, map_view, stencil_view, pred_op);
          } else if (MapTransform && Inplace) {
            raft::matrix::gather(h, inout_view, map_view, params.col_batch_size, transform_op);
          } else if (MapTransform) {
            raft::matrix::gather(h, in_view, map_view, out_view, transform_op);
          } else if (Inplace) {
            raft::matrix::gather(h, inout_view, map_view, params.col_batch_size);
          } else {
            raft::matrix::gather(h, in_view, map_view, out_view);
          }
        }
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN);

    if (Inplace) {
      RAFT_CUDA_TRY(cudaMemcpy2DAsync(d_out_act.data(),
                                      params.ncols * sizeof(MatrixT),
                                      d_in.data(),
                                      ld_in * sizeof(MatrixT),
                                      params.ncols * sizeof(MatrixT),
                                      map_length,
                                      cudaMemcpyDefault,
                                      raft::resource::get_cuda_stream(handle)));
    }

    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream = 0;
  GatherInputs<IdxT> params;
  std::vector<MatrixT> h_in, h_out, h_stencil;
  std::vector<MapT> h_map;
  rmm::device_uvector<MatrixT> d_in, d_out_exp, d_out_act, d_stencil;
  rmm::device_uvector<MapT> d_map;
};

#define GATHER_TEST(test_type, test_name, test_inputs)                                            \
  typedef RAFT_DEPAREN(test_type) test_name;                                                      \
  TEST_P(test_name, Result)                                                                       \
  {                                                                                               \
    ASSERT_TRUE(                                                                                  \
      devArrMatch(d_out_exp.data(), d_out_act.data(), d_out_exp.size(), raft::Compare<float>())); \
  }                                                                                               \
  INSTANTIATE_TEST_CASE_P(GatherTests, test_name, ::testing::ValuesIn(test_inputs))

const std::vector<GatherInputs<int>> inputs_i32 = raft::util::itertools::product<GatherInputs<int>>(
  {25, 2000}, {6, 31, 129}, {0, 1, 11}, {11, 999}, {2, 3, 6}, {1234ULL});
const std::vector<GatherInputs<int64_t>> inputs_i64 =
  raft::util::itertools::product<GatherInputs<int64_t>>(
    {25, 2000}, {6, 31, 129}, {0, 1, 11}, {11, 999}, {2, 3, 6}, {1234ULL});
const std::vector<GatherInputs<int>> inplace_inputs_i32 =
  raft::util::itertools::product<GatherInputs<int>>(
    {25, 2000}, {6, 31, 129}, {0, 1, 11}, {11, 999}, {0, 1, 2, 3, 6, 100}, {1234ULL});
const std::vector<GatherInputs<int64_t>> inplace_inputs_i64 =
  raft::util::itertools::product<GatherInputs<int64_t>>(
    {25, 2000}, {6, 31, 129}, {0, 1, 11}, {11, 999}, {0, 1, 2, 3, 6, 100}, {1234ULL});

GATHER_TEST((GatherTest<false, false, false, float, uint32_t, int>), GatherTestFU32I32, inputs_i32);
GATHER_TEST((GatherTest<false, true, false, float, uint32_t, int>),
            GatherTransformTestFU32I32,
            inputs_i32);
GATHER_TEST((GatherTest<true, false, false, float, uint32_t, int>),
            GatherIfTestFU32I32,
            inputs_i32);
GATHER_TEST((GatherTest<true, true, false, float, uint32_t, int>),
            GatherIfTransformTestFU32I32,
            inputs_i32);
GATHER_TEST((GatherTest<true, true, false, double, uint32_t, int>),
            GatherIfTransformTestDU32I32,
            inputs_i32);
GATHER_TEST((GatherTest<true, true, false, float, uint32_t, int64_t>),
            GatherIfTransformTestFU32I64,
            inputs_i64);
GATHER_TEST((GatherTest<true, true, false, float, int64_t, int64_t>),
            GatherIfTransformTestFI64I64,
            inputs_i64);
GATHER_TEST((GatherTest<false, false, true, float, uint32_t, int>),
            GatherInplaceTestFU32I32,
            inplace_inputs_i32);
GATHER_TEST((GatherTest<false, false, true, float, uint32_t, int64_t>),
            GatherInplaceTestFU32I64,
            inplace_inputs_i64);
GATHER_TEST((GatherTest<false, false, true, float, int64_t, int64_t>),
            GatherInplaceTestFI64I64,
            inplace_inputs_i64);
}  // end namespace raft
