/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/reduce_cols_by_key.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace linalg {

template <typename T, typename KeyT, typename IdxT>
void naiveReduceColsByKey(const T* in,
                          const KeyT* keys,
                          T* out_ref,
                          IdxT nrows,
                          IdxT ncols,
                          IdxT nkeys,
                          cudaStream_t stream)
{
  std::vector<KeyT> h_keys(ncols, 0u);
  raft::copy(&(h_keys[0]), keys, ncols, stream);
  std::vector<T> h_in(nrows * ncols);
  raft::copy(&(h_in[0]), in, nrows * ncols, stream);
  raft::interruptible::synchronize(stream);
  std::vector<T> out(nrows * nkeys, T(0));
  for (IdxT i = 0; i < nrows; ++i) {
    for (IdxT j = 0; j < ncols; ++j) {
      out[i * nkeys + h_keys[j]] += h_in[i * ncols + j];
    }
  }
  raft::copy(out_ref, &(out[0]), nrows * nkeys, stream);
  raft::interruptible::synchronize(stream);
}

template <typename T, typename IdxT>
struct ReduceColsInputs {
  T tolerance;
  IdxT rows;
  IdxT cols;
  IdxT nkeys;
  unsigned long long int seed;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const ReduceColsInputs<T, IdxT>& p)
{
  os << "{" << p.tolerance << "," << p.rows << "," << p.cols << "," << p.nkeys << "," << p.seed
     << "}";
  return os;
}

template <typename T, typename KeyT, typename IdxT>
class ReduceColsTest : public ::testing::TestWithParam<ReduceColsInputs<T, IdxT>> {
 protected:
  ReduceColsTest() : in(0, stream), out_ref(0, stream), out(0, stream), keys(0, stream) {}

  void SetUp() override
  {
    params = ::testing::TestWithParam<ReduceColsInputs<T, IdxT>>::GetParam();
    raft::random::RngState r(params.seed);
    raft::resources handle;
    auto stream = resource::get_cuda_stream(handle).get();
    auto nrows  = params.rows;
    auto ncols  = params.cols;
    auto nkeys  = params.nkeys;
    in.resize(nrows * ncols, stream);
    keys.resize(ncols, stream);
    out_ref.resize(nrows * nkeys, stream);
    out.resize(nrows * nkeys, stream);
    uniform(handle, r, in.data(), nrows * ncols, T(-1.0), T(1.0));
    uniformInt(handle, r, keys.data(), ncols, KeyT{0}, static_cast<KeyT>(params.nkeys));
    naiveReduceColsByKey(in.data(), keys.data(), out_ref.data(), nrows, ncols, nkeys, stream);
    auto input_view  = raft::make_device_matrix_view<const T>(in.data(), nrows, ncols);
    auto output_view = raft::make_device_matrix_view(out.data(), nrows, nkeys);
    auto keys_view   = raft::make_device_vector_view<const KeyT>(keys.data(), ncols);
    reduce_cols_by_key(handle, input_view, keys_view, output_view, nkeys);
    raft::interruptible::synchronize(stream);
  }

 protected:
  cudaStream_t stream = 0;
  ReduceColsInputs<T, IdxT> params;
  rmm::device_uvector<T> in, out_ref, out;
  rmm::device_uvector<KeyT> keys;
};

#define RCBK_TEST(test_type, test_name, test_inputs)                       \
  typedef RAFT_DEPAREN(test_type) test_name;                               \
  TEST_P(test_name, Result)                                                \
  {                                                                        \
    ASSERT_TRUE(raft::devArrMatch(out_ref.data(),                          \
                                  out.data(),                              \
                                  params.rows * params.nkeys,              \
                                  raft::CompareApprox(params.tolerance))); \
  }                                                                        \
  INSTANTIATE_TEST_CASE_P(ReduceColsTests, test_name, ::testing::ValuesIn(test_inputs))

const std::vector<ReduceColsInputs<float, int>> inputsf_i32 =
  raft::util::itertools::product<ReduceColsInputs<float, int>>(
    {0.001f}, {1, 9, 63, 1024}, {1234, 9999, 101010}, {7, 42, 127, 515, 2022}, {1234ULL});
const std::vector<ReduceColsInputs<double, int>> inputsd_i32 =
  raft::util::itertools::product<ReduceColsInputs<double, int>>(
    {0.000001}, {1, 9, 63, 1024}, {1234, 9999, 101010}, {7, 42, 127, 515, 2022}, {1234ULL});
const std::vector<ReduceColsInputs<float, uint32_t>> inputsf_u32 =
  raft::util::itertools::product<ReduceColsInputs<float, uint32_t>>({0.001f},
                                                                    {1u, 9u, 63u, 1024u},
                                                                    {1234u, 9999u, 101010u},
                                                                    {7u, 42u, 127u, 515u, 2022u},
                                                                    {1234ULL});
const std::vector<ReduceColsInputs<float, int64_t>> inputsf_i64 =
  raft::util::itertools::product<ReduceColsInputs<float, int64_t>>(
    {0.001f}, {1, 9, 63, 1024}, {1234, 9999, 101010}, {7, 42, 127, 515, 2022}, {1234ULL});

RCBK_TEST((ReduceColsTest<float, uint32_t, int>), ReduceColsTestFU32I32, inputsf_i32);
RCBK_TEST((ReduceColsTest<double, uint32_t, int>), ReduceColsTestDU32I32, inputsd_i32);
RCBK_TEST((ReduceColsTest<float, int, uint32_t>), ReduceColsTestFI32U32, inputsf_u32);
RCBK_TEST((ReduceColsTest<float, uint32_t, int64_t>), ReduceColsTestFI32I64, inputsf_i64);

}  // end namespace linalg
}  // end namespace raft
