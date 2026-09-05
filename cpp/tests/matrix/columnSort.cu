/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/col_wise_sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>

namespace raft {
namespace matrix {

template <typename T>
std::vector<int>* sort_indexes(const std::vector<T>& v)
{
  // initialize original index locations
  std::vector<int>* idx = new std::vector<int>(v.size());
  std::iota((*idx).begin(), (*idx).end(), 0);

  // sort indexes based on comparing values in v
  std::sort((*idx).begin(), (*idx).end(), [&v](int i1, int i2) { return v[i1] < v[i2]; });
  return idx;
}

template <typename T>
struct columnSort {
  T tolerance;
  int n_row;
  int n_col;
  bool testKeys;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const columnSort<T>& dims)
{
  return os;
}

template <typename T>
class ColumnSort : public ::testing::TestWithParam<columnSort<T>> {
 protected:
  ColumnSort()
    : keyIn(0, resource::get_cuda_stream(handle)),
      keySorted(0, resource::get_cuda_stream(handle)),
      keySortGolden(0, resource::get_cuda_stream(handle)),
      valueOut(0, resource::get_cuda_stream(handle)),
      goldenValOut(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    params  = ::testing::TestWithParam<columnSort<T>>::GetParam();
    int len = params.n_row * params.n_col;
    keyIn.resize(len, resource::get_cuda_stream(handle));
    valueOut.resize(len, resource::get_cuda_stream(handle));
    goldenValOut.resize(len, resource::get_cuda_stream(handle));
    if (params.testKeys) {
      keySorted.resize(len, resource::get_cuda_stream(handle));
      keySortGolden.resize(len, resource::get_cuda_stream(handle));
    }

    std::vector<T> vals(len);
    std::vector<int> cValGolden(len);
    std::iota(vals.begin(), vals.end(),
              1.0f);  // will have to change input param type
    std::random_shuffle(vals.begin(), vals.end());

    std::vector<T> cKeyGolden(len);

    for (int i = 0; i < params.n_row; i++) {
      std::vector<T> tmp(vals.begin() + i * params.n_col, vals.begin() + (i + 1) * params.n_col);
      auto cpuOut = sort_indexes(tmp);
      std::copy((*cpuOut).begin(), (*cpuOut).end(), cValGolden.begin() + i * params.n_col);
      delete cpuOut;

      if (params.testKeys) {
        std::sort(tmp.begin(), tmp.end());
        std::copy(tmp.begin(), tmp.end(), cKeyGolden.begin() + i * params.n_col);
      }
    }

    raft::update_device(keyIn.data(), &vals[0], len, resource::get_cuda_stream(handle));
    raft::update_device(
      goldenValOut.data(), &cValGolden[0], len, resource::get_cuda_stream(handle));

    if (params.testKeys)
      raft::update_device(
        keySortGolden.data(), &cKeyGolden[0], len, resource::get_cuda_stream(handle));

    auto key_in_view = raft::make_device_matrix_view<const T, int, row_major>(
      keyIn.data(), params.n_row, params.n_col);
    auto value_out_view = raft::make_device_matrix_view<int, int, row_major>(
      valueOut.data(), params.n_row, params.n_col);
    auto key_sorted_view = raft::make_device_matrix_view<T, int, row_major>(
      keySorted.data(), params.n_row, params.n_col);

    raft::matrix::sort_cols_per_row(
      handle, key_in_view, value_out_view, std::make_optional(key_sorted_view));

    RAFT_CUDA_TRY(cudaStreamSynchronize(resource::get_cuda_stream(handle).get()));
  }

 protected:
  raft::resources handle;
  columnSort<T> params;
  rmm::device_uvector<T> keyIn, keySorted, keySortGolden;
  rmm::device_uvector<int> valueOut, goldenValOut;  // valueOut are indexes
};

const std::vector<columnSort<float>> inputsf1 = {{0.000001f, 503, 2000, false},
                                                 {0.000001f, 113, 20000, true},
                                                 {0.000001f, 503, 2000, false},
                                                 {0.000001f, 113, 20000, true}};

typedef ColumnSort<float> ColumnSortF;
TEST_P(ColumnSortF, Result)
{
  // Remove this condition once the implementation of of descending sort is
  // fixed.
  ASSERT_TRUE(devArrMatch(valueOut.data(),
                          goldenValOut.data(),
                          params.n_row * params.n_col,
                          raft::CompareApprox<float>(params.tolerance)));
  if (params.testKeys) {
    ASSERT_TRUE(devArrMatch(keySorted.data(),
                            keySortGolden.data(),
                            params.n_row * params.n_col,
                            raft::CompareApprox<float>(params.tolerance)));
  }
}

INSTANTIATE_TEST_CASE_P(ColumnSortTests, ColumnSortF, ::testing::ValuesIn(inputsf1));

}  // end namespace matrix
}  // end namespace raft
