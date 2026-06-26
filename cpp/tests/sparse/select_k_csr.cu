/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/matrix/select_k.cuh>
#include <raft/util/cuda_utils.cuh>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

namespace raft {
namespace sparse {

template <typename index_t>
struct SelectKCsrInputs {
  index_t n_rows;
  index_t n_cols;
  index_t top_k;
  float sparsity;
  bool select_min;
  bool customized_indices;
};

template <typename T>
struct CompareApproxWithInf {
  CompareApproxWithInf(T eps_) : eps(eps_) {}
  bool operator()(const T& a, const T& b) const
  {
    if (std::isinf(a) && std::isinf(b)) return true;
    T diff  = std::abs(a - b);
    T m     = std::max(std::abs(a), std::abs(b));
    T ratio = diff > eps ? diff / m : diff;

    return (ratio <= eps);
  }

 private:
  T eps;
};

template <typename value_t, typename index_t>
class SelectKCsrTest : public ::testing::TestWithParam<SelectKCsrInputs<index_t>> {
 public:
  SelectKCsrTest()
    : stream(resource::get_cuda_stream(handle)),
      params(::testing::TestWithParam<SelectKCsrInputs<index_t>>::GetParam()),
      indices_d(0, stream),
      customized_indices_d(0, stream),
      indptr_d(0, stream),
      values_d(0, stream),
      dst_values_d(0, stream),
      dst_values_expected_d(0, stream),
      dst_indices_d(0, stream),
      dst_indices_expected_d(0, stream)
  {
  }

 protected:
  index_t create_sparse_matrix(index_t m, index_t n, value_t sparsity, std::vector<bool>& matrix)
  {
    index_t total_elements = static_cast<index_t>(m * n);
    index_t num_ones       = static_cast<index_t>((total_elements * 1.0f) * sparsity);
    index_t res            = num_ones;

    for (index_t i = 0; i < total_elements; ++i) {
      matrix[i] = false;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_idx(0, total_elements - 1);

    while (num_ones > 0) {
      size_t index = dis_idx(gen);
      if (matrix[index] == false) {
        matrix[index] = true;
        num_ones--;
      }
    }
    return res;
  }

  void convert_to_csr(std::vector<bool>& matrix,
                      index_t rows,
                      index_t cols,
                      std::vector<index_t>& indices,
                      std::vector<index_t>& indptr)
  {
    index_t offset_indptr   = 0;
    index_t offset_values   = 0;
    indptr[offset_indptr++] = 0;

    for (index_t i = 0; i < rows; ++i) {
      for (index_t j = 0; j < cols; ++j) {
        if (matrix[i * cols + j]) {
          indices[offset_values] = static_cast<index_t>(j);
          offset_values++;
        }
      }
      indptr[offset_indptr++] = static_cast<index_t>(offset_values);
    }
  }

  void cpu_select_k(const std::vector<index_t>& indptr_h,
                    const std::vector<index_t>& indices_h,
                    const std::vector<value_t>& values_h,
                    std::optional<std::vector<index_t>>& in_idx_h,
                    index_t n_rows,
                    index_t n_cols,
                    index_t top_k,
                    std::vector<value_t>& out_values_h,
                    std::vector<index_t>& out_indices_h,
                    bool select_min = true)
  {
    auto comp = [select_min](const std::pair<value_t, index_t>& a,
                             const std::pair<value_t, index_t>& b) {
      return select_min ? a.first < b.first : a.first >= b.first;
    };

    for (index_t row = 0; row < n_rows; ++row) {
      std::priority_queue<std::pair<value_t, index_t>,
                          std::vector<std::pair<value_t, index_t>>,
                          decltype(comp)>
        pq(comp);

      for (index_t idx = indptr_h[row]; idx < indptr_h[row + 1]; ++idx) {
        pq.push({values_h[idx], (in_idx_h.has_value()) ? (*in_idx_h)[idx] : indices_h[idx]});
        if (pq.size() > size_t(top_k)) { pq.pop(); }
      }

      std::vector<std::pair<value_t, index_t>> row_pairs;
      while (!pq.empty()) {
        row_pairs.push_back(pq.top());
        pq.pop();
      }

      if (select_min) {
        std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
          return a.first <= b.first;
        });
      } else {
        std::sort(row_pairs.begin(), row_pairs.end(), [](const auto& a, const auto& b) {
          return a.first >= b.first;
        });
      }
      for (index_t col = 0; col < top_k; col++) {
        if (col < index_t(row_pairs.size())) {
          out_values_h[row * top_k + col]  = row_pairs[col].first;
          out_indices_h[row * top_k + col] = row_pairs[col].second;
        }
      }
    }
  }

  void random_array(value_t* array, size_t size)
  {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<value_t> dis(-10.0, 10.0);
    std::unordered_set<value_t> uset;

    while (uset.size() < size) {
      uset.insert(dis(gen));
    }
    typename std::unordered_set<value_t>::iterator it = uset.begin();
    for (size_t i = 0; i < size; ++i) {
      array[i] = *(it++);
    }
  }

  template <typename data_t>
  std::optional<data_t> get_opt_var(data_t x)
  {
    if (params.customized_indices) {
      return x;
    } else {
      return std::nullopt;
    }
  }

  void SetUp() override
  {
    std::vector<bool> dense_values_h(params.n_rows * params.n_cols, false);
    nnz = create_sparse_matrix(params.n_rows, params.n_cols, params.sparsity, dense_values_h);

    std::vector<value_t> values_h(nnz);
    std::vector<index_t> indices_h(nnz);
    std::vector<index_t> customized_indices_h(nnz);
    std::vector<index_t> indptr_h(params.n_rows + 1);

    convert_to_csr(dense_values_h, params.n_rows, params.n_cols, indices_h, indptr_h);

    std::vector<value_t> dst_values_h(params.n_rows * params.top_k,
                                      std::numeric_limits<value_t>::infinity());
    std::vector<index_t> dst_indices_h(params.n_rows * params.top_k, static_cast<index_t>(0));

    dst_values_d.resize(params.n_rows * params.top_k, stream);
    dst_indices_d.resize(params.n_rows * params.top_k, stream);
    values_d.resize(nnz, stream);

    update_device(dst_values_d.data(), dst_values_h.data(), dst_values_h.size(), stream);
    update_device(dst_indices_d.data(), dst_indices_h.data(), dst_indices_h.size(), stream);

    if (params.customized_indices) {
      customized_indices_d.resize(nnz, stream);
      update_device(customized_indices_d.data(),
                    customized_indices_h.data(),
                    customized_indices_h.size(),
                    stream);
    }

    resource::sync_stream(handle);

    if (values_h.size()) {
      random_array(values_h.data(), values_h.size());
      raft::copy(values_d.data(), values_h.data(), values_h.size(), stream);
      resource::sync_stream(handle);
    }

    auto optional_indices_h = get_opt_var(customized_indices_h);

    cpu_select_k(indptr_h,
                 indices_h,
                 values_h,
                 optional_indices_h,
                 params.n_rows,
                 params.n_cols,
                 params.top_k,
                 dst_values_h,
                 dst_indices_h,
                 params.select_min);

    indices_d.resize(nnz, stream);
    indptr_d.resize(params.n_rows + 1, stream);

    dst_values_expected_d.resize(params.n_rows * params.top_k, stream);
    dst_indices_expected_d.resize(params.n_rows * params.top_k, stream);

    update_device(values_d.data(), values_h.data(), values_h.size(), stream);
    update_device(indices_d.data(), indices_h.data(), indices_h.size(), stream);
    update_device(indptr_d.data(), indptr_h.data(), indptr_h.size(), stream);
    update_device(dst_values_expected_d.data(), dst_values_h.data(), dst_values_h.size(), stream);
    update_device(
      dst_indices_expected_d.data(), dst_indices_h.data(), dst_indices_h.size(), stream);

    resource::sync_stream(handle);
  }

  void Run()
  {
    auto in_val_structure = raft::make_device_compressed_structure_view<index_t, index_t, index_t>(
      indptr_d.data(),
      indices_d.data(),
      params.n_rows,
      params.n_cols,
      static_cast<index_t>(indices_d.size()));

    auto in_val =
      raft::make_device_csr_matrix_view<const value_t>(values_d.data(), in_val_structure);

    std::optional<raft::device_vector_view<const index_t, index_t>> in_idx;

    in_idx = get_opt_var(
      raft::make_device_vector_view<const index_t, index_t>(customized_indices_d.data(), nnz));

    auto out_val = raft::make_device_matrix_view<value_t, index_t, raft::row_major>(
      dst_values_d.data(), params.n_rows, params.top_k);
    auto out_idx = raft::make_device_matrix_view<index_t, index_t, raft::row_major>(
      dst_indices_d.data(), params.n_rows, params.top_k);

    raft::execute_with_dry_run_check(
      handle,
      [&](raft::resources const& h) {
        raft::sparse::matrix::select_k(
          h, in_val, in_idx, out_val, out_idx, params.select_min, true);
      },
      raft::alloc_behavior::ARGUMENT_DRIVEN);

    ASSERT_TRUE(raft::devArrMatch<index_t>(dst_indices_expected_d.data(),
                                           out_idx.data_handle(),
                                           params.n_rows * params.top_k,
                                           raft::Compare<index_t>(),
                                           stream));

    ASSERT_TRUE(raft::devArrMatch<value_t>(dst_values_expected_d.data(),
                                           out_val.data_handle(),
                                           params.n_rows * params.top_k,
                                           CompareApproxWithInf<value_t>(1e-6f),
                                           stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SelectKCsrInputs<index_t> params;

  index_t nnz;

  rmm::device_uvector<value_t> values_d;
  rmm::device_uvector<index_t> indptr_d;
  rmm::device_uvector<index_t> indices_d;
  rmm::device_uvector<index_t> customized_indices_d;

  rmm::device_uvector<value_t> dst_values_d;
  rmm::device_uvector<value_t> dst_values_expected_d;

  rmm::device_uvector<index_t> dst_indices_d;
  rmm::device_uvector<index_t> dst_indices_expected_d;
};

using SelectKCsrTest_float_int = SelectKCsrTest<float, int>;
TEST_P(SelectKCsrTest_float_int, Result) { Run(); }

using SelectKCsrTest_double_int64 = SelectKCsrTest<double, int64_t>;
TEST_P(SelectKCsrTest_double_int64, Result) { Run(); }

template <typename index_t>
const std::vector<SelectKCsrInputs<index_t>> selectk_inputs = {
  {10, 32, 10, 0.0, true, false},
  {10, 32, 10, 0.0, true, true},
  {10, 32, 10, 0.01, true, false},  // kWarpImmediate
  {10, 32, 10, 0.1, true, true},
  {10, 32, 251, 0.1, true, false},  // kWarpImmediate
  {10, 32, 251, 0.6, true, true},
  {1000, 1024 * 100, 1, 0.1, true, false},  // kWarpImmediate
  {1000, 1024 * 100, 1, 0.2, true, true},
  {1024, 1024, 258, 0.3, true, false},  // kRadix11bitsExtraPass
  {1024, 1024, 600, 0.2, true, true},
  {1024, 1024, 1024, 0.3, true, false},  // kRadix11bitsExtraPass
  {1024, 1024, 1024, 0.2, true, true},
  {100, 1024 * 1000, 251, 0.1, true, false},  // kWarpDistributedShm
  {100, 1024 * 1000, 251, 0.2, true, true},
  {1024, 1024 * 10, 251, 0.3, true, false},  // kWarpImmediate
  {1024, 1024 * 10, 251, 0.2, true, true},
  {1000, 1024 * 20, 1000, 0.2, true, false},  // kRadix11bits
  {1000, 1024 * 20, 1000, 0.3, true, true},
  {2048, 1024 * 10, 1000, 0.2, true, false},  // kRadix11bitsExtraPass
  {2048, 1024 * 10, 1000, 0.3, true, true},
  {2048, 1024 * 10, 2100, 0.1, true, false},  // kRadix11bitsExtraPass
  {2048, 1024 * 10, 2100, 0.2, true, true},
  {10, 32, 10, 0.0, false, false},
  {10, 32, 10, 0.0, false, true},
  {10, 32, 10, 0.01, false, false},  // kWarpImmediate
  {10, 32, 10, 0.1, false, true},
  {10, 32, 251, 0.1, false, false},  // kWarpImmediate
  {10, 32, 251, 0.6, false, true},
  {1000, 1024 * 100, 1, 0.1, false, false},  // kWarpImmediate
  {1000, 1024 * 100, 1, 0.2, false, true},
  {1024, 1024, 258, 0.3, false, false},  // kRadix11bitsExtraPass
  {1024, 1024, 600, 0.2, false, true},
  {1024, 1024, 1024, 0.3, false, false},  // kRadix11bitsExtraPass
  {1024, 1024, 1024, 0.2, false, true},
  {100, 1024 * 1000, 251, 0.1, false, false},  // kWarpDistributedShm
  {100, 1024 * 1000, 251, 0.2, false, true},
  {1024, 1024 * 10, 251, 0.3, false, false},  // kWarpImmediate
  {1024, 1024 * 10, 251, 0.2, false, true},
  {1000, 1024 * 20, 1000, 0.2, false, false},  // kRadix11bits
  {1000, 1024 * 20, 1000, 0.3, false, true},
  {2048, 1024 * 10, 1000, 0.2, false, false},  // kRadix11bitsExtraPass
  {2048, 1024 * 10, 1000, 0.3, false, true},
  {2048, 1024 * 10, 2100, 0.1, false, false},  // kRadix11bitsExtraPass
  {2048, 1024 * 10, 2100, 0.2, false, true}};

INSTANTIATE_TEST_CASE_P(SelectKCsrTest,
                        SelectKCsrTest_float_int,
                        ::testing::ValuesIn(selectk_inputs<int>));
INSTANTIATE_TEST_CASE_P(SelectKCsrTest,
                        SelectKCsrTest_double_int64,
                        ::testing::ValuesIn(selectk_inputs<int64_t>));

}  // namespace sparse
}  // namespace raft
