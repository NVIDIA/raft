/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <common/benchmark.hpp>

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/copy.cuh>
#include <raft/random/make_blobs.cuh>
#include <raft/random/rng_state.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/matrix/select_k.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/itertools.hpp>

#include <rmm/device_uvector.hpp>

#include <random>
#include <sstream>
#include <unordered_set>
#include <vector>

namespace raft::bench::sparse {

template <typename index_t>
struct bench_param {
  index_t n_rows;
  index_t n_cols;
  index_t top_k;
  float sparsity;
  bool select_min         = true;
  bool customized_indices = false;
};

template <typename index_t>
inline auto operator<<(std::ostream& os, const bench_param<index_t>& params) -> std::ostream&
{
  os << params.n_rows << "#" << params.n_cols << "#" << params.top_k << "#" << params.sparsity;
  return os;
}

template <typename value_t, typename index_t>
struct SelectKCsrTest : public fixture {
  SelectKCsrTest(const bench_param<index_t>& p)
    : fixture(true),
      params(p),
      handle(stream),
      values_d(0, stream),
      indptr_d(0, stream),
      indices_d(0, stream),
      customized_indices_d(0, stream),
      dst_values_d(0, stream),
      dst_indices_d(0, stream)
  {
    std::vector<bool> dense_values_h(params.n_rows * params.n_cols);
    nnz = create_sparse_matrix(params.n_rows, params.n_cols, params.sparsity, dense_values_h);

    std::vector<index_t> indices_h(nnz);
    std::vector<index_t> customized_indices_h(nnz);
    std::vector<index_t> indptr_h(params.n_rows + 1);

    convert_to_csr(dense_values_h, params.n_rows, params.n_cols, indices_h, indptr_h);

    std::vector<value_t> dst_values_h(params.n_rows * params.top_k, static_cast<value_t>(2.0f));
    std::vector<index_t> dst_indices_h(params.n_rows * params.top_k,
                                       static_cast<index_t>(params.n_rows * params.n_cols * 100));

    dst_values_d.resize(params.n_rows * params.top_k, stream);
    dst_indices_d.resize(params.n_rows * params.top_k, stream);
    values_d.resize(nnz, stream);

    if (nnz) {
      auto blobs_values = raft::make_device_matrix<value_t, index_t>(handle, 1, nnz);
      auto labels       = raft::make_device_vector<index_t, index_t>(handle, 1);

      raft::random::make_blobs<value_t, index_t>(blobs_values.data_handle(),
                                                 labels.data_handle(),
                                                 1,
                                                 nnz,
                                                 1,
                                                 stream.get(),
                                                 false,
                                                 nullptr,
                                                 nullptr,
                                                 value_t(1.0),
                                                 false,
                                                 value_t(-10.0f),
                                                 value_t(10.0f),
                                                 uint64_t(2024));
      raft::copy(values_d.data(), blobs_values.data_handle(), nnz, stream);
      resource::sync_stream(handle);
    }

    indices_d.resize(nnz, stream);
    indptr_d.resize(params.n_rows + 1, stream);

    update_device(indices_d.data(), indices_h.data(), indices_h.size(), stream);
    update_device(indptr_d.data(), indptr_h.data(), indptr_h.size(), stream);

    if (params.customized_indices) {
      customized_indices_d.resize(nnz, stream);
      update_device(customized_indices_d.data(),
                    customized_indices_h.data(),
                    customized_indices_h.size(),
                    stream);
    }
  }

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

  template <typename data_t>
  std::optional<data_t> get_opt_var(data_t x)
  {
    if (params.customized_indices) {
      return x;
    } else {
      return std::nullopt;
    }
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

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

    raft::sparse::matrix::select_k(handle, in_val, in_idx, out_val, out_idx, params.select_min);
    resource::sync_stream(handle);
    loop_on_state(state, [this, &in_val, &in_idx, &out_val, &out_idx]() {
      raft::sparse::matrix::select_k(
        handle, in_val, in_idx, out_val, out_idx, params.select_min, false);
      resource::sync_stream(handle);
    });
  }

 protected:
  const raft::device_resources handle;

  bench_param<index_t> params;
  index_t nnz;

  rmm::device_uvector<value_t> values_d;
  rmm::device_uvector<index_t> indptr_d;
  rmm::device_uvector<index_t> indices_d;
  rmm::device_uvector<index_t> customized_indices_d;

  rmm::device_uvector<value_t> dst_values_d;
  rmm::device_uvector<index_t> dst_indices_d;
};  // struct SelectKCsrTest

template <typename index_t>
const std::vector<bench_param<index_t>> getInputs()
{
  std::vector<bench_param<index_t>> param_vec;
  struct TestParams {
    index_t m;
    index_t n;
    index_t k;
  };

  const std::vector<TestParams> params_group{
    {20000, 500, 1},    {20000, 500, 2},    {20000, 500, 4},   {20000, 500, 8},
    {20000, 500, 16},   {20000, 500, 32},   {20000, 500, 64},  {20000, 500, 128},
    {20000, 500, 256},

    {1000, 10000, 1},   {1000, 10000, 2},   {1000, 10000, 4},  {1000, 10000, 8},
    {1000, 10000, 16},  {1000, 10000, 32},  {1000, 10000, 64}, {1000, 10000, 128},
    {1000, 10000, 256},

    {100, 100000, 1},   {100, 100000, 2},   {100, 100000, 4},  {100, 100000, 8},
    {100, 100000, 16},  {100, 100000, 32},  {100, 100000, 64}, {100, 100000, 128},
    {100, 100000, 256},

    {10, 1000000, 1},   {10, 1000000, 2},   {10, 1000000, 4},  {10, 1000000, 8},
    {10, 1000000, 16},  {10, 1000000, 32},  {10, 1000000, 64}, {10, 1000000, 128},
    {10, 1000000, 256},

    {10, 1000000, 1},   {10, 1000000, 2},   {10, 1000000, 4},  {10, 1000000, 8},
    {10, 1000000, 16},  {10, 1000000, 32},  {10, 1000000, 64}, {10, 1000000, 128},
    {10, 1000000, 256},

    {10, 1000000, 1},   {10, 1000000, 16},  {10, 1000000, 64}, {10, 1000000, 128},
    {10, 1000000, 256},

    {10, 1000000, 1},   {10, 1000000, 16},  {10, 1000000, 64}, {10, 1000000, 128},
    {10, 1000000, 256}, {1000, 10000, 1},   {1000, 10000, 16}, {1000, 10000, 64},
    {1000, 10000, 128}, {1000, 10000, 256},

    {10, 1000000, 1},   {10, 1000000, 16},  {10, 1000000, 64}, {10, 1000000, 128},
    {10, 1000000, 256}, {1000, 10000, 1},   {1000, 10000, 16}, {1000, 10000, 64},
    {1000, 10000, 128}, {1000, 10000, 256}};

  param_vec.reserve(params_group.size());
  for (TestParams params : params_group) {
    param_vec.push_back(bench_param<index_t>({params.m, params.n, params.k, 0.1}));
  }
  for (TestParams params : params_group) {
    param_vec.push_back(bench_param<index_t>({params.m, params.n, params.k, 0.2}));
  }
  for (TestParams params : params_group) {
    param_vec.push_back(bench_param<index_t>({params.m, params.n, params.k, 0.5}));
  }
  return param_vec;
}

RAFT_BENCH_REGISTER((SelectKCsrTest<float, uint32_t>), "", getInputs<uint32_t>());

}  // namespace raft::bench::sparse
