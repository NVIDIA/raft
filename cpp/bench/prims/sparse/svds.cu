/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/sparse/solver/lanczos_svds.cuh>
#include <raft/sparse/solver/randomized_svds.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace raft::bench::sparse {

constexpr std::uint64_t kCsrFileMagic = 0x3152534354464152ULL;

struct csr_file_header {
  std::uint64_t magic;
  std::uint32_t version;
  std::uint32_t dtype;
  std::int64_t rows;
  std::int64_t cols;
  std::int64_t nnz;
};

struct svds_input {
  int rows;
  int cols;
  int nnz;
  int nnz_per_row;
  int k;
  int ncv;
  int n_oversamples;
  int n_power_iters;
  int max_iterations;
  bool file_backed;
};

csr_file_header read_csr_file_header(std::string const& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file) { throw std::runtime_error("Could not open CSR benchmark file: " + path); }

  csr_file_header header{};
  file.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (!file) { throw std::runtime_error("Could not read CSR benchmark file header: " + path); }
  if (header.magic != kCsrFileMagic || header.version != 1) {
    throw std::runtime_error("Invalid CSR benchmark file header: " + path);
  }
  if (header.dtype != 1) {
    throw std::runtime_error("CSR benchmark file must contain float32 values: " + path);
  }
  if (header.rows <= 0 || header.cols <= 0 || header.nnz <= 0) {
    throw std::runtime_error("CSR benchmark file has invalid dimensions: " + path);
  }
  if (header.rows > std::numeric_limits<int>::max() ||
      header.cols > std::numeric_limits<int>::max() ||
      header.nnz > std::numeric_limits<int>::max()) {
    throw std::runtime_error("CSR benchmark file dimensions exceed int32 limits: " + path);
  }
  return header;
}

int env_int(char const* name, int default_value)
{
  auto const* value = std::getenv(name);
  return value == nullptr ? default_value : std::atoi(value);
}

std::vector<double> read_reference_singular_values(std::string const& path, int k)
{
  std::ifstream file(path);
  if (!file) { throw std::runtime_error("Could not open reference singular values file: " + path); }

  std::vector<double> values;
  values.reserve(k);
  double value = 0.0;
  while (file >> value) {
    values.push_back(value);
    if (static_cast<int>(values.size()) == k) { break; }
  }
  if (static_cast<int>(values.size()) != k) {
    throw std::runtime_error("Reference singular values file has fewer than k values: " + path);
  }
  return values;
}

template <typename value_t>
class svds_bench_base : public fixture {
 public:
  explicit svds_bench_base(svds_input const& p)
    : fixture(true),
      params(p),
      nnz(p.nnz),
      indptr(raft::make_device_vector<int, uint32_t>(handle, p.rows + 1)),
      indices(raft::make_device_vector<int, uint32_t>(handle, nnz)),
      values(raft::make_device_vector<value_t, uint32_t>(handle, nnz)),
      singular_values(raft::make_device_vector<value_t, uint32_t>(handle, p.k)),
      U(raft::make_device_matrix<value_t, uint32_t, raft::col_major>(handle, p.rows, p.k)),
      Vt(raft::make_device_matrix<value_t, uint32_t, raft::col_major>(handle, p.k, p.cols))
  {
    if (params.file_backed) {
      load_csr_from_file();
    } else {
      initialize_synthetic_csr();
    }
  }

  void generate_metrics(::benchmark::State& state) override
  {
    state.counters["rows"]        = benchmark::Counter(params.rows);
    state.counters["cols"]        = benchmark::Counter(params.cols);
    state.counters["nnz"]         = benchmark::Counter(nnz);
    state.counters["nnz/row"]     = benchmark::Counter(static_cast<double>(nnz) / params.rows);
    state.counters["components"]  = benchmark::Counter(params.k);
    state.counters["subspace"]    = benchmark::Counter(params.ncv);
    state.counters["oversamples"] = benchmark::Counter(params.n_oversamples);
    add_reference_metrics(state);
    add_orthogonality_metrics(state);
  }

 protected:
  auto csr_view()
  {
    auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
      indptr.data_handle(), indices.data_handle(), params.rows, params.cols, nnz);
    return raft::make_device_csr_matrix_view<value_t const, int, int, int>(values.data_handle(),
                                                                           csr_structure);
  }

  void initialize_synthetic_csr()
  {
    std::vector<int> h_indptr(params.rows + 1);
    std::vector<int> h_indices(nnz);
    std::vector<value_t> h_values(nnz);

    for (int row = 0; row <= params.rows; ++row) {
      h_indptr[row] = row * params.nnz_per_row;
    }

    std::vector<int> row_indices(params.nnz_per_row);
    int const step = std::max(1, params.cols / std::max(1, params.nnz_per_row));
    for (int row = 0; row < params.rows; ++row) {
      for (int j = 0; j < params.nnz_per_row; ++j) {
        row_indices[j] = (row * 131 + j * step + j * 17) % params.cols;
      }
      std::sort(row_indices.begin(), row_indices.end());
      for (int j = 0; j < params.nnz_per_row; ++j) {
        auto offset       = static_cast<std::size_t>(row) * params.nnz_per_row + j;
        h_indices[offset] = row_indices[j];
        h_values[offset] =
          value_t(0.1) + value_t((row * 17 + j * 29 + row_indices[j] * 7) % 1000) / value_t(1000);
      }
    }

    raft::update_device(indptr.data_handle(), h_indptr.data(), h_indptr.size(), stream);
    raft::update_device(indices.data_handle(), h_indices.data(), h_indices.size(), stream);
    raft::update_device(values.data_handle(), h_values.data(), h_values.size(), stream);
    resource::sync_stream(handle, stream);
  }

  void load_csr_from_file()
  {
    auto const* path = std::getenv("RAFT_SVDS_CSR_FILE");
    if (path == nullptr) { throw std::runtime_error("RAFT_SVDS_CSR_FILE must be set"); }

    auto header = read_csr_file_header(path);
    if (header.rows != params.rows || header.cols != params.cols || header.nnz != params.nnz) {
      throw std::runtime_error(
        "CSR benchmark file dimensions changed after benchmark registration");
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
      throw std::runtime_error(std::string("Could not open CSR benchmark file: ") + path);
    }
    file.seekg(sizeof(csr_file_header), std::ios::beg);

    std::vector<int> h_indptr(static_cast<std::size_t>(params.rows) + 1);
    std::vector<int> h_indices(static_cast<std::size_t>(nnz));
    std::vector<value_t> h_values(static_cast<std::size_t>(nnz));

    file.read(reinterpret_cast<char*>(h_indptr.data()), h_indptr.size() * sizeof(int));
    file.read(reinterpret_cast<char*>(h_indices.data()), h_indices.size() * sizeof(int));
    file.read(reinterpret_cast<char*>(h_values.data()), h_values.size() * sizeof(value_t));
    if (!file) {
      throw std::runtime_error(std::string("Could not read CSR benchmark file: ") + path);
    }

    raft::update_device(indptr.data_handle(), h_indptr.data(), h_indptr.size(), stream);
    raft::update_device(indices.data_handle(), h_indices.data(), h_indices.size(), stream);
    raft::update_device(values.data_handle(), h_values.data(), h_values.size(), stream);
    resource::sync_stream(handle, stream);
  }

  void add_reference_metrics(::benchmark::State& state)
  {
    auto const* ref_path = std::getenv("RAFT_SVDS_REF_S");
    if (ref_path == nullptr) { return; }

    auto ref = read_reference_singular_values(ref_path, params.k);
    std::vector<value_t> actual(params.k);
    raft::update_host(actual.data(), singular_values.data_handle(), params.k, stream);
    resource::sync_stream(handle, stream);

    double max_rel  = 0.0;
    double mean_rel = 0.0;
    for (int i = 0; i < params.k; ++i) {
      auto denom = std::max(std::abs(ref[i]), 1e-30);
      auto rel   = std::abs(static_cast<double>(actual[i]) - ref[i]) / denom;
      max_rel    = std::max(max_rel, rel);
      mean_rel += rel;
    }
    mean_rel /= params.k;

    state.counters["s_rel_max"]  = benchmark::Counter(max_rel);
    state.counters["s_rel_mean"] = benchmark::Counter(mean_rel);
    state.counters["s0_rel"]     = benchmark::Counter(
      std::abs(static_cast<double>(actual[0]) - ref[0]) / std::max(std::abs(ref[0]), 1e-30));
  }

  void add_orthogonality_metrics(::benchmark::State& state)
  {
    auto UtU =
      raft::make_device_matrix<value_t, uint32_t, raft::col_major>(handle, params.k, params.k);
    auto VVt =
      raft::make_device_matrix<value_t, uint32_t, raft::col_major>(handle, params.k, params.k);

    value_t one  = 1;
    value_t zero = 0;

    raft::linalg::gemm(handle,
                       U.data_handle(),
                       params.rows,
                       params.k,
                       U.data_handle(),
                       UtU.data_handle(),
                       params.k,
                       params.k,
                       CUBLAS_OP_T,
                       CUBLAS_OP_N,
                       one,
                       zero,
                       stream);

    raft::linalg::gemm(handle,
                       Vt.data_handle(),
                       params.k,
                       params.cols,
                       Vt.data_handle(),
                       VVt.data_handle(),
                       params.k,
                       params.k,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       one,
                       zero,
                       stream);

    std::vector<value_t> h_utu(static_cast<std::size_t>(params.k) * params.k);
    std::vector<value_t> h_vvt(static_cast<std::size_t>(params.k) * params.k);
    raft::update_host(h_utu.data(), UtU.data_handle(), h_utu.size(), stream);
    raft::update_host(h_vvt.data(), VVt.data_handle(), h_vvt.size(), stream);
    resource::sync_stream(handle, stream);

    auto compute_stats = [this](std::vector<value_t> const& gram) {
      double fro_sq  = 0.0;
      double max_abs = 0.0;
      for (int col = 0; col < params.k; ++col) {
        for (int row = 0; row < params.k; ++row) {
          auto const expected = row == col ? 1.0 : 0.0;
          auto const diff     = std::abs(
            static_cast<double>(gram[static_cast<std::size_t>(col) * params.k + row]) - expected);
          fro_sq += diff * diff;
          max_abs = std::max(max_abs, diff);
        }
      }
      return std::array<double, 2>{std::sqrt(fro_sq), max_abs};
    };

    auto const u_stats           = compute_stats(h_utu);
    auto const v_stats           = compute_stats(h_vvt);
    state.counters["u_orth_fro"] = benchmark::Counter(u_stats[0]);
    state.counters["u_orth_max"] = benchmark::Counter(u_stats[1]);
    state.counters["v_orth_fro"] = benchmark::Counter(v_stats[0]);
    state.counters["v_orth_max"] = benchmark::Counter(v_stats[1]);
  }

  svds_input params;
  int nnz;
  raft::device_vector<int, uint32_t> indptr;
  raft::device_vector<int, uint32_t> indices;
  raft::device_vector<value_t, uint32_t> values;
  raft::device_vector<value_t, uint32_t> singular_values;
  raft::device_matrix<value_t, uint32_t, raft::col_major> U;
  raft::device_matrix<value_t, uint32_t, raft::col_major> Vt;
};

template <typename value_t>
class lanczos_svds_bench : public svds_bench_base<value_t> {
 public:
  using svds_bench_base<value_t>::svds_bench_base;

  void run_benchmark(::benchmark::State& state) override
  {
    auto csr_matrix = this->csr_view();

    raft::sparse::solver::sparse_lanczos_svd_config<value_t> config;
    config.n_components   = this->params.k;
    config.ncv            = this->params.ncv;
    config.tolerance      = value_t(1e-4);
    config.max_iterations = this->params.max_iterations;
    config.seed           = 1234;

    this->loop_on_state(
      state,
      [&]() {
        raft::sparse::solver::sparse_lanczos_svd(this->handle,
                                                 config,
                                                 csr_matrix,
                                                 this->singular_values.view(),
                                                 this->U.view(),
                                                 this->Vt.view());
      },
      false);
  }
};

template <typename value_t>
class lanczos_mgs2_svds_bench : public svds_bench_base<value_t> {
 public:
  using svds_bench_base<value_t>::svds_bench_base;

  void run_benchmark(::benchmark::State& state) override
  {
    auto csr_matrix = this->csr_view();

    raft::sparse::solver::sparse_lanczos_svd_config<value_t> config;
    config.n_components               = this->params.k;
    config.ncv                        = this->params.ncv;
    config.tolerance                  = value_t(1e-4);
    config.max_iterations             = this->params.max_iterations;
    config.seed                       = 1234;
    config.use_mgs2_orthogonalization = true;

    this->loop_on_state(
      state,
      [&]() {
        raft::sparse::solver::sparse_lanczos_svd(this->handle,
                                                 config,
                                                 csr_matrix,
                                                 this->singular_values.view(),
                                                 this->U.view(),
                                                 this->Vt.view());
      },
      false);
  }
};

template <typename value_t>
class randomized_svds_bench : public svds_bench_base<value_t> {
 public:
  using svds_bench_base<value_t>::svds_bench_base;

  void run_benchmark(::benchmark::State& state) override
  {
    auto csr_matrix = this->csr_view();

    raft::sparse::solver::sparse_svd_config<value_t> config;
    config.n_components  = this->params.k;
    config.n_oversamples = this->params.n_oversamples;
    config.n_power_iters = this->params.n_power_iters;
    config.seed          = 1234;

    this->loop_on_state(
      state,
      [&]() {
        raft::sparse::solver::sparse_randomized_svd(this->handle,
                                                    config,
                                                    csr_matrix,
                                                    this->singular_values.view(),
                                                    this->U.view(),
                                                    this->Vt.view());
      },
      false);
  }
};

std::vector<svds_input> get_svds_inputs()
{
  std::vector<svds_input> inputs = {
    {10000, 2000, 10000 * 16, 16, 16, 48, 32, 2, 20, false},
    {50000, 5000, 50000 * 16, 16, 32, 80, 48, 2, 20, false},
  };

  auto const* path = std::getenv("RAFT_SVDS_CSR_FILE");
  if (path != nullptr) {
    auto header = read_csr_file_header(path);
    inputs.push_back({static_cast<int>(header.rows),
                      static_cast<int>(header.cols),
                      static_cast<int>(header.nnz),
                      0,
                      env_int("RAFT_SVDS_K", 50),
                      env_int("RAFT_SVDS_NCV", 0),
                      env_int("RAFT_SVDS_RANDOMIZED_OVERSAMPLES", 10),
                      env_int("RAFT_SVDS_RANDOMIZED_POWER_ITERS", 2),
                      env_int("RAFT_SVDS_MAXITER", 100),
                      true});
  }
  return inputs;
}

std::vector<svds_input> const svds_inputs = get_svds_inputs();

RAFT_BENCH_REGISTER((lanczos_svds_bench<float>), "", svds_inputs);
RAFT_BENCH_REGISTER((lanczos_mgs2_svds_bench<float>), "", svds_inputs);
RAFT_BENCH_REGISTER((randomized_svds_bench<float>), "", svds_inputs);

}  // namespace raft::bench::sparse
