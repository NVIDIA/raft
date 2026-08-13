/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/cublas_macros.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/cublaslt_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/custom_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cache.hpp>
#include <raft/util/cuda_data_type.hpp>

#include <cuda_fp16.hpp>

#include <cublasLt.h>

#include <type_traits>
#include <utility>

namespace raft {
namespace linalg::detail {

/** Get the cublas compute type for the combination of input types. */
template <typename S, typename A, typename B, typename C>
auto get_matmul_type() -> cublasComputeType_t
{
  static_assert(std::is_same_v<S, float> && std::is_same_v<A, float> && std::is_same_v<B, float> &&
                  std::is_same_v<C, float>,
                "Unsupported combination of input types. Consult cublas API for supported types.");
  return CUBLAS_COMPUTE_32F;
}

template <>
inline auto get_matmul_type<float, float, float, float>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32F;
}
template <>
inline auto get_matmul_type<float, half, half, float>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32F;
}
template <>
inline auto get_matmul_type<float, int8_t, int8_t, float>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32F;
}
template <>
inline auto get_matmul_type<float, half, half, half>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32F;
}
template <>
inline auto get_matmul_type<half, half, half, half>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_16F;
}
template <>
inline auto get_matmul_type<int32_t, int8_t, int8_t, int32_t>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32I;
}
template <>
inline auto get_matmul_type<float, int8_t, int8_t, int8_t>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_32I;
}
template <>
inline auto get_matmul_type<double, double, double, double>() -> cublasComputeType_t
{
  return CUBLAS_COMPUTE_64F;
}

/** Unique representation of a matrix multiplication (assuming fixed types). */
struct matmul_key_t {
  uint64_t m;
  uint64_t n;
  uint64_t k;
  uint64_t lda;
  uint64_t ldb;
  uint64_t ldc;
  bool trans_a;
  bool trans_b;
};

inline auto operator==(const matmul_key_t& a, const matmul_key_t& b) -> bool
{
  return a.m == b.m && a.n == b.n && a.k == b.k && a.lda == b.lda && a.ldb == b.ldb &&
         a.ldc == b.ldc && a.trans_a == b.trans_a && a.trans_b == b.trans_b;
}

struct matmul_key_hash {
  inline auto operator()(const matmul_key_t& x) const noexcept -> std::size_t
  {
    return x.m * x.n * x.k + x.lda * x.ldb * x.ldc + size_t{x.trans_a} + size_t{x.trans_b} * 2;
  }
};

/**
 * cuBLASLt 13.6.0, shipped with CUDA 13.3, may select algorithm 68 once A's physical span reaches
 * 2^31 elements. That algorithm fails during execution for FP32.
 */
inline auto needs_cublaslt_13_6_workaround(const matmul_key_t& args, std::size_t version) noexcept
  -> bool
{
  constexpr uint64_t max_safe_span = (uint64_t{1} << 31) - 1;
  const auto a_columns             = args.trans_a ? args.m : args.k;
  return version == 130600 && args.lda != 0 && a_columns > max_safe_span / args.lda;
}

/**
 * Querying with a physical A leading dimension that is not 16-byte aligned suppresses algorithm 68.
 * The returned algorithm is then used with the real descriptors.
 */
inline auto get_cublaslt_13_6_heuristic_args(const matmul_key_t& args) noexcept -> matmul_key_t
{
  constexpr uint64_t fp32_elements_per_16_bytes = 4;
  auto heuristic_args                           = args;
  if (heuristic_args.lda % fp32_elements_per_16_bytes == 0) { ++heuristic_args.lda; }
  return heuristic_args;
}

inline auto get_cublaslt_algorithm_id(const cublasLtMatmulHeuristicResult_t& heuristic) -> int
{
  int algorithm_id{};
  std::size_t bytes_written{};
  RAFT_CUBLAS_TRY(cublasLtMatmulAlgoConfigGetAttribute(
    &heuristic.algo, CUBLASLT_ALGO_CONFIG_ID, &algorithm_id, sizeof(algorithm_id), &bytes_written));
  return algorithm_id;
}

/** Descriptor for a column-major cublasLt matrix. */
struct cublastlt_matrix_layout {
  cublasLtMatrixLayout_t res{nullptr};
  inline cublastlt_matrix_layout(cudaDataType dtype, uint64_t rows, uint64_t cols, uint64_t ld)
  {
    RAFT_CUBLAS_TRY(cublasLtMatrixLayoutCreate(&res, dtype, rows, cols, ld));
  }
  inline cublastlt_matrix_layout(const cublastlt_matrix_layout&)                    = delete;
  inline auto operator=(const cublastlt_matrix_layout&) -> cublastlt_matrix_layout& = delete;
  inline cublastlt_matrix_layout(cublastlt_matrix_layout&& other) noexcept
    : res(std::exchange(other.res, nullptr))
  {
  }
  inline auto operator=(cublastlt_matrix_layout&& other) noexcept -> cublastlt_matrix_layout&
  {
    std::swap(res, other.res);
    return *this;
  }

  inline ~cublastlt_matrix_layout() noexcept
  {
    if (res != nullptr) { RAFT_CUBLAS_TRY_NO_THROW(cublasLtMatrixLayoutDestroy(res)); }
  }

  // NOLINTNEXTLINE
  inline operator cublasLtMatrixLayout_t() const noexcept { return res; }

  template <typename T>
  static inline auto for_matmul(bool col_major, uint64_t rows, uint64_t cols, uint64_t ld)
    -> cublastlt_matrix_layout
  {
    return cublastlt_matrix_layout{
      get_cuda_data_type<T>(), col_major ? rows : cols, col_major ? cols : rows, ld};
  }

  template <typename T>
  static inline auto for_strided_batched_matmul(bool col_major,
                                                uint64_t rows,
                                                uint64_t cols,
                                                uint64_t ld,
                                                int32_t batch_count,
                                                int64_t batch_stride) -> cublastlt_matrix_layout
  {
    RAFT_EXPECTS(batch_count > 0, "cuBLASLt batch count must be positive");
    auto layout = for_matmul<T>(col_major, rows, cols, ld);
    RAFT_CUBLAS_TRY(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, sizeof(batch_count)));
    RAFT_CUBLAS_TRY(cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride, sizeof(batch_stride)));
    return layout;
  }
};

/** Descriptor for a cublasLt matmul function. */
struct cublastlt_matmul_desc {
  cublasLtMatmulDesc_t res{nullptr};
  inline cublastlt_matmul_desc(cublasComputeType_t compute_type, cudaDataType scale_type)
  {
    RAFT_CUBLAS_TRY(cublasLtMatmulDescCreate(&res, compute_type, scale_type));
  }
  inline cublastlt_matmul_desc(const cublastlt_matmul_desc&)                    = delete;
  inline auto operator=(const cublastlt_matmul_desc&) -> cublastlt_matmul_desc& = delete;
  inline cublastlt_matmul_desc(cublastlt_matmul_desc&& other) noexcept
    : res(std::exchange(other.res, nullptr))
  {
  }
  inline auto operator=(cublastlt_matmul_desc&& other) noexcept -> cublastlt_matmul_desc&
  {
    std::swap(res, other.res);
    return *this;
  }

  inline ~cublastlt_matmul_desc() noexcept
  {
    if (res != nullptr) { RAFT_CUBLAS_TRY_NO_THROW(cublasLtMatmulDescDestroy(res)); }
  }

  // NOLINTNEXTLINE
  inline operator cublasLtMatmulDesc_t() const noexcept { return res; }

  template <typename S, typename A, typename B, typename C, bool DevicePointerMode = false>
  static inline auto for_matmul(bool transpose_a,
                                bool transpose_b,
                                cublasComputeType_t compute_type = get_matmul_type<S, A, B, C>())
    -> cublastlt_matmul_desc
  {
    auto desc = cublastlt_matmul_desc{compute_type, get_cuda_data_type<S>()};
    if constexpr (DevicePointerMode) {
      const cublasPointerMode_t mode = CUBLAS_POINTER_MODE_DEVICE;
      RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode)));
    }
    const cublasOperation_t trans_op = CUBLAS_OP_T;
    if (transpose_a) {
      RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_op, sizeof(trans_op)));
    }
    if (transpose_b) {
      RAFT_CUBLAS_TRY(cublasLtMatmulDescSetAttribute(
        desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_op, sizeof(trans_op)));
    }
    return desc;
  }
};

/** Preference descriptor for a cublasLt matmul heuristic query. */
struct cublastlt_matmul_preference {
  cublasLtMatmulPreference_t res{nullptr};

  inline cublastlt_matmul_preference() { RAFT_CUBLAS_TRY(cublasLtMatmulPreferenceCreate(&res)); }
  inline cublastlt_matmul_preference(const cublastlt_matmul_preference&) = delete;
  inline auto operator=(const cublastlt_matmul_preference&)
    -> cublastlt_matmul_preference& = delete;

  inline ~cublastlt_matmul_preference() noexcept
  {
    RAFT_CUBLAS_TRY_NO_THROW(cublasLtMatmulPreferenceDestroy(res));
  }

  // NOLINTNEXTLINE
  inline operator cublasLtMatmulPreference_t() const noexcept { return res; }
};

/** Full description of matmul. */
struct matmul_desc {
  cublastlt_matmul_desc desc;
  cublastlt_matrix_layout a;
  cublastlt_matrix_layout b;
  cublastlt_matrix_layout c;
  cublasLtMatmulHeuristicResult_t heuristics;

  template <typename S, typename A, typename B, typename C, bool DevicePointerMode = false>
  static inline auto create(raft::resources const& res, const matmul_key_t& args) -> matmul_desc
  {
    matmul_desc r{
      cublastlt_matmul_desc::for_matmul<S, A, B, C, DevicePointerMode>(args.trans_a, args.trans_b),
      cublastlt_matrix_layout::for_matmul<A>(!(args.trans_a), args.m, args.k, args.lda),
      cublastlt_matrix_layout::for_matmul<B>(!(args.trans_b), args.k, args.n, args.ldb),
      cublastlt_matrix_layout::for_matmul<C>(true, args.m, args.n, args.ldc)};

    bool use_cublaslt_13_6_workaround = false;
    if constexpr (std::is_same_v<S, float> && std::is_same_v<A, float> &&
                  std::is_same_v<B, float> && std::is_same_v<C, float>) {
      use_cublaslt_13_6_workaround = needs_cublaslt_13_6_workaround(args, cublasLtGetVersion());
    }

    int algo_count;
    cublastlt_matmul_preference preference;
    const auto query_heuristic = [&](cublasLtMatrixLayout_t a_layout,
                                     cublasLtMatrixLayout_t c_layout) {
      RAFT_CUBLAS_TRY(cublasLtMatmulAlgoGetHeuristic(resource::get_cublaslt_handle(res),
                                                     r.desc,
                                                     a_layout,
                                                     r.b,
                                                     c_layout,
                                                     c_layout,
                                                     preference,
                                                     1,
                                                     &r.heuristics,
                                                     &algo_count));
    };

    if (use_cublaslt_13_6_workaround) {
      const auto heuristic_args = get_cublaslt_13_6_heuristic_args(args);
      const auto heuristic_a    = cublastlt_matrix_layout::for_matmul<A>(
        !(heuristic_args.trans_a), heuristic_args.m, heuristic_args.k, heuristic_args.lda);
      query_heuristic(heuristic_a, r.c);
    } else {
      query_heuristic(r.a, r.c);
    }

    RAFT_EXPECTS(algo_count > 0, "cuBLASLt did not return a matmul algorithm");
    constexpr int faulty_algorithm = 68;
    if (use_cublaslt_13_6_workaround) {
      RAFT_EXPECTS(get_cublaslt_algorithm_id(r.heuristics) != faulty_algorithm,
                   "cuBLASLt 13.6.0 returned faulty algorithm 68 for the workaround query");
    }
    return r;
  }
};

/** Cache with the default constructor; tagged with input types to use separate caches. */
template <typename S, typename A, typename B, typename C, bool DevicePointerMode>
struct matmul_cache {
  /** Number of matmul invocations to cache. */
  static constexpr size_t kDefaultSize = 100;
  cache::lru<matmul_key_t, matmul_key_hash, std::equal_to<>, std::shared_ptr<matmul_desc>> value{
    kDefaultSize};
};

/**
 * A helper structure to allocate alpha and beta pointers if not provided.
 * This designed to do a minimal amount of synchronization.
 */
template <bool DevicePointerMode, typename S>
struct coef_wrapper {
  const S* alpha;
  const S* beta;
};

template <typename S>
struct coef_wrapper<false, S> {
  S alpha_default = 1;
  S beta_default  = 0;
  const S* alpha;
  const S* beta;
  coef_wrapper(const S* alpha_in, const S* beta_in, rmm::cuda_stream_view)
    : alpha(alpha_in == nullptr ? &alpha_default : alpha_in),
      beta(beta_in == nullptr ? &beta_default : beta_in)
  {
  }
};

template <typename S>
struct coef_wrapper<true, S> {
  S* store = nullptr;
  rmm::cuda_stream_view stream;
  const S* alpha;
  const S* beta;
  coef_wrapper(const S* alpha_in, const S* beta_in, rmm::cuda_stream_view stream)
    : stream(stream), alpha(alpha_in), beta(beta_in)
  {
    if (alpha != nullptr && beta != nullptr) { return; }
    S defaults[2] = {1, 0};
    RAFT_CUDA_TRY(cudaMallocAsync(&store, 2 * sizeof(S), stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(store, defaults, 2 * sizeof(S), cudaMemcpyHostToDevice, stream));
    if (alpha == nullptr) { alpha = &store[0]; }
    if (beta == nullptr) { beta = &store[1]; }
  }
  ~coef_wrapper() noexcept
  {
    if (store != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaFreeAsync(store, stream)); }
  }
};

/**
 * Run a strided-batched cublasLt matmul without an algorithm or workspace preference.
 *
 * The matrix dimensions and leading dimensions follow the same column-major convention as
 * `matmul`. Batch strides are measured in elements.
 */
template <bool DevicePointerMode = false, typename S, typename A, typename B, typename C>
void matmul_strided_batched(raft::resources const& res,
                            bool trans_a,
                            bool trans_b,
                            uint64_t m,
                            uint64_t n,
                            uint64_t k,
                            const S* alpha,
                            const A* a_ptr,
                            uint64_t lda,
                            int64_t stride_a,
                            const B* b_ptr,
                            uint64_t ldb,
                            int64_t stride_b,
                            const S* beta,
                            C* c_ptr,
                            uint64_t ldc,
                            int64_t stride_c,
                            int32_t batch_count,
                            cublasComputeType_t compute_type)
{
  if (resource::get_dry_run_flag(res)) { return; }
  common::nvtx::range<common::nvtx::domain::raft> batch_scope(
    "linalg::detail::matmul_strided_batched(m = %d, n = %d, k = %d, batch_count = %d)",
    m,
    n,
    k,
    batch_count);

  auto operation = cublastlt_matmul_desc::for_matmul<S, A, B, C, DevicePointerMode>(
    trans_a, trans_b, compute_type);

  auto a_layout = cublastlt_matrix_layout::for_strided_batched_matmul<A>(
    !trans_a, m, k, lda, batch_count, stride_a);
  auto b_layout = cublastlt_matrix_layout::for_strided_batched_matmul<B>(
    !trans_b, k, n, ldb, batch_count, stride_b);
  auto c_layout =
    cublastlt_matrix_layout::for_strided_batched_matmul<C>(true, m, n, ldc, batch_count, stride_c);

  auto stream = resource::get_cuda_stream(res);
  coef_wrapper<DevicePointerMode, S> coefficients(alpha, beta, stream);
  RAFT_CUBLAS_TRY(cublasLtMatmul(resource::get_cublaslt_handle(res),
                                 operation,
                                 coefficients.alpha,
                                 a_ptr,
                                 a_layout,
                                 b_ptr,
                                 b_layout,
                                 coefficients.beta,
                                 c_ptr,
                                 c_layout,
                                 c_ptr,
                                 c_layout,
                                 nullptr,
                                 nullptr,
                                 0,
                                 stream));
}

/**
 * Compatibility version of the cublasLt matmul wrapper: It takes the cudaStream_t argument
 * explicitly rather than through the raft::resources. This function is used by other legacy
 * functions, which take the cudaStream_t argument explicitly; by using `legacy_matmul`, such
 * functions do not need to duplicate the raft resources handle to set the explicit stream before
 * passing it to `matmul` (thus avoid the extra overheads associated with that).
 *
 * The use of this function in any new code in deprecated.
 */
template <bool DevicePointerMode = false, typename S, typename A, typename B, typename C>
[[deprecated]] void legacy_matmul(raft::resources const& res,
                                  bool trans_a,
                                  bool trans_b,
                                  uint64_t m,
                                  uint64_t n,
                                  uint64_t k,
                                  const S* alpha,
                                  const A* a_ptr,
                                  uint64_t lda,
                                  const B* b_ptr,
                                  uint64_t ldb,
                                  const S* beta,
                                  C* c_ptr,
                                  uint64_t ldc,
                                  cudaStream_t stream)
{
  // We pass nullptr to the workspace, so the extra memory usage should be zero.
  if (resource::get_dry_run_flag(res)) { return; }
  common::nvtx::range<common::nvtx::domain::raft> batch_scope(
    "linalg::matmul(m = %d, n = %d, k = %d)", m, n, k);
  std::shared_ptr<matmul_desc> mm_desc{nullptr};
  matmul_key_t mm_key{m, n, k, lda, ldb, ldc, trans_a, trans_b};
  auto& cache =
    resource::get_custom_resource<matmul_cache<S, A, B, C, DevicePointerMode>>(res)->value;
  if (!cache.get(mm_key, &mm_desc)) {
    mm_desc.reset(new matmul_desc{matmul_desc::create<S, A, B, C, DevicePointerMode>(res, mm_key)});
    cache.set(mm_key, mm_desc);
  }
  // Allocate alpha and beta pointers if not provided.
  coef_wrapper<DevicePointerMode, S> w(alpha, beta, stream);
  RAFT_CUBLAS_TRY(cublasLtMatmul(resource::get_cublaslt_handle(res),
                                 mm_desc->desc,
                                 w.alpha,
                                 a_ptr,
                                 mm_desc->a,
                                 b_ptr,
                                 mm_desc->b,
                                 w.beta,
                                 c_ptr,
                                 mm_desc->c,
                                 c_ptr,
                                 mm_desc->c,
                                 &(mm_desc->heuristics.algo),
                                 nullptr,
                                 0,
                                 stream));
}

/**
 * @brief the wrapper of cublasLt matmul function
 *  It computes the following equation: C = alpha .* opA(A) * opB(B) + beta .* C
 *
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @tparam S the type of scale parameters alpha, beta
 * @tparam A the element type of matrix A
 * @tparam B the element type of matrix B
 * @tparam C the element type of matrix C
 *
 * @param [in] res raft resources
 * @param [in] trans_a cublas transpose op for A
 * @param [in] trans_b cublas transpose op for B
 * @param [in] m number of rows of C
 * @param [in] n number of columns of C
 * @param [in] k number of rows of opB(B) / number of columns of opA(A)
 * @param [in] alpha host or device scalar, if nullptr, the default value 1 will be used
 * @param [in] a_ptr such a matrix that the shape of column-major opA(A) is [m, k]
 * @param [in] lda leading dimension of A
 * @param [in] b_ptr such a matrix that the shape of column-major opA(B) is [k, n]
 * @param [in] ldb leading dimension of B
 * @param [in] beta host or device scalar, if nullptr, the default value 0 will be used
 * @param [inout] c_ptr column-major matrix of size [m, n]
 * @param [in] ldc leading dimension of C
 */
template <bool DevicePointerMode = false, typename S, typename A, typename B, typename C>
void matmul(raft::resources const& res,
            bool trans_a,
            bool trans_b,
            uint64_t m,
            uint64_t n,
            uint64_t k,
            const S* alpha,
            const A* a_ptr,
            uint64_t lda,
            const B* b_ptr,
            uint64_t ldb,
            const S* beta,
            C* c_ptr,
            uint64_t ldc)
{
  return legacy_matmul<DevicePointerMode>(res,
                                          trans_a,
                                          trans_b,
                                          m,
                                          n,
                                          k,
                                          alpha,
                                          a_ptr,
                                          lda,
                                          b_ptr,
                                          ldb,
                                          beta,
                                          c_ptr,
                                          ldc,
                                          resource::get_cuda_stream(res));
}

}  // namespace linalg::detail
}  // namespace raft
