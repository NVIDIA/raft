/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../linalg/matrix_vector_op.cuh"
#include "../test_utils.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/matrix/linewise_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <cuda_profiler_api.h>
#include <thrust/for_each.h>

#include <gtest/gtest.h>

namespace raft {
namespace matrix {

constexpr std::size_t PTR_PADDING = 128;

struct LinewiseTestParams {
  double tolerance;
  std::size_t workSizeBytes;
  uint64_t seed;
  bool checkCorrectness;
  int inAlignOffset;
  int outAlignOffset;
};

template <typename T, typename I, typename ParamsReader>
struct LinewiseTest : public ::testing::TestWithParam<typename ParamsReader::Params> {
  const raft::resources handle;
  const LinewiseTestParams params;
  cuda::stream_ref stream;

  LinewiseTest()
    : testing::TestWithParam<typename ParamsReader::Params>(),
      params(
        ParamsReader::read(::testing::TestWithParam<typename ParamsReader::Params>::GetParam())),
      handle(),
      stream(resource::get_cuda_stream(handle))
  {
  }

  template <typename layout>
  void runLinewiseSum(T* out, const T* in, const I lineLen, const I nLines, const T* vec)
  {
    constexpr auto rowmajor = std::is_same_v<layout, row_major>;

    I m = rowmajor ? lineLen : nLines;
    I n = rowmajor ? nLines : lineLen;

    auto in_view  = raft::make_device_matrix_view<const T, I, layout>(in, n, m);
    auto out_view = raft::make_device_matrix_view<T, I, layout>(out, n, m);

    auto vec_view = raft::make_device_vector_view<const T>(vec, lineLen);
    if (raft::is_row_major(in_view)) {
      matrix::linewise_op<raft::Apply::ALONG_ROWS>(
        handle, in_view, out_view, raft::add_op{}, vec_view);
    } else {
      matrix::linewise_op<raft::Apply::ALONG_COLUMNS>(
        handle, in_view, out_view, raft::add_op{}, vec_view);
    }
  }

  template <typename layout>
  void runLinewiseSum(
    T* out, const T* in, const I lineLen, const I nLines, const T* vec1, const T* vec2)
  {
    auto f                  = [] __device__(T a, T b, T c) -> T { return a + b + c; };
    constexpr auto rowmajor = std::is_same_v<layout, row_major>;

    I m = rowmajor ? lineLen : nLines;
    I n = rowmajor ? nLines : lineLen;

    auto in_view   = raft::make_device_matrix_view<const T, I, layout>(in, n, m);
    auto out_view  = raft::make_device_matrix_view<T, I, layout>(out, n, m);
    auto vec1_view = raft::make_device_vector_view<const T, I>(vec1, lineLen);
    auto vec2_view = raft::make_device_vector_view<const T, I>(vec2, lineLen);

    if (raft::is_row_major(in_view)) {
      matrix::linewise_op<raft::Apply::ALONG_ROWS>(
        handle, in_view, out_view, f, vec1_view, vec2_view);
    } else {
      matrix::linewise_op<raft::Apply::ALONG_COLUMNS>(
        handle, in_view, out_view, f, vec1_view, vec2_view);
    }
  }

  rmm::device_uvector<T> genData(size_t workSizeBytes)
  {
    raft::random::RngState r(params.seed);
    const std::size_t workSizeElems = workSizeBytes / sizeof(T);
    rmm::device_uvector<T> blob(workSizeElems, stream);
    uniform(handle, r, blob.data(), workSizeElems, T(-1.0), T(1.0));
    return blob;
  }

  template <typename layout>
  void runLinewiseSumPadded(raft::device_aligned_matrix_view<T, I, layout> out,
                            raft::device_aligned_matrix_view<const T, I, layout> in,
                            const I lineLen,
                            const I nLines,
                            const bool alongLines,
                            const T* vec)
  {
    auto vec_view = raft::make_device_vector_view<const T, I>(vec, alongLines ? lineLen : nLines);
    if (alongLines) {
      matrix::linewise_op<raft::Apply::ALONG_ROWS>(handle, in, out, raft::add_op{}, vec_view);
    } else {
      matrix::linewise_op<raft::Apply::ALONG_COLUMNS>(handle, in, out, raft::add_op{}, vec_view);
    }
  }

  /**
   * Suggest multiple versions of matrix dimensions (n, m), such that
   *
   * (2 * n * m + numVectors * m + minUnused) * sizeof(T) <= workSize.
   *
   * This way I know I can create two matrices and numVectors vectors of size m,
   * such that they fit into the allocated workSet.
   */
  std::vector<std::tuple<I, I>> suggestDimensions(I numVectors)
  {
    const std::size_t workSizeElems = params.workSizeBytes / sizeof(T);
    std::vector<std::tuple<I, I>> out;
    const double b = double(numVectors);
    const double s = double(workSizeElems) - double(PTR_PADDING * 2 * (2 + b));
    double squareN = 0.25 * (sqrt(8.0 * s + b * b) - b);

    auto solveForN       = [s, b](I m) -> double { return (s - b * double(m)) / double(2 * m); };
    auto solveForM       = [s, b](I n) -> double { return s / double(2 * n + b); };
    auto addIfMakesSense = [&out](double x, double y) {
      if (x <= 0 || y <= 0) return;
      I n = I(floor(x));
      I m = I(floor(y));
      if (n > 0 && m > 0) out.push_back(std::make_tuple(n, m));
    };
    std::vector<double> sizes = {15, 16, 17, 256, 257, 263, 1024};
    addIfMakesSense(squareN, squareN);
    for (I k : sizes) {
      addIfMakesSense(solveForN(k), k);
      addIfMakesSense(k, solveForM(k));
    }

    return out;
  }

  std::tuple<T*, const T*, const T*, const T*> assignSafePtrs(rmm::device_uvector<T>& blob,
                                                              I n,
                                                              I m)
  {
    typedef raft::Pow2<PTR_PADDING> Align;
    T* out = Align::roundUp(blob.data()) + params.outAlignOffset;
    const T* in =
      const_cast<const T*>(Align::roundUp(out + n * m + PTR_PADDING)) + params.inAlignOffset;
    const T* vec1 = Align::roundUp(in + n * m + PTR_PADDING);
    const T* vec2 = Align::roundUp(vec1 + m + PTR_PADDING);
    ASSERT(blob.data() + blob.size() >= vec2 + PTR_PADDING,
           "Failed to allocate pointers: the workset is not big enough.");
    return std::make_tuple(out, in, vec1, vec2);
  }

  testing::AssertionResult run(std::vector<std::tuple<I, I>>&& dims, rmm::device_uvector<T>&& blob)
  {
    rmm::device_uvector<T> blob_val(params.checkCorrectness ? blob.size() / 2 : 0, stream);

    stream.sync();
    cudaProfilerStart();
    testing::AssertionResult r = testing::AssertionSuccess();
    for (auto [n, m] : dims) {
      if (!r) break;
      auto [out, in, vec1, vec2] = assignSafePtrs(blob, n, m);
      common::nvtx::range dims_scope("Dims-%zu-%zu", std::size_t(n), std::size_t(m));
      for (auto alongRows : ::testing::Bool()) {
        common::nvtx::range dir_scope(alongRows ? "alongRows" : "acrossRows");
        auto lineLen = alongRows ? m : n;
        auto nLines  = alongRows ? n : m;
        {
          {
            common::nvtx::range vecs_scope("one vec");
            if (alongRows) {
              runLinewiseSum<raft::row_major>(out, in, lineLen, nLines, vec1);
            } else {
              runLinewiseSum<raft::col_major>(out, in, lineLen, nLines, vec1);
            }
          }
          if (params.checkCorrectness) {
            linalg::naiveMatVec(
              blob_val.data(), in, vec1, lineLen, nLines, true, alongRows, T(1), stream.get());
            r = devArrMatch(blob_val.data(), out, n * m, CompareApprox<T>(params.tolerance))
                << " " << (alongRows ? "alongRows" : "acrossRows")
                << " with one vec; lineLen: " << lineLen << "; nLines " << nLines;
            if (!r) break;
          }
          {
            common::nvtx::range vecs_scope("two vecs");
            if (alongRows) {
              runLinewiseSum<raft::row_major>(out, in, lineLen, nLines, vec1, vec2);

            } else {
              runLinewiseSum<raft::col_major>(out, in, lineLen, nLines, vec1, vec2);
            }
          }
          if (params.checkCorrectness) {
            linalg::naiveMatVec(blob_val.data(),
                                in,
                                vec1,
                                vec2,
                                lineLen,
                                nLines,
                                true,
                                alongRows,
                                T(1),
                                stream.get());
            r = devArrMatch(blob_val.data(), out, n * m, CompareApprox<T>(params.tolerance))
                << " " << (alongRows ? "alongRows" : "acrossRows")
                << " with two vecs;  lineLen: " << lineLen << "; nLines " << nLines;
            if (!r) break;
          }
        }
      }
    }
    cudaProfilerStop();

    return r;
  }

  testing::AssertionResult runWithPaddedSpan(std::vector<std::tuple<I, I>>&& dims,
                                             rmm::device_uvector<T>&& blob)
  {
    rmm::device_uvector<T> blob_val(params.checkCorrectness ? blob.size() / 2 : 0, stream);

    stream.sync();
    cudaProfilerStart();
    testing::AssertionResult r = testing::AssertionSuccess();
    for (auto alongRows : ::testing::Bool()) {
      for (auto [n, m] : dims) {
        if (!r) break;
        // take dense testdata
        auto [out, in, vec1, vec2] = assignSafePtrs(blob, n, m);
        common::nvtx::range dims_scope("Dims-%zu-%zu", std::size_t(n), std::size_t(m));
        common::nvtx::range dir_scope(alongRows ? "alongRows" : "acrossRows");

        auto lineLen = alongRows ? m : n;
        auto nLines  = alongRows ? n : m;

        // create a padded span based on testdata (just for functional testing)
        size_t matrix_size_padded;
        if (alongRows) {
          auto extents = matrix_extent<I>{n, m};
          typename raft::layout_right_padded<T>::template mapping<matrix_extent<I>> layout{extents};
          matrix_size_padded = layout.required_span_size();
        } else {
          auto extents = matrix_extent<I>{n, m};
          typename raft::layout_left_padded<T>::template mapping<matrix_extent<I>> layout{extents};
          matrix_size_padded = layout.required_span_size();
        }

        rmm::device_uvector<T> blob_in(matrix_size_padded, stream);
        rmm::device_uvector<T> blob_out(matrix_size_padded, stream);

        {
          auto in2 = in;

          // actual testrun
          common::nvtx::range vecs_scope("one vec");
          if (alongRows) {
            auto inSpan = make_device_aligned_matrix_view<T, I, raft::layout_right_padded<T>>(
              blob_in.data(), n, m);
            auto outSpan = make_device_aligned_matrix_view<T, I, raft::layout_right_padded<T>>(
              blob_out.data(), n, m);
            // prep padded input data
            thrust::for_each_n(rmm::exec_policy(stream),
                               cuda::make_counting_iterator(0ul),
                               nLines * lineLen,
                               [inSpan, in2, lineLen] __device__(size_t i) {
                                 inSpan(i / lineLen, i % lineLen) = in2[i];
                               });
            auto inSpanConst =
              make_device_aligned_matrix_view<const T, I, raft::layout_right_padded<T>>(
                blob_in.data(), n, m);
            runLinewiseSumPadded<raft::layout_right_padded<T>>(
              outSpan, inSpanConst, lineLen, nLines, alongRows, vec1);

            if (params.checkCorrectness) {
              runLinewiseSum<raft::row_major>(out, in, lineLen, nLines, vec1);
              auto out_dense = blob_val.data();
              thrust::for_each_n(rmm::exec_policy(stream),
                                 cuda::make_counting_iterator(0ul),
                                 nLines * lineLen,
                                 [outSpan, out_dense, lineLen] __device__(size_t i) {
                                   out_dense[i] = outSpan(i / lineLen, i % lineLen);
                                 });
              r = devArrMatch(out_dense, out, n * m, CompareApprox<T>(params.tolerance))
                  << " " << (alongRows ? "alongRows" : "acrossRows")
                  << " with one vec;  lineLen: " << lineLen << "; nLines " << nLines;
              if (!r) break;
            }

          } else {
            auto inSpan = make_device_aligned_matrix_view<T, I, raft::layout_left_padded<T>>(
              blob_in.data(), n, m);
            auto outSpan = make_device_aligned_matrix_view<T, I, raft::layout_left_padded<T>>(
              blob_out.data(), n, m);
            // prep padded input data
            thrust::for_each_n(rmm::exec_policy(stream),
                               cuda::make_counting_iterator(0ul),
                               nLines * lineLen,
                               [inSpan, in2, lineLen] __device__(size_t i) {
                                 inSpan(i % lineLen, i / lineLen) = in2[i];
                               });
            auto inSpanConst =
              make_device_aligned_matrix_view<const T, I, raft::layout_left_padded<T>>(
                blob_in.data(), n, m);
            runLinewiseSumPadded<raft::layout_left_padded<T>>(
              outSpan, inSpanConst, lineLen, nLines, alongRows, vec1);

            if (params.checkCorrectness) {
              runLinewiseSum<raft::col_major>(out, in, lineLen, nLines, vec1);
              auto out_dense = blob_val.data();
              thrust::for_each_n(rmm::exec_policy(stream),
                                 cuda::make_counting_iterator(0ul),
                                 nLines * lineLen,
                                 [outSpan, out_dense, lineLen] __device__(size_t i) {
                                   out_dense[i] = outSpan(i % lineLen, i / lineLen);
                                 });
              r = devArrMatch(out_dense, out, n * m, CompareApprox<T>(params.tolerance))
                  << " " << (alongRows ? "alongRows" : "acrossRows")
                  << " with one vec;  lineLen: " << lineLen << "; nLines " << nLines;
              if (!r) break;
            }
          }
        }
      }
    }
    cudaProfilerStop();

    return r;
  }

  testing::AssertionResult run()
  {
    return run(suggestDimensions(2), genData(params.workSizeBytes));
  }

  testing::AssertionResult runWithPaddedSpan()
  {
    return runWithPaddedSpan(suggestDimensions(2), genData(params.workSizeBytes));
  }

  testing::AssertionResult runEdgeCases()
  {
    std::vector<I> sizes = {1, 2, 3, 4, 7, 16};
    std::vector<std::tuple<I, I>> dims;
    for (auto m : sizes) {
      for (auto n : sizes) {
        dims.push_back(std::make_tuple(m, n));
      }
    }

    return run(std::move(dims), genData(1024 * 1024));
  }
};

#define TEST_IT(fun, TestClass, ElemType, IndexType)                                         \
  typedef LinewiseTest<ElemType, IndexType, TestClass> TestClass##_##ElemType##_##IndexType; \
  TEST_P(TestClass##_##ElemType##_##IndexType, fun) { ASSERT_TRUE(fun()); }                  \
  INSTANTIATE_TEST_SUITE_P(LinewiseOp, TestClass##_##ElemType##_##IndexType, TestClass##Params)

#define TEST_IT_SPAN(fun, TestClass, ElemType, IndexType)                                        \
  typedef LinewiseTest<ElemType, IndexType, TestClass> TestClass##Span_##ElemType##_##IndexType; \
  TEST_P(TestClass##Span_##ElemType##_##IndexType, fun) { ASSERT_TRUE(fun()); }                  \
  INSTANTIATE_TEST_SUITE_P(LinewiseOpSpan, TestClass##Span_##ElemType##_##IndexType, SpanParams)

auto SpanParams = ::testing::Combine(::testing::Values(0), ::testing::Values(0));

auto TinyParams = ::testing::Combine(::testing::Values(0, 1, 2, 4), ::testing::Values(0, 1, 2, 3));

struct Tiny {
  typedef std::tuple<int, int> Params;
  static LinewiseTestParams read(Params ps)
  {
    return {/** .tolerance */ 0.00001,
            /** .workSizeBytes */ 0 /* not used anyway */,
            /** .seed */ 42ULL,
            /** .checkCorrectness */ true,
            /** .inAlignOffset */ std::get<0>(ps),
            /** .outAlignOffset */ std::get<1>(ps)};
  }
};

auto MegabyteParams = TinyParams;

struct Megabyte {
  typedef std::tuple<int, int> Params;
  static LinewiseTestParams read(Params ps)
  {
    return {/** .tolerance */ 0.00001,
            /** .workSizeBytes */ 1024 * 1024,
            /** .seed */ 42ULL,
            /** .checkCorrectness */ true,
            /** .inAlignOffset */ std::get<0>(ps),
            /** .outAlignOffset */ std::get<1>(ps)};
  }
};

auto GigabyteParams = ::testing::Combine(::testing::Values(0, 1, 2), ::testing::Values(0, 1, 2));

struct Gigabyte {
  typedef std::tuple<int, int> Params;
  static LinewiseTestParams read(Params ps)
  {
    return {/** .tolerance */ 0.00001,
            /** .workSizeBytes */ 1024 * 1024 * 1024,
            /** .seed */ 42ULL,
            /** .checkCorrectness */ false,
            /** .inAlignOffset */ std::get<0>(ps),
            /** .outAlignOffset */ std::get<1>(ps)};
  }
};

auto TenGigsParams = GigabyteParams;

struct TenGigs {
  typedef std::tuple<int, int> Params;
  static LinewiseTestParams read(Params ps)
  {
    return {/** .tolerance */ 0.00001,
            /** .workSizeBytes */ 10ULL * 1024ULL * 1024ULL * 1024ULL,
            /** .seed */ 42ULL,
            /** .checkCorrectness */ false,
            /** .inAlignOffset */ std::get<0>(ps),
            /** .outAlignOffset */ std::get<1>(ps)};
  }
};

TEST_IT(runEdgeCases, Tiny, float, int);
TEST_IT(runEdgeCases, Tiny, double, int);
TEST_IT(run, Megabyte, float, int);
TEST_IT(run, Megabyte, double, int);
TEST_IT(run, Gigabyte, float, int);
TEST_IT(run, Gigabyte, double, int);
TEST_IT(run, TenGigs, float, uint64_t);
TEST_IT(run, TenGigs, double, uint64_t);

TEST_IT_SPAN(runWithPaddedSpan, Megabyte, float, int);
TEST_IT_SPAN(runWithPaddedSpan, Megabyte, double, int);
TEST_IT_SPAN(runWithPaddedSpan, Gigabyte, float, int);
TEST_IT_SPAN(runWithPaddedSpan, Gigabyte, double, int);

}  // namespace matrix
}  // end namespace raft
