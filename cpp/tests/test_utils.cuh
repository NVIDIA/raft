/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "test_utils.h"

#include <raft/core/dry_run_resources.hpp>
#include <raft/core/memory_stats_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/iterator>
#include <thrust/for_each.h>

#include <gtest/gtest.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace raft {

/*
 * @brief Helper function to compare 2 device n-D arrays with custom comparison
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected expected value(s)
 * @param actual actual values
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 * @{
 */
template <typename T, typename L>
testing::AssertionResult devArrMatch(
  const T* expected, const T* actual, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> exp_h(new T[size]);
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(exp_h.get(), expected, size, stream);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < size; ++i) {
    auto exp = exp_h.get()[i];
    auto act = act_h.get()[i];
    if (!eq_compare(exp, act)) {
      return testing::AssertionFailure() << "actual=" << act << " != expected=" << exp << " @" << i;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(
  T expected, const T* actual, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < size; ++i) {
    auto act = act_h.get()[i];
    if (!eq_compare(expected, act)) {
      return testing::AssertionFailure()
             << "actual=" << act << " != expected=" << expected << " @" << i;
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(const T* expected,
                                     const T* actual,
                                     size_t rows,
                                     size_t cols,
                                     L eq_compare,
                                     cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> exp_h(new T[size]);
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(exp_h.get(), expected, size, stream);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      auto idx = i * cols + j;  // row major assumption!
      auto exp = exp_h.get()[idx];
      auto act = act_h.get()[idx];
      if (!eq_compare(exp, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << exp << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename L>
testing::AssertionResult devArrMatch(
  T expected, const T* actual, size_t rows, size_t cols, L eq_compare, cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      auto idx = i * cols + j;  // row major assumption!
      auto act = act_h.get()[idx];
      if (!eq_compare(expected, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << expected << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

/*
 * @brief Helper function to compare a device n-D arrays with an expected array
 * on the host, using a custom comparison
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected_h host array of expected value(s)
 * @param actual_d device array actual values
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 */
template <typename T, typename L>
testing::AssertionResult devArrMatchHost(
  const T* expected_h, const T* actual_d, size_t size, L eq_compare, cudaStream_t stream = 0)
{
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual_d, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  bool ok   = true;
  auto fail = testing::AssertionFailure();
  for (size_t i(0); i < size; ++i) {
    auto exp = expected_h[i];
    auto act = act_h.get()[i];
    if (!eq_compare(exp, act)) {
      ok = false;
      fail << "actual=" << act << " != expected=" << exp << " @" << i << "; ";
    }
  }
  if (!ok) return fail;
  return testing::AssertionSuccess();
}

/**
 * @brief Helper function to compare host vectors using a custom comparison
 * @tparam T the element type
 * @tparam L the comparator lambda or object function
 * @param expected_h host vector of expected value(s)
 * @param actual_h host vector actual values
 * @param eq_compare the comparator
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 */
template <typename T, typename L>
testing::AssertionResult hostVecMatch(const std::vector<T>& expected_h,
                                      const std::vector<T>& actual_h,
                                      L eq_compare)
{
  auto n = actual_h.size();
  if (n != expected_h.size())
    return testing::AssertionFailure()
           << "vector sizez mismatch: "
           << "actual=" << n << " != expected=" << expected_h.size() << "; ";
  for (size_t i = 0; i < n; ++i) {
    auto exp       = expected_h[i];
    auto act       = actual_h[i];
    bool are_equal = [&]() {
      if constexpr (std::is_invocable_v<L, decltype(exp), decltype(act), size_t>) {
        return eq_compare(exp, act, i);
      } else {
        return eq_compare(exp, act);
      }
    }();
    if (!are_equal) {
      return testing::AssertionFailure()
             << "actual=" << act << " != expected=" << exp << " @" << i << "; ";
    }
  }
  return testing::AssertionSuccess();
}

/*
 * @brief Helper function to compare diagonal values of a 2D matrix
 * @tparam T the data type of the arrays
 * @tparam L the comparator lambda or object function
 * @param expected expected value along diagonal
 * @param actual actual matrix
 * @param eq_compare the comparator
 * @param stream cuda stream
 * @return the testing assertion to be later used by ASSERT_TRUE/EXPECT_TRUE
 */
template <typename T, typename L>
testing::AssertionResult diagonalMatch(
  T expected, const T* actual, size_t rows, size_t cols, L eq_compare, cudaStream_t stream = 0)
{
  size_t size = rows * cols;
  std::unique_ptr<T[]> act_h(new T[size]);
  raft::update_host<T>(act_h.get(), actual, size, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  for (size_t i(0); i < rows; ++i) {
    for (size_t j(0); j < cols; ++j) {
      if (i != j) continue;
      auto idx = i * cols + j;  // row major assumption!
      auto act = act_h.get()[idx];
      if (!eq_compare(expected, act)) {
        return testing::AssertionFailure()
               << "actual=" << act << " != expected=" << expected << " @" << i << "," << j;
      }
    }
  }
  return testing::AssertionSuccess();
}

template <typename T, typename IdxT>
typename std::enable_if_t<std::is_floating_point_v<T>> gen_uniform(const raft::resources& handle,
                                                                   T* out,
                                                                   raft::random::RngState& rng,
                                                                   IdxT len,
                                                                   T range_min = T(-1),
                                                                   T range_max = T(1))
{
  raft::random::uniform(handle, rng, out, len, range_min, range_max);
}

template <typename T, typename IdxT>
typename std::enable_if_t<std::is_integral_v<T>> gen_uniform(const raft::resources& handle,
                                                             T* out,
                                                             raft::random::RngState& rng,
                                                             IdxT len,
                                                             T range_min = T(0),
                                                             T range_max = T(100))
{
  raft::random::uniformInt(handle, rng, out, len, range_min, range_max);
}

template <typename T1, typename T2, typename IdxT>
void gen_uniform(const raft::resources& handle,
                 raft::KeyValuePair<T1, T2>* out,
                 raft::random::RngState& rng,
                 IdxT len)
{
  auto stream = resource::get_cuda_stream(handle);
  rmm::device_uvector<T1> keys(len, stream);
  rmm::device_uvector<T2> values(len, stream);

  gen_uniform(handle, keys.data(), rng, len);
  gen_uniform(handle, values.data(), rng, len);

  const T1* d_keys   = keys.data();
  const T2* d_values = values.data();
  auto counting      = cuda::make_counting_iterator<IdxT>(0);
  thrust::for_each(rmm::exec_policy(stream),
                   counting,
                   counting + len,
                   [out, d_keys, d_values] __device__(int idx) {
                     out[idx].key   = d_keys[idx];
                     out[idx].value = d_values[idx];
                   });
}

/** @} */

/** time the function call 'func' using cuda events */
#define TIMEIT_LOOP(ms, count, func)                       \
  do {                                                     \
    cudaEvent_t start, stop;                               \
    RAFT_CUDA_TRY(cudaEventCreate(&start));                \
    RAFT_CUDA_TRY(cudaEventCreate(&stop));                 \
    RAFT_CUDA_TRY(cudaEventRecord(start));                 \
    for (int i = 0; i < count; ++i) {                      \
      func;                                                \
    }                                                      \
    RAFT_CUDA_TRY(cudaEventRecord(stop));                  \
    RAFT_CUDA_TRY(cudaEventSynchronize(stop));             \
    ms = 0.f;                                              \
    RAFT_CUDA_TRY(cudaEventElapsedTime(&ms, start, stop)); \
    ms /= args.runs;                                       \
  } while (0)

inline std::vector<float> read_csv(std::string filename, bool skip_first_n_columns = 1)
{
  std::vector<float> result;
  std::ifstream myFile(filename);
  if (!myFile.is_open()) throw std::runtime_error("Could not open file");

  std::string line, colname;
  int val;

  if (myFile.good()) {
    std::getline(myFile, line);
    std::stringstream ss(line);
    while (std::getline(ss, colname, ',')) {}
  }

  int n_lines = 0;
  while (std::getline(myFile, line)) {
    std::stringstream ss(line);
    int colIdx = 0;
    while (ss >> val) {
      if (colIdx >= skip_first_n_columns) {
        result.push_back(val);
        if (ss.peek() == ',') ss.ignore();
      }
      colIdx++;
    }
    n_lines++;
  }

  printf("lines read: %d\n", n_lines);
  myFile.close();
  return result;
}

enum class alloc_behavior {
  NO_ALLOCATIONS,
  ARGUMENT_DRIVEN,
  DATA_DRIVEN,
};

/**
 * @brief Execute an action and check dry-run protocol compliance.
 *
 * Runs @p action once in dry-run mode (via dry_run_execute) to record predicted
 * allocations, then runs it for real with all six memory resources wrapped in
 * statistics adaptors to record actual peak usage. Compares the predicted and
 * actual peaks according to the specified @p behavior.
 *
 * @tparam Action callable with signature void(raft::resources const&)
 */
template <typename Action>
void execute_with_dry_run_check(raft::resources const& res,
                                Action&& action,
                                alloc_behavior behavior,
                                std::size_t min_alloc = 0)
{
  auto dry = raft::util::dry_run_execute(res, action);

  raft::memory_stats_resources stat_res(res);
  std::forward<Action>(action)(static_cast<const raft::resources&>(stat_res));
  resource::sync_stream(stat_res);
  auto actual = stat_res.get_bytes_peak();

  auto total_dry    = dry.total();
  auto total_actual = actual.total();

  if (dry.device_workspace != actual.device_workspace ||
      dry.device_large_workspace != actual.device_large_workspace ||
      dry.device_global != actual.device_global || dry.device_managed != actual.device_managed ||
      dry.host != actual.host || dry.host_pinned != actual.host_pinned) {
    printf(
      "  dry-run: ws=%zu large_ws=%zu global=%zu managed=%zu host=%zu pinned=%zu (total=%zu)\n"
      "  actual:  ws=%zu large_ws=%zu global=%zu managed=%zu host=%zu pinned=%zu (total=%zu)\n",
      dry.device_workspace,
      dry.device_large_workspace,
      dry.device_global,
      dry.device_managed,
      dry.host,
      dry.host_pinned,
      total_dry,
      actual.device_workspace,
      actual.device_large_workspace,
      actual.device_global,
      actual.device_managed,
      actual.host,
      actual.host_pinned,
      total_actual);
  }

  EXPECT_GE(total_actual, min_alloc);
  EXPECT_GE(total_dry, min_alloc);

  switch (behavior) {
    case alloc_behavior::NO_ALLOCATIONS:
      EXPECT_EQ(dry.device_workspace, std::size_t{0});
      EXPECT_EQ(dry.device_large_workspace, std::size_t{0});
      EXPECT_EQ(dry.device_global, std::size_t{0});
      EXPECT_EQ(dry.device_managed, std::size_t{0});
      EXPECT_EQ(dry.host, std::size_t{0});
      EXPECT_EQ(dry.host_pinned, std::size_t{0});
      EXPECT_EQ(actual.device_workspace, std::size_t{0});
      EXPECT_EQ(actual.device_large_workspace, std::size_t{0});
      EXPECT_EQ(actual.device_global, std::size_t{0});
      EXPECT_EQ(actual.device_managed, std::size_t{0});
      EXPECT_EQ(actual.host, std::size_t{0});
      EXPECT_EQ(actual.host_pinned, std::size_t{0});
      break;
    case alloc_behavior::ARGUMENT_DRIVEN:
      EXPECT_EQ(dry.device_workspace, actual.device_workspace);
      EXPECT_EQ(dry.device_large_workspace, actual.device_large_workspace);
      EXPECT_EQ(dry.device_global, actual.device_global);
      EXPECT_EQ(dry.device_managed, actual.device_managed);
      EXPECT_EQ(dry.host, actual.host);
      EXPECT_EQ(dry.host_pinned, actual.host_pinned);
      break;
    case alloc_behavior::DATA_DRIVEN:
      EXPECT_GE(dry.device_workspace, actual.device_workspace);
      EXPECT_GE(dry.device_large_workspace, actual.device_large_workspace);
      EXPECT_GE(dry.device_global, actual.device_global);
      EXPECT_GE(dry.device_managed, actual.device_managed);
      EXPECT_GE(dry.host, actual.host);
      EXPECT_GE(dry.host_pinned, actual.host_pinned);
      break;
  }
}

};  // end namespace raft
