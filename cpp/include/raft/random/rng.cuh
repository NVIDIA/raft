/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/rng_impl.cuh"
#include "detail/rng_impl_deprecated.cuh"  // necessary for now (to be removed)
#include "rng_state.hpp"

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>

#include <cassert>
#include <optional>
#include <type_traits>
#include <variant>

namespace raft {
namespace random {

/**
 * \defgroup univariate_random_sampling Univariate random sampling
 * @{
 */

/**
 * @brief Generate uniformly distributed numbers in the given range
 *
 * @tparam OutputValueType Data type of output random number
 * @tparam Index Data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output array
 * @param[in] start start of the range
 * @param[in] end end of the range
 */
template <typename OutputValueType, typename IndexType>
void uniform(raft::resources const& handle,
             RngState& rng_state,
             raft::device_vector_view<OutputValueType, IndexType> out,
             OutputValueType start,
             OutputValueType end)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::uniform(
    rng_state, out.data_handle(), out.extent(0), start, end, resource::get_cuda_stream(handle));
}

/**
 * @}
 */

/**
 * @brief Legacy overload of `uniform` taking raw pointers
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] start start of the range
 * @param[in] end end of the range
 */
template <typename OutType, typename LenType = int>
void uniform(raft::resources const& handle,
             RngState& rng_state,
             OutType* ptr,
             LenType len,
             OutType start,
             OutType end)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::uniform(rng_state, ptr, len, start, end, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate uniformly distributed integers in the given range
 *
 * @tparam OutputValueType Integral type; value type of the output vector
 * @tparam IndexType Type used to represent length of the output vector
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output vector of random numbers
 * @param[in] start start of the range
 * @param[in] end end of the range
 */
template <typename OutputValueType, typename IndexType>
void uniformInt(raft::resources const& handle,
                RngState& rng_state,
                raft::device_vector_view<OutputValueType, IndexType> out,
                OutputValueType start,
                OutputValueType end)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  static_assert(
    std::is_same<OutputValueType, typename std::remove_cv<OutputValueType>::type>::value,
    "uniformInt: The output vector must be a view of nonconst, "
    "so that we can write to it.");
  static_assert(std::is_integral<OutputValueType>::value,
                "uniformInt: The elements of the output vector must have integral type.");
  detail::uniformInt(
    rng_state, out.data_handle(), out.extent(0), start, end, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `uniformInt`
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] start start of the range
 * @param[in] end end of the range
 */
template <typename OutType, typename LenType = int>
void uniformInt(raft::resources const& handle,
                RngState& rng_state,
                OutType* ptr,
                LenType len,
                OutType start,
                OutType end)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::uniformInt(rng_state, ptr, len, start, end, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate normal distributed numbers
 *   with a given mean and standard deviation
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output array
 * @param[in] mu mean of the distribution
 * @param[in] sigma std-dev of the distribution
 */
template <typename OutputValueType, typename IndexType>
void normal(raft::resources const& handle,
            RngState& rng_state,
            raft::device_vector_view<OutputValueType, IndexType> out,
            OutputValueType mu,
            OutputValueType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::normal(
    rng_state, out.data_handle(), out.extent(0), mu, sigma, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `normal`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] mu mean of the distribution
 * @param[in] sigma std-dev of the distribution
 */
template <typename OutType, typename LenType = int>
void normal(raft::resources const& handle,
            RngState& rng_state,
            OutType* ptr,
            LenType len,
            OutType mu,
            OutType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::normal(rng_state, ptr, len, mu, sigma, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate normal distributed integers
 *
 * @tparam OutputValueType Integral type; value type of the output vector
 * @tparam IndexType Integral type of the output vector's length
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output array
 * @param[in] mu mean of the distribution
 * @param[in] sigma standard deviation of the distribution
 */
template <typename OutputValueType, typename IndexType>
void normalInt(raft::resources const& handle,
               RngState& rng_state,
               raft::device_vector_view<OutputValueType, IndexType> out,
               OutputValueType mu,
               OutputValueType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  static_assert(
    std::is_same<OutputValueType, typename std::remove_cv<OutputValueType>::type>::value,
    "normalInt: The output vector must be a view of nonconst, "
    "so that we can write to it.");
  static_assert(std::is_integral<OutputValueType>::value,
                "normalInt: The output vector's value type must be an integer.");

  detail::normalInt(
    rng_state, out.data_handle(), out.extent(0), mu, sigma, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `normalInt`
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] mu mean of the distribution
 * @param[in] sigma std-dev of the distribution
 */
template <typename IntType, typename LenType = int>
void normalInt(raft::resources const& handle,
               RngState& rng_state,
               IntType* ptr,
               LenType len,
               IntType mu,
               IntType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::normalInt(rng_state, ptr, len, mu, sigma, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate normal distributed table according to the given set of
 * means and scalar standard deviations.
 *
 * Each row in this table conforms to a normally distributed n-dim vector
 * whose mean is the input vector and standard deviation is the corresponding
 * vector or scalar. Correlations among the dimensions itself are assumed to
 * be absent.
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[in] mu_vec mean vector (of length `out.extent(1)`)
 * @param[in] sigma Either the standard-deviation vector
 *            (of length `out.extent(1)`) of each component,
 *            or a scalar standard deviation for all components.
 * @param[out] out the output table
 */
template <typename OutputValueType, typename IndexType>
void normalTable(
  raft::resources const& handle,
  RngState& rng_state,
  raft::device_vector_view<const OutputValueType, IndexType> mu_vec,
  std::variant<raft::device_vector_view<const OutputValueType, IndexType>, OutputValueType> sigma,
  raft::device_matrix_view<OutputValueType, IndexType, raft::row_major> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  const OutputValueType* sigma_vec_ptr = nullptr;
  OutputValueType sigma_value{};

  using sigma_vec_type = raft::device_vector_view<const OutputValueType, IndexType>;
  if (std::holds_alternative<sigma_vec_type>(sigma)) {
    auto sigma_vec = std::get<sigma_vec_type>(sigma);
    RAFT_EXPECTS(sigma_vec.extent(0) == out.extent(1),
                 "normalTable: The sigma vector "
                 "has length %zu, which does not equal the number of columns "
                 "in the output table %zu.",
                 static_cast<size_t>(sigma_vec.extent(0)),
                 static_cast<size_t>(out.extent(1)));
    // The extra length check makes this work even if sigma_vec views a std::vector,
    // where .data() need not return nullptr even if .size() is zero.
    sigma_vec_ptr = sigma_vec.extent(0) == 0 ? nullptr : sigma_vec.data_handle();
  } else {
    sigma_value = std::get<OutputValueType>(sigma);
  }

  RAFT_EXPECTS(mu_vec.extent(0) == out.extent(1),
               "normalTable: The mu vector "
               "has length %zu, which does not equal the number of columns "
               "in the output table %zu.",
               static_cast<size_t>(mu_vec.extent(0)),
               static_cast<size_t>(out.extent(1)));

  detail::normalTable(rng_state,
                      out.data_handle(),
                      out.extent(0),
                      out.extent(1),
                      mu_vec.data_handle(),
                      sigma_vec_ptr,
                      sigma_value,
                      resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `normalTable`.
 *
 * Each row in this table conforms to a normally distributed n-dim vector
 * whose mean is the input vector and standard deviation is the corresponding
 * vector or scalar. Correlations among the dimensions itself are assumed to
 * be absent.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output table (dim = n_rows x n_cols)
 * @param[in] n_rows number of rows in the table
 * @param[in] n_cols number of columns in the table
 * @param[in] mu_vec mean vector (dim = n_cols x 1).
 * @param[in] sigma_vec std-dev vector of each component (dim = n_cols x 1). Pass
 * a nullptr to use the same scalar 'sigma' across all components
 * @param[in] sigma scalar sigma to be used if 'sigma_vec' is nullptr
 */
template <typename OutType, typename LenType = int>
void normalTable(raft::resources const& handle,
                 RngState& rng_state,
                 OutType* ptr,
                 LenType n_rows,
                 LenType n_cols,
                 const OutType* mu_vec,
                 const OutType* sigma_vec,
                 OutType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::normalTable(
    rng_state, ptr, n_rows, n_cols, mu_vec, sigma_vec, sigma, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Fill a vector with the given value
 *
 * @tparam OutputValueType Value type of the output vector
 * @tparam IndexType Integral type used to represent length of the output vector
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[in] val value with which to fill the output vector
 * @param[out] out the output vector
 */
template <typename OutputValueType, typename IndexType>
void fill(raft::resources const& handle,
          RngState& rng_state,
          OutputValueType val,
          raft::device_vector_view<OutputValueType, IndexType> out)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::fill(rng_state, out.data_handle(), out.extent(0), val, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `fill`
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] val value to be filled
 */
template <typename OutType, typename LenType = int>
void fill(
  raft::resources const& handle, RngState& rng_state, OutType* ptr, LenType len, OutType val)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::fill(rng_state, ptr, len, val, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate bernoulli distributed boolean array
 *
 * @tparam OutputValueType Type of each element of the output vector;
 *         must be able to represent boolean values (e.g., `bool`)
 * @tparam IndexType Integral type of the output vector's length
 * @tparam Type Data type in which to compute the probabilities
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output vector
 * @param[in] prob coin-toss probability for heads
 */
template <typename OutputValueType, typename IndexType, typename Type>
void bernoulli(raft::resources const& handle,
               RngState& rng_state,
               raft::device_vector_view<OutputValueType, IndexType> out,
               Type prob)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::bernoulli(
    rng_state, out.data_handle(), out.extent(0), prob, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `bernoulli`
 *
 * @tparam Type    data type in which to compute the probabilities
 * @tparam OutType output data type
 * @tparam LenType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr    the output array
 * @param[in]  len    the number of elements in the output
 * @param[in]  prob   coin-toss probability for heads
 */
template <typename Type, typename OutType = bool, typename LenType = int>
void bernoulli(
  raft::resources const& handle, RngState& rng_state, OutType* ptr, LenType len, Type prob)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::bernoulli(rng_state, ptr, len, prob, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate bernoulli distributed array and applies scale
 *
 * @tparam OutputValueType Data type in which to compute the probabilities
 * @tparam IndexType Integral type of the output vector's length
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output vector
 * @param[in] prob coin-toss probability for heads
 * @param[in] scale scaling factor
 */
template <typename OutputValueType, typename IndexType>
void scaled_bernoulli(raft::resources const& handle,
                      RngState& rng_state,
                      raft::device_vector_view<OutputValueType, IndexType> out,
                      OutputValueType prob,
                      OutputValueType scale)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::scaled_bernoulli(
    rng_state, out.data_handle(), out.extent(0), prob, scale, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `scaled_bernoulli`
 *
 * @tparam OutType data type in which to compute the probabilities
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] prob coin-toss probability for heads
 * @param[in] scale scaling factor
 */
template <typename OutType, typename LenType = int>
void scaled_bernoulli(raft::resources const& handle,
                      RngState& rng_state,
                      OutType* ptr,
                      LenType len,
                      OutType prob,
                      OutType scale)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::scaled_bernoulli(rng_state, ptr, len, prob, scale, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate Gumbel distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] mu mean value
 * @param[in] beta scale value
 * @note https://en.wikipedia.org/wiki/Gumbel_distribution
 */
template <typename OutputValueType, typename IndexType = int>
void gumbel(raft::resources const& handle,
            RngState& rng_state,
            raft::device_vector_view<OutputValueType, IndexType> out,
            OutputValueType mu,
            OutputValueType beta)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::gumbel(
    rng_state, out.data_handle(), out.extent(0), mu, beta, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `gumbel`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] mu mean value
 * @param[in] beta scale value
 * @note https://en.wikipedia.org/wiki/Gumbel_distribution
 */
template <typename OutType, typename LenType = int>
void gumbel(raft::resources const& handle,
            RngState& rng_state,
            OutType* ptr,
            LenType len,
            OutType mu,
            OutType beta)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::gumbel(rng_state, ptr, len, mu, beta, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate lognormal distributed numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out the output array
 * @param[in] mu mean of the distribution
 * @param[in] sigma standard deviation of the distribution
 */
template <typename OutputValueType, typename IndexType>
void lognormal(raft::resources const& handle,
               RngState& rng_state,
               raft::device_vector_view<OutputValueType, IndexType> out,
               OutputValueType mu,
               OutputValueType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::lognormal(
    rng_state, out.data_handle(), out.extent(0), mu, sigma, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `lognormal`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr the output array
 * @param[in] len the number of elements in the output
 * @param[in] mu mean of the distribution
 * @param[in] sigma standard deviation of the distribution
 */
template <typename OutType, typename LenType = int>
void lognormal(raft::resources const& handle,
               RngState& rng_state,
               OutType* ptr,
               LenType len,
               OutType mu,
               OutType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::lognormal(rng_state, ptr, len, mu, sigma, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate logistic distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] mu mean value
 * @param[in] scale scale value
 */
template <typename OutputValueType, typename IndexType = int>
void logistic(raft::resources const& handle,
              RngState& rng_state,
              raft::device_vector_view<OutputValueType, IndexType> out,
              OutputValueType mu,
              OutputValueType scale)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::logistic(
    rng_state, out.data_handle(), out.extent(0), mu, scale, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `logistic`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] mu mean value
 * @param[in] scale scale value
 */
template <typename OutType, typename LenType = int>
void logistic(raft::resources const& handle,
              RngState& rng_state,
              OutType* ptr,
              LenType len,
              OutType mu,
              OutType scale)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::logistic(rng_state, ptr, len, mu, scale, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate exponentially distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] lambda the exponential distribution's lambda parameter
 */
template <typename OutputValueType, typename IndexType>
void exponential(raft::resources const& handle,
                 RngState& rng_state,
                 raft::device_vector_view<OutputValueType, IndexType> out,
                 OutputValueType lambda)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::exponential(
    rng_state, out.data_handle(), out.extent(0), lambda, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `exponential`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] lambda the exponential distribution's lambda parameter
 */
template <typename OutType, typename LenType = int>
void exponential(
  raft::resources const& handle, RngState& rng_state, OutType* ptr, LenType len, OutType lambda)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::exponential(rng_state, ptr, len, lambda, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate rayleigh distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] sigma the distribution's sigma parameter
 */
template <typename OutputValueType, typename IndexType>
void rayleigh(raft::resources const& handle,
              RngState& rng_state,
              raft::device_vector_view<OutputValueType, IndexType> out,
              OutputValueType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::rayleigh(
    rng_state, out.data_handle(), out.extent(0), sigma, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `rayleigh`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] sigma the distribution's sigma parameter
 */
template <typename OutType, typename LenType = int>
void rayleigh(
  raft::resources const& handle, RngState& rng_state, OutType* ptr, LenType len, OutType sigma)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::rayleigh(rng_state, ptr, len, sigma, resource::get_cuda_stream(handle));
}
/**
 * @ingroup univariate_random_sampling
 * @brief Generate laplace distributed random numbers
 *
 * @tparam OutputValueType data type of output random number
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] mu the mean
 * @param[in] scale the scale
 */
template <typename OutputValueType, typename IndexType>
void laplace(raft::resources const& handle,
             RngState& rng_state,
             raft::device_vector_view<OutputValueType, IndexType> out,
             OutputValueType mu,
             OutputValueType scale)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::laplace(
    rng_state, out.data_handle(), out.extent(0), mu, scale, resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy raw pointer overload of `laplace`.
 *
 * @tparam OutType data type of output random number
 * @tparam LenType data type used to represent length of the arrays
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] ptr output array
 * @param[in] len number of elements in the output array
 * @param[in] mu the mean
 * @param[in] scale the scale
 */
template <typename OutType, typename LenType = int>
void laplace(raft::resources const& handle,
             RngState& rng_state,
             OutType* ptr,
             LenType len,
             OutType mu,
             OutType scale)
{
  if (resource::get_dry_run_flag(handle)) { return; }
  detail::laplace(rng_state, ptr, len, mu, scale, resource::get_cuda_stream(handle));
}

/**
 * @ingroup univariate_random_sampling
 * @brief Generate random integers, where the probability of i is weights[i]/sum(weights)
 *
 * Usage example:
 * @code{.cpp}
 *  #include <raft/core/device_mdarray.hpp>
 *  #include <raft/core/resources.hpp>
 *  #include <raft/random/rng.cuh>
 *
 *  raft::resources handle;
 *  ...
 *  raft::random::RngState rng(seed);
 *  auto indices = raft::make_device_vector<int>(handle, n_samples);
 *  raft::random::discrete(handle, rng, indices.view(), weights);
 * @endcode
 *
 * @tparam OutType integer output type
 * @tparam WeightType weight type
 * @tparam IndexType data type used to represent length of the arrays
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output array
 * @param[in] weights weight array
 */
template <typename OutType, typename WeightType, typename IndexType>
std::enable_if_t<std::is_integral_v<OutType>> discrete(
  raft::resources const& handle,
  RngState& rng_state,
  raft::device_vector_view<OutType, IndexType> out,
  raft::device_vector_view<const WeightType, IndexType> weights)
{
  detail::discrete(resource::get_dry_run_flag(handle),
                   rng_state,
                   out.data_handle(),
                   weights.data_handle(),
                   out.extent(0),
                   weights.extent(0),
                   resource::get_cuda_stream(handle));
}

/**
 * @brief Legacy version of @c sample_without_replacement (see above)
 *   that takes raw arrays instead of device mdspan.
 *
 * @tparam DataT data type
 * @tparam WeightsT weights type
 * @tparam IdxT index type
 *
 * @param[in] handle raft handle for resource management
 * @param[in] rng_state random number generator state
 * @param[out] out output sampled array (of length 'sampledLen')
 * @param[out] outIdx indices of the sampled array (of length 'sampledLen'). Pass
 * a nullptr if this is not required.
 * @param[in] in input array to be sampled (of length 'len')
 * @param[in] wts weights array (of length 'len'). Pass a nullptr if uniform
 * sampling is desired
 * @param[in] sampledLen output sampled array length
 * @param[in] len input array length
 */
template <typename DataT, typename WeightsT, typename IdxT = int>
void sampleWithoutReplacement(raft::resources const& handle,
                              RngState& rng_state,
                              DataT* out,
                              IdxT* outIdx,
                              const DataT* in,
                              const WeightsT* wts,
                              IdxT sampledLen,
                              IdxT len)
{
  detail::sampleWithoutReplacement(resource::get_dry_run_flag(handle),
                                   rng_state,
                                   out,
                                   outIdx,
                                   in,
                                   wts,
                                   sampledLen,
                                   len,
                                   resource::get_cuda_stream(handle));
}

/** @brief Sample from range 0..N-1.
 *
 * Elements are sampled uniformly. The method aims to sample without replacement,
 * but there is a small probability of a few having duplicate elements.
 *
 * The algorithm will allocate a workspace of size 4*n_samples*sizeof(IdxT) internally.
 *
 * We use max N random numbers. Depending on how large n_samples is w.r.t to N, we
 * either use rejection sampling, or sort the [0..N-1] values using random keys.
 *
 * @tparam IdxT type of indices that we sample
 * @tparam MatIdxT extent type of the returned mdarray
 *
 * @param res RAFT resource handle
 * @param state random number generator state
 * @param N number of elements to sample from. We will sample values in range 0..N-1.
 * @param n_samples number of samples to return
 *
 * @return device mdarray with the random samples
 */
template <typename IdxT, typename MatIdxT = IdxT>
auto excess_subsample(raft::resources const& res, RngState& state, IdxT N, IdxT n_samples)
{
  return detail::excess_subsample(res, state, N, n_samples);
}

/**
 * @brief Generates the 'a' and 'b' parameters for a modulo affine
 *        transformation equation: `(ax + b) % n`
 *
 * @tparam IdxT integer type
 *
 * @param[in] rng_state random number generator state
 * @param[in]  n the modulo range
 * @param[out] a slope parameter
 * @param[out] b intercept parameter
 */
template <typename IdxT>
void affine_transform_params(RngState const& rng_state, IdxT n, IdxT& a, IdxT& b)
{
  detail::affine_transform_params(rng_state, n, a, b);
}

///////////////////////////////////////////////////////////////////////////
// Everything below this point is deprecated and will be removed         //
///////////////////////////////////////////////////////////////////////////

// without the macro, clang-format seems to go insane
#define DEPR [[deprecated("Use 'RngState' with the new flat functions instead")]]

using detail::bernoulli;
using detail::exponential;
using detail::fill;
using detail::gumbel;
using detail::laplace;
using detail::logistic;
using detail::lognormal;
using detail::normal;
using detail::normalInt;
using detail::normalTable;
using detail::rayleigh;
using detail::scaled_bernoulli;
using detail::uniform;
using detail::uniformInt;

using detail::sampleWithoutReplacement;

class DEPR Rng : public detail::RngImpl {
 public:
  /**
   * @brief ctor
   * @param _s 64b seed used to initialize the RNG
   * @param _t backend device RNG generator type
   * @note Refer to the `Rng::seed` method for details about seeding the engine
   */
  Rng(uint64_t _s, GeneratorType _t = GenPhilox) : detail::RngImpl(_s, _t) {}

  /**
   * @brief Generates the 'a' and 'b' parameters for a modulo affine
   *        transformation equation: `(ax + b) % n`
   *
   * @tparam IdxT integer type
   *
   * @param[in]  n the modulo range
   * @param[out] a slope parameter
   * @param[out] b intercept parameter
   */
  template <typename IdxT>
  void affine_transform_params(IdxT n, IdxT& a, IdxT& b)
  {
    detail::RngImpl::affine_transform_params(n, a, b);
  }

  /**
   * @brief Generate uniformly distributed numbers in the given range
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param start start of the range
   * @param end end of the range
   * @param stream stream where to launch the kernel
   * @{
   */
  template <typename OutType, typename LenType = int>
  void uniform(OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
  {
    detail::RngImpl::uniform(ptr, len, start, end, stream);
  }

  template <typename OutType, typename LenType = int>
  void uniformInt(OutType* ptr, LenType len, OutType start, OutType end, cudaStream_t stream)
  {
    detail::RngImpl::uniformInt(ptr, len, start, end, stream);
  }
  /** @} */

  /**
   * @brief Generate normal distributed numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param mu mean of the distribution
   * @param sigma std-dev of the distribution
   * @param stream stream where to launch the kernel
   * @{
   */
  template <typename OutType, typename LenType = int>
  void normal(OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
  {
    detail::RngImpl::normal(ptr, len, mu, sigma, stream);
  }

  template <typename IntType, typename LenType = int>
  void normalInt(IntType* ptr, LenType len, IntType mu, IntType sigma, cudaStream_t stream)
  {
    detail::RngImpl::normalInt(ptr, len, mu, sigma, stream);
  }
  /** @} */

  /**
   * @brief Generate normal distributed table according to the given set of
   * means and scalar standard deviations.
   *
   * Each row in this table conforms to a normally distributed n-dim vector
   * whose mean is the input vector and standard deviation is the corresponding
   * vector or scalar. Correlations among the dimensions itself is assumed to
   * be absent.
   *
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output table (dim = n_rows x n_cols)
   * @param n_rows number of rows in the table
   * @param n_cols number of columns in the table
   * @param mu_vec mean vector (dim = n_cols x 1).
   * @param sigma_vec std-dev vector of each component (dim = n_cols x 1). Pass
   * a nullptr to use the same scalar 'sigma' across all components
   * @param sigma scalar sigma to be used if 'sigma_vec' is nullptr
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void normalTable(OutType* ptr,
                   LenType n_rows,
                   LenType n_cols,
                   const OutType* mu_vec,
                   const OutType* sigma_vec,
                   OutType sigma,
                   cudaStream_t stream)
  {
    detail::RngImpl::normalTable(ptr, n_rows, n_cols, mu_vec, sigma_vec, sigma, stream);
  }

  /**
   * @brief Fill an array with the given value
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param val value to be filled
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void fill(OutType* ptr, LenType len, OutType val, cudaStream_t stream)
  {
    detail::RngImpl::fill(ptr, len, val, stream);
  }

  /**
   * @brief Generate bernoulli distributed boolean array
   *
   * @tparam Type    data type in which to compute the probabilities
   * @tparam OutType output data type
   * @tparam LenType data type used to represent length of the arrays
   *
   * @param[out] ptr    the output array
   * @param[in]  len    the number of elements in the output
   * @param[in]  prob   coin-toss probability for heads
   * @param[in]  stream stream where to launch the kernel
   */
  template <typename Type, typename OutType = bool, typename LenType = int>
  void bernoulli(OutType* ptr, LenType len, Type prob, cudaStream_t stream)
  {
    detail::RngImpl::bernoulli(ptr, len, prob, stream);
  }

  /**
   * @brief Generate bernoulli distributed array and applies scale
   * @tparam Type data type in which to compute the probabilities
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param prob coin-toss probability for heads
   * @param scale scaling factor
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void scaled_bernoulli(OutType* ptr, LenType len, OutType prob, OutType scale, cudaStream_t stream)
  {
    detail::RngImpl::scaled_bernoulli(ptr, len, prob, scale, stream);
  }

  /**
   * @brief Generate Gumbel distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu mean value
   * @param beta scale value
   * @param stream stream where to launch the kernel
   * @note https://en.wikipedia.org/wiki/Gumbel_distribution
   */
  template <typename OutType, typename LenType = int>
  void gumbel(OutType* ptr, LenType len, OutType mu, OutType beta, cudaStream_t stream)
  {
    detail::RngImpl::gumbel(ptr, len, mu, beta, stream);
  }

  /**
   * @brief Generate lognormal distributed numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr the output array
   * @param len the number of elements in the output
   * @param mu mean of the distribution
   * @param sigma std-dev of the distribution
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void lognormal(OutType* ptr, LenType len, OutType mu, OutType sigma, cudaStream_t stream)
  {
    detail::RngImpl::lognormal(ptr, len, mu, sigma, stream);
  }

  /**
   * @brief Generate logistic distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu mean value
   * @param scale scale value
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void logistic(OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
  {
    detail::RngImpl::logistic(ptr, len, mu, scale, stream);
  }

  /**
   * @brief Generate exponentially distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param lambda the lambda
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void exponential(OutType* ptr, LenType len, OutType lambda, cudaStream_t stream)
  {
    detail::RngImpl::exponential(ptr, len, lambda, stream);
  }

  /**
   * @brief Generate rayleigh distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param sigma the sigma
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void rayleigh(OutType* ptr, LenType len, OutType sigma, cudaStream_t stream)
  {
    detail::RngImpl::rayleigh(ptr, len, sigma, stream);
  }

  /**
   * @brief Generate laplace distributed random numbers
   * @tparam Type data type of output random number
   * @tparam LenType data type used to represent length of the arrays
   * @param ptr output array
   * @param len number of elements in the output array
   * @param mu the mean
   * @param scale the scale
   * @param stream stream where to launch the kernel
   */
  template <typename OutType, typename LenType = int>
  void laplace(OutType* ptr, LenType len, OutType mu, OutType scale, cudaStream_t stream)
  {
    detail::RngImpl::laplace(ptr, len, mu, scale, stream);
  }

  void advance(uint64_t max_streams, uint64_t max_calls_per_subsequence)
  {
    detail::RngImpl::advance(max_streams, max_calls_per_subsequence);
  }

  /**
   * @brief Sample the input array without replacement, optionally based on the
   * input weight vector for each element in the array
   *
   * Implementation here is based on the `one-pass sampling` algo described here:
   * https://www.ethz.ch/content/dam/ethz/special-interest/baug/ivt/ivt-dam/vpl/reports/1101-1200/ab1141.pdf
   *
   * @note In the sampled array the elements which are picked will always appear
   * in the increasing order of their weights as computed using the exponential
   * distribution. So, if you're particular about the order (for eg. array
   * permutations), then this might not be the right choice!
   *
   * @tparam DataT data type
   * @tparam WeightsT weights type
   * @tparam IdxT index type
   * @param handle
   * @param out output sampled array (of length 'sampledLen')
   * @param outIdx indices of the sampled array (of length 'sampledLen'). Pass
   * a nullptr if this is not required.
   * @param in input array to be sampled (of length 'len')
   * @param wts weights array (of length 'len'). Pass a nullptr if uniform
   * sampling is desired
   * @param sampledLen output sampled array length
   * @param len input array length
   * @param stream cuda stream
   */
  template <typename DataT, typename WeightsT, typename IdxT = int>
  void sampleWithoutReplacement(raft::resources const& handle,
                                DataT* out,
                                IdxT* outIdx,
                                const DataT* in,
                                const WeightsT* wts,
                                IdxT sampledLen,
                                IdxT len,
                                cudaStream_t stream)
  {
    detail::RngImpl::sampleWithoutReplacement(
      handle, out, outIdx, in, wts, sampledLen, len, stream);
  }
};

#undef DEPR

};  // namespace random
}  // namespace raft
