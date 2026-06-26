/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <cmath>

namespace RAFT_EXPORT raft {
namespace core {
/**
 * @defgroup bitset Bitset
 * @{
 */
/**
 * @brief View of a RAFT Bitset.
 *
 * This lightweight structure stores a pointer to a bitset in device memory with it's length.
 * It provides a test() device function to check if a given index is set in the bitset.
 *
 * @tparam bitset_t Underlying type of the bitset array. Default is uint32_t.
 * @tparam index_t Indexing type used. Default is uint32_t.
 */
template <typename bitset_t = uint32_t, typename index_t = uint32_t>
struct bitset_view {
  static constexpr index_t bitset_element_size = sizeof(bitset_t) * 8;

  /**
   * @brief Create a bitset view from a device pointer to the bitset.
   *
   * @param bitset_ptr Device pointer to the bitset
   * @param bitset_len Number of bits in the bitset
   * @param original_nbits Original number of bits used when the bitset was created, to handle
   * potential mismatches of data types. This is useful for using ANN indexes when a bitset was
   * originally created with a different data type than the ones currently supported in cuVS ANN
   * indexes.
   */
  _RAFT_HOST_DEVICE bitset_view(bitset_t* bitset_ptr,
                                index_t bitset_len,
                                index_t original_nbits = 0)
    : bitset_ptr_{bitset_ptr}, bitset_len_{bitset_len}, original_nbits_{original_nbits}
  {
  }
  /**
   * @brief Create a bitset view from a device vector view of the bitset.
   *
   * @param bitset_span Device vector view of the bitset
   * @param bitset_len Number of bits in the bitset
   * @param original_nbits Original number of bits used when the bitset was created, to handle
   * potential mismatches of data types. This is useful for using ANN indexes when a bitset was
   * originally created with a different data type than the ones currently supported in cuVS ANN
   * indexes.
   */
  _RAFT_HOST_DEVICE bitset_view(raft::device_vector_view<bitset_t, index_t> bitset_span,
                                index_t bitset_len,
                                index_t original_nbits = 0)
    : bitset_ptr_{bitset_span.data_handle()},
      bitset_len_{bitset_len},
      original_nbits_{original_nbits}
  {
  }
  /**
   * @brief Device function to test if a given index is set in the bitset.
   *
   * @param sample_index Single index to test
   * @return bool True if index has not been unset in the bitset
   */
  inline _RAFT_HOST_DEVICE auto test(const index_t sample_index) const -> bool;
  /**
   * @brief Device function to test if a given index is set in the bitset.
   *
   * @param sample_index Single index to test
   * @return bool True if index has not been unset in the bitset
   */
  inline _RAFT_HOST_DEVICE auto operator[](const index_t sample_index) const -> bool;
  /**
   * @brief Device function to set a given index to set_value in the bitset.
   *
   * @param sample_index index to set
   * @param set_value Value to set the bit to (true or false)
   */
  inline _RAFT_DEVICE void set(const index_t sample_index, bool set_value) const;

  /**
   * @brief Get the device pointer to the bitset.
   */
  inline _RAFT_HOST_DEVICE auto data() -> bitset_t* { return bitset_ptr_; }
  inline _RAFT_HOST_DEVICE auto data() const -> const bitset_t* { return bitset_ptr_; }
  /**
   * @brief Get the number of bits of the bitset representation.
   */
  inline _RAFT_HOST_DEVICE auto size() const -> index_t { return bitset_len_; }

  /**
   * @brief Get the number of elements used by the bitset representation.
   */
  inline _RAFT_HOST_DEVICE auto n_elements() const -> index_t
  {
    return raft::div_rounding_up_safe(bitset_len_, bitset_element_size);
  }

  inline auto to_mdspan() -> raft::device_vector_view<bitset_t, index_t>
  {
    return raft::make_device_vector_view<bitset_t, index_t>(bitset_ptr_, n_elements());
  }
  inline auto to_mdspan() const -> raft::device_vector_view<const bitset_t, index_t>
  {
    return raft::make_device_vector_view<const bitset_t, index_t>(bitset_ptr_, n_elements());
  }
  /**
   * @brief Returns the number of bits set to true in count_gpu_scalar.
   *
   * @param[in] res RAFT resources
   * @param[out] count_gpu_scalar Device scalar to store the count
   */
  void count(const raft::resources& res, raft::device_scalar_view<index_t> count_gpu_scalar) const;
  /**
   * @brief Returns the number of bits set to true.
   *
   * @param res RAFT resources
   * @return index_t Number of bits set to true
   */
  auto count(const raft::resources& res) const -> index_t
  {
    auto count_gpu_scalar = raft::make_device_scalar<index_t>(res, 0.0);
    count(res, count_gpu_scalar.view());
    index_t count_cpu = 0;
    if (!resource::get_dry_run_flag(res)) {
      raft::update_host(
        &count_cpu, count_gpu_scalar.data_handle(), 1, resource::get_cuda_stream(res));
      resource::sync_stream(res);
    }
    return count_cpu;
  }

  /**
   * @brief Repeats the bitset data and copies it to the output device pointer.
   *
   * This function takes the original bitset data stored in the device memory
   * and repeats it a specified number of times into a new location in the device memory.
   * The bits are copied bit-by-bit to ensure that even if the number of bits (bitset_len_)
   * is not a multiple of the bitset element size (e.g., 32 for uint32_t), the bits are
   * tightly packed without any gaps between rows.
   *
   * @param res RAFT resources for managing CUDA streams and execution policies.
   * @param times Number of times the bitset data should be repeated in the output.
   * @param output_device_ptr Device pointer where the repeated bitset data will be stored.
   *
   * The caller must ensure that the output device pointer has enough memory allocated
   * to hold `times * bitset_len` bits, where `bitset_len` is the number of bits in the original
   * bitset. This function uses Thrust parallel algorithms to efficiently perform the operation on
   * the GPU.
   */
  void repeat(const raft::resources& res, index_t times, bitset_t* output_device_ptr) const;

  /**
   * @brief Calculate the sparsity (fraction of 0s) of the bitset.
   *
   * This function computes the sparsity of the bitset, defined as the ratio of unset bits (0s)
   * to the total number of bits in the set. If the total number of bits is zero, the function
   * returns 1.0, indicating the set is fully sparse.
   *
   * @param res RAFT resources for managing CUDA streams and execution policies.
   * @return double The sparsity of the bitset, i.e., the fraction of unset bits.
   *
   * This API will synchronize on the stream of `res`.
   */
  double sparsity(const raft::resources& res) const;

  /**
   * @brief Calculates the number of `bitset_t` elements required to store a bitset.
   *
   * This function computes the number of `bitset_t` elements needed to store a bitset, ensuring
   * that all bits are accounted for. If the bitset length is not a multiple of the `bitset_t` size
   * (in bits), the calculation rounds up to include the remaining bits in an additional `bitset_t`
   * element.
   *
   * @param bitset_len The total length of the bitset in bits.
   * @return size_t The number of `bitset_t` elements required to store the bitset.
   */
  static inline size_t eval_n_elements(size_t bitset_len)
  {
    const size_t bits_per_element = sizeof(bitset_t) * 8;
    return (bitset_len + bits_per_element - 1) / bits_per_element;
  }

  /**
   * @brief Get the original number of bits of the bitset.
   */
  auto get_original_nbits() const -> index_t { return original_nbits_; }
  void set_original_nbits(index_t original_nbits) { original_nbits_ = original_nbits; }

  /**
   * @brief Converts to a Compressed Sparse Row (CSR) format matrix.
   *
   * This method transforms the bitset view into a CSR matrix representation, where each '1' bit in
   * the bitset corresponds to a non-zero entry in the CSR matrix. The bitset format supports
   * only a single-row matrix, so if the CSR matrix requires multiple rows, the bitset data is
   * repeated for each row in the output.
   *
   * Example usage:
   *
   * @code{.cpp}
   * #include <raft/core/resource/cuda_stream.hpp>
   * #include <raft/sparse/convert/csr.cuh>
   * #include <rmm/device_uvector.hpp>
   *
   * using bitset_t = uint32_t;
   * using index_t  = int;
   * using value_t  = float;
   *
   * raft::resources handle;
   * auto stream    = resource::get_cuda_stream(handle);
   * index_t n_rows = 3;
   * index_t n_cols = 30;
   *
   * // Compute bitset size and initialize device memory
   * index_t bitset_size = (n_cols + sizeof(bitset_t) * 8 - 1) / (sizeof(bitset_t) * 8);
   * rmm::device_uvector<bitset_t> bitset_d(bitset_size, stream);
   * std::vector<bitset_t> bitset_h = {
   *   bitset_t(0b11001010),
   * };  // Example bitset, with 4 non-zero entries.
   *
   * raft::copy(bitset_d.data(), bitset_h.data(), bitset_h.size(), stream);
   *
   * // Create bitset view and CSR matrix
   * auto bitset_view = raft::core::bitset_view<bitset_t, index_t>(bitset_d.data(), n_cols);
   * auto csr = raft::make_device_csr_matrix<value_t, index_t>(handle, n_rows, n_cols, 4 * n_rows);
   *
   * // Convert bitset to CSR
   * bitset_view.to_csr(handle, csr);
   * resource::sync_stream(handle);
   *
   * // Results:
   * // csr.indptr  = [0, 4, 8, 12];
   * // csr.indices = [1, 3, 6, 7,
   * //                1, 3, 6, 7,
   * //                1, 3, 6, 7];
   * // csr.values  = [1, 1, 1, 1,
   * //                1, 1, 1, 1,
   * //                1, 1, 1, 1];
   * @endcode
   *
   * @tparam csr_matrix_t Specifies the CSR matrix type, constrained to raft::device_csr_matrix.
   *
   * @param[in] res RAFT resources for managing CUDA streams and execution policies.
   * @param[out] csr Output parameter where the resulting CSR matrix is stored. Each '1' bit in
   * the bitset corresponds to a non-zero element in the CSR matrix.
   *
   * The caller must ensure that: The `csr` matrix is pre-allocated with dimensions and non-zero
   * count matching the expected output, i.e., `nnz_for_csr = nnz_for_bitset * n_rows`.
   */
  template <typename csr_matrix_t>
  void to_csr(const raft::resources& res, csr_matrix_t& csr) const;

 private:
  bitset_t* bitset_ptr_;
  index_t bitset_len_;
  index_t original_nbits_;
};

/**
 * @brief RAFT Bitset.
 *
 * This structure encapsulates a bitset in device memory. It provides a view() method to get a
 * device-usable lightweight view of the bitset.
 * Each index is represented by a single bit in the bitset. The total number of bytes used is
 * ceil(bitset_len / 8).
 * @tparam bitset_t Underlying type of the bitset array. Default is uint32_t.
 * @tparam index_t Indexing type used. Default is uint32_t.
 */
template <typename bitset_t = uint32_t, typename index_t = uint32_t>
struct bitset {
  static constexpr index_t bitset_element_size = sizeof(bitset_t) * 8;

  /**
   * @brief Construct a new bitset object with a list of indices to unset.
   *
   * @param res RAFT resources
   * @param mask_index List of indices to unset in the bitset
   * @param bitset_len Length of the bitset
   * @param default_value Default value to set the bits to. Default is true.
   */
  bitset(const raft::resources& res,
         raft::device_vector_view<const index_t, index_t> mask_index,
         index_t bitset_len,
         bool default_value = true);

  /**
   * @brief Construct a new bitset object
   *
   * @param res RAFT resources
   * @param bitset_len Length of the bitset
   * @param default_value Default value to set the bits to. Default is true.
   */
  bitset(const raft::resources& res, index_t bitset_len, bool default_value = true);
  // Disable copy constructor
  bitset(const bitset&)            = delete;
  bitset(bitset&&)                 = default;
  bitset& operator=(const bitset&) = delete;
  bitset& operator=(bitset&&)      = default;

  /**
   * @brief Create a device-usable view of the bitset.
   *
   * @return bitset_view<bitset_t, index_t>
   */
  inline auto view() -> raft::core::bitset_view<bitset_t, index_t>
  {
    return bitset_view<bitset_t, index_t>(to_mdspan(), bitset_len_);
  }
  [[nodiscard]] inline auto view() const -> raft::core::bitset_view<const bitset_t, index_t>
  {
    return bitset_view<const bitset_t, index_t>(to_mdspan(), bitset_len_);
  }

  /**
   * @brief Get the device pointer to the bitset.
   */
  inline auto data() -> bitset_t* { return bitset_.data(); }
  inline auto data() const -> const bitset_t* { return bitset_.data(); }
  /**
   * @brief Get the number of bits of the bitset representation.
   */
  inline auto size() const -> index_t { return bitset_len_; }

  /**
   * @brief Get the number of elements used by the bitset representation.
   */
  inline auto n_elements() const -> index_t
  {
    return raft::div_rounding_up_safe(bitset_len_, bitset_element_size);
  }

  /** @brief Get an mdspan view of the current bitset */
  inline auto to_mdspan() -> raft::device_vector_view<bitset_t, index_t>
  {
    return raft::make_device_vector_view<bitset_t, index_t>(bitset_.data(), n_elements());
  }
  [[nodiscard]] inline auto to_mdspan() const -> raft::device_vector_view<const bitset_t, index_t>
  {
    return raft::make_device_vector_view<const bitset_t, index_t>(bitset_.data(), n_elements());
  }

  /** @brief Resize the bitset. If the requested size is larger, new memory is allocated and set to
   * the default value.
   * @param res RAFT resources
   * @param new_bitset_len new size of the bitset
   * @param default_value default value to initialize the new bits to
   */
  void resize(const raft::resources& res, index_t new_bitset_len, bool default_value = true);

  /**
   * @brief Test a list of indices in a bitset.
   *
   * @tparam output_t Output type of the test. Default is bool.
   * @param res RAFT resources
   * @param queries List of indices to test
   * @param output List of outputs
   */
  template <typename output_t = bool>
  void test(const raft::resources& res,
            raft::device_vector_view<const index_t, index_t> queries,
            raft::device_vector_view<output_t, index_t> output) const;
  /**
   * @brief Set a list of indices in a bitset to set_value.
   *
   * @param res RAFT resources
   * @param mask_index indices to remove from the bitset
   * @param set_value Value to set the bits to (true or false)
   */
  void set(const raft::resources& res,
           raft::device_vector_view<const index_t, index_t> mask_index,
           bool set_value = false);
  /**
   * @brief Flip all the bits in a bitset.
   * @param res RAFT resources
   */
  void flip(const raft::resources& res);
  /**
   * @brief Reset the bits in a bitset.
   *
   * @param res RAFT resources
   * @param default_value Value to set the bits to (true or false)
   */
  void reset(const raft::resources& res, bool default_value = true);
  /**
   * @brief Returns the number of bits set to true in count_gpu_scalar.
   *
   * @param[in] res RAFT resources
   * @param[out] count_gpu_scalar Device scalar to store the count
   */
  void count(const raft::resources& res, raft::device_scalar_view<index_t> count_gpu_scalar);
  /**
   * @brief Returns the number of bits set to true.
   *
   * @param res RAFT resources
   * @return index_t Number of bits set to true
   */
  auto count(const raft::resources& res) -> index_t
  {
    auto count_gpu_scalar = raft::make_device_scalar<index_t>(res, 0.0);
    count(res, count_gpu_scalar.view());
    index_t count_cpu = 0;
    if (!resource::get_dry_run_flag(res)) {
      raft::update_host(
        &count_cpu, count_gpu_scalar.data_handle(), 1, resource::get_cuda_stream(res));
      resource::sync_stream(res);
    }
    return count_cpu;
  }
  /**
   * @brief Checks if any of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool any(const raft::resources& res) { return count(res) > 0; }
  /**
   * @brief Checks if all of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool all(const raft::resources& res) { return count(res) == bitset_len_; }
  /**
   * @brief Checks if none of the bits are set to true in the bitset.
   * @param res RAFT resources
   */
  bool none(const raft::resources& res) { return count(res) == 0; }

 private:
  raft::device_uvector<bitset_t> bitset_;
  index_t bitset_len_;
};

/** @} */
}  // namespace core
}  // namespace RAFT_EXPORT raft
