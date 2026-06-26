/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/bitset.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/operators.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/map.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/util/device_atomics.cuh>
#include <raft/util/popc.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace RAFT_EXPORT raft {
namespace core {

template <typename index_t>
_RAFT_HOST_DEVICE void inline compute_original_nbits_position(const index_t original_nbits,
                                                              const index_t nbits,
                                                              const index_t sample_index,
                                                              index_t& new_bit_index,
                                                              index_t& new_bit_offset)
{
  const index_t original_bit_index  = sample_index / original_nbits;
  const index_t original_bit_offset = sample_index % original_nbits;
  new_bit_index                     = original_bit_index * original_nbits / nbits;
  new_bit_offset                    = 0;
  if (original_nbits > nbits) {
    new_bit_index += original_bit_offset / nbits;
    new_bit_offset = original_bit_offset % nbits;
  } else {
    index_t ratio = nbits / original_nbits;
    new_bit_offset += (original_bit_index % ratio) * original_nbits;
    new_bit_offset += original_bit_offset % nbits;
  }
}

template <typename bitset_t, typename index_t>
_RAFT_HOST_DEVICE inline bool bitset_view<bitset_t, index_t>::test(const index_t sample_index) const
{
  const index_t nbits = sizeof(bitset_t) * 8;
  index_t bit_index   = 0;
  index_t bit_offset  = 0;
  if (original_nbits_ == 0 || nbits == original_nbits_) {
    bit_index  = sample_index / bitset_element_size;
    bit_offset = sample_index % bitset_element_size;
  } else {
    compute_original_nbits_position(original_nbits_, nbits, sample_index, bit_index, bit_offset);
  }
  const bitset_t bit_element = bitset_ptr_[bit_index];
  const bool is_bit_set      = (bit_element & (bitset_t{1} << bit_offset)) != 0;
  return is_bit_set;
}

template <typename bitset_t, typename index_t>
_RAFT_HOST_DEVICE bool bitset_view<bitset_t, index_t>::operator[](const index_t sample_index) const
{
  return test(sample_index);
}

template <typename bitset_t, typename index_t>
_RAFT_DEVICE void bitset_view<bitset_t, index_t>::set(const index_t sample_index,
                                                      bool set_value) const
{
  const index_t nbits = sizeof(bitset_t) * 8;
  index_t bit_index   = 0;
  index_t bit_offset  = 0;

  if (original_nbits_ == 0 || nbits == original_nbits_) {
    bit_index  = sample_index / bitset_element_size;
    bit_offset = sample_index % bitset_element_size;
  } else {
    compute_original_nbits_position(original_nbits_, nbits, sample_index, bit_index, bit_offset);
  }
  const bitset_t bitmask = bitset_t{1} << bit_offset;
  if (set_value) {
    atomicOr(bitset_ptr_ + bit_index, bitmask);
  } else {
    const bitset_t bitmask2 = ~bitmask;
    atomicAnd(bitset_ptr_ + bit_index, bitmask2);
  }
}

template <typename bitset_t, typename index_t>
void bitset_view<bitset_t, index_t>::count(const raft::resources& res,
                                           raft::device_scalar_view<index_t> count_gpu_scalar) const
{
  auto max_len = raft::make_host_scalar_view<const index_t, index_t>(&bitset_len_);
  auto values  = raft::make_device_vector_view<const bitset_t, index_t>(bitset_ptr_, n_elements());
  raft::popc(res, values, max_len, count_gpu_scalar);
}

template <typename bitset_t, typename index_t>
RAFT_KERNEL bitset_repeat_kernel(const bitset_t* src,
                                 bitset_t* output,
                                 index_t src_bit_len,
                                 index_t repeat_times)
{
  constexpr index_t bits_per_element = sizeof(bitset_t) * 8;
  int output_idx                     = blockIdx.x * blockDim.x + threadIdx.x;

  index_t total_bits  = src_bit_len * repeat_times;
  index_t output_size = (total_bits + bits_per_element - 1) / bits_per_element;
  index_t src_size    = (src_bit_len + bits_per_element - 1) / bits_per_element;

  if (output_idx < output_size) {
    bitset_t result     = 0;
    index_t bit_written = 0;

    index_t start_bit = output_idx * bits_per_element;

    while (bit_written < bits_per_element && start_bit + bit_written < total_bits) {
      index_t bit_idx      = (start_bit + bit_written) % src_bit_len;
      index_t src_word_idx = bit_idx / bits_per_element;
      index_t src_offset   = bit_idx % bits_per_element;

      index_t remaining_bits = min(bits_per_element - bit_written, src_bit_len - bit_idx);

      bitset_t src_value = (src[src_word_idx] >> src_offset);

      if (src_offset + remaining_bits > bits_per_element) {
        bitset_t next_value = src[(src_word_idx + 1) % src_size];
        src_value |= (next_value << (bits_per_element - src_offset));
      }
      src_value &= ((bitset_t{1} << remaining_bits) - 1);
      result |= (src_value << bit_written);
      bit_written += remaining_bits;
    }
    output[output_idx] = result;
  }
}

template <typename bitset_t, typename index_t>
void bitset_repeat(raft::resources const& handle,
                   const bitset_t* d_src,
                   bitset_t* d_output,
                   index_t src_bit_len,
                   index_t repeat_times)
{
  if (src_bit_len == 0 || repeat_times == 0) return;
  auto stream = resource::get_cuda_stream(handle);

  constexpr index_t bits_per_element = sizeof(bitset_t) * 8;
  const index_t total_bits           = src_bit_len * repeat_times;
  const index_t output_size          = (total_bits + bits_per_element - 1) / bits_per_element;

  int threadsPerBlock = 128;
  int blocksPerGrid   = (output_size + threadsPerBlock - 1) / threadsPerBlock;
  bitset_repeat_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
    d_src, d_output, src_bit_len, repeat_times);

  return;
}

template <typename bitset_t, typename index_t>
void bitset_view<bitset_t, index_t>::repeat(const raft::resources& res,
                                            index_t times,
                                            bitset_t* output_device_ptr) const
{
  // Only a copy and kernel run below this point.
  if (resource::get_dry_run_flag(res)) { return; }
  constexpr index_t bits_per_element = sizeof(bitset_t) * 8;

  if (bitset_len_ % bits_per_element == 0) {
    index_t num_elements_to_copy = bitset_len_ / bits_per_element;

    for (index_t i = 0; i < times; ++i) {
      raft::copy(output_device_ptr + i * num_elements_to_copy,
                 bitset_ptr_,
                 num_elements_to_copy,
                 raft::resource::get_cuda_stream(res));
    }
  } else {
    bitset_repeat(res, bitset_ptr_, output_device_ptr, bitset_len_, times);
  }
}

template <typename bitset_t, typename index_t>
double bitset_view<bitset_t, index_t>::sparsity(const raft::resources& res) const
{
  index_t size_h = this->size();
  if (0 == size_h) { return static_cast<double>(1.0); }
  index_t count_h = this->count(res);

  return static_cast<double>((1.0 * (size_h - count_h)) / (1.0 * size_h));
}

template <typename bitset_t, typename index_t>
template <typename csr_matrix_t>
void bitset_view<bitset_t, index_t>::to_csr(const raft::resources& res, csr_matrix_t& csr) const
{
  raft::sparse::convert::bitset_to_csr(res, *this, csr);
}

template <typename bitset_t, typename index_t>
bitset<bitset_t, index_t>::bitset(const raft::resources& res,
                                  raft::device_vector_view<const index_t, index_t> mask_index,
                                  index_t bitset_len,
                                  bool default_value)
  : bitset_{std::size_t(raft::div_rounding_up_safe(bitset_len, bitset_element_size)),
            raft::resource::get_cuda_stream(res)},
    bitset_len_{bitset_len}
{
  reset(res, default_value);
  set(res, mask_index, !default_value);
}

template <typename bitset_t, typename index_t>
bitset<bitset_t, index_t>::bitset(const raft::resources& res,
                                  index_t bitset_len,
                                  bool default_value)
  : bitset_{std::size_t(raft::div_rounding_up_safe(bitset_len, bitset_element_size)),
            raft::resource::get_cuda_stream(res)},
    bitset_len_{bitset_len}
{
  reset(res, default_value);
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::resize(const raft::resources& res,
                                       index_t new_bitset_len,
                                       bool default_value)
{
  auto old_size = raft::div_rounding_up_safe(bitset_len_, bitset_element_size);
  auto new_size = raft::div_rounding_up_safe(new_bitset_len, bitset_element_size);
  bitset_.resize(new_size);
  bitset_len_ = new_bitset_len;
  if (old_size < new_size) {
    // If the new size is larger, set the new bits to the default value
    auto new_elements_view = raft::make_device_vector_view<bitset_t, index_t>(
      bitset_.data() + old_size, new_size - old_size);
    raft::linalg::map(res,
                      new_elements_view,
                      raft::const_op<bitset_t>{default_value ? static_cast<bitset_t>(-1)
                                                             : static_cast<bitset_t>(0)});
  }
}

template <typename bitset_t, typename index_t>
template <typename output_t>
void bitset<bitset_t, index_t>::test(const raft::resources& res,
                                     raft::device_vector_view<const index_t, index_t> queries,
                                     raft::device_vector_view<output_t, index_t> output) const
{
  RAFT_EXPECTS(output.extent(0) == queries.extent(0), "Output and queries must be same size");
  auto bitset_view = view();
  raft::linalg::map(
    res,
    output,
    [bitset_view] __device__(index_t query) { return bitset_view.test(query); },
    queries);
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::set(const raft::resources& res,
                                    raft::device_vector_view<const index_t, index_t> mask_index,
                                    bool set_value)
{
  auto this_bitset_view = view();
  // Use mask_index as output since we're just writing back the same values
  auto mask_index_output = raft::make_device_vector_view<index_t, index_t>(
    const_cast<index_t*>(mask_index.data_handle()), mask_index.extent(0));
  raft::linalg::map_offset(
    res,
    mask_index_output,
    [this_bitset_view, set_value] __device__(index_t idx, const index_t sample_index) -> index_t {
      this_bitset_view.set(sample_index, set_value);
      return sample_index;  // Return the input value as output (in-place)
    },
    mask_index);
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::flip(const raft::resources& res)
{
  auto bitset_span = this->to_mdspan();
  raft::linalg::map(
    res,
    bitset_span,
    [] __device__(bitset_t element) { return bitset_t(~element); },
    raft::make_const_mdspan(bitset_span));
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::reset(const raft::resources& res, bool default_value)
{
  auto bitset_view = raft::make_device_vector_view<bitset_t, index_t>(bitset_.data(), n_elements());
  raft::linalg::map(
    res,
    bitset_view,
    raft::const_op<bitset_t>{default_value ? static_cast<bitset_t>(-1) : static_cast<bitset_t>(0)});
}

template <typename bitset_t, typename index_t>
void bitset<bitset_t, index_t>::count(const raft::resources& res,
                                      raft::device_scalar_view<index_t> count_gpu_scalar)
{
  auto max_len = raft::make_host_scalar_view<const index_t, index_t>(&bitset_len_);
  auto values =
    raft::make_device_vector_view<const bitset_t, index_t>(bitset_.data(), n_elements());
  raft::popc(res, values, max_len, count_gpu_scalar);
}

}  // namespace core
}  // namespace RAFT_EXPORT raft
