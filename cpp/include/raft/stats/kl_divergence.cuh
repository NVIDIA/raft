/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __KL_DIVERGENCE_H
#define __KL_DIVERGENCE_H

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/stats/detail/kl_divergence.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate KL Divergence
 * <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">more info on KL
 * Divergence</a>
 *
 * @tparam DataT: Data type of the input array
 * @param modelPDF: the model array of probability density functions of type DataT
 * @param candidatePDF: the candidate array of probability density functions of type DataT
 * @param size: the size of the data points of type int
 * @param stream: the cudaStream object
 */
template <typename DataT>
DataT kl_divergence(const DataT* modelPDF, const DataT* candidatePDF, int size, cudaStream_t stream)
{
  return detail::kl_divergence(false, modelPDF, candidatePDF, size, stream);
}

/**
 * @defgroup kl_divergence Kullback-Leibler Divergence
 * @{
 */

/**
 * @brief Function to calculate KL Divergence
 * <a href="https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence">more info on KL
 * Divergence</a>
 *
 * @tparam value_t: Data type of the input array
 * @tparam idx_t index type
 * @param[in] handle the raft handle
 * @param[in] modelPDF: the model array of probability density functions of type value_t
 * @param[in] candidatePDF: the candidate array of probability density functions of type value_t
 * @return the KL Divergence value
 */
template <typename value_t, typename idx_t>
value_t kl_divergence(raft::resources const& handle,
                      raft::device_vector_view<const value_t, idx_t> modelPDF,
                      raft::device_vector_view<const value_t, idx_t> candidatePDF)
{
  RAFT_EXPECTS(modelPDF.size() == candidatePDF.size(), "Size mismatch");
  RAFT_EXPECTS(modelPDF.is_exhaustive(), "modelPDF must be contiguous");
  RAFT_EXPECTS(candidatePDF.is_exhaustive(), "candidatePDF must be contiguous");
  return detail::kl_divergence(resource::get_dry_run_flag(handle),
                               modelPDF.data_handle(),
                               candidatePDF.data_handle(),
                               modelPDF.extent(0),
                               resource::get_cuda_stream(handle));
}

/** @} */  // end group kl_divergence

};  // end namespace stats
};  // namespace raft
#endif
