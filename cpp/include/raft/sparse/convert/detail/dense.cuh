/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusparse_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/sparse/linalg/detail/cusparse_utils.hpp>

#include <cuda_runtime.h>

#include <cusparse_v2.h>

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

template <typename SparseMatrixViewType,
          typename ValueType,
          typename IndexType,
          typename LayoutPolicy>
void sparse_to_dense(raft::resources const& handle,
                     SparseMatrixViewType sparse,
                     raft::device_matrix_view<ValueType, IndexType, LayoutPolicy> dense)
{
  auto stream            = raft::resource::get_cuda_stream(handle);
  auto cusparse_handle   = raft::resource::get_cusparse_handle(handle);
  auto sparse_descriptor = raft::sparse::linalg::detail::create_descriptor(sparse);
  auto dense_descriptor  = raft::sparse::linalg::detail::create_descriptor(dense);

  RAFT_CUSPARSE_TRY(cusparseSetStream(cusparse_handle, stream));
  std::size_t buffer_size;
  RAFT_CUSPARSE_TRY(cusparseSparseToDense_bufferSize(cusparse_handle,
                                                     sparse_descriptor,
                                                     dense_descriptor,
                                                     CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                                     &buffer_size));
  auto buffer = raft::make_device_vector<char, std::size_t>(handle, buffer_size);
  RAFT_CUSPARSE_TRY(cusparseSparseToDense(cusparse_handle,
                                          sparse_descriptor,
                                          dense_descriptor,
                                          CUSPARSE_SPARSETODENSE_ALG_DEFAULT,
                                          buffer.data_handle()));

  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroySpMat(sparse_descriptor));
  RAFT_CUSPARSE_TRY_NO_THROW(cusparseDestroyDnMat(dense_descriptor));
}

};  // namespace detail
};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // namespace raft
