/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/linalg/reduce.cuh>
#include <raft/matrix/init.cuh>
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/convert/dense.cuh>
#include <raft/sparse/matrix/preprocessing.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/op/sort.cuh>

namespace raft::util {

template <typename T1, typename T2>
struct check_zeroes {
  float __device__ operator()(const T1& value, const T2& idx)
  {
    if (value == 0) {
      return 0.f;
    } else {
      return 1.f;
    }
  }
};

template <typename T1, typename T2, typename DenseVectorView, typename ResultVectorView>
void preproc(raft::resources& handle,
             DenseVectorView dense_values,
             ResultVectorView results,
             int num_rows,
             int num_cols,
             bool tf_idf)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  // create matrix and copy to device
  auto host_dense_vals = raft::make_host_vector<T2, int64_t>(handle, dense_values.size());
  raft::copy(
    host_dense_vals.data_handle(), dense_values.data_handle(), dense_values.size(), stream);

  auto host_matrix =
    raft::make_host_matrix_view<T2, int64_t>(host_dense_vals.data_handle(), num_rows, num_cols);
  auto device_matrix = raft::make_device_matrix<T2, int64_t>(handle, num_rows, num_cols);

  raft::copy(device_matrix.data_handle(), host_matrix.data_handle(), host_matrix.size(), stream);

  // get sum reduce for each row (length of the document)
  auto output_rows_lengths = raft::make_device_matrix<T2, int64_t>(handle, 1, num_rows);
  raft::linalg::reduce<true, true>(output_rows_lengths.data_handle(),
                                   device_matrix.data_handle(),
                                   num_cols,
                                   num_rows,
                                   0.0f,
                                   stream);
  auto h_output_rows_lengths = raft::make_host_matrix<T2, int64_t>(handle, 1, num_rows);
  raft::copy(h_output_rows_lengths.data_handle(),
             output_rows_lengths.data_handle(),
             output_rows_lengths.size(),
             stream);

  // find the avg size of a document
  auto output_rows_length_sum = raft::make_device_scalar<T2>(handle, 0);
  raft::linalg::mapReduce(output_rows_length_sum.data_handle(),
                          num_rows,
                          0.0f,
                          raft::identity_op(),
                          raft::add_op(),
                          stream,
                          output_rows_lengths.data_handle());
  auto h_output_rows_length_sum = raft::make_host_scalar<T2>(handle, 0);
  raft::copy(h_output_rows_length_sum.data_handle(),
             output_rows_length_sum.data_handle(),
             output_rows_length_sum.size(),
             stream);
  T2 avg_row_length = (T2)h_output_rows_length_sum(0) / num_rows;

  // find the number of docs(row) each vocab(col) word is in
  auto output_cols_cnt = raft::make_device_matrix<T2, int64_t>(handle, 1, num_cols);
  raft::linalg::reduce<true, false>(output_cols_cnt.data_handle(),
                                    device_matrix.data_handle(),
                                    num_cols,
                                    num_rows,
                                    0.0f,
                                    stream,
                                    false,
                                    check_zeroes<T2, T2>());
  auto h_output_cols_cnt = raft::make_host_matrix<T2, int64_t>(handle, 1, num_cols);
  raft::copy(
    h_output_cols_cnt.data_handle(), output_cols_cnt.data_handle(), output_cols_cnt.size(), stream);

  // perform bm25/tfidf calculations
  auto out_device_matrix = raft::make_device_matrix<T2, int64_t>(handle, num_rows, num_cols);
  raft::matrix::fill<T2>(handle, out_device_matrix.view(), 0.0f);
  auto out_host_matrix = raft::make_host_matrix<T2, int64_t>(handle, num_rows, num_cols);
  auto out_host_vector = raft::make_host_vector<T2, int64_t>(handle, results.size());

  float k1  = 1.6f;
  float b   = 0.75f;
  int count = 0;
  float result;
  for (int row = 0; row < num_rows; row++) {
    for (int col = 0; col < num_cols; col++) {
      float val = host_matrix(row, col);
      if (val == 0) {
        out_host_matrix(row, col) = 0.0f;
      } else {
        float tf     = (float)raft::log<float>(val);
        float idf_in = (float)num_rows / h_output_cols_cnt(0, col);
        float idf    = (float)raft::log<float>(idf_in + 1);
        if (tf_idf) {
          result = (float)tf * idf;
        } else {
          float bm25 = ((k1 + 1) * tf) /
                       (k1 * ((1 - b) + b * (h_output_rows_lengths(0, row) / avg_row_length)) + tf);
          result = (float)idf * bm25;
        }
        out_host_matrix(row, col) = result;
        out_host_vector(count)    = result;
        count++;
      }
    }
  }
  raft::copy(results.data_handle(), out_host_vector.data_handle(), out_host_vector.size(), stream);
}

template <typename T1, typename T2, typename ResultVectorView>
void calc_tfidf_bm25(raft::resources& handle,
                     raft::device_csr_matrix_view<T2, T1, T1, T1> csr_in,
                     ResultVectorView results,
                     bool tf_idf = false)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int num_rows        = csr_in.structure_view().get_n_rows();
  int num_cols        = csr_in.structure_view().get_n_cols();

  auto dense_values = raft::make_device_vector<T2, int64_t>(handle, num_rows * num_cols);

  raft::sparse::convert::sparse_to_dense(
    handle,
    csr_in,
    raft::make_device_matrix_view<T2, int64_t, raft::row_major>(
      dense_values.data_handle(), num_rows, num_cols));

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  preproc<T1, T2>(handle, dense_values.view(), results, num_rows, num_cols, tf_idf);
}

template <typename T1, typename T2>
void create_dataset(raft::resources& handle,
                    raft::device_vector_view<T1, int64_t> rows,
                    raft::device_vector_view<T1, int64_t> columns,
                    raft::device_vector_view<T2, int64_t> values,
                    int max_term_occurence_doc = 5,
                    int num_rows_unique        = 7,
                    int num_cols_unique        = 7,
                    int seed                   = 12345)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::random::RngState rng(seed);

  auto d_out = raft::make_device_vector<T1, int64_t>(handle, rows.size() * 2);

  int theta_guide = max(num_rows_unique, num_cols_unique);
  auto theta      = raft::make_device_vector<T2, int64_t>(handle, theta_guide * 4);

  raft::random::uniform(handle, rng, theta.view(), 0.0f, 1.0f);

  raft::random::rmat_rectangular_gen(d_out.data_handle(),
                                     rows.data_handle(),
                                     columns.data_handle(),
                                     theta.data_handle(),
                                     num_rows_unique,
                                     num_cols_unique,
                                     int(values.size()),
                                     stream,
                                     rng);

  auto vals = raft::make_device_vector<T1, int64_t>(handle, rows.size());
  raft::random::uniformInt(handle, rng, vals.view(), 1, max_term_occurence_doc);
  raft::linalg::map(handle, values, raft::cast_op<T2>{}, raft::make_const_mdspan(vals.view()));
}

};  // namespace raft::util
