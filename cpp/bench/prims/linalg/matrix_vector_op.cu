/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <common/benchmark.hpp>

#include <raft/linalg/matrix_vector_op.cuh>

#include <rmm/device_uvector.hpp>

namespace raft::bench::linalg {

template <typename IdxT>
struct mat_vec_op_inputs {
  IdxT rows, cols;
  bool rowMajor, bcastAlongRows;
  IdxT inAlignOffset, outAlignOffset;
};  // struct mat_vec_op_inputs

template <typename IdxT>
inline auto operator<<(std::ostream& os, const mat_vec_op_inputs<IdxT>& p) -> std::ostream&
{
  os << p.rows << "#" << p.cols << "#" << p.rowMajor << "#" << p.bcastAlongRows << "#"
     << p.inAlignOffset << "#" << p.outAlignOffset;
  return os;
}

template <typename OpT, typename T, typename IdxT>
struct mat_vec_op : public fixture {
  mat_vec_op(const mat_vec_op_inputs<IdxT>& p)
    : params(p),
      out(p.rows * p.cols + params.outAlignOffset, stream),
      in(p.rows * p.cols + params.inAlignOffset, stream),
      vec1(p.bcastAlongRows ? p.cols : p.rows, stream),
      vec2(p.bcastAlongRows ? p.cols : p.rows, stream)
  {
  }

  void run_benchmark(::benchmark::State& state) override
  {
    std::ostringstream label_stream;
    label_stream << params;
    state.SetLabel(label_stream.str());

    loop_on_state(state, [this]() {
      if constexpr (OpT::useTwoVectors) {
        if (params.rowMajor) {
          if (params.bcastAlongRows) {
            raft::linalg::matrixVectorOp<true, true>(out.data() + params.outAlignOffset,
                                                     in.data() + params.inAlignOffset,
                                                     vec1.data(),
                                                     vec2.data(),
                                                     params.cols,
                                                     params.rows,
                                                     OpT{},
                                                     stream.get());
          } else {
            raft::linalg::matrixVectorOp<true, false>(out.data() + params.outAlignOffset,
                                                      in.data() + params.inAlignOffset,
                                                      vec1.data(),
                                                      vec2.data(),
                                                      params.cols,
                                                      params.rows,
                                                      OpT{},
                                                      stream.get());
          }
        } else {
          if (params.bcastAlongRows) {
            raft::linalg::matrixVectorOp<false, true>(out.data() + params.outAlignOffset,
                                                      in.data() + params.inAlignOffset,
                                                      vec1.data(),
                                                      vec2.data(),
                                                      params.cols,
                                                      params.rows,
                                                      OpT{},
                                                      stream.get());
          } else {
            raft::linalg::matrixVectorOp<false, false>(out.data() + params.outAlignOffset,
                                                       in.data() + params.inAlignOffset,
                                                       vec1.data(),
                                                       vec2.data(),
                                                       params.cols,
                                                       params.rows,
                                                       OpT{},
                                                       stream.get());
          }
        }
      } else {
        if (params.rowMajor) {
          if (params.bcastAlongRows) {
            raft::linalg::matrixVectorOp<true, true>(out.data() + params.outAlignOffset,
                                                     in.data() + params.inAlignOffset,
                                                     vec1.data(),
                                                     params.cols,
                                                     params.rows,
                                                     OpT{},
                                                     stream.get());
          } else {
            raft::linalg::matrixVectorOp<true, false>(out.data() + params.outAlignOffset,
                                                      in.data() + params.inAlignOffset,
                                                      vec1.data(),
                                                      params.cols,
                                                      params.rows,
                                                      OpT{},
                                                      stream.get());
          }
        } else {
          if (params.bcastAlongRows) {
            raft::linalg::matrixVectorOp<false, true>(out.data() + params.outAlignOffset,
                                                      in.data() + params.inAlignOffset,
                                                      vec1.data(),
                                                      params.cols,
                                                      params.rows,
                                                      OpT{},
                                                      stream.get());
          } else {
            raft::linalg::matrixVectorOp<false, false>(out.data() + params.outAlignOffset,
                                                       in.data() + params.inAlignOffset,
                                                       vec1.data(),
                                                       params.cols,
                                                       params.rows,
                                                       OpT{},
                                                       stream.get());
          }
        }
      }
    });
  }

 private:
  mat_vec_op_inputs<IdxT> params;
  rmm::device_uvector<T> out, in, vec1, vec2;
};  // struct MatVecOp

template <typename IdxT>
std::vector<mat_vec_op_inputs<IdxT>> get_mv_inputs()
{
  std::vector<mat_vec_op_inputs<IdxT>> out;

  // Scalability benchmark with round dimensions
  std::vector<IdxT> rows = {1000, 100000, 1000000};
  std::vector<IdxT> cols = {8, 64, 256, 1024};
  for (bool rowMajor : {true, false}) {
    for (bool alongRows : {true, false}) {
      for (IdxT rows_ : rows) {
        for (IdxT cols_ : cols) {
          out.push_back({rows_, cols_, rowMajor, alongRows, 0, 0});
        }
      }
    }
  }

  // Odd dimensions, misalignment
  std::vector<std::tuple<IdxT, IdxT>> rowcols = {
    {44739207, 7},
    {44739207, 15},
    {44739207, 16},
    {44739207, 17},
    {2611236, 256},
    {2611236, 257},
    {2611236, 263},
  };
  for (bool rowMajor : {true, false}) {
    for (bool alongRows : {true, false}) {
      for (auto rc : rowcols) {
        for (IdxT inAlignOffset : {0, 1}) {
          for (IdxT outAlignOffset : {0, 1}) {
            out.push_back({std::get<0>(rc),
                           std::get<1>(rc),
                           rowMajor,
                           alongRows,
                           inAlignOffset,
                           outAlignOffset});
          }
        }
      }
    }
  }
  return out;
}

const std::vector<mat_vec_op_inputs<int>> mv_input_i32     = get_mv_inputs<int>();
const std::vector<mat_vec_op_inputs<int64_t>> mv_input_i64 = get_mv_inputs<int64_t>();

template <typename T>
struct Add1Vec {
  static constexpr bool useTwoVectors = false;
  HDI T operator()(T a, T b) const { return a + b; };
};
template <typename T>
struct Add2Vec {
  static constexpr bool useTwoVectors = true;
  HDI T operator()(T a, T b, T c) const { return a + b + c; };
};

RAFT_BENCH_REGISTER((mat_vec_op<Add1Vec<float>, float, int>), "", mv_input_i32);
RAFT_BENCH_REGISTER((mat_vec_op<Add1Vec<double>, double, int>), "", mv_input_i32);
RAFT_BENCH_REGISTER((mat_vec_op<Add2Vec<float>, float, int>), "", mv_input_i32);
RAFT_BENCH_REGISTER((mat_vec_op<Add2Vec<double>, double, int>), "", mv_input_i32);
RAFT_BENCH_REGISTER((mat_vec_op<Add1Vec<float>, float, int64_t>), "", mv_input_i64);
RAFT_BENCH_REGISTER((mat_vec_op<Add1Vec<double>, double, int64_t>), "", mv_input_i64);
RAFT_BENCH_REGISTER((mat_vec_op<Add2Vec<float>, float, int64_t>), "", mv_input_i64);
RAFT_BENCH_REGISTER((mat_vec_op<Add2Vec<double>, double, int64_t>), "", mv_input_i64);

}  // namespace raft::bench::linalg
