/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/detail/macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/kernel_launch.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <cuda/std/tuple>

#include <algorithm>

namespace raft {
namespace matrix {
namespace detail {

/** This type simplifies returning arrays and passing them as arguments */
template <typename Type, int VecElems>
struct VecArg {
  Type val[VecElems];
};

/** Executes the operation with the given matrix element and an arbitrary number of vector elements
 * contained in the given tuple. The index_sequence is used here for compile-time indexing of the
 * tuple in the fold expression. */
template <typename MatT, typename Lambda, class Tuple, size_t... Is>
__device__ __forceinline__ MatT
RunMatVecOp(Lambda op, MatT mat, Tuple&& args, std::index_sequence<Is...>)
{
  return op(mat, (cuda::std::get<Is>(args))...);
}

template <typename Type, typename IdxType, std::size_t VecBytes, int BlockSize>
struct Linewise {
  static constexpr IdxType VecElems = VecBytes / sizeof(Type);

  typedef raft::TxN_t<Type, VecElems> Vec;
  typedef raft::Pow2<VecBytes> AlignBytes;
  typedef raft::Pow2<VecElems> AlignElems;
  typedef raft::Pow2<raft::WarpSize> AlignWarp;

  /**
   * Compute op(matrix_in, vec_1, vec_2, ...) where vectors are applied across the
   * matrix rows (one vector element per matrix row).
   *
   * It's assumed that `in` and `out` are aligned to the cuda-vector-size,
   * and their length is multiple of that.
   *
   * Block work arrangement: blocked;
   *     one warp works on a contiguous chunk of a matrix. Since the matrix is represented
   *     as a flat array, such an arrangement minimizes the number of times when a single
   *     thread needs to reload the vector value at an index corresponding to the current
   *     matrix row. Ideally, a thread would load a value from a vector only once, but that
   *     is not possible if the vector size (= number of matrix rows) is too small or not
   *     aligned with the cuda-vector-size.
   *
   * Note about rowDiv/rowMod:
   *     these two represent the row/column indices in the original input matrices, before
   *     it was converted to (Vec::io_t*) type (which possibly involves shifting a pointer
   *     a bit to align to the cuda-vector-size). Thus, they are used to track the index for
   *     the argument vectors only (the vector pointers are not altered in any way).
   *
   *
   * @tparam Vecs a pack of pointers to vectors (Type*)
   * @param [out] out (aligned part of) the output matrix
   * @param [in] in (aligned part of) the input matrix
   * @param [in] in_end end of the (aligned part of the) input matrix
   * @param [in] rowLen number of elements in a row (NOT the vector size)
   * @param [in] rowDiv the index in the vectors (= row num in the original unaligned input matrix)
   * @param [in] rowMod the index within a row in the original unaligned input matrix.
   * @param [in] op the function to apply
   * @param [in] vecs pointers to the argument vectors.
   *
   */
  template <typename Lambda, typename... Vecs>
  static __device__ __forceinline__ void vectorCols(typename Vec::io_t* out,
                                                    const typename Vec::io_t* in,
                                                    const typename Vec::io_t* in_end,
                                                    const IdxType rowLen,
                                                    IdxType rowDiv,
                                                    IdxType rowMod,
                                                    Lambda op,
                                                    const Vecs*... vecs) noexcept
  {
    constexpr IdxType warpPad = (AlignWarp::Value - 1) * VecElems;
    constexpr auto index      = std::index_sequence_for<Vecs...>();
    cuda::std::tuple<Vecs...> args;
    Vec v, w;
    bool update = true;
    for (; in < in_end; in += AlignWarp::Value, out += AlignWarp::Value, rowMod += warpPad) {
      *v.vectorized_data() = __ldcv(in);
      while (rowMod >= rowLen) {
        rowMod -= rowLen;
        rowDiv++;
        update = true;
      }
      if (update) {
        args   = cuda::std::make_tuple((vecs[rowDiv])...);
        update = false;
      }
#pragma unroll VecElems
      for (int k = 0; k < VecElems; k++, rowMod++) {
        if (rowMod == rowLen) {
          rowMod = 0;
          rowDiv++;
          args = cuda::std::make_tuple((vecs[rowDiv])...);
        }
        w.val.data[k] = RunMatVecOp(op, v.val.data[k], args, index);
      }
      *out = *w.vectorized_data();
    }
  }

  /**
   * Compute op(matrix_in, vec_1, vec_2, ...) where vectors are applied along
   * matrix rows (vector and matrix indices are 1-1).
   *
   * It's assumed that `in` and `out` are aligned to the cuda-vector-size,
   * and their length is multiple of that.
   *
   * Block work arrangement: striped;
   *     the grid size is chosen in such a way, that one thread always processes
   *     the same vector elements. That's why there is no need to read the
   *     vector arguments multiple times.
   *
   * @tparam Args a pack of raft::TxN_t<Type, VecElems>
   * @param [out] out (aligned part of) the output matrix
   * @param [in] in (aligned part of) the input matrix
   * @param [in] len total length of (the aligned part of) the input/output matrices
   * @param [in] op the function to apply
   * @param [in] args the cuda-vector-sized chunks on input vectors (raft::TxN_t<Type, VecElems>)
   */
  template <typename Lambda, typename... Args>
  static __device__ __forceinline__ void vectorRows(typename Vec::io_t* out,
                                                    const typename Vec::io_t* in,
                                                    const IdxType len,
                                                    Lambda op,
                                                    Args... args) noexcept
  {
    Vec v;
    const IdxType d = BlockSize * gridDim.x;
    for (IdxType i = threadIdx.x + blockIdx.x * BlockSize; i < len; i += d) {
      *v.vectorized_data() = __ldcv(in + i);
#pragma unroll VecElems
      for (int k = 0; k < VecElems; k++)
        v.val.data[k] = op(v.val.data[k], args.val[k]...);
      __stwt(out + i, *v.vectorized_data());
    }
  }

  /**
   * The helper for `vectorRows`. Loads the `raft::TxN_t<Type, VecElems>` chunk
   * of a vector. Most of the time this is not aligned, so we load it thread-striped
   * within a block and then use the shared memory to get a contiguous chunk.
   *
   * @tparam VecT Type of the vector to load
   * @param [in] shm a shared memory region for rearranging the data among threads
   * @param [in] p pointer to a vector
   * @param [in] blockOffset the offset of the current block into a vector.
   * @param [in] rowLen the length of a vector.
   * @return a contiguous chunk of a vector, suitable for `vectorRows`.
   */
  template <typename VecT>
  static __device__ __forceinline__ VecArg<VecT, VecElems> loadVec(VecT* shm,
                                                                   const VecT* p,
                                                                   const IdxType blockOffset,
                                                                   const IdxType rowLen) noexcept
  {
    IdxType j = blockOffset + threadIdx.x;
#pragma unroll VecElems
    for (int k = threadIdx.x; k < VecElems * BlockSize; k += BlockSize, j += BlockSize) {
      while (j >= rowLen)
        j -= rowLen;
      shm[k] = p[j];
    }
    __syncthreads();
    {
      VecArg<VecT, VecElems> out;
#pragma unroll VecElems
      for (int i = 0; i < VecElems; i++)
        out.val[i] = shm[threadIdx.x * VecElems + i];
      return out;
    }
  }

  /**
   * @brief Same as loadVec, but pads data with Ones
   *
   * @tparam VecT Type of the vector to load
   * @param shm
   * @param p
   * @param blockOffset
   * @param rowLen
   * @param rowLenPadded
   * @return a contiguous chunk of a vector, suitable for `vectorRows`.
   */
  template <typename VecT>
  static __device__ __forceinline__ VecArg<VecT, VecElems> loadVecPadded(
    VecT* shm,
    const VecT* p,
    const IdxType blockOffset,
    const IdxType rowLen,
    const IdxType rowLenPadded) noexcept
  {
    IdxType j = blockOffset + threadIdx.x;
#pragma unroll VecElems
    for (int k = threadIdx.x; k < VecElems * BlockSize; k += BlockSize, j += BlockSize) {
      while (j >= rowLenPadded)
        j -= rowLenPadded;
      shm[k] = j < rowLen ? p[j] : VecT(1);
    }
    __syncthreads();
    {
      VecArg<VecT, VecElems> out;
#pragma unroll VecElems
      for (int i = 0; i < VecElems; i++)
        out.val[i] = shm[threadIdx.x * VecElems + i];
      return out;
    }
  }
};

/**
 * This kernel prepares the inputs for the `vectorCols` function where the most of the
 * work happens; see `vectorCols` for details.
 *
 * The work arrangement is blocked; a single block works on a contiguous chunk of flattened
 * matrix data and does not care about the gridDim.
 *
 * @param [out] out the output matrix
 * @param [in] in the input matrix
 * @param [in] arrOffset such an offset into the matrices that makes them aligned to the
 * cuda-vector-size
 * @param [in] rowLen number of elements in a row (NOT the vector size)
 * @param [in] len the total length of the aligned part of the matrices
 * @param [in] elemsPerThread how many elements are processed by a single thread in total
 * @param [in] op the function to apply
 * @param [in] vecs pointers to the argument vectors
 */
template <typename Type,
          typename IdxType,
          std::size_t VecBytes,
          int BlockSize,
          typename Lambda,
          typename... Vecs>
RAFT_KERNEL __launch_bounds__(BlockSize)
  matrixLinewiseVecColsMainKernel(Type* out,
                                  const Type* in,
                                  const IdxType arrOffset,
                                  const IdxType rowLen,
                                  const IdxType len,
                                  const IdxType elemsPerThread,
                                  Lambda op,
                                  const Vecs*... vecs)
{
  typedef Linewise<Type, IdxType, VecBytes, BlockSize> L;

  IdxType t = L::AlignWarp::mod(threadIdx.x);
  t = arrOffset + elemsPerThread * (blockIdx.x * BlockSize + threadIdx.x - t) + t * L::VecElems;

  return L::vectorCols(reinterpret_cast<typename L::Vec::io_t*>(out + t),
                       reinterpret_cast<const typename L::Vec::io_t*>(in + t),
                       reinterpret_cast<const typename L::Vec::io_t*>(
                         in + min(t + elemsPerThread * L::AlignWarp::Value, len)),
                       rowLen,
                       t / rowLen,
                       t % rowLen,
                       op,
                       vecs...);
}

/**
 * This kernel is similar to `matrixLinewiseVecColsMainKernel`, but processes only the unaligned
 * head and tail parts of the matrix.
 * This kernel is always launched in just two blocks; the first block processes the head of the
 * matrix, the second block processes the tail. It uses the same `vectorCols` function, but
 * sets `VecElems = 1`
 *
 * @param [out] out the output matrix
 * @param [in] in the input matrix
 * @param [in] arrOffset the length of the unaligned head - such an offset into the matrices that
 * makes them aligned to the `VecBytes`
 * @param [in] arrTail the offset to the unaligned tail
 * @param [in] rowLen number of elements in a row (NOT the vector size)
 * @param [in] len the total length of the matrices (rowLen * nRows)
 * @param [in] op the function to apply
 * @param [in] vecs pointers to the argument vectors
 */
template <typename Type, typename IdxType, std::size_t MaxOffset, typename Lambda, typename... Vecs>
RAFT_KERNEL __launch_bounds__(MaxOffset, 2) matrixLinewiseVecColsTailKernel(Type* out,
                                                                            const Type* in,
                                                                            const IdxType arrOffset,
                                                                            const IdxType arrTail,
                                                                            const IdxType rowLen,
                                                                            const IdxType len,
                                                                            Lambda op,
                                                                            const Vecs*... vecs)
{
  // Note, L::VecElems == 1
  typedef Linewise<Type, IdxType, sizeof(Type), MaxOffset> L;
  IdxType threadOffset, elemsPerWarp;
  if (blockIdx.x == 0) {
    // first block: offset = 0, length = arrOffset
    threadOffset = threadIdx.x;
    elemsPerWarp = threadOffset < arrOffset;
  } else {
    // second block: offset = arrTail, length = len - arrTail
    threadOffset = arrTail + threadIdx.x;
    elemsPerWarp = threadOffset < len;
  }
  const IdxType rowDiv = threadOffset / rowLen;
  const IdxType rowMod = threadOffset % rowLen;
  return L::vectorCols(
    reinterpret_cast<typename L::Vec::io_t*>(out + threadOffset),
    reinterpret_cast<const typename L::Vec::io_t*>(in + threadOffset),
    reinterpret_cast<const typename L::Vec::io_t*>(in + threadOffset + elemsPerWarp),
    rowLen,
    rowDiv,
    rowMod,
    op,
    vecs...);
}

/** Helper function to get the largest type from a variadic list of types */
template <typename... Types>
constexpr size_t maxSizeOf()
{
  size_t maxSize = 0;
  ((maxSize = std::max(maxSize, sizeof(Types))), ...);
  return maxSize;
}

/**
 * This kernel prepares the inputs for the `vectorRows` function where the most of the
 * work happens; see `vectorRows` for details.
 *
 * The work arrangement is striped; the gridDim should be selected in such a way, that
 * on each iteration a thread processes the same indices along rows:
 *   `(gridDim.x * BlockSize * VecElems) % rowLen == 0`.
 *
 * @param [out] out the start of the *aligned* part of the output matrix
 * @param [in] in the start of the *aligned* part of the input matrix
 * @param [in] arrOffset such an offset into the matrices that makes them aligned to `VecBytes`
 * @param [in] rowLen number of elements in a row (= the vector size)
 * @param [in] len the total length of the aligned part of the matrices
 * @param [in] op the function to apply
 * @param [in] vecs pointers to the argument vectors
 */
template <typename Type,
          typename IdxType,
          std::size_t VecBytes,
          int BlockSize,
          typename Lambda,
          typename... Vecs>
RAFT_KERNEL __launch_bounds__(BlockSize) matrixLinewiseVecRowsMainKernel(Type* out,
                                                                         const Type* in,
                                                                         const IdxType arrOffset,
                                                                         const IdxType rowLen,
                                                                         const IdxType len,
                                                                         Lambda op,
                                                                         const Vecs*... vecs)
{
  typedef Linewise<Type, IdxType, VecBytes, BlockSize> L;
  constexpr uint workSize         = L::VecElems * BlockSize;
  constexpr size_t maxVecItemSize = maxSizeOf<Vecs...>();
  uint workOffset                 = workSize * maxVecItemSize;
  __shared__ __align__(
    maxVecItemSize *
    L::VecElems) char shm[workSize * maxVecItemSize * ((sizeof...(Vecs)) > 1 ? 2 : 1)];
  const IdxType blockOffset = (arrOffset + BlockSize * L::VecElems * blockIdx.x) % rowLen;
  return L::vectorRows(reinterpret_cast<typename L::Vec::io_t*>(out),
                       reinterpret_cast<const typename L::Vec::io_t*>(in),
                       L::AlignElems::div(len),
                       op,
                       (workOffset ^= workSize * maxVecItemSize,
                        L::loadVec((Vecs*)(shm + workOffset), vecs, blockOffset, rowLen))...);
}

/**
 * Simplified version of `matrixLinewiseVecRowsMainKernel` for use with padded data.
 * Data is required to be aligned and padded.
 *
 * @param [out] out the start of the *aligned* part of the output matrix
 * @param [in] in the start of the *aligned* part of the input matrix
 * @param [in] arrOffset such an offset into the matrices that makes them aligned to `VecBytes`
 * @param [in] rowLen number of elements in a row (= the vector size)
 * @param [in] len the total length of the aligned part of the matrices
 * @param [in] op the function to apply
 * @param [in] vecs pointers to the argument vectors
 */
template <typename Type,
          typename IdxType,
          std::size_t VecBytes,
          int BlockSize,
          typename Lambda,
          typename... Vecs>
RAFT_KERNEL __launch_bounds__(BlockSize) matrixLinewiseVecRowsSpanKernel(Type* out,
                                                                         const Type* in,
                                                                         const IdxType rowLen,
                                                                         const IdxType rowLenPadded,
                                                                         const IdxType lenPadded,
                                                                         Lambda op,
                                                                         const Vecs*... vecs)
{
  typedef Linewise<Type, IdxType, VecBytes, BlockSize> L;
  constexpr uint workSize         = L::VecElems * BlockSize;
  constexpr size_t maxVecItemSize = maxSizeOf<Vecs...>();
  uint workOffset                 = workSize * maxVecItemSize;
  __shared__ __align__(
    maxVecItemSize *
    L::VecElems) char shm[workSize * maxVecItemSize * ((sizeof...(Vecs)) > 1 ? 2 : 1)];
  const IdxType blockOffset = (BlockSize * L::VecElems * blockIdx.x) % rowLenPadded;
  return L::vectorRows(
    reinterpret_cast<typename L::Vec::io_t*>(out),
    reinterpret_cast<const typename L::Vec::io_t*>(in),
    L::AlignElems::div(lenPadded),
    op,
    (workOffset ^= workSize * maxVecItemSize,
     L::loadVecPadded((Vecs*)(shm + workOffset), vecs, blockOffset, rowLen, rowLenPadded))...);
}

/**
 * This kernel is similar to `matrixLinewiseVecRowsMainKernel`, but processes only the unaligned
 * head and tail parts of the matrix.
 * This kernel is always launched in just two blocks; the first block processes the head of the
 * matrix, the second block processes the tail. It uses the same `vectorRows` function, but
 * sets `VecElems = 1`
 *
 * @param [out] out the output matrix
 * @param [in] in the input matrix
 * @param [in] arrOffset the length of the unaligned head - such an offset into the matrices that
 * makes them aligned to the `VecBytes`
 * @param [in] arrTail the offset to the unaligned tail
 * @param [in] rowLen number of elements in a row (= the vector size)
 * @param [in] len the total length of the matrices (rowLen * nRows)
 * @param [in] op the function to apply
 * @param [in] vecs pointers to the argument vectors
 */
template <typename Type, typename IdxType, std::size_t MaxOffset, typename Lambda, typename... Vecs>
RAFT_KERNEL __launch_bounds__(MaxOffset, 2) matrixLinewiseVecRowsTailKernel(Type* out,
                                                                            const Type* in,
                                                                            const IdxType arrOffset,
                                                                            const IdxType arrTail,
                                                                            const IdxType rowLen,
                                                                            const IdxType len,
                                                                            Lambda op,
                                                                            const Vecs*... vecs)
{
  // Note, L::VecElems == 1
  constexpr uint workSize         = MaxOffset;
  constexpr size_t maxVecItemSize = maxSizeOf<Vecs...>();
  uint workOffset                 = workSize * maxVecItemSize;
  __shared__ char shm[workSize * maxVecItemSize * ((sizeof...(Vecs)) > 1 ? 2 : 1)];
  typedef Linewise<Type, IdxType, sizeof(Type), MaxOffset> L;
  if (blockIdx.x == 0) {
    // first block: offset = 0, length = arrOffset
    L::vectorRows(reinterpret_cast<typename L::Vec::io_t*>(out),
                  reinterpret_cast<const typename L::Vec::io_t*>(in),
                  arrOffset,
                  op,
                  (workOffset ^= workSize * maxVecItemSize,
                   L::loadVec((Vecs*)(shm + workOffset), vecs, 0, rowLen))...);
  } else {
    // second block: offset = arrTail, length = len - arrTail
    // NB: I subtract MaxOffset (= blockDim.x) to get the correct indexing for block 1
    L::vectorRows(reinterpret_cast<typename L::Vec::io_t*>(out + arrTail - MaxOffset),
                  reinterpret_cast<const typename L::Vec::io_t*>(in + arrTail - MaxOffset),
                  len - arrTail + MaxOffset,
                  op,
                  (workOffset ^= workSize * maxVecItemSize,
                   L::loadVec((Vecs*)(shm + workOffset), vecs, arrTail % rowLen, rowLen))...);
  }
}

/** Fully occupy GPU this many times for better work balancing. */
static inline constexpr uint OptimalSmOccupancy = 16;

/**
 * Calculate the grid size to be `OptimalSmOccupancy * FullyOccupiedGPU`, where `FullyOccupiedGPU`
 * is the maximum number of blocks fitting in all available SMs.
 *
 * @tparam BlockSize blockDim of the kernel.
 * @return OptimalSmOccupancy * FullyOccupiedGPU
 */
template <int BlockSize>
inline uint getOptimalGridSize()
{
  int devId, smCount, maxBlockSize;
  RAFT_CUDA_TRY(cudaGetDevice(&devId));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, devId));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&maxBlockSize, cudaDevAttrMaxThreadsPerBlock, devId));
  return OptimalSmOccupancy * static_cast<uint>(smCount * maxBlockSize / BlockSize);
}

template <typename Type,
          typename IdxType,
          std::size_t VecBytes,
          int BlockSize,
          typename Lambda,
          typename... Vecs>
void matrixLinewiseVecCols(Type* out,
                           const Type* in,
                           const IdxType rowLen,
                           const IdxType nRows,
                           Lambda op,
                           cudaStream_t stream,
                           const Vecs*... vecs)
{
  typedef raft::Pow2<VecBytes> AlignBytes;
  constexpr std::size_t VecElems = VecBytes / sizeof(Type);
  const IdxType totalLen         = rowLen * nRows;
  const Type* alignedStart       = AlignBytes::roundUp(in);
  const IdxType alignedOff       = IdxType(alignedStart - in);
  const IdxType alignedEnd       = IdxType(AlignBytes::roundDown(in + totalLen) - in);
  const IdxType alignedLen       = alignedEnd - alignedOff;
  if (alignedLen > 0) {
    constexpr dim3 bs(BlockSize, 1, 1);
    // Minimum size of the grid to make the device well occupied
    const uint occupy = getOptimalGridSize<BlockSize>();
    // does not make sense to have more blocks than this
    const uint maxBlocks = raft::ceildiv<uint>(uint(alignedLen), bs.x * VecElems);
    const dim3 gs(std::min(maxBlocks, occupy), 1, 1);
    // The work arrangement is blocked on the block and warp levels;
    //   see more details at Linewise::vectorCols.
    // The value below determines how many scalar elements are processed by on thread in total.
    const IdxType elemsPerThread =
      raft::ceildiv<IdxType>(alignedLen, gs.x * VecElems * BlockSize) * VecElems;
    raft::launch_kernel(
      stream,
      gs,
      bs,
      matrixLinewiseVecColsMainKernel<Type, IdxType, VecBytes, BlockSize, Lambda, Vecs...>,
      out,
      in,
      alignedOff,
      rowLen,
      alignedLen,
      elemsPerThread,
      op,
      vecs...);
  }
  if (alignedLen < totalLen) {
    // should be not smaller than the warp size for better branching
    constexpr std::size_t MaxOffset = std::max(std::size_t(raft::WarpSize), VecBytes);
    raft::launch_kernel(stream,
                        dim3(2, 1, 1),
                        dim3(MaxOffset, 1, 1),
                        matrixLinewiseVecColsTailKernel<Type, IdxType, MaxOffset, Lambda, Vecs...>,
                        out,
                        in,
                        alignedOff,
                        alignedEnd,
                        rowLen,
                        totalLen,
                        op,
                        vecs...);
  }
}

/**
 *  input/output data is expected to be aligned and padded
 *  we simply extend the operation over the padded elements to be fully aligned
 */
template <typename Type,
          typename IdxType,
          typename LayoutPolicy,
          std::size_t VecBytes,
          int BlockSize,
          typename Lambda,
          typename... Vecs>
void matrixLinewiseVecColsSpan(
  raft::device_aligned_matrix_view<Type, IdxType, LayoutPolicy> out,
  raft::device_aligned_matrix_view<const Type, IdxType, LayoutPolicy> in,
  const IdxType rowLen,
  const IdxType nRows,
  Lambda op,
  cudaStream_t stream,
  const Vecs*... vecs)
{
  typedef raft::Pow2<VecBytes> AlignBytes;
  constexpr std::size_t VecElems = VecBytes / sizeof(Type);

  typedef raft::Pow2<raft::layout_left_padded<Type>::padding> AlignPadding;

  const uint paddedRowLen  = AlignPadding::roundUp(rowLen);
  const IdxType alignedLen = paddedRowLen * nRows;

  if (rowLen * nRows > 0) {
    constexpr dim3 bs(BlockSize, 1, 1);
    // Minimum size of the grid to make the device well occupied
    const uint occupy = getOptimalGridSize<BlockSize>();
    // does not make sense to have more blocks than this
    const uint maxBlocks = raft::ceildiv<uint>(uint(alignedLen), bs.x * VecElems);
    const dim3 gs(std::min(maxBlocks, occupy), 1, 1);
    // The work arrangement is blocked on the block and warp levels;
    //   see more details at Linewise::vectorCols.
    // The value below determines how many scalar elements are processed by on thread in total.
    const IdxType elemsPerThread =
      raft::ceildiv<IdxType>(alignedLen, gs.x * VecElems * BlockSize) * VecElems;
    raft::launch_kernel(
      stream,
      gs,
      bs,
      matrixLinewiseVecColsMainKernel<Type, IdxType, VecBytes, BlockSize, Lambda, Vecs...>,
      out.data_handle(),
      in.data_handle(),
      0,
      paddedRowLen,
      alignedLen,
      elemsPerThread,
      op,
      vecs...);
  }
}

template <typename Type,
          typename IdxType,
          std::size_t VecBytes,
          int BlockSize,
          typename Lambda,
          typename... Vecs>
void matrixLinewiseVecRows(Type* out,
                           const Type* in,
                           const IdxType rowLen,
                           const IdxType nRows,
                           Lambda op,
                           cudaStream_t stream,
                           const Vecs*... vecs)
{
  typedef raft::Pow2<VecBytes> AlignBytes;
  constexpr std::size_t VecElems = VecBytes / sizeof(Type);
  const IdxType totalLen         = rowLen * nRows;
  const Type* alignedStart       = AlignBytes::roundUp(in);
  const IdxType alignedOff       = IdxType(alignedStart - in);
  const IdxType alignedEnd       = IdxType(AlignBytes::roundDown(in + totalLen) - in);
  const IdxType alignedLen       = alignedEnd - alignedOff;
  if (alignedLen > 0) {
    constexpr dim3 bs(BlockSize, 1, 1);
    // The work arrangement is striped;
    //   see more details at Linewise::vectorRows.
    // Below is the work amount performed by one block in one iteration.
    constexpr uint block_work_size = bs.x * uint(VecElems);
    /* Here I would define `grid_work_size = lcm(block_work_size, rowLen)` (Least Common Multiple)
       This way, the grid spans a set of one or more rows each iteration, and, most importantly,
       on every iteration each row processes the same set of indices within a row (= the same set
       of vector indices).
       This means, each block needs to load the values from the vector arguments only once.
       Sadly, sometimes `grid_work_size > rowLen*nRows`, and sometimes grid_work_size > UINT_MAX.
       That's why I don't declare it here explicitly.
       Instead, I straightaway compute the
         expected_grid_size = lcm(block_work_size, rowLen) / block_work_size
     */
    const uint expected_grid_size = rowLen / raft::gcd(block_work_size, uint(rowLen));
    // Minimum size of the grid to make the device well occupied
    const uint occupy = getOptimalGridSize<BlockSize>();
    const dim3 gs(std::min(
                    // does not make sense to have more blocks than this
                    raft::ceildiv<uint>(uint(alignedLen), block_work_size),
                    // increase the grid size to be not less than `occupy` while
                    // still being the multiple of `expected_grid_size`
                    raft::ceildiv<uint>(occupy, expected_grid_size) * expected_grid_size),
                  1,
                  1);

    raft::launch_kernel(
      stream,
      gs,
      bs,
      matrixLinewiseVecRowsMainKernel<Type, IdxType, VecBytes, BlockSize, Lambda, Vecs...>,
      out + alignedOff,
      alignedStart,
      alignedOff,
      rowLen,
      alignedLen,
      op,
      vecs...);
  }
  if (alignedLen < totalLen) {
    // should be not smaller than the warp size for better branching
    constexpr std::size_t MaxOffset = std::max(std::size_t(raft::WarpSize), VecBytes);
    raft::launch_kernel(stream,
                        dim3(2, 1, 1),
                        dim3(MaxOffset, 1, 1),
                        matrixLinewiseVecRowsTailKernel<Type, IdxType, MaxOffset, Lambda, Vecs...>,
                        out,
                        in,
                        alignedOff,
                        alignedEnd,
                        rowLen,
                        totalLen,
                        op,
                        vecs...);
  }
}

/**
 *  input/output data is expected to be aligned and padded
 *  we simply extend the operation over the padded elements to be fully aligned
 *  special treatment for 'Vecs' is needed as no elements are available for the padded region
 */
template <typename Type,
          typename IdxType,
          typename LayoutPolicy,
          std::size_t VecBytes,
          int BlockSize,
          typename Lambda,
          typename... Vecs>
void matrixLinewiseVecRowsSpan(
  raft::device_aligned_matrix_view<Type, IdxType, LayoutPolicy> out,
  raft::device_aligned_matrix_view<const Type, IdxType, LayoutPolicy> in,
  const IdxType rowLen,
  const IdxType nRows,
  Lambda op,
  cudaStream_t stream,
  const Vecs*... vecs)
{
  constexpr std::size_t VecElems = VecBytes / sizeof(Type);
  typedef raft::Pow2<VecBytes> AlignBytes;

  typedef raft::Pow2<raft::layout_right_padded<Type>::padding> AlignPadding;

  const uint paddedRowLen  = AlignPadding::roundUp(rowLen);
  const IdxType alignedLen = paddedRowLen * nRows;

  if (rowLen * nRows > 0) {
    constexpr dim3 bs(BlockSize, 1, 1);
    // The work arrangement is striped;
    //   see more details at Linewise::vectorRows.
    // Below is the work amount performed by one block in one iteration.
    constexpr uint block_work_size = bs.x * uint(VecElems);
    /* Here I would define `grid_work_size = lcm(block_work_size, rowLen)` (Least Common Multiple)
       This way, the grid spans a set of one or more rows each iteration, and, most importantly,
       on every iteration each row processes the same set of indices within a row (= the same set
       of vector indices).
       This means, each block needs to load the values from the vector arguments only once.
       Sadly, sometimes `grid_work_size > rowLen*nRows`, and sometimes grid_work_size > UINT_MAX.
       That's why I don't declare it here explicitly.
       Instead, I straightaway compute the
         expected_grid_size = lcm(block_work_size, rowLen) / block_work_size
     */
    const uint expected_grid_size = paddedRowLen / raft::gcd(block_work_size, uint(paddedRowLen));
    // Minimum size of the grid to make the device well occupied
    const uint occupy = getOptimalGridSize<BlockSize>();
    const dim3 gs(std::min(
                    // does not make sense to have more blocks than this
                    raft::ceildiv<uint>(uint(alignedLen), block_work_size),
                    // increase the grid size to be not less than `occupy` while
                    // still being the multiple of `expected_grid_size`
                    raft::ceildiv<uint>(occupy, expected_grid_size) * expected_grid_size),
                  1,
                  1);

    raft::launch_kernel(
      stream,
      gs,
      bs,
      matrixLinewiseVecRowsSpanKernel<Type, IdxType, VecBytes, BlockSize, Lambda, Vecs...>,
      out.data_handle(),
      in.data_handle(),
      rowLen,
      paddedRowLen,
      alignedLen,
      op,
      vecs...);
  }
}

/**
 * Select one of the implementations:
 *   a. vectors applied along/across lines
 *   b. recursively try different VecBytes, such that alignments of `in` and `out`
 *      are the same.
 *
 * @tparam VecBytes - size of the load/store ops in bytes.
 * @tparam BlockSize - is fixed and should not affect the performance.
 */
template <std::size_t VecBytes = 16, int BlockSize = 256>
struct MatrixLinewiseOp {
  template <bool alongLines, typename Type, typename IdxType, typename Lambda, typename... Vecs>
  static void run(Type* out,
                  const Type* in,
                  const IdxType lineLen,
                  const IdxType nLines,
                  Lambda op,
                  cudaStream_t stream,
                  const Vecs*... vecs)
  {
    if constexpr (VecBytes > sizeof(Type)) {
      if (!raft::Pow2<VecBytes>::areSameAlignOffsets(in, out)) {
        constexpr std::size_t NextVecBytes =
          (VecBytes >> 1) > sizeof(Type) ? (VecBytes >> 1) : sizeof(Type);
        return MatrixLinewiseOp<NextVecBytes, BlockSize>::template run<alongLines>(
          out, in, lineLen, nLines, op, stream, vecs...);
      }
    }
    if constexpr (alongLines)
      return matrixLinewiseVecRows<Type, IdxType, VecBytes, BlockSize, Lambda, Vecs...>(
        out, in, lineLen, nLines, op, stream, vecs...);
    else
      return matrixLinewiseVecCols<Type, IdxType, VecBytes, BlockSize, Lambda, Vecs...>(
        out, in, lineLen, nLines, op, stream, vecs...);
  }

  template <bool alongLines,
            typename Type,
            typename IdxType,
            typename LayoutPolicy,
            typename Lambda,
            typename... Vecs>
  static void runPadded(raft::device_aligned_matrix_view<Type, IdxType, LayoutPolicy> out,
                        raft::device_aligned_matrix_view<const Type, IdxType, LayoutPolicy> in,
                        const IdxType lineLen,
                        const IdxType nLines,
                        Lambda op,
                        cudaStream_t stream,
                        const Vecs*... vecs)
  {
    constexpr auto is_rowmajor = std::is_same_v<LayoutPolicy, raft::layout_right_padded<Type>>;
    constexpr auto is_colmajor = std::is_same_v<LayoutPolicy, raft::layout_left_padded<Type>>;

    static_assert(is_rowmajor || is_colmajor,
                  "layout for in and out must be either padded row or col major");

    // also statically assert padded matrix alignment == 2^i*VecBytes
    RAFT_EXPECTS(raft::Pow2<VecBytes>::areSameAlignOffsets(in.data_handle(), out.data_handle()),
                 "The matrix views in and out does not have correct alignment");

    if constexpr (alongLines)
      return matrixLinewiseVecRowsSpan<Type,
                                       IdxType,
                                       LayoutPolicy,
                                       VecBytes,
                                       BlockSize,
                                       Lambda,
                                       Vecs...>(out, in, lineLen, nLines, op, stream, vecs...);
    else
      return matrixLinewiseVecColsSpan<Type,
                                       IdxType,
                                       LayoutPolicy,
                                       VecBytes,
                                       BlockSize,
                                       Lambda,
                                       Vecs...>(out, in, lineLen, nLines, op, stream, vecs...);
  }
};

}  // end namespace detail
}  // end namespace matrix
}  // namespace raft
