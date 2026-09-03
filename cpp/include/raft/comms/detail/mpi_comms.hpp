/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/comms/detail/util.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>

#include <cuda/stream>

#include <mpi.h>
#include <nccl.h>

#include <cstdio>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#define RAFT_MPI_TRY(call)                                                                    \
  do {                                                                                        \
    int status = call;                                                                        \
    if (MPI_SUCCESS != status) {                                                              \
      int mpi_error_string_lenght = 0;                                                        \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                                            \
      MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght);                   \
      RAFT_EXPECTS(                                                                           \
        MPI_SUCCESS == status, "ERROR: MPI call='%s'. Reason:%s\n", #call, mpi_error_string); \
    }                                                                                         \
  } while (0)

// FIXME: Remove after consumer rename
#ifndef MPI_TRY
#define MPI_TRY(call) RAFT_MPI_TRY(call)
#endif

#define RAFT_MPI_TRY_NO_THROW(call)                                         \
  do {                                                                      \
    int status = call;                                                      \
    if (MPI_SUCCESS != status) {                                            \
      int mpi_error_string_lenght = 0;                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                          \
      MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght); \
      printf("MPI call='%s' at file=%s line=%d failed with %s ",            \
             #call,                                                         \
             __FILE__,                                                      \
             __LINE__,                                                      \
             mpi_error_string);                                             \
    }                                                                       \
  } while (0)

// FIXME: Remove after consumer rename
#ifndef MPI_TRY_NO_THROW
#define MPI_TRY_NO_THROW(call) RAFT_MPI_TRY_NO_THROW(call)
#endif
namespace raft {
namespace comms {
namespace detail {

constexpr MPI_Datatype get_mpi_datatype(const datatype_t datatype)
{
  switch (datatype) {
    case datatype_t::CHAR: return MPI_CHAR;
    case datatype_t::UINT8: return MPI_UNSIGNED_CHAR;
    case datatype_t::INT32: return MPI_INT;
    case datatype_t::UINT32: return MPI_UNSIGNED;
    case datatype_t::INT64: return MPI_LONG_LONG;
    case datatype_t::UINT64: return MPI_UNSIGNED_LONG_LONG;
    case datatype_t::FLOAT32: return MPI_FLOAT;
    case datatype_t::FLOAT64: return MPI_DOUBLE;
    default:
      // Execution should never reach here. This takes care of compiler warning.
      return MPI_DOUBLE;
  }
}

constexpr MPI_Op get_mpi_op(const op_t op)
{
  switch (op) {
    case op_t::SUM: return MPI_SUM;
    case op_t::PROD: return MPI_PROD;
    case op_t::MIN: return MPI_MIN;
    case op_t::MAX: return MPI_MAX;
    default:
      // Execution should never reach here. This takes care of compiler warning.
      return MPI_MAX;
  }
}

class mpi_comms : public comms_iface {
 public:
  mpi_comms(MPI_Comm comm, const bool owns_mpi_comm, cuda::stream_ref stream)
    : owns_mpi_comm_(owns_mpi_comm),
      mpi_comm_(comm),
      size_(0),
      rank_(1),
      status_(stream),
      next_request_id_(0),
      stream_(stream)
  {
    int mpi_is_initialized = 0;
    RAFT_MPI_TRY(MPI_Initialized(&mpi_is_initialized));
    RAFT_EXPECTS(mpi_is_initialized, "ERROR: MPI is not initialized!");
    RAFT_MPI_TRY(MPI_Comm_size(mpi_comm_, &size_));
    RAFT_MPI_TRY(MPI_Comm_rank(mpi_comm_, &rank_));
    // get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id;
    if (0 == rank_) RAFT_NCCL_TRY(ncclGetUniqueId(&id));
    RAFT_MPI_TRY(MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, mpi_comm_));

    // initializing NCCL
    ncclConfig_t nccl_config = NCCL_CONFIG_INITIALIZER;
    nccl_config.splitShare   = 1;
    RAFT_NCCL_TRY(ncclCommInitRankConfig(&nccl_comm_, size_, id, rank_, &nccl_config));

    initialize();
  }

  mpi_comms(MPI_Comm mpi_comm, bool owns_mpi_comm, ncclComm_t nccl_comm, cuda::stream_ref stream)
    : owns_mpi_comm_(owns_mpi_comm),
      mpi_comm_(mpi_comm),
      nccl_comm_(nccl_comm),
      size_(0),
      rank_(1),
      status_(stream),
      next_request_id_(0),
      stream_(stream)
  {
    int mpi_is_initialized = 0;
    RAFT_MPI_TRY(MPI_Initialized(&mpi_is_initialized));
    RAFT_EXPECTS(mpi_is_initialized, "ERROR: MPI is not initialized!");
    RAFT_MPI_TRY(MPI_Comm_size(mpi_comm_, &size_));
    RAFT_MPI_TRY(MPI_Comm_rank(mpi_comm_, &rank_));

    initialize();
  }

  void initialize()
  {
    status_.set_value_to_zero_async(stream_);
    buf_ = status_.data();
  }

  virtual ~mpi_comms()
  {
    // finalizing NCCL
    RAFT_NCCL_TRY_NO_THROW(ncclCommDestroy(nccl_comm_));
    if (owns_mpi_comm_) { RAFT_MPI_TRY_NO_THROW(MPI_Comm_free(&mpi_comm_)); }
  }

  int get_size() const { return size_; }

  int get_rank() const { return rank_; }

  std::unique_ptr<comms_iface> comm_split(int color, int key) const
  {
    MPI_Comm new_mpi_comm;
    RAFT_MPI_TRY(MPI_Comm_split(mpi_comm_, color, key, &new_mpi_comm));
    ncclComm_t new_nccl_comm{};
    RAFT_NCCL_TRY(ncclCommSplit(nccl_comm_, color, key, &new_nccl_comm, nullptr));
    return std::unique_ptr<comms_iface>(new mpi_comms(new_mpi_comm, true, new_nccl_comm, stream_));
  }

  void barrier() const
  {
    allreduce(buf_, buf_, 1, datatype_t::INT32, op_t::SUM, stream_);

    ASSERT(sync_stream(stream_) == status_t::SUCCESS,
           "ERROR: syncStream failed. This can be caused by a failed rank_.");
  }

  void isend(const void* buf, size_t size, int dest, int tag, request_t* request) const
  {
    MPI_Request mpi_req;
    request_t req_id;
    if (free_requests_.empty()) {
      req_id = next_request_id_++;
    } else {
      auto it = free_requests_.begin();
      req_id  = *it;
      free_requests_.erase(it);
    }
    RAFT_MPI_TRY(MPI_Isend(buf, size, MPI_BYTE, dest, tag, mpi_comm_, &mpi_req));
    requests_in_flight_.insert(std::make_pair(req_id, mpi_req));
    *request = req_id;
  }

  void irecv(void* buf, size_t size, int source, int tag, request_t* request) const
  {
    MPI_Request mpi_req;
    request_t req_id;
    if (free_requests_.empty()) {
      req_id = next_request_id_++;
    } else {
      auto it = free_requests_.begin();
      req_id  = *it;
      free_requests_.erase(it);
    }

    RAFT_MPI_TRY(MPI_Irecv(buf, size, MPI_BYTE, source, tag, mpi_comm_, &mpi_req));
    requests_in_flight_.insert(std::make_pair(req_id, mpi_req));
    *request = req_id;
  }

  void waitall(int count, request_t array_of_requests[]) const
  {
    std::vector<MPI_Request> requests;
    requests.reserve(count);
    for (int i = 0; i < count; ++i) {
      auto req_it = requests_in_flight_.find(array_of_requests[i]);
      RAFT_EXPECTS(requests_in_flight_.end() != req_it,
                   "ERROR: waitall on invalid request: %d",
                   array_of_requests[i]);
      requests.push_back(req_it->second);
      free_requests_.insert(req_it->first);
      requests_in_flight_.erase(req_it);
    }
    RAFT_MPI_TRY(MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE));
  }

  void allreduce(const void* sendbuff,
                 void* recvbuff,
                 size_t count,
                 datatype_t datatype,
                 op_t op,
                 cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclAllReduce(
      sendbuff, recvbuff, count, get_nccl_datatype(datatype), get_nccl_op(op), nccl_comm_, stream));
  }

  void bcast(void* buff, size_t count, datatype_t datatype, int root, cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(
      ncclBroadcast(buff, buff, count, get_nccl_datatype(datatype), root, nccl_comm_, stream));
  }

  void bcast(const void* sendbuff,
             void* recvbuff,
             size_t count,
             datatype_t datatype,
             int root,
             cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclBroadcast(
      sendbuff, recvbuff, count, get_nccl_datatype(datatype), root, nccl_comm_, stream));
  }

  void reduce(const void* sendbuff,
              void* recvbuff,
              size_t count,
              datatype_t datatype,
              op_t op,
              int root,
              cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclReduce(sendbuff,
                             recvbuff,
                             count,
                             get_nccl_datatype(datatype),
                             get_nccl_op(op),
                             root,
                             nccl_comm_,
                             stream));
  }

  void allgather(const void* sendbuff,
                 void* recvbuff,
                 size_t sendcount,
                 datatype_t datatype,
                 cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclAllGather(
      sendbuff, recvbuff, sendcount, get_nccl_datatype(datatype), nccl_comm_, stream));
  }

  void allgatherv(const void* sendbuf,
                  void* recvbuf,
                  const size_t* recvcounts,
                  const size_t* displs,
                  datatype_t datatype,
                  cudaStream_t stream) const
  {
    RAFT_EXPECTS(size_ <= 2048,
                 "# NCCL operations between ncclGroupStart & ncclGroupEnd shouldn't exceed 2048.");
    // From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" -
    // https://arxiv.org/pdf/1812.05964.pdf Listing 1 on page 4.
    RAFT_NCCL_TRY(ncclGroupStart());
    for (int root = 0; root < size_; ++root) {
      RAFT_NCCL_TRY(
        ncclBroadcast(sendbuf,
                      static_cast<char*>(recvbuf) + displs[root] * get_datatype_size(datatype),
                      recvcounts[root],
                      get_nccl_datatype(datatype),
                      root,
                      nccl_comm_,
                      stream));
    }
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void gather(const void* sendbuff,
              void* recvbuff,
              size_t sendcount,
              datatype_t datatype,
              int root,
              cudaStream_t stream) const
  {
    size_t dtype_size = get_datatype_size(datatype);
    RAFT_NCCL_TRY(ncclGroupStart());
    if (get_rank() == root) {
      for (int r = 0; r < get_size(); ++r) {
        RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + sendcount * r * dtype_size,
                               sendcount,
                               get_nccl_datatype(datatype),
                               r,
                               nccl_comm_,
                               stream));
      }
    }
    RAFT_NCCL_TRY(
      ncclSend(sendbuff, sendcount, get_nccl_datatype(datatype), root, nccl_comm_, stream));
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void gatherv(const void* sendbuff,
               void* recvbuff,
               size_t sendcount,
               const size_t* recvcounts,
               const size_t* displs,
               datatype_t datatype,
               int root,
               cudaStream_t stream) const
  {
    size_t dtype_size = get_datatype_size(datatype);
    RAFT_NCCL_TRY(ncclGroupStart());
    if (get_rank() == root) {
      for (int r = 0; r < get_size(); ++r) {
        RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuff) + displs[r] * dtype_size,
                               recvcounts[r],
                               get_nccl_datatype(datatype),
                               r,
                               nccl_comm_,
                               stream));
      }
    }
    RAFT_NCCL_TRY(
      ncclSend(sendbuff, sendcount, get_nccl_datatype(datatype), root, nccl_comm_, stream));
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void reducescatter(const void* sendbuff,
                     void* recvbuff,
                     size_t recvcount,
                     datatype_t datatype,
                     op_t op,
                     cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclReduceScatter(sendbuff,
                                    recvbuff,
                                    recvcount,
                                    get_nccl_datatype(datatype),
                                    get_nccl_op(op),
                                    nccl_comm_,
                                    stream));
  }

  status_t sync_stream(cudaStream_t stream) const { return nccl_sync_stream(nccl_comm_, stream); }

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_send(const void* buf, size_t size, int dest, cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclSend(buf, size, ncclUint8, dest, nccl_comm_, stream));
  }

  // if a thread is sending & receiving at the same time, use device_sendrecv to avoid deadlock
  void device_recv(void* buf, size_t size, int source, cudaStream_t stream) const
  {
    RAFT_NCCL_TRY(ncclRecv(buf, size, ncclUint8, source, nccl_comm_, stream));
  }

  void device_sendrecv(const void* sendbuf,
                       size_t sendsize,
                       int dest,
                       void* recvbuf,
                       size_t recvsize,
                       int source,
                       cudaStream_t stream) const
  {
    // ncclSend/ncclRecv pair needs to be inside ncclGroupStart/ncclGroupEnd to avoid deadlock
    RAFT_NCCL_TRY(ncclGroupStart());
    RAFT_NCCL_TRY(ncclSend(sendbuf, sendsize, ncclUint8, dest, nccl_comm_, stream));
    RAFT_NCCL_TRY(ncclRecv(recvbuf, recvsize, ncclUint8, source, nccl_comm_, stream));
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void device_multicast_sendrecv(const void* sendbuf,
                                 std::vector<size_t> const& sendsizes,
                                 std::vector<size_t> const& sendoffsets,
                                 std::vector<int> const& dests,
                                 void* recvbuf,
                                 std::vector<size_t> const& recvsizes,
                                 std::vector<size_t> const& recvoffsets,
                                 std::vector<int> const& sources,
                                 cudaStream_t stream) const
  {
    // ncclSend/ncclRecv pair needs to be inside ncclGroupStart/ncclGroupEnd to avoid deadlock
    RAFT_NCCL_TRY(ncclGroupStart());
    for (size_t i = 0; i < sendsizes.size(); ++i) {
      RAFT_NCCL_TRY(ncclSend(static_cast<const char*>(sendbuf) + sendoffsets[i],
                             sendsizes[i],
                             ncclUint8,
                             dests[i],
                             nccl_comm_,
                             stream));
    }
    for (size_t i = 0; i < recvsizes.size(); ++i) {
      RAFT_NCCL_TRY(ncclRecv(static_cast<char*>(recvbuf) + recvoffsets[i],
                             recvsizes[i],
                             ncclUint8,
                             sources[i],
                             nccl_comm_,
                             stream));
    }
    RAFT_NCCL_TRY(ncclGroupEnd());
  }

  void group_start() const { RAFT_NCCL_TRY(ncclGroupStart()); }

  void group_end() const { RAFT_NCCL_TRY(ncclGroupEnd()); }

 private:
  bool owns_mpi_comm_;
  MPI_Comm mpi_comm_;

  cudaStream_t stream_;
  rmm::device_scalar<int32_t> status_;
  int32_t* buf_;

  ncclComm_t nccl_comm_;
  int size_;
  int rank_;
  mutable request_t next_request_id_;
  mutable std::unordered_map<request_t, MPI_Request> requests_in_flight_;
  mutable std::unordered_set<request_t> free_requests_;
};

}  // end namespace detail
};  // end namespace comms
};  // namespace raft
