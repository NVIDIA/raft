/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/comms/detail/ucp_helper.hpp>
#include <raft/comms/detail/util.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/detail/macros.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime.h>

#include <nccl.h>
#include <stdlib.h>
#include <time.h>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include <ucxx/api.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <memory>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace raft {
namespace comms {
namespace detail {

using ucp_endpoint_array_t  = std::shared_ptr<ucp_ep_h*>;
using ucxx_endpoint_array_t = std::shared_ptr<ucxx::Endpoint**>;
using ucp_worker_t          = ucp_worker_h;
using ucxx_worker_t         = ucxx::Worker*;

struct ucx_objects_t {
 public:
  std::variant<ucp_endpoint_array_t, ucxx_endpoint_array_t> endpoints;
  std::variant<ucp_worker_t, ucxx_worker_t> worker;
};

class std_comms : public comms_iface {
 public:
  std_comms() = delete;

  /**
   * @brief Constructor for collective + point-to-point operation.
   * @param nccl_comm initialized nccl comm
   * @param ucp_worker initialized ucp_worker instance
   * @param eps shared pointer to array of ucp endpoints
   * @param num_ranks number of ranks in the cluster
   * @param rank rank of the current worker
   * @param stream cuda stream for synchronizing and ordering collective operations
   * @param subcomms_ucp use ucp for subcommunicators
   */
  std_comms(ncclComm_t nccl_comm,
            ucx_objects_t ucx_objects,
            int num_ranks,
            int rank,
            rmm::cuda_stream_view stream,
            bool subcomms_ucp = true)
    : nccl_comm_(nccl_comm),
      stream_(stream),
      status_(stream),
      num_ranks_(num_ranks),
      rank_(rank),
      subcomms_ucp_(subcomms_ucp),
      ucx_objects_(ucx_objects),
      own_nccl_comm_(false),
      next_request_id_(0)
  {
    initialize();
  };

  /**
   * @brief constructor for collective-only operation
   * @param nccl_comm initialized nccl communicator
   * @param num_ranks size of the cluster
   * @param rank rank of the current worker
   * @param stream stream for ordering collective operations
   */
  std_comms(const ncclComm_t nccl_comm,
            int num_ranks,
            int rank,
            rmm::cuda_stream_view stream,
            bool own_nccl_comm = false)
    : nccl_comm_(nccl_comm),
      stream_(stream),
      status_(stream),
      num_ranks_(num_ranks),
      rank_(rank),
      subcomms_ucp_(false),
      own_nccl_comm_(own_nccl_comm)
  {
    initialize();
  };

  void initialize()
  {
    status_.set_value_to_zero_async(stream_);
    buf_ = status_.data();
  }

  ~std_comms()
  {
    requests_in_flight_.clear();
    free_requests_.clear();

    if (own_nccl_comm_) {
      RAFT_NCCL_TRY_NO_THROW(ncclCommDestroy(nccl_comm_));
      nccl_comm_ = nullptr;
    }
  }

  int get_size() const { return num_ranks_; }

  int get_rank() const { return rank_; }

  std::unique_ptr<comms_iface> comm_split(int color, int key) const
  {
    ncclComm_t new_nccl_comm{};
    RAFT_NCCL_TRY(ncclCommSplit(nccl_comm_, color, key, &new_nccl_comm, nullptr));
    int new_nccl_comm_size{};
    RAFT_NCCL_TRY(ncclCommCount(new_nccl_comm, &new_nccl_comm_size));
    return std::unique_ptr<comms_iface>(
      new std_comms(new_nccl_comm, new_nccl_comm_size, key, stream_, true));
  }

  void barrier() const
  {
    allreduce(buf_, buf_, 1, datatype_t::INT32, op_t::SUM, stream_);

    ASSERT(sync_stream(stream_) == status_t::SUCCESS,
           "ERROR: syncStream failed. This can be caused by a failed rank_.");
  }

  void get_request_id(request_t* req) const
  {
    request_t req_id;

    if (this->free_requests_.empty())
      req_id = this->next_request_id_++;
    else {
      auto it = this->free_requests_.begin();
      req_id  = *it;
      this->free_requests_.erase(it);
    }
    *req = req_id;
  }

  void isend(const void* buf, size_t size, int dest, int tag, request_t* request) const
  {
    if (std::holds_alternative<ucxx_worker_t>(ucx_objects_.worker)) {
      get_request_id(request);

      ucxx::Endpoint* ep_ptr = (*std::get<ucxx_endpoint_array_t>(ucx_objects_.endpoints))[dest];

      ucp_tag_t ucp_tag = build_message_tag(get_rank(), tag);
      auto ucxx_req     = ep_ptr->tagSend(const_cast<void*>(buf), size, ucxx::Tag(ucp_tag));

      requests_in_flight_.insert(std::make_pair(*request, ucxx_req));
    } else {
      ASSERT(std::get<ucp_worker_t>(ucx_objects_.worker) != nullptr,
             "ERROR: UCX comms not initialized on communicator.");

      get_request_id(request);
      ucp_ep_h ep_ptr = (*std::get<ucp_endpoint_array_t>(ucx_objects_.endpoints))[dest];

      ucp_request* ucp_req = (ucp_request*)malloc(sizeof(ucp_request));

      this->ucp_handler_.ucp_isend(ucp_req, ep_ptr, buf, size, tag, default_tag_mask, get_rank());

      requests_in_flight_.insert(std::make_pair(*request, ucp_req));
    }
  }

  void irecv(void* buf, size_t size, int source, int tag, request_t* request) const
  {
    if (std::holds_alternative<ucxx_worker_t>(ucx_objects_.worker)) {
      get_request_id(request);

      ucxx::Endpoint* ep_ptr = (*std::get<ucxx_endpoint_array_t>(ucx_objects_.endpoints))[source];

      ucp_tag_t ucp_tag = build_message_tag(source, tag);
      auto ucxx_req =
        ep_ptr->tagRecv(buf, size, ucxx::Tag(ucp_tag), ucxx::TagMask(default_tag_mask));

      requests_in_flight_.insert(std::make_pair(*request, ucxx_req));
    } else {
      ASSERT(std::get<ucp_worker_t>(ucx_objects_.worker) != nullptr,
             "ERROR: UCX comms not initialized on communicator.");

      get_request_id(request);

      ucp_ep_h ep_ptr = (*std::get<ucp_endpoint_array_t>(ucx_objects_.endpoints))[source];

      ucp_tag_t tag_mask = default_tag_mask;

      ucp_request* ucp_req = (ucp_request*)malloc(sizeof(ucp_request));
      ucp_handler_.ucp_irecv(ucp_req,
                             std::get<ucp_worker_t>(ucx_objects_.worker),
                             ep_ptr,
                             buf,
                             size,
                             tag,
                             tag_mask,
                             source);

      requests_in_flight_.insert(std::make_pair(*request, ucp_req));
    }
  }

  void waitall(int count, request_t array_of_requests[]) const
  {
    if (std::holds_alternative<ucxx_worker_t>(ucx_objects_.worker)) {
      ucxx_worker_t worker = std::get<ucxx_worker_t>(ucx_objects_.worker);

      std::vector<std::shared_ptr<ucxx::Request>> requests;
      requests.reserve(count);

      time_t start = time(NULL);

      for (int i = 0; i < count; ++i) {
        auto req_it = requests_in_flight_.find(array_of_requests[i]);
        ASSERT(requests_in_flight_.end() != req_it,
               "ERROR: waitall on invalid request: %d",
               array_of_requests[i]);
        requests.push_back(std::get<std::shared_ptr<ucxx::Request>>(req_it->second));
        free_requests_.insert(req_it->first);
        requests_in_flight_.erase(req_it);
      }

      while (requests.size() > 0) {
        time_t now = time(NULL);

        // Timeout if we have not gotten progress or completed any requests
        // in 10 or more seconds.
        ASSERT(now - start < 10, "Timed out waiting for requests.");

        for (std::vector<std::shared_ptr<ucxx::Request>>::iterator it = requests.begin();
             it != requests.end();) {
          bool restart = false;  // resets the timeout when any progress was made

          if (worker->isProgressThreadRunning()) {
            // Progress thread is running in background, just yield to let it work
            std::this_thread::yield();
          } else {
            // Causes UCXX to progress through the send/recv message queue
            while (!worker->progress()) {
              restart = true;
            }
          }

          auto req = *it;

          // If the message needs release, we know it will be sent/received
          // asynchronously, so we will need to track and verify its state
          if (req->isCompleted()) {
            auto status = req->getStatus();
            ASSERT(req->getStatus() == UCS_OK,
                   "UCX Request Error: %d (%s)\n",
                   status,
                   ucs_status_string(status));
          }

          // If a message was sent synchronously (eg. completed before
          // `isend`/`irecv` completed) or an asynchronous message
          // is complete, we can go ahead and clean it up.
          if (req->isCompleted()) {
            restart = true;

            auto status = req->getStatus();
            ASSERT(req->getStatus() == UCS_OK,
                   "UCX Request Error: %d (%s)\n",
                   status,
                   ucs_status_string(status));

            // remove from pending requests
            it = requests.erase(it);
          } else {
            ++it;
          }
          // if any progress was made, reset the timeout start time
          if (restart) { start = time(NULL); }
        }
      }
    } else {
      ucp_worker_t worker = std::get<ucp_worker_t>(ucx_objects_.worker);
      ASSERT(worker != nullptr, "ERROR: UCX comms not initialized on communicator.");

      std::vector<ucp_request*> requests;
      requests.reserve(count);

      time_t start = time(NULL);

      for (int i = 0; i < count; ++i) {
        auto req_it = requests_in_flight_.find(array_of_requests[i]);
        ASSERT(requests_in_flight_.end() != req_it,
               "ERROR: waitall on invalid request: %d",
               array_of_requests[i]);
        requests.push_back(std::get<ucp_request*>(req_it->second));
        free_requests_.insert(req_it->first);
        requests_in_flight_.erase(req_it);
      }

      while (requests.size() > 0) {
        time_t now = time(NULL);

        // Timeout if we have not gotten progress or completed any requests
        // in 10 or more seconds.
        ASSERT(now - start < 10, "Timed out waiting for requests.");

        for (std::vector<ucp_request*>::iterator it = requests.begin(); it != requests.end();) {
          bool restart = false;  // resets the timeout when any progress was made

          // Causes UCP to progress through the send/recv message queue
          while (ucp_worker_progress(worker) != 0) {
            restart = true;
          }

          auto req = *it;

          // If the message needs release, we know it will be sent/received
          // asynchronously, so we will need to track and verify its state
          if (req->needs_release) {
            ASSERT(UCS_PTR_IS_PTR(req->req), "UCX Request Error. Request is not valid UCX pointer");
            ASSERT(!UCS_PTR_IS_ERR(req->req), "UCX Request Error: %d\n", UCS_PTR_STATUS(req->req));
            ASSERT(req->req->completed == 1 || req->req->completed == 0,
                   "request->completed not a valid value: %d\n",
                   req->req->completed);
          }

          // If a message was sent synchronously (eg. completed before
          // `isend`/`irecv` completed) or an asynchronous message
          // is complete, we can go ahead and clean it up.
          if (!req->needs_release || req->req->completed == 1) {
            restart = true;

            // perform cleanup
            ucp_handler_.free_ucp_request(req);

            // remove from pending requests
            it = requests.erase(it);
          } else {
            ++it;
          }
          // if any progress was made, reset the timeout start time
          if (restart) { start = time(NULL); }
        }
      }
    }
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
    // From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" -
    // https://arxiv.org/pdf/1812.05964.pdf Listing 1 on page 4.
    RAFT_EXPECTS(num_ranks_ <= 2048,
                 "# NCCL operations between ncclGroupStart & ncclGroupEnd shouldn't exceed 2048.");
    RAFT_NCCL_TRY(ncclGroupStart());
    for (int root = 0; root < num_ranks_; ++root) {
      size_t dtype_size = get_datatype_size(datatype);
      RAFT_NCCL_TRY(ncclBroadcast(sendbuf,
                                  static_cast<char*>(recvbuf) + displs[root] * dtype_size,
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
  ncclComm_t nccl_comm_;
  cudaStream_t stream_;

  rmm::device_scalar<int32_t> status_;

  int32_t* buf_;
  int num_ranks_;
  int rank_;

  bool subcomms_ucp_;
  bool own_nccl_comm_;

  comms_ucp_handler ucp_handler_;
  ucx_objects_t ucx_objects_;
  mutable request_t next_request_id_;
  mutable std::unordered_map<request_t,
                             std::variant<struct ucp_request*, std::shared_ptr<ucxx::Request>>>
    requests_in_flight_;
  mutable std::unordered_set<request_t> free_requests_;
};
}  // namespace detail
}  // end namespace comms
}  // namespace raft
