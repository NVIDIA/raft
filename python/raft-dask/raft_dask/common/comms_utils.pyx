# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cpython.long cimport PyLong_AsVoidPtr
from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free, malloc
from libcpp cimport bool


cdef extern from "nccl.h":

    cdef struct ncclComm
    ctypedef ncclComm *ncclComm_t

cdef extern from "raft/core/handle.hpp" namespace "raft":
    cdef cppclass device_resources:
        device_resources() except +

cdef extern from "raft/core/device_resources.hpp" namespace "raft":
    cdef cppclass device_resources:
        device_resources() except +

cdef extern from "raft/comms/std_comms.hpp" namespace "raft::comms":

    void build_comms_nccl_ucx(device_resources *handle,
                              ncclComm_t comm,
                              bint is_ucxx,
                              void *ucp_worker,
                              void *eps,
                              int size,
                              int rank) except +

    void build_comms_nccl_only(device_resources *handle,
                               ncclComm_t comm,
                               int size,
                               int rank) except +

cdef extern from "raft/comms/comms_test.hpp" namespace "raft::comms":

    bool test_collective_allreduce(const device_resources &h, int root) \
        except +
    bool test_collective_broadcast(const device_resources &h, int root) \
        except +
    bool test_collective_reduce(const device_resources &h, int root) except +
    bool test_collective_allgather(const device_resources &h, int root) \
        except +
    bool test_collective_gather(const device_resources &h, int root) except +
    bool test_collective_gatherv(const device_resources &h, int root) except +
    bool test_collective_reducescatter(const device_resources &h, int root) \
        except +
    bool test_pointToPoint_simple_send_recv(const device_resources &h,
                                            int numTrials) except +
    bool test_pointToPoint_device_send_or_recv(const device_resources &h,
                                               int numTrials) except +
    bool test_pointToPoint_device_sendrecv(const device_resources &h,
                                           int numTrials) except +
    bool test_pointToPoint_device_multicast_sendrecv(const device_resources &h,
                                                     int numTrials) except +
    bool test_commsplit(const device_resources &h, int n_colors) except +


def perform_test_comms_allreduce(handle, root):
    """
    Performs an allreduce on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const device_resources* h = \
        <device_resources*><size_t>handle.getHandle()
    return test_collective_allreduce(deref(h), root)


def perform_test_comms_reduce(handle, root):
    """
    Performs a reduce on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const device_resources* h = \
        <device_resources*><size_t>handle.getHandle()
    return test_collective_reduce(deref(h), root)


def perform_test_comms_reducescatter(handle, root):
    """
    Performs a reducescatter on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const device_resources* h = \
        <device_resources*><size_t>handle.getHandle()
    return test_collective_reducescatter(deref(h), root)


def perform_test_comms_bcast(handle, root):
    """
    Performs a broadcast on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const device_resources* h = \
        <device_resources*><size_t>handle.getHandle()
    return test_collective_broadcast(deref(h), root)


def perform_test_comms_allgather(handle, root):
    """
    Performs an allgather on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const device_resources* h = \
        <device_resources*><size_t>handle.getHandle()
    return test_collective_allgather(deref(h), root)


def perform_test_comms_gather(handle, root):
    """
    Performs a gather on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    root : int
           Rank of the root worker
    """
    cdef const device_resources* h = \
        <device_resources*><size_t>handle.getHandle()
    return test_collective_gather(deref(h), root)


def perform_test_comms_gatherv(handle, root):
    """
    Performs a gatherv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    root : int
           Rank of the root worker
    """
    cdef const device_resources* h = \
        <device_resources*><size_t>handle.getHandle()
    return test_collective_gatherv(deref(h), root)


def perform_test_comms_send_recv(handle, n_trials):
    """
    Performs a p2p send/recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    n_trilas : int
               Number of test trials
    """
    cdef const device_resources *h = \
        <device_resources*><size_t>handle.getHandle()
    return test_pointToPoint_simple_send_recv(deref(h), <int>n_trials)


def perform_test_comms_device_send_or_recv(handle, n_trials):
    """
    Performs a p2p device send or recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    n_trilas : int
               Number of test trials
    """
    cdef const device_resources *h = \
        <device_resources*><size_t>handle.getHandle()
    return test_pointToPoint_device_send_or_recv(deref(h), <int>n_trials)


def perform_test_comms_device_sendrecv(handle, n_trials):
    """
    Performs a p2p device concurrent send&recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    n_trilas : int
               Number of test trials
    """
    cdef const device_resources *h = \
        <device_resources*><size_t>handle.getHandle()
    return test_pointToPoint_device_sendrecv(deref(h), <int>n_trials)


def perform_test_comms_device_multicast_sendrecv(handle, n_trials):
    """
    Performs a p2p device concurrent multicast send&recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    n_trilas : int
               Number of test trials
    """
    cdef const device_resources *h = \
        <device_resources *> <size_t> handle.getHandle()
    return test_pointToPoint_device_multicast_sendrecv(deref(h), <int>n_trials)


def perform_test_comm_split(handle, n_colors):
    """
    Performs a p2p send/recv on the current worker

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    """
    cdef const device_resources * h = \
        < device_resources * > < size_t > handle.getHandle()
    return test_commsplit(deref(h), < int > n_colors)


def inject_comms_on_handle_coll_only(handle, nccl_inst, size, rank, verbose):
    """
    Given a handle and initialized nccl comm, creates a comms_t
    instance and injects it into the handle.

        Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    nccl_inst : raft.dask.common.nccl
                Initialized nccl comm to use
    size : int
           Number of workers in cluster
    rank : int
           Rank of current worker

    """

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <device_resources*>handle_size_t

    cdef size_t nccl_comm_size_t = <size_t>nccl_inst.get_comm()
    nccl_comm_ = <ncclComm_t*>nccl_comm_size_t

    build_comms_nccl_only(handle_,
                          deref(nccl_comm_),
                          size,
                          rank)


def inject_comms_on_handle(handle, nccl_inst, is_ucxx, ucp_worker, eps, size,
                           rank, verbose):
    """
    Given a handle and initialized comms, creates a comms_t instance
    and injects it into the handle.

    Parameters
    ----------
    handle : raft.common.Handle
             handle containing comms_t to use
    nccl_inst : raft.dask.common.nccl
                Initialized nccl comm to use
    ucp_worker : size_t pointer to initialized ucp_worker_h instance
    eps: size_t pointer to array of initialized ucp_ep_h instances
    size : int
           Number of workers in cluster
    rank : int
           Rank of current worker
    """
    cdef size_t *ucp_eps = <size_t*> malloc(len(eps)*sizeof(size_t))

    for i in range(len(eps)):
        if eps[i] is not None:
            if is_ucxx:
                ep_st = <uintptr_t>eps[i].ucxx_endpoint
            else:
                ep_st = <uintptr_t>eps[i].get_ucp_endpoint()
            ucp_eps[i] = <size_t>ep_st
        else:
            ucp_eps[i] = 0

    cdef void* ucp_worker_st = <void*><size_t>ucp_worker

    cdef size_t handle_size_t = <size_t>handle.getHandle()
    handle_ = <device_resources*>handle_size_t

    cdef size_t nccl_comm_size_t = <size_t>nccl_inst.get_comm()
    nccl_comm_ = <ncclComm_t*>nccl_comm_size_t

    build_comms_nccl_ucx(handle_,
                         deref(nccl_comm_),
                         is_ucxx,
                         <void*>ucp_worker_st,
                         <void*>ucp_eps,
                         size,
                         rank)

    free(ucp_eps)
