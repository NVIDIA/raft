# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import logging
import os
import time
import uuid
import warnings
from collections import OrderedDict

from dask.distributed import default_client
from dask_cuda.utils import nvml_device_index

from pylibraft.common.handle import Handle

from .comms_utils import (
    inject_comms_on_handle,
    inject_comms_on_handle_coll_only,
)
from .nccl import nccl
from .ucx import UCX
from .utils import parse_host_port

logger = logging.getLogger(__name__)


class Comms:
    """
    Initializes and manages underlying NCCL and UCX comms handles across
    the workers of a Dask cluster. It is expected that `init()` will be
    called explicitly. It is recommended to also call `destroy()` when
    the comms are no longer needed so the underlying resources can be
    cleaned up. This class is not meant to be thread-safe.

    Examples
    --------
    .. code-block:: python

        # The following code block assumes we have wrapped a C++
        # function in a Python function called `run_algorithm`,
        # which takes a `raft::handle_t` as a single argument.
        # Once the `Comms` instance is successfully initialized,
        # the underlying `raft::handle_t` will contain an instance
        # of `raft::comms::comms_t`

        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client, get_worker, wait

        from raft_dask.common import Comms, local_handle

        cluster = LocalCUDACluster()
        client = Client(cluster)

        def _use_comms(sessionId):
            return run_algorithm(local_handle(sessionId, dask_worker=get_worker()))

        comms = Comms(client=client)
        comms.init()

        futures = [client.submit(_use_comms,
                                 comms.sessionId,
                                 workers=[w],
                                 pure=False) # Don't memoize
                       for w in comms.worker_addresses]
        wait(futures, timeout=5)

        comms.destroy()
        client.close()
        cluster.close()
    """

    valid_nccl_placements = ("client", "worker", "scheduler")

    def __init__(
        self,
        comms_p2p=False,
        client=None,
        verbose=False,
        streams_per_handle=0,
        nccl_root_location="scheduler",
    ):
        """
        Construct a new CommsContext instance

        Parameters
        ----------
        comms_p2p : bool
                    Initialize UCX endpoints?
        client : dask.distributed.Client [optional]
                 Dask client to use
        verbose : bool
                  Print verbose logging
        nccl_root_location : string
                  Indicates where the NCCL's root node should be located.
                  ['client', 'worker', 'scheduler' (default)]

        """
        self.client = client if client is not None else default_client()

        self.comms_p2p = comms_p2p

        self.nccl_root_location = nccl_root_location.lower()
        if self.nccl_root_location not in Comms.valid_nccl_placements:
            raise ValueError(
                f"nccl_root_location must be one of: "
                f"{Comms.valid_nccl_placements}"
            )

        self.streams_per_handle = streams_per_handle

        self.sessionId = uuid.uuid4().bytes

        self.nccl_initialized = False
        self.ucx_initialized = False

        self.verbose = verbose

        if verbose:
            print("Initializing comms!")

    def __del__(self):
        if self.nccl_initialized or self.ucx_initialized:
            self.destroy()

    def create_nccl_uniqueid(self):
        if self.nccl_root_location == "client":
            self.uniqueId = nccl.get_unique_id()
        elif self.nccl_root_location == "worker":
            self.uniqueId = self.client.run(
                _func_set_worker_as_nccl_root,
                sessionId=self.sessionId,
                verbose=self.verbose,
                workers=[self.worker_addresses[0]],
                wait=True,
            )[self.worker_addresses[0]]
        else:
            self.uniqueId = self.client.run_on_scheduler(
                _func_set_scheduler_as_nccl_root,
                sessionId=self.sessionId,
                verbose=self.verbose,
            )

    def worker_info(self, workers):
        """
        Builds a dictionary of { (worker_address, worker_port) :
                                (worker_rank, worker_port ) }
        """
        ranks = _func_worker_ranks(self.client, workers)
        ports = (
            _func_ucp_ports(self.client, workers) if self.comms_p2p else None
        )

        output = {}
        for k in ranks.keys():
            output[k] = {"rank": ranks[k]}
            if self.comms_p2p:
                output[k]["port"] = ports[k]
        return output

    def init(self, workers=None):
        """
        Initializes the underlying comms. NCCL is required but
        UCX is only initialized if `comms_p2p == True`

        Parameters
        ----------
        workers : Sequence
                  Unique collection of workers for initializing comms.
        """

        self.worker_addresses = list(
            OrderedDict.fromkeys(
                self.client.scheduler_info(n_workers=-1)["workers"].keys()
                if workers is None
                else workers
            )
        )

        if self.nccl_initialized or self.ucx_initialized:
            warnings.warn("Comms have already been initialized.")
            return

        worker_info = self.worker_info(self.worker_addresses)
        worker_info = {w: worker_info[w] for w in self.worker_addresses}

        self.create_nccl_uniqueid()

        self.client.run(
            _func_init_all,
            self.sessionId,
            self.uniqueId,
            self.comms_p2p,
            worker_info,
            self.verbose,
            self.streams_per_handle,
            workers=self.worker_addresses,
            wait=True,
        )

        self.nccl_initialized = True

        if self.comms_p2p:
            self.ucx_initialized = True

        if self.verbose:
            print("Initialization complete.")

    def destroy(self):
        """
        Shuts down initialized comms and cleans up resources. This will
        be called automatically by the Comms destructor, but may be called
        earlier to save resources.
        """
        self.client.run(
            _func_destroy_all,
            self.sessionId,
            self.comms_p2p,
            self.verbose,
            wait=True,
            workers=self.worker_addresses,
        )

        if self.nccl_root_location == "scheduler":
            self.client.run_on_scheduler(
                _func_destroy_scheduler_session, self.sessionId
            )

        if self.verbose:
            print("Destroying comms.")

        self.nccl_initialized = False
        self.ucx_initialized = False


def local_handle(sessionId, dask_worker=None):
    """
    Simple helper function for retrieving the local handle_t instance
    for a comms session on a worker.

    Parameters
    ----------
    sessionId : str
                session identifier from an initialized comms instance
    dask_worker : dask_worker object
                  (Note: if called by client.run(), this is supplied by Dask
                   and not the client)

    Returns
    -------
    handle : raft.Handle or None
    """
    state = get_raft_comm_state(sessionId, dask_worker)
    return state["handle"] if "handle" in state else None


def get_raft_comm_state(sessionId, state_object=None, dask_worker=None):
    """
    Retrieves cuML comms state on the scheduler node, for the given sessionId,
    creating a new session if it does not exist. If no session id is given,
    returns the state dict for all sessions.

    Parameters
    ----------
    sessionId : SessionId value to retrieve from the dask_scheduler instances
    state_object : Object (either Worker, or Scheduler) on which the raft
                   comm state will retrieved (or created)
    dask_worker : dask_worker object
                  (Note: if called by client.run(), this is supplied by Dask
                   and not the client)

    Returns
    -------
    session state : str
                    session state associated with sessionId
    """
    state_object = state_object if state_object is not None else dask_worker

    if not hasattr(state_object, "_raft_comm_state"):
        state_object._raft_comm_state = {}

    if (
        sessionId is not None
        and sessionId not in state_object._raft_comm_state
    ):
        state_object._raft_comm_state[sessionId] = {"ts": time.time()}

    if sessionId is not None:
        return state_object._raft_comm_state[sessionId]

    return state_object._raft_comm_state


def set_nccl_root(sessionId, state_object):
    if sessionId is None:
        raise ValueError("sessionId cannot be None.")

    raft_comm_state = get_raft_comm_state(
        sessionId=sessionId, state_object=state_object
    )

    if "nccl_uid" not in raft_comm_state:
        raft_comm_state["nccl_uid"] = nccl.get_unique_id()

    return raft_comm_state["nccl_uid"]


def get_ucx(dask_worker=None):
    """
    A simple convenience wrapper to make sure UCP listener and
    endpoints are only ever assigned once per worker.

    Parameters
    ----------
    dask_worker : dask_worker object
                  (Note: if called by client.run(), this is supplied by Dask
                   and not the client)
    """
    raft_comm_state = get_raft_comm_state(
        sessionId="ucp", state_object=dask_worker
    )
    if "ucx" not in raft_comm_state:
        raft_comm_state["ucx"] = UCX.get()

    return raft_comm_state["ucx"]


def _func_destroy_scheduler_session(sessionId, dask_scheduler):
    """
    Remove session date from _raft_comm_state, associated with sessionId

    Parameters
    ----------
    sessionId : session Id to be destroyed.
    dask_scheduler : dask_scheduler object
                    (Note: this is supplied by DASK, not the client)
    """
    if sessionId is not None and sessionId in dask_scheduler._raft_comm_state:
        del dask_scheduler._raft_comm_state[sessionId]
    else:
        return 1

    return 0


def _func_set_scheduler_as_nccl_root(sessionId, verbose, dask_scheduler):
    """
    Creates a persistent nccl uniqueId on the scheduler node.


    Parameters
    ----------
    sessionId : Associated session to attach the unique ID to.
    verbose : Indicates whether or not to emit additional information
    dask_scheduler : dask scheduler object,
                    (Note: this is supplied by DASK, not the client)

    Return
    ------
    uniqueId : byte str
                NCCL uniqueId, associating the DASK scheduler as its root node.
    """
    if verbose:
        logger.info(
            msg=f"Setting scheduler as NCCL root for sessionId, '{sessionId}'"
        )

    nccl_uid = set_nccl_root(sessionId=sessionId, state_object=dask_scheduler)

    if verbose:
        logger.info("Done setting scheduler as NCCL root.")

    return nccl_uid


def _func_set_worker_as_nccl_root(sessionId, verbose, dask_worker=None):
    """
    Creates a persistent nccl uniqueId on the scheduler node.


    Parameters
    ----------
    sessionId : Associated session to attach the unique ID to.
    verbose : Indicates whether or not to emit additional information
    dask_worker : dask_worker object
                  (Note: if called by client.run(), this is supplied by Dask
                   and not the client)

    Return
    ------
    uniqueId : byte str
                NCCL uniqueId, associating this DASK worker as its root node.
    """
    if verbose:
        dask_worker.log_event(
            topic="info",
            msg=f"Setting worker as NCCL root for session, '{sessionId}'",
        )

    nccl_uid = set_nccl_root(sessionId=sessionId, state_object=dask_worker)

    if verbose:
        dask_worker.log_event(
            topic="info", msg="Done setting scheduler as NCCL root."
        )

    return nccl_uid


def _func_ucp_listener_port(dask_worker=None):
    return get_ucx(dask_worker=dask_worker).listener_port()


async def _func_init_all(
    sessionId,
    uniqueId,
    comms_p2p,
    worker_info,
    verbose,
    streams_per_handle,
    dask_worker=None,
):
    raft_comm_state = get_raft_comm_state(
        sessionId=sessionId, state_object=dask_worker
    )
    raft_comm_state["nccl_uid"] = uniqueId
    raft_comm_state["wid"] = worker_info[dask_worker.address]["rank"]
    raft_comm_state["nworkers"] = len(worker_info)

    if verbose:
        dask_worker.log_event(topic="info", msg="Initializing NCCL.")
        start = time.time()

    _func_init_nccl(sessionId, uniqueId, dask_worker=dask_worker)

    if verbose:
        elapsed = time.time() - start
        dask_worker.log_event(
            topic="info", msg=f"NCCL Initialization took: {elapsed} seconds."
        )

    if comms_p2p:
        if verbose:
            dask_worker.log_event(
                topic="info", msg="Initializing UCX Endpoints"
            )

        if verbose:
            start = time.time()
        await _func_ucp_create_endpoints(
            sessionId, worker_info, dask_worker=dask_worker
        )

        if verbose:
            elapsed = time.time() - start
            msg = (
                f"Done initializing UCX endpoints."
                f"Took: {elapsed} seconds.\nBuilding handle."
            )
            dask_worker.log_event(topic="info", msg=msg)

        _func_build_handle_p2p(
            sessionId, streams_per_handle, verbose, dask_worker=dask_worker
        )

        if verbose:
            dask_worker.log_event(topic="info", msg="Done building handle.")

    else:
        _func_build_handle(
            sessionId, streams_per_handle, verbose, dask_worker=dask_worker
        )


def _func_init_nccl(sessionId, uniqueId, dask_worker=None):
    """
    Initialize ncclComm_t on worker

    Parameters
    ----------
    sessionId : str
                session identifier from a comms instance
    uniqueId : array[byte]
               The NCCL unique Id generated from the
               client.
    dask_worker : dask_worker object
                  (Note: if called by client.run(), this is supplied by Dask
                   and not the client)
    """

    raft_comm_state = get_raft_comm_state(
        sessionId=sessionId, state_object=dask_worker, dask_worker=dask_worker
    )
    wid = raft_comm_state["wid"]
    nWorkers = raft_comm_state["nworkers"]

    try:
        n = nccl()
        n.init(nWorkers, uniqueId, wid)
        raft_comm_state["nccl"] = n
    except Exception as e:
        dask_worker.log_event(
            topic="error", msg=f"An error occurred initializing NCCL: {e}."
        )
        raise


def _func_build_handle_p2p(
    sessionId, streams_per_handle, verbose, dask_worker=None
):
    """
    Builds a handle_t on the current worker given the initialized comms

    Parameters
    ----------
    sessionId : str id to reference state for current comms instance.
    streams_per_handle : int number of internal streams to create
    verbose : bool print verbose logging output
    dask_worker : dask_worker object
                  (Note: if called by client.run(), this is supplied by Dask
                   and not the client)
    """
    if verbose:
        dask_worker.log_event(topic="info", msg="Building p2p handle.")

    ucx = get_ucx(dask_worker)
    ucx_worker = ucx.get_worker()
    raft_comm_state = get_raft_comm_state(
        sessionId=sessionId, state_object=dask_worker
    )

    handle = Handle(n_streams=streams_per_handle)
    nccl_comm = raft_comm_state["nccl"]
    eps = raft_comm_state["ucp_eps"]
    nWorkers = raft_comm_state["nworkers"]
    workerId = raft_comm_state["wid"]

    if verbose:
        dask_worker.log_event(topic="info", msg="Injecting comms on handle.")

    inject_comms_on_handle(
        handle,
        nccl_comm,
        True,
        ucx_worker,
        eps,
        nWorkers,
        workerId,
        verbose,
    )

    if verbose:
        dask_worker.log_event(
            topic="info", msg="Finished injecting comms on handle."
        )

    raft_comm_state["handle"] = handle


def _func_build_handle(
    sessionId, streams_per_handle, verbose, dask_worker=None
):
    """
    Builds a handle_t on the current worker given the initialized comms

    Parameters
    ----------
    sessionId : str id to reference state for current comms instance.
    streams_per_handle : int number of internal streams to create
    verbose : bool print verbose logging output
    dask_worker : dask_worker object
                  (Note: if called by client.run(), this is supplied by Dask
                   and not the client)
    """
    if verbose:
        dask_worker.log_event(topic="info", msg="Injecting comms on handle.")

    handle = Handle(n_streams=streams_per_handle)

    raft_comm_state = get_raft_comm_state(
        sessionId=sessionId, state_object=dask_worker
    )

    workerId = raft_comm_state["wid"]
    nWorkers = raft_comm_state["nworkers"]

    nccl_comm = raft_comm_state["nccl"]
    inject_comms_on_handle_coll_only(
        handle, nccl_comm, nWorkers, workerId, verbose
    )
    raft_comm_state["handle"] = handle


def _func_store_initial_state(
    nworkers, sessionId, uniqueId, wid, dask_worker=None
):
    raft_comm_state = get_raft_comm_state(
        sessionId=sessionId, state_object=dask_worker
    )
    raft_comm_state["nccl_uid"] = uniqueId
    raft_comm_state["wid"] = wid
    raft_comm_state["nworkers"] = nworkers


async def _func_ucp_create_endpoints(sessionId, worker_info, dask_worker):
    """
    Runs on each worker to create ucp endpoints to all other workers

    Parameters
    ----------
    sessionId : str
                uuid unique id for this instance
    worker_info : dict
                  Maps worker addresses to NCCL ranks & UCX ports
    dask_worker : dask_worker object
                  (Note: if called by client.run(), this is supplied by Dask
                   and not the client)
    """
    eps = [None] * len(worker_info)
    count = 1

    for k in worker_info:
        ip, port = parse_host_port(k)

        ep = await get_ucx(dask_worker=dask_worker).get_endpoint(
            ip, worker_info[k]["port"]
        )

        eps[worker_info[k]["rank"]] = ep
        count += 1

    raft_comm_state = get_raft_comm_state(
        sessionId=sessionId, state_object=dask_worker
    )
    raft_comm_state["ucp_eps"] = eps


async def _func_destroy_all(
    sessionId, comms_p2p, verbose=False, dask_worker=None
):
    if verbose:
        dask_worker.log_event(
            topic="info", msg="Destroying NCCL session state."
        )

    raft_comm_state = get_raft_comm_state(
        sessionId=sessionId, state_object=dask_worker
    )
    if "nccl" in raft_comm_state:
        raft_comm_state["nccl"].destroy()
        del raft_comm_state["nccl"]
        if verbose:
            dask_worker.log_event(
                topic="info", msg="NCCL session state destroyed."
            )
    else:
        if verbose:
            dask_worker.log_event(
                topic="warning",
                msg=f"Session state for, '{sessionId}', "
                f"does not contain expected 'nccl' element",
            )

    if verbose:
        dask_worker.log_event(
            topic="info",
            msg=f"Destroying CUDA handle for sessionId, '{sessionId}.'",
        )

    if "handle" in raft_comm_state:
        del raft_comm_state["handle"]
    else:
        if verbose:
            dask_worker.log_event(
                topic="warning",
                msg=f"Session state for, '{sessionId}', "
                f"does not contain expected 'handle' element",
            )


def _func_ucp_ports(client, workers):
    return client.run(_func_ucp_listener_port, workers=workers)


def _func_worker_ranks(client, workers):
    """
    For each worker connected to the client, compute a global rank which takes
    into account the NVML device index and the worker IP
    (group workers on same host and order by NVML device).
    Note that the reason for sorting was nvbug 4149999 and is presumably
    fixed afterNCCL 2.19.3.

    Parameters
    ----------
        client (object): Dask client object.
        workers (list): List of worker addresses.
    """
    # TODO: Add Test this function
    # Running into build issues preventing testing
    nvml_device_index_d = client.run(_get_nvml_device_index, workers=workers)
    # Sort workers first by IP and then by the nvml device index:
    worker_info_list = [
        (_get_worker_ip(worker), nvml_device_index, worker)
        for worker, nvml_device_index in nvml_device_index_d.items()
    ]
    worker_info_list.sort()
    return {wi[2]: i for i, wi in enumerate(worker_info_list)}


def _get_nvml_device_index():
    """
    Return NVML device index based on environment variable
    'CUDA_VISIBLE_DEVICES'.
    """
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
    return nvml_device_index(0, CUDA_VISIBLE_DEVICES)


def _get_worker_ip(worker_address):
    """
    Extract the worker IP address from the worker address string.

    Parameters
    ----------
        worker_address (str): Full address string of the worker
    """
    return ":".join(worker_address.split(":")[0:2])
