Using RAFT Comms
================

RAFT provides a communications abstraction for writing distributed algorithms which can scale up to multiple GPUs and scale out to multiple nodes. The communications abstraction is largely based on MPI and NCCL, and allows the user to decouple the design of algorithms from the environments where the algorithms are executed, enabling “write-once deploy everywhere” semantics. Currently, the distributed algorithms in both cuGraph and cuML are being deployed in both MPI and Dask clusters while cuML’s distributed algorithms are also being deployed on GPUs in Apache Spark clusters. This is a powerful concept as distributed algorithms can be non-trivial to write and so maintainability is eased and bug fixes reach further by increasing reuse as much as possible.

While users of RAFT’s communications layer largely get MPI integration for free just by installing MPI and using `mpirun` to run their applications, the `raft-dask` Python package provides a mechanism for executing algorithms written using RAFT’s communications layer in a Dask cluster. It will help to walk through a small example of how one would build an algorithm with RAFT’s communications layer.

First, an instance of `raft::comms_t` is passed through the `raft::resources` instance and code is written to utilize collective and/or point-to-point communications as needed.

.. code-block:: cpp
   :caption: Example function written with the RAFT comms API

   #include <raft/core/comms.hpp>
   #include <raft/core/device_mdspan.hpp>
   #include <raft/util/cudart_utils.hpp>

   void test_allreduce(raft::resources const &handle, int root) {
     raft::comms::comms_t const& communicator = resource::get_comms(handle);
     cudaStream_t stream = resource::get_cuda_stream(handle);
     raft::device_scalar<int> temp_scalar(stream);

     int to_send = 1;
     raft::copy(temp_scalar.data(), &to_send, 1, stream);
     communicator.allreduce(temp_scalar.data(), temp_scalar.data(), 1,
                            raft::comms::opt_t::SUM, stream);
     resource::sync_stream(handle);
   }

This exact function can now be executed in several different types of GPU clusters. For example, it can be executed with MPI by initializing an instance of `raft::comms::mpi_comms` with the `MPI_Comm`:

.. code-block:: cpp
   :caption: Example of running test_allreduce() in MPI

   #include <raft/core/mpi_comms.hpp>
   #include <raft/core/resources.hpp>

   raft::resources resource_handle;
   // ...
   // initialize MPI_Comm
   // ...
   raft::comms::initialize_mpi_comms(resource_handle,  mpi_comm);
   // ...
   test_allreduce(resource_handle, 0);

Deploying our`test_allreduce` function in Dask requires a lightweight Python interface, which we can accomplish using `pylibraft` and exposing the function through Cython:

.. code-block:: cython
   :caption: Example of wrapping test_allreduce() w/ cython

   from pylibraft.common.handle cimport device_resources
   from cython.operator cimport dereference as deref

   cdef extern from “allreduce_test.hpp”:
       void test_allreduce(device_resources const &handle, int root) except +

   def run_test_allreduce(handle, root):
       cdef const device_resources* h = <device_resources*><size_t>handle.getHandle()

   test_allreduce(deref(h), root)

Finally, we can use `raft_dask` to execute our new algorithm in a Dask cluster (please note this also uses `LocalCUDACluster` from the NVIDIA Dask-CUDA library):

.. code-block:: python
   :caption: Example of running test_allreduce() in Dask

   from raft_dask.common import Comms, local_handle
   from dask.distributed import Client, wait
   from dask_cuda import LocalCUDACluster
   cluster = LocalCUDACluster()
   client = Client(cluster)

   # Create and initialize Comms instance
   comms = Comms(client=client)
   comms.init()

   def func_run_allreduce(sessionId, root):
     handle = local_handle(sessionId)
     run_test_allreduce(handle, root)

   # Invoke run_test_allreduce on all workers
   dfs = [
     client.submit(
       func_run_allreduce,
       comms.sessionId,
       0,
       pure=False,
       workers=[w]
     )
     for w in comms.worker_addresses
   ]

   # Wait until processing is done
   wait(dfs, timeout=5)

   comms.destroy()
   client.close()
   cluster.close()
