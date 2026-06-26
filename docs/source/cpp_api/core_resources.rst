Resources
=========

.. role:: py(code)
   :language: c++
   :class: highlight

All resources which are specific to a computing environment like host or device are contained within, and managed by,
`raft::resources`. This design simplifies the APIs and eases user burden by making the APIs opaque by default but allowing customization based on user preference.


Vocabulary
----------

``#include <raft/core/resource/resource_types.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_types
     :project: RAFT
     :members:
     :content-only:


Device Resources
----------------

`raft::device_resources` is a convenience over using `raft::resources` directly. It provides accessor methods to retrieve resources such as the CUDA stream, stream pool, and handles to the various CUDA math libraries like cuBLAS and cuSOLVER.

``#include <raft/core/device_resources.hpp>``

namespace *raft::core*

.. doxygenclass:: raft::device_resources
    :project: RAFT
    :members:

Device Resources Manager
------------------------

While `raft::device_resources` provides a convenient way to access
device-related resources for a sequence of RAFT calls, it is sometimes useful
to be able to limit those resources across an entire application. For
instance, in highly multi-threaded applications, it can be helpful to limit
the total number of streams rather than relying on the default stream per
thread. `raft::device_resources_manager` offers a way to access
`raft::device_resources` instances that draw from a limited pool of
underlying device resources.

``#include <raft/core/device_resources_manager.hpp>``

namespace *raft::core*

.. doxygenclass:: raft::device_resources_manager
    :project: RAFT
    :members:

SNMG Device Resources
---------------------

The `raft::device_resources_snmg` provides a convenient way to configure
a SNMG (single-node multi-GPU) clique for MG algorithms. It initiates
device-related resources for a set of devices. Calling nccl-related functions
in `raft/core/resource/nccl_comm.hpp` will initialize nccl comm on each device resource.
GPUs can be addressed and exchanges be made over multiple threads for performance.

``#include <raft/core/device_resources_snmg.hpp>``

namespace *raft::core*

.. doxygenclass:: raft::device_resources_snmg
    :project: RAFT
    :members:

Resource Functions
------------------

Comms
~~~~~

``#include <raft/core/resource/comms.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_comms
     :project: RAFT
     :members:
     :content-only:

cuBLAS Handle
~~~~~~~~~~~~~

``#include <raft/core/resource/cublas_handle.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cublas
     :project: RAFT
     :members:
     :content-only:

cuBLASLt Handle
~~~~~~~~~~~~~~~

``#include <raft/core/resource/cublaslt_handle.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cublaslt
     :project: RAFT
     :members:
     :content-only:

CUDA Stream
~~~~~~~~~~~

``#include <raft/core/resource/cuda_stream.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cuda_stream
     :project: RAFT
     :members:
     :content-only:


CUDA Stream Pool
~~~~~~~~~~~~~~~~

``#include <raft/core/resource/cuda_stream_pool.hpp>``

namespace *raft::resource*

.. doxygengroup:: resource_stream_pool
    :project: RAFT
    :members:
    :content-only:

cuSolverDn Handle
~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/cusolver_dn_handle.hpp>``
namespace *raft::resource*

 .. doxygengroup:: resource_cusolver_dn
     :project: RAFT
     :members:
     :content-only:

cuSolverSp Handle
~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/cusolver_sp_handle.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cusolver_sp
     :project: RAFT
     :members:
     :content-only:

cuSparse Handle
~~~~~~~~~~~~~~~

``#include <raft/core/resource/cusparse_handle.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_cusparse
     :project: RAFT
     :members:
     :content-only:

Device ID
~~~~~~~~~

``#include <raft/core/resource/device_id.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_device_id
     :project: RAFT
     :members:
     :content-only:


Device Memory Resource
~~~~~~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/device_memory_resource.hpp>``

namespace *raft::resource*

 .. doxygengroup:: device_memory_resource
     :project: RAFT
     :members:
     :content-only:

Memory Statistics Resources
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``#include <raft/core/memory_stats_resources.hpp>``

namespace *raft*

.. doxygenstruct:: raft::memory_stats
    :project: RAFT
    :members:

.. doxygenclass:: raft::memory_stats_resources
    :project: RAFT
    :members:

Memory Tracking Resources
~~~~~~~~~~~~~~~~~~~~~~~~~

``#include <raft/core/memory_tracking_resources.hpp>``

namespace *raft*

.. doxygenclass:: raft::memory_tracking_resources
    :project: RAFT
    :members:

Dry-Run Resources
~~~~~~~~~~~~~~~~~

``#include <raft/core/dry_run_resources.hpp>``

namespace *raft*

.. doxygenclass:: raft::dry_run_resources
    :project: RAFT
    :members:

namespace *raft::util*

.. doxygenfunction:: raft::util::dry_run_execute
    :project: RAFT

Device Properties
~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/device_properties.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_device_props
     :project: RAFT
     :members:
     :content-only:

Sub Communicators
~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/sub_comms.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_sub_comms
     :project: RAFT
     :members:
     :content-only:

Thrust Exec Policy
~~~~~~~~~~~~~~~~~~

``#include <raft/core/resource/thrust_policy.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_thrust_policy
     :project: RAFT
     :members:
     :content-only:

Custom runtime-shared resources
~~~~~~~~~~~~~~~~~~~~~~

A custom resource is an arbitrary default-constructible C++ class.
The consumer of the API can keep such a resource in the `raft::resources` handle.
For example, consider a function that is expected to be called repeatedly and
involves a costly kernel configuration. One can cache the kernel configuration in
a custom resource.
The cost of accessing it is one hashmap lookup.

``#include <raft/core/resource/custom_resource.hpp>``

namespace *raft::resource*

 .. doxygengroup:: resource_custom
     :project: RAFT
     :members:
     :content-only:
