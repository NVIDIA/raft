#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.vector cimport vector

from rmm.librmm.cuda_stream_pool cimport cuda_stream_pool
from rmm.librmm.cuda_stream_ref cimport stream_ref


# Keeping `handle_t` around for backwards compatibility at the
# cython layer but users are encourage to switch to device_resources
cdef extern from "raft/core/handle.hpp" namespace "raft" nogil:
    cdef cppclass handle_t:
        handle_t() except +
        handle_t(stream_ref stream_view) except +
        handle_t(stream_ref stream_view,
                 shared_ptr[cuda_stream_pool] stream_pool) except +
        stream_ref get_stream() except +
        void sync_stream() except +


cdef extern from "raft/core/device_resources.hpp" namespace "raft" nogil:
    cdef cppclass device_resources:
        device_resources() except +
        device_resources(stream_ref stream_view) except +
        device_resources(stream_ref stream_view,
                         shared_ptr[cuda_stream_pool] stream_pool) except +
        stream_ref get_stream() except +
        void sync_stream() except +

cdef class DeviceResources:
    cdef unique_ptr[device_resources] c_obj
    cdef shared_ptr[cuda_stream_pool] stream_pool
    cdef int n_streams

cdef extern from "raft/core/device_resources_snmg.hpp" namespace "raft":
    cdef cppclass device_resources_snmg(device_resources):
        device_resources_snmg() except +
        device_resources_snmg(const vector[int]& device_ids) except +
        device_resources_snmg(const device_resources_snmg&) except +

cdef class DeviceResourcesSNMG:
    cdef unique_ptr[device_resources_snmg] c_obj
    cdef object device_ids
