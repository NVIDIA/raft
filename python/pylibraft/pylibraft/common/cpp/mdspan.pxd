#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport int8_t, int64_t, uint8_t, uint64_t
from libcpp.string cimport string

from pylibraft.common.handle cimport device_resources


cdef extern from "raft/core/mdspan_types.hpp" \
        namespace "raft":
    cdef cppclass row_major:
        pass
    cdef cppclass col_major:
        pass
    cdef cppclass matrix_extent[IndexType]:
        pass


cdef extern from "raft/core/device_mdspan.hpp" namespace "raft" nogil:

    cdef cppclass device_vector_view[ElementType, IndexType]:
        pass

    cdef cppclass device_scalar_view[ElementType, IndexType]:
        pass

    cdef cppclass device_matrix_view[ElementType, IndexType, LayoutType]:
        pass

    cdef device_matrix_view[ElementType, IndexType, LayoutPolicy] \
        make_device_matrix_view[ElementType, IndexType, LayoutPolicy](
            ElementType* ptr, IndexType n_rows, IndexType n_cols) except +

    cdef device_vector_view[ElementType, IndexType] \
        make_device_vector_view[ElementType, IndexType](
            ElementType* ptr, IndexType n) except +

    cdef device_scalar_view[ElementType, IndexType] \
        make_device_vector_view[ElementType, IndexType](
            ElementType* ptr) except +


cdef extern from "raft/core/host_mdspan.hpp" \
        namespace "raft" nogil:

    cdef cppclass host_matrix_view[ElementType, IndexType, LayoutPolicy]:
        pass

    cdef cppclass host_vector_view[ElementType, IndexType]:
        pass

    cdef cppclass host_scalar_view[ElementType, IndexType]:
        pass

    cdef cppclass host_mdspan[ElementType, Extents, LayoutPolicy]:
        pass

    cdef host_matrix_view[ElementType, IndexType, LayoutPolicy] \
        make_host_matrix_view[ElementType, IndexType, LayoutPolicy](
            ElementType* ptr, IndexType n_rows, IndexType n_cols) except +

    cdef host_vector_view[ElementType, IndexType] \
        make_host_vector_view[ElementType, IndexType](
            ElementType* ptr, IndexType n) except +

    cdef host_scalar_view[ElementType, IndexType] \
        make_host_scalar_view[ElementType, IndexType](
            ElementType *ptr) except +

cdef extern from "<sstream>" namespace "std" nogil:
    cdef cppclass ostringstream:
        ostringstream() except +
        string str() except +


cdef extern from "<ostream>" namespace "std" nogil:

    cdef cppclass ostream:
        pass

cdef extern from "raft/core/mdspan.hpp" namespace "raft" nogil:
    cdef cppclass dextents[IndexType, Rank]:
        pass

cdef extern from "raft/core/serialize.hpp" namespace "raft" nogil:

    cdef void serialize_mdspan[ElementType, Extents, LayoutPolicy](
        const device_resources& handle, ostream& os,
        const host_mdspan[ElementType, Extents, LayoutPolicy]& obj)
