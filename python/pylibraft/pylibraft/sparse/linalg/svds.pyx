#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import warnings

import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uint32_t, uint64_t, uintptr_t
from libcpp cimport bool

from pylibraft.common import Handle, cai_wrapper, device_ndarray
from pylibraft.common.handle import auto_sync_handle

from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    device_vector_view,
    make_device_matrix_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.cpp.optional cimport optional
from pylibraft.common.handle cimport device_resources


cdef extern from "raft/sparse/solver/solver_types.hpp" \
        namespace "raft::sparse::solver" nogil:

    cdef cppclass sparse_svd_config[ValueTypeT]:
        int n_components
        int n_oversamples
        int n_power_iters
        optional[uint64_t] seed

    cdef cppclass sparse_lanczos_svd_config[ValueTypeT]:
        int n_components
        int ncv
        ValueTypeT tolerance
        int max_iterations
        optional[uint64_t] seed
        bool use_mgs2_orthogonalization


cdef extern from "raft_runtime/solver/randomized_svds.hpp" \
        namespace "raft::runtime::solver" nogil:

    cdef void sparse_randomized_svd_f \
            "raft::runtime::solver::sparse_randomized_svd"(
        const device_resources &handle,
        const sparse_svd_config[float] &config,
        device_vector_view[int, uint32_t] indptr,
        device_vector_view[int, uint32_t] indices,
        device_vector_view[float, uint32_t] data,
        int n_rows, int n_cols, int nnz,
        device_vector_view[float, uint32_t] singular_values,
        optional[device_matrix_view[float, uint32_t, col_major]] U,
        optional[device_matrix_view[float, uint32_t, col_major]] Vt) except +

    cdef void sparse_randomized_svd_d \
            "raft::runtime::solver::sparse_randomized_svd"(
        const device_resources &handle,
        const sparse_svd_config[double] &config,
        device_vector_view[int, uint32_t] indptr,
        device_vector_view[int, uint32_t] indices,
        device_vector_view[double, uint32_t] data,
        int n_rows, int n_cols, int nnz,
        device_vector_view[double, uint32_t] singular_values,
        optional[device_matrix_view[double, uint32_t, col_major]] U,
        optional[device_matrix_view[double, uint32_t, col_major]] Vt) except +


cdef extern from "raft_runtime/solver/lanczos_svds.hpp" \
        namespace "raft::runtime::solver" nogil:

    cdef void sparse_lanczos_svd_f \
            "raft::runtime::solver::sparse_lanczos_svd"(
        const device_resources &handle,
        const sparse_lanczos_svd_config[float] &config,
        device_vector_view[int, uint32_t] indptr,
        device_vector_view[int, uint32_t] indices,
        device_vector_view[float, uint32_t] data,
        int n_rows, int n_cols, int nnz,
        device_vector_view[float, uint32_t] singular_values,
        optional[device_matrix_view[float, uint32_t, col_major]] U,
        optional[device_matrix_view[float, uint32_t, col_major]] Vt) except +

    cdef void sparse_lanczos_svd_d \
            "raft::runtime::solver::sparse_lanczos_svd"(
        const device_resources &handle,
        const sparse_lanczos_svd_config[double] &config,
        device_vector_view[int, uint32_t] indptr,
        device_vector_view[int, uint32_t] indices,
        device_vector_view[double, uint32_t] data,
        int n_rows, int n_cols, int nnz,
        device_vector_view[double, uint32_t] singular_values,
        optional[device_matrix_view[double, uint32_t, col_major]] U,
        optional[device_matrix_view[double, uint32_t, col_major]] Vt) except +


@auto_sync_handle
def svds(A, k=6, n_oversamples=10, n_power_iters=2,
         seed=None, return_singular_vectors=True, handle=None,
         solver="randomized", ncv=None, tol=1e-4, maxiter=None,
         orthogonalization="cgs2"):
    """
    Compute the largest ``k`` singular values and corresponding singular
    vectors of a sparse matrix.

    Computes the truncated SVD: ``A ~ U @ diag(S) @ Vt``.

    Args:
        A (cupyx.scipy.sparse.csr_matrix): Sparse CSR matrix of shape
            ``(m, n)``. Must be of type
            :class:`cupyx.scipy.sparse.csr_matrix`.
        k (int): Number of singular values and vectors to compute. Must be
            ``1 <= k < min(m, n)``. Default 6.
        n_oversamples (int): Number of extra random vectors for better
            approximation. Total subspace dimension is ``k + n_oversamples``.
            Used only when ``solver="randomized"``. Default 10.
        n_power_iters (int): Number of power iteration passes. More
            iterations improve accuracy for matrices with slowly decaying
            singular values. Used only when ``solver="randomized"``.
            Default 2.
        seed (int or None): Random seed for reproducibility. If ``None``,
            a non-deterministic seed is used.
        return_singular_vectors (bool or {{"u", "vh"}}): Controls which
            singular vectors are returned (matches
            :func:`scipy.sparse.linalg.svds`).

            - ``True`` (default): return ``(U, S, Vt)``.
            - ``False``: skip both vector matrices, return ``(None, S, None)``.
            - ``"u"``: skip ``Vt``, return ``(U, S, None)``.
            - ``"vh"``: skip ``U``, return ``(None, S, Vt)``.

            Skipping a side avoids the corresponding output buffer. Depending
            on the selected solver, some intermediate vector products may
            still be required for numerical refinement.
        handle: RAFT resource handle. If ``None``, a default is created.
        solver ({{ "randomized", "lanczos" }}): Solver to use. The default
            ``"randomized"`` preserves the existing approximate randomized SVD
            behavior. ``"lanczos"`` uses Lanczos bidiagonalization and is the
            higher-accuracy, ARPACK-like sparse SVD path.
        ncv (int or None): Number of Lanczos vectors. Used only when
            ``solver="lanczos"``. If ``None``, a default is selected. Larger
            values can improve convergence margin and orthogonality for
            difficult spectra, but increase sparse matrix-vector work.
        tol (float): Lanczos residual tolerance. Used only when
            ``solver="lanczos"``.
        maxiter (int or None): Maximum Lanczos restart iterations. Used only
            when ``solver="lanczos"``. If ``None``, defaults to 100. The
            Lanczos solver raises an error if all requested singular triplets
            do not converge within this limit.
        orthogonalization ({{ "cgs2", "mgs2" }}): Lanczos reorthogonalization
            method. ``"cgs2"`` is the GPU-efficient default. ``"mgs2"`` is
            more launch-heavy and intended as an alternate path for difficult
            float32 spectra.

    Returns:
        tuple:
            ``(U, S, Vt)`` where ``U`` is left singular vectors ``(m, k)``,
            ``S`` is singular values ``(k,)`` in descending order, and
            ``Vt`` is right singular vectors ``(k, n)``. ``U`` and/or ``Vt``
            may be ``None`` depending on ``return_singular_vectors``.

    .. seealso::
        :func:`scipy.sparse.linalg.svds`

    .. note::
        With ``solver="randomized"``, this function uses randomized SVD
        (Halko et al. 2009) with CholeskyQR2 orthogonalization (Tomas et al.
        2024). With ``solver="lanczos"``, it uses implicitly restarted
        Lanczos bidiagonalization with full reorthogonalization and final
        ``A @ V`` post-refinement of the returned left singular vectors.

    """

    if A is None:
        raise ValueError("'A' cannot be None!")

    if not cupy_sparse.issparse(A):
        raise TypeError(
            "Expected a cupyx.scipy.sparse matrix, got %s" % type(A).__name__
        )

    if not isinstance(A, cupy_sparse.csr_matrix):
        raise TypeError(
            "Expected a cupyx.scipy.sparse.csr_matrix, got %s. "
            "Use A.tocsr() to convert." % type(A).__name__
        )

    if solver not in ("randomized", "lanczos"):
        raise ValueError(
            "solver must be 'randomized' or 'lanczos', got %r" % (solver,)
        )

    if orthogonalization not in ("cgs2", "mgs2"):
        raise ValueError(
            "orthogonalization must be 'cgs2' or 'mgs2', got %r"
            % (orthogonalization,)
        )

    if not (
        return_singular_vectors is True
        or return_singular_vectors is False
        or return_singular_vectors == "u"
        or return_singular_vectors == "vh"
    ):
        raise ValueError(
            "return_singular_vectors must be True, False, 'u', or 'vh', "
            "got %r" % (return_singular_vectors,)
        )
    want_U = (return_singular_vectors is True) or (return_singular_vectors == "u")
    want_Vt = (return_singular_vectors is True) or (return_singular_vectors == "vh")

    m, n = A.shape

    if k < 1 or k >= min(m, n):
        raise ValueError(
            f"k must satisfy 1 <= k < min(m, n), got k={k}, m={m}, n={n}"
        )

    # Extract CSR arrays and ensure int32 indices. raft's runtime layer takes
    # `int` (int32) for nnz / indptr / indices, so overflow on conversion
    # cannot be recovered downstream; error out before the cast.
    # The runtime API is currently int32-only. Native int64 support should
    # mirror the sparse Lanczos eigensolver overloads when it is added.
    indptr = A.indptr
    indices = A.indices
    data = A.data

    INT32_MAX = (1 << 31) - 1

    # Validate indptr and indices dtypes independently. CuPy CSR matrices
    # normally have matching dtypes, but don't assume it.
    for name, arr in (("indptr", indptr), ("indices", indices)):
        if arr.dtype != np.int32 and arr.dtype != np.int64:
            raise TypeError(
                "dtype %s.dtype=%s not supported, expected int32 or int64"
                % (name, arr.dtype)
            )

    if indptr.dtype == np.int64 or indices.dtype == np.int64:
        if A.nnz > INT32_MAX:
            raise OverflowError(
                f"nnz={A.nnz} exceeds int32 max ({INT32_MAX}); "
                "raft sparse SVD requires nnz to fit in int32."
            )
        if n > INT32_MAX:
            raise OverflowError(
                f"n_cols={n} exceeds int32 max ({INT32_MAX}); "
                "raft sparse SVD requires column indices to fit in int32."
            )
        warnings.warn(
            "Input matrix has int64 indices which will be converted to "
            "int32. The conversion is safe for this matrix (nnz and "
            "column indices fit in int32).",
            UserWarning,
            stacklevel=2,
        )
        if indptr.dtype == np.int64:
            indptr = indptr.astype(np.int32)
        if indices.dtype == np.int64:
            indices = indices.astype(np.int32)

    indptr = cai_wrapper(indptr)
    indices = cai_wrapper(indices)
    data = cai_wrapper(data)

    ValueType = data.dtype
    nnz = A.nnz

    indptr_ptr = <uintptr_t>indptr.data
    indices_ptr = <uintptr_t>indices.data
    data_ptr = <uintptr_t>data.data

    cdef optional[uint64_t] seed_opt
    if seed is not None:
        seed_opt = <uint64_t>seed

    # Allocate outputs
    S_out = device_ndarray.empty((k,), dtype=ValueType, order='F')
    S_cai = cai_wrapper(S_out)
    S_ptr = <uintptr_t>S_cai.data

    U_out = None
    Vt_out = None
    cdef uintptr_t U_ptr = 0
    cdef uintptr_t Vt_ptr = 0
    if want_U:
        U_out = device_ndarray.empty((m, k), dtype=ValueType, order='F')
        U_ptr = <uintptr_t>cai_wrapper(U_out).data
    if want_Vt:
        Vt_out = device_ndarray.empty((k, n), dtype=ValueType, order='F')
        Vt_ptr = <uintptr_t>cai_wrapper(Vt_out).data

    handle = handle if handle is not None else Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    cdef sparse_svd_config[float] cfg_float
    cdef sparse_svd_config[double] cfg_double
    cdef sparse_lanczos_svd_config[float] lanczos_cfg_float
    cdef sparse_lanczos_svd_config[double] lanczos_cfg_double
    cdef optional[device_matrix_view[float, uint32_t, col_major]] U_opt_f
    cdef optional[device_matrix_view[float, uint32_t, col_major]] Vt_opt_f
    cdef optional[device_matrix_view[double, uint32_t, col_major]] U_opt_d
    cdef optional[device_matrix_view[double, uint32_t, col_major]] Vt_opt_d
    cdef int ncv_value = 0 if ncv is None else <int>ncv
    cdef int maxiter_value = 100 if maxiter is None else <int>maxiter
    cdef bool use_mgs2_value = orthogonalization == "mgs2"

    if ValueType == np.float32:
        if want_U:
            U_opt_f = make_device_matrix_view[float, uint32_t, col_major](
                <float *>U_ptr, <uint32_t>m, <uint32_t>k)
        if want_Vt:
            Vt_opt_f = make_device_matrix_view[float, uint32_t, col_major](
                <float *>Vt_ptr, <uint32_t>k, <uint32_t>n)
        if solver == "randomized":
            cfg_float.n_components = k
            cfg_float.n_oversamples = n_oversamples
            cfg_float.n_power_iters = n_power_iters
            cfg_float.seed = seed_opt
            sparse_randomized_svd_f(
                deref(h),
                cfg_float,
                make_device_vector_view(<int *>indptr_ptr, <uint32_t>(m + 1)),
                make_device_vector_view(<int *>indices_ptr, <uint32_t>nnz),
                make_device_vector_view(<float *>data_ptr, <uint32_t>nnz),
                m, n, nnz,
                make_device_vector_view(<float *>S_ptr, <uint32_t>k),
                U_opt_f,
                Vt_opt_f,
            )
        else:
            lanczos_cfg_float.n_components = k
            lanczos_cfg_float.ncv = ncv_value
            lanczos_cfg_float.tolerance = <float>tol
            lanczos_cfg_float.max_iterations = maxiter_value
            lanczos_cfg_float.seed = seed_opt
            lanczos_cfg_float.use_mgs2_orthogonalization = use_mgs2_value
            sparse_lanczos_svd_f(
                deref(h),
                lanczos_cfg_float,
                make_device_vector_view(<int *>indptr_ptr, <uint32_t>(m + 1)),
                make_device_vector_view(<int *>indices_ptr, <uint32_t>nnz),
                make_device_vector_view(<float *>data_ptr, <uint32_t>nnz),
                m, n, nnz,
                make_device_vector_view(<float *>S_ptr, <uint32_t>k),
                U_opt_f,
                Vt_opt_f,
            )
    elif ValueType == np.float64:
        if want_U:
            U_opt_d = make_device_matrix_view[double, uint32_t, col_major](
                <double *>U_ptr, <uint32_t>m, <uint32_t>k)
        if want_Vt:
            Vt_opt_d = make_device_matrix_view[double, uint32_t, col_major](
                <double *>Vt_ptr, <uint32_t>k, <uint32_t>n)
        if solver == "randomized":
            cfg_double.n_components = k
            cfg_double.n_oversamples = n_oversamples
            cfg_double.n_power_iters = n_power_iters
            cfg_double.seed = seed_opt
            sparse_randomized_svd_d(
                deref(h),
                cfg_double,
                make_device_vector_view(<int *>indptr_ptr, <uint32_t>(m + 1)),
                make_device_vector_view(<int *>indices_ptr, <uint32_t>nnz),
                make_device_vector_view(<double *>data_ptr, <uint32_t>nnz),
                m, n, nnz,
                make_device_vector_view(<double *>S_ptr, <uint32_t>k),
                U_opt_d,
                Vt_opt_d,
            )
        else:
            lanczos_cfg_double.n_components = k
            lanczos_cfg_double.ncv = ncv_value
            lanczos_cfg_double.tolerance = <double>tol
            lanczos_cfg_double.max_iterations = maxiter_value
            lanczos_cfg_double.seed = seed_opt
            lanczos_cfg_double.use_mgs2_orthogonalization = use_mgs2_value
            sparse_lanczos_svd_d(
                deref(h),
                lanczos_cfg_double,
                make_device_vector_view(<int *>indptr_ptr, <uint32_t>(m + 1)),
                make_device_vector_view(<int *>indices_ptr, <uint32_t>nnz),
                make_device_vector_view(<double *>data_ptr, <uint32_t>nnz),
                m, n, nnz,
                make_device_vector_view(<double *>S_ptr, <uint32_t>k),
                U_opt_d,
                Vt_opt_d,
            )
    else:
        raise TypeError("dtype ValueType=%s not supported, "
                        "expected float32 or float64" % ValueType)

    U_ret = cp.asarray(U_out) if U_out is not None else None
    Vt_ret = cp.asarray(Vt_out) if Vt_out is not None else None
    return (U_ret, cp.asarray(S_out), Vt_ret)
