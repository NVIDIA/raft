# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy
import cupyx.scipy.sparse.linalg  # NOQA
import numpy
import pytest
import scipy
from cupyx.scipy import sparse

from pylibraft.sparse.linalg import eigsh, svds


def shaped_random(
    shape, xp=cupy, dtype=numpy.float32, scale=10, seed=0, order="C"
):
    """
    Returns an array filled with random values.

    Args
    ----
        shape(tuple): Shape of returned ndarray.
        xp(numpy or cupy): Array module to use.
        dtype(dtype): Dtype of returned ndarray.
        scale(float): Scaling factor of elements.
        seed(int): Random seed.

    Returns
    -------
        numpy.ndarray or cupy.ndarray: The array with
        given shape, array module,

    If ``dtype`` is ``numpy.bool_``, the elements are
    independently drawn from ``True`` and ``False``
    with same probabilities.
    Otherwise, the array is filled with samples
    independently and identically drawn
    from uniform distribution over :math:`[0, scale)`
    with specified dtype.
    """
    numpy.random.seed(seed)
    dtype = numpy.dtype(dtype)
    if dtype == "?":
        a = numpy.random.randint(2, size=shape)
    elif dtype.kind == "c":
        a = numpy.random.rand(*shape) + 1j * numpy.random.rand(*shape)
        a *= scale
    else:
        a = numpy.random.rand(*shape) * scale
    return xp.asarray(a, dtype=dtype, order=order)


class TestEigsh:
    n = 30
    density = 0.33
    tol = {numpy.float32: 1e-5, numpy.complex64: 1e-5, "default": 1e-12}
    res_tol = {"f": 1e-5, "d": 1e-12}
    return_eigenvectors = True

    def _make_matrix(self, dtype, xp):
        shape = (self.n, self.n)
        a = shaped_random(shape, xp, dtype=dtype)
        mask = shaped_random(shape, xp, dtype="f", scale=1)
        a[mask > self.density] = 0
        a = a * a.conj().T
        return a

    def _test_eigsh(self, a, k, xp, sp, which):
        scipy_csr = sp.sparse.csr_matrix(
            (a.data.get(), a.indices.get(), a.indptr.get()), shape=a.shape
        )
        expected_ret = sp.sparse.linalg.eigsh(
            scipy_csr,
            k=k,
            return_eigenvectors=self.return_eigenvectors,
            which=which,
        )

        actual_ret = eigsh(
            a,
            k=k,
            which=which,
        )
        if self.return_eigenvectors:
            w, x = actual_ret
            exp_w, _ = expected_ret
            # Check the residuals to see if eigenvectors are correct.
            ax_xw = a @ x - xp.multiply(x, w.reshape(1, k))
            res = xp.linalg.norm(ax_xw) / xp.linalg.norm(w)
            tol = self.res_tol[numpy.dtype(a.dtype).char.lower()]
            assert res < tol
        else:
            w = actual_ret
            exp_w = expected_ret
        w = xp.sort(w)
        assert cupy.allclose(w, exp_w, rtol=tol, atol=tol)

    @pytest.mark.parametrize("format", ["csr"])  # , 'csc', 'coo'])
    @pytest.mark.parametrize("k", [3, 6, 12])
    @pytest.mark.parametrize("dtype", ["f", "d"])
    @pytest.mark.parametrize("which", ["LA", "LM", "SA"])
    def test_sparse(self, format, k, dtype, which, xp=cupy, sp=sparse):
        if format == "csc":
            pytest.xfail("may be buggy")  # trans=True

        a = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat(format)
        return self._test_eigsh(a, k, xp, scipy, which)

    def test_invalid(self):
        xp, sp = cupy, sparse
        a = xp.diag(xp.ones((self.n,), dtype="f"))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(xp.ones((2, 1), dtype="f"), which="SA")
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=0, which="SA")
        a = xp.diag(xp.ones((self.n,), dtype="f"))
        with pytest.raises(ValueError):
            sp.linalg.eigsh(xp.ones((1,), dtype="f"), which="SA")
        with pytest.raises(TypeError):
            sp.linalg.eigsh(xp.ones((2, 2), dtype="i"), which="SA")
        with pytest.raises(ValueError):
            sp.linalg.eigsh(a, k=self.n, which="SA")

    def test_starting_vector(self):
        # Make symmetric matrix
        aux = self._make_matrix("f", cupy)
        aux = sparse.coo_matrix(aux).asformat("csr")
        matrix = (aux + aux.T) / 2.0

        # Find reference eigenvector
        ew, ev = eigsh(matrix, k=1)
        v = ev[:, 0]

        # Obtain non-converged eigenvector from random initial guess.
        ew_aux, ev_aux = eigsh(matrix, k=1, ncv=1, maxiter=0)
        v_aux = cupy.copysign(ev_aux[:, 0], v)

        # Obtain eigenvector using known eigenvector as initial guess.
        ew_v0, ev_v0 = eigsh(matrix, k=1, v0=v.copy(), ncv=1, maxiter=0)
        v_v0 = cupy.copysign(ev_v0[:, 0], v)

        assert cupy.linalg.norm(v - v_v0) < cupy.linalg.norm(v - v_aux)

    @pytest.mark.parametrize("dtype", ["f", "d"])
    @pytest.mark.parametrize("seed", [42, 123, 0])
    def test_reproducibility(self, dtype, seed):
        """Test that eigsh produces reproducible results when given a seed."""
        xp, sp = cupy, sparse

        # Make symmetric matrix
        a = self._make_matrix(dtype, xp)
        a = sp.coo_matrix(a).asformat("csr")

        # Run eigsh twice with the same seed
        w1, v1 = eigsh(a, k=3, which="SA", seed=seed)
        w2, v2 = eigsh(a, k=3, which="SA", seed=seed)

        # Results should be identical
        assert cupy.allclose(w1, w2), "Eigenvalues differ with same seed"
        assert cupy.allclose(v1, v2), "Eigenvectors differ with same seed"


class TestSvds:
    """Tests for sparse randomized SVD."""

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    @pytest.mark.parametrize("k", [3, 10])
    def test_singular_values_positive_descending(self, dtype, k):
        """Singular values must be positive and in descending order."""
        m, n = 100, 60
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=dtype)
        U, S, Vt = svds(A, k=k, seed=42)
        S_host = cupy.asnumpy(S)
        assert all(S_host > 0), "Singular values must be positive"
        assert all(
            S_host[i] >= S_host[i + 1] for i in range(len(S_host) - 1)
        ), "Singular values must be descending"

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    @pytest.mark.parametrize("k", [3, 10])
    def test_orthogonality(self, dtype, k):
        """U columns and Vt rows must be orthonormal."""
        m, n = 100, 60
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=dtype)
        U, S, Vt = svds(A, k=k, seed=42)

        tol = 1e-4 if dtype == numpy.float32 else 1e-8
        I_k = cupy.eye(k, dtype=dtype)

        UtU = U.T @ U
        assert cupy.allclose(UtU, I_k, atol=tol), "U is not orthogonal"

        VtVtT = Vt @ Vt.T
        assert cupy.allclose(VtVtT, I_k, atol=tol), "Vt rows not orthonormal"

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_reconstruction_error(self, dtype):
        """||A - U @ diag(S) @ Vt|| must be bounded by the (k+1)-th singular value."""
        m, n = 80, 50
        k = 5
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.4, format="csr", dtype=dtype)
        U, S, Vt = svds(A, k=k, n_power_iters=4, seed=42)

        A_dense = A.toarray()
        recon = (U * S[None, :]) @ Vt
        err = cupy.linalg.norm(A_dense - recon)
        norm_A = cupy.linalg.norm(A_dense)

        # Relative error should be reasonable (not reconstructing full matrix)
        assert err / norm_A < 1.0, (
            f"Reconstruction error too large: {float(err / norm_A):.4f}"
        )

    def test_shapes(self):
        """Output shapes must match (m,k), (k,), (k,n)."""
        m, n, k = 100, 60, 5
        A = sparse.random(m, n, density=0.3, format="csr", dtype=numpy.float32)
        U, S, Vt = svds(A, k=k, seed=42)
        assert U.shape == (m, k)
        assert S.shape == (k,)
        assert Vt.shape == (k, n)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    @pytest.mark.parametrize("orthogonalization", ["cgs2", "mgs2"])
    def test_lanczos_diagonal_spectrum(self, dtype, orthogonalization):
        """Lanczos SVD must recover the leading singular triplets."""
        diag = cupy.asarray([8, 7, 6, 5, 4, 3, 2, 1], dtype=dtype)
        A = sparse.diags(diag, offsets=0, shape=(12, 8), format="csr")
        U, S, Vt = svds(
            A,
            k=3,
            solver="lanczos",
            ncv=6,
            tol=1e-6,
            maxiter=20,
            seed=42,
            orthogonalization=orthogonalization,
        )

        tol = 2e-3 if dtype == numpy.float32 else 1e-8
        assert cupy.allclose(S, diag[:3], atol=tol)
        assert cupy.allclose(U.T @ U, cupy.eye(3, dtype=dtype), atol=tol)
        assert cupy.allclose(Vt @ Vt.T, cupy.eye(3, dtype=dtype), atol=tol)

    def test_lanczos_return_singular_vectors_modes(self):
        """Lanczos supports scipy-style vector return modes."""
        diag = cupy.asarray([8, 7, 6, 5, 4, 3, 2, 1], dtype=numpy.float32)
        A = sparse.diags(diag, offsets=0, shape=(12, 8), format="csr")

        U_ref, S_ref, Vt_ref = svds(A, k=3, solver="lanczos", ncv=6, seed=42)

        U_f, S_f, Vt_f = svds(
            A,
            k=3,
            solver="lanczos",
            ncv=6,
            seed=42,
            return_singular_vectors=False,
        )
        assert U_f is None
        assert Vt_f is None
        assert cupy.allclose(S_f, S_ref)

        U_u, S_u, Vt_u = svds(
            A,
            k=3,
            solver="lanczos",
            ncv=6,
            seed=42,
            return_singular_vectors="u",
        )
        assert U_u.shape == U_ref.shape
        assert Vt_u is None
        assert cupy.allclose(S_u, S_ref)

        U_v, S_v, Vt_v = svds(
            A,
            k=3,
            solver="lanczos",
            ncv=6,
            seed=42,
            return_singular_vectors="vh",
        )
        assert U_v is None
        assert Vt_v.shape == Vt_ref.shape
        assert cupy.allclose(S_v, S_ref)

    def test_lanczos_rotated_clustered_spectrum(self):
        """Lanczos recovers clustered singular values for a non-diagonal sparse matrix."""
        m, n, k = 64, 48, 8
        expected = numpy.asarray(
            [10.0, 9.9995, 9.999, 9.5, 9.4995, 9.0, 8.5, 8.0],
            dtype=numpy.float32,
        )
        all_s = numpy.concatenate(
            [expected, numpy.asarray([0.75, 0.25], dtype=numpy.float32)]
        )
        indptr = [0]
        indices = []
        values = []
        for block in range(len(all_s) // 2):
            col0 = 2 * block
            s0, s1 = all_s[2 * block], all_s[2 * block + 1]
            theta = 0.17 + 0.11 * block
            phi = 0.31 + 0.07 * block
            cu, su = numpy.cos(theta), numpy.sin(theta)
            cv, sv = numpy.cos(phi), numpy.sin(phi)
            block_values = [
                cu * s0 * cv + su * s1 * sv,
                cu * s0 * sv - su * s1 * cv,
                su * s0 * cv - cu * s1 * sv,
                su * s0 * sv + cu * s1 * cv,
            ]
            indices.extend([col0, col0 + 1])
            values.extend(block_values[:2])
            indptr.append(len(values))
            indices.extend([col0, col0 + 1])
            values.extend(block_values[2:])
            indptr.append(len(values))
        indptr.extend([len(values)] * (m + 1 - len(indptr)))

        A = sparse.csr_matrix(
            (
                cupy.asarray(values, dtype=numpy.float32),
                cupy.asarray(indices, dtype=numpy.int32),
                cupy.asarray(indptr, dtype=numpy.int32),
            ),
            shape=(m, n),
        )

        U, S, Vt = svds(
            A,
            k=k,
            solver="lanczos",
            ncv=20,
            tol=1e-5,
            maxiter=120,
            seed=2027,
        )

        assert cupy.allclose(S, cupy.asarray(expected), atol=2e-3)
        eye = cupy.eye(k, dtype=numpy.float64)
        U64 = U.astype(cupy.float64, copy=False)
        Vt64 = Vt.astype(cupy.float64, copy=False)
        assert cupy.linalg.norm(U64.T @ U64 - eye) < 2e-3
        assert cupy.linalg.norm(Vt64 @ Vt64.T - eye) < 2e-3
        residual = cupy.linalg.norm(A @ Vt.T - U * S[None, :]) / S[0]
        assert residual < 1e-4

    @pytest.mark.parametrize("shape", [(96, 28), (24, 96)])
    def test_lanczos_rank_deficient_and_edge_shapes(self, shape):
        """Lanczos handles rank deficiency and tall/wide sparse inputs."""
        diag = cupy.zeros(min(shape), dtype=numpy.float32)
        diag[:8] = cupy.asarray([12, 8, 6, 3, 1, 0, 0, 0], dtype=diag.dtype)
        A = sparse.diags(diag, offsets=0, shape=shape, format="csr")

        U, S, Vt = svds(
            A,
            k=4,
            solver="lanczos",
            ncv=12,
            tol=1e-5,
            maxiter=80,
            seed=2026,
        )

        assert cupy.allclose(S, diag[:4], atol=2e-3)
        eye = cupy.eye(4, dtype=numpy.float64)
        U64 = U.astype(cupy.float64, copy=False)
        Vt64 = Vt.astype(cupy.float64, copy=False)
        assert cupy.linalg.norm(U64.T @ U64 - eye) < 2e-3
        assert cupy.linalg.norm(Vt64 @ Vt64.T - eye) < 2e-3
        residual = cupy.linalg.norm(A @ Vt.T - U * S[None, :]) / S[0]
        assert residual < 1e-4

    def test_lanczos_non_convergence_raises(self):
        """Lanczos reports non-convergence instead of returning forced Ritz vectors."""
        diag = cupy.zeros(40, dtype=numpy.float32)
        diag[:6] = cupy.asarray([10, 9, 8, 7, 6, 5], dtype=diag.dtype)
        A = sparse.diags(diag, offsets=0, shape=(64, 40), format="csr")

        with pytest.raises(RuntimeError, match="failed to converge"):
            svds(
                A,
                k=4,
                solver="lanczos",
                ncv=6,
                tol=0,
                maxiter=1,
                seed=2026,
            )

    def test_invalid_solver(self):
        """Unknown sparse SVD solvers must be rejected."""
        A = sparse.random(20, 15, density=0.3, format="csr")
        with pytest.raises(ValueError, match="solver"):
            svds(A, k=3, solver="bogus")
        with pytest.raises(ValueError, match="orthogonalization"):
            svds(A, k=3, solver="lanczos", orthogonalization="bogus")

    def test_int64_indices(self):
        """int64 indices should be auto-converted and work, with a warning."""
        m, n, k = 50, 30, 3
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=numpy.float32)

        # cupyx CSR's constructor normalizes index dtype back to int32, so
        # bypass it via field reassignment to actually hold int64 indices.
        A.indptr = A.indptr.astype(cupy.int64)
        A.indices = A.indices.astype(cupy.int64)
        assert A.indptr.dtype == numpy.int64

        with pytest.warns(UserWarning, match="int64"):
            U, S, Vt = svds(A, k=k, seed=42)
        assert S.shape == (k,)
        assert all(cupy.asnumpy(S) > 0)

    def test_invalid_k(self):
        """K out of range must raise ValueError."""
        A = sparse.random(20, 15, density=0.3, format="csr")
        with pytest.raises(ValueError):
            svds(A, k=0)
        with pytest.raises(ValueError):
            svds(A, k=15)

    def test_invalid_input_matrix(self):
        """Non-CSR / non-sparse / None inputs must be rejected with a clear error."""
        # None
        with pytest.raises(ValueError, match="'A' cannot be None"):
            svds(None, k=3)

        # Dense array (not sparse at all)
        dense = cupy.random.rand(20, 15).astype(numpy.float32)
        with pytest.raises(TypeError, match="cupyx.scipy.sparse"):
            svds(dense, k=3)

        # Sparse but wrong layout (COO / CSC)
        A_coo = sparse.random(
            20, 15, density=0.3, format="coo", dtype=numpy.float32
        )
        with pytest.raises(TypeError, match="csr_matrix"):
            svds(A_coo, k=3)

        A_csc = sparse.random(
            20, 15, density=0.3, format="csc", dtype=numpy.float32
        )
        with pytest.raises(TypeError, match="csr_matrix"):
            svds(A_csc, k=3)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_reproducibility(self, dtype):
        """Same seed must give same results."""
        m, n, k = 80, 50, 5
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=dtype)

        _, S1, _ = svds(A, k=k, seed=123)
        _, S2, _ = svds(A, k=k, seed=123)

        tol = 1e-5 if dtype == numpy.float32 else 1e-10
        assert cupy.allclose(S1, S2, atol=tol), (
            "Not reproducible with same seed"
        )

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_return_singular_vectors_modes(self, dtype):
        """All four scipy-style return_singular_vectors modes."""
        m, n, k = 60, 40, 4
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=dtype)

        # Reference: full output
        U_ref, S_ref, Vt_ref = svds(A, k=k, seed=42)
        assert U_ref.shape == (m, k)
        assert Vt_ref.shape == (k, n)

        # False: singular values only
        U_f, S_f, Vt_f = svds(A, k=k, seed=42, return_singular_vectors=False)
        assert U_f is None
        assert Vt_f is None
        assert cupy.allclose(S_f, S_ref)

        # "u": U + S
        U_u, S_u, Vt_u = svds(A, k=k, seed=42, return_singular_vectors="u")
        assert U_u.shape == (m, k)
        assert Vt_u is None
        assert cupy.allclose(S_u, S_ref)
        # U-based sign correction matches the full-output reference exactly
        tol = 1e-4 if dtype == numpy.float32 else 1e-10
        assert cupy.allclose(U_u, U_ref, atol=tol)

        # "vh": Vt + S
        U_v, S_v, Vt_v = svds(A, k=k, seed=42, return_singular_vectors="vh")
        assert U_v is None
        assert Vt_v.shape == (k, n)
        assert cupy.allclose(S_v, S_ref)
        # Vt rows should remain orthonormal even though signs may differ
        I_k = cupy.eye(k, dtype=dtype)
        assert cupy.allclose(Vt_v @ Vt_v.T, I_k, atol=tol)

    def test_return_singular_vectors_invalid(self):
        """Invalid return_singular_vectors must raise ValueError."""
        A = sparse.random(20, 15, density=0.3, format="csr")
        with pytest.raises(ValueError):
            svds(A, k=3, return_singular_vectors="bogus")
        with pytest.raises(ValueError):
            svds(A, k=3, return_singular_vectors=1)

    @pytest.mark.parametrize("dtype", [numpy.float32, numpy.float64])
    def test_return_singular_vectors_deterministic(self, dtype):
        """Each mode must be deterministic across runs with the same seed."""
        m, n, k = 50, 30, 3
        cupy.random.seed(42)
        A = sparse.random(m, n, density=0.3, format="csr", dtype=dtype)

        # cuSOLVER gesvd inside the algorithm is not bit-deterministic across
        # runs even with a fixed RNG seed, so allow a small numerical tolerance.
        atol = 1e-4 if dtype == numpy.float32 else 1e-10
        for mode in (False, "u", "vh"):
            res_a = svds(A, k=k, seed=7, return_singular_vectors=mode)
            res_b = svds(A, k=k, seed=7, return_singular_vectors=mode)
            assert cupy.allclose(res_a[1], res_b[1], atol=atol)
            if mode in (True, "u"):
                assert cupy.allclose(res_a[0], res_b[0], atol=atol)
            if mode in (True, "vh"):
                assert cupy.allclose(res_a[2], res_b[2], atol=atol)
