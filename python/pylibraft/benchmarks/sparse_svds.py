# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib.util
import math
import time
from pathlib import Path

import cupy as cp
import cupyx.scipy.sparse as cupy_sparse
import cupyx.scipy.sparse.linalg as cupy_sparse_linalg
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from anndata import read_h5ad

CSR_FILE_MAGIC = 0x3152534354464152


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_data_path() -> Path:
    root = _repo_root()
    candidates = [
        root.parent / "rapids-singlecell-notebooks/h5/pca.h5ad",
        root.parent / "rapids_singlecell/data/pbmc3k_raw.h5ad",
        root.parent / "scanpy/src/scanpy/datasets/10x_pbmc68k_reduced.h5ad",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _import_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_rsc_modules(rsc_source: Path):
    sparse_pca = rsc_source / "src/rapids_singlecell/preprocessing/_sparse_pca"
    lanczos = _import_from_path(
        "rsc_svd_lanczos", sparse_pca / "_svd_lanczos.py"
    )
    randomized = _import_from_path(
        "rsc_block_lanczos", sparse_pca / "_block_lanczos.py"
    )
    return lanczos.lanczos_svd, randomized.randomized_svd


def _load_raft_svds():
    try:
        from pylibraft.sparse.linalg import svds
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not import pylibraft.sparse.linalg.svds: {exc}"
        ) from exc
    return svds


def _read_h5ad_csr(path: Path, *, layer: str | None, dtype: np.dtype):
    adata = read_h5ad(path)
    X = adata.layers[layer] if layer is not None else adata.X
    if scipy.sparse.issparse(X):
        X = X.tocsr()
    else:
        X = scipy.sparse.csr_matrix(X)
    X.sort_indices()
    if X.dtype != dtype:
        X = X.astype(dtype)
    return X


def _slice_matrix(X, *, max_rows: int | None, max_cols: int | None):
    if max_rows is not None:
        X = X[:max_rows, :]
    if max_cols is not None:
        X = X[:, :max_cols]
    X = X.tocsr()
    X.sort_indices()
    return X


def _to_cupy_csr(X):
    return cupy_sparse.csr_matrix(
        (cp.asarray(X.data), cp.asarray(X.indices), cp.asarray(X.indptr)),
        shape=X.shape,
    )


def _export_csr_bin(X, path: Path):
    if X.dtype != np.float32:
        raise TypeError(
            f"CSR binary export currently supports float32 values, got {X.dtype}"
        )
    dtype_code = 1

    int32_max = np.iinfo(np.int32).max
    if X.nnz > int32_max:
        raise OverflowError(f"nnz={X.nnz} exceeds int32 max")
    if X.shape[1] > int32_max:
        raise OverflowError(f"n_cols={X.shape[1]} exceeds int32 max")

    path.parent.mkdir(parents=True, exist_ok=True)
    header_dtype = np.dtype(
        [
            ("magic", "<u8"),
            ("version", "<u4"),
            ("dtype", "<u4"),
            ("rows", "<i8"),
            ("cols", "<i8"),
            ("nnz", "<i8"),
        ]
    )
    header = np.array(
        [(CSR_FILE_MAGIC, 1, dtype_code, X.shape[0], X.shape[1], X.nnz)],
        dtype=header_dtype,
    )
    with path.open("wb") as f:
        header.tofile(f)
        np.asarray(X.indptr, dtype=np.int32).tofile(f)
        np.asarray(X.indices, dtype=np.int32).tofile(f)
        np.asarray(X.data, dtype=X.dtype).tofile(f)


def _time_cuda(fn, *, warmups: int, repeats: int):
    for _ in range(warmups):
        result = fn()
        cp.cuda.Stream.null.synchronize()

    times_ms = []
    result = None
    for _ in range(repeats):
        start = cp.cuda.Event()
        stop = cp.cuda.Event()
        start.record()
        result = fn()
        stop.record()
        stop.synchronize()
        times_ms.append(cp.cuda.get_elapsed_time(start, stop))
    return result, times_ms


def _time_cpu(fn, *, warmups: int, repeats: int):
    for _ in range(warmups):
        fn()

    times_ms = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times_ms.append((time.perf_counter() - start) * 1000)
    return result, times_ms


def _orthogonality_error(Q, *, rows_are_vectors: bool):
    if Q is None:
        return math.nan
    Q_metric = Q.astype(cp.float64, copy=False)
    k = Q.shape[0] if rows_are_vectors else Q.shape[1]
    gram = Q_metric @ Q_metric.T if rows_are_vectors else Q_metric.T @ Q_metric
    eye = cp.eye(k, dtype=Q_metric.dtype)
    return float(cp.linalg.norm(gram - eye).get())


def _residual_error(A, U, S, Vt):
    if U is None or Vt is None:
        return math.nan
    AV = A @ Vt.T
    US = U * S[cp.newaxis, :]
    return float((cp.linalg.norm(AV - US) / cp.maximum(S[0], 1e-30)).get())


def _orthogonality_error_cpu(Q, *, rows_are_vectors: bool):
    if Q is None:
        return math.nan
    Q_metric = Q.astype(np.float64, copy=False)
    k = Q.shape[0] if rows_are_vectors else Q.shape[1]
    gram = Q_metric @ Q_metric.T if rows_are_vectors else Q_metric.T @ Q_metric
    eye = np.eye(k, dtype=Q_metric.dtype)
    return float(np.linalg.norm(gram - eye))


def _residual_error_cpu(A, U, S, Vt):
    if U is None or Vt is None:
        return math.nan
    AV = A @ Vt.T
    US = U * S[np.newaxis, :]
    return float(np.linalg.norm(AV - US) / max(float(S[0]), 1e-30))


def _sort_svd_desc(U, S, Vt):
    order = cp.argsort(S)[::-1]
    U_sorted = U[:, order] if U is not None else None
    Vt_sorted = Vt[order, :] if Vt is not None else None
    return U_sorted, S[order], Vt_sorted


def _sort_svd_desc_cpu(U, S, Vt):
    order = np.argsort(S)[::-1]
    U_sorted = U[:, order] if U is not None else None
    Vt_sorted = Vt[order, :] if Vt is not None else None
    return U_sorted, S[order], Vt_sorted


def _cupy_svds(A, args, ncv):
    cp.random.seed(args.seed)
    U, S, Vt = cupy_sparse_linalg.svds(
        A,
        k=args.k,
        ncv=ncv,
        tol=args.tol,
        maxiter=args.maxiter,
        return_singular_vectors=True,
    )
    return _sort_svd_desc(U, S, Vt)


def _scipy_svds(A, args, *, solver: str):
    U, S, Vt = scipy.sparse.linalg.svds(
        A,
        k=args.k,
        tol=args.tol,
        maxiter=args.maxiter,
        return_singular_vectors=True,
        solver=solver,
        random_state=args.seed,
    )
    return _sort_svd_desc_cpu(U, S, Vt)


def _run_solver(name, fn, A, args, *, compute_metrics: bool):
    def call():
        return fn()

    (U, S, Vt), times_ms = _time_cuda(
        call, warmups=args.warmups, repeats=args.repeats
    )
    cp.cuda.Stream.null.synchronize()
    s0 = float(S[0].get())
    s_head = cp.asnumpy(S[: min(5, S.shape[0])])
    row = {
        "solver": name,
        "median_ms": float(np.median(times_ms)),
        "mean_ms": float(np.mean(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "s0": s0,
        "s_head": s_head,
        "u_orth": math.nan,
        "vt_orth": math.nan,
        "residual": math.nan,
    }
    if compute_metrics:
        row["u_orth"] = _orthogonality_error(U, rows_are_vectors=False)
        row["vt_orth"] = _orthogonality_error(Vt, rows_are_vectors=True)
        row["residual"] = _residual_error(A, U, S, Vt)
    return row


def _run_solver_cpu(name, fn, A, args, *, compute_metrics: bool):
    (U, S, Vt), times_ms = _time_cpu(
        fn, warmups=args.warmups, repeats=args.repeats
    )
    s0 = float(S[0])
    s_head = np.asarray(S[: min(5, S.shape[0])])
    row = {
        "solver": name,
        "median_ms": float(np.median(times_ms)),
        "mean_ms": float(np.mean(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "s0": s0,
        "s_head": s_head,
        "u_orth": math.nan,
        "vt_orth": math.nan,
        "residual": math.nan,
    }
    if compute_metrics:
        row["u_orth"] = _orthogonality_error_cpu(U, rows_are_vectors=False)
        row["vt_orth"] = _orthogonality_error_cpu(Vt, rows_are_vectors=True)
        row["residual"] = _residual_error_cpu(A, U, S, Vt)
    return row


def _format_row(row):
    s_head = np.array2string(row["s_head"], precision=4, separator=", ")
    return (
        f"{row['solver']:<22} median={row['median_ms']:>9.3f} ms "
        f"mean={row['mean_ms']:>9.3f} ms min={row['min_ms']:>9.3f} ms "
        f"s0={row['s0']:.6g} "
        f"u_orth={row['u_orth']:.3e} vt_orth={row['vt_orth']:.3e} "
        f"residual={row['residual']:.3e} s[:5]={s_head}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark sparse SVD solvers on CSR/H5AD inputs."
    )
    parser.add_argument("--data", type=Path, default=_default_data_path())
    parser.add_argument("--layer", default=None)
    parser.add_argument(
        "--rsc-source",
        type=Path,
        default=_repo_root().parent / "rapids_singlecell",
    )
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--ncv", type=int, default=None)
    parser.add_argument("--n-oversamples", type=int, default=10)
    parser.add_argument("--n-power-iters", type=int, default=2)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default="float32"
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-cols", type=int, default=None)
    parser.add_argument("--export-csr-bin", type=Path, default=None)
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument(
        "--solvers",
        default="raft_lanczos,raft_randomized",
        help="Comma-separated subset of raft_lanczos, raft_lanczos_mgs2, "
        "raft_randomized, rsc_lanczos, rsc_randomized, cupy_svds, "
        "scipy_propack, scipy_lobpcg.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = np.dtype(args.dtype)
    ncv = args.ncv
    selected = {
        name.strip() for name in args.solvers.split(",") if name.strip()
    }

    X = _read_h5ad_csr(args.data, layer=args.layer, dtype=dtype)
    X = _slice_matrix(X, max_rows=args.max_rows, max_cols=args.max_cols)
    if args.export_csr_bin is not None:
        _export_csr_bin(X, args.export_csr_bin)
        print(
            f"exported CSR binary to {args.export_csr_bin} "
            f"shape={X.shape} nnz={X.nnz} dtype={X.dtype}"
        )
        if args.export_only:
            return

    needs_gpu = any(
        name == "cupy_svds"
        or name.startswith("raft_")
        or name.startswith("rsc_")
        for name in selected
    )
    A = _to_cupy_csr(X) if needs_gpu else None

    print(
        f"data={args.data} shape={X.shape} nnz={X.nnz} density={X.nnz / (X.shape[0] * X.shape[1]):.6g} "
        f"dtype={X.dtype} k={args.k} ncv={ncv if ncv is not None else 'default'} repeats={args.repeats}"
    )

    needs_raft = any(name.startswith("raft_") for name in selected)
    needs_rsc = any(name.startswith("rsc_") for name in selected)

    raft_svds = _load_raft_svds() if needs_raft else None
    rsc_lanczos, rsc_randomized = (
        _load_rsc_modules(args.rsc_source) if needs_rsc else (None, None)
    )

    gpu_solvers = {}
    cpu_solvers = {
        "scipy_propack": lambda: _scipy_svds(X, args, solver="propack"),
        "scipy_lobpcg": lambda: _scipy_svds(X, args, solver="lobpcg"),
    }
    if A is not None:
        gpu_solvers["cupy_svds"] = lambda: _cupy_svds(A, args, ncv)
    if raft_svds is not None:
        gpu_solvers["raft_lanczos"] = lambda: raft_svds(
            A,
            k=args.k,
            solver="lanczos",
            ncv=ncv,
            tol=args.tol,
            maxiter=args.maxiter,
            seed=args.seed,
            orthogonalization="cgs2",
        )
        gpu_solvers["raft_lanczos_mgs2"] = lambda: raft_svds(
            A,
            k=args.k,
            solver="lanczos",
            ncv=ncv,
            tol=args.tol,
            maxiter=args.maxiter,
            seed=args.seed,
            orthogonalization="mgs2",
        )
        gpu_solvers["raft_randomized"] = lambda: raft_svds(
            A,
            k=args.k,
            solver="randomized",
            n_oversamples=args.n_oversamples,
            n_power_iters=args.n_power_iters,
            seed=args.seed,
        )

    if rsc_lanczos is not None:
        gpu_solvers["rsc_lanczos"] = lambda: rsc_lanczos(
            A,
            k=args.k,
            ncv=ncv,
            tol=args.tol,
            max_iter=args.maxiter,
            random_state=args.seed,
            refine_results=True,
        )
        gpu_solvers["rsc_randomized"] = lambda: rsc_randomized(
            A,
            k=args.k,
            n_oversamples=args.n_oversamples,
            n_iter=args.n_power_iters,
            random_state=args.seed,
        )

    solvers = {**gpu_solvers, **cpu_solvers}
    unknown = selected.difference(solvers)
    if unknown:
        raise ValueError(f"Unknown or unavailable solvers: {sorted(unknown)}")

    solver_order = [
        "cupy_svds",
        "raft_lanczos",
        "raft_lanczos_mgs2",
        "raft_randomized",
        "rsc_lanczos",
        "rsc_randomized",
        "scipy_propack",
        "scipy_lobpcg",
    ]
    for name in [name for name in solver_order if name in selected]:
        try:
            if name in gpu_solvers:
                row = _run_solver(
                    name,
                    gpu_solvers[name],
                    A,
                    args,
                    compute_metrics=args.metrics,
                )
            else:
                row = _run_solver_cpu(
                    name,
                    cpu_solvers[name],
                    X,
                    args,
                    compute_metrics=args.metrics,
                )
            print(_format_row(row))
        except Exception as exc:  # noqa: BLE001
            print(f"{name:<22} failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
