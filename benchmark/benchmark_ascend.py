#!/usr/bin/env python3
"""Ascend-only FlagSparse versus PyTorch-NPU benchmark.

This entry point deliberately does not touch the CUDA/ROCm benchmark runners.
It uses SciPy on CPU only for correctness and synchronizes ``torch.npu`` around
the timed region.  ``ops-sparse`` is a C ``aclsparse`` library; unless a Python
bridge is supplied, PyTorch-NPU is reported as the fallback baseline.
"""

from __future__ import annotations

import argparse
import ctypes.util
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass
class Case:
    m: int
    n: int
    nnz: int
    dense_cols: int
    dtype_name: str


def _sync(torch):
    torch.npu.synchronize()


def _bench(torch, fn, warmup: int, iters: int):
    for _ in range(warmup):
        fn()
    _sync(torch)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        _sync(torch)
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "min_ms": float(samples[0]),
        "p95_ms": float(np.percentile(samples, 95)),
    }


def _make_csr(torch, case: Case, device):
    rng = np.random.default_rng(20260909 + case.m + case.n + case.nnz)
    rows = rng.integers(0, case.m, size=case.nnz, dtype=np.int64)
    cols = rng.integers(0, case.n, size=case.nnz, dtype=np.int64)
    vals = rng.standard_normal(case.nnz).astype(np.float32)
    matrix = sp.coo_matrix((vals, (rows, cols)), shape=(case.m, case.n)).tocsr()
    matrix.sum_duplicates()
    data = torch.tensor(matrix.data, device=device, dtype=torch.float32)
    indices = torch.tensor(matrix.indices, device=device, dtype=torch.int32)
    indptr = torch.tensor(matrix.indptr, device=device, dtype=torch.int32)
    return matrix, data, indices, indptr


def _scipy_spmv(matrix, x):
    return matrix @ x


def _scipy_spmm(matrix, b):
    return matrix @ b


def _scipy_sddmm(matrix, x, y):
    dense = x @ y.T
    # SDDMM samples ``x @ y.T`` at the CSR nonzero coordinates; it does not
    # multiply by the input CSR values (those only participate when beta != 0).
    rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr))
    return np.asarray(dense[rows, matrix.indices])


def _torch_npu_csr_row_ids(indptr, n_rows):
    """Build CSR row ids with operations supported by PyTorch-NPU 910B."""
    import torch

    counts = indptr[1:].to(torch.int64) - indptr[:-1].to(torch.int64)
    rows = torch.arange(n_rows, device=indptr.device, dtype=torch.int64)
    return torch.repeat_interleave(rows, counts)


def _torch_npu_csr_spmv(data, indices, row_ids, x, n_rows):
    """PyTorch-NPU CSR SpMV fallback without torch.sparse.mm."""
    import torch

    out = torch.zeros((n_rows,), device=data.device, dtype=data.dtype)
    out.index_add_(0, row_ids, data * x[indices.to(torch.int64)])
    return out


def _torch_npu_csr_spmm(data, indices, row_ids, B, n_rows):
    """PyTorch-NPU CSR SpMM fallback without SparseCSR addmm."""
    import torch

    out = torch.zeros((n_rows, B.shape[1]), device=data.device, dtype=data.dtype)
    values = data[:, None] * B[indices.to(torch.int64)]
    out.index_add_(0, row_ids, values)
    return out


def run(case: Case, warmup: int, iters: int, device_id: int = 0):
    import torch
    import flagsparse as fs

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Ascend benchmark requires torch_npu and an available NPU")
    torch.npu.set_device(int(device_id))
    device = torch.device(f"npu:{int(device_id)}")
    # ops-sparse is distributed as a C ``aclsparse`` library.  A Python import
    # alone is not sufficient to claim that it was timed, so report both probes
    # explicitly and keep PyTorch-NPU as the honest fallback when no bridge is
    # available.
    try:
        import ops_sparse  # type: ignore
        ops_status = "python_module_imported (not called by this runner)"
    except Exception as exc:
        ops_status = f"no_python_bridge: {exc}"
    acl_candidates = [
        os.environ.get("OPSSPARSE_HOME", ""),
        os.environ.get("OPS_SPARSE_HOME", ""),
    ]
    acl_lib = None
    for root in acl_candidates:
        if root:
            candidate = Path(root) / "lib" / "libaclsparse.so"
            if candidate.exists():
                acl_lib = str(candidate)
                break
    acl_lib = acl_lib or ctypes.util.find_library("aclsparse")
    if acl_lib:
        ops_status += f"; C library found: {acl_lib} (C bridge required)"
    else:
        ops_status += "; libaclsparse.so not found"
    matrix, data, indices, indptr = _make_csr(torch, case, device)
    x = torch.randn(case.n, device=device, dtype=torch.float32)
    b = torch.randn(case.n, case.dense_cols, device=device, dtype=torch.float32)
    sx = torch.randn(case.m, 32, device=device, dtype=torch.float32)
    sy = torch.randn(case.n, 32, device=device, dtype=torch.float32)
    row_ids = _torch_npu_csr_row_ids(indptr, case.m)

    rows = np.repeat(np.arange(case.m, dtype=np.int64), np.diff(matrix.indptr))
    scipy_spmv = _scipy_spmv(matrix, x.cpu().numpy())
    scipy_spmm = _scipy_spmm(matrix, b.cpu().numpy())
    scipy_sddmm = _scipy_sddmm(matrix, sx.cpu().numpy(), sy.cpu().numpy())

    results = [{
        "device": f"npu:{int(device_id)}",
        "ops_sparse": ops_status,
        "ops_sparse_910b_supported": ["spmv_csr", "sddmm_csr", "scatter"],
        "tested_ops": ["spmv_csr", "spmm_csr", "sddmm_csr", "gather", "scatter"],
        "baseline_policy": "ops-sparse/aclsparse when a callable bridge is supplied; otherwise PyTorch-NPU",
    }]

    def record(name, fs_fn, pt_fn, scipy_ref, output_to_numpy):
        fs_time = pt_time = None
        fs_err = pt_err = None
        status_parts = []
        try:
            fs_out = fs_fn()
            _sync(torch)
            fs_time = _bench(torch, fs_fn, warmup, iters)
            fs_np = output_to_numpy(fs_out)
            fs_err = float(np.max(np.abs(fs_np - scipy_ref))) if fs_np.size else 0.0
        except Exception as exc:
            status_parts.append(f"FlagSparse: {exc}")
        try:
            pt_out = pt_fn()
            _sync(torch)
            pt_time = _bench(torch, pt_fn, warmup, iters)
            pt_np = output_to_numpy(pt_out)
            pt_err = float(np.max(np.abs(pt_np - scipy_ref))) if pt_np.size else 0.0
            status_parts.append("PyTorch-NPU: PASS")
        except Exception as exc:
            status_parts.append(f"PyTorch-NPU: {exc}")
        results.append({"op": name, "flagsparse": fs_time, "pytorch": pt_time,
                        "scipy_max_abs_error": {"flagsparse": fs_err, "pytorch": pt_err},
                        "status": "; ".join(status_parts) if status_parts else "unknown"})

    record("spmv_csr", lambda: fs.flagsparse_spmv_csr(data, indices, indptr, x, (case.m, case.n)),
           lambda: _torch_npu_csr_spmv(data, indices, row_ids, x, case.m), scipy_spmv,
           lambda t: t.detach().cpu().numpy())
    record("spmm_csr", lambda: fs.flagsparse_spmm_csr(data, indices, indptr, b, (case.m, case.n)),
           lambda: _torch_npu_csr_spmm(data, indices, row_ids, b, case.m), scipy_spmm,
           lambda t: t.detach().cpu().numpy())
    record("sddmm_csr", lambda: fs.flagsparse_sddmm_csr(data, indices, indptr, sx, sy, (case.m, case.n)),
           lambda: (sx @ sy.T)[torch.tensor(rows, device=device), indices.to(torch.int64)],
           scipy_sddmm, lambda t: t.detach().cpu().numpy())

    # Gather/scatter are PyTorch indexing baselines; they are not advertised as
    # equivalent to a dedicated ops-sparse kernel.
    gather_idx = torch.arange(min(case.nnz, case.n), device=device, dtype=torch.int64)
    dense = torch.randn(case.n, device=device)
    values = torch.randn(gather_idx.numel(), device=device)
    record("gather", lambda: fs.flagsparse_gather(dense, gather_idx),
           lambda: torch.gather(dense, 0, gather_idx),
           dense.detach().cpu().numpy()[gather_idx.cpu().numpy()],
           lambda t: t.detach().cpu().numpy())
    # Keep independent buffers: flagsparse_scatter mutates its input in place,
    # while index_copy returns a new tensor.  Sharing one buffer would make the
    # second baseline depend on the first benchmark and invalidate correctness.
    scatter_fs = dense.detach().clone()
    scatter_pt = dense.detach().clone()
    scatter_initial = scatter_fs.detach().clone()
    scatter_ref = scatter_initial.detach().cpu().numpy().copy()
    scatter_ref[gather_idx.cpu().numpy()] = values.detach().cpu().numpy()
    record("scatter", lambda: (fs.flagsparse_scatter(scatter_fs, gather_idx, values) or scatter_fs),
           lambda: scatter_pt.index_copy(0, gather_idx, values), scatter_ref,
           lambda t: t.detach().cpu().numpy())
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--m", type=int, default=4096)
    p.add_argument("--n", type=int, default=4096)
    p.add_argument("--nnz", type=int, default=131072)
    p.add_argument("--dense-cols", type=int, default=64)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--device", type=int, default=0, help="Ascend NPU device ordinal")
    p.add_argument("--op", choices=("spmv_csr", "spmm_csr", "sddmm_csr", "gather", "scatter"), default=None)
    p.add_argument("--csv-summary", default=None, help="Write a runner-compatible one-row CSV summary")
    args = p.parse_args()
    case = Case(args.m, args.n, args.nnz, args.dense_cols, "float32")
    import json
    try:
        payload = run(case, args.warmup, args.iters, device_id=args.device)
    except (ImportError, RuntimeError) as exc:
        # Keep dependency/runtime failures machine-readable so a host without
        # torch_npu can still be checked in CI and the exact blocker is visible.
        payload = {
            "status": "blocked",
            "runtime": "Ascend/CANN",
            "reason": str(exc),
            "hint": "Install a matching torch + torch_npu + Triton environment and source CANN set_env.sh.",
        }
        print(json.dumps(payload, indent=2))
        raise SystemExit(2)
    if args.op is not None:
        payload = [item for item in payload if item.get("op") == args.op] if isinstance(payload, list) else payload
    if args.csv_summary:
        rows = []
        for item in payload if isinstance(payload, list) else []:
            if "op" not in item:
                continue
            fs = item.get("flagsparse") or {}
            pt = item.get("pytorch") or {}
            err = item.get("scipy_max_abs_error") or {}
            fs_ms = fs.get("mean_ms")
            pt_ms = pt.get("mean_ms")
            rows.append({
                "dtype": "float32",
                "shape": args.op or "ascend",
                "triton_ms": "" if fs_ms is None else fs_ms,
                "pytorch_ms": "" if pt_ms is None else pt_ms,
                "speedup": "" if fs_ms is None or not fs_ms else (pt_ms / fs_ms if pt_ms is not None else ""),
                "max_abs_err": "" if err.get("flagsparse") is None else err.get("flagsparse"),
                "status": item.get("status", ""),
            })
        with open(args.csv_summary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["dtype", "shape", "triton_ms", "pytorch_ms", "speedup", "max_abs_err", "status"])
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
