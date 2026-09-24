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
from scipy.io import mmread
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
    matrix_path: Path | None = None


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
    if case.matrix_path is not None:
        loaded = mmread(case.matrix_path)
        matrix = loaded.tocsr() if sp.issparse(loaded) else sp.csr_matrix(loaded)
        matrix.sum_duplicates()
        if np.iscomplexobj(matrix.data):
            matrix = matrix.real.tocsr()
        dtype = getattr(torch, case.dtype_name)
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        matrix = matrix.astype(np_dtype, copy=False)
        actual_case = Case(
            matrix.shape[0],
            matrix.shape[1],
            matrix.nnz,
            case.dense_cols,
            case.dtype_name,
            case.matrix_path,
        )
    else:
        actual_case = case
        rng = np.random.default_rng(20260909 + case.m + case.n + case.nnz)
        rows = rng.integers(0, case.m, size=case.nnz, dtype=np.int64)
        cols = rng.integers(0, case.n, size=case.nnz, dtype=np.int64)
        dtype = getattr(torch, case.dtype_name)
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        vals = rng.standard_normal(case.nnz).astype(np_dtype)
        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(case.m, case.n)).tocsr()
        matrix.sum_duplicates()
    data = torch.tensor(matrix.data, device=device, dtype=dtype)
    indices = torch.tensor(matrix.indices, device=device, dtype=torch.int32)
    indptr = torch.tensor(matrix.indptr, device=device, dtype=torch.int32)
    return matrix, data, indices, indptr, actual_case


def _make_coo(torch, case: Case, device):
    if case.matrix_path is not None:
        loaded = mmread(case.matrix_path)
        matrix = loaded.tocoo(copy=False) if sp.issparse(loaded) else sp.coo_matrix(loaded)
        if np.iscomplexobj(matrix.data):
            matrix = matrix.real.tocoo(copy=False)
        dtype = getattr(torch, case.dtype_name)
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        matrix = matrix.astype(np_dtype, copy=False)
        actual_case = Case(
            matrix.shape[0],
            matrix.shape[1],
            matrix.nnz,
            case.dense_cols,
            case.dtype_name,
            case.matrix_path,
        )
    else:
        actual_case = case
        rng = np.random.default_rng(20260909 + case.m + case.n + case.nnz)
        rows = rng.integers(0, case.m, size=case.nnz, dtype=np.int64)
        cols = rng.integers(0, case.n, size=case.nnz, dtype=np.int64)
        dtype = getattr(torch, case.dtype_name)
        np_dtype = np.float64 if dtype == torch.float64 else np.float32
        vals = rng.standard_normal(case.nnz).astype(np_dtype)
        matrix = sp.coo_matrix((vals, (rows, cols)), shape=(case.m, case.n))

    data = torch.tensor(matrix.data, device=device, dtype=dtype)
    row = torch.tensor(matrix.row, device=device, dtype=torch.int32)
    col = torch.tensor(matrix.col, device=device, dtype=torch.int32)
    return matrix, data, row, col, actual_case


def _scipy_spmv(matrix, x):
    return matrix @ x


def _scipy_numpy(tensor):
    """Convert NPU tensors to NumPy; NumPy has no bfloat16 dtype."""
    value = tensor.detach().cpu()
    if str(value.dtype) == "torch.bfloat16":
        value = value.float()
    return value.numpy()


# Which implementation each delivery operator actually executes on Ascend.  Every one
# of the seven falls back to torch_npu, because 910B's Triton has no shmem extension for
# the generic gather kernel and does not lower the CSR/COO kernels reliably -- see the
# _is_ascend_runtime() branches in each operator module.  This matters for reading the
# speedup column: the baseline is PyTorch-NPU, so on this backend BOTH sides of the
# ratio are the same library and a value near 1.000x means "we ARE the baseline", not
# "our kernel matches it".  tests/ci/test_ascend_implementation_labels.py keeps this
# table honest against the sources.
ASCEND_OPERATOR_IMPLEMENTATION = {
    "gather": "torch_npu",
    "scatter": "torch_npu",
    "spmv_csr": "torch_npu",
    "spmv_coo": "torch_npu",
    "spmm_csr": "torch_npu",
    "spmm_coo": "torch_npu",
    "sddmm_csr": "torch_npu",
}


def _ascend_implementation(op_name):
    """Label for the code path an operator runs here; 'triton' if it has no fallback."""
    return ASCEND_OPERATOR_IMPLEMENTATION.get(op_name, "triton")


def _ascend_complex_index_capability_reason(exc):
    """Classify only unsupported complex gather/scatter index operations."""
    text = f"{type(exc).__name__}: {exc}".lower()
    is_complex = "complex" in text or "dt_complex" in text
    is_index_api = any(token in text for token in ("index", "gather", "scatter"))
    is_unsupported = any(
        token in text
        for token in (
            "not support",
            "unsupported",
            "not implemented",
            "no kernel",
            "no implementation",
        )
    )
    if is_complex and is_index_api and is_unsupported:
        return f"Ascend complex gather/scatter capability unavailable: {exc}"
    return None


def _randn(torch, shape, device, dtype):
    if dtype == torch.complex64:
        real = torch.randn(shape, device="cpu", dtype=torch.float32)
        imag = torch.randn(shape, device="cpu", dtype=torch.float32)
        return torch.complex(real, imag).to(device)
    if dtype == torch.complex128:
        real = torch.randn(shape, device="cpu", dtype=torch.float64)
        imag = torch.randn(shape, device="cpu", dtype=torch.float64)
        return torch.complex(real, imag).to(device)
    return torch.randn(shape, device=device, dtype=dtype)


def _torch_gather(torch, dense, indices):
    if dense.is_complex():
        components = torch.view_as_real(dense).reshape(-1)
        real_indices = indices.to(torch.int64) * 2
        real = torch.gather(components, 0, real_indices)
        imag = torch.gather(components, 0, real_indices + 1)
        return torch.view_as_complex(torch.stack((real, imag), dim=-1))
    return torch.gather(dense, 0, indices)


def _torch_scatter(torch, dense, indices, values):
    dense.zero_()
    if dense.is_complex():
        dense_real = torch.view_as_real(dense)
        values_real = torch.view_as_real(values)
        dense_real.index_copy_(0, indices, values_real)
    else:
        dense.index_copy_(0, indices, values)
    return dense


def _case_for_gather_scatter(case: Case):
    if case.matrix_path is None:
        return case
    loaded = mmread(case.matrix_path)
    matrix = loaded.tocsr() if sp.issparse(loaded) else sp.csr_matrix(loaded)
    matrix.sum_duplicates()
    return Case(
        matrix.shape[0],
        matrix.shape[1],
        matrix.nnz,
        case.dense_cols,
        case.dtype_name,
        case.matrix_path,
    )


def _scipy_spmm(matrix, b):
    return matrix @ b


def _scipy_sddmm(matrix, x, y, chunk_nnz=262144):
    """Reference SDDMM without materializing the dense m-by-n product."""
    rows = np.repeat(np.arange(matrix.shape[0], dtype=np.int64), np.diff(matrix.indptr))
    result = np.empty(matrix.nnz, dtype=np.result_type(x.dtype, y.dtype))
    for begin in range(0, matrix.nnz, chunk_nnz):
        end = min(begin + chunk_nnz, matrix.nnz)
        result[begin:end] = np.sum(
            x[rows[begin:end]] * y[matrix.indices[begin:end]], axis=1
        )
    return result


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


def _torch_npu_coo_spmv(data, row, col, x, n_rows):
    import torch

    out = torch.zeros((n_rows,), device=data.device, dtype=data.dtype)
    out.index_add_(0, row.to(torch.int64), data * x[col.to(torch.int64)])
    return out


def _torch_npu_coo_spmm(data, row, col, B, n_rows):
    import torch

    out = torch.zeros((n_rows, B.shape[1]), device=data.device, dtype=data.dtype)
    values = data[:, None] * B[col.to(torch.int64)]
    out.index_add_(0, row.to(torch.int64), values)
    return out


def _accuracy_tolerance(dtype):
    import torch

    if dtype in (torch.float16, torch.bfloat16, torch.float32, torch.complex64):
        return 2e-2
    return 1e-8


def _outputs_pass_accuracy(candidate, baseline, dtype):
    tolerance = _accuracy_tolerance(dtype)
    return np.allclose(
        candidate,
        baseline,
        rtol=tolerance,
        atol=tolerance,
        equal_nan=True,
    )


def _torch_npu_sddmm_sampled(x, y, row_ids, indices, chunk_nnz=262144):
    """PyTorch-NPU SDDMM reference with bounded temporary memory."""
    import torch

    cols = indices.to(torch.int64)
    out = torch.empty((indices.numel(),), device=x.device, dtype=x.dtype)
    for begin in range(0, int(indices.numel()), chunk_nnz):
        end = min(begin + chunk_nnz, int(indices.numel()))
        out[begin:end] = torch.sum(x[row_ids[begin:end]] * y[cols[begin:end]], dim=1)
    return out


def row_status(fs_time) -> str:
    """The `status` cell, in the vocabulary the runner aggregates on.

    This column used to carry the descriptive string that is now `reason` -- on a
    fully successful case, "PyTorch-NPU: PASS", because the FlagSparse branch
    appends nothing when it works. _performance_row_status_is_usable() compares
    the whole cell against {PASS, PASSED, OK, SUCCESS}, so every Ascend row was
    ineligible for the per-dtype aggregate however good the measurement was, and
    the delivery table printed `-` beside a healthy operator.

    Module level, and not inlined at its one call site, so the contract with that
    gate is testable away from an NPU: this defect is invisible on any box that
    cannot run this script.
    """
    return "PASS" if fs_time is not None else "FAIL"


def run(case: Case, warmup: int, iters: int, device_id: int = 0, op: str | None = None):
    import torch
    import flagsparse as fs

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Ascend benchmark requires torch_npu and an available NPU")
    torch.npu.set_device(int(device_id))
    device = torch.device(f"npu:{int(device_id)}")
    if case.dtype_name in ("complex64", "complex128") and op not in ("gather", "scatter"):
        raise ValueError("Ascend complex benchmark dtypes are supported only for gather/scatter")
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
    coo_matrix = coo_data = coo_row = coo_col = None
    if op in ("gather", "scatter"):
        case = _case_for_gather_scatter(case)
        matrix = data = indices = indptr = None
    elif op in ("spmv_coo", "spmm_coo"):
        coo_matrix, coo_data, coo_row, coo_col, case = _make_coo(
            torch, case, device
        )
        matrix = data = indices = indptr = None
    else:
        matrix, data, indices, indptr, case = _make_csr(torch, case, device)
    dtype = getattr(torch, case.dtype_name)

    results = [{
        "device": f"npu:{int(device_id)}",
        "dtype": case.dtype_name,
        "matrix": case.matrix_path.name if case.matrix_path is not None else "synthetic",
        "shape": f"{case.m}x{case.n};nnz={case.nnz}",
        "ops_sparse": ops_status,
        "ops_sparse_910b_supported": ["spmv_csr", "sddmm_csr", "scatter"],
        "tested_ops": [
            "spmv_csr",
            "spmv_coo",
            "spmm_csr",
            "spmm_coo",
            "sddmm_csr",
            "gather",
            "scatter",
        ],
        "baseline_policy": "ops-sparse/aclsparse when a callable bridge is supplied; otherwise PyTorch-NPU",
    }]

    def record(name, fs_fn, pt_fn, scipy_ref, output_to_numpy):
        fs_time = pt_time = None
        fs_err = pt_err = None
        fs_out_np = pt_out_np = None
        status_parts = []
        fs_capability_reason = None
        baseline_capability_reason = None
        try:
            fs_out = fs_fn()
            _sync(torch)
            fs_out_np = output_to_numpy(fs_out)
            fs_err = float(np.max(np.abs(fs_out_np - scipy_ref))) if fs_out_np.size else 0.0
        except Exception as exc:
            status_parts.append(f"FlagSparse: {exc}")
            if device.type == "npu" and case.dtype_name in ("complex64", "complex128"):
                fs_capability_reason = _ascend_complex_index_capability_reason(exc)
        try:
            pt_out = pt_fn()
            _sync(torch)
            pt_out_np = output_to_numpy(pt_out)
            pt_err = float(np.max(np.abs(pt_out_np - scipy_ref))) if pt_out_np.size else 0.0
        except Exception as exc:
            status_parts.append(f"PyTorch-NPU: {exc}")
            if device.type == "npu" and case.dtype_name in ("complex64", "complex128"):
                baseline_capability_reason = _ascend_complex_index_capability_reason(exc)
                if baseline_capability_reason:
                    status_parts.append(baseline_capability_reason)
        outputs_match = (
            fs_out_np is not None
            and pt_out_np is not None
            and _outputs_pass_accuracy(
                fs_out_np,
                pt_out_np,
                dtype,
            )
        )
        if fs_out_np is not None and pt_out_np is not None and not outputs_match:
            status_parts.append("FlagSparse and PyTorch-NPU outputs differ beyond tolerance")
        if not status_parts:
            fs_time = _bench(torch, fs_fn, warmup, iters)
            pt_time = _bench(torch, pt_fn, warmup, iters)
        row_state = (
            "SKIP"
            if fs_capability_reason or baseline_capability_reason
            else "PASS" if fs_time is not None and pt_time is not None else "FAIL"
        )
        if row_state == "PASS":
            status_parts.append("PyTorch-NPU: PASS")
        results.append({"op": name, "implementation": _ascend_implementation(name),
                        "dtype": case.dtype_name,
                        "matrix": case.matrix_path.name if case.matrix_path is not None else "synthetic",
                        "shape": f"{case.m}x{case.n};nnz={case.nnz}",
                        "flagsparse": fs_time, "pytorch": pt_time,
                        "scipy_max_abs_error": {"flagsparse": fs_err, "pytorch": pt_err},
                        "status": row_state,
                        "reason": "; ".join(status_parts) if status_parts else "unknown"})

    if op in (None, "spmv_csr"):
        x = torch.randn(case.n, device=device, dtype=dtype)
        row_ids = _torch_npu_csr_row_ids(indptr, case.m)
        record("spmv_csr", lambda: fs.flagsparse_spmv_csr(data, indices, indptr, x, (case.m, case.n)),
               lambda: _torch_npu_csr_spmv(data, indices, row_ids, x, case.m),
               _scipy_spmv(matrix, _scipy_numpy(x)), _scipy_numpy)
    if op in (None, "spmm_csr"):
        b = torch.randn(case.n, case.dense_cols, device=device, dtype=dtype)
        row_ids = _torch_npu_csr_row_ids(indptr, case.m)
        record("spmm_csr", lambda: fs.flagsparse_spmm_csr(data, indices, indptr, b, (case.m, case.n)),
               lambda: _torch_npu_csr_spmm(data, indices, row_ids, b, case.m),
               _scipy_spmm(matrix, _scipy_numpy(b)), _scipy_numpy)
    if op in (None, "sddmm_csr"):
        sx = torch.randn(case.m, 32, device=device, dtype=dtype)
        sy = torch.randn(case.n, 32, device=device, dtype=dtype)
        row_ids = _torch_npu_csr_row_ids(indptr, case.m)
        record("sddmm_csr", lambda: fs.flagsparse_sddmm_csr(data, indices, indptr, sx, sy, (case.m, case.n)),
               lambda: _torch_npu_sddmm_sampled(sx, sy, row_ids, indices),
               _scipy_sddmm(matrix, _scipy_numpy(sx), _scipy_numpy(sy)), _scipy_numpy)

    if op == "spmv_coo":
        x = torch.randn(case.n, device=device, dtype=dtype)
        record(
            "spmv_coo",
            lambda: fs.flagsparse_spmv_coo(
                coo_data, coo_row, coo_col, x, (case.m, case.n), op="non"
            ),
            lambda: _torch_npu_coo_spmv(
                coo_data, coo_row, coo_col, x, case.m
            ),
            _scipy_spmv(coo_matrix, _scipy_numpy(x)),
            _scipy_numpy,
        )
    if op == "spmm_coo":
        b = torch.randn(case.n, case.dense_cols, device=device, dtype=dtype)
        record(
            "spmm_coo",
            lambda: fs.flagsparse_spmm_coo(
                coo_data, coo_row, coo_col, b, (case.m, case.n), op="non"
            ),
            lambda: _torch_npu_coo_spmm(
                coo_data, coo_row, coo_col, b, case.m
            ),
            _scipy_spmm(coo_matrix, _scipy_numpy(b)),
            _scipy_numpy,
        )

    # Gather/scatter are PyTorch indexing baselines; they are not advertised as
    # equivalent to a dedicated ops-sparse kernel.
    if op in (None, "gather", "scatter"):
        gather_idx = torch.arange(min(case.nnz, case.n), device=device, dtype=torch.int64)
        try:
            dense = _randn(torch, case.n, device, dtype)
            values = _randn(torch, gather_idx.numel(), device, dtype)
        except Exception as exc:
            capability_reason = None
            if device.type == "npu" and case.dtype_name in ("complex64", "complex128"):
                capability_reason = _ascend_complex_index_capability_reason(exc)
            if not capability_reason:
                raise
            for name in (("gather", "scatter") if op is None else (op,)):
                results.append({
                    "op": name,
                    "dtype": case.dtype_name,
                    "matrix": case.matrix_path.name if case.matrix_path is not None else "synthetic",
                    "shape": f"{case.m}x{case.n};nnz={case.nnz}",
                    "flagsparse": None,
                    "pytorch": None,
                    "scipy_max_abs_error": {"flagsparse": None, "pytorch": None},
                    "status": "SKIP",
                    "reason": capability_reason,
                })
            return results
    if op in (None, "gather"):
        record("gather", lambda: fs.flagsparse_gather(dense, gather_idx),
               lambda: _torch_gather(torch, dense, gather_idx),
               _scipy_numpy(dense)[gather_idx.cpu().numpy()], _scipy_numpy)
    # Keep independent buffers: flagsparse_scatter mutates its input in place,
    # while index_copy returns a new tensor.  Sharing one buffer would make the
    # second baseline depend on the first benchmark and invalidate correctness.
    if op in (None, "scatter"):
        scatter_fs = dense.detach().clone()
        scatter_pt = dense.detach().clone()
        scatter_ref = np.zeros_like(_scipy_numpy(scatter_fs))
        scatter_ref[gather_idx.cpu().numpy()] = _scipy_numpy(values)
        record("scatter", lambda: (fs.flagsparse_scatter(scatter_fs, gather_idx, values) or scatter_fs),
               lambda: _torch_scatter(torch, scatter_pt, gather_idx, values), scatter_ref,
               _scipy_numpy)
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
    p.add_argument("--op", choices=("spmv_csr", "spmv_coo", "spmm_csr", "spmm_coo", "sddmm_csr", "gather", "scatter"), default=None)
    p.add_argument("--dtypes", default="float32", help="Comma-separated value dtypes")
    p.add_argument("--input", type=Path, default=None,
                   help="MatrixMarket .mtx file or directory; every .mtx is benchmarked")
    p.add_argument("--csv-summary", default=None, help="Write a runner-compatible one-row CSV summary")
    args = p.parse_args()
    dtype_names = [item.strip() for item in args.dtypes.split(",") if item.strip()]
    allowed_dtypes = {"float16", "bfloat16", "float32", "float64", "complex64", "complex128"}
    unknown = sorted(set(dtype_names) - allowed_dtypes)
    if not dtype_names or unknown:
        p.error("--dtypes must contain names from: " + ", ".join(sorted(allowed_dtypes)))
    if set(dtype_names) & {"complex64", "complex128"} and args.op not in ("gather", "scatter"):
        p.error("complex dtypes are supported only for --op gather or --op scatter")
    import json
    if args.input is None:
        matrix_paths: list[Path | None] = [None]
    elif args.input.is_file():
        matrix_paths = [args.input]
    elif args.input.is_dir():
        matrix_paths = sorted(args.input.glob("*.mtx"))
        if not matrix_paths:
            p.error(f"no .mtx files found in {args.input}")
    else:
        p.error(f"--input does not exist: {args.input}")
    payload = []
    for dtype_name in dtype_names:
        for matrix_path in matrix_paths:
            try:
                dtype_payload = run(
                    Case(args.m, args.n, args.nnz, args.dense_cols, dtype_name, matrix_path),
                    args.warmup,
                    args.iters,
                    device_id=args.device,
                    op=args.op,
                )
            except (ImportError, RuntimeError) as exc:
                dtype_payload = [{"dtype": dtype_name, "matrix": str(matrix_path or "synthetic"),
                                  "status": "blocked", "reason": str(exc)}]
            payload.extend(dtype_payload)
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
                "dtype": item.get("dtype", "float32"),
                "matrix": item.get("matrix", ""),
                "shape": item.get("op", args.op or "ascend"),
                # Not necessarily Triton on this backend -- see
                # ASCEND_OPERATOR_IMPLEMENTATION.  The column name is kept because the
                # runner's schema matches on it; this field says what actually ran.
                "implementation": item.get("implementation", "triton"),
                "triton_ms": "" if fs_ms is None else fs_ms,
                "pytorch_ms": "" if pt_ms is None else pt_ms,
                "speedup": "" if fs_ms is None or not fs_ms else (pt_ms / fs_ms if pt_ms is not None else ""),
                "max_abs_err": "" if err.get("flagsparse") is None else err.get("flagsparse"),
                "status": item.get("status", ""),
                "reason": item.get("reason", ""),
            })
        with open(args.csv_summary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["dtype", "matrix", "shape", "implementation",
                            "triton_ms", "pytorch_ms",
                            "speedup", "max_abs_err", "status", "reason"],
            )
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
