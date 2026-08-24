# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native BSR SpMM benchmark and correctness script."""

import argparse
import csv
import glob
import math
import os
import sys
import time
import warnings
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
from flagsparse.sparse_operations import spmm_bsr as bsr_ops

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:
    cp = None
    cpx_sparse = None

try:
    import numpy as np
    from scipy.sparse import bsr_matrix as scipy_bsr_matrix
except ImportError as exc:
    np = None
    scipy_bsr_matrix = None
    SCIPY_IMPORT_ERROR = str(exc)
else:
    SCIPY_IMPORT_ERROR = None


VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = (torch.int32, torch.int64)
OPS = ("non", "trans", "conj")
SUPPORTED_OPS = OPS
ALGS = ("auto", "spmm_bsr_base", "base", "all")
TEST_SIZES = ((64, 96, 16), (160, 1024, 32), (128, 256, 48))
DEFAULT_BLOCK_DIMS = (2,)
WARMUP = 10
ITERS = 50

PERF_FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "op",
    "layout",
    "alg",
    "block_dim",
    "ref",
    "out_rows",
    "padded_out_rows",
    "pad_rows",
    "n_rows",
    "n_cols",
    "nnzb",
    "stored_nnz",
    "pad_ratio",
    "dense_cols",
    "b_stride",
    "c_stride",
    "ms",
    "gpu_ms",
    "process_cpu_ms",
    "torch_ms",
    "cusparse_ms",
    "scipy_cpu_ms",
    "torch_vs_alg_speedup",
    "cusparse_vs_alg_speedup",
    "scipy_vs_alg_speedup",
    "err_vs_ref",
    "err_vs_torch",
    "err_vs_cusparse",
    "scipy_cpu_err",
    "status",
    "reason",
    "torch_reason",
    "cusparse_reason",
    "scipy_reason",
]
TIMING_FIELDS = ["process_gpu_ms", "compute_ms"]


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {"int32": torch.int32, "int64": torch.int64}


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _reference_dtype(dtype):
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _reference_tolerance(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    return 1e-6, 1e-5


def _error_ratio(actual, expected, dtype):
    if actual is None or expected is None:
        return None
    atol, rtol = _reference_tolerance(dtype)
    if expected.numel() == 0:
        return 0.0
    diff = torch.abs(actual - expected).to(torch.float64)
    denom = (atol + rtol * torch.abs(expected)).to(torch.float64)
    ratios = diff / denom
    return float(torch.max(ratios).item()) if ratios.numel() else 0.0


def _parse_csv_tokens(value, mapping, option_name):
    tokens = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{option_name} must not be empty")
    invalid = [token for token in tokens if token not in mapping]
    if invalid:
        raise ValueError(
            f"unsupported {option_name}: {', '.join(invalid)}; allowed: {', '.join(mapping)}"
        )
    return [mapping[token] for token in tokens]


def _parse_ops(value):
    token = str(value).strip().lower()
    if token == "all":
        return list(OPS)
    ops = [item.strip().lower() for item in token.split(",") if item.strip()]
    invalid = [op for op in ops if op not in OPS]
    if not ops or invalid:
        raise ValueError(f"unsupported --ops: {', '.join(invalid or ops)}")
    return ops


def _parse_algs(value):
    token = str(value or "auto").strip().lower().replace("-", "_")
    algs = [item.strip().lower().replace("-", "_") for item in token.split(",") if item.strip()]
    aliases = {"base": "spmm_bsr_base"}
    algs = [aliases.get(alg, alg) for alg in algs]
    invalid = [alg for alg in algs if alg not in ALGS and alg != "spmm_bsr_base"]
    if not algs or invalid:
        raise ValueError("unsupported --alg; allowed: auto, all, base, spmm_bsr_base")
    return algs


def _expand_algs(algs, op, dtype):
    del dtype
    if op not in SUPPORTED_OPS:
        return []
    out = []
    for alg in algs:
        if alg in ("auto", "all"):
            out.append("spmm_bsr_base")
        else:
            out.append(alg)
    deduped = []
    for alg in out:
        if alg not in deduped:
            deduped.append(alg)
    return deduped


def _parse_block_dims(value):
    dims = []
    for item in str(value or "2").split(","):
        item = item.strip()
        if not item:
            continue
        dim = int(item)
        if dim <= 1:
            raise ValueError("--block-dims values must be greater than 1")
        dims.append(dim)
    if not dims:
        raise ValueError("--block-dims must not be empty")
    return dims


def _normalize_layout_name(layout):
    token = str(layout).strip().lower()
    if token in ("row", "row_major", "row-major", "c", "c_order", "auto", "default"):
        return "row"
    if token in ("col", "column", "col_major", "column-major", "f", "fortran"):
        return "col"
    raise ValueError("layout must be one of: row, col, all")


def _layout_names(value):
    token = str(value).strip().lower()
    if token == "all":
        return ["row", "col"]
    return [_normalize_layout_name(token)]


def _materialize_dense_layout(tensor, layout):
    layout = _normalize_layout_name(layout)
    if layout == "row":
        return tensor.contiguous()
    out = torch.empty_strided(
        tuple(tensor.shape),
        (1, max(1, int(tensor.shape[0]))),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    out.copy_(tensor)
    return out


def _stride_string(tensor):
    return "x".join(str(int(v)) for v in tensor.stride())


def _random_values(shape, dtype, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.complex64:
        return torch.complex(
            torch.randn(shape, dtype=torch.float32, device=device),
            torch.randn(shape, dtype=torch.float32, device=device),
        )
    if dtype == torch.complex128:
        return torch.complex(
            torch.randn(shape, dtype=torch.float64, device=device),
            torch.randn(shape, dtype=torch.float64, device=device),
        )
    raise TypeError(f"unsupported dtype: {dtype}")


def _zero_value(dtype):
    return 0j if dtype in (torch.complex64, torch.complex128) else 0.0


def _mtx_value_for_dtype(raw_value, dtype):
    if dtype in (torch.complex64, torch.complex128):
        return complex(raw_value)
    return float(raw_value.real if isinstance(raw_value, complex) else raw_value)


def _padded_shape(shape, block_dim):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    return (
        ((n_rows + block_dim - 1) // block_dim) * block_dim,
        ((n_cols + block_dim - 1) // block_dim) * block_dim,
    )


def _op_transposes(op):
    return str(op).lower() in ("trans", "conj")


def _logical_b_rows(shape, op):
    return int(shape[0]) if _op_transposes(op) else int(shape[1])


def _logical_out_rows(shape, op):
    return int(shape[1]) if _op_transposes(op) else int(shape[0])


def _padded_b_rows(shape, block_dim, op):
    padded_rows, padded_cols = _padded_shape(shape, block_dim)
    return padded_rows if _op_transposes(op) else padded_cols


def _padded_out_rows(shape, block_dim, op):
    padded_rows, padded_cols = _padded_shape(shape, block_dim)
    return padded_cols if _op_transposes(op) else padded_rows


def _entries_to_bsr(entries, shape, dtype, index_dtype, block_dim, device):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    n_block_rows = (n_rows + block_dim - 1) // block_dim
    blocks = {}
    for (row, col), value in entries.items():
        brow = int(row) // block_dim
        bcol = int(col) // block_dim
        inner_row = int(row) % block_dim
        inner_col = int(col) % block_dim
        block = blocks.setdefault(
            (brow, bcol),
            [_zero_value(dtype) for _ in range(block_dim * block_dim)],
        )
        block[inner_row * block_dim + inner_col] += _mtx_value_for_dtype(value, dtype)
    row_blocks = [[] for _ in range(n_block_rows)]
    for key in sorted(blocks):
        row_blocks[key[0]].append(key)
    data_values = []
    indices_values = []
    indptr_values = [0]
    for keys in row_blocks:
        for key in keys:
            indices_values.append(key[1])
            data_values.extend(blocks[key])
        indptr_values.append(len(indices_values))
    data = torch.tensor(data_values, dtype=dtype, device=device)
    data = data.reshape(-1, block_dim, block_dim).contiguous()
    indices = torch.tensor(indices_values, dtype=index_dtype, device=device).contiguous()
    indptr = torch.tensor(indptr_values, dtype=index_dtype, device=device).contiguous()
    return data, indices, indptr


def _dense_to_bsr(dense, index_dtype, block_dim):
    rows, cols = torch.nonzero(dense != 0, as_tuple=True)
    entries = {
        (int(row.item()), int(col.item())): dense[row, col].item()
        for row, col in zip(rows, cols)
    }
    return _entries_to_bsr(
        entries,
        tuple(dense.shape),
        dense.dtype,
        index_dtype,
        block_dim,
        dense.device,
    )


def _read_mtx_entries(path):
    entries = {}
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip().split()
        if len(header) < 5 or header[0] != "%%MatrixMarket":
            raise ValueError("invalid MatrixMarket header")
        field = header[3].lower()
        symmetry = header[4].lower()
        line = fh.readline()
        while line.startswith("%"):
            line = fh.readline()
        n_rows, n_cols, _nnz = [int(v) for v in line.strip().split()[:3]]
        pattern = field == "pattern"
        complex_field = field == "complex"
        for line in fh:
            if not line.strip() or line.startswith("%"):
                continue
            parts = line.strip().split()
            row = int(parts[0]) - 1
            col = int(parts[1]) - 1
            if pattern:
                value = 1.0
            elif complex_field:
                value = complex(float(parts[2]), float(parts[3]))
            else:
                value = float(parts[2])
            entries[(row, col)] = entries.get((row, col), 0) + value
            if symmetry in ("symmetric", "hermitian") and row != col:
                mirror = value.conjugate() if symmetry == "hermitian" else value
                entries[(col, row)] = entries.get((col, row), 0) + mirror
    return entries, (n_rows, n_cols)


def _make_synthetic_case(M, K, dtype, index_dtype, block_dim, device):
    p = min(0.25, max(0.06, 32.0 / max(M * K, 1)))
    mask = torch.rand(M, K, device=device) < p
    if int(mask.sum().item()) == 0:
        mask[0, 0] = True
    dense = torch.where(
        mask,
        _random_values((M, K), dtype, device) * 0.125,
        torch.zeros((), dtype=dtype, device=device),
    )
    data, indices, indptr = _dense_to_bsr(dense, index_dtype, block_dim)
    return "synthetic", data, indices, indptr, (M, K)


def _bsr_block_rows(indptr):
    counts = indptr[1:].to(torch.int64) - indptr[:-1].to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(indptr.numel() - 1, dtype=torch.int64, device=indptr.device),
        counts,
    )


def _bsr_to_torch_coo(data, indices, indptr, shape, block_dim):
    block_rows = _bsr_block_rows(indptr)
    nnzb = int(data.shape[0])
    if nnzb == 0:
        empty = torch.empty(0, dtype=torch.int64, device=data.device)
        return torch.sparse_coo_tensor(
            torch.stack([empty, empty]),
            data.reshape(-1),
            size=shape,
            device=data.device,
            dtype=data.dtype,
        ).coalesce()
    local = torch.arange(block_dim * block_dim, dtype=torch.int64, device=data.device)
    inner_rows = local // block_dim
    inner_cols = local % block_dim
    rows = block_rows[:, None] * block_dim + inner_rows[None, :]
    cols = indices.to(torch.int64)[:, None] * block_dim + inner_cols[None, :]
    values = data.reshape(nnzb, block_dim * block_dim)
    mask = (rows < int(shape[0])) & (cols < int(shape[1])) & (values != 0)
    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]),
        values,
        size=shape,
        device=data.device,
        dtype=data.dtype,
    ).coalesce()


def _torch_spmm_coo_reference(data, indices, indptr, B, shape, dtype, block_dim, op):
    ref_dtype = _reference_dtype(dtype)
    A = _bsr_to_torch_coo(
        data.to(ref_dtype),
        indices,
        indptr,
        shape,
        block_dim,
    )
    if op == "non":
        return torch.sparse.mm(A, B[: int(shape[1]), :].to(ref_dtype)).to(dtype)
    if op == "trans":
        return torch.sparse.mm(A.transpose(0, 1), B[: int(shape[0]), :].to(ref_dtype)).to(dtype)
    if op == "conj":
        return torch.sparse.mm(A.conj().transpose(0, 1), B[: int(shape[0]), :].to(ref_dtype)).to(dtype)
    raise ValueError(f"unsupported op: {op}")


def _cuda_event_benchmark(op, warmup, iters):
    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    count = max(1, int(iters))
    start.record()
    for _ in range(count):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / count


def _pad_dense_for_bsr_run(B, padded_rows):
    if B.shape[0] == padded_rows:
        return B
    if B.stride(0) == 1 and B.shape[0] > 1:
        padded = torch.empty_strided(
            (padded_rows, B.shape[1]),
            (1, max(1, int(padded_rows))),
            dtype=B.dtype,
            device=B.device,
        )
        padded.zero_()
    else:
        padded = torch.zeros((padded_rows, B.shape[1]), dtype=B.dtype, device=B.device)
    padded[: B.shape[0], :].copy_(B)
    return padded


def _time_flagsparse_bsr(data, indices, indptr, B, shape, block_dim, alg, op, warmup, iters, timing=False):
    prepared = fs.prepare_spmm_bsr_route(
        data, indices, indptr, shape, block_dim=block_dim, alg=alg, op=op
    )
    B_for_bsr = _pad_dense_for_bsr_run(B, _padded_b_rows(shape, block_dim, op))
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmm_bsr_run(prepared, B_for_bsr, alg=alg, op=op),
        warmup,
        iters,
    )
    _meta_out, meta = fs.flagsparse_spmm_bsr_run(
        prepared,
        B_for_bsr,
        alg=alg,
        op=op,
        return_meta=True,
        timing=bool(timing),
    )
    process_cpu_ms = float(meta.get("process_cpu_ms", 0.0) or 0.0)
    process_gpu_ms = meta.get("process_gpu_ms") if timing else None
    compute_ms = meta.get("compute_ms") if timing else None
    if timing:
        if process_gpu_ms is None:
            process_gpu_ms = 0.0
        if compute_ms is None:
            compute_ms = gpu_ms
    return {
        "out": out,
        "ms": process_cpu_ms + gpu_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": process_cpu_ms,
        "process_gpu_ms": process_gpu_ms,
        "compute_ms": compute_ms,
    }


def _time_pytorch_bsr(data, indices, indptr, B, shape, block_dim, op, warmup, iters):
    if op in ("trans", "conj"):
        return None, "PyTorch CUDA BSR transpose-family SpMM baseline is unsupported; no fallback", None
    padded_shape = _padded_shape(shape, block_dim)
    padded_B = B
    if B.shape[0] != padded_shape[1]:
        padded_B = torch.zeros((padded_shape[1], B.shape[1]), dtype=B.dtype, device=B.device)
        padded_B[: B.shape[0], :].copy_(B)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse BSR tensor support is in beta state.*",
            category=UserWarning,
        )
        A = torch.sparse_bsr_tensor(
            indptr,
            indices,
            data,
            size=padded_shape,
            device=data.device,
            dtype=data.dtype,
        )
    out, ms = _cuda_event_benchmark(lambda: torch.sparse.mm(A, padded_B), warmup, iters)
    return ms, None, out


def _cupy_bsr_unavailable_reason():
    if cp is None or cpx_sparse is None:
        return "CuPy/cupyx.scipy.sparse is not available"
    if not hasattr(cpx_sparse, "bsr_matrix"):
        return "CuPy cupyx.scipy.sparse has no bsr_matrix baseline"
    return None


def _time_cusparse_bsr(data, indices, indptr, B, shape, block_dim, op, warmup, iters):
    backend, backend_reason = bsr_ops._spmm_bsr_sparse_ref_backend(
        data.dtype, indices.dtype, op=op
    )
    if backend == "hipsparse":
        # DCU/ROCm: the generic SpMM API has no BSR format, so the vendor
        # baseline is the legacy hipsparseXbsrmm entry point.  Without this
        # branch the DCU run has no vendor column at all.
        padded = _padded_shape(shape, block_dim)
        B_use = B
        padded_b_rows = _padded_b_rows(shape, block_dim, op)
        if B.shape[0] != padded_b_rows:
            B_use = torch.zeros(
                (padded_b_rows, B.shape[1]), dtype=B.dtype, device=B.device
            )
            B_use[: B.shape[0], :].copy_(B)
        ref = bsr_ops._benchmark_spmm_bsr_sparse_ref(
            data, indices, indptr, B_use, padded, block_dim, warmup, iters, op=op
        )
        if ref["values"] is None:
            return None, ref["reason"] or "hipSPARSE BSR SpMM reference skipped", None
        return ref["ms"], None, ref["values"]
    if backend is None and getattr(torch.version, "hip", None) is not None:
        return None, backend_reason, None
    reason = _cupy_bsr_unavailable_reason()
    if reason:
        return None, reason, None
    if indices.dtype != torch.int32 or indptr.dtype != torch.int32:
        return None, "CuPy BSR baseline requires int32 indices/indptr; no implicit index fallback", None
    padded_shape = _padded_shape(shape, block_dim)
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
    B_use = B
    padded_b_rows = _padded_b_rows(shape, block_dim, op)
    if B.shape[0] != padded_b_rows:
        B_use = torch.zeros((padded_b_rows, B.shape[1]), dtype=B.dtype, device=B.device)
        B_use[: B.shape[0], :].copy_(B)
    B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B_use))
    A = cpx_sparse.bsr_matrix((data_cp, ind_cp, ptr_cp), shape=padded_shape)
    if op == "non":
        fn = lambda: A @ B_cp
    elif op == "trans":
        fn = lambda: A.T @ B_cp
    elif op == "conj":
        fn = lambda: A.conj().T @ B_cp
    else:
        raise ValueError(f"unsupported op: {op}")
    for _ in range(max(0, int(warmup))):
        _ = fn()
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    count = max(1, int(iters))
    start.record()
    for _ in range(count):
        out_cp = fn()
    end.record()
    end.synchronize()
    out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
    return cp.cuda.get_elapsed_time(start, end) / count, None, out


def _time_scipy_bsr_cpu(data, indices, indptr, B, shape, block_dim, op, warmup, iters):
    if scipy_bsr_matrix is None or np is None:
        return None, f"SciPy BSR baseline is not available: {SCIPY_IMPORT_ERROR}", None
    padded_shape = _padded_shape(shape, block_dim)
    data_np = data.detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy()
    indptr_np = indptr.detach().cpu().numpy()
    B_use = _pad_dense_for_bsr_run(B, _padded_b_rows(shape, block_dim, op))
    B_np = B_use.detach().cpu().numpy()
    A = scipy_bsr_matrix((data_np, indices_np, indptr_np), shape=padded_shape)
    if op == "non":
        fn = lambda: A @ B_np
    elif op == "trans":
        fn = lambda: A.T @ B_np
    elif op == "conj":
        fn = lambda: A.conj().T @ B_np
    else:
        raise ValueError(f"unsupported op: {op}")
    out_np = None
    for _ in range(max(0, int(warmup))):
        out_np = fn()
    count = max(1, int(iters))
    start = time.perf_counter()
    for _ in range(count):
        out_np = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / count
    out = torch.as_tensor(out_np, dtype=data.dtype, device=data.device)
    return elapsed_ms, None, out


def _run_case(matrix_name, data, indices, indptr, shape, dtype, index_dtype, block_dim, dense_cols, layout, alg, op, warmup, iters, timing, run_cusparse):
    B = _materialize_dense_layout(
        _random_values((_logical_b_rows(shape, op), int(dense_cols)), dtype, data.device) * 0.125,
        layout,
    )
    ref = _torch_spmm_coo_reference(data, indices, indptr, B, shape, dtype, block_dim, op)
    logical_out_rows = _logical_out_rows(shape, op)
    padded_out_rows = _padded_out_rows(shape, block_dim, op)
    row = {
        "matrix": matrix_name,
        "dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "layout": layout,
        "alg": alg,
        "block_dim": block_dim,
        "ref": "torch_spmm_coo",
        "out_rows": logical_out_rows,
        "padded_out_rows": padded_out_rows,
        "pad_rows": padded_out_rows - logical_out_rows,
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "nnzb": int(data.shape[0]),
        "stored_nnz": int(data.numel()),
        "pad_ratio": (float(data.numel()) / max(1, int(torch.count_nonzero(data).item()))),
        "dense_cols": int(dense_cols),
        "b_stride": _stride_string(B),
        "c_stride": "",
        "ms": None,
        "gpu_ms": None,
        "process_cpu_ms": 0.0,
        "torch_ms": None,
        "cusparse_ms": None,
        "scipy_cpu_ms": None,
        "torch_vs_alg_speedup": None,
        "cusparse_vs_alg_speedup": None,
        "scipy_vs_alg_speedup": None,
        "err_vs_ref": None,
        "err_vs_torch": None,
        "err_vs_cusparse": None,
        "scipy_cpu_err": None,
        "status": "ERROR",
        "reason": "",
        "torch_reason": "",
        "cusparse_reason": "",
        "scipy_reason": "",
        "process_gpu_ms": None,
        "compute_ms": None,
    }
    try:
        bsr = _time_flagsparse_bsr(
            data, indices, indptr, B, shape, block_dim, alg, op, warmup, iters, timing=timing
        )
        out_logical = bsr["out"][:logical_out_rows, :]
        row.update(
            {
                "ms": bsr["ms"],
                "gpu_ms": bsr["gpu_ms"],
                "process_cpu_ms": bsr["process_cpu_ms"],
                "process_gpu_ms": bsr["process_gpu_ms"],
                "compute_ms": bsr["compute_ms"],
                "c_stride": _stride_string(bsr["out"]),
                "err_vs_ref": _error_ratio(out_logical, ref, dtype),
            }
        )
        row["status"] = "PASS" if row["err_vs_ref"] is not None and row["err_vs_ref"] <= 1.0 else "FAIL"
        if row["status"] == "FAIL":
            row["reason"] = "correctness check failed"
    except Exception as exc:
        row["reason"] = str(exc)
        return row
    try:
        torch_ms, torch_reason, torch_out = _time_pytorch_bsr(
            data, indices, indptr, B, shape, block_dim, op, warmup, iters
        )
        row["torch_ms"] = torch_ms
        row["torch_reason"] = torch_reason or ""
        if torch_out is not None:
            row["err_vs_torch"] = _error_ratio(torch_out[:logical_out_rows, :], ref, dtype)
    except Exception as exc:
        row["torch_reason"] = str(exc)
    if run_cusparse:
        try:
            cu_ms, cu_reason, cu_out = _time_cusparse_bsr(
                data, indices, indptr, B, shape, block_dim, op, warmup, iters
            )
            row["cusparse_ms"] = cu_ms
            row["cusparse_reason"] = cu_reason or ""
            if cu_out is not None:
                row["err_vs_cusparse"] = _error_ratio(cu_out[:logical_out_rows, :], ref, dtype)
        except Exception as exc:
            row["cusparse_reason"] = str(exc)
    try:
        scipy_ms, scipy_reason, scipy_out = _time_scipy_bsr_cpu(
            data, indices, indptr, B, shape, block_dim, op, warmup, iters
        )
        row["scipy_cpu_ms"] = scipy_ms
        row["scipy_reason"] = scipy_reason or ""
        if scipy_out is not None:
            row["scipy_cpu_err"] = _error_ratio(scipy_out[:logical_out_rows, :], ref, dtype)
    except Exception as exc:
        row["scipy_reason"] = str(exc)
    row["torch_vs_alg_speedup"] = _ratio(row["torch_ms"], row["ms"])
    row["cusparse_vs_alg_speedup"] = _ratio(row["cusparse_ms"], row["ms"])
    row["scipy_vs_alg_speedup"] = _ratio(row["scipy_cpu_ms"], row["ms"])
    return row


def _resolve_input_paths(input_paths):
    paths = []
    for path in input_paths:
        if os.path.isfile(path) and path.lower().endswith(".mtx"):
            paths.append(os.path.abspath(path))
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    return paths


def _print_notes(run_cusparse):
    print("FlagSparse BSR SpMM follows padded block-grid semantics; native output is padded and correctness checks slice back to logical rows/cols by op.")
    print("Accuracy reference: Ref=torch_spmm_coo expands the same BSR arrays to COO and runs torch.sparse.mm; this is correctness-only, not the FlagSparse compute path.")
    print("PyTorch BSR baseline is attempted only for same-format supported cases; CUDA BSR transpose-family is recorded as N/A with no fallback.")
    if scipy_bsr_matrix is None:
        print(f"SciPy CPU BSR baseline: unavailable ({SCIPY_IMPORT_ERROR}); SciPy(ms)=N/A.")
    else:
        print("SciPy CPU BSR baseline: same BSR arrays with padded shape; CPU-vs-GPU speedup is diagnostic only.")
    if run_cusparse:
        reason = _cupy_bsr_unavailable_reason()
        if reason:
            print(f"CuPy baseline: unavailable for BSR ({reason}); CU(ms)=N/A.")


def _print_row(row, timing=False):
    print(
        f"{row['matrix']:<28} {row['dtype']:<10} {row['index_dtype']:<5} {row['op']:<4} {row['layout']:<4} {row['alg']:<14} "
        f"{row['block_dim']:>4} {row['n_rows']:>7} {row['n_cols']:>7} {row['nnzb']:>8} {row['dense_cols']:>5} "
        f"{_fmt(row['ms']):>9} {_fmt(row['gpu_ms']):>9} {_fmt(row['process_cpu_ms']):>9} "
        f"{_fmt(row['torch_ms']):>9} {_fmt(row['cusparse_ms']):>9} {_fmt(row['scipy_cpu_ms']):>9} "
        f"{_fmt(row['torch_vs_alg_speedup'], 2):>8} {_fmt(row['scipy_vs_alg_speedup'], 2):>8} "
        f"{_fmt(row['err_vs_ref'], 2):>10} {_fmt(row['scipy_cpu_err'], 2):>10} {row['status']:>6}"
        + (
            f" {_fmt(row.get('process_gpu_ms')):>9} {_fmt(row.get('compute_ms')):>9}"
            if timing
            else ""
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv-bsr", type=str, default=None, metavar="FILE")
    parser.add_argument("--dtypes", default="float32,float64,complex64,complex128")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--block-dims", default="2")
    parser.add_argument("--ops", default="non")
    parser.add_argument("--alg", default="auto")
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--layout", default="row")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for native BSR SpMM benchmark")
    dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
    index_dtypes = _parse_csv_tokens(args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes")
    block_dims = _parse_block_dims(args.block_dims)
    ops = _parse_ops(args.ops)
    algs = _parse_algs(args.alg)
    layouts = _layout_names(args.layout)
    run_cusparse = not args.no_cusparse
    _print_notes(run_cusparse)

    fields = PERF_FIELDS + (TIMING_FIELDS if args.timing else [])
    rows = []
    if args.csv_bsr:
        Path(args.csv_bsr).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    fh = None
    if args.csv_bsr:
        fh = open(args.csv_bsr, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()

    try:
        print("-" * 160)
        print(
            f"{'Matrix':<28} {'DType':<10} {'Index':<5} {'Op':<4} {'Lay':<4} {'Alg':<14} "
            f"{'BDim':>4} {'Rows':>7} {'Cols':>7} {'NNZB':>8} {'DCols':>5} "
            f"{'MS':>9} {'GPU':>9} {'CPUProc':>9} {'PT':>9} {'CU':>9} {'SciPy':>9} "
            f"{'PT/Alg':>8} {'Sci/Alg':>8} {'Err':>10} {'SciErr':>10} {'Status':>6}"
            + (f" {'GPUProc':>9} {'Compute':>9}" if args.timing else "")
        )
        print("-" * 160)
        device = torch.device("cuda")
        cases = []
        if args.synthetic:
            for M, K, dense_cols in TEST_SIZES:
                cases.append(("synthetic", None, (M, K), dense_cols))
        for path in _resolve_input_paths(args.mtx):
            cases.append((os.path.basename(path), path, None, args.dense_cols))
        if not cases:
            raise ValueError("provide --synthetic or at least one .mtx input")
        for dtype in dtypes:
            for index_dtype in index_dtypes:
                for block_dim in block_dims:
                    for case_name, path, synthetic_shape, dense_cols in cases:
                        if path is None:
                            data, indices, indptr, shape = _make_synthetic_case(
                                synthetic_shape[0], synthetic_shape[1], dtype, index_dtype, block_dim, device
                            )[1:]
                        else:
                            entries, shape = _read_mtx_entries(path)
                            data, indices, indptr = _entries_to_bsr(
                                entries, shape, dtype, index_dtype, block_dim, device
                            )
                        for op in ops:
                            expanded_algs = _expand_algs(algs, op, dtype)
                            if not expanded_algs:
                                row = {
                                    field: None for field in fields
                                }
                                row.update(
                                    {
                                        "matrix": case_name,
                                        "dtype": _dtype_name(dtype),
                                        "index_dtype": _dtype_name(index_dtype),
                                        "op": op,
                                        "layout": "",
                                        "alg": "",
                                        "block_dim": block_dim,
                                        "ref": "torch_spmm_coo",
                                        "n_rows": int(shape[0]),
                                        "n_cols": int(shape[1]),
                                        "status": "SKIP",
                                        "reason": f"spmm_bsr does not support op={op!r}",
                                    }
                                )
                                rows.append(row)
                                _print_row(row, timing=args.timing)
                                if writer:
                                    writer.writerow({field: row.get(field) for field in fields})
                                    fh.flush()
                                continue
                            for layout in layouts:
                                for alg in expanded_algs:
                                    row = _run_case(
                                        case_name,
                                        data,
                                        indices,
                                        indptr,
                                        shape,
                                        dtype,
                                        index_dtype,
                                        block_dim,
                                        dense_cols,
                                        layout,
                                        alg,
                                        op,
                                        args.warmup,
                                        args.iters,
                                        args.timing,
                                        run_cusparse,
                                    )
                                    rows.append(row)
                                    _print_row(row, timing=args.timing)
                                    if writer:
                                        writer.writerow({field: row.get(field) for field in fields})
                                        fh.flush()
                                    if args.fail_fast and row["status"] in ("FAIL", "ERROR"):
                                        raise RuntimeError(row.get("reason") or "case failed")
    finally:
        if fh is not None:
            fh.close()
    failures = sum(1 for row in rows if row.get("status") in ("FAIL", "ERROR"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
