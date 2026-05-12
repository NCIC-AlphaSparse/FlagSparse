"""
Protected CSR SpMM benchmark: base vs alg1 vs alg2 vs references.
Alg1/Alg2 totals report CPU-wall preprocessing plus CUDA-event compute time.

Usage:
    python tests/test_spmm_opt_alg2.py --synthetic --dense-cols 32 --with-cusparse
    python tests/test_spmm_opt_alg2.py path/to/mtx_dir --csv spmm_opt_alg2.csv --with-cusparse
"""

import argparse
import csv
import glob
import math
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
import flagsparse.sparse_operations.spmm_csr as spmm_csr_mod
import flagsparse.sparse_operations.spmm_csr_opt_alg2 as spmm_alg2_mod

from test_spmm_opt import _seeded_dense_matrix, load_mtx_to_csr_torch


VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 50
DEFAULT_DENSE_COLS = 32
DEFAULT_SEED = 0
SUMMARY_FIELDS = [
    "matrix",
    "value_dtype",
    "dense_cols",
    "seed",
    "n_rows",
    "n_cols",
    "nnz",
    "avg_nnz_per_row",
    "max_row_nnz",
    "base_ms",
    "opt_ms",
    "alg1_ms",
    "alg1_preprocess_ms",
    "alg1_compute_ms",
    "alg2_ms",
    "alg2_preprocess_ms",
    "alg2_compute_ms",
    "torch_ms",
    "cusparse_ms",
    "base_vs_alg1_speedup",
    "torch_vs_alg1_speedup",
    "cusparse_vs_alg1_speedup",
    "base_vs_alg2_speedup",
    "torch_vs_alg2_speedup",
    "cusparse_vs_alg2_speedup",
    "alg1_vs_alg2_speedup",
    "base_vs_torch_err",
    "base_vs_cusparse_err",
    "opt_vs_torch_err",
    "opt_vs_cusparse_err",
    "opt_alg2_vs_torch_err",
    "opt_alg2_vs_cusparse_err",
    "base_status_vs_torch",
    "base_status_vs_cusparse",
    "opt_status_vs_torch",
    "opt_status_vs_cusparse",
    "opt_alg2_status_vs_torch",
    "opt_alg2_status_vs_cusparse",
    "matrix_status",
]


def _resolve_input_paths(input_paths):
    paths = []
    for path in input_paths:
        if os.path.isfile(path) and path.lower().endswith(".mtx"):
            paths.append(os.path.abspath(path))
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    return paths


def _reference_tolerance(dtype):
    if dtype == torch.float32:
        return 1e-4, 1e-2
    return 1e-12, 1e-10


def _status_from_error(error_value):
    if error_value is None:
        return "SKIP"
    return "PASS" if float(error_value) <= 1.0 else "FAIL"


def _error_profile(candidate, reference, dtype):
    if reference is None:
        return {
            "global_err": None,
            "status": "SKIP",
            "row_max_ratio": None,
            "row_worst_col": None,
        }
    atol, rtol = _reference_tolerance(dtype)
    if candidate.numel() == 0:
        row_max_ratio = torch.zeros((candidate.shape[0],), dtype=torch.float64, device=candidate.device)
        row_worst_col = torch.zeros((candidate.shape[0],), dtype=torch.int64, device=candidate.device)
        return {
            "global_err": 0.0,
            "status": "PASS",
            "row_max_ratio": row_max_ratio,
            "row_worst_col": row_worst_col,
        }
    diff = torch.abs(candidate - reference).to(torch.float64)
    denom = (atol + rtol * torch.abs(reference)).to(torch.float64)
    ratio = diff / denom
    row_max_ratio, row_worst_col = torch.max(ratio, dim=1)
    global_err = float(torch.max(row_max_ratio).item()) if row_max_ratio.numel() > 0 else 0.0
    return {
        "global_err": global_err,
        "status": _status_from_error(global_err),
        "row_max_ratio": row_max_ratio,
        "row_worst_col": row_worst_col,
    }


def _timed_spmm_base(data, indices, indptr, B, shape, warmup, iters):
    out = fs.flagsparse_spmm_csr(data, indices, indptr, B, shape)
    _, elapsed = _benchmark(
        lambda: fs.flagsparse_spmm_csr(data, indices, indptr, B, shape),
        warmup,
        iters,
    )
    return out, elapsed


def _timed_spmm_opt(data, indices, indptr, B, shape, warmup, iters):
    prepared = fs.prepare_spmm_csr_opt(data, indices, indptr, shape)
    count = max(1, int(iters))
    runtime_prepared = spmm_csr_mod._build_spmm_csr_opt_runtime_symbolic_triton(
        prepared
    )
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(count):
        runtime_prepared = spmm_csr_mod._build_spmm_csr_opt_runtime_symbolic_triton(
            prepared
        )
    torch.cuda.synchronize()
    preprocess_ms = (time.perf_counter() - t0) * 1000.0 / count

    def op():
        out, _ = spmm_csr_mod._triton_spmm_csr_impl_opt_prepared(
            runtime_prepared, B
        )
        return out

    out = op()
    torch.cuda.synchronize()
    for _ in range(max(0, int(warmup))):
        _ = op()
    torch.cuda.synchronize()

    measured_value = out
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        measured_value = op()
    end.record()
    torch.cuda.synchronize()
    compute_ms = start.elapsed_time(end) / count
    return measured_value, preprocess_ms + compute_ms, preprocess_ms, compute_ms


def _timed_spmm_opt_alg2(data, indices, indptr, B, shape, warmup, iters):
    prepared = fs.prepare_spmm_csr_opt_alg2(data, indices, indptr, shape)
    count = max(1, int(iters))
    opt_buckets = spmm_alg2_mod._build_spmm_opt_alg2_buckets_triton_symbolic(
        prepared.row_lengths,
        prepared.data.dtype,
    )
    prepared.opt_buckets = opt_buckets
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(count):
        opt_buckets = spmm_alg2_mod._build_spmm_opt_alg2_buckets_triton_symbolic(
            prepared.row_lengths,
            prepared.data.dtype,
        )
    prepared.opt_buckets = opt_buckets
    torch.cuda.synchronize()
    preprocess_ms = (time.perf_counter() - t0) * 1000.0 / count

    def op():
        out, meta = spmm_alg2_mod._triton_spmm_csr_impl_opt_alg2_prepared(
            prepared,
            B,
            opt_buckets=opt_buckets,
            return_meta=True,
        )
        return out, meta

    first, meta = op()
    torch.cuda.synchronize()
    for _ in range(max(0, int(warmup))):
        _ = op()
    torch.cuda.synchronize()

    measured_prepared = prepared
    measured_meta = meta
    measured_value = first
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        measured_value, measured_meta = op()
    end.record()
    torch.cuda.synchronize()
    compute_ms = start.elapsed_time(end) / count
    return (
        measured_value,
        preprocess_ms + compute_ms,
        measured_prepared,
        measured_meta,
        preprocess_ms,
        compute_ms,
    )


def _timed_torch_reference(data, indices, indptr, B, shape, dtype, warmup, iters):
    device = data.device
    ref_dtype = torch.float64 if dtype == torch.float32 else dtype
    B_ref = B.to(ref_dtype)
    sparse = torch.sparse_csr_tensor(
        indptr.to(torch.int64),
        indices.to(torch.int64),
        data.to(ref_dtype),
        size=shape,
        device=device,
    )
    ref = torch.sparse.mm(sparse, B_ref).to(dtype)
    timed_value, elapsed = _benchmark(
        lambda: torch.sparse.mm(sparse, B_ref).to(dtype),
        warmup,
        iters,
    )
    return ref, elapsed


def _timed_sparse_backend(data, indices, indptr, B, shape, warmup, iters, enabled):
    backend_name = "hipsparse_ref" if getattr(torch.version, "hip", None) else "cusparse_ref"
    if not enabled:
        return None, None, backend_name, "disabled"
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx
    except Exception as exc:
        return None, None, backend_name, str(exc)

    try:
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
        ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
        ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
        sparse = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
        out_cp, elapsed = _benchmark(lambda: sparse @ B_cp, warmup, iters)
        out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
        return out, elapsed, backend_name, None
    except Exception as exc:
        return None, None, backend_name, str(exc)


def _benchmark(op, warmup, iters):
    out = op()
    torch.cuda.synchronize()
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(max(1, int(iters))):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / max(1, int(iters))


def _build_csr_from_row_lengths(row_lengths, n_cols, dtype, index_dtype, device, pattern="random"):
    data_parts = []
    col_parts = []
    indptr = [0]
    for row, row_nnz in enumerate(row_lengths):
        row_nnz = int(max(0, min(int(row_nnz), n_cols)))
        if row_nnz == 0:
            indptr.append(indptr[-1])
            continue
        if pattern == "banded":
            center = row % max(n_cols, 1)
            offsets = torch.arange(row_nnz, device=device, dtype=torch.int64)
            cols = (center + offsets) % max(n_cols, 1)
        else:
            cols = torch.randperm(n_cols, device=device, dtype=torch.int64)[:row_nnz]
        cols, _ = torch.sort(cols)
        vals = torch.randn(row_nnz, dtype=dtype, device=device)
        data_parts.append(vals)
        col_parts.append(cols.to(index_dtype))
        indptr.append(indptr[-1] + row_nnz)

    if data_parts:
        data = torch.cat(data_parts)
        indices = torch.cat(col_parts)
    else:
        data = torch.empty(0, dtype=dtype, device=device)
        indices = torch.empty(0, dtype=index_dtype, device=device)
    indptr_tensor = torch.tensor(indptr, dtype=torch.int64, device=device)
    return data, indices, indptr_tensor


def _synthetic_row_lengths(case_name, n_rows, n_cols):
    if case_name == "short_uniform":
        return [min(8, n_cols)] * n_rows
    if case_name == "medium_uniform":
        return [min(64, n_cols)] * n_rows
    if case_name == "long_uniform":
        return [min(256, n_cols)] * n_rows
    if case_name == "heavy_tail":
        rows = [min(8, n_cols)] * n_rows
        long_rows = max(1, n_rows // 32)
        for row in range(long_rows):
            rows[row] = min(max(256, n_cols // 2), n_cols)
        return rows
    if case_name == "banded_regular":
        return [min(32, n_cols)] * n_rows
    if case_name == "powerlaw_irregular":
        rows = []
        max_len = max(2, min(n_cols, 512))
        for row in range(n_rows):
            length = int(max(1, round(max_len / math.sqrt(row + 1))))
            rows.append(min(length, n_cols))
        return rows
    raise ValueError(f"Unknown synthetic case: {case_name}")


def build_synthetic_case(case_name, dtype, index_dtype, device):
    if case_name == "short_uniform":
        shape = (4096, 4096)
        pattern = "random"
    elif case_name == "medium_uniform":
        shape = (2048, 4096)
        pattern = "random"
    elif case_name == "long_uniform":
        shape = (1024, 4096)
        pattern = "random"
    elif case_name == "heavy_tail":
        shape = (4096, 8192)
        pattern = "random"
    elif case_name == "banded_regular":
        shape = (4096, 4096)
        pattern = "banded"
    elif case_name == "powerlaw_irregular":
        shape = (8192, 8192)
        pattern = "random"
    else:
        raise ValueError(f"Unknown synthetic case: {case_name}")

    n_rows, n_cols = shape
    row_lengths = _synthetic_row_lengths(case_name, n_rows, n_cols)
    data, indices, indptr = _build_csr_from_row_lengths(
        row_lengths,
        n_cols,
        dtype,
        index_dtype,
        device,
        pattern=pattern,
    )
    return data, indices, indptr, shape


def run_one_case(
    case_name,
    data,
    indices,
    indptr,
    shape,
    dtype,
    index_dtype,
    dense_cols,
    warmup,
    iters,
    seed,
    with_cusparse,
    return_details=False,
):
    device = data.device
    n_rows, n_cols = shape
    B = _seeded_dense_matrix((n_cols, dense_cols), dtype, device, seed)
    base_out, base_ms = _timed_spmm_base(data, indices, indptr, B, shape, warmup, iters)
    opt_out, opt_ms, alg1_preprocess_ms, alg1_compute_ms = _timed_spmm_opt(
        data,
        indices,
        indptr,
        B,
        shape,
        warmup,
        iters,
    )
    (
        alg2_out,
        alg2_ms,
        prepared_alg2,
        alg2_meta,
        alg2_preprocess_ms,
        alg2_compute_ms,
    ) = _timed_spmm_opt_alg2(
        data,
        indices,
        indptr,
        B,
        shape,
        warmup,
        iters,
    )
    torch_ref, torch_ms = _timed_torch_reference(data, indices, indptr, B, shape, dtype, warmup, iters)
    cusparse_ref, cusparse_ms, sparse_backend_name, sparse_backend_reason = _timed_sparse_backend(
        data,
        indices,
        indptr,
        B,
        shape,
        warmup,
        iters,
        with_cusparse,
    )

    base_vs_torch = _error_profile(base_out, torch_ref, dtype)
    opt_vs_torch = _error_profile(opt_out, torch_ref, dtype)
    alg2_vs_torch = _error_profile(alg2_out, torch_ref, dtype)
    base_vs_cusparse = _error_profile(base_out, cusparse_ref, dtype) if cusparse_ref is not None else _error_profile(base_out, None, dtype)
    opt_vs_cusparse = _error_profile(opt_out, cusparse_ref, dtype) if cusparse_ref is not None else _error_profile(opt_out, None, dtype)
    alg2_vs_cusparse = _error_profile(alg2_out, cusparse_ref, dtype) if cusparse_ref is not None else _error_profile(alg2_out, None, dtype)
    row_lengths = (indptr[1:] - indptr[:-1]).to(torch.int64)
    max_row_nnz = int(row_lengths.max().item()) if row_lengths.numel() > 0 else 0

    summary = {
        "matrix": case_name,
        "value_dtype": str(dtype).replace("torch.", ""),
        "dense_cols": dense_cols,
        "seed": seed,
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
        "nnz": int(data.numel()),
        "avg_nnz_per_row": (float(data.numel()) / float(n_rows) if n_rows > 0 else 0.0),
        "max_row_nnz": max_row_nnz,
        "base_ms": base_ms,
        "opt_ms": opt_ms,
        "alg1_ms": opt_ms,
        "alg1_preprocess_ms": alg1_preprocess_ms,
        "alg1_compute_ms": alg1_compute_ms,
        "alg2_ms": alg2_ms,
        "alg2_preprocess_ms": alg2_preprocess_ms,
        "alg2_compute_ms": alg2_compute_ms,
        "torch_ms": torch_ms,
        "cusparse_ms": cusparse_ms,
        "base_vs_alg1_speedup": (
            base_ms / opt_ms if opt_ms is not None and opt_ms > 0 else None
        ),
        "torch_vs_alg1_speedup": (
            torch_ms / opt_ms if torch_ms is not None and opt_ms is not None and opt_ms > 0 else None
        ),
        "cusparse_vs_alg1_speedup": (
            cusparse_ms / opt_ms if cusparse_ms is not None and opt_ms is not None and opt_ms > 0 else None
        ),
        "base_vs_alg2_speedup": (
            base_ms / alg2_ms if alg2_ms is not None and alg2_ms > 0 else None
        ),
        "torch_vs_alg2_speedup": (
            torch_ms / alg2_ms if torch_ms is not None and alg2_ms is not None and alg2_ms > 0 else None
        ),
        "cusparse_vs_alg2_speedup": (
            cusparse_ms / alg2_ms if cusparse_ms is not None and alg2_ms is not None and alg2_ms > 0 else None
        ),
        "alg1_vs_alg2_speedup": (
            opt_ms / alg2_ms if opt_ms is not None and alg2_ms is not None and alg2_ms > 0 else None
        ),
        "base_vs_opt_alg2_speedup": (
            base_ms / alg2_ms if alg2_ms is not None and alg2_ms > 0 else None
        ),
        "torch_vs_opt_alg2_speedup": (
            torch_ms / alg2_ms if torch_ms is not None and alg2_ms is not None and alg2_ms > 0 else None
        ),
        "cusparse_vs_opt_alg2_speedup": (
            cusparse_ms / alg2_ms if cusparse_ms is not None and alg2_ms is not None and alg2_ms > 0 else None
        ),
        "base_vs_torch_err": base_vs_torch["global_err"],
        "base_vs_cusparse_err": base_vs_cusparse["global_err"],
        "opt_vs_torch_err": opt_vs_torch["global_err"],
        "opt_vs_cusparse_err": opt_vs_cusparse["global_err"],
        "opt_alg2_vs_torch_err": alg2_vs_torch["global_err"],
        "opt_alg2_vs_cusparse_err": alg2_vs_cusparse["global_err"],
        "base_status_vs_torch": base_vs_torch["status"],
        "base_status_vs_cusparse": base_vs_cusparse["status"],
        "opt_status_vs_torch": opt_vs_torch["status"],
        "opt_status_vs_cusparse": opt_vs_cusparse["status"],
        "opt_alg2_status_vs_torch": alg2_vs_torch["status"],
        "opt_alg2_status_vs_cusparse": alg2_vs_cusparse["status"],
        "matrix_status": alg2_vs_torch["status"],
    }

    if not return_details:
        return summary
    return {
        "summary": summary,
        "prepared_alg2": prepared_alg2,
        "alg2_meta": alg2_meta,
        "outputs": {
            "base_triton": base_out,
            "opt_triton": opt_out,
            "opt_alg2_triton": alg2_out,
            "torch_ref": torch_ref,
            "cusparse_ref": cusparse_ref,
        },
        "profiles": {
            "base_vs_torch": base_vs_torch,
            "base_vs_cusparse": base_vs_cusparse,
            "opt_vs_torch": opt_vs_torch,
            "opt_vs_cusparse": opt_vs_cusparse,
            "opt_alg2_vs_torch": alg2_vs_torch,
            "opt_alg2_vs_cusparse": alg2_vs_cusparse,
        },
        "sparse_backend_name": sparse_backend_name,
        "sparse_backend_reason": sparse_backend_reason,
    }


def _fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def _speed(value):
    return "N/A" if value is None else f"{value:.2f}x"


def _err(value):
    return "N/A" if value is None else f"{value:.2e}"


def print_row(row):
    name = os.path.basename(row["matrix"])[:27]
    print(
        f"{name:<28} {row['n_rows']:>7} {row['n_cols']:>7} {row['nnz']:>10} {row['dense_cols']:>8}  "
        f"{_fmt(row['base_ms']):>9} {_fmt(row['alg1_ms']):>9} "
        f"{_fmt(row['alg1_preprocess_ms']):>9} {_fmt(row['alg1_compute_ms']):>9} "
        f"{_fmt(row['alg2_ms']):>9} {_fmt(row['alg2_preprocess_ms']):>9} {_fmt(row['alg2_compute_ms']):>9} "
        f"{_fmt(row['torch_ms']):>9} {_fmt(row['cusparse_ms']):>9}  "
        f"{_speed(row['base_vs_alg1_speedup']):>8} "
        f"{_speed(row['base_vs_alg2_speedup']):>8} "
        f"{_speed(row['alg1_vs_alg2_speedup']):>8} "
        f"{_speed(row['torch_vs_alg2_speedup']):>8} "
        f"{_speed(row['cusparse_vs_alg2_speedup']):>8}  "
        f"{_err(row['opt_vs_torch_err']):>10} {_err(row['opt_vs_cusparse_err']):>10} "
        f"{_err(row['opt_alg2_vs_torch_err']):>10} {_err(row['opt_alg2_vs_cusparse_err']):>10} "
        f"{row['matrix_status']:>6}"
    )


def run_batch(paths, dtype, index_dtype, dense_cols, warmup, iters, seed, with_cusparse):
    device = torch.device("cuda")
    results = []
    for path in paths:
        try:
            data, indices, indptr, shape = load_mtx_to_csr_torch(path, dtype=dtype, device=device)
            indices = indices.to(index_dtype)
            result = run_one_case(
                os.path.basename(path),
                data,
                indices,
                indptr,
                shape,
                dtype,
                index_dtype,
                dense_cols,
                warmup,
                iters,
                seed,
                with_cusparse,
                return_details=False,
            )
        except Exception as exc:
            print(f"  ERROR on {os.path.basename(path)}: {exc}")
            continue
        results.append(result)
        print_row(result)
    return results


def run_synthetic(case_names, dtype, index_dtype, dense_cols, warmup, iters, seed, with_cusparse):
    device = torch.device("cuda")
    results = []
    for case_name in case_names:
        data, indices, indptr, shape = build_synthetic_case(case_name, dtype, index_dtype, device)
        result = run_one_case(
            case_name,
            data,
            indices,
            indptr,
            shape,
            dtype,
            index_dtype,
            dense_cols,
            warmup,
            iters,
            seed,
            with_cusparse,
            return_details=False,
        )
        results.append(result)
        print_row(result)
    return results


def _write_csv(csv_path, rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description="Protected CSR SpMM benchmark with alg1 and alg2.")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--synthetic", action="store_true", help="Run built-in synthetic cases")
    parser.add_argument(
        "--synthetic-cases",
        nargs="*",
        default=["short_uniform", "medium_uniform", "long_uniform", "heavy_tail", "banded_regular", "powerlaw_irregular"],
        help="Subset of synthetic cases to run",
    )
    parser.add_argument("--csv", type=str, default=None, metavar="FILE", help="Export summary CSV")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--dense-cols", type=int, default=DEFAULT_DENSE_COLS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--with-cusparse", action="store_true")
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "float64": torch.float64}[args.dtype]
    index_dtype = torch.int32
    print("=" * 220)
    print("FLAGSPARSE SpMM Alg1 / Alg2 Protected Benchmark")
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  |  dtype: {args.dtype}  |  dense_cols: {args.dense_cols}  "
        f"|  with_cusparse: {args.with_cusparse}"
    )
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8}  "
        f"{'Base(ms)':>9} {'Alg1(ms)':>9} {'A1Prep':>9} {'A1Comp':>9} "
        f"{'Alg2(ms)':>9} {'A2Prep':>9} {'A2Comp':>9} {'Torch(ms)':>9} {'CU(ms)':>9}  "
        f"{'B/A1':>8} {'B/A2':>8} {'A1/A2':>8} {'T/A2':>8} {'CU/A2':>8}  "
        f"{'Err(A1/T)':>10} {'Err(A1/CU)':>10} {'Err(A2/T)':>10} {'Err(A2/CU)':>10} {'Status':>6}"
    )
    print("A*Prep columns are CPU wall time; A*Comp columns are CUDA event compute time.")
    print("=" * 220)

    rows = []
    if args.synthetic:
        rows.extend(
            run_synthetic(
                args.synthetic_cases,
                dtype,
                index_dtype,
                args.dense_cols,
                args.warmup,
                args.iters,
                args.seed,
                args.with_cusparse,
            )
        )

    paths = _resolve_input_paths(args.mtx)
    if paths:
        rows.extend(
            run_batch(
                paths,
                dtype,
                index_dtype,
                args.dense_cols,
                args.warmup,
                args.iters,
                args.seed,
                args.with_cusparse,
            )
        )

    if not rows:
        print("No synthetic cases or .mtx inputs selected.")
        return

    print("=" * 220)
    passed = sum(1 for row in rows if row["matrix_status"] == "PASS")
    print(f"Passed: {passed} / {len(rows)}")
    if args.csv:
        _write_csv(args.csv, rows)
        print(f"Wrote {len(rows)} rows to {args.csv}")


if __name__ == "__main__":
    main()
