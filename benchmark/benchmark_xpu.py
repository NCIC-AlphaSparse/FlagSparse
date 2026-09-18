#!/usr/bin/env python3
"""Benchmark FlagSparse kernels against equivalent PyTorch-XPU expressions."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import mmread

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@dataclass(frozen=True)
class Case:
    rows: int
    cols: int
    nnz: int
    dense_cols: int
    dtype_name: str


def _sync(torch) -> None:
    # torch_xmlir presents Kunlunxin XPU through torch.cuda; torch.xpu is the
    # unsupported upstream namespace in this environment.
    torch.cuda.synchronize()


def _measure(torch, fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    _sync(torch)
    samples = []
    for _ in range(iters):
        _sync(torch)
        start = time.perf_counter()
        fn()
        _sync(torch)
        samples.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.median(samples))


def _dtype(torch, name: str):
    value = getattr(torch, name, None)
    if value is None:
        raise ValueError(f"unknown torch dtype: {name}")
    return value


def _csr_inputs(torch, case: Case, device, matrix_path: Path | None = None):
    """Build a deterministic balanced CSR pattern before device upload."""

    if matrix_path is not None:
        matrix = mmread(str(matrix_path)).tocsr()
        matrix.sum_duplicates()
        actual = Case(
            int(matrix.shape[0]), int(matrix.shape[1]), int(matrix.nnz),
            case.dense_cols, case.dtype_name,
        )
        dtype = _dtype(torch, case.dtype_name)
        indptr = torch.as_tensor(np.asarray(matrix.indptr), dtype=torch.int32)
        indices = torch.as_tensor(np.asarray(matrix.indices), dtype=torch.int32)
        counts = indptr[1:].to(torch.int64) - indptr[:-1].to(torch.int64)
        row_ids = torch.repeat_interleave(torch.arange(actual.rows, dtype=torch.int64), counts)
        values = torch.as_tensor(np.asarray(matrix.data), dtype=dtype)
        return values.to(device), indices.to(device), indptr.to(device), row_ids.to(device), actual, matrix_path.name

    if case.nnz < case.rows:
        raise ValueError("nnz must be at least rows so every CSR row has one value")
    dtype = _dtype(torch, case.dtype_name)
    per_row, extra = divmod(case.nnz, case.rows)
    counts = torch.full((case.rows,), per_row, dtype=torch.int64)
    if extra:
        counts[:extra] += 1
    indptr = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0).to(torch.int32)))
    row_ids = torch.repeat_interleave(torch.arange(case.rows, dtype=torch.int64), counts)
    positions = torch.arange(case.nnz, dtype=torch.int64)
    indices = ((positions * 17 + row_ids * 13) % case.cols).to(torch.int32)
    generator = torch.Generator(device="cpu").manual_seed(20260915)
    values = torch.randn(case.nnz, dtype=dtype, generator=generator)
    return values.to(device), indices.to(device), indptr.to(device), row_ids.to(device), case, ""


def _max_abs_error(actual, expected) -> float:
    difference = (actual.detach().float().cpu() - expected.detach().float().cpu()).abs()
    return float(difference.max().item()) if difference.numel() else 0.0


def _matches_tolerance(actual, expected, rtol: float, atol: float) -> bool:
    """Compare on the host because torch_xmlir eager lacks some XPU predicates."""

    actual_host = actual.detach().float().cpu()
    expected_host = expected.detach().float().cpu()
    return bool(
        (actual_host - expected_host).abs().le(atol + rtol * expected_host.abs()).all().item()
    )


def _tolerance(dtype_name: str) -> tuple[float, float]:
    return (1e-2, 1e-2) if dtype_name in {"float16", "bfloat16"} else (1e-4, 1e-4)


def _run_case(torch, fs, op: str, case: Case, warmup: int, iters: int, device_id: int, matrix_path: Path | None = None):
    try:
        import torch_xmlir  # noqa: F401 - installs the XPU torch.cuda shim.
    except ImportError as exc:
        raise RuntimeError("Kunlunxin XPU benchmark requires torch_xmlir") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Kunlunxin XPU benchmark requires an available torch_xmlir XPU")
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")
    dtype = _dtype(torch, case.dtype_name)
    values, indices, indptr, row_ids, case, matrix_name = _csr_inputs(torch, case, device, matrix_path)
    columns = indices.to(torch.int64)
    generator = torch.Generator(device="cpu").manual_seed(20260916)

    if op == "gather":
        dense = torch.randn(case.cols, dtype=dtype, generator=generator).to(device)
        gather_indices = columns[: min(case.nnz, case.cols)]
        fs_out = torch.empty(gather_indices.numel(), dtype=dtype, device=device)
        baseline_out = torch.empty_like(fs_out)
        candidate = lambda: fs.flagsparse_gather(dense, gather_indices, out=fs_out)
        baseline = lambda: torch.index_select(dense, 0, gather_indices, out=baseline_out)
    elif op == "scatter":
        initial = torch.randn(case.cols, dtype=dtype, generator=generator).to(device)
        size = min(case.nnz, case.cols)
        # A unique set makes both implementations a direct index-copy comparison.
        scatter_indices = torch.arange(size, device=device, dtype=torch.int32)
        scatter_values = torch.randn(size, dtype=dtype, generator=generator).to(device)
        fs_out = initial.detach().clone()
        baseline_out = initial.detach().clone()

        def candidate():
            fs.flagsparse_scatter(fs_out, scatter_indices, scatter_values, reset_output=True)
            return fs_out

        def baseline():
            baseline_out.zero_()
            baseline_out.index_copy_(0, scatter_indices.to(torch.int64), scatter_values)
            return baseline_out
    elif op == "spmv_csr":
        x = torch.randn(case.cols, dtype=dtype, generator=generator).to(device)
        fs_out = torch.empty(case.rows, dtype=dtype, device=device)
        baseline_out = torch.empty_like(fs_out)
        candidate = lambda: fs.flagsparse_spmv_csr(values, indices, indptr, x, (case.rows, case.cols), out=fs_out)

        def baseline():
            baseline_out.zero_()
            baseline_out.index_add_(0, row_ids, values * x[columns])
            return baseline_out
    elif op == "spmv_coo":
        x = torch.randn(case.cols, dtype=dtype, generator=generator).to(device)
        fs_out = torch.empty(case.rows, dtype=dtype, device=device)
        baseline_out = torch.empty_like(fs_out)
        candidate = lambda: fs.flagsparse_spmv_coo(
            values, row_ids.to(torch.int32), indices, x, (case.rows, case.cols), out=fs_out
        )

        def baseline():
            baseline_out.zero_()
            baseline_out.index_add_(0, row_ids, values * x[columns])
            return baseline_out
    elif op == "spmm_csr":
        dense = torch.randn(case.cols, case.dense_cols, dtype=dtype, generator=generator).to(device)
        fs_out = torch.empty(case.rows, case.dense_cols, dtype=dtype, device=device)
        baseline_out = torch.empty_like(fs_out)
        candidate = lambda: fs.flagsparse_spmm_csr(values, indices, indptr, dense, (case.rows, case.cols), out=fs_out)

        def baseline():
            baseline_out.zero_()
            baseline_out.index_add_(0, row_ids, values[:, None] * dense[columns])
            return baseline_out
    elif op == "sddmm_csr":
        left = torch.randn(case.rows, case.dense_cols, dtype=dtype, generator=generator).to(device)
        right = torch.randn(case.cols, case.dense_cols, dtype=dtype, generator=generator).to(device)
        fs_out = torch.empty_like(values)
        baseline_out = torch.empty_like(values)
        candidate = lambda: fs.flagsparse_sddmm_csr(values, indices, indptr, left, right, (case.rows, case.cols), out=fs_out)

        def baseline():
            baseline_out.copy_((left[row_ids] * right[columns]).sum(dim=1))
            return baseline_out
    else:
        raise ValueError(f"unsupported XPU baseline operator: {op}")

    # Check the same tensors that will subsequently be timed.  Both closures
    # overwrite their output buffers, so the warmup loops do not alter this result.
    candidate_result = candidate()
    baseline_result = baseline()
    _sync(torch)
    rtol, atol = _tolerance(case.dtype_name)
    max_abs_err = _max_abs_error(candidate_result, baseline_result)
    if not _matches_tolerance(candidate_result, baseline_result, rtol, atol):
        return {
            "dtype": case.dtype_name,
            "shape": f"matrix={matrix_name},m={case.rows},n={case.cols},nnz={case.nnz},k={case.dense_cols}",
            "triton_ms": "",
            "pytorch_ms": "",
            "speedup": "",
            "triton_speedup_vs_pytorch": "",
            "max_abs_err": max_abs_err,
            "status": "MISMATCH",
            "baseline": "pytorch",
            "reason": f"max_abs_err={max_abs_err:.8g} exceeds rtol={rtol:g}, atol={atol:g}",
        }

    triton_ms = _measure(torch, candidate, warmup, iters)
    pytorch_ms = _measure(torch, baseline, warmup, iters)
    speedup = pytorch_ms / triton_ms if triton_ms > 0.0 else ""
    return {
        "dtype": case.dtype_name,
        "shape": f"matrix={matrix_name},m={case.rows},n={case.cols},nnz={case.nnz},k={case.dense_cols}",
        "triton_ms": triton_ms,
        "pytorch_ms": pytorch_ms,
        "speedup": speedup,
        "triton_speedup_vs_pytorch": speedup,
        "max_abs_err": max_abs_err,
        "status": "PASS",
        "baseline": "pytorch",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", required=True, choices=("gather", "scatter", "spmv_csr", "spmv_coo", "spmm_csr", "sddmm_csr"))
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--nnz", type=int, default=131072)
    parser.add_argument("--dense-cols", type=int, default=64)
    parser.add_argument("--dtypes", default="float32", help="Comma-separated torch floating dtypes")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--device", type=int, default=0)
    matrix_source = parser.add_mutually_exclusive_group()
    matrix_source.add_argument("--matrix", type=Path, help="One Matrix Market input")
    matrix_source.add_argument("--matrix-dir", type=Path, help="Directory of Matrix Market (*.mtx) inputs")
    parser.add_argument("--csv-summary", required=True)
    args = parser.parse_args()
    if args.m <= 0 or args.n <= 0 or args.nnz <= 0 or args.dense_cols <= 0:
        parser.error("matrix dimensions, nnz, and dense-cols must be positive")
    if args.warmup < 0 or args.iters <= 0:
        parser.error("warmup must be non-negative and iters must be positive")

    import torch
    import flagsparse as fs

    dtype_names = [value.strip() for value in args.dtypes.split(",") if value.strip()]
    matrix_paths = [None]
    if args.matrix is not None:
        if not args.matrix.is_file():
            parser.error(f"matrix file does not exist: {args.matrix}")
        matrix_paths = [args.matrix]
    elif args.matrix_dir is not None:
        matrix_paths = sorted(args.matrix_dir.glob("*.mtx"))
        if not matrix_paths:
            parser.error(f"no .mtx files in {args.matrix_dir}")
    rows = []
    for dtype_name in dtype_names:
        for matrix_path in matrix_paths:
            try:
                rows.append(_run_case(torch, fs, args.op, Case(args.m, args.n, args.nnz, args.dense_cols, dtype_name), args.warmup, args.iters, args.device, matrix_path))
            except Exception as exc:
                label = matrix_path.name if matrix_path is not None else args.op
                rows.append({"dtype": dtype_name, "shape": label, "triton_ms": "", "pytorch_ms": "", "speedup": "", "triton_speedup_vs_pytorch": "", "max_abs_err": "", "status": "ERROR", "baseline": "pytorch", "reason": str(exc)})

    path = Path(args.csv_summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ("dtype", "shape", "triton_ms", "pytorch_ms", "speedup", "triton_speedup_vs_pytorch", "max_abs_err", "status", "baseline", "reason")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{args.op} {row['dtype']}: {row['status']}")
        if row.get("reason"):
            print(row["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
