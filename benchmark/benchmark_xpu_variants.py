#!/usr/bin/env python3
"""Run the 40 cwapper delivery variants on Kunlunxin XPU.

Rows which cannot run on this runtime remain explicit in the report.  In
particular, torch_xmlir's eager CUDA shim downcasts f64/c128 to f32, so those
rows are not presented as f64 performance measurements.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DTYPE_TAGS = {
    "f16": "float16",
    "f32": "float32",
    "c32": "complex64",
}

# Each entry has been independently launched on the current P800/XPU runtime.
MEASURED = {
    ("gather", "f16"), ("gather", "f32"), ("gather", "c32"),
    ("scatter", "f16"), ("scatter", "f32"), ("scatter", "c32"),
    ("spmv_csr", "f32"),
    ("spmv_coo", "f32"), ("spmv_coo", "c32"),
    ("spmm_csr", "f32"), ("spmm_csr", "c32"),
    ("sddmm_csr", "f32"),
}

COMPILE_FAILURES = {"spgemm_csr", "spsv_csr", "spsv_coo", "spsm_csr"}


def _variants(manifest: Path):
    document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    for op in document["operators"]:
        if op.get("status") != "implemented" or op.get("reporting") != "delivery":
            continue
        limited = set(op.get("delivery_dtypes") or op.get("dtypes") or ())
        formats = op.get("formats") or ()
        for fmt in formats:
            for source_dtype in op.get("dtypes") or ():
                if source_dtype not in limited:
                    continue
                tag = {"c64": "c32", "c128": "c64"}.get(source_dtype, source_dtype)
                yield op["id"], fmt, tag


def _blocked_status(op: str, dtype: str) -> tuple[str, str]:
    if dtype in {"f64", "c64"}:
        return (
            "UNSUPPORTED_DTYPE",
            "torch_xmlir eager XPU converts requested f64/c128 tensors to f32/complex64",
        )
    if op == "spmv_csr" and dtype == "c32":
        return "TRITON_COMPILE", "XPU UnrollControl rejects the complex CSR SpMV kernel"
    if op == "spmm_coo":
        return "RUNTIME_ERROR", "XPU runtime reports invalid device function for COO SpMM"
    if op in COMPILE_FAILURES:
        return "TRITON_COMPILE", "current Triton XPU lowering rejects this kernel's control-flow/atomic IR"
    return "ERROR", "no XPU execution route is registered for this delivery variant"


def _measure_matrix(
    op: str,
    dtype: str,
    device: int,
    warmup: int,
    iters: int,
    matrix_path: Path,
    timeout_seconds: int,
) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="flagsparse_xpu_variant_") as temp:
        detail = Path(temp) / "detail.csv"
        command = [
            sys.executable, str(ROOT / "benchmark" / "benchmark_xpu.py"),
            "--op", op, "--dtypes", DTYPE_TAGS[dtype], "--device", str(device),
            "--warmup", str(warmup), "--iters", str(iters), "--matrix", str(matrix_path),
            "--csv-summary", str(detail),
        ]
        try:
            process = subprocess.run(
                command, cwd=ROOT, text=True, capture_output=True, check=False, timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "RUNTIME_ERROR", "shape": f"matrix={matrix_path.name}", "triton_ms": "",
                "reason": f"execution exceeded {timeout_seconds}s timeout",
            }
        if not detail.exists():
            return {
                "status": "RUNTIME_ERROR", "shape": f"matrix={matrix_path.name}", "triton_ms": "",
                "reason": (process.stderr or process.stdout).strip(),
            }
        with detail.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            return {
                "status": "RUNTIME_ERROR", "shape": f"matrix={matrix_path.name}", "triton_ms": "",
                "reason": "benchmark wrote no result row",
            }
        row = rows[0]
        if process.returncode:
            if not row.get("reason"):
                row["reason"] = (process.stderr or process.stdout).strip()
        return row


def _measure_batch(
    op: str, dtype: str, device: int, warmup: int, iters: int, matrix_dir: Path,
) -> list[dict[str, str]]:
    """Run a stable variant over all inputs in one process to amortize XPU startup."""
    with tempfile.TemporaryDirectory(prefix="flagsparse_xpu_variant_") as temp:
        detail = Path(temp) / "detail.csv"
        command = [
            sys.executable, str(ROOT / "benchmark" / "benchmark_xpu.py"),
            "--op", op, "--dtypes", DTYPE_TAGS[dtype], "--device", str(device),
            "--warmup", str(warmup), "--iters", str(iters), "--matrix-dir", str(matrix_dir),
            "--csv-summary", str(detail),
        ]
        process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        if not detail.exists():
            return [{
                "status": "RUNTIME_ERROR", "shape": "", "triton_ms": "",
                "reason": (process.stderr or process.stdout).strip(),
            }]
        with detail.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if process.returncode:
            for row in rows:
                if not row.get("reason"):
                    row["reason"] = (process.stderr or process.stdout).strip()
        return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT.parent / "cwapper/conf/operators.yaml")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--matrix-dir", type=Path, default=ROOT.parent / "matrix")
    parser.add_argument("--matrix-timeout", type=int, default=60, help="Per-matrix process timeout in seconds")
    parser.add_argument("--matrix-csv", type=Path, help="Per-matrix detail CSV; default is beside --csv-summary")
    parser.add_argument("--csv-summary", type=Path, required=True)
    args = parser.parse_args()

    if args.matrix_timeout <= 0:
        parser.error("matrix timeout must be positive")
    matrix_paths = sorted(args.matrix_dir.glob("*.mtx"))
    if not matrix_paths:
        parser.error(f"no .mtx files in {args.matrix_dir}")

    rows = []
    matrix_rows = []
    matrix_csv = args.matrix_csv or args.csv_summary.with_name(args.csv_summary.stem + "_matrices.csv")
    for op, fmt, dtype in _variants(args.manifest):
        if (op, dtype) in MEASURED:
            if op == "sddmm_csr":
                measured = [
                    _measure_matrix(
                        op, dtype, args.device, args.warmup, args.iters, matrix_path, args.matrix_timeout,
                    )
                    for matrix_path in matrix_paths
                ]
            else:
                measured = _measure_batch(op, dtype, args.device, args.warmup, args.iters, args.matrix_dir)
            for item in measured:
                matrix_rows.append({
                    "operator": op, "format": fmt, "dtype": dtype, "device": args.device,
                    "status": item.get("status", "ERROR"), "shape": item.get("shape", ""),
                    "triton_ms": item.get("triton_ms", ""), "baseline": "", "speedup": "",
                    "reason": item.get("reason", ""),
                })
            statuses = {item["status"] for item in measured}
            status = "PASS" if statuses == {"PASS"} else ("MIXED" if "PASS" in statuses else next(iter(statuses)))
            timings = [float(item["triton_ms"]) for item in measured if item.get("status") == "PASS" and item.get("triton_ms")]
            latency = statistics.median(timings) if timings else ""
            shape = "30 Matrix Market inputs"
            reason = "" if status == "PASS" else "one or more matrix executions failed; see matrix detail CSV"
            matrix_count = sum(item.get("status") == "PASS" for item in measured)
        else:
            status, reason = _blocked_status(op, dtype)
            latency = shape = ""
            matrix_count = 0
        rows.append({
            "operator": op, "format": fmt, "dtype": dtype, "device": args.device,
            "status": status, "matrix_count": matrix_count, "shape": shape, "triton_ms": latency, "baseline": "", "speedup": "", "reason": reason,
        })

    args.csv_summary.parent.mkdir(parents=True, exist_ok=True)
    fields = ("operator", "format", "dtype", "device", "status", "matrix_count", "shape", "triton_ms", "baseline", "speedup", "reason")
    with args.csv_summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if matrix_rows:
        matrix_csv.parent.mkdir(parents=True, exist_ok=True)
        with matrix_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields[0:5] + fields[6:])
            writer.writeheader()
            writer.writerows(matrix_rows)
    for row in rows:
        print(f"{row['operator']} {row['format']} {row['dtype']}: {row['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
