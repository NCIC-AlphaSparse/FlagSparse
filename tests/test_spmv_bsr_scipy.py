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

"""Native BSR SpMV benchmark with an optional SciPy CPU BSR baseline."""

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

try:
    import numpy as np
    from scipy.sparse import bsr_matrix as scipy_bsr_matrix
except Exception as exc:  # pragma: no cover - optional diagnostic dependency
    np = None
    scipy_bsr_matrix = None
    SCIPY_IMPORT_ERROR = str(exc)
else:
    SCIPY_IMPORT_ERROR = None

from test_spmv_bsr import (  # noqa: E402
    DEFAULT_BLOCK_DIMS,
    DTYPE_MAP,
    INDEX_DTYPE_MAP,
    INDEX_DTYPES,
    ITERS,
    SUPPORTED_OPS,
    TEST_SIZES,
    VALUE_DTYPES,
    WARMUP,
    _allclose_error_ratio,
    _base_row,
    _dense_to_bsr,
    _dtype_name,
    _error_stats,
    _expand_algs,
    _fmt,
    _fmt_err,
    _logical_out_size,
    _logical_x_size,
    _op_transposes,
    _pad_vector,
    _parse_algs,
    _parse_block_dims,
    _parse_csv_tokens,
    _parse_ops,
    _padded_shape,
    _padded_x_size,
    _print_baseline_notes,
    _random_values,
    _reference_tolerance,
    _resolve_block_dims,
    _sep,
    _spmv_coo_reference,
    _status,
    _time_cusparse,
    _time_flagsparse_bsr,
    _time_pytorch,
    _time_pytorch_padded,
    _entries_to_bsr_torch,
    load_mtx_entries,
)


def _scipy_unavailable_reason():
    if np is None or scipy_bsr_matrix is None:
        return f"NumPy/SciPy is not available: {SCIPY_IMPORT_ERROR}"
    return None


def _time_scipy_bsr_cpu(data, indices, indptr, x, shape, block_dim, op, warmup, iters):
    reason = _scipy_unavailable_reason()
    if reason:
        return None, reason, None
    padded_shape = _padded_shape(shape, block_dim)
    x_len = padded_shape[0] if _op_transposes(op) else padded_shape[1]

    data_np = data.detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy()
    indptr_np = indptr.detach().cpu().numpy()
    x_np = x.detach().cpu().numpy()
    x_padded = np.zeros((x_len,), dtype=x_np.dtype)
    x_padded[: x_np.shape[0]] = x_np
    A = scipy_bsr_matrix((data_np, indices_np, indptr_np), shape=padded_shape)

    if op == "trans":
        fn = lambda: A.T @ x_padded
    elif op == "conj":
        fn = lambda: A.conj().T @ x_padded
    else:
        fn = lambda: A @ x_padded

    out = None
    for _ in range(max(0, int(warmup))):
        out = fn()
    count = max(1, int(iters))
    start = time.perf_counter()
    for _ in range(count):
        out = fn()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / count
    return elapsed_ms, None, out


def _spd_value(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _spd_text(numerator, denominator):
    value = _spd_value(numerator, denominator)
    return "N/A" if value is None else f"{value:.2f}x"


def _print_scipy_notes(run_cusparse=True):
    _print_baseline_notes(run_cusparse=run_cusparse)
    print(
        "SciPy baseline: SciPyCPU(ms) uses scipy.sparse.bsr_matrix on CPU with padded shape; "
        "GPU->CPU transfer and SciPy construction are setup and excluded."
    )
    print(
        "CPU SciPy vs GPU FlagSparse is not a same-device speedup; SciPy/Alg is diagnostic only."
    )
    reason = _scipy_unavailable_reason()
    if reason:
        print(f"SciPy baseline: unavailable ({reason}); SciPyCPU(ms)=N/A.")


def _header(timing=False):
    split = f" {'ProcGPU':>9} {'Compute':>9}" if timing else ""
    return (
        f"{'Matrix':<28} {'Alg':>15} {'Op':>5} {'BDim':>5} {'Ref':>8} "
        f"{'Out':>7} {'PadOut':>7} {'Rows':>7} {'Cols':>7} {'NNZB':>9} "
        f"{'BSR(ms)':>9} {'BSRGPU':>9} {'CPUProc':>9}{split} "
        f"{'SciPyCPU':>9} {'SciPy/Alg':>9} {'SciPyErr':>10} "
        f"{'PT(ms)':>9} {'PTPad':>9} {'CU(ms)':>9} {'BSRErr':>10} {'Status':>6}"
    )


def _print_row(row, timing=False):
    name = str(row["matrix"])[:27]
    if len(str(row["matrix"])) > 27:
        name += "..."
    split = (
        f" {_fmt(row.get('process_gpu_ms')):>9} {_fmt(row.get('compute_ms')):>9}"
        if timing
        else ""
    )
    print(
        f"{name:<28} {row.get('algorithm', 'base')[:15]:>15} {row['op']:>5} "
        f"{row['block_dim']:>5} {row['reference']:>8} {row['out_size']:>7} "
        f"{row['padded_out_size']:>7} {row['n_rows']:>7} {row['n_cols']:>7} "
        f"{row['nnzb']:>9} {_fmt(row['bsr_ms']):>9} {_fmt(row['bsr_gpu_ms']):>9} "
        f"{_fmt(row['process_cpu_ms']):>9}{split} "
        f"{_fmt(row.get('scipy_cpu_ms')):>9} {_spd_text(row.get('scipy_cpu_ms'), row.get('bsr_ms')):>9} "
        f"{_fmt_err(row.get('scipy_cpu_err')):>10} {_fmt(row['pytorch_ms']):>9} "
        f"{_fmt(row['pytorch_padded_ms']):>9} {_fmt(row['cusparse_ms']):>9} "
        f"{_fmt_err(row['err']):>10} {row['status']:>6}"
    )
    for label, key in (
        ("error", "error"),
        ("scipy", "scipy_reason"),
        ("pt", "pytorch_error"),
        ("pt_padded", "pytorch_padded_error"),
        ("cusparse", "cusparse_error"),
    ):
        value = row.get(key)
        if value:
            print(f"  {label}: {str(value)[:240]}")


def _run_one_case(
    data,
    indices,
    indptr,
    shape,
    dtype,
    index_dtype,
    op,
    alg,
    matrix_name,
    block_dim,
    warmup,
    iters,
    timing=False,
    run_cusparse=True,
    logical_nnz=None,
):
    data = data.contiguous()
    indices = indices.to(index_dtype).contiguous()
    indptr = indptr.to(index_dtype).contiguous()
    row = _base_row(
        matrix_name,
        dtype,
        index_dtype,
        op,
        alg,
        shape,
        data,
        block_dim,
        logical_nnz=logical_nnz,
    )
    row.update(
        {
            "scipy_cpu_ms": None,
            "scipy_cpu_err": None,
            "scipy_vs_alg_speedup": None,
            "scipy_reason": None,
        }
    )
    row["process_gpu_ms"] = 0.0 if timing else None
    if op not in SUPPORTED_OPS:
        row["status"] = "SKIP"
        row["error"] = "unsupported BSR SpMV op"
        return row

    x = _random_values((_logical_x_size(shape, op),), dtype, data.device)
    atol, rtol = _reference_tolerance(dtype)
    try:
        bsr = _time_flagsparse_bsr(
            data, indices, indptr, x, shape, block_dim, op, alg, warmup, iters, timing=timing
        )
    except Exception as exc:
        row["error"] = f"flagsparse_spmv_bsr failed: {exc}"
        return row
    row.update(
        {
            "bsr_ms": bsr["ms"],
            "bsr_gpu_ms": bsr["gpu_ms"],
            "process_cpu_ms": bsr["process_cpu_ms"],
            "process_gpu_ms": bsr["process_gpu_ms"],
            "compute_ms": bsr["compute_ms"],
        }
    )

    try:
        y_ref = _spmv_coo_reference(data, indices, indptr, x, shape, dtype, block_dim, op)
        logical_out = _logical_out_size(shape, op)
        y_bsr = bsr["out"][:logical_out]
        stats = _error_stats(y_bsr, y_ref, atol, rtol)
        err = stats["ratio"]
        row.update(
            {
                "padded_out_size": int(bsr["out"].numel()),
                "out_size": logical_out,
                "pad_rows": max(0, int(bsr["out"].numel()) - int(logical_out)),
                "err": err,
                "bsr_err": err,
                "max_abs_err": stats["max_abs"],
                "max_rel_err": stats["max_rel"],
                "max_err_index": stats["index"],
                "actual_at_max": stats["actual"],
                "expected_at_max": stats["expected"],
            }
        )
    except Exception as exc:
        row["error"] = f"reference failed after BSR run: {exc}"
        return row

    try:
        scipy_ms, scipy_reason, scipy_out = _time_scipy_bsr_cpu(
            data, indices, indptr, x, shape, block_dim, op, warmup, iters
        )
        row["scipy_cpu_ms"] = scipy_ms
        row["scipy_reason"] = scipy_reason
        if scipy_out is not None:
            scipy_logical = torch.as_tensor(
                scipy_out[:logical_out], dtype=dtype, device=y_ref.device
            )
            row["scipy_cpu_err"] = _allclose_error_ratio(scipy_logical, y_ref, atol, rtol)
            row["scipy_vs_alg_speedup"] = _spd_value(scipy_ms, row["bsr_ms"])
    except Exception as exc:
        row["scipy_reason"] = str(exc)

    pytorch_out = None
    try:
        row["pytorch_ms"], row["pytorch_error"], pytorch_out = _time_pytorch(
            data, indices, indptr, x, shape, block_dim, op, warmup, iters
        )
        if pytorch_out is not None:
            row["pytorch_err"] = _allclose_error_ratio(pytorch_out, y_ref, atol, rtol)
            row["bsr_vs_pytorch_err"] = _allclose_error_ratio(
                y_bsr, pytorch_out, atol, rtol
            )
    except Exception as exc:
        row["pytorch_error"] = str(exc)

    padded_shape = _padded_shape(shape, block_dim)
    if (
        pytorch_out is not None
        and int(padded_shape[0]) == int(shape[0])
        and int(padded_shape[1]) == int(shape[1])
    ):
        row["pytorch_padded_ms"] = row["pytorch_ms"]
        row["pytorch_padded_error"] = None
        row["pytorch_padded_err"] = row["pytorch_err"]
        row["bsr_vs_pytorch_padded_err"] = row["bsr_vs_pytorch_err"]
        row["pytorch_padded_mode"] = "same_as_pt"
    else:
        try:
            (
                row["pytorch_padded_ms"],
                row["pytorch_padded_error"],
                pytorch_padded_out,
            ) = _time_pytorch_padded(
                data, indices, indptr, x, shape, block_dim, op, warmup, iters
            )
            row["pytorch_padded_mode"] = "padded_shape"
            if pytorch_padded_out is not None:
                pytorch_padded_logical = pytorch_padded_out[:logical_out]
                row["pytorch_padded_err"] = _allclose_error_ratio(
                    pytorch_padded_logical, y_ref, atol, rtol
                )
                row["bsr_vs_pytorch_padded_err"] = _allclose_error_ratio(
                    y_bsr, pytorch_padded_logical, atol, rtol
                )
        except Exception as exc:
            row["pytorch_padded_error"] = str(exc)
            row["pytorch_padded_mode"] = "padded_shape"

    if run_cusparse:
        try:
            row["cusparse_ms"], row["cusparse_error"] = _time_cusparse(
                data, indices, indptr, x, shape, block_dim, op, warmup, iters
            )
        except Exception as exc:
            row["cusparse_error"] = str(exc)

    ok = (not math.isnan(err)) and err <= 1.0
    row["status"] = _status(ok)
    row["error"] = None if ok else "correctness check failed"
    return row


def _write_csv(rows, csv_path, timing):
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
        "algorithm",
        "reference",
        "block_dim",
        "out_size",
        "padded_out_size",
        "pad_rows",
        "n_rows",
        "n_cols",
        "nnzb",
        "logical_nnz",
        "stored_nnz",
        "padding_ratio",
        "bsr_ms",
        "bsr_gpu_ms",
        "process_cpu_ms",
        "process_gpu_ms",
        "compute_ms",
        "scipy_cpu_ms",
        "scipy_cpu_err",
        "scipy_vs_alg_speedup",
        "scipy_reason",
        "pytorch_ms",
        "pytorch_error",
        "pytorch_err",
        "pytorch_padded_ms",
        "pytorch_padded_error",
        "pytorch_padded_err",
        "pytorch_padded_mode",
        "bsr_vs_pytorch_err",
        "bsr_vs_pytorch_padded_err",
        "max_abs_err",
        "max_rel_err",
        "max_err_index",
        "actual_at_max",
        "expected_at_max",
        "cusparse_ms",
        "cusparse_error",
        "bsr_err",
        "err",
        "status",
        "error",
    ]
    if not timing:
        fieldnames = [
            field for field in fieldnames if field not in ("process_gpu_ms", "compute_ms")
        ]
    csv_parent = Path(csv_path).parent
    if str(csv_parent) not in ("", "."):
        csv_parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    print(f"Wrote {len(rows)} rows to {csv_path}")


def _error_row(matrix_name, dtype, index_dtype, op, block_dim, error):
    block_dim_value = int(block_dim if block_dim != "auto" else 4)
    return {
        "matrix": matrix_name,
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "algorithm": "base",
        "reference": "spmv-coo",
        "block_dim": block_dim_value,
        "out_size": "ERR",
        "padded_out_size": "ERR",
        "pad_rows": "ERR",
        "n_rows": "ERR",
        "n_cols": "ERR",
        "nnzb": "ERR",
        "logical_nnz": "ERR",
        "stored_nnz": "ERR",
        "padding_ratio": "ERR",
        "bsr_ms": None,
        "bsr_gpu_ms": None,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": None,
        "compute_ms": None,
        "scipy_cpu_ms": None,
        "scipy_cpu_err": None,
        "scipy_vs_alg_speedup": None,
        "scipy_reason": None,
        "pytorch_ms": None,
        "pytorch_error": None,
        "pytorch_err": None,
        "pytorch_padded_ms": None,
        "pytorch_padded_error": None,
        "pytorch_padded_err": None,
        "pytorch_padded_mode": None,
        "bsr_vs_pytorch_err": None,
        "bsr_vs_pytorch_padded_err": None,
        "max_abs_err": None,
        "max_rel_err": None,
        "max_err_index": None,
        "actual_at_max": None,
        "expected_at_max": None,
        "cusparse_ms": None,
        "cusparse_error": None,
        "bsr_err": None,
        "err": None,
        "status": "ERROR",
        "error": str(error),
    }


def run_synthetic(
    value_dtypes=None,
    index_dtypes=None,
    block_dims=None,
    ops=None,
    algs=None,
    warmup=WARMUP,
    iters=ITERS,
    timing=False,
    run_cusparse=True,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    block_dims = list(DEFAULT_BLOCK_DIMS) if block_dims is None else block_dims
    ops = SUPPORTED_OPS if ops is None else ops
    algs = ["base"] if algs is None else algs
    print("=" * 140)
    print("FLAGSPARSE SpMV BSR BENCHMARK WITH SCIPY CPU BSR BASELINE")
    print("=" * 140)
    print("Timing policy: bsr_ms = process_cpu_ms + bsr_gpu_ms; SciPy construction is setup.")
    _print_scipy_notes(run_cusparse=run_cusparse)
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for block_dim in block_dims:
                block_dim = 4 if block_dim == "auto" else int(block_dim)
                for op in ops:
                    op_algs = _expand_algs(algs, op)
                    print(_sep(timing))
                    print(
                        f"dtype: {_dtype_name(dtype)} | index_dtype: {_dtype_name(index_dtype)} | "
                        f"block_dim: {block_dim} | op: {op} | alg: {','.join(op_algs)}"
                    )
                    print(_sep(timing))
                    print(_header(timing))
                    print(_sep(timing))
                    for m, n in TEST_SIZES:
                        dense = _random_values((m, n), dtype, device)
                        dense *= (torch.rand(m, n, device=device) < 0.1).to(dtype=dtype)
                        logical_nnz = int(torch.count_nonzero(dense).item())
                        data, indices, indptr = _dense_to_bsr(dense, index_dtype, block_dim)
                        for alg in op_algs:
                            row = _run_one_case(
                                data,
                                indices,
                                indptr,
                                (m, n),
                                dtype,
                                index_dtype,
                                op,
                                alg,
                                f"{m}x{n}",
                                block_dim,
                                warmup,
                                iters,
                                timing=timing,
                                run_cusparse=run_cusparse,
                                logical_nnz=logical_nnz,
                            )
                            _print_row(row, timing=timing)
                    print(_sep(timing))
                    print()


def run_csv(
    mtx_paths,
    csv_path,
    value_dtypes=None,
    index_dtypes=None,
    block_dims=None,
    ops=None,
    algs=None,
    warmup=WARMUP,
    iters=ITERS,
    timing=False,
    run_cusparse=True,
    fail_fast=False,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    block_dims = list(DEFAULT_BLOCK_DIMS) if block_dims is None else block_dims
    ops = SUPPORTED_OPS if ops is None else ops
    algs = ["base"] if algs is None else algs
    rows = []
    _print_scipy_notes(run_cusparse=run_cusparse)
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                op_algs = _expand_algs(algs, op)
                print(_sep(timing))
                print(
                    f"Value dtype: {_dtype_name(dtype)} | Index dtype: {_dtype_name(index_dtype)} | "
                    f"op: {op} | alg: {','.join(op_algs)}"
                )
                print(_sep(timing))
                print(_header(timing))
                print(_sep(timing))
                for path in mtx_paths:
                    try:
                        entries, shape = load_mtx_entries(path)
                        resolved_block_dims = _resolve_block_dims(block_dims, entries, shape)
                        for block_dim in resolved_block_dims:
                            data, indices, indptr = _entries_to_bsr_torch(
                                entries, shape, dtype, index_dtype, int(block_dim), device
                            )
                            for alg in op_algs:
                                row = _run_one_case(
                                    data,
                                    indices,
                                    indptr,
                                    shape,
                                    dtype,
                                    index_dtype,
                                    op,
                                    alg,
                                    os.path.basename(path),
                                    int(block_dim),
                                    warmup,
                                    iters,
                                    timing=timing,
                                    run_cusparse=run_cusparse,
                                    logical_nnz=len(entries),
                                )
                                if fail_fast and row.get("status") == "ERROR":
                                    raise RuntimeError(row.get("error") or "BSR SpMV case failed")
                                rows.append(row)
                                _print_row(row, timing=timing)
                    except Exception as exc:
                        if fail_fast:
                            raise
                        row = _error_row(
                            os.path.basename(path),
                            dtype,
                            index_dtype,
                            op,
                            block_dims[0],
                            exc,
                        )
                        rows.append(row)
                        _print_row(row, timing=timing)
                print(_sep(timing))
    _write_csv(rows, csv_path, timing)


def main():
    parser = argparse.ArgumentParser(
        description="Native BSR SpMV benchmark/test with optional SciPy CPU BSR baseline."
    )
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv-bsr", type=str, default=None, metavar="FILE")
    parser.add_argument("--dtypes", default="float32,float64,complex64,complex128")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--block-dims", default="4")
    parser.add_argument("--ops", default="non")
    parser.add_argument(
        "--alg",
        default="base",
        help="BSR algorithm: base, blockrow_reduce, auto, compare, or comma-separated values",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    try:
        value_dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
        index_dtypes = _parse_csv_tokens(args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes")
        block_dims = _parse_block_dims(args.block_dims)
        ops = _parse_ops(args.ops)
        algs = _parse_algs(args.alg)
    except ValueError as exc:
        parser.error(str(exc))

    if args.synthetic:
        run_synthetic(
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            block_dims=block_dims,
            ops=ops,
            algs=algs,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
            run_cusparse=not args.no_cusparse,
        )
        return

    paths = []
    for path in args.mtx:
        if os.path.isfile(path) and path.endswith(".mtx"):
            paths.append(path)
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    if args.csv_bsr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-bsr")
            return
        run_csv(
            paths,
            args.csv_bsr,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            block_dims=block_dims,
            ops=ops,
            algs=algs,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
            run_cusparse=not args.no_cusparse,
            fail_fast=args.fail_fast,
        )
        return
    if not paths:
        print("No .mtx files. Use --synthetic or --csv-bsr with inputs.")
        return
    run_csv(
        paths,
        "test_spmv_bsr_scipy.csv",
        value_dtypes=value_dtypes,
        index_dtypes=index_dtypes,
        block_dims=block_dims,
        ops=ops,
        algs=algs,
        warmup=args.warmup,
        iters=args.iters,
        timing=args.timing,
        run_cusparse=not args.no_cusparse,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
