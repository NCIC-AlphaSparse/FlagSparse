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

"""Native CSC SpMV benchmark and correctness script."""

import argparse
import csv
import glob
import math
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
from flagsparse.sparse_operations import _common as ast_ops

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except ImportError:
    cp = None
    cpx_sparse = None


VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = (torch.int32, torch.int64)
OPS = ("non", "trans", "conj")
TEST_SIZES = ((64, 96), (160, 1024), (128, 256))
WARMUP = 10
ITERS = 50


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {"int32": torch.int32, "int64": torch.int64}


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
    token = "non,trans,conj" if value is None else str(value).strip().lower()
    if token == "all":
        return list(OPS)
    ops = [item.strip().lower() for item in token.split(",") if item.strip()]
    invalid = [op for op in ops if op not in OPS]
    if not ops or invalid:
        raise ValueError(f"unsupported --ops: {', '.join(invalid or ops)}")
    return ops


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


def _op_transposes(op):
    return op in ("trans", "conj")


def _x_size_for_op(shape, op):
    return int(shape[0]) if _op_transposes(op) else int(shape[1])


def _out_size_for_op(shape, op):
    return int(shape[1]) if _op_transposes(op) else int(shape[0])


def _dense_to_csc(dense, index_dtype):
    rows, cols = dense.nonzero(as_tuple=True)
    if rows.numel() == 0:
        data = dense.new_empty((0,))
        indices = torch.empty(0, dtype=index_dtype, device=dense.device)
        indptr = torch.zeros(
            int(dense.shape[1]) + 1, dtype=index_dtype, device=dense.device
        )
        return data, indices, indptr
    order = torch.argsort(cols * max(1, int(dense.shape[0])) + rows)
    rows = rows[order]
    cols = cols[order]
    data = dense[rows, cols].contiguous()
    col_counts = torch.bincount(cols, minlength=int(dense.shape[1]))
    indptr = torch.zeros(
        int(dense.shape[1]) + 1, dtype=torch.int64, device=dense.device
    )
    indptr[1:] = torch.cumsum(col_counts, dim=0)
    return data, rows.to(index_dtype).contiguous(), indptr.to(index_dtype)


def _csc_col_indices(indptr):
    counts = indptr[1:].to(torch.int64) - indptr[:-1].to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(indptr.numel() - 1, dtype=torch.int64, device=indptr.device),
        counts,
    )


def _csc_to_torch_coo(data, indices, indptr, shape):
    cols = _csc_col_indices(indptr)
    row = indices.to(torch.int64)
    return torch.sparse_coo_tensor(
        torch.stack([row, cols]),
        data,
        size=shape,
        device=data.device,
        dtype=data.dtype,
    ).coalesce()


def _pytorch_reference(data, indices, indptr, x, shape, dtype, op):
    ref_dtype = _reference_dtype(dtype)
    A = _csc_to_torch_coo(
        data.to(ref_dtype),
        indices,
        indptr,
        shape,
    )
    x_ref = x.to(ref_dtype)
    if op == "non":
        out = torch.sparse.mm(A, x_ref.unsqueeze(1)).squeeze(1)
    elif op == "trans":
        out = torch.sparse.mm(A.transpose(0, 1), x_ref.unsqueeze(1)).squeeze(1)
    elif op == "conj":
        out = torch.sparse.mm(A.conj().transpose(0, 1), x_ref.unsqueeze(1)).squeeze(1)
    else:
        raise ValueError(f"unsupported op: {op}")
    return out.to(dtype)


def _allclose_error_ratio(actual, expected, atol, rtol):
    if expected.numel() == 0:
        return 0.0
    diff = torch.abs(actual - expected).to(torch.float64)
    denom = atol + rtol * torch.abs(expected).to(torch.float64)
    return float(torch.max(diff / denom).item())


def _cuda_event_benchmark(op, warmup, iters):
    out = None
    count = max(1, int(iters))
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(count):
        out = op()
    end.record()
    torch.cuda.synchronize()
    return out, start.elapsed_time(end) / count


def _time_flagsparse_csc(
    data, indices, indptr, x, shape, op, warmup, iters, timing=False
):
    prepared = fs.prepare_spmv_csc(data, indices, indptr, shape, op=op)
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmv_csc(x=x, prepared=prepared),
        warmup,
        iters,
    )
    return {
        "out": out,
        "ms": gpu_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": gpu_ms if timing else None,
    }


def _time_pytorch(data, indices, indptr, x, shape, op, warmup, iters):
    A = _csc_to_torch_coo(data, indices, indptr, shape)
    if op == "non":
        fn = lambda: torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
    elif op == "trans":
        At = A.transpose(0, 1)
        fn = lambda: torch.sparse.mm(At, x.unsqueeze(1)).squeeze(1)
    else:
        AH = A.conj().transpose(0, 1)
        fn = lambda: torch.sparse.mm(AH, x.unsqueeze(1)).squeeze(1)
    _, ms = _cuda_event_benchmark(fn, warmup, iters)
    return ms


def _time_cusparse(data, indices, indptr, x, shape, op, warmup, iters):
    backend, _ = ast_ops._spmv_csc_sparse_ref_backend(data.dtype, indices.dtype, op=op)
    if backend == "hipsparse":
        # DCU/ROCm: hipSPARSE replaces cuSPARSE as the vendor baseline, via
        # hipsparseCreateCsc + the generic SpMV.  Without this branch the DCU
        # run reports no vendor baseline at all and the CSC timings have
        # nothing to be compared against.
        ref = ast_ops._benchmark_spmv_csc_sparse_ref(
            data, indices, indptr, x, shape, warmup, iters, op=op
        )
        return ref["ms"]
    if cp is None or cpx_sparse is None:
        return None
    if data.dtype not in (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ):
        return None
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.to(torch.int64)))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    A = cpx_sparse.csc_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    if op == "non":
        fn = lambda: A @ x_cp
    elif op == "trans":
        fn = lambda: A.T @ x_cp
    else:
        fn = lambda: A.conj().T @ x_cp
    for _ in range(max(0, int(warmup))):
        _ = fn()
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    count = max(1, int(iters))
    start.record()
    for _ in range(count):
        _ = fn()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / count


def _fmt(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _spd(base, other):
    if base is None or other is None or other <= 0:
        return "N/A"
    return f"{base / other:.2f}x"


def _status(ok):
    return "PASS" if ok else "FAIL"


def _header(timing=False):
    split = f" {'ProcGPU':>9} {'Compute':>9}" if timing else ""
    return (
        f"{'Matrix':<28} {'Op':>5} {'Out':>7} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10}  "
        f"{'CSC(ms)':>9} {'CSCGPU':>9} {'CPUProc':>9}{split} "
        f"{'PT(ms)':>9} {'CU(ms)':>9}  {'CSC/PT':>8} {'CSC/CU':>8} "
        f"{'Err':>10} {'Status':>6}"
    )


def _sep(timing=False):
    return "-" * (150 if timing else 130)


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
        f"{name:<28} {row['op']:>5} {row['out_size']:>7} {row['n_rows']:>7} {row['n_cols']:>7} {row['nnz']:>10}  "
        f"{_fmt(row['csc_ms']):>9} {_fmt(row['csc_gpu_ms']):>9} {_fmt(row['process_cpu_ms']):>9}{split} "
        f"{_fmt(row['pytorch_ms']):>9} {_fmt(row['cusparse_ms']):>9}  "
        f"{_spd(row['pytorch_ms'], row['csc_ms']):>8} {_spd(row['cusparse_ms'], row['csc_ms']):>8} "
        f"{_fmt_err(row['err']):>10} {row['status']:>6}"
    )
    error = row.get("error")
    if error:
        print(f"  error: {str(error)[:240]}")


def _run_one_case(
    data,
    indices,
    indptr,
    shape,
    dtype,
    index_dtype,
    op,
    matrix_name,
    warmup,
    iters,
    timing=False,
    run_cusparse=True,
):
    indices = indices.to(index_dtype).contiguous()
    indptr = indptr.to(index_dtype).contiguous()
    x = _random_values((_x_size_for_op(shape, op),), dtype, data.device)
    atol, rtol = _reference_tolerance(dtype)
    base_row = {
        "matrix": matrix_name,
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "out_size": _out_size_for_op(shape, op),
        "n_rows": int(shape[0]),
        "n_cols": int(shape[1]),
        "nnz": int(data.numel()),
        "csc_ms": None,
        "csc_gpu_ms": None,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": None,
        "pytorch_ms": None,
        "cusparse_ms": None,
        "err": None,
        "status": "ERROR",
        "error": None,
    }
    try:
        csc = _time_flagsparse_csc(
            data, indices, indptr, x, shape, op, warmup, iters, timing=timing
        )
    except Exception as exc:
        base_row["error"] = f"flagsparse_spmv_csc failed: {exc}"
        return base_row
    base_row.update(
        {
            "csc_ms": csc["ms"],
            "csc_gpu_ms": csc["gpu_ms"],
            "process_cpu_ms": csc["process_cpu_ms"],
            "process_gpu_ms": csc["process_gpu_ms"],
            "compute_ms": csc["compute_ms"],
        }
    )
    try:
        y_ref = _pytorch_reference(data, indices, indptr, x, shape, dtype, op)
        err = _allclose_error_ratio(csc["out"], y_ref, atol, rtol)
    except Exception as exc:
        base_row["error"] = f"reference failed after CSC run: {exc}"
        return base_row
    pt_ms = None
    cu_ms = None
    try:
        pt_ms = _time_pytorch(data, indices, indptr, x, shape, op, warmup, iters)
    except Exception:
        pass
    if run_cusparse:
        try:
            cu_ms = _time_cusparse(data, indices, indptr, x, shape, op, warmup, iters)
        except Exception:
            pass
    ok = (not math.isnan(err)) and err <= 1.0
    base_row.update(
        {
            "pytorch_ms": pt_ms,
            "cusparse_ms": cu_ms,
            "err": err,
            "status": _status(ok),
            "error": None if ok else "correctness check failed",
        }
    )
    return base_row


def _mtx_value_for_dtype(raw_value, dtype):
    if dtype in (torch.complex64, torch.complex128):
        return complex(raw_value)
    return float(raw_value.real if isinstance(raw_value, complex) else raw_value)


def load_mtx_to_csc_torch(path, dtype=torch.float32, device=None):
    """Load a .mtx into CSC torch tensors via the C-accelerated scipy reader
    (see tests/mtx_fast.py); the former pure-Python parser took minutes on
    large SuiteSparse matrices. Returns (data, row_indices, col_indptr, shape)."""
    from mtx_fast import load_csc

    data, indices, indptr, shape = load_csc(path, dtype=dtype, device=device)
    return data, indices, indptr, shape


def run_synthetic(
    value_dtypes=None,
    index_dtypes=None,
    ops=None,
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
    ops = OPS if ops is None else ops
    print("=" * 140)
    print("FLAGSPARSE SpMV CSC BENCHMARK (native CSC Triton)")
    print("=" * 140)
    print(
        "Timing policy: csc_ms = process_cpu_ms + csc_gpu_ms; CSC v1 has no process phase."
    )
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                print(_sep(timing))
                print(
                    f"dtype: {_dtype_name(dtype)} | index_dtype: {_dtype_name(index_dtype)} | op: {op}"
                )
                print(_sep(timing))
                print(_header(timing))
                print(_sep(timing))
                for m, n in TEST_SIZES:
                    dense = _random_values((m, n), dtype, device)
                    dense *= (torch.rand(m, n, device=device) < 0.1).to(dtype=dtype)
                    data, indices, indptr = _dense_to_csc(dense, index_dtype)
                    row = _run_one_case(
                        data,
                        indices,
                        indptr,
                        (m, n),
                        dtype,
                        index_dtype,
                        op,
                        f"{m}x{n}",
                        warmup,
                        iters,
                        timing=timing,
                        run_cusparse=run_cusparse,
                    )
                    _print_row(row, timing=timing)
                print(_sep(timing))
                print()


def run_csv(
    mtx_paths,
    csv_path,
    value_dtypes=None,
    index_dtypes=None,
    ops=None,
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
    ops = OPS if ops is None else ops
    rows = []
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                print(_sep(timing))
                print(
                    f"Value dtype: {_dtype_name(dtype)} | Index dtype: {_dtype_name(index_dtype)} | op: {op}"
                )
                print(_sep(timing))
                print(_header(timing))
                print(_sep(timing))
                for path in mtx_paths:
                    try:
                        data, indices, indptr, shape = load_mtx_to_csc_torch(
                            path, dtype=dtype, device=device
                        )
                        row = _run_one_case(
                            data,
                            indices,
                            indptr,
                            shape,
                            dtype,
                            index_dtype,
                            op,
                            os.path.basename(path),
                            warmup,
                            iters,
                            timing=timing,
                            run_cusparse=run_cusparse,
                        )
                    except Exception as exc:
                        if fail_fast:
                            raise
                        row = {
                            "matrix": os.path.basename(path),
                            "value_dtype": _dtype_name(dtype),
                            "index_dtype": _dtype_name(index_dtype),
                            "op": op,
                            "out_size": "ERR",
                            "n_rows": "ERR",
                            "n_cols": "ERR",
                            "nnz": "ERR",
                            "csc_ms": None,
                            "csc_gpu_ms": None,
                            "process_cpu_ms": None,
                            "process_gpu_ms": None,
                            "compute_ms": None,
                            "pytorch_ms": None,
                            "cusparse_ms": None,
                            "err": None,
                            "status": "ERROR",
                            "error": str(exc),
                        }
                    if fail_fast and row.get("status") == "ERROR":
                        raise RuntimeError(row.get("error") or "CSC SpMV case failed")
                    rows.append(row)
                    _print_row(row, timing=timing)
                print(_sep(timing))
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
        "out_size",
        "n_rows",
        "n_cols",
        "nnz",
        "csc_ms",
        "csc_gpu_ms",
        "process_cpu_ms",
        "process_gpu_ms",
        "compute_ms",
        "pytorch_ms",
        "cusparse_ms",
        "err",
        "status",
        "error",
    ]
    if not timing:
        fieldnames = [
            field
            for field in fieldnames
            if field not in ("process_gpu_ms", "compute_ms")
        ]
    csv_parent = Path(csv_path).parent
    if str(csv_parent) not in ("", "."):
        csv_parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: ("" if value is None else value) for key, value in row.items()}
            )
    print(f"Wrote {len(rows)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Native CSC SpMV benchmark/test.")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--csv-csc", type=str, default=None, metavar="FILE")
    parser.add_argument("--dtypes", default="float32,float64,complex64,complex128")
    parser.add_argument("--index-dtypes", default="int32,int64")
    parser.add_argument("--ops", default="non,trans,conj")
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    try:
        value_dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
        index_dtypes = _parse_csv_tokens(
            args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes"
        )
        ops = _parse_ops(args.ops)
    except ValueError as exc:
        parser.error(str(exc))
    if args.synthetic:
        run_synthetic(
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            ops=ops,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
            run_cusparse=not args.no_cusparse,
            fail_fast=args.fail_fast,
        )
        return
    paths = []
    for path in args.mtx:
        if os.path.isfile(path) and path.endswith(".mtx"):
            paths.append(path)
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    if args.csv_csc:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-csc")
            return
        run_csv(
            paths,
            args.csv_csc,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            ops=ops,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
            run_cusparse=not args.no_cusparse,
            fail_fast=args.fail_fast,
        )
        return
    if not paths:
        print("No .mtx files. Use --synthetic or --csv-csc with inputs.")
        return
    run_csv(
        paths,
        "spmv_csc_results.csv",
        value_dtypes=value_dtypes,
        index_dtypes=index_dtypes,
        ops=ops,
        warmup=args.warmup,
        iters=args.iters,
        timing=args.timing,
        run_cusparse=not args.no_cusparse,
        fail_fast=args.fail_fast,
    )


if __name__ == "__main__":
    main()
