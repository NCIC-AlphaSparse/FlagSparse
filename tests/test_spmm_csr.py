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

"""AlphaSparse-style CSR SpMM route benchmark.

This test is intentionally route-based: each output row represents one
algorithm for one matrix/dtype/op case, so future CSR algorithms can be swept
and ranked without changing the CSV schema.
"""

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
from flagsparse.sparse_operations import _common as fs_common
import flagsparse.sparse_operations.spmm_csr as spmm_ops

from test_spmm import (
    _build_dense_matrix,
    _build_pytorch_reference,
    _normalize_csv_path,
    load_mtx_to_csr_torch,
)

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {
    "int32": torch.int32,
    "int64": torch.int64,
}
DEFAULT_DTYPE_NAMES = ("float32", "float64", "complex64", "complex128")
DEFAULT_RUN_DTYPE_NAMES = ("float32", "float64")
DEFAULT_INDEX_DTYPE_NAMES = ("int32", "int64")
DEFAULT_OP_NAMES = tuple(spmm_ops.SPMM_OP_NAMES.values())
CUSPARSE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
TLE_CSR_SPMM_ALGORITHMS = {"alpha_alg1_tle_opt", "alpha_alg1_tle_opt2"}
MAIN_CSR_SPMM_ALGORITHMS = {
    "csr_base",
    "csr_base_accuracy",
    "alpha_alg1_tle_opt",
    "alpha_alg1_tle_opt2",
    "spmm_csr_alg1",
    "spmm_csr_alg2",
    "spmm_csr_alg2_accuracy",
    "spmm_csr_alg2_accuracy_hp",
}

PERF_FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "op",
    "layout",
    "alg",
    "n_rows",
    "n_cols",
    "nnz",
    "dense_cols",
    "b_stride",
    "c_stride",
    "ms",
    "gpu_ms",
    "process_cpu_ms",
    "torch_ms",
    "cusparse_ms",
    "torch_vs_alg_speedup",
    "cusparse_vs_alg_speedup",
    "err_vs_torch",
    "err_vs_cusparse",
    "status",
    "reason",
    "cusparse_reason",
]

TIMING_FIELDS = ["process_gpu_ms", "compute_ms"]

DIAG_FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "op",
    "layout",
    "alg",
    "launch_config_scope",
    "launch_config_count",
    "bucket_count",
    "long_row_count",
    "long_part_count",
    "num_warps",
    "num_stages",
    "block_n",
    "block_nnz",
    "warp_size",
    "factor",
    "block_rows",
    "block_cols",
    "grid_m",
    "grid_n",
    "launch_version",
    "dense_layout",
    "b_stride",
    "c_stride",
    "output_layout",
]

BEST_FIELDS = [
    "matrix",
    "dtype",
    "index_dtype",
    "op",
    "layout",
    "best_alg",
    "best_ms",
    "best_gpu_ms",
    "best_torch_speedup",
    "best_cusparse_speedup",
]

LAYOUT_NAMES = ("row", "col")


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


def _normalize_layout_name(layout):
    token = str(layout).strip().lower()
    if token in ("row", "row_major", "row-major", "c", "c_order"):
        return "row"
    if token in (
        "col",
        "column",
        "col_major",
        "column_major",
        "col-major",
        "column-major",
        "f",
        "fortran",
    ):
        return "col"
    raise ValueError("layout must be one of: row, col, all")


def _layout_names(value):
    value = str(value).strip().lower()
    if value == "all":
        return list(LAYOUT_NAMES)
    return [_normalize_layout_name(value)]


def _materialize_dense_layout_for_test(tensor, layout):
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
    if tensor is None:
        return ""
    return "x".join(str(int(v)) for v in tensor.stride())


def _reference_tolerance(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    if dtype == torch.float16:
        return 1e-3, 2e-3
    if dtype == torch.bfloat16:
        return 0.016, 1e-1
    return 1e-6, 1e-5


def _error_profile(candidate, reference, dtype):
    if candidate is None or reference is None:
        return {"global_err": None, "status": "SKIP"}
    atol, rtol = _reference_tolerance(dtype)
    if candidate.numel() == 0:
        return {"global_err": 0.0, "status": "PASS"}
    diff = torch.abs(candidate - reference).to(torch.float64)
    denom = (atol + rtol * torch.abs(reference)).to(torch.float64)
    ratio = diff / denom
    global_err = float(torch.max(ratio).item()) if ratio.numel() > 0 else 0.0
    return {"global_err": global_err, "status": "PASS" if global_err <= 1.0 else "FAIL"}


def _resolve_input_paths(input_paths):
    paths = []
    for path in input_paths:
        if os.path.isfile(path) and path.lower().endswith(".mtx"):
            paths.append(os.path.abspath(path))
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))
    return paths


def _parse_csv_names(value, all_names, option_name, explicit_names=None):
    value = str(value).strip().lower()
    if value == "all":
        return list(all_names)
    allowed = tuple(all_names if explicit_names is None else explicit_names)
    names = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not names:
        raise ValueError(f"{option_name} must not be empty")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(
            f"unsupported {option_name}: {', '.join(invalid)}; allowed: all,{','.join(allowed)}"
        )
    return names


def _parse_algs(value):
    value = str(value).strip().lower()
    if value in ("auto", "all"):
        return [value]
    allowed = set(fs.SPMM_CSR_ALGORITHMS)
    names = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not names:
        raise ValueError("--alg must not be empty")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(
            f"unsupported --alg: {', '.join(invalid)}; allowed: auto,all,{','.join(sorted(allowed))}"
        )
    return names


def _expand_algs(alg_names, op, dtype, exclude_tle=False):
    expanded = []
    for alg in alg_names:
        if alg == "all":
            expanded.extend(fs.list_spmm_csr_algorithms(op=op, dtype=dtype))
        elif alg == "auto":
            expanded.append("auto")
        else:
            expanded.append(alg)
    if exclude_tle:
        expanded = [
            alg
            for alg in expanded
            if alg not in TLE_CSR_SPMM_ALGORITHMS
            and fs.resolve_spmm_csr_algorithm(alg, op, dtype).name
            not in TLE_CSR_SPMM_ALGORITHMS
        ]
    deduped = []
    for alg in expanded:
        if alg not in deduped:
            deduped.append(alg)
    return deduped


def _selected_tle_algs(alg_names):
    selected = set()
    for alg in alg_names:
        if alg == "all":
            selected.update(("alpha_alg1_tle_opt", "alpha_alg1_tle_opt2"))
        elif alg in ("alpha_alg1_tle_opt", "alpha_alg1_tle_opt2"):
            selected.add(alg)
    return tuple(
        name
        for name in ("alpha_alg1_tle_opt", "alpha_alg1_tle_opt2")
        if name in selected
    )


def _print_tle_availability(alg_names):
    selected = _selected_tle_algs(alg_names)
    if not selected:
        return
    checks = {
        "alpha_alg1_tle_opt": (
            fs.is_alpha_spmm_alg1_tle_opt_available,
            fs.alpha_spmm_alg1_tle_opt_unavailable_reason,
        ),
        "alpha_alg1_tle_opt2": (
            fs.is_alpha_spmm_alg1_tle_opt2_available,
            fs.alpha_spmm_alg1_tle_opt2_unavailable_reason,
        ),
    }
    print("TLE runtime availability:")
    for name in selected:
        available_fn, reason_fn = checks[name]
        available = bool(available_fn())
        print(f"{name}: {'available' if available else 'unavailable'}")
        if not available:
            print(f"  reason: {reason_fn()}")


def _cuda_event_benchmark(op, warmup, iters):
    out = None
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


def _cupy_event_benchmark(op, warmup, iters):
    import cupy as cp

    out = None
    for _ in range(max(0, int(warmup))):
        out = op()
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    for _ in range(max(1, int(iters))):
        out = op()
    end.record()
    end.synchronize()
    return out, cp.cuda.get_elapsed_time(start, end) / max(1, int(iters))


def _time_route(
    prepared, B, alg, warmup, iters, timing=False, diagnose=False, layout="row"
):
    out, gpu_ms = _cuda_event_benchmark(
        lambda: fs.flagsparse_spmm_csr_run(prepared, B, alg=alg, dense_layout=layout),
        warmup,
        iters,
    )
    _, meta = fs.flagsparse_spmm_csr_run(
        prepared,
        B,
        alg=alg,
        dense_layout=layout,
        return_meta=True,
        timing=bool(timing),
        diagnostics=bool(diagnose),
    )
    process_cpu_ms = float(meta.get("process_cpu_ms", 0.0) or 0.0)
    row = {
        "alg": meta.get("alg", alg),
        "ms": process_cpu_ms + gpu_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": process_cpu_ms,
        "process_gpu_ms": None,
        "compute_ms": None,
        "dense_layout": meta.get("dense_layout", layout),
        "b_stride": meta.get("b_stride"),
        "c_stride": meta.get("c_stride"),
        "output_layout": meta.get("output_layout"),
        "diagnostics": meta.get("diagnostics", {}),
        "out": out,
    }
    if timing:
        row["process_gpu_ms"] = meta.get("process_gpu_ms")
        row["compute_ms"] = meta.get("compute_ms")
        if row["compute_ms"] is None and row["alg"] == "csr_base":
            row["compute_ms"] = gpu_ms
        if row["process_gpu_ms"] is None and row["alg"] == "csr_base":
            row["process_gpu_ms"] = 0.0
    return row


def _skip_row(
    path,
    dtype,
    index_dtype_name,
    op,
    layout,
    alg,
    shape,
    nnz,
    dense_cols,
    b_stride,
    torch_ms,
    cusparse_ms,
    reason,
    timing,
    cusparse_reason="",
):
    n_rows, n_cols = shape
    row = {
        "matrix": os.path.basename(path),
        "dtype": _dtype_name(dtype),
        "index_dtype": index_dtype_name,
        "op": op,
        "layout": layout,
        "alg": alg,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(nnz),
        "dense_cols": dense_cols,
        "b_stride": b_stride,
        "c_stride": "",
        "ms": None,
        "gpu_ms": None,
        "process_cpu_ms": None,
        "torch_ms": torch_ms,
        "cusparse_ms": cusparse_ms,
        "torch_vs_alg_speedup": None,
        "cusparse_vs_alg_speedup": None,
        "err_vs_torch": None,
        "err_vs_cusparse": None,
        "status": "SKIP",
        "reason": reason,
        "cusparse_reason": cusparse_reason or "",
    }
    if timing:
        row["process_gpu_ms"] = None
        row["compute_ms"] = None
    return row


def _vendor_label():
    return fs_common._expected_vendor_sparse_label()


def _vendor_column_label():
    return fs_common._expected_vendor_sparse_short().lower()


def _time_vendor_sparse_ref(
    data, indices, indptr, shape, B, op, warmup, iters, layout="row"
):
    vendor = fs_common._expected_vendor_sparse_backend()
    if vendor == "hipsparse":
        try:
            sparse_ref = spmm_ops._benchmark_spmm_csr_sparse_ref(
                data,
                indices,
                indptr,
                B,
                shape,
                warmup=warmup,
                iters=iters,
                op=op,
                dense_layout=layout,
            )
        except Exception as exc:
            return None, None, str(exc)
        if sparse_ref["backend"] is None:
            return None, None, sparse_ref["reason"]
        return sparse_ref["values"], sparse_ref["ms"], None
    if vendor != "cupy_cusparse":
        return (
            None,
            None,
            f"{fs_common._sparse_backend_label(vendor)} CSR SpMM baseline is not wired for this runner",
        )

    if data.dtype not in CUSPARSE_DTYPES:
        return None, None, "dtype not supported by CuPy/cuSPARSE reference"
    if op != "non":
        return (
            None,
            None,
            f"CuPy/cuSPARSE CSR SpMM baseline supports op=non only in this runner; op={op} is unsupported",
        )
    if layout != "row":
        return (
            None,
            None,
            f"CuPy/cuSPARSE CSR SpMM baseline supports row-major dense RHS only in this runner; layout={layout} is unsupported",
        )
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx_sparse
    except Exception as exc:
        return None, None, f"CuPy/cuSPARSE unavailable: {exc}"
    try:
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
        indices_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(indices.to(torch.int64))
        )
        indptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
        A = cpx_sparse.csr_matrix((data_cp, indices_cp, indptr_cp), shape=shape)
        out_cp, ms = _cupy_event_benchmark(lambda: A @ B_cp, warmup, iters)
        out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
        return out, ms, None
    except Exception as exc:
        return None, None, str(exc)


def run_one_case(
    path,
    dtype,
    index_dtype_name,
    index_dtype,
    op,
    layout,
    alg_names,
    dense_cols,
    warmup,
    iters,
    run_cusparse,
    timing,
    diagnose,
    exclude_tle=False,
):
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(
        path, dtype=dtype, device=device
    )
    indices = indices.to(index_dtype)
    indptr = indptr.to(index_dtype)
    n_rows, n_cols = shape
    b_rows = n_rows if op in ("trans", "conj") else n_cols
    B = _materialize_dense_layout_for_test(
        _build_dense_matrix(b_rows, dense_cols, dtype, device),
        layout,
    )
    b_stride = _stride_string(B)
    ref, torch_op, _torch_format = _build_pytorch_reference(
        data, indices, indptr, shape, B, op=op
    )
    _torch_out, torch_ms = _cuda_event_benchmark(torch_op, warmup, iters)
    cusparse_out = None
    cusparse_ms = None
    cusparse_reason = ""
    if run_cusparse:
        cusparse_out, cusparse_ms, cusparse_reason = _time_vendor_sparse_ref(
            data, indices, indptr, shape, B, op, warmup, iters, layout=layout
        )

    rows = []
    diag_rows = []
    prepared = fs.prepare_spmm_csr_route(
        data, indices, indptr, shape, op=op, alg="auto"
    )
    for alg in _expand_algs(alg_names, op, dtype, exclude_tle=exclude_tle):
        try:
            resolved = fs.resolve_spmm_csr_algorithm(alg, op, dtype)
            if layout == "col" and resolved.name not in MAIN_CSR_SPMM_ALGORITHMS:
                rows.append(
                    _skip_row(
                        path,
                        dtype,
                        index_dtype_name,
                        op,
                        layout,
                        alg,
                        shape,
                        data.numel(),
                        dense_cols,
                        b_stride,
                        torch_ms,
                        cusparse_ms,
                        "col-major layout is currently supported only by CSR SpMM main algorithms",
                        timing,
                        cusparse_reason=cusparse_reason,
                    )
                )
                continue
            result = _time_route(
                prepared,
                B,
                alg,
                warmup,
                iters,
                timing=timing,
                diagnose=diagnose,
                layout=layout,
            )
        except (fs.SpmmCsrAlgorithmUnavailable, ValueError, TypeError, RuntimeError) as exc:
            rows.append(
                _skip_row(
                    path,
                    dtype,
                    index_dtype_name,
                    op,
                    layout,
                    alg,
                    shape,
                    data.numel(),
                    dense_cols,
                    b_stride,
                    torch_ms,
                    cusparse_ms,
                    f"{type(exc).__name__}: {exc}",
                    timing,
                    cusparse_reason=cusparse_reason,
                )
            )
            continue
        out = result.pop("out")
        diagnostics = result.pop("diagnostics")
        torch_profile = _error_profile(out, ref, dtype)
        cusparse_profile = _error_profile(out, cusparse_out, dtype)
        row = {
            "matrix": os.path.basename(path),
            "dtype": _dtype_name(dtype),
            "index_dtype": index_dtype_name,
            "op": op,
            "layout": layout,
            "alg": result["alg"],
            "n_rows": n_rows,
            "n_cols": n_cols,
            "nnz": int(data.numel()),
            "dense_cols": dense_cols,
            "b_stride": _stride_string(B),
            "c_stride": _stride_string(out),
            "ms": result["ms"],
            "gpu_ms": result["gpu_ms"],
            "process_cpu_ms": result["process_cpu_ms"],
            "torch_ms": torch_ms,
            "cusparse_ms": cusparse_ms,
            "torch_vs_alg_speedup": _ratio(torch_ms, result["ms"]),
            "cusparse_vs_alg_speedup": _ratio(cusparse_ms, result["ms"]),
            "err_vs_torch": torch_profile["global_err"],
            "err_vs_cusparse": cusparse_profile["global_err"],
            "status": torch_profile["status"],
            "reason": "",
            "cusparse_reason": cusparse_reason or "",
        }
        if timing:
            row["process_gpu_ms"] = result["process_gpu_ms"]
            row["compute_ms"] = result["compute_ms"]
        rows.append(row)
        if diagnose:
            diag = {
                "matrix": os.path.basename(path),
                "dtype": _dtype_name(dtype),
                "index_dtype": index_dtype_name,
                "op": op,
                "layout": layout,
                "alg": result["alg"],
            }
            for field in DIAG_FIELDS:
                if field not in diag:
                    diag[field] = diagnostics.get(field)
            diag_rows.append(diag)
    return rows, diag_rows


def _best_rows(rows):
    groups = {}
    for row in rows:
        if row.get("status") != "PASS" or row.get("ms") is None:
            continue
        key = (
            row["matrix"],
            row["dtype"],
            row["index_dtype"],
            row["op"],
            row["layout"],
        )
        groups.setdefault(key, []).append(row)
    best = []
    for (matrix, dtype, index_dtype, op, layout), group in sorted(groups.items()):
        selected = min(group, key=lambda item: item["ms"])
        best.append(
            {
                "matrix": matrix,
                "dtype": dtype,
                "index_dtype": index_dtype,
                "op": op,
                "layout": layout,
                "best_alg": selected["alg"],
                "best_ms": selected["ms"],
                "best_gpu_ms": selected["gpu_ms"],
                "best_torch_speedup": selected["torch_vs_alg_speedup"],
                "best_cusparse_speedup": selected["cusparse_vs_alg_speedup"],
            }
        )
    return best


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_row(row):
    print(
        f"{row['matrix']:<28} {row['dtype']:<10} {row['index_dtype']:<5} {row['op']:<5} {row['layout']:<4} {row['alg']:<10} "
        f"{_fmt(row['ms']):>9} {_fmt(row['gpu_ms']):>9} {_fmt(row['process_cpu_ms']):>9} "
        f"{_fmt(row['torch_ms']):>9} {_fmt(row['cusparse_ms']):>9} "
        f"{_fmt(row['torch_vs_alg_speedup'], 2):>9} {_fmt(row['cusparse_vs_alg_speedup'], 2):>9} "
        f"{_fmt(row['err_vs_torch'], 2):>10} {row['status']:>6}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="AlphaSparse-style CSR SpMM route benchmark."
    )
    parser.add_argument("input", nargs="+", help=".mtx file(s) or directories")
    parser.add_argument(
        "--alg", default="auto", help="auto, all, or comma-separated algorithms"
    )
    parser.add_argument(
        "--dtype",
        default=",".join(DEFAULT_RUN_DTYPE_NAMES),
        help=(
            "Comma-separated dtype names. Default: float32,float64. "
            "all runs float32,float64,complex64,complex128; float16/bfloat16 are opt-in."
        ),
    )
    parser.add_argument(
        "--op", default="all", help="all or comma-separated ops: non,trans,conj"
    )
    parser.add_argument("--index-dtype", default="all", help="int32, int64, or all")
    parser.add_argument("--layout", default="row", help="row, col, or all")
    parser.add_argument("--dense-cols", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", default=None, help="Performance CSV path")
    parser.add_argument(
        "--no-cusparse",
        action="store_true",
        help="Disable vendor sparse reference (cuSPARSE on CUDA, hipSPARSE on ROCm)",
    )
    parser.add_argument(
        "--no-hipsparse",
        dest="no_cusparse",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--timing", action="store_true", help="Add process_gpu_ms/compute_ms columns"
    )
    parser.add_argument(
        "--exclude-tle",
        action="store_true",
        help="Exclude alpha_alg1_tle_opt and alpha_alg1_tle_opt2 from --alg sweeps",
    )
    parser.add_argument(
        "--diagnose", action="store_true", help="Write separate diagnose metadata CSV"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("A CUDA/ROCm PyTorch device is not available.")
        return
    paths = _resolve_input_paths(args.input)
    if not paths:
        print("No .mtx files found.")
        return
    try:
        dtype_names = _parse_csv_names(
            args.dtype,
            DEFAULT_DTYPE_NAMES,
            "--dtype",
            explicit_names=tuple(DTYPE_MAP),
        )
        op_names = _parse_csv_names(args.op, DEFAULT_OP_NAMES, "--op")
        index_dtype_names = _parse_csv_names(
            args.index_dtype,
            DEFAULT_INDEX_DTYPE_NAMES,
            "--index-dtype",
            explicit_names=tuple(INDEX_DTYPE_MAP),
        )
        layout_names = _layout_names(args.layout)
        alg_names = _parse_algs(args.alg)
    except ValueError as exc:
        parser.error(str(exc))

    torch.manual_seed(int(args.seed))
    fields = PERF_FIELDS + (TIMING_FIELDS if args.timing else [])
    rows = []
    diag_rows = []
    if args.exclude_tle:
        print("TLE runtime availability: skipped by --exclude-tle")
    else:
        _print_tle_availability(alg_names)
    vendor_label = _vendor_label()
    vendor_column = _vendor_column_label()
    first_dtype = DTYPE_MAP[dtype_names[0]]
    first_index_dtype = INDEX_DTYPE_MAP[index_dtype_names[0]]
    vendor_backend, vendor_reason = spmm_ops._spmm_csr_sparse_ref_backend(
        first_dtype, first_index_dtype, first_index_dtype
    )
    for line in fs_common._backend_summary_lines(
        op_name="SpMM CSR",
        native_format="CSR",
        correctness_ref="PyTorch CSR/COO",
        vendor_backend=vendor_backend,
        vendor_reason=vendor_reason,
        run_vendor=not args.no_cusparse,
    ):
        print(line)
    print(
        f"Vendor sparse baseline: {vendor_label}; unsupported combinations are reported as N/A with reason."
    )
    print(
        f"{'Matrix':<28} {'DType':<10} {'Idx':<5} {'Op':<5} {'Lay':<4} {'Alg':<10} "
        f"{'ms':>9} {'gpu_ms':>9} {'cpu_ms':>9} {'torch':>9} {vendor_column:>9} "
        f"{'PT/Alg':>9} {'V/Alg':>9} {'ErrPT':>10} {'Status':>6}"
    )
    for dtype_name in dtype_names:
        dtype = DTYPE_MAP[dtype_name]
        for index_dtype_name in index_dtype_names:
            index_dtype = INDEX_DTYPE_MAP[index_dtype_name]
            for op in op_names:
                for layout in layout_names:
                    for path in paths:
                        try:
                            case_rows, case_diag = run_one_case(
                                path,
                                dtype,
                                index_dtype_name,
                                index_dtype,
                                op,
                                layout,
                                alg_names,
                                args.dense_cols,
                                args.warmup,
                                args.iters,
                                not args.no_cusparse,
                                args.timing,
                                args.diagnose,
                                exclude_tle=args.exclude_tle,
                            )
                            rows.extend(case_rows)
                            diag_rows.extend(case_diag)
                            for row in case_rows:
                                _print_row(row)
                        except Exception as exc:
                            print(
                                f"  ERROR on {os.path.basename(path)} dtype={dtype_name} "
                                f"index_dtype={index_dtype_name} op={op} layout={layout}: {exc}"
                            )
    if args.csv:
        csv_path = _normalize_csv_path(args.csv)
        _write_csv(csv_path, rows, fields)
        root, ext = os.path.splitext(csv_path)
        best_path = f"{root}.best{ext}"
        _write_csv(best_path, _best_rows(rows), BEST_FIELDS)
        print(f"Wrote {len(rows)} rows to {csv_path}")
        print(f"Wrote best summary to {best_path}")
        if args.diagnose:
            diag_path = f"{root}.diagnose{ext}"
            _write_csv(diag_path, diag_rows, DIAG_FIELDS)
            print(f"Wrote diagnose metadata to {diag_path}")


if __name__ == "__main__":
    main()
