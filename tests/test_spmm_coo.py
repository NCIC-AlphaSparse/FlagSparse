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

"""
COO SpMM tests: load SuiteSparse .mtx, batch run, output error and performance.
Supports: multi .mtx files, value_dtype / index_dtype, CSV export, synthetic cases,
API validation checks, and PyTorch / CuPy comparison baselines.

This test module targets the current FlagSparse native COO SpMM implementation.
The default public route is a sorted row-run Triton COO kernel. A second native
atomic COO route is retained for internal parity checks and debug.
"""

import argparse
import csv
import glob
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as ast
import flagsparse.sparse_operations.spmm_coo as ast_ops

VALUE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]
INDEX_DTYPES = [torch.int32, torch.int64]
CSV_VALUE_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
CSV_INDEX_DTYPES = [torch.int32, torch.int64]
OP_NAMES = tuple(ast_ops.SPMM_COO_OP_NAMES.values())
LAYOUT_NAMES = ("row", "col")
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
    "bucket_counts",
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
TEST_CASES = [
    (512, 512, 4096, 16),
    (1024, 1024, 16384, 32),
    (2048, 2048, 65536, 64),
    (4096, 4096, 131072, 64),
]
COO_TILE_CASES = [
    (256, 256, 4096, 4),
    (256, 256, 4096, 5),
    (256, 256, 4096, 12),
    (256, 256, 4096, 24),
    (256, 256, 4096, 48),
    (256, 256, 4096, 96),
]
WARMUP = 10
ITERS = 50
DEFAULT_BLOCK_N = None
DEFAULT_BLOCK_NNZ = 256
DUPLICATE_CASE_DENSE_COLS = 48


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(value):
    return "N/A" if value is None else f"{value:.4f}"


def _fmt_speedup(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return "N/A"
    return f"{other_ms / triton_ms:.2f}x"


def _speedup_ratio(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return None
    return other_ms / triton_ms


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _parse_algs(value, route="rowrun"):
    if value is None:
        if route == "atomic":
            return ["coo_atomic"]
        if route == "compare":
            return ["coo_rowrun", "coo_atomic"]
        return ["auto"]
    value = str(value).strip().lower()
    if value in ("auto", "all"):
        return [value]
    allowed = set(ast.SPMM_COO_ALGORITHMS)
    aliases = {
        "rowrun": "coo_rowrun",
        "atomic": "coo_atomic",
        "alg1": "spmm_coo_alg1",
        "coo_alg1": "spmm_coo_alg1",
    }
    names = [
        aliases.get(token.strip().lower(), token.strip().lower())
        for token in value.split(",")
        if token.strip()
    ]
    if not names:
        raise ValueError("--alg must not be empty")
    invalid = [name for name in names if name not in allowed]
    if invalid:
        raise ValueError(
            f"unsupported --alg: {', '.join(invalid)}; allowed: auto,all,{','.join(sorted(allowed))}"
        )
    return names


def _expand_algs(alg_names, op, dtype):
    expanded = []
    for alg in alg_names:
        if alg == "all":
            expanded.extend(ast.list_spmm_coo_algorithms(op=op, dtype=dtype))
        elif alg == "auto":
            expanded.append("auto")
        else:
            expanded.append(alg)
    deduped = []
    for alg in expanded:
        if alg not in deduped:
            deduped.append(alg)
    return deduped


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


def _parse_op_names(value):
    token = str(value).strip().lower()
    if token == "all":
        return list(OP_NAMES)
    names = [item.strip().lower() for item in token.split(",") if item.strip()]
    invalid = [name for name in names if name not in OP_NAMES]
    if not names or invalid:
        raise ValueError(
            f"unsupported --op: {', '.join(invalid or names)}; allowed: all,{','.join(OP_NAMES)}"
        )
    return names


def _normalize_layout_name(layout):
    token = str(layout).strip().lower()
    if token in ("row", "row_major", "row-major", "c", "c_order", "auto", "default"):
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


def _is_complex_dtype(dtype):
    return dtype in (torch.complex64, torch.complex128)


def _fmt_err(value):
    return "N/A" if value is None else f"{value:.2e}"


def _fmt_check(value):
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def _status_label(value):
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def _normalize_csv_path(csv_path):
    csv_path = str(csv_path)
    if not csv_path.lower().endswith(".csv"):
        csv_path = f"{csv_path}.csv"
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return csv_path


def _fmt_launch_value(value):
    return "auto" if value is None else str(value)


def _build_values(length, value_dtype, device):
    shape = (length,)
    if value_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=value_dtype, device=device)
    if value_dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if value_dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"Unsupported value dtype: {value_dtype}")


def _build_dense_matrix(n_rows, n_cols, value_dtype, device):
    shape = (n_rows, n_cols)
    if value_dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        return torch.randn(shape, dtype=value_dtype, device=device)
    if value_dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if value_dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"Unsupported value dtype: {value_dtype}")


def _tolerance_for_dtype(value_dtype):
    if value_dtype == torch.float16:
        return 1e-3, 2e-3
    if value_dtype == torch.bfloat16:
        return 0.016, 1e-1
    if value_dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    return 1e-6, 1e-5


def _scaled_allclose_error(candidate, reference, value_dtype=None):
    if candidate.numel() == 0:
        return 0.0
    dtype = reference.dtype if value_dtype is None else value_dtype
    atol, rtol = _tolerance_for_dtype(dtype)
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    return float(torch.max(diff / denom).item())


def _error_profile(candidate, reference, dtype):
    if candidate is None or reference is None:
        return {"global_err": None, "status": "SKIP"}
    if candidate.numel() == 0:
        return {"global_err": 0.0, "status": "PASS"}
    err = _scaled_allclose_error(candidate, reference, dtype)
    return {"global_err": err, "status": "PASS" if err <= 1.0 else "FAIL"}


def _write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: ("" if value is None else value) for key, value in row.items()}
            )


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


def load_mtx_to_coo_torch(file_path, dtype=torch.float32, device=None):
    """Load a .mtx into COO torch tensors via the C-accelerated scipy reader
    (see tests/mtx_fast.py); the former pure-Python parser took minutes on
    large SuiteSparse matrices. Returns (data, row, col, shape)."""
    from mtx_fast import load_coo

    data, row, col, shape = load_coo(file_path, dtype=dtype, device=device)
    return data, row, col, shape


def _normalize_route(route):
    route = str(route).strip().lower()
    if route not in ("rowrun", "atomic", "compare"):
        raise ValueError("route must be one of: rowrun, atomic, compare")
    return route


def _selected_route(route):
    route = _normalize_route(route)
    return "rowrun" if route == "compare" else route


def _route_label(route):
    labels = {
        "rowrun": "COO native row-run",
        "atomic": "COO native atomic",
        "compare": "COO native row-run (compare mode)",
    }
    if route not in labels:
        raise ValueError(f"Unsupported route label: {route}")
    return labels[route]


def _empty_pairwise_summary():
    return {
        "match": None,
        "error_ratio": None,
        "max_abs_error": None,
        "max_relative_error": None,
        "sum_relative_error": None,
    }


def _prepare_canonical_case(data, row, col, shape, B, op="non", layout="row"):
    layout = _normalize_layout_name(layout)
    op_code = ast_ops._normalize_spmm_coo_op(op)
    data, row, col, shape = ast_ops._materialize_spmm_coo_op(
        data, row, col, shape, op_code
    )
    native_data, native_row, native_col, native_B, n_rows, n_cols, n_dense_cols = (
        ast_ops._prepare_spmm_coo_inputs(
            data,
            row,
            col,
            B,
            shape,
            dense_layout=layout,
        )
    )
    (
        canonical_data,
        canonical_row,
        canonical_col,
        canonical_B,
        _,
        _,
        _,
        output_dtype,
        _,
    ) = ast_ops._prepare_spmm_coo_canonical_prepared(
        native_data,
        native_row,
        native_col,
        native_B,
        n_rows,
        n_cols,
        n_dense_cols,
        dense_layout=layout,
    )
    cusparse_data, cusparse_row, cusparse_col = ast_ops._coalesce_coo_entries(
        native_data,
        native_row,
        native_col,
        (n_rows, n_cols),
    )
    cusparse_data, cusparse_row, cusparse_col = ast_ops._sort_coo_lex_inplace(
        cusparse_data,
        cusparse_row,
        cusparse_col,
        n_cols,
    )
    native_coo = ast_ops._build_torch_sparse_coo(
        native_data, native_row, native_col, shape
    )
    return {
        "native_data": native_data,
        "native_row": native_row,
        "native_col": native_col,
        "native_B": native_B,
        "native_coo": native_coo,
        "cusparse_data": cusparse_data,
        "cusparse_row": cusparse_row,
        "cusparse_col": cusparse_col,
        "canonical_data": canonical_data,
        "canonical_row": canonical_row,
        "canonical_col": canonical_col,
        "canonical_B": canonical_B,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_dense_cols": n_dense_cols,
        "output_dtype": output_dtype,
        "op": ast_ops._spmm_coo_op_to_name(op_code),
        "dense_layout": layout,
    }


def _build_pytorch_reference(
    data, row, col, shape, B, prepared=None, op="non", layout="row"
):
    prepared = (
        _prepare_canonical_case(data, row, col, shape, B, op=op, layout=layout)
        if prepared is None
        else prepared
    )
    expected = ast_ops._build_spmm_coo_pytorch_reference_from_canonical(
        prepared["canonical_data"],
        prepared["canonical_row"],
        prepared["canonical_col"],
        prepared["canonical_B"],
        (prepared["n_rows"], prepared["n_cols"]),
        prepared["output_dtype"],
    )
    pytorch_op = lambda: torch.sparse.mm(prepared["native_coo"], prepared["native_B"])
    return expected, pytorch_op, "COO", None


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


def _time_coo_algorithm(
    prepared, B, alg, warmup, iters, timing=False, diagnose=False, layout="row"
):
    out, gpu_ms = _cuda_event_benchmark(
        lambda: ast.flagsparse_spmm_coo_run(prepared, B, alg=alg, dense_layout=layout),
        warmup,
        iters,
    )
    _, meta = ast.flagsparse_spmm_coo_run(
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
        if row["process_gpu_ms"] is None:
            row["process_gpu_ms"] = 0.0
        if row["compute_ms"] is None:
            row["compute_ms"] = gpu_ms
    return row


def _skip_alg_row(
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


def _time_cusparse_coo(prepared_case, ref_C, dtype, warmup, iters, layout="row"):
    if dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
        return None, None, "dtype not supported by CuPy/cuSPARSE reference"
    sparse_ref_backend, _ = ast_ops._spmm_coo_sparse_ref_backend(
        prepared_case["cusparse_data"].dtype, prepared_case["cusparse_row"].dtype
    )
    if sparse_ref_backend == "hipsparse":
        # DCU/ROCm: hipSPARSE replaces cuSPARSE as the vendor baseline. COO SpMM
        # materialises the op before the call, so every op is covered. Without
        # this branch the DCU run silently reports the baseline as unavailable.
        try:
            ref = ast_ops._benchmark_spmm_coo_sparse_ref(
                prepared_case["cusparse_data"],
                prepared_case["cusparse_row"],
                prepared_case["cusparse_col"],
                prepared_case["native_B"],
                (prepared_case["n_rows"], prepared_case["n_cols"]),
                warmup,
                iters,
            )
        except Exception as exc:
            return None, None, str(exc)
        if ref["values"] is None:
            return None, None, ref["reason"] or "hipSPARSE COO SpMM reference skipped"
        del ref_C
        return ref["values"], ref["ms"], ref["reason"] or ""
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx
    except Exception as exc:
        return None, None, f"CuPy/cuSPARSE unavailable: {exc}"
    try:
        data_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(prepared_case["cusparse_data"])
        )
        row_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(prepared_case["cusparse_row"].to(torch.int64))
        )
        col_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(prepared_case["cusparse_col"].to(torch.int64))
        )
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(prepared_case["native_B"]))
        A_coo = cpx.coo_matrix(
            (data_cp, (row_cp, col_cp)),
            shape=(prepared_case["n_rows"], prepared_case["n_cols"]),
        )

        def _run(rhs):
            return A_coo @ rhs

        try:
            out_cp, ms = _cupy_event_benchmark(_run, B_cp, warmup, iters)
            reason = ""
        except Exception:
            if layout != "col":
                raise
            B_cp = cp.asfortranarray(B_cp)
            out_cp, ms = _cupy_event_benchmark(_run, B_cp, warmup, iters)
            reason = "used cp.asfortranarray fallback for col-major B"
        out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
        del ref_C
        return out, ms, reason
    except Exception as exc:
        return None, None, str(exc)


def _cupy_event_benchmark(op, arg, warmup, iters):
    import cupy as cp

    out = None
    for _ in range(max(0, int(warmup))):
        out = op(arg)
    cp.cuda.runtime.deviceSynchronize()
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    count = max(1, int(iters))
    start.record()
    for _ in range(count):
        out = op(arg)
    end.record()
    end.synchronize()
    return out, cp.cuda.get_elapsed_time(start, end) / count


def run_one_alg_case(
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
    progress=False,
):
    matrix_name = os.path.basename(path)

    def _start(stage):
        if progress:
            print(f"[{matrix_name}] {stage} ...", flush=True)
        return time.perf_counter()

    def _done(stage, started_at, extra=""):
        if progress:
            suffix = f" {extra}" if extra else ""
            print(
                f"[{matrix_name}] {stage} done in {(time.perf_counter() - started_at) * 1000.0:.2f} ms{suffix}",
                flush=True,
            )

    device = torch.device("cuda")
    stage_t0 = _start("load mtx")
    data, row, col, shape = load_mtx_to_coo_torch(path, dtype=dtype, device=device)
    row = row.to(index_dtype)
    col = col.to(index_dtype)
    n_rows, n_cols = shape
    _done("load mtx", stage_t0, f"shape=({n_rows},{n_cols}) nnz={int(data.numel())}")
    b_rows = n_rows if ast_ops._spmm_coo_op_transposes(op) else n_cols
    stage_t0 = _start(f"build dense B shape=({b_rows},{dense_cols})")
    B = _materialize_dense_layout_for_test(
        _build_dense_matrix(b_rows, dense_cols, dtype, device),
        layout,
    )
    b_stride = _stride_string(B)
    _done("build dense B", stage_t0, f"stride={b_stride}")
    stage_t0 = _start("prepare canonical reference")
    case = _prepare_canonical_case(data, row, col, shape, B, op=op, layout=layout)
    ref, pytorch_op, _torch_format, pytorch_reason = _build_pytorch_reference(
        data,
        row,
        col,
        shape,
        B,
        prepared=case,
        op=op,
        layout=layout,
    )
    _done("prepare canonical reference", stage_t0)
    torch_ms = None
    try:
        stage_t0 = _start("time PyTorch COO reference")
        _torch_out, torch_ms = _cuda_event_benchmark(pytorch_op, warmup, iters)
        _done("time PyTorch COO reference", stage_t0, f"ms={_fmt_ms(torch_ms)}")
    except Exception as exc:
        pytorch_reason = (
            str(exc) if pytorch_reason is None else f"{pytorch_reason}; timing: {exc}"
        )
        _done("time PyTorch COO reference", stage_t0, f"failed={exc}")

    cusparse_out = None
    cusparse_ms = None
    cusparse_reason = ""
    if run_cusparse:
        stage_t0 = _start("time CuPy/cuSPARSE COO reference")
        cusparse_out, cusparse_ms, cusparse_reason = _time_cusparse_coo(
            case,
            ref,
            dtype,
            warmup,
            iters,
            layout=layout,
        )
        _done(
            "time CuPy/cuSPARSE COO reference",
            stage_t0,
            f"ms={_fmt_ms(cusparse_ms)} reason={cusparse_reason or ''}",
        )

    rows = []
    diag_rows = []
    try:
        stage_t0 = _start("prepare COO route")
        prepared = ast.prepare_spmm_coo_route(data, row, col, shape, op=op, alg="auto")
        _done("prepare COO route", stage_t0, f"row_runs={prepared.n_segs}")
    except Exception as exc:
        _done("prepare COO route", stage_t0, f"failed={exc}")
        for alg in _expand_algs(alg_names, op, dtype):
            rows.append(
                _skip_alg_row(
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
                    f"prepare: {exc}",
                    timing,
                    cusparse_reason=cusparse_reason,
                )
            )
        return rows, diag_rows

    for alg in _expand_algs(alg_names, op, dtype):
        stage_t0 = None
        try:
            ast.resolve_spmm_coo_algorithm(alg, op, dtype)
            stage_t0 = _start(f"run {alg}")
            result = _time_coo_algorithm(
                prepared,
                B,
                alg,
                warmup,
                iters,
                timing=timing,
                diagnose=diagnose,
                layout=layout,
            )
            _done(f"run {alg}", stage_t0, f"ms={_fmt_ms(result['ms'])}")
        except (ast.SpmmCooAlgorithmUnavailable, ValueError, TypeError) as exc:
            if stage_t0 is not None:
                _done(f"run {alg}", stage_t0, f"skip={exc}")
            rows.append(
                _skip_alg_row(
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
                    str(exc),
                    timing,
                    cusparse_reason=cusparse_reason,
                )
            )
            continue
        out = result.pop("out")
        diagnostics = result.pop("diagnostics")
        torch_profile = _error_profile(out, ref, dtype)
        cusparse_profile = _error_profile(out, cusparse_out, dtype)
        row_out = {
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
            "reason": pytorch_reason or "",
            "cusparse_reason": cusparse_reason or "",
        }
        if timing:
            row_out["process_gpu_ms"] = result["process_gpu_ms"]
            row_out["compute_ms"] = result["compute_ms"]
        rows.append(row_out)
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


def _prepare_spmm_coo_timing_base(data, row, col, B, shape, op="non", layout="row"):
    layout = _normalize_layout_name(layout)
    op_code = ast_ops._normalize_spmm_coo_op(op)
    data, row, col, shape = ast_ops._materialize_spmm_coo_op(
        data, row, col, shape, op_code
    )
    native_data, native_row, native_col, native_B, n_rows, n_cols, n_dense_cols = (
        ast_ops._prepare_spmm_coo_inputs(
            data,
            row,
            col,
            B,
            shape,
            dense_layout=layout,
        )
    )
    output_dtype = native_data.dtype
    compute_dtype = ast_ops._spmm_coo_compute_dtype(output_dtype)
    data_compute = (
        native_data if compute_dtype == output_dtype else native_data.to(compute_dtype)
    )
    B_compute = (
        native_B if compute_dtype == output_dtype else native_B.to(compute_dtype)
    )
    launch = ast_ops._resolve_spmm_coo_launch_config(
        n_dense_cols,
        int(native_data.numel()),
        block_n=None,
        block_nnz=DEFAULT_BLOCK_NNZ,
    )
    return {
        "data": data_compute,
        "row": native_row,
        "col": native_col,
        "B": B_compute,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_dense_cols": n_dense_cols,
        "output_dtype": output_dtype,
        "layout": layout,
        "launch": launch,
    }


def _spmm_coo_rowrun_process(base):
    canonical_data, canonical_row, canonical_col = ast_ops._coalesce_coo_entries(
        base["data"],
        base["row"],
        base["col"],
        (base["n_rows"], base["n_cols"]),
    )
    canonical_data, canonical_row, canonical_col = ast_ops._sort_coo_lex_inplace(
        canonical_data,
        canonical_row,
        canonical_col,
        base["n_cols"],
    )
    seg_starts = ast_ops._seg_starts_from_sorted_rows(
        canonical_row,
        int(canonical_data.numel()),
        canonical_data.device,
    )
    return {
        "data": canonical_data,
        "row": canonical_row,
        "col": canonical_col,
        "B": base["B"],
        "seg_starts": seg_starts,
    }


def _spmm_coo_rowrun_compute(base, plan):
    launch = base["launch"]
    return ast_ops._triton_spmm_coo_rowrun_impl(
        plan["data"],
        plan["row"],
        plan["col"],
        plan["B"],
        base["n_rows"],
        base["n_dense_cols"],
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
        output_dtype=base["output_dtype"],
        dense_layout=base["layout"],
        seg_starts=plan["seg_starts"],
    )


def _spmm_coo_atomic_compute(base):
    launch = base["launch"]
    return ast_ops._triton_spmm_coo_atomic_impl(
        base["data"],
        base["row"],
        base["col"],
        base["B"],
        base["n_rows"],
        base["n_dense_cols"],
        block_n=launch["block_n"],
        block_nnz=launch["block_nnz"],
        output_dtype=base["output_dtype"],
        dense_layout=base["layout"],
    )


def _benchmark_spmm_coo_route_policy(
    data,
    row,
    col,
    B,
    shape,
    warmup,
    iters,
    route="rowrun",
    block_n=None,
    block_nnz=DEFAULT_BLOCK_NNZ,
    op="non",
    layout="row",
    timing=False,
):
    selected_route = _selected_route(route)
    base = _prepare_spmm_coo_timing_base(data, row, col, B, shape, op=op, layout=layout)
    if block_n is not None or block_nnz != DEFAULT_BLOCK_NNZ:
        base["launch"] = ast_ops._resolve_spmm_coo_launch_config(
            base["n_dense_cols"],
            int(base["data"].numel()),
            block_n=block_n,
            block_nnz=block_nnz,
        )

    if selected_route == "rowrun":

        def full_op():
            plan = _spmm_coo_rowrun_process(base)
            return _spmm_coo_rowrun_compute(base, plan)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = full_op()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - t0) * 1000.0
        values, gpu_ms = _cuda_event_benchmark(full_op, warmup, iters)
        process_gpu_ms = None
        compute_ms = None
        total_ms = gpu_ms
        if timing:
            plan, process_gpu_ms = _cuda_event_benchmark(
                lambda: _spmm_coo_rowrun_process(base), warmup, iters
            )
            values, compute_ms = _cuda_event_benchmark(
                lambda: _spmm_coo_rowrun_compute(base, plan), warmup, iters
            )
            total_ms = process_gpu_ms + compute_ms
    else:

        def full_op():
            return _spmm_coo_atomic_compute(base)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = full_op()
        torch.cuda.synchronize()
        first_call_ms = (time.perf_counter() - t0) * 1000.0
        values, gpu_ms = _cuda_event_benchmark(full_op, warmup, iters)
        process_gpu_ms = 0.0 if timing else None
        compute_ms = None
        total_ms = gpu_ms
        if timing:
            values, compute_ms = _cuda_event_benchmark(full_op, warmup, iters)
            total_ms = compute_ms

    return {
        "values": values,
        "ms": total_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": process_gpu_ms,
        "compute_ms": compute_ms,
        "first_call_ms": first_call_ms,
    }


def _benchmark_spmm_coo_route(
    data,
    row,
    col,
    B,
    shape,
    warmup,
    iters,
    route="rowrun",
    block_n=None,
    block_nnz=DEFAULT_BLOCK_NNZ,
    prepared=None,
    op="non",
    layout="row",
    timing=False,
):
    selected_route = _selected_route(route)
    del prepared
    result = _benchmark_spmm_coo_route_policy(
        data,
        row,
        col,
        B,
        shape,
        warmup,
        iters,
        block_n=block_n,
        block_nnz=block_nnz,
        route=selected_route,
        op=op,
        layout=layout,
        timing=timing,
    )
    return result["values"], result["ms"], result["first_call_ms"], result


def _summarize_route_output(
    values, reference, value_dtype, ms=None, first_call_ms=None, cusparse_values=None
):
    metrics = ast_ops._spmm_validation_metrics(values, reference)
    atol, rtol = _tolerance_for_dtype(value_dtype)
    summary = {
        "ms": ms,
        "first_call_ms": first_call_ms,
        "ok_pt": torch.allclose(values, reference, atol=atol, rtol=rtol),
        "err_pt": _scaled_allclose_error(values, reference, value_dtype),
        "max_abs_error": metrics["max_abs_error"],
        "max_relative_error": metrics["max_relative_error"],
        "ok_cu": None,
        "err_cu": None,
        "error": None,
    }
    if cusparse_values is not None:
        summary["ok_cu"] = torch.allclose(values, cusparse_values, atol=atol, rtol=rtol)
        summary["err_cu"] = _scaled_allclose_error(values, cusparse_values, value_dtype)
    return summary


def _pairwise_route_summary(candidate, reference, value_dtype):
    return ast_ops._spmm_coo_pairwise_summary(candidate, reference, value_dtype)


def _format_debug_scalar(value):
    if value is None:
        return "-"
    if torch.is_tensor(value):
        value = value.item()
    if isinstance(value, complex):
        return f"{value.real:.16e}{value.imag:+.16e}j"
    return f"{float(value):.16e}"


def _build_compare_debug_summary(
    row, reference, route_outputs, cusparse_values, value_dtype
):
    if reference is None or reference.numel() == 0:
        return None

    atol, rtol = _tolerance_for_dtype(value_dtype)
    candidates = []
    for label in ("rowrun", "atomic"):
        values = route_outputs.get(label)
        if values is not None:
            candidates.append((label, values))
    if cusparse_values is not None:
        candidates.append(("cusparse", cusparse_values))

    best = None
    for label, candidate in candidates:
        if (
            candidate is None
            or candidate.shape != reference.shape
            or candidate.numel() == 0
        ):
            continue
        diff = torch.abs(candidate - reference)
        denom = atol + rtol * torch.abs(reference)
        ratio = diff / denom
        flat_idx = int(torch.argmax(ratio).item())
        error_ratio = float(ratio.reshape(-1)[flat_idx].item())
        if best is None or error_ratio > best["error_ratio"]:
            row_idx = flat_idx // reference.shape[1]
            dense_col = flat_idx % reference.shape[1]
            best = {
                "route": label,
                "row": row_idx,
                "dense_col": dense_col,
                "error_ratio": error_ratio,
            }

    if best is None:
        return None

    row_idx = best["row"]
    dense_col = best["dense_col"]
    row64 = row.to(torch.int64)
    row_nnz = int((row64 == row_idx).sum().item())

    def _scalar_at(values):
        if values is None:
            return None
        return values[row_idx, dense_col]

    return {
        "route": best["route"],
        "row": row_idx,
        "dense_col": dense_col,
        "row_nnz": row_nnz,
        "error_ratio": best["error_ratio"],
        "rowrun": _format_debug_scalar(_scalar_at(route_outputs.get("rowrun"))),
        "atomic": _format_debug_scalar(_scalar_at(route_outputs.get("atomic"))),
        "pt": _format_debug_scalar(reference[row_idx, dense_col]),
        "cu": _format_debug_scalar(_scalar_at(cusparse_values)),
    }


def _assert_spmm_coo_matches_reference(
    data,
    row,
    col,
    B,
    shape,
    value_dtype,
    out=None,
    block_n=None,
    block_nnz=DEFAULT_BLOCK_NNZ,
    op="non",
    layout="row",
):
    layout = _normalize_layout_name(layout)
    result = ast.flagsparse_spmm_coo(
        data,
        row,
        col,
        B,
        shape,
        block_n=block_n,
        block_nnz=block_nnz,
        out=out,
        op=op,
        dense_layout=layout,
    )
    ref_C, _, _, _ = _build_pytorch_reference(
        data, row, col, shape, B, op=op, layout=layout
    )
    atol, rtol = _tolerance_for_dtype(value_dtype)
    if not torch.allclose(result, ref_C, atol=atol, rtol=rtol):
        metrics = ast_ops._spmm_validation_metrics(result, ref_C)
        raise AssertionError(
            "reference mismatch: "
            f"err={_scaled_allclose_error(result, ref_C, value_dtype):.3e}, "
            f"max_abs={metrics['max_abs_error']:.3e}, "
            f"atol={atol:.3e}, "
            f"rtol={rtol:.3e}"
        )
    if out is not None and result.data_ptr() != out.data_ptr():
        raise AssertionError(
            "flagsparse_spmm_coo did not return the provided out tensor"
        )
    return result, ref_C


def _build_duplicate_unsorted_case(
    value_dtype, index_dtype, device, n_dense_cols=DUPLICATE_CASE_DENSE_COLS
):
    shape = (4, 6)
    row = torch.tensor([2, 0, 2, 1, 2, 0, 3, 2], dtype=index_dtype, device=device)
    col = torch.tensor([1, 4, 1, 3, 0, 4, 2, 5], dtype=index_dtype, device=device)
    data = _build_values(row.numel(), value_dtype, device)
    B = _build_dense_matrix(shape[1], n_dense_cols, value_dtype, device)
    return data, row, col, B, shape


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    n_dense_cols=32,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    route="rowrun",
    op="non",
    layout="row",
    timing=False,
):
    route = _normalize_route(route)
    selected_route = _selected_route(route)
    op_name = ast_ops._spmm_coo_op_to_name(op)
    layout = _normalize_layout_name(layout)
    device = torch.device("cuda")
    data, row, col, shape = load_mtx_to_coo_torch(
        mtx_path, dtype=value_dtype, device=device
    )
    row = row.to(index_dtype)
    col = col.to(index_dtype)
    n_rows, n_cols = shape
    nnz = data.numel()
    b_rows = n_rows if ast_ops._spmm_coo_op_transposes(op_name) else n_cols
    B = _materialize_dense_layout_for_test(
        _build_dense_matrix(b_rows, n_dense_cols, value_dtype, device),
        layout,
    )
    prepared = None
    atol, rtol = _tolerance_for_dtype(value_dtype)

    result = {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz,
        "dense_cols": n_dense_cols,
        "route": selected_route,
        "op": op_name,
        "layout": layout,
        "b_stride": _stride_string(B),
        "c_stride": "",
        "error": None,
        "triton_ms": None,
        "triton_gpu_ms": None,
        "process_cpu_ms": None,
        "process_gpu_ms": None,
        "compute_ms": None,
        "triton_first_call_ms": None,
        "cusparse_ms": None,
        "pytorch_ms": None,
        "err_pt": None,
        "err_cu": None,
        "triton_abs_err": None,
        "cusparse_abs_err": None,
        "triton_relative_error_diag": None,
        "cusparse_relative_error_diag": None,
        "triton_ok_pt": None,
        "triton_ok_cu": None,
        "cusparse_reason": None,
        "pytorch_reason": None,
        "pytorch_format": None,
        "status": "UNKNOWN",
        "compare": None,
    }
    try:
        prepared = _prepare_canonical_case(
            data, row, col, shape, B, op=op_name, layout=layout
        )
        ref_C, pytorch_op, pytorch_format, pytorch_reason = _build_pytorch_reference(
            data,
            row,
            col,
            shape,
            B,
            prepared=prepared,
            op=op_name,
            layout=layout,
        )
        result["pytorch_format"] = pytorch_format
        result["pytorch_reason"] = pytorch_reason
    except Exception as exc:
        result["error"] = f"ref: {exc}"
        result["status"] = "REF_FAIL"
        return result

    triton_C = None
    try:
        triton_C, triton_ms, triton_first_call_ms, triton_timing = (
            _benchmark_spmm_coo_route(
                data,
                row,
                col,
                B,
                shape,
                warmup,
                iters,
                route=selected_route,
                block_n=block_n,
                block_nnz=block_nnz,
                prepared=prepared,
                op=op_name,
                layout=layout,
                timing=timing,
            )
        )
        result["triton_ms"] = triton_ms
        result["triton_gpu_ms"] = triton_timing.get("gpu_ms")
        result["process_cpu_ms"] = triton_timing.get("process_cpu_ms")
        result["process_gpu_ms"] = triton_timing.get("process_gpu_ms")
        result["compute_ms"] = triton_timing.get("compute_ms")
        result["triton_first_call_ms"] = triton_first_call_ms
        result["c_stride"] = _stride_string(triton_C)
    except Exception as exc:
        # Continue to PyTorch / CuPy timing when Triton fails (same as CSR SpMM test).
        result["error"] = f"triton: {exc}"
        result["triton_ok_pt"] = False

    if triton_C is not None:
        triton_summary = _summarize_route_output(triton_C, ref_C, value_dtype)
        result["triton_abs_err"] = triton_summary["max_abs_error"]
        result["triton_relative_error_diag"] = triton_summary["max_relative_error"]
        result["err_pt"] = triton_summary["err_pt"]
        result["triton_ok_pt"] = triton_summary["ok_pt"]
    else:
        result["triton_ok_pt"] = False

    try:
        _, result["pytorch_ms"] = ast_ops._benchmark_cuda_op(
            pytorch_op,
            warmup=warmup,
            iters=iters,
        )
    except Exception as exc:
        reason = str(exc)
        if result["pytorch_reason"]:
            result["pytorch_reason"] = f"{result['pytorch_reason']}; timing: {reason}"
        else:
            result["pytorch_reason"] = reason

    cs_C_t = None
    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    # Vendor baseline per backend: hipSPARSE on DCU/ROCm, cuSPARSE via CuPy on CUDA.
    sparse_ref_backend, sparse_ref_reason = ast_ops._spmm_coo_sparse_ref_backend(
        value_dtype, prepared["cusparse_row"].dtype
    )
    if run_cusparse and sparse_ref_backend == "hipsparse":
        try:
            cs_C_t, result["cusparse_ms"] = ast_ops._benchmark_prepared_cuda_op(
                lambda: ast_ops._prepare_spmm_coo_ref_hipsparse(
                    prepared["cusparse_data"],
                    prepared["cusparse_row"],
                    prepared["cusparse_col"],
                    prepared["native_B"],
                    (prepared["n_rows"], prepared["n_cols"]),
                ),
                ast_ops._run_spmm_coo_ref_hipsparse_prepared,
                ast_ops._destroy_spmm_coo_ref_hipsparse_prepared,
                warmup=warmup,
                iters=iters,
            )
            cusparse_metrics = ast_ops._spmm_validation_metrics(cs_C_t, ref_C)
            result["cusparse_abs_err"] = cusparse_metrics["max_abs_error"]
            result["cusparse_relative_error_diag"] = cusparse_metrics[
                "max_relative_error"
            ]
            if triton_C is not None:
                result["err_cu"] = _scaled_allclose_error(triton_C, cs_C_t, value_dtype)
                result["triton_ok_cu"] = torch.allclose(
                    triton_C, cs_C_t, atol=atol, rtol=rtol
                )
        except Exception as exc:
            result["cusparse_ms"] = None
            result["err_cu"] = None
            result["cusparse_abs_err"] = None
            result["cusparse_relative_error_diag"] = None
            result["triton_ok_cu"] = None
            result["cusparse_reason"] = str(exc)
    elif run_cusparse:
        if value_dtype not in _cupy_supported_dtypes:
            result["cusparse_reason"] = (
                "float16/bfloat16 not supported by CuPy sparse; skipped"
            )
        else:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cpx

                data_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(prepared["cusparse_data"])
                )
                row_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(
                        prepared["cusparse_row"].to(torch.int64)
                    )
                )
                col_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(
                        prepared["cusparse_col"].to(torch.int64)
                    )
                )
                B_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(prepared["native_B"])
                )
                A_coo = cpx.coo_matrix(
                    (data_cp, (row_cp, col_cp)),
                    shape=(prepared["n_rows"], prepared["n_cols"]),
                )

                def _run_cusparse_timing(rhs):
                    torch.cuda.synchronize()
                    for _ in range(warmup):
                        _ = A_coo @ rhs
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(iters):
                        _ = A_coo @ rhs
                    end.record()
                    torch.cuda.synchronize()
                    return start.elapsed_time(end) / iters

                try:
                    result["cusparse_ms"] = _run_cusparse_timing(B_cp)
                except Exception:
                    if layout != "col":
                        raise
                    B_cp = cp.asfortranarray(B_cp)
                    result["cusparse_reason"] = (
                        "used cp.asfortranarray fallback for col-major B"
                    )
                    result["cusparse_ms"] = _run_cusparse_timing(B_cp)

                cs_C = A_coo @ B_cp
                cs_C_t = torch.utils.dlpack.from_dlpack(cs_C.toDlpack())
                cusparse_metrics = ast_ops._spmm_validation_metrics(cs_C_t, ref_C)
                result["cusparse_abs_err"] = cusparse_metrics["max_abs_error"]
                result["cusparse_relative_error_diag"] = cusparse_metrics[
                    "max_relative_error"
                ]
                if triton_C is not None:
                    result["err_cu"] = _scaled_allclose_error(
                        triton_C, cs_C_t, value_dtype
                    )
                    result["triton_ok_cu"] = torch.allclose(
                        triton_C, cs_C_t, atol=atol, rtol=rtol
                    )
            except Exception as exc:
                result["cusparse_ms"] = None
                result["err_cu"] = None
                result["cusparse_abs_err"] = None
                result["cusparse_relative_error_diag"] = None
                result["triton_ok_cu"] = None
                result["cusparse_reason"] = str(exc)

    if route == "compare":
        route_outputs = {}
        route_summaries = {}
        if triton_C is not None:
            route_outputs[selected_route] = triton_C
            route_summary = _summarize_route_output(
                triton_C,
                ref_C,
                value_dtype,
                ms=triton_ms,
                first_call_ms=triton_first_call_ms,
                cusparse_values=cs_C_t,
            )
            route_summary.update(
                {
                    "gpu_ms": result.get("triton_gpu_ms"),
                    "process_cpu_ms": result.get("process_cpu_ms"),
                    "process_gpu_ms": result.get("process_gpu_ms"),
                    "compute_ms": result.get("compute_ms"),
                }
            )
            route_summaries[selected_route] = route_summary
        for extra_route in ("rowrun", "atomic"):
            if extra_route in route_outputs:
                continue
            try:
                extra_C, extra_ms, extra_first_call_ms, extra_timing = (
                    _benchmark_spmm_coo_route(
                        data,
                        row,
                        col,
                        B,
                        shape,
                        warmup,
                        iters,
                        route=extra_route,
                        block_n=block_n,
                        block_nnz=block_nnz,
                        prepared=prepared,
                        op=op_name,
                        layout=layout,
                        timing=timing,
                    )
                )
                route_outputs[extra_route] = extra_C
                route_summary = _summarize_route_output(
                    extra_C,
                    ref_C,
                    value_dtype,
                    ms=extra_ms,
                    first_call_ms=extra_first_call_ms,
                    cusparse_values=cs_C_t,
                )
                route_summary.update(
                    {
                        "gpu_ms": extra_timing.get("gpu_ms"),
                        "process_cpu_ms": extra_timing.get("process_cpu_ms"),
                        "process_gpu_ms": extra_timing.get("process_gpu_ms"),
                        "compute_ms": extra_timing.get("compute_ms"),
                    }
                )
                route_summaries[extra_route] = route_summary
            except Exception as exc:
                route_summaries[extra_route] = {
                    "ms": None,
                    "first_call_ms": None,
                    "ok_pt": False,
                    "err_pt": None,
                    "max_abs_error": None,
                    "max_relative_error": None,
                    "ok_cu": None,
                    "err_cu": None,
                    "error": str(exc),
                }

        parity = {
            "rowrun_vs_atomic": _empty_pairwise_summary(),
        }
        if "rowrun" in route_outputs and "atomic" in route_outputs:
            parity["rowrun_vs_atomic"] = _pairwise_route_summary(
                route_outputs["rowrun"], route_outputs["atomic"], value_dtype
            )

        cu_match = (
            None
            if cs_C_t is None
            else torch.allclose(cs_C_t, ref_C, atol=atol, rtol=rtol)
        )
        compare_debug = None
        rowrun_summary = route_summaries.get("rowrun") or {}
        atomic_summary = route_summaries.get("atomic") or {}
        if (
            rowrun_summary.get("ok_pt") is False
            or atomic_summary.get("ok_pt") is False
            or cu_match is False
        ):
            compare_debug = _build_compare_debug_summary(
                prepared["canonical_row"], ref_C, route_outputs, cs_C_t, value_dtype
            )

        result["compare"] = {
            "routes": route_summaries,
            "parity": parity,
            "cusparse_reference_match": cu_match,
            "cusparse_reference_error": (
                None
                if cs_C_t is None
                else _scaled_allclose_error(cs_C_t, ref_C, value_dtype)
            ),
            "debug": compare_debug,
        }

    result["status"] = (
        "PASS" if (result["triton_ok_pt"] or result["triton_ok_cu"]) else "FAIL"
    )
    return result


def run_mtx_batch(
    paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    n_dense_cols=32,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    route="rowrun",
    op="non",
    layout="row",
    timing=False,
    on_result=None,
):
    results = []
    for path in paths:
        entry = run_one_mtx(
            path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=warmup,
            iters=iters,
            run_cusparse=run_cusparse,
            n_dense_cols=n_dense_cols,
            block_n=block_n,
            block_nnz=block_nnz,
            route=route,
            op=op,
            layout=layout,
            timing=timing,
        )
        results.append(entry)
        if on_result is not None:
            on_result(entry)
    return results


def _benchmark_spmm_coo_synthetic_policy(
    n_rows=4096,
    n_cols=4096,
    nnz=65536,
    n_dense_cols=32,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=20,
    iters=200,
    block_n=None,
    block_nnz=DEFAULT_BLOCK_NNZ,
    run_cusparse=True,
    route="rowrun",
    compare_routes=False,
    op="non",
    dense_layout="row",
    timing=False,
):
    device = torch.device("cuda")
    route = _selected_route(route)
    layout = _normalize_layout_name(dense_layout)
    op_name = ast_ops._spmm_coo_op_to_name(op)
    data, row, col = ast_ops._build_random_coo(
        n_rows,
        n_cols,
        int(nnz),
        value_dtype,
        index_dtype,
        device,
    )
    shape = (int(n_rows), int(n_cols))
    b_rows = n_rows if ast_ops._spmm_coo_op_transposes(op_name) else n_cols
    B = _materialize_dense_layout_for_test(
        _build_dense_matrix(b_rows, n_dense_cols, value_dtype, device),
        layout,
    )
    prepared = _prepare_canonical_case(
        data, row, col, shape, B, op=op_name, layout=layout
    )
    expected, pytorch_op, pytorch_format, pytorch_reason = _build_pytorch_reference(
        data,
        row,
        col,
        shape,
        B,
        prepared=prepared,
        op=op_name,
        layout=layout,
    )
    launch = ast_ops._resolve_spmm_coo_launch_config(
        prepared["n_dense_cols"],
        int(prepared["canonical_data"].numel()),
        block_n=block_n,
        block_nnz=block_nnz,
    )
    seg_starts = ast_ops._seg_starts_from_sorted_rows(
        prepared["canonical_row"],
        int(prepared["canonical_data"].numel()),
        device,
    )
    n_row_runs = int(seg_starts.numel()) - 1 if seg_starts is not None else 0

    triton_C, triton_ms, triton_first_call_ms, triton_timing = (
        _benchmark_spmm_coo_route(
            data,
            row,
            col,
            B,
            shape,
            warmup,
            iters,
            route=route,
            block_n=launch["block_n"],
            block_nnz=launch["block_nnz"],
            prepared=prepared,
            op=op_name,
            layout=layout,
            timing=timing,
        )
    )
    triton_summary = _summarize_route_output(triton_C, expected, value_dtype)

    pytorch_values = expected
    pytorch_ms = None
    try:
        pytorch_values, pytorch_ms = ast_ops._benchmark_cuda_op(
            pytorch_op, warmup=warmup, iters=iters
        )
    except Exception as exc:
        pytorch_reason = (
            str(exc) if pytorch_reason is None else f"{pytorch_reason}; timing: {exc}"
        )

    cusparse_ms = None
    cusparse_match = None
    cusparse_reason = None
    cusparse_values = None
    cusparse_summary = None
    bench_ref_backend, bench_ref_reason = ast_ops._spmm_coo_sparse_ref_backend(
        value_dtype, prepared["cusparse_row"].dtype
    )
    if run_cusparse and bench_ref_backend == "hipsparse":
        try:
            cusparse_values, cusparse_ms = ast_ops._benchmark_prepared_cuda_op(
                lambda: ast_ops._prepare_spmm_coo_ref_hipsparse(
                    prepared["cusparse_data"],
                    prepared["cusparse_row"],
                    prepared["cusparse_col"],
                    prepared["native_B"],
                    (prepared["n_rows"], prepared["n_cols"]),
                ),
                ast_ops._run_spmm_coo_ref_hipsparse_prepared,
                ast_ops._destroy_spmm_coo_ref_hipsparse_prepared,
                warmup=warmup,
                iters=iters,
            )
            cusparse_summary = ast_ops._spmm_validation_metrics(
                cusparse_values, expected
            )
            atol, rtol = _tolerance_for_dtype(value_dtype)
            cusparse_match = torch.allclose(
                cusparse_values, expected, atol=atol, rtol=rtol
            )
        except Exception as exc:
            cusparse_reason = str(exc)
    elif run_cusparse:
        if value_dtype not in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ):
            cusparse_reason = "float16/bfloat16 not supported by CuPy sparse; skipped"
        else:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as cpx

                data_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(prepared["cusparse_data"])
                )
                row_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(
                        prepared["cusparse_row"].to(torch.int64)
                    )
                )
                col_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(
                        prepared["cusparse_col"].to(torch.int64)
                    )
                )
                B_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(prepared["native_B"])
                )
                A_coo = cpx.coo_matrix(
                    (data_cp, (row_cp, col_cp)),
                    shape=(prepared["n_rows"], prepared["n_cols"]),
                )
                cusparse_values_cp, cusparse_ms = ast_ops._benchmark_cuda_op(
                    lambda: A_coo @ B_cp,
                    warmup=warmup,
                    iters=iters,
                )
                cusparse_values = torch.utils.dlpack.from_dlpack(
                    cusparse_values_cp.toDlpack()
                )
                cusparse_summary = ast_ops._spmm_validation_metrics(
                    cusparse_values, expected
                )
                atol, rtol = _tolerance_for_dtype(value_dtype)
                cusparse_match = torch.allclose(
                    cusparse_values, expected, atol=atol, rtol=rtol
                )
            except Exception as exc:
                cusparse_reason = str(exc)

    route_results = None
    route_samples = None
    parity = None
    if compare_routes:
        route_outputs = {route: triton_C}
        route_results = {
            route: {
                "route": route,
                "ms": triton_ms,
                "gpu_ms": triton_timing.get("gpu_ms"),
                "process_cpu_ms": triton_timing.get("process_cpu_ms"),
                "process_gpu_ms": triton_timing.get("process_gpu_ms"),
                "compute_ms": triton_timing.get("compute_ms"),
                "first_call_ms": triton_first_call_ms,
                "match_reference": triton_summary["ok_pt"],
                "error_ratio": triton_summary["err_pt"],
                "max_abs_error": triton_summary["max_abs_error"],
                "max_relative_error": triton_summary["max_relative_error"],
                "error": None,
            }
        }
        for extra_route in ("rowrun", "atomic"):
            if extra_route in route_outputs:
                continue
            try:
                extra_C, extra_ms, extra_first_call_ms, extra_timing = (
                    _benchmark_spmm_coo_route(
                        data,
                        row,
                        col,
                        B,
                        shape,
                        warmup,
                        iters,
                        route=extra_route,
                        block_n=launch["block_n"],
                        block_nnz=launch["block_nnz"],
                        prepared=prepared,
                        op=op_name,
                        layout=layout,
                        timing=timing,
                    )
                )
                extra_summary = _summarize_route_output(extra_C, expected, value_dtype)
                route_outputs[extra_route] = extra_C
                route_results[extra_route] = {
                    "route": extra_route,
                    "ms": extra_ms,
                    "gpu_ms": extra_timing.get("gpu_ms"),
                    "process_cpu_ms": extra_timing.get("process_cpu_ms"),
                    "process_gpu_ms": extra_timing.get("process_gpu_ms"),
                    "compute_ms": extra_timing.get("compute_ms"),
                    "first_call_ms": extra_first_call_ms,
                    "match_reference": extra_summary["ok_pt"],
                    "error_ratio": extra_summary["err_pt"],
                    "max_abs_error": extra_summary["max_abs_error"],
                    "max_relative_error": extra_summary["max_relative_error"],
                    "error": None,
                }
            except Exception as exc:
                route_results[extra_route] = {
                    "route": extra_route,
                    "ms": None,
                    "first_call_ms": None,
                    "match_reference": False,
                    "error": str(exc),
                }
        parity = {"rowrun_vs_atomic": _empty_pairwise_summary()}
        if "rowrun" in route_outputs and "atomic" in route_outputs:
            parity["rowrun_vs_atomic"] = _pairwise_route_summary(
                route_outputs["rowrun"], route_outputs["atomic"], value_dtype
            )
        route_samples = route_outputs

    threshold = ast_ops._spmm_relative_threshold(value_dtype)
    return {
        "parameters": {
            "format": "coo",
            "internal_format": f"native-{route}",
            "route": route,
            "op": op_name,
            "dense_layout": layout,
            "b_stride": tuple(int(v) for v in prepared["native_B"].stride()),
            "c_stride": tuple(int(v) for v in triton_C.stride()),
            "n_rows": prepared["n_rows"],
            "n_cols": prepared["n_cols"],
            "nnz": int(nnz),
            "n_dense_cols": n_dense_cols,
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "warmup": warmup,
            "iters": iters,
            "block_n": launch["block_n"],
            "block_nnz": launch["block_nnz"],
            "required_nnz_tiles": launch["required_nnz_tiles"],
            "heuristic_warp_size": launch["heuristic_warp_size"],
            "heuristic_factor": launch["heuristic_factor"],
            "n_row_runs": n_row_runs,
            "run_cusparse": run_cusparse,
        },
        "performance": {
            "pytorch_ms": pytorch_ms,
            "triton_ms": triton_ms,
            "triton_gpu_ms": triton_timing.get("gpu_ms"),
            "process_cpu_ms": triton_timing.get("process_cpu_ms"),
            "process_gpu_ms": triton_timing.get("process_gpu_ms"),
            "compute_ms": triton_timing.get("compute_ms"),
            "triton_first_call_ms": triton_first_call_ms,
            "cusparse_ms": cusparse_ms,
            "triton_speedup_vs_pytorch": _speedup_ratio(pytorch_ms, triton_ms),
            "triton_speedup_vs_cusparse": _speedup_ratio(cusparse_ms, triton_ms),
        },
        "verification": {
            "triton_match_reference": triton_summary["ok_pt"],
            "triton_match_pytorch": triton_summary["ok_pt"],
            "triton_max_error": triton_summary["max_abs_error"],
            "triton_max_abs_error": triton_summary["max_abs_error"],
            "triton_max_relative_error": triton_summary["max_relative_error"],
            "triton_sum_relative_error": None,
            "triton_relative_threshold": threshold,
            "triton_strict_allclose_match": triton_summary["ok_pt"],
            "pytorch_match_reference": True,
            "cusparse_match_reference": cusparse_match,
            "cusparse_match_pytorch": cusparse_match,
            "cusparse_max_error": (
                cusparse_summary["max_abs_error"]
                if cusparse_summary is not None
                else None
            ),
            "cusparse_max_abs_error": (
                cusparse_summary["max_abs_error"]
                if cusparse_summary is not None
                else None
            ),
            "cusparse_max_relative_error": (
                cusparse_summary["max_relative_error"]
                if cusparse_summary is not None
                else None
            ),
            "cusparse_sum_relative_error": None,
            "cusparse_relative_threshold": threshold,
            "cusparse_strict_allclose_match": cusparse_match,
        },
        "backend_status": {
            "pytorch_unavailable_reason": pytorch_reason,
            "pytorch_sparse_format": pytorch_format,
            "cusparse_unavailable_reason": cusparse_reason,
            "flagsparse_internal_route": f"coo-native-{route}",
        },
        "samples": {
            "pytorch": pytorch_values,
            "triton": triton_C,
            "reference": expected,
            "cusparse": cusparse_values,
        },
        "route_results": route_results,
        "parity": parity,
        "route_samples": route_samples,
    }


def _print_spmm_coo_mtx_header(
    value_dtype, index_dtype, route, layout=None, timing=False
):
    route = _normalize_route(route)
    print(
        f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}"
    )
    if layout is not None:
        print(f"Dense layout: {layout}")
    print(
        f"Formats: FlagSparse={_route_label(route)}, cuSPARSE=COO dense-mm, PyTorch=COO."
    )
    print(
        "Timing: FS(ms)=process_cpu_ms+FS_GPU(ms); --timing adds process_gpu_ms/compute_ms split."
    )
    print(
        "Rowrun process includes COO execution-plan rebuild: coalesce/sort + seg_starts."
    )
    print(
        "Atomic has no current execution-plan preprocessing; host launch config and input normalization are excluded."
    )
    print(
        "Timing stays in native dtype. For float32, correctness references use float64 compute then cast."
    )
    print(
        "PT/CU show per-reference correctness. Err(PT)/Err(CU)=max(|diff| / (atol + rtol*|ref|))."
    )
    print("PyTorch uses COO sparse.mm as the only correctness reference path.")
    if route == "compare":
        print(
            "Compare mode also benchmarks native atomic (debug-only) after the main table."
        )
    width = 226 if timing else 202
    print("-" * width)
    split = f"{'ProcGPU':>9} {'Compute':>9} " if timing else ""
    print(
        f"{'Matrix':<28} {'Op':>5} {'Lay':>4} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} "
        f"{'FS(ms)':>9} {'FS_GPU':>9} {'CPUProc':>9} {split}"
        f"{'cuSPARSE':>9} {'PyTorch':>9} {'FS/CU':>7} {'FS/PT':>7} {'PT':>6} {'CU':>6} {'Err(PT)':>10} {'Err(CU)':>10}"
    )
    print("-" * width)


def _print_spmm_coo_mtx_row(entry, timing=False):
    name = os.path.basename(entry["path"])[:27]
    n_rows, n_cols = entry["shape"]
    triton_ms = entry.get("triton_ms")
    cu_ms = entry.get("cusparse_ms")
    pt_ms = entry.get("pytorch_ms")
    split = (
        f"{_fmt_ms(entry.get('process_gpu_ms')):>9} {_fmt_ms(entry.get('compute_ms')):>9} "
        if timing
        else ""
    )
    print(
        f"{name:<28} {entry.get('op', 'non'):>5} {entry.get('layout', 'row'):>4} {n_rows:>7} {n_cols:>7} {entry['nnz']:>10} {entry['dense_cols']:>8} "
        f"{_fmt_ms(triton_ms):>9} {_fmt_ms(entry.get('triton_gpu_ms')):>9} {_fmt_ms(entry.get('process_cpu_ms')):>9} {split}"
        f"{_fmt_ms(cu_ms):>9} {_fmt_ms(pt_ms):>9} "
        f"{_fmt_speedup(cu_ms, triton_ms):>7} {_fmt_speedup(pt_ms, triton_ms):>7} "
        f"{_fmt_check(entry.get('triton_ok_pt')):>6} {_fmt_check(entry.get('triton_ok_cu')):>6} "
        f"{_fmt_err(entry.get('err_pt')):>10} {_fmt_err(entry.get('err_cu')):>10}"
    )
    err = entry.get("error")
    if err:
        msg = str(err).replace("\n", " ")
        if len(msg) > 200:
            msg = msg[:197] + "..."
        print(f"  NOTE: {msg}")


def print_mtx_results(
    results, value_dtype, index_dtype, route="rowrun", layout=None, timing=False
):
    route = _normalize_route(route)
    _print_spmm_coo_mtx_header(
        value_dtype, index_dtype, route, layout=layout, timing=timing
    )
    for entry in results:
        _print_spmm_coo_mtx_row(entry, timing=timing)
    print("-" * (226 if timing else 202))


def print_compare_results(results, value_dtype, index_dtype):
    if not any(entry.get("compare") for entry in results):
        return

    print("Compare details (PT-COO / CU-COO / native parity)")
    print("Row/PT is the main default-route diagnostic; Atomic/PT is debug-only.")
    print("-" * 174)
    print(
        f"{'Matrix':<28} {'Lay':>4} {'Row/PT':>7} {'Atomic/PT':>9} {'CU/PT':>7} {'Row/Atomic':>11} "
        f"{'Err(Row/PT)':>12} {'Err(Atomic/PT)':>14} {'Err(CU/PT)':>10} {'Err(Row/Atomic)':>15}"
    )
    print("-" * 174)
    for entry in results:
        compare = entry.get("compare") or {}
        routes = compare.get("routes") or {}
        parity = compare.get("parity") or {}
        rowrun = routes.get("rowrun") or {}
        atomic = routes.get("atomic") or {}
        row_atomic = parity.get("rowrun_vs_atomic") or {}
        print(
            f"{os.path.basename(entry['path'])[:27]:<28} "
            f"{entry.get('layout', 'row'):>4} "
            f"{_fmt_check(rowrun.get('ok_pt')):>7} {_fmt_check(atomic.get('ok_pt')):>9} {_fmt_check(compare.get('cusparse_reference_match')):>7} "
            f"{_fmt_check(row_atomic.get('match')):>11} "
            f"{_fmt_err(rowrun.get('err_pt')):>12} {_fmt_err(atomic.get('err_pt')):>14} {_fmt_err(compare.get('cusparse_reference_error')):>10} "
            f"{_fmt_err(row_atomic.get('error_ratio')):>15}"
        )
    print("-" * 174)

    debug_rows = []
    for entry in results:
        compare = entry.get("compare") or {}
        debug = compare.get("debug")
        if debug is not None:
            debug_rows.append((os.path.basename(entry["path"])[:27], debug))
    if not debug_rows:
        return

    print("Worst mismatch summary for failing compare cases")
    print("-" * 178)
    print(
        f"{'Matrix':<28} {'Route':>8} {'Row':>8} {'DenseCol':>9} {'RowNNZ':>8} {'Err':>10} "
        f"{'Rowrun':>18} {'Atomic':>18} {'PT':>18} {'CU':>18}"
    )
    print("-" * 178)
    for name, debug in debug_rows:
        print(
            f"{name:<28} {debug['route']:>8} {debug['row']:>8} {debug['dense_col']:>9} {debug['row_nnz']:>8} {debug['error_ratio']:>10.2e} "
            f"{debug['rowrun']:>18} {debug['atomic']:>18} {debug['pt']:>18} {debug['cu']:>18}"
        )
    print("-" * 178)


def run_all_dtypes_export_csv(
    paths,
    csv_path,
    warmup=10,
    iters=50,
    run_cusparse=True,
    n_dense_cols=32,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    route="rowrun",
    value_dtypes=None,
    index_dtypes=None,
    op_names=None,
    layout_names=None,
    timing=False,
    alg_names=None,
    diagnose=False,
):
    alg_names = _parse_algs(None, route=route) if alg_names is None else alg_names
    csv_path = _normalize_csv_path(csv_path)
    rows = []
    diag_rows = []
    fieldnames = list(PERF_FIELDS)
    if timing:
        fieldnames += TIMING_FIELDS
    best_path = csv_path[:-4] + ".best.csv"
    diag_path = csv_path[:-4] + ".diagnose.csv"
    value_dtypes = CSV_VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = CSV_INDEX_DTYPES if index_dtypes is None else index_dtypes
    op_names = ["non"] if op_names is None else op_names
    layout_names = ["row"] if layout_names is None else layout_names
    for value_dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op_name in op_names:
                for layout_name in layout_names:
                    print("=" * 150)
                    print(
                        f"COO SpMM registry | dtype={_dtype_name(value_dtype)} index={_dtype_name(index_dtype)} "
                        f"op={op_name} layout={layout_name} alg={','.join(alg_names)}",
                        flush=True,
                    )
                    for matrix_idx, path in enumerate(paths, start=1):
                        print(
                            f"[{matrix_idx}/{len(paths)}] [{os.path.basename(path)}] START",
                            flush=True,
                        )
                        case_rows, case_diag_rows = run_one_alg_case(
                            path,
                            value_dtype,
                            _dtype_name(index_dtype),
                            index_dtype,
                            op_name,
                            layout_name,
                            alg_names,
                            n_dense_cols,
                            warmup,
                            iters,
                            run_cusparse,
                            timing,
                            diagnose,
                            progress=True,
                        )
                        rows.extend(case_rows)
                        diag_rows.extend(case_diag_rows)
                        for row in case_rows:
                            print(
                                f"{row['matrix']:<32} {row['alg']:<16} {row['status']:<5} "
                                f"ms={_fmt_ms(row['ms'])} torch={_fmt_ms(row['torch_ms'])} "
                                f"err={_fmt_err(row['err_vs_torch'])} reason={row.get('reason') or row.get('cusparse_reason') or ''}",
                                flush=True,
                            )
    _write_csv(csv_path, rows, fieldnames)
    _write_csv(best_path, _best_rows(rows), BEST_FIELDS)
    if diagnose:
        _write_csv(diag_path, diag_rows, DIAG_FIELDS)
        print(f"Wrote {len(diag_rows)} diagnose rows to {diag_path}")
    print(f"Wrote best rows to {best_path}")
    print(f"Wrote {len(rows)} rows to {csv_path}")


def run_api_validation_checks():
    if not torch.cuda.is_available():
        print("API checks skipped: CUDA is not available.")
        return 0

    device = torch.device("cuda")
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    row = torch.tensor([0, 0, 1], dtype=torch.int32, device=device)
    col = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    B = torch.randn((2, 4), dtype=torch.float32, device=device)
    dup_data, dup_row, dup_col, dup_B, dup_shape = _build_duplicate_unsorted_case(
        torch.float32, torch.int32, device
    )

    negative_cases = [
        (
            "shape must be 2D",
            lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2,)),
            ValueError,
        ),
        (
            "B must be 2D",
            lambda: ast.flagsparse_spmm_coo(data, row, col, B[0], (2, 2)),
            ValueError,
        ),
        (
            "dtype mismatch",
            lambda: ast.flagsparse_spmm_coo(
                data, row, col, B.to(torch.float64), (2, 2)
            ),
            TypeError,
        ),
        (
            "shape mismatch",
            lambda: ast.flagsparse_spmm_coo(
                data,
                row,
                col,
                torch.randn((3, 4), dtype=torch.float32, device=device),
                (2, 2),
            ),
            ValueError,
        ),
        (
            "row length mismatch",
            lambda: ast.flagsparse_spmm_coo(data, row[:-1], col, B, (2, 2)),
            ValueError,
        ),
        (
            "col length mismatch",
            lambda: ast.flagsparse_spmm_coo(data, row, col[:-1], B, (2, 2)),
            ValueError,
        ),
        (
            "row out of range",
            lambda: ast.flagsparse_spmm_coo(
                data,
                torch.tensor([0, 2, 1], dtype=torch.int32, device=device),
                col,
                B,
                (2, 2),
            ),
            IndexError,
        ),
        (
            "col out of range",
            lambda: ast.flagsparse_spmm_coo(
                data,
                row,
                torch.tensor([0, 3, 1], dtype=torch.int32, device=device),
                B,
                (2, 2),
            ),
            IndexError,
        ),
        (
            "block_n positive",
            lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2, 2), block_n=0),
            ValueError,
        ),
        (
            "block_nnz positive",
            lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2, 2), block_nnz=0),
            ValueError,
        ),
        (
            "invalid op",
            lambda: ast.flagsparse_spmm_coo(data, row, col, B, (2, 2), op="bad"),
            ValueError,
        ),
        (
            "op transpose conflict",
            lambda: ast.flagsparse_spmm_coo(
                data, row, col, B, (2, 2), op="non", transpose=True
            ),
            ValueError,
        ),
        (
            "out shape mismatch",
            lambda: ast.flagsparse_spmm_coo(
                data,
                row,
                col,
                B,
                (2, 2),
                out=torch.empty((3, 4), dtype=torch.float32, device=device),
            ),
            ValueError,
        ),
        (
            "trans out shape mismatch",
            lambda: ast.flagsparse_spmm_coo(
                data,
                row,
                col,
                torch.randn((2, 4), dtype=torch.float32, device=device),
                (2, 3),
                op="trans",
                out=torch.empty((2, 4), dtype=torch.float32, device=device),
            ),
            ValueError,
        ),
        (
            "out device mismatch",
            lambda: ast.flagsparse_spmm_coo(
                data, row, col, B, (2, 2), out=torch.empty((2, 4), dtype=torch.float32)
            ),
            ValueError,
        ),
    ]

    failed = 0
    print("-" * 96)
    print("API validation checks")
    print("-" * 96)
    for name, fn, exc_type in negative_cases:
        try:
            fn()
            print(f"FAIL  {name:<32} expected {exc_type.__name__}")
            failed += 1
        except exc_type:
            print(f"PASS  {name:<32} raised {exc_type.__name__}")
        except Exception as exc:
            print(f"FAIL  {name:<32} raised {type(exc).__name__}: {exc}")
            failed += 1

    positive_checks = []

    def _positive_out_path():
        out = torch.empty((2, 4), dtype=torch.float32, device=device)
        _assert_spmm_coo_matches_reference(
            data, row, col, B, (2, 2), torch.float32, out=out
        )

    positive_checks.append(("out path success", _positive_out_path))

    def _positive_trans_path():
        _assert_spmm_coo_matches_reference(
            data, row, col, B, (2, 2), torch.float32, op="trans"
        )

    positive_checks.append(("trans op success", _positive_trans_path))

    def _positive_conj_path():
        complex_data = data.to(torch.complex64) * (1 + 2j)
        complex_B = B.to(torch.complex64) * (2 - 1j)
        _assert_spmm_coo_matches_reference(
            complex_data,
            row,
            col,
            complex_B,
            (2, 2),
            torch.complex64,
            op="conj",
        )

    positive_checks.append(("conj op success", _positive_conj_path))

    def _positive_empty_matrix():
        empty_data = torch.tensor([], dtype=torch.float32, device=device)
        empty_row = torch.tensor([], dtype=torch.int32, device=device)
        empty_col = torch.tensor([], dtype=torch.int32, device=device)
        dense = torch.randn((2, 4), dtype=torch.float32, device=device)
        result, _ = _assert_spmm_coo_matches_reference(
            empty_data,
            empty_row,
            empty_col,
            dense,
            (2, 2),
            torch.float32,
        )
        if result.shape != (2, 4):
            raise AssertionError(
                f"unexpected empty-matrix result shape: {tuple(result.shape)}"
            )

    positive_checks.append(("empty matrix success", _positive_empty_matrix))

    def _positive_empty_dense_cols():
        dense = torch.empty((2, 0), dtype=torch.float32, device=device)
        result, _ = _assert_spmm_coo_matches_reference(
            data,
            row,
            col,
            dense,
            (2, 2),
            torch.float32,
        )
        if result.shape != (2, 0):
            raise AssertionError(
                f"unexpected empty-dense result shape: {tuple(result.shape)}"
            )

    positive_checks.append(("empty dense cols success", _positive_empty_dense_cols))

    def _positive_noncontiguous_b():
        dense = _build_dense_matrix(4, 2, torch.float32, device).transpose(0, 1)
        if dense.is_contiguous():
            raise AssertionError("expected non-contiguous test matrix")
        _assert_spmm_coo_matches_reference(data, row, col, dense, (2, 2), torch.float32)

    positive_checks.append(("noncontiguous B success", _positive_noncontiguous_b))

    def _positive_col_layout():
        dense = _materialize_dense_layout_for_test(B, "col")
        result, _ = _assert_spmm_coo_matches_reference(
            data,
            row,
            col,
            dense,
            (2, 2),
            torch.float32,
            layout="col",
        )
        if tuple(result.stride()) != (1, result.shape[0]):
            raise AssertionError(
                f"unexpected col-major output stride: {tuple(result.stride())}"
            )

    positive_checks.append(("col layout success", _positive_col_layout))

    def _positive_complex_col_layout():
        complex_data = data.to(torch.complex64) * (1 + 2j)
        complex_B = _materialize_dense_layout_for_test(
            B.to(torch.complex64) * (2 - 1j), "col"
        )
        result, _ = _assert_spmm_coo_matches_reference(
            complex_data,
            row,
            col,
            complex_B,
            (2, 2),
            torch.complex64,
            layout="col",
        )
        if tuple(result.stride()) != (1, result.shape[0]):
            raise AssertionError(
                f"unexpected complex col-major output stride: {tuple(result.stride())}"
            )

    positive_checks.append(("complex col layout success", _positive_complex_col_layout))

    def _positive_unsorted_duplicate():
        _assert_spmm_coo_matches_reference(
            dup_data,
            dup_row,
            dup_col,
            dup_B,
            dup_shape,
            torch.float32,
        )

    positive_checks.append(("unsorted duplicate success", _positive_unsorted_duplicate))

    for name, fn in positive_checks:
        try:
            fn()
            print(f"PASS  {name:<32} returned correct result")
        except Exception as exc:
            print(f"FAIL  {name:<32} raised {type(exc).__name__}: {exc}")
            failed += 1

    print("-" * 96)
    return failed


def run_coo_tile_branch_coverage(warmup=WARMUP, iters=ITERS, run_cusparse=True):
    if not torch.cuda.is_available():
        print("COO branch coverage skipped: CUDA is not available.")
        return 0

    print("=" * 144)
    print("COO native row-run dense-column coverage")
    print("=" * 144)
    print(
        f"{'DenseN':>8} {'BLOCK_N':>8} {'NNZTile':>8} {'Runs':>7} {'Tiles':>7} {'Warp':>6} {'Factor':>7} "
        f"{'PyTorch(ms)':>12} {'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>12} {'PT':>6} {'CU':>6} {'Err(FS)':>11}"
    )
    print("-" * 144)

    failed = 0
    note = None
    for n_rows, n_cols, nnz, n_dense_cols in COO_TILE_CASES:
        result = ast_ops.benchmark_spmm_coo_case(
            n_rows=n_rows,
            n_cols=n_cols,
            nnz=nnz,
            n_dense_cols=n_dense_cols,
            value_dtype=torch.float32,
            index_dtype=torch.int32,
            warmup=warmup,
            iters=iters,
            block_n=DEFAULT_BLOCK_N,
            block_nnz=DEFAULT_BLOCK_NNZ,
            run_cusparse=run_cusparse,
        )
        params = result["parameters"]
        perf = result["performance"]
        verify = result["verification"]
        backend = result["backend_status"]
        samples = result["samples"]
        triton_ok = verify.get(
            "triton_strict_allclose_match", verify.get("triton_match_reference")
        )
        cusparse_ok = verify.get(
            "cusparse_strict_allclose_match", verify.get("cusparse_match_reference")
        )
        status = "PASS" if triton_ok else "FAIL"
        if status != "PASS":
            failed += 1
        if backend.get("cusparse_unavailable_reason"):
            note = backend["cusparse_unavailable_reason"]
        triton_err = _scaled_allclose_error(
            samples["triton"], samples["reference"], torch.float32
        )
        print(
            f"{n_dense_cols:>8} {params['block_n']:>8} {params['block_nnz']:>8} {params['n_row_runs']:>7} {params['required_nnz_tiles']:>7} {params['heuristic_warp_size']:>6} {params['heuristic_factor']:>7} "
            f"{_fmt_ms(perf.get('pytorch_ms')):>12} {_fmt_ms(perf.get('triton_ms')):>14} {_fmt_ms(perf.get('cusparse_ms')):>12} "
            f"{_fmt_check(triton_ok):>6} {_fmt_check(cusparse_ok):>6} {_fmt_err(triton_err):>11}"
        )
    print("-" * 144)
    if note:
        print(f"cuSPARSE note: {note}")
    print()
    return failed


def _print_synthetic_compare_results(compare_rows):
    if not compare_rows:
        return

    print("Compare details (PT-COO / CU-COO / native parity)")
    print("Row/PT is the main default-route diagnostic; Atomic/PT is debug-only.")
    print("-" * 168)
    print(
        f"{'Lay':>4} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} {'Row/PT':>7} {'Atomic/PT':>9} {'CU/PT':>7} {'Row/Atomic':>11} "
        f"{'Err(Row/PT)':>12} {'Err(Atomic/PT)':>14} {'Err(CU/PT)':>10} {'Err(Row/Atomic)':>15}"
    )
    print("-" * 168)
    for entry in compare_rows:
        print(
            f"{entry.get('layout', 'row'):>4} {entry['n_rows']:>7} {entry['n_cols']:>7} {entry['nnz']:>10} {entry['dense_cols']:>8} "
            f"{_fmt_check(entry.get('row_pt')):>7} {_fmt_check(entry.get('atomic_pt')):>9} {_fmt_check(entry.get('cu_pt')):>7} "
            f"{_fmt_check(entry.get('row_atomic')):>11} "
            f"{_fmt_err(entry.get('err_row_pt')):>12} {_fmt_err(entry.get('err_atomic_pt')):>14} {_fmt_err(entry.get('err_cu_pt')):>10} "
            f"{_fmt_err(entry.get('err_row_atomic')):>15}"
        )
    print("-" * 168)
    print()


def run_comprehensive_synthetic(
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    run_api_checks=True,
    run_coo_coverage=True,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    route="rowrun",
    op_names=None,
    layout_names=None,
    timing=False,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    route = _normalize_route(route)
    selected_route = _selected_route(route)
    op_names = ["non"] if op_names is None else op_names
    layout_names = ["row"] if layout_names is None else layout_names

    print("=" * 150)
    print("FLAGSPARSE SpMM BENCHMARK (synthetic COO @ dense)")
    print("=" * 150)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  |  Warmup: {warmup}  Iters: {iters}  "
        f"BLOCK_N: {_fmt_launch_value(block_n)}  BLOCK_NNZ: {_fmt_launch_value(block_nnz)}  Route: {route}  Ops: {','.join(op_names)}  Layouts: {','.join(layout_names)}"
    )
    print(
        f"Formats: FlagSparse={_route_label(route)}, cuSPARSE=COO dense-mm (when supported), PyTorch=COO."
    )
    print(
        "Timing: FS(ms)=process_cpu_ms+FS_GPU(ms); --timing adds process_gpu_ms/compute_ms split."
    )
    print(
        "Rowrun process includes COO execution-plan rebuild: coalesce/sort + seg_starts."
    )
    print(
        "Atomic has no current execution-plan preprocessing; host launch config and input normalization are excluded."
    )
    print(
        "For float32, PT checks the float64-based correctness reference while CU reflects native cuSPARSE float32 consistency."
    )
    if route == "compare":
        print(
            "Compare mode also benchmarks native atomic (debug-only) for each synthetic case."
        )
    print()

    total = 0
    failed = 0
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            compare_rows = []
            width = 194 if timing else 174
            print("-" * width)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<12}  |  Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * width)
            split_header = f"{'ProcGPU':>9} {'Compute':>9} " if timing else ""
            print(
                f"{'Op':>5} {'Lay':>4} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} {'BN':>4} {'BNNZ':>6} {'Runs':>5} {'Tiles':>5} "
                f"{'FS(ms)':>9} {'FS_GPU':>9} {'CPUProc':>9} {split_header}"
                f"{'PyTorch':>9} {'cuSPARSE':>9} {'FS/PT':>8} {'FS/CU':>8} {'PT':>6} {'CU':>6} {'Err(FS)':>11} {'Err(CU)':>12}"
            )
            print("-" * width)
            combo_reason = None
            for op_name in op_names:
                for layout_name in layout_names:
                    for n_rows, n_cols, nnz, n_dense_cols in TEST_CASES:
                        result = _benchmark_spmm_coo_synthetic_policy(
                            n_rows=n_rows,
                            n_cols=n_cols,
                            nnz=nnz,
                            n_dense_cols=n_dense_cols,
                            value_dtype=value_dtype,
                            index_dtype=index_dtype,
                            warmup=warmup,
                            iters=iters,
                            block_n=block_n,
                            block_nnz=block_nnz,
                            run_cusparse=run_cusparse,
                            route=selected_route,
                            compare_routes=(route == "compare"),
                            op=op_name,
                            dense_layout=layout_name,
                            timing=timing,
                        )
                        total += 1
                        params = result["parameters"]
                        perf = result["performance"]
                        verify = result["verification"]
                        backend = result["backend_status"]
                        samples = result["samples"]
                        triton_ok = verify.get(
                            "triton_strict_allclose_match",
                            verify.get("triton_match_reference"),
                        )
                        cusparse_ok = verify.get(
                            "cusparse_strict_allclose_match",
                            verify.get("cusparse_match_reference"),
                        )
                        status = "PASS" if triton_ok else "FAIL"
                        if status != "PASS":
                            failed += 1
                        if backend.get("cusparse_unavailable_reason"):
                            combo_reason = backend["cusparse_unavailable_reason"]
                        triton_err = _scaled_allclose_error(
                            samples["triton"], samples["reference"], value_dtype
                        )
                        cusparse_err = None
                        if samples.get("cusparse") is not None:
                            cusparse_err = _scaled_allclose_error(
                                samples["triton"], samples["cusparse"], value_dtype
                            )
                        split_values = (
                            f"{_fmt_ms(perf.get('process_gpu_ms')):>9} {_fmt_ms(perf.get('compute_ms')):>9} "
                            if timing
                            else ""
                        )
                        print(
                            f"{op_name:>5} {layout_name:>4} {n_rows:>7} {n_cols:>7} {nnz:>10} {n_dense_cols:>8} {params['block_n']:>4} {params['block_nnz']:>6} {params['n_row_runs']:>5} {params['required_nnz_tiles']:>5} "
                            f"{_fmt_ms(perf.get('triton_ms')):>9} {_fmt_ms(perf.get('triton_gpu_ms')):>9} {_fmt_ms(perf.get('process_cpu_ms')):>9} {split_values}"
                            f"{_fmt_ms(perf.get('pytorch_ms')):>9} {_fmt_ms(perf.get('cusparse_ms')):>9} "
                            f"{_fmt_speedup(perf.get('pytorch_ms'), perf.get('triton_ms')):>8} {_fmt_speedup(perf.get('cusparse_ms'), perf.get('triton_ms')):>8} "
                            f"{_fmt_check(triton_ok):>6} {_fmt_check(cusparse_ok):>6} {_fmt_err(triton_err):>11} {_fmt_err(cusparse_err):>12}"
                        )
                        if route == "compare":
                            route_results = result.get("route_results") or {}
                            parity = result.get("parity") or {}
                            compare_rows.append(
                                {
                                    "layout": layout_name,
                                    "n_rows": n_rows,
                                    "n_cols": n_cols,
                                    "nnz": nnz,
                                    "dense_cols": n_dense_cols,
                                    "row_pt": (route_results.get("rowrun") or {}).get(
                                        "match_reference"
                                    ),
                                    "atomic_pt": (
                                        route_results.get("atomic") or {}
                                    ).get("match_reference"),
                                    "cu_pt": verify.get("cusparse_match_reference"),
                                    "row_atomic": (
                                        parity.get("rowrun_vs_atomic") or {}
                                    ).get("match"),
                                    "err_row_pt": (
                                        route_results.get("rowrun") or {}
                                    ).get("error_ratio"),
                                    "err_atomic_pt": (
                                        route_results.get("atomic") or {}
                                    ).get("error_ratio"),
                                    "err_cu_pt": (
                                        verify.get("cusparse_max_relative_error")
                                        if verify.get("cusparse_match_reference")
                                        is not None
                                        else None
                                    ),
                                    "err_row_atomic": (
                                        parity.get("rowrun_vs_atomic") or {}
                                    ).get("error_ratio"),
                                }
                            )
            print("-" * width)
            if combo_reason:
                print(f"  cuSPARSE: {combo_reason}")
            print()
            if route == "compare":
                _print_synthetic_compare_results(compare_rows)

    coo_failed = 0
    if run_coo_coverage:
        if route == "rowrun":
            coo_failed = run_coo_tile_branch_coverage(
                warmup=warmup, iters=iters, run_cusparse=run_cusparse
            )
        else:
            print(
                f"COO dense-column coverage is row-run specific; skipped for route {route}."
            )
            print()
    api_failed = run_api_validation_checks() if run_api_checks else 0
    print("=" * 150)
    print(
        f"Total synthetic cases: {total}  Failed synthetic cases: {failed}  "
        f"Failed COO branch cases: {coo_failed}  Failed API checks: {api_failed}"
    )
    print("=" * 150)


def main():
    parser = argparse.ArgumentParser(
        description="COO SpMM test: SuiteSparse .mtx batch run, error and performance."
    )
    parser.add_argument(
        "mtx", nargs="*", help=".mtx file path(s), or directory(ies) to glob for *.mtx"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run synthetic benchmark instead of .mtx",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=[
            "float16",
            "bfloat16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        ],
        help="Value dtype (default: float32)",
    )
    parser.add_argument(
        "--index-dtype",
        default="int32",
        choices=["int32", "int64"],
        help="Index dtype (default: int32)",
    )
    parser.add_argument(
        "--dtypes",
        default="float32,float64,complex64,complex128",
        help="Comma-separated value dtype grid for CSV export: float32,float64,complex64,complex128",
    )
    parser.add_argument(
        "--index-dtypes",
        default="int32,int64",
        help="Comma-separated index dtype grid for CSV export: int32,int64",
    )
    parser.add_argument(
        "--op",
        default="non",
        help="SpMM op: non, trans, conj, all, or comma-separated list",
    )
    parser.add_argument(
        "--layout",
        default="row",
        choices=["row", "col", "all"],
        help="Dense RHS/output layout: row, col, or all",
    )
    parser.add_argument(
        "--dense-cols", type=int, default=32, help="Dense RHS column count"
    )
    parser.add_argument(
        "--block-n",
        type=int,
        default=DEFAULT_BLOCK_N,
        help="Output column tile override (default: auto from dense-column heuristic)",
    )
    parser.add_argument(
        "--block-nnz",
        type=int,
        default=DEFAULT_BLOCK_NNZ,
        help="COO nnz tile width override (default: 256)",
    )
    parser.add_argument(
        "--route",
        default="rowrun",
        choices=["rowrun", "atomic", "compare"],
        help="Native COO route to benchmark/test (default: rowrun)",
    )
    parser.add_argument(
        "--alg",
        default=None,
        help="COO SpMM algorithm: auto, all, or comma-separated registered names",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Add process_gpu_ms/compute_ms split timing columns",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Write algorithm diagnostics to .diagnose.csv",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument(
        "--no-cusparse", action="store_true", help="Skip cuSPARSE baseline"
    )
    parser.add_argument(
        "--skip-api-checks",
        action="store_true",
        help="Skip API validation checks in synthetic mode",
    )
    parser.add_argument(
        "--skip-coo-coverage",
        action="store_true",
        help="Skip dense-column COO heuristic coverage in synthetic mode",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="FILE",
        help="Run selected dtype/index grids on all .mtx and write results to one CSV",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    index_map = {"int32": torch.int32, "int64": torch.int64}
    value_dtype = dtype_map[args.dtype]
    index_dtype = index_map[args.index_dtype]
    try:
        op_names = _parse_op_names(args.op)
        layout_names = _layout_names(args.layout)
        alg_names = _parse_algs(args.alg, route=args.route)
    except ValueError as exc:
        parser.error(str(exc))

    if args.synthetic:
        run_comprehensive_synthetic(
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            run_api_checks=not args.skip_api_checks,
            run_coo_coverage=not args.skip_coo_coverage,
            block_n=args.block_n,
            block_nnz=args.block_nnz,
            route=args.route,
            op_names=op_names,
            layout_names=layout_names,
            timing=args.timing,
        )
        return

    paths = []
    for path in args.mtx:
        if os.path.isfile(path) and path.endswith(".mtx"):
            paths.append(path)
        elif os.path.isdir(path):
            paths.extend(sorted(glob.glob(os.path.join(path, "*.mtx"))))

    if not paths and not args.csv:
        print(
            "No .mtx files given. Use: python test_spmm_coo.py <file.mtx> [file2.mtx ...] or <dir/>"
        )
        print("Or run synthetic: python test_spmm_coo.py --synthetic")
        print(
            "Or run all dtypes and export CSV: python test_spmm_coo.py <dir/> --csv results.csv"
        )
        return

    if args.csv is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        if args.route == "compare" and args.alg is None:
            print("CSV export only supports --route rowrun or --route atomic.")
            return
        csv_path = _normalize_csv_path(args.csv)
        print("=" * 100)
        print("FLAGSPARSE COO SpMM - selected dtype/index grid, export to CSV")
        print("=" * 100)
        try:
            csv_value_dtypes = _parse_csv_tokens(args.dtypes, dtype_map, "--dtypes")
            csv_index_dtypes = _parse_csv_tokens(
                args.index_dtypes, index_map, "--index-dtypes"
            )
            csv_op_names = _parse_op_names(args.op)
            csv_layout_names = _layout_names(args.layout)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  DenseN: {args.dense_cols}  |  Alg: {args.alg or args.route}  |  CSV: {csv_path}"
        )
        print(
            f"dtypes: {args.dtypes}  |  index_dtypes: {args.index_dtypes}  |  ops: {args.op}  |  layouts: {args.layout}"
        )
        run_all_dtypes_export_csv(
            paths,
            csv_path,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            n_dense_cols=args.dense_cols,
            block_n=args.block_n,
            block_nnz=args.block_nnz,
            route=args.route,
            value_dtypes=csv_value_dtypes,
            index_dtypes=csv_index_dtypes,
            op_names=csv_op_names,
            layout_names=csv_layout_names,
            timing=args.timing,
            alg_names=alg_names,
            diagnose=args.diagnose,
        )
        return

    print("=" * 140)
    print("FLAGSPARSE COO SpMM - SuiteSparse .mtx batch (error + performance)")
    print("=" * 140)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(
        f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  dense_cols: {args.dense_cols}  "
        f"op: {args.op}  layout: {args.layout}  warmup: {args.warmup}  iters: {args.iters}  block_n: {_fmt_launch_value(args.block_n)}  "
        f"block_nnz: {_fmt_launch_value(args.block_nnz)}  alg: {args.alg or args.route}"
    )
    print()
    if args.route == "compare" and args.alg is None:
        for op_name in op_names:
            for layout_name in layout_names:
                results = run_mtx_batch(
                    paths,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    run_cusparse=not args.no_cusparse,
                    n_dense_cols=args.dense_cols,
                    block_n=args.block_n,
                    block_nnz=args.block_nnz,
                    route=args.route,
                    op=op_name,
                    layout=layout_name,
                    timing=args.timing,
                )
                print_mtx_results(
                    results,
                    value_dtype,
                    index_dtype,
                    route=args.route,
                    layout=layout_name,
                    timing=args.timing,
                )
                print_compare_results(results, value_dtype, index_dtype)
        return

    for op_name in op_names:
        for layout_name in layout_names:
            for path in paths:
                rows, _diag_rows = run_one_alg_case(
                    path,
                    value_dtype,
                    _dtype_name(index_dtype),
                    index_dtype,
                    op_name,
                    layout_name,
                    alg_names,
                    args.dense_cols,
                    args.warmup,
                    args.iters,
                    not args.no_cusparse,
                    args.timing,
                    args.diagnose,
                )
                for row in rows:
                    print(
                        f"{row['matrix']:<32} {row['alg']:<16} {row['status']:<5} "
                        f"ms={_fmt_ms(row['ms'])} gpu={_fmt_ms(row['gpu_ms'])} "
                        f"torch={_fmt_ms(row['torch_ms'])} err={_fmt_err(row['err_vs_torch'])} "
                        f"reason={row.get('reason') or row.get('cusparse_reason') or ''}"
                    )


if __name__ == "__main__":
    main()
