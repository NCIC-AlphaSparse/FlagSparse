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
Experimental AlphaSparse ALG1 benchmark: base vs alpha_spmm_alg1_tle_opt vs
alpha_spmm_alg1_tle_opt2, with Torch and CuPy/cuSPARSE references.
"""

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

from flagsparse.sparse_operations.spmm_csr import (
    _normalize_spmm_base_device_props,
    _prepare_spmm_csr_inputs,
    _resolve_spmm_base_triton_launch,
    _triton_spmm_csr_impl,
)
from test_spmm_opt import _seeded_dense_matrix, load_mtx_to_csr_torch

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 50
DEFAULT_DENSE_COLS = 32
DEFAULT_SEED = 0

ROUTES = (
    "base",
    "alpha_spmm_alg1_tle_opt",
    "alpha_spmm_alg1_tle_opt2",
    "torch_ref",
    "cupy_cusparse_ref",
)

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
]
for _route in ROUTES:
    SUMMARY_FIELDS.extend(
        [
            f"{_route}_symbolic_ms",
            f"{_route}_compute_ms",
            f"{_route}_total_ms",
        ]
    )
SUMMARY_FIELDS.extend(
    [
        "torch_ref_compute_speedup_vs_base",
        "torch_ref_compute_speedup_vs_alpha_spmm_alg1_tle_opt",
        "torch_ref_compute_speedup_vs_alpha_spmm_alg1_tle_opt2",
        "cupy_cusparse_ref_compute_speedup_vs_base",
        "cupy_cusparse_ref_compute_speedup_vs_alpha_spmm_alg1_tle_opt",
        "cupy_cusparse_ref_compute_speedup_vs_alpha_spmm_alg1_tle_opt2",
        "alpha_spmm_alg1_tle_opt2_compute_speedup_vs_alpha_spmm_alg1_tle_opt",
        "torch_ref_total_speedup_vs_base",
        "torch_ref_total_speedup_vs_alpha_spmm_alg1_tle_opt",
        "torch_ref_total_speedup_vs_alpha_spmm_alg1_tle_opt2",
        "cupy_cusparse_ref_total_speedup_vs_base",
        "cupy_cusparse_ref_total_speedup_vs_alpha_spmm_alg1_tle_opt",
        "cupy_cusparse_ref_total_speedup_vs_alpha_spmm_alg1_tle_opt2",
        "alpha_spmm_alg1_tle_opt2_total_speedup_vs_alpha_spmm_alg1_tle_opt",
        "base_vs_torch_ref_err",
        "alpha_spmm_alg1_tle_opt_vs_torch_ref_err",
        "alpha_spmm_alg1_tle_opt2_vs_torch_ref_err",
        "base_vs_cupy_cusparse_ref_err",
        "alpha_spmm_alg1_tle_opt_vs_cupy_cusparse_ref_err",
        "alpha_spmm_alg1_tle_opt2_vs_cupy_cusparse_ref_err",
        "base_status_vs_torch_ref",
        "alpha_spmm_alg1_tle_opt_status_vs_torch_ref",
        "alpha_spmm_alg1_tle_opt2_status_vs_torch_ref",
        "base_status_vs_cupy_cusparse_ref",
        "alpha_spmm_alg1_tle_opt_status_vs_cupy_cusparse_ref",
        "alpha_spmm_alg1_tle_opt2_status_vs_cupy_cusparse_ref",
        "alpha_spmm_alg1_tle_opt_status",
        "alpha_spmm_alg1_tle_opt_reason",
        "alpha_spmm_alg1_tle_opt2_status",
        "alpha_spmm_alg1_tle_opt2_reason",
        "torch_ref_status",
        "torch_ref_reason",
        "cupy_cusparse_ref_status",
        "cupy_cusparse_ref_reason",
        "matrix_status",
    ]
)

LAUNCH_FIELDS = [
    "matrix",
    "route",
    "device_name",
    "sm_count",
    "dtype",
    "dense_cols",
    "warp_size",
    "factor",
    "block_size",
    "block_rows",
    "block_cols",
    "num_warps",
    "num_stages",
    "loop_strategy",
    "launch_version",
    "grid_m",
    "grid_n",
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


def _ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return float(numerator) / float(denominator)


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
    return {"global_err": global_err, "status": _status_from_error(global_err)}


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


def _prepare_base_inputs(data, indices, indptr, B, shape):
    prepared_inputs = _prepare_spmm_csr_inputs(data, indices, indptr, B, shape)
    data_p, indices_p, indptr_p, B_p, n_rows, _n_cols, n_dense_cols = prepared_inputs
    max_row_nnz = (
        int(torch.max(indptr_p[1:] - indptr_p[:-1]).item()) if n_rows > 0 else 0
    )
    device_props = _normalize_spmm_base_device_props(data_p.device)
    launch = _resolve_spmm_base_triton_launch(
        data_p.dtype,
        n_dense_cols,
        max_row_nnz,
        device_props=device_props,
    )
    return {
        "data": data_p,
        "indices": indices_p,
        "indptr": indptr_p,
        "B": B_p,
        "n_rows": n_rows,
        "n_dense_cols": n_dense_cols,
        "launch": launch,
    }


def _timed_spmm_base(data, indices, indptr, B, shape, warmup, iters):
    prepared = _prepare_base_inputs(data, indices, indptr, B, shape)
    out, compute_ms = _benchmark(
        lambda: _triton_spmm_csr_impl(
            prepared["data"],
            prepared["indices"],
            prepared["indptr"],
            prepared["B"],
            prepared["n_rows"],
            prepared["n_dense_cols"],
            block_n=prepared["launch"]["block_n"],
            block_nnz=prepared["launch"]["block_nnz"],
            num_warps=prepared["launch"]["num_warps"],
            num_stages=prepared["launch"]["num_stages"],
        ),
        warmup,
        iters,
    )
    symbolic_ms = 0.0
    return out, symbolic_ms, compute_ms, symbolic_ms + compute_ms, prepared


def _timed_alpha_spmm_alg1_tle_opt(data, indices, indptr, B, shape, warmup, iters):
    if not fs.is_alpha_spmm_alg1_tle_opt_available():
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            fs.alpha_spmm_alg1_tle_opt_unavailable_reason(),
        )
    try:
        prepared = fs.prepare_alpha_spmm_alg1_tle_opt(data, indices, indptr, shape)
        meta = fs.build_alpha_spmm_alg1_tle_opt_meta(prepared, B)
        out, compute_ms = _benchmark(
            lambda: fs.flagsparse_alpha_spmm_alg1_tle_opt(
                B=B, prepared=prepared, meta=meta
            ),
            warmup,
            iters,
        )
    except Exception as exc:
        return None, None, None, None, None, None, f"{type(exc).__name__}: {exc}"
    symbolic_ms = 0.0
    return out, symbolic_ms, compute_ms, symbolic_ms + compute_ms, prepared, meta, None


def _timed_alpha_spmm_alg1_tle_opt2(data, indices, indptr, B, shape, warmup, iters):
    if not fs.is_alpha_spmm_alg1_tle_opt2_available():
        return (
            None,
            None,
            None,
            None,
            None,
            None,
            fs.alpha_spmm_alg1_tle_opt2_unavailable_reason(),
        )
    try:
        prepared = fs.prepare_alpha_spmm_alg1_tle_opt2(data, indices, indptr, shape)
        meta = fs.build_alpha_spmm_alg1_tle_opt2_meta(prepared, B)
        out, compute_ms = _benchmark(
            lambda: fs.flagsparse_alpha_spmm_alg1_tle_opt2(
                B=B, prepared=prepared, meta=meta
            ),
            warmup,
            iters,
        )
    except Exception as exc:
        return None, None, None, None, None, None, f"{type(exc).__name__}: {exc}"
    symbolic_ms = 0.0
    return out, symbolic_ms, compute_ms, symbolic_ms + compute_ms, prepared, meta, None


def _timed_torch_reference(data, indices, indptr, B, shape, dtype, warmup, iters):
    device = data.device
    try:
        sparse = torch.sparse_csr_tensor(
            indptr.to(torch.int64),
            indices.to(torch.int64),
            data,
            size=shape,
            device=device,
        )
        out, compute_ms = _benchmark(lambda: torch.sparse.mm(sparse, B), warmup, iters)
        if dtype == torch.float32:
            ref_sparse = torch.sparse_csr_tensor(
                indptr.to(torch.int64),
                indices.to(torch.int64),
                data.double(),
                size=shape,
                device=device,
            )
            out = torch.sparse.mm(ref_sparse, B.double()).float()
    except Exception as exc:
        return None, None, None, None, f"{type(exc).__name__}: {exc}"
    symbolic_ms = 0.0
    return out, symbolic_ms, compute_ms, symbolic_ms + compute_ms, None


def _timed_sparse_backend(data, indices, indptr, B, shape, warmup, iters, enabled):
    if not enabled:
        return None, None, None, None, "disabled"
    try:
        import cupy as cp
        import cupyx.scipy.sparse as cpx
    except Exception as exc:
        return None, None, None, None, str(exc)

    try:

        def prepare():
            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
            ind_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(indices.to(torch.int64))
            )
            ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
            B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
            return cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape), B_cp

        sparse, B_cp = prepare()
        out_cp, compute_ms = _benchmark(lambda: sparse @ B_cp, warmup, iters)
        out = torch.utils.dlpack.from_dlpack(out_cp.toDlpack())
    except Exception as exc:
        return None, None, None, None, str(exc)
    symbolic_ms = 0.0
    return out, symbolic_ms, compute_ms, symbolic_ms + compute_ms, None


def _build_csr_from_row_lengths(
    row_lengths, n_cols, dtype, index_dtype, device, pattern="random"
):
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
    elif case_name == "medium_uniform":
        shape = (2048, 4096)
    elif case_name == "long_uniform":
        shape = (1024, 4096)
    elif case_name == "heavy_tail":
        shape = (4096, 8192)
    elif case_name == "powerlaw_irregular":
        shape = (8192, 8192)
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
    )
    return data, indices, indptr, shape


def _build_summary_status(profiles):
    required = (profiles["base_vs_torch_ref"]["status"],)
    if any(status != "PASS" for status in required):
        return "FAIL"
    for key in (
        "alpha_spmm_alg1_tle_opt_vs_torch_ref",
        "alpha_spmm_alg1_tle_opt2_vs_torch_ref",
    ):
        if profiles[key]["status"] not in ("PASS", "SKIP"):
            return "FAIL"
    return "PASS"


def _timing_dict(prefix, symbolic_ms, compute_ms, total_ms):
    return {
        f"{prefix}_symbolic_ms": symbolic_ms,
        f"{prefix}_compute_ms": compute_ms,
        f"{prefix}_total_ms": total_ms,
    }


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
    del index_dtype
    device = data.device
    n_rows, n_cols = shape
    B = _seeded_dense_matrix((n_cols, dense_cols), dtype, device, seed)
    base_out, base_symbolic_ms, base_compute_ms, base_total_ms, prepared_base = (
        _timed_spmm_base(data, indices, indptr, B, shape, warmup, iters)
    )
    (
        opt_out,
        opt_symbolic_ms,
        opt_compute_ms,
        opt_total_ms,
        prepared_opt,
        opt_meta,
        opt_reason,
    ) = _timed_alpha_spmm_alg1_tle_opt(data, indices, indptr, B, shape, warmup, iters)
    (
        opt2_out,
        opt2_symbolic_ms,
        opt2_compute_ms,
        opt2_total_ms,
        prepared_opt2,
        opt2_meta,
        opt2_reason,
    ) = _timed_alpha_spmm_alg1_tle_opt2(data, indices, indptr, B, shape, warmup, iters)
    torch_out, torch_symbolic_ms, torch_compute_ms, torch_total_ms, torch_reason = (
        _timed_torch_reference(data, indices, indptr, B, shape, dtype, warmup, iters)
    )
    cupy_out, cupy_symbolic_ms, cupy_compute_ms, cupy_total_ms, cupy_reason = (
        _timed_sparse_backend(
            data, indices, indptr, B, shape, warmup, iters, with_cusparse
        )
    )

    profiles = {
        "base_vs_torch_ref": _error_profile(base_out, torch_out, dtype),
        "alpha_spmm_alg1_tle_opt_vs_torch_ref": _error_profile(
            opt_out, torch_out, dtype
        ),
        "alpha_spmm_alg1_tle_opt2_vs_torch_ref": _error_profile(
            opt2_out, torch_out, dtype
        ),
        "base_vs_cupy_cusparse_ref": _error_profile(base_out, cupy_out, dtype),
        "alpha_spmm_alg1_tle_opt_vs_cupy_cusparse_ref": _error_profile(
            opt_out, cupy_out, dtype
        ),
        "alpha_spmm_alg1_tle_opt2_vs_cupy_cusparse_ref": _error_profile(
            opt2_out, cupy_out, dtype
        ),
    }
    max_row_nnz = int((indptr[1:] - indptr[:-1]).max().item()) if n_rows > 0 else 0
    summary = {
        "matrix": case_name,
        "value_dtype": str(dtype).replace("torch.", ""),
        "dense_cols": dense_cols,
        "seed": seed,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(data.numel()),
        "avg_nnz_per_row": float(data.numel()) / float(max(1, n_rows)),
        "max_row_nnz": max_row_nnz,
        **_timing_dict("base", base_symbolic_ms, base_compute_ms, base_total_ms),
        **_timing_dict(
            "alpha_spmm_alg1_tle_opt", opt_symbolic_ms, opt_compute_ms, opt_total_ms
        ),
        **_timing_dict(
            "alpha_spmm_alg1_tle_opt2", opt2_symbolic_ms, opt2_compute_ms, opt2_total_ms
        ),
        **_timing_dict(
            "torch_ref", torch_symbolic_ms, torch_compute_ms, torch_total_ms
        ),
        **_timing_dict(
            "cupy_cusparse_ref", cupy_symbolic_ms, cupy_compute_ms, cupy_total_ms
        ),
        "torch_ref_compute_speedup_vs_base": _ratio(torch_compute_ms, base_compute_ms),
        "torch_ref_compute_speedup_vs_alpha_spmm_alg1_tle_opt": _ratio(
            torch_compute_ms, opt_compute_ms
        ),
        "torch_ref_compute_speedup_vs_alpha_spmm_alg1_tle_opt2": _ratio(
            torch_compute_ms, opt2_compute_ms
        ),
        "cupy_cusparse_ref_compute_speedup_vs_base": _ratio(
            cupy_compute_ms, base_compute_ms
        ),
        "cupy_cusparse_ref_compute_speedup_vs_alpha_spmm_alg1_tle_opt": _ratio(
            cupy_compute_ms, opt_compute_ms
        ),
        "cupy_cusparse_ref_compute_speedup_vs_alpha_spmm_alg1_tle_opt2": _ratio(
            cupy_compute_ms, opt2_compute_ms
        ),
        "alpha_spmm_alg1_tle_opt2_compute_speedup_vs_alpha_spmm_alg1_tle_opt": _ratio(
            opt_compute_ms, opt2_compute_ms
        ),
        "torch_ref_total_speedup_vs_base": _ratio(torch_total_ms, base_total_ms),
        "torch_ref_total_speedup_vs_alpha_spmm_alg1_tle_opt": _ratio(
            torch_total_ms, opt_total_ms
        ),
        "torch_ref_total_speedup_vs_alpha_spmm_alg1_tle_opt2": _ratio(
            torch_total_ms, opt2_total_ms
        ),
        "cupy_cusparse_ref_total_speedup_vs_base": _ratio(cupy_total_ms, base_total_ms),
        "cupy_cusparse_ref_total_speedup_vs_alpha_spmm_alg1_tle_opt": _ratio(
            cupy_total_ms, opt_total_ms
        ),
        "cupy_cusparse_ref_total_speedup_vs_alpha_spmm_alg1_tle_opt2": _ratio(
            cupy_total_ms, opt2_total_ms
        ),
        "alpha_spmm_alg1_tle_opt2_total_speedup_vs_alpha_spmm_alg1_tle_opt": _ratio(
            opt_total_ms, opt2_total_ms
        ),
        "base_vs_torch_ref_err": profiles["base_vs_torch_ref"]["global_err"],
        "alpha_spmm_alg1_tle_opt_vs_torch_ref_err": profiles[
            "alpha_spmm_alg1_tle_opt_vs_torch_ref"
        ]["global_err"],
        "alpha_spmm_alg1_tle_opt2_vs_torch_ref_err": profiles[
            "alpha_spmm_alg1_tle_opt2_vs_torch_ref"
        ]["global_err"],
        "base_vs_cupy_cusparse_ref_err": profiles["base_vs_cupy_cusparse_ref"][
            "global_err"
        ],
        "alpha_spmm_alg1_tle_opt_vs_cupy_cusparse_ref_err": profiles[
            "alpha_spmm_alg1_tle_opt_vs_cupy_cusparse_ref"
        ]["global_err"],
        "alpha_spmm_alg1_tle_opt2_vs_cupy_cusparse_ref_err": profiles[
            "alpha_spmm_alg1_tle_opt2_vs_cupy_cusparse_ref"
        ]["global_err"],
        "base_status_vs_torch_ref": profiles["base_vs_torch_ref"]["status"],
        "alpha_spmm_alg1_tle_opt_status_vs_torch_ref": profiles[
            "alpha_spmm_alg1_tle_opt_vs_torch_ref"
        ]["status"],
        "alpha_spmm_alg1_tle_opt2_status_vs_torch_ref": profiles[
            "alpha_spmm_alg1_tle_opt2_vs_torch_ref"
        ]["status"],
        "base_status_vs_cupy_cusparse_ref": profiles["base_vs_cupy_cusparse_ref"][
            "status"
        ],
        "alpha_spmm_alg1_tle_opt_status_vs_cupy_cusparse_ref": profiles[
            "alpha_spmm_alg1_tle_opt_vs_cupy_cusparse_ref"
        ]["status"],
        "alpha_spmm_alg1_tle_opt2_status_vs_cupy_cusparse_ref": profiles[
            "alpha_spmm_alg1_tle_opt2_vs_cupy_cusparse_ref"
        ]["status"],
        "alpha_spmm_alg1_tle_opt_status": "SKIP" if opt_out is None else "PASS",
        "alpha_spmm_alg1_tle_opt_reason": opt_reason,
        "alpha_spmm_alg1_tle_opt2_status": "SKIP" if opt2_out is None else "PASS",
        "alpha_spmm_alg1_tle_opt2_reason": opt2_reason,
        "torch_ref_status": "SKIP" if torch_out is None else "PASS",
        "torch_ref_reason": torch_reason,
        "cupy_cusparse_ref_status": "SKIP" if cupy_out is None else "PASS",
        "cupy_cusparse_ref_reason": cupy_reason,
        "matrix_status": _build_summary_status(profiles),
    }
    if not return_details:
        return summary
    return {
        "summary": summary,
        "prepared_base": prepared_base,
        "prepared_alpha_tle_opt": prepared_opt,
        "prepared_alpha_tle_opt2": prepared_opt2,
        "meta_alpha_tle_opt": opt_meta,
        "meta_alpha_tle_opt2": opt2_meta,
        "B": B,
        "profiles": profiles,
    }


def _print_header():
    opt_available = fs.is_alpha_spmm_alg1_tle_opt_available()
    opt_status = (
        "available"
        if opt_available
        else f"unavailable ({fs.alpha_spmm_alg1_tle_opt_unavailable_reason()})"
    )
    opt2_available = fs.is_alpha_spmm_alg1_tle_opt2_available()
    opt2_status = (
        "available"
        if opt2_available
        else f"unavailable ({fs.alpha_spmm_alg1_tle_opt2_unavailable_reason()})"
    )
    print(f"TLEOpt alpha_spmm_alg1_tle_opt: {opt_status}")
    print(f"TLEOpt2 alpha_spmm_alg1_tle_opt2: {opt2_status}")
    print("-" * 174)
    print(
        f"{'Matrix':<24} {'dtype':>7} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} "
        f"{'Base(ms)':>9} {'TLEOpt(ms)':>11} {'TLEOpt2(ms)':>12} {'TorchRef(ms)':>12} "
        f"{'CuPyRef(ms)':>11} {'Opt2/Opt':>9} {'Err(Opt)':>10} {'Err(Opt2)':>10} {'Status':>8}"
    )
    print("-" * 174)


def _fmt_ms(value):
    return "N/A" if value is None else f"{value:.4f}"


def _fmt_ratio(value):
    return "N/A" if value is None else f"{value:.2f}x"


def _fmt_err(value):
    return "N/A" if value is None else f"{value:.2e}"


def _print_summary_row(summary):
    print(
        f"{summary['matrix'][:23]:<24} {summary['value_dtype']:>7} {summary['n_rows']:>7} "
        f"{summary['n_cols']:>7} {summary['nnz']:>10} {summary['dense_cols']:>8} "
        f"{_fmt_ms(summary['base_compute_ms']):>9} "
        f"{_fmt_ms(summary['alpha_spmm_alg1_tle_opt_compute_ms']):>11} "
        f"{_fmt_ms(summary['alpha_spmm_alg1_tle_opt2_compute_ms']):>12} "
        f"{_fmt_ms(summary['torch_ref_compute_ms']):>12} "
        f"{_fmt_ms(summary['cupy_cusparse_ref_compute_ms']):>11} "
        f"{_fmt_ratio(summary['alpha_spmm_alg1_tle_opt2_compute_speedup_vs_alpha_spmm_alg1_tle_opt']):>9} "
        f"{_fmt_err(summary['alpha_spmm_alg1_tle_opt_vs_torch_ref_err']):>10} "
        f"{_fmt_err(summary['alpha_spmm_alg1_tle_opt2_vs_torch_ref_err']):>10} "
        f"{summary['matrix_status']:>8}"
    )
    for key in (
        "alpha_spmm_alg1_tle_opt",
        "alpha_spmm_alg1_tle_opt2",
        "torch_ref",
        "cupy_cusparse_ref",
    ):
        if summary.get(f"{key}_status") == "SKIP" and summary.get(f"{key}_reason"):
            print(f"  {key} skipped: {summary[f'{key}_reason']}")


def _write_csv(path, rows, fieldnames):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: ("" if value is None else value) for key, value in row.items()}
            )


def _build_launch_row(matrix_name, route, dtype_name, dense_cols, prepared, meta):
    if prepared is None or meta is None:
        return {
            "matrix": matrix_name,
            "route": route,
            "dtype": dtype_name,
            "dense_cols": dense_cols,
        }
    return {
        "matrix": matrix_name,
        "route": route,
        "device_name": meta["device_name"],
        "sm_count": meta["sm_count"],
        "dtype": dtype_name,
        "dense_cols": dense_cols,
        "warp_size": meta["warp_size"],
        "factor": meta["factor"],
        "block_size": meta["block_size"],
        "block_rows": meta["block_rows"],
        "block_cols": meta["block_cols"],
        "num_warps": meta["num_warps"],
        "num_stages": meta["num_stages"],
        "loop_strategy": meta.get("loop_strategy", ""),
        "launch_version": meta.get("launch_version", ""),
        "grid_m": meta["grid_m"],
        "grid_n": meta["grid_n"],
    }


def _append_launch_rows(launch_rows, result):
    summary = result["summary"]
    dtype_name = summary["value_dtype"]
    dense_cols = summary["dense_cols"]
    launch_rows.append(
        _build_launch_row(
            summary["matrix"],
            "alpha_spmm_alg1_tle_opt",
            dtype_name,
            dense_cols,
            result["prepared_alpha_tle_opt"],
            result["meta_alpha_tle_opt"],
        )
    )
    launch_rows.append(
        _build_launch_row(
            summary["matrix"],
            "alpha_spmm_alg1_tle_opt2",
            dtype_name,
            dense_cols,
            result["prepared_alpha_tle_opt2"],
            result["meta_alpha_tle_opt2"],
        )
    )


def _check_required(args, summary):
    if args.require_tle_opt and summary["alpha_spmm_alg1_tle_opt_status"] == "SKIP":
        raise RuntimeError(
            "alpha_spmm_alg1_tle_opt was skipped: "
            + str(summary.get("alpha_spmm_alg1_tle_opt_reason") or "")
        )
    if args.require_tle_opt2 and summary["alpha_spmm_alg1_tle_opt2_status"] == "SKIP":
        raise RuntimeError(
            "alpha_spmm_alg1_tle_opt2 was skipped: "
            + str(summary.get("alpha_spmm_alg1_tle_opt2_reason") or "")
        )


def main():
    parser = argparse.ArgumentParser(
        description="Experimental AlphaSparse ALG1 TLEOpt SpMM benchmark."
    )
    parser.add_argument("input_path", nargs="*", help=".mtx file or directory")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument(
        "--csv", type=str, default=None, help="Write summary CSV and <stem>_launch.csv"
    )
    parser.add_argument("--dense-cols", type=int, default=DEFAULT_DENSE_COLS)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--with-cusparse",
        action="store_true",
        default=True,
        help="Run CuPy/cuSPARSE reference timing (default).",
    )
    parser.add_argument(
        "--no-cusparse",
        action="store_false",
        dest="with_cusparse",
        help="Disable CuPy/cuSPARSE reference timing.",
    )
    parser.add_argument("--require-tle-opt", action="store_true")
    parser.add_argument("--require-tle-opt2", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    rows = []
    launch_rows = []
    if args.require_tle_opt and not fs.is_alpha_spmm_alg1_tle_opt_available():
        raise RuntimeError(
            "alpha_spmm_alg1_tle_opt is unavailable: "
            + fs.alpha_spmm_alg1_tle_opt_unavailable_reason()
        )
    if args.require_tle_opt2 and not fs.is_alpha_spmm_alg1_tle_opt2_available():
        raise RuntimeError(
            "alpha_spmm_alg1_tle_opt2 is unavailable: "
            + fs.alpha_spmm_alg1_tle_opt2_unavailable_reason()
        )
    _print_header()
    if args.synthetic:
        synthetic_cases = [
            "short_uniform",
            "medium_uniform",
            "long_uniform",
            "heavy_tail",
            "powerlaw_irregular",
        ]
        for value_dtype in VALUE_DTYPES:
            for dense_cols in (4, 5, 12, 24, 48, 96):
                for case_name in synthetic_cases:
                    data, indices, indptr, shape = build_synthetic_case(
                        case_name, value_dtype, torch.int32, device
                    )
                    result = run_one_case(
                        f"{case_name}_n{dense_cols}",
                        data,
                        indices,
                        indptr,
                        shape,
                        value_dtype,
                        torch.int32,
                        dense_cols,
                        args.warmup,
                        args.iters,
                        args.seed,
                        args.with_cusparse,
                        return_details=bool(args.csv),
                    )
                    summary = result["summary"] if args.csv else result
                    if args.csv:
                        _append_launch_rows(launch_rows, result)
                    rows.append(summary)
                    _print_summary_row(summary)
                    _check_required(args, summary)
    else:
        paths = _resolve_input_paths(args.input_path)
        for value_dtype in VALUE_DTYPES:
            for index_dtype in INDEX_DTYPES:
                for path in paths:
                    data, indices, indptr, shape = load_mtx_to_csr_torch(
                        path, dtype=value_dtype, device=device
                    )
                    indices = indices.to(index_dtype)
                    result = run_one_case(
                        os.path.basename(path),
                        data,
                        indices,
                        indptr,
                        shape,
                        value_dtype,
                        index_dtype,
                        args.dense_cols,
                        args.warmup,
                        args.iters,
                        args.seed,
                        args.with_cusparse,
                        return_details=bool(args.csv),
                    )
                    summary = result["summary"] if args.csv else result
                    if args.csv:
                        _append_launch_rows(launch_rows, result)
                    rows.append(summary)
                    _print_summary_row(summary)
                    _check_required(args, summary)
    print("-" * 174)
    if args.csv:
        csv_path = os.path.abspath(args.csv)
        stem, ext = os.path.splitext(csv_path)
        launch_csv_path = f"{stem}_launch{ext or '.csv'}"
        _write_csv(csv_path, rows, SUMMARY_FIELDS)
        _write_csv(launch_csv_path, launch_rows, LAUNCH_FIELDS)
        print(f"Wrote {len(rows)} rows to {csv_path}")
        print(f"Wrote {len(launch_rows)} launch rows to {launch_csv_path}")


if __name__ == "__main__":
    main()
