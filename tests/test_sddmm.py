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
SDDMM tests: load SuiteSparse .mtx as CSR pattern and benchmark
out = alpha * dot(X[row], Y[col]) + beta * in.
cuSPARSE baseline is cusparseSDDMM via torch.sparse.sampled_addmm.

acc_mode notes:
- acc_mode=f32 keeps the native float32 accumulate path for float32 inputs.
- acc_mode=f64 upgrades only the internal accumulation of float32 inputs to
  float64 while still returning float32 outputs.
- float64 inputs always keep the existing float64 route; acc_mode only affects
  float32 runs in this test harness.
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
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

import flagsparse as ast
import flagsparse.sparse_operations.sddmm_csr as ast_ops
from test_spmm import load_mtx_to_csr_torch

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
WARMUP = 5
ITERS = 20
DEFAULT_K = 64
CSV_K_DIMS = (32, 64, 128, 256)
BASELINE_ATOL = 1e-4
BASELINE_RTOL = 1e-2
ACC64_ATOL = 1e-6
ACC64_RTOL = 1e-5
DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}
INDEX_DTYPE_MAP = {
    "int32": torch.int32,
}


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


def _fmt_err(value):
    return "N/A" if value is None else f"{value:.2e}"


def _fmt_check(value):
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def _status_label(value):
    if isinstance(value, str):
        return value
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def _is_resource_error(message):
    text = str(message).lower()
    resource_tokens = (
        "out of memory",
        "cudaerroroutofmemory",
        "cuda error out of memory",
        "insufficient resources",
        "resource exhausted",
        "memoryerror",
        "cublas_status_alloc_failed",
        "cusparse_status_insufficient_resources",
    )
    return any(token in text for token in resource_tokens)


def _benchmark_cusparse_sddmm(
    indices, indptr, shape, x, y, data_in, alpha, beta, warmup, iters
):
    """Real cuSPARSE SDDMM (``cusparseSDDMM``) under the scenario-B contract.

    Scenario B means single-shot with validation on both sides, so the timed window
    holds everything a caller must pay for one SDDMM: descriptor construction with
    ``check_invariants=True`` (PyTorch's equivalent of FlagSparse's ``validate=True``,
    checking the same CSR invariants) plus the op itself. cuSPARSE redoes its
    descriptor/bufferSize/preprocess work on every call and PyTorch exposes no way to
    reuse it, so nothing here can be hoisted out even in principle.

    This replaced a CuPy ``sum(x[rows] * y[cols], axis=1)`` gather that was reported in
    the ``cusparse_ms`` column while not being cuSPARSE at all -- it materialises
    nnz x k temporaries and was 3-15x slower than the real thing, inflating the
    reported speedup to 12-38x.
    """
    indptr64 = indptr.to(torch.int64).contiguous()
    indices64 = indices.to(torch.int64).contiguous()
    # sampled_addmm computes beta*input + alpha*(x @ y.T) masked to input's pattern,
    # which matches out = alpha*dot(x[row], y[col]) + beta*data when the descriptor
    # carries data_in as its values.
    vals = (
        data_in
        if (beta != 0.0 and data_in is not None)
        else torch.zeros(int(indices.numel()), dtype=x.dtype, device=x.device)
    )
    y_t = y.t()

    def op():
        sp = torch.sparse_csr_tensor(
            indptr64, indices64, vals, size=shape, check_invariants=True
        )
        return torch.sparse.sampled_addmm(
            sp, x, y_t, beta=float(beta), alpha=float(alpha)
        ).values()

    return ast_ops._benchmark_cuda_op(op, warmup=warmup, iters=iters)


def _normalize_csv_path(csv_path):
    csv_path = str(csv_path)
    if not csv_path.lower().endswith(".csv"):
        csv_path = f"{csv_path}.csv"
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return csv_path


def _parse_mapped_tokens(value, mapping, default_names, option_name):
    raw = ",".join(default_names) if value is None else str(value)
    tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{option_name} must not be empty")
    invalid = [token for token in tokens if token not in mapping]
    if invalid:
        raise ValueError(
            f"unsupported {option_name}: {', '.join(invalid)}; allowed: {', '.join(mapping)}"
        )
    return [mapping[token] for token in tokens]


def _parse_k_dims(value, default_dims):
    raw = ",".join(str(k) for k in default_dims) if value is None else str(value)
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--k must not be empty")
    k_dims = []
    for token in tokens:
        try:
            k_dim = int(token)
        except ValueError as exc:
            raise ValueError(f"--k values must be integers, got {token!r}") from exc
        if k_dim < 0:
            raise ValueError("--k must be non-negative")
        k_dims.append(k_dim)
    return k_dims


def _resolve_tolerance(value_dtype, acc_mode):
    if value_dtype == torch.float32:
        if acc_mode == "f64":
            return ACC64_ATOL, ACC64_RTOL
        return BASELINE_ATOL, BASELINE_RTOL
    return ast_ops._tolerance_for_dtype(value_dtype)


def _scaled_allclose_error(candidate, reference, atol, rtol):
    if candidate.numel() == 0:
        return 0.0
    diff = torch.abs(candidate - reference)
    denom = atol + rtol * torch.abs(reference)
    return float(torch.max(diff / denom).item())


def _benchmark_reference_sddmm(
    data, indices, indptr, x, y, alpha, beta, value_dtype, warmup, iters
):
    indptr64 = indptr.to(torch.int64)
    if value_dtype == torch.float32:
        x_ref = x.to(torch.float64)
        y_ref = y.to(torch.float64)
        data_ref = data.to(torch.float64) if data is not None else None

        op = lambda: ast_ops._sddmm_reference(
            indices, indptr64, x_ref, y_ref, data_ref, alpha, beta
        ).to(torch.float32)
    else:
        op = lambda: ast_ops._sddmm_reference(
            indices, indptr64, x, y, data, alpha, beta
        )
    ref_values, ref_ms = ast_ops._benchmark_cuda_op(op, warmup=warmup, iters=iters)
    return ref_values, ref_ms


def _benchmark_triton_sddmm(
    data, indices, indptr, shape, x, y, alpha, beta, warmup, iters, acc_mode
):
    torch.cuda.synchronize()
    t_prepare0 = time.perf_counter()
    prepared = ast.prepare_sddmm_csr(indices, indptr, shape, k_hint=int(x.shape[1]))
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0

    torch.cuda.synchronize()
    t_first0 = time.perf_counter()
    if x.dtype == torch.float32 and acc_mode == "f64":
        _ = ast_ops._run_sddmm_prepared(
            prepared,
            x.contiguous(),
            y.contiguous(),
            data.contiguous() if data is not None else None,
            alpha,
            beta,
            out=None,
            allow_fallback=False,
            variant="acc64",
        )[0]
    else:
        _ = ast.flagsparse_sddmm_csr(
            data=data, x=x, y=y, alpha=alpha, beta=beta, prepared=prepared
        )
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t_first0) * 1000.0

    if x.dtype == torch.float32 and acc_mode == "f64":
        # Scenario B: prepare inside the timed window here too, so the acc64 diagnostic
        # is measured on the same contract as the default path.
        op = lambda: ast_ops._run_sddmm_prepared(
            ast.prepare_sddmm_csr(indices, indptr, shape, k_hint=int(x.shape[1])),
            x.contiguous(),
            y.contiguous(),
            data.contiguous() if data is not None else None,
            alpha,
            beta,
            out=None,
            allow_fallback=False,
            variant="acc64",
        )[0]
        triton_values, triton_ms = ast_ops._benchmark_cuda_op(
            op, warmup=warmup, iters=iters
        )
        _, meta = ast_ops._run_sddmm_prepared(
            prepared,
            x.contiguous(),
            y.contiguous(),
            data.contiguous() if data is not None else None,
            alpha,
            beta,
            out=None,
            allow_fallback=False,
            variant="acc64",
        )
    else:
        # Scenario B: prepare is inside the timed window (raw CSR args, not prepared=),
        # with validate=True, so this measures the whole cost of one SDDMM against a
        # cuSPARSE call that likewise cannot amortise its own setup. prepare_ms is
        # still reported separately below. Matches the SpGEMM decision.
        triton_values, triton_ms = ast_ops._benchmark_cuda_op(
            lambda: ast.flagsparse_sddmm_csr(
                data=data,
                indices=indices,
                indptr=indptr,
                shape=shape,
                x=x,
                y=y,
                alpha=alpha,
                beta=beta,
            ),
            warmup=warmup,
            iters=iters,
        )
        _, meta = ast.flagsparse_sddmm_csr(
            data=data,
            x=x,
            y=y,
            alpha=alpha,
            beta=beta,
            prepared=prepared,
            return_meta=True,
        )
    meta["prepare_ms"] = prepare_ms
    return triton_values, triton_ms, first_call_ms, meta


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    k_dim=DEFAULT_K,
    alpha=1.0,
    beta=0.0,
    run_cusparse=True,
    acc_mode="f32",
):
    device = torch.device("cuda")
    _pattern_values, indices, indptr, shape = load_mtx_to_csr_torch(
        mtx_path, dtype=value_dtype, device=device
    )
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz = int(indices.numel())
    data_in = torch.randn(nnz, dtype=value_dtype, device=device)
    x = torch.randn((n_rows, k_dim), dtype=value_dtype, device=device)
    y = torch.randn((n_cols, k_dim), dtype=value_dtype, device=device)

    result = {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz,
        "nnz_pattern": nnz,
        "k": int(k_dim),
        "alpha": float(alpha),
        "beta": float(beta),
        "error": None,
        "triton_ms": None,
        "triton_first_call_ms": None,
        "prepare_ms": None,
        "pytorch_ms": None,
        "cusparse_ms": None,
        "err_pt": None,
        "err_cu": None,
        "triton_ok_pt": None,
        "triton_ok_cu": None,
        "cu_status": "REF_UNAVAILABLE",
        "cu_reason": None,
        "cusparse_reason": None,
        "triton_started": False,
        "cu_started": False,
        "fallback_used": False,
        "status": "UNKNOWN",
    }

    triton_values = None
    try:
        result["triton_started"] = True
        triton_values, triton_ms, triton_first_ms, meta = _benchmark_triton_sddmm(
            data_in, indices, indptr, shape, x, y, alpha, beta, warmup, iters, acc_mode
        )
        result["triton_ms"] = triton_ms
        result["triton_first_call_ms"] = triton_first_ms
        result["prepare_ms"] = meta.get("prepare_ms")
        result["fallback_used"] = bool(meta.get("fallback_used", False))
    except Exception as exc:
        result["error"] = f"triton: {exc}"

    try:
        ref, result["pytorch_ms"] = _benchmark_reference_sddmm(
            data_in,
            indices,
            indptr,
            x,
            y,
            alpha,
            beta,
            value_dtype,
            warmup,
            iters,
        )
    except Exception as exc:
        result["error"] = (
            str(exc) if result["error"] is None else f"{result['error']}; ref: {exc}"
        )
        result["status"] = "REF_FAIL"
        return result

    if triton_values is not None:
        atol, rtol = _resolve_tolerance(value_dtype, acc_mode)
        result["triton_ok_pt"] = bool(
            torch.allclose(triton_values, ref, atol=atol, rtol=rtol)
        )
        result["err_pt"] = _scaled_allclose_error(triton_values, ref, atol, rtol)
    else:
        result["triton_ok_pt"] = False

    if run_cusparse:
        try:
            result["cu_started"] = True
            cu_vals, cusparse_ms = _benchmark_cusparse_sddmm(
                indices=indices,
                indptr=indptr,
                shape=shape,
                x=x,
                y=y,
                data_in=data_in,
                alpha=alpha,
                beta=beta,
                warmup=warmup,
                iters=iters,
            )
            result["cusparse_ms"] = cusparse_ms
            # cuSPARSE is now a real implementation of the same op, so it is a
            # correctness check as well as the performance baseline.
            if triton_values is not None and cu_vals is not None:
                atol, rtol = _resolve_tolerance(value_dtype, acc_mode)
                result["triton_ok_cu"] = bool(
                    torch.allclose(triton_values, cu_vals, atol=atol, rtol=rtol)
                )
                result["err_cu"] = _scaled_allclose_error(
                    triton_values, cu_vals, atol, rtol
                )
                result["cu_status"] = "PASS" if result["triton_ok_cu"] else "FAIL"
            else:
                result["cu_status"] = "PERF_ONLY"
        except Exception as exc:
            result["cu_status"] = (
                "PERF_RESOURCE" if _is_resource_error(exc) else "PERF_UNAVAILABLE"
            )
            result["cu_reason"] = str(exc)
    else:
        result["cu_status"] = "PERF_ONLY"
        result["cu_reason"] = "cuSPARSE baseline is disabled by CLI"

    result["cusparse_reason"] = result["cu_reason"]
    result["status"] = "PASS" if result["triton_ok_pt"] else "FAIL"
    return result


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    k_dim=DEFAULT_K,
    alpha=1.0,
    beta=0.0,
    run_cusparse=True,
    on_result=None,
    acc_mode="f32",
):
    results = []
    for path in mtx_paths:
        entry = run_one_mtx(
            path,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=warmup,
            iters=iters,
            k_dim=k_dim,
            alpha=alpha,
            beta=beta,
            run_cusparse=run_cusparse,
            acc_mode=acc_mode,
        )
        results.append(entry)
        if on_result is not None:
            on_result(entry)
    return results


def _print_sddmm_mtx_header(value_dtype, index_dtype, k_dim, alpha, beta, acc_mode):
    print(
        f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}"
    )
    print(
        "Formats: FlagSparse=CSR SDDMM vs cuSPARSE SDDMM (cusparseSDDMM via "
        "torch.sparse.sampled_addmm). PyTorch = correctness reference only, not a "
        "performance baseline."
    )
    print(
        "Metric: scenario B -- single-shot, validation on both sides. Both timings "
        "include their own setup (FlagSparse prepare+validate; cuSPARSE "
        "descriptor+check_invariants+preprocess), neither is amortised."
    )
    print(
        f"Equation: out = alpha*dot(x[row], y[col]) + beta*in  |  K={k_dim}  alpha={alpha}  beta={beta}  acc_mode={acc_mode}"
    )
    print("-" * 196)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'K':>6} "
        f"{'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>13} {'PyTorch(ms)':>11} "
        f"{'FS/CU':>7} {'PT':>6} {'CU_Status':>12} {'Err(PT)':>10} {'Err(CU)':>10} {'Prep(ms)':>9}"
    )
    print("-" * 196)


def _print_sddmm_mtx_row(entry):
    name = os.path.basename(entry["path"])[:27]
    n_rows, n_cols = entry["shape"]
    cusparse_ms = entry.get("cusparse_ms")
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {entry['nnz_pattern']:>10} {entry['k']:>6} "
        f"{_fmt_ms(entry.get('triton_ms')):>14} {_fmt_ms(cusparse_ms):>13} {_fmt_ms(entry.get('pytorch_ms')):>11} "
        f"{_fmt_speedup(cusparse_ms, entry.get('triton_ms')):>7} "
        f"{_fmt_check(entry.get('triton_ok_pt')):>6} {_status_label(entry.get('cu_status')):>12} "
        f"{_fmt_err(entry.get('err_pt')):>10} {_fmt_err(entry.get('err_cu')):>10} {_fmt_ms(entry.get('prepare_ms')):>9}"
    )
    err = entry.get("error")
    cu_reason = entry.get("cu_reason")
    if err:
        msg = str(err).replace("\n", " ")
        if len(msg) > 220:
            msg = msg[:217] + "..."
        print(f"  NOTE: {msg}")
    if cu_reason:
        msg = str(cu_reason).replace("\n", " ")
        if len(msg) > 220:
            msg = msg[:217] + "..."
        print(f"  CU_NOTE: {msg}")


def print_mtx_results(results, value_dtype, index_dtype, k_dim, alpha, beta, acc_mode):
    _print_sddmm_mtx_header(value_dtype, index_dtype, k_dim, alpha, beta, acc_mode)
    for entry in results:
        _print_sddmm_mtx_row(entry)
    print("-" * 196)


def run_all_dtypes_export_csv(
    paths,
    csv_path,
    value_dtypes=None,
    index_dtypes=None,
    warmup=WARMUP,
    iters=ITERS,
    k_dims=None,
    alpha=1.0,
    beta=0.0,
    run_cusparse=True,
    acc_mode="f32",
):
    csv_path = _normalize_csv_path(csv_path)
    rows = []
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    k_dims = CSV_K_DIMS if k_dims is None else k_dims
    for value_dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for k_dim in k_dims:
                print("=" * 164)
                _print_sddmm_mtx_header(
                    value_dtype, index_dtype, k_dim, alpha, beta, acc_mode
                )
                results = run_mtx_batch(
                    paths,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=warmup,
                    iters=iters,
                    k_dim=k_dim,
                    alpha=alpha,
                    beta=beta,
                    run_cusparse=run_cusparse,
                    on_result=_print_sddmm_mtx_row,
                    acc_mode=acc_mode,
                )
                print("-" * 196)
                for entry in results:
                    n_rows, n_cols = entry["shape"]
                    cusparse_ms = entry.get("cusparse_ms")
                    pytorch_ms = entry.get("pytorch_ms")
                    triton_ms = entry.get("triton_ms")
                    rows.append(
                        {
                            "matrix": os.path.basename(entry["path"]),
                            "value_dtype": _dtype_name(value_dtype),
                            "index_dtype": _dtype_name(index_dtype),
                            "n_rows": n_rows,
                            "n_cols": n_cols,
                            "nnz": entry["nnz"],
                            "triton_ms": triton_ms,
                            "cusparse_ms": cusparse_ms,
                            "pytorch_ms": pytorch_ms,
                            # Scenario B is the single performance metric. There is
                            # deliberately no speedup-vs-pytorch column: the PyTorch
                            # path is a correctness reference that materialises
                            # nnz x k temporaries, so its latency is not a baseline.
                            # run_flagsparse_pytest.py resolves the reported metric by
                            # first non-empty match in PERFORMANCE_SPEEDUP_SCHEMAS,
                            # where vs_pytorch precedes vs_cusparse -- emitting it
                            # would silently override this metric.
                            "triton_speedup_vs_cusparse": _speedup_ratio(
                                cusparse_ms, triton_ms
                            ),
                            "pt_status": _status_label(entry.get("triton_ok_pt")),
                            "cu_status": _status_label(entry.get("cu_status")),
                            "status": entry.get("status"),
                            "err_pt": entry.get("err_pt"),
                            "err_cu": entry.get("err_cu"),
                            "error": entry.get("error"),
                            "cu_reason": entry.get("cu_reason"),
                            "triton_started": entry.get("triton_started"),
                            "cu_started": entry.get("cu_started"),
                            "fallback_used": entry.get("fallback_used"),
                            "nnz_pattern": entry.get("nnz_pattern"),
                            "k": entry.get("k"),
                            "alpha": entry.get("alpha"),
                            "beta": entry.get("beta"),
                            "prepare_ms": entry.get("prepare_ms"),
                        }
                    )
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "n_rows",
        "n_cols",
        "nnz",
        "triton_ms",
        "cusparse_ms",
        "pytorch_ms",
        "triton_speedup_vs_cusparse",
        "pt_status",
        "cu_status",
        "status",
        "err_pt",
        "err_cu",
        "error",
        "cu_reason",
        "triton_started",
        "cu_started",
        "fallback_used",
        "nnz_pattern",
        "k",
        "alpha",
        "beta",
        "prepare_ms",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: ("" if value is None else value) for key, value in row.items()}
            )
    print(f"Wrote {len(rows)} rows to {csv_path}")


def run_api_validation_checks():
    if not torch.cuda.is_available():
        print("API checks skipped: CUDA is not available.")
        return 0
    device = torch.device("cuda")
    indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    shape = (2, 2)
    x = torch.randn((2, 8), dtype=torch.float32, device=device)
    y = torch.randn((2, 8), dtype=torch.float32, device=device)
    data = torch.randn(3, dtype=torch.float32, device=device)

    negative_cases = [
        (
            "indices must int32",
            lambda: ast.flagsparse_sddmm_csr(
                indices=indices.to(torch.int64), indptr=indptr, x=x, y=y, shape=shape
            ),
            TypeError,
        ),
        (
            "x/y K mismatch",
            lambda: ast.flagsparse_sddmm_csr(
                indices=indices, indptr=indptr, x=x, y=y[:, :4], shape=shape
            ),
            ValueError,
        ),
        (
            "data length mismatch",
            lambda: ast.flagsparse_sddmm_csr(
                data=torch.randn(2, dtype=torch.float32, device=device),
                indices=indices,
                indptr=indptr,
                x=x,
                y=y,
                shape=shape,
            ),
            ValueError,
        ),
        (
            "beta needs data",
            lambda: ast.flagsparse_sddmm_csr(
                indices=indices, indptr=indptr, x=x, y=y, shape=shape, beta=0.5
            ),
            ValueError,
        ),
        (
            "K=0 out shape mismatch",
            lambda: ast.flagsparse_sddmm_csr(
                data=data,
                indices=indices,
                indptr=indptr,
                x=x[:, :0],
                y=y[:, :0],
                shape=shape,
                out=torch.empty(2, dtype=torch.float32, device=device),
            ),
            ValueError,
        ),
    ]
    failed = 0
    print("-" * 96)
    print("API validation checks (SDDMM)")
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

    try:
        out = ast.flagsparse_sddmm_csr(
            data=data,
            indices=indices,
            indptr=indptr,
            x=x,
            y=y,
            shape=shape,
            alpha=1.25,
            beta=0.5,
        )
        if out.shape != (3,):
            raise AssertionError("unexpected output shape")
        print("PASS  positive path returned correct output shape")
    except Exception as exc:
        print(f"FAIL  positive path raised {type(exc).__name__}: {exc}")
        failed += 1
    print("-" * 96)
    return failed


def _expand_mtx_paths(raw_paths):
    paths = []
    for p in raw_paths:
        if os.path.isfile(p) and p.lower().endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    seen = set()
    uniq = []
    for path in paths:
        ap = os.path.abspath(path)
        if ap not in seen:
            uniq.append(ap)
            seen.add(ap)
    return uniq


def main():
    parser = argparse.ArgumentParser(description="FlagSparse SDDMM CSR tests")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument(
        "--dtype",
        type=str,
        default=None,
        help="Value dtype(s): float32,float64. CSV default: float32,float64; non-CSV default: float32.",
    )
    parser.add_argument(
        "--index-dtype",
        type=str,
        default="int32",
        help="Index dtype(s), currently only int32 is supported by SDDMM CSR.",
    )
    parser.add_argument(
        "--acc_mode",
        type=str,
        default="f32",
        choices=["f32", "f64"],
        help="For float32 runs, choose native f32 accumulation or float64 accumulation.",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument(
        "--k",
        type=str,
        default=None,
        help="Dense feature dimension K; accepts one value or comma-separated values, e.g. 64 or 32,64,128. CSV default: 32,64,128,256; non-CSV default: 64.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--no-cusparse-ref",
        action="store_true",
        help="Skip the cuSPARSE SDDMM performance baseline",
    )
    parser.add_argument(
        "--no-cusparse",
        action="store_true",
        help="Alias of --no-cusparse-ref",
    )
    parser.add_argument("--csv", type=str, default=None, metavar="FILE")
    parser.add_argument("--skip-api-checks", action="store_true")
    args = parser.parse_args()

    try:
        value_dtypes = _parse_mapped_tokens(
            args.dtype,
            DTYPE_MAP,
            ("float32", "float64") if args.csv is not None else ("float32",),
            "--dtype",
        )
        index_dtypes = _parse_mapped_tokens(
            args.index_dtype,
            INDEX_DTYPE_MAP,
            ("int32",),
            "--index-dtype",
        )
        k_dims = _parse_k_dims(
            args.k,
            CSV_K_DIMS if args.csv is not None else (DEFAULT_K,),
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    if not args.skip_api_checks:
        failed = run_api_validation_checks()
        if failed > 0:
            raise SystemExit(1)
    run_cusparse_ref = not (args.no_cusparse_ref or args.no_cusparse)

    paths = _expand_mtx_paths(args.mtx)
    if not paths and not args.csv:
        print(
            "No .mtx files given. Use: python test_sddmm.py <file.mtx> [file2.mtx ...] or <dir/>"
        )
        print(
            "Or export the current dtype to CSV: python test_sddmm.py <dir/> --csv results.csv"
        )
        return

    if args.csv is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        csv_path = _normalize_csv_path(args.csv)
        print("=" * 110)
        print("FLAGSPARSE SDDMM - export to CSV")
        print("=" * 110)
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  dtypes: {','.join(_dtype_name(d) for d in value_dtypes)}  |  index_dtypes: {','.join(_dtype_name(d) for d in index_dtypes)}  |  acc_mode: {args.acc_mode}  |  K: {','.join(str(k) for k in k_dims)}  |  alpha: {args.alpha}  |  beta: {args.beta}  |  CSV: {csv_path}"
        )
        run_all_dtypes_export_csv(
            paths,
            csv_path,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            warmup=args.warmup,
            iters=args.iters,
            k_dims=k_dims,
            alpha=args.alpha,
            beta=args.beta,
            run_cusparse=run_cusparse_ref,
            acc_mode=args.acc_mode,
        )
        return

    print("=" * 150)
    print("FLAGSPARSE SDDMM - SuiteSparse .mtx batch (CSR pattern-guided)")
    print("=" * 150)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(
        f"dtypes: {','.join(_dtype_name(d) for d in value_dtypes)}  index_dtypes: {','.join(_dtype_name(d) for d in index_dtypes)}  acc_mode: {args.acc_mode}  K: {','.join(str(k) for k in k_dims)}  alpha: {args.alpha}  beta: {args.beta}  warmup: {args.warmup}  iters: {args.iters}"
    )
    print()
    total_passed = 0
    total_cases = 0
    for value_dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for k_dim in k_dims:
                results = run_mtx_batch(
                    paths,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=args.warmup,
                    iters=args.iters,
                    k_dim=k_dim,
                    alpha=args.alpha,
                    beta=args.beta,
                    run_cusparse=run_cusparse_ref,
                    acc_mode=args.acc_mode,
                )
                print_mtx_results(
                    results,
                    value_dtype,
                    index_dtype,
                    k_dim,
                    args.alpha,
                    args.beta,
                    args.acc_mode,
                )
                passed = sum(1 for entry in results if entry.get("status") == "PASS")
                total_passed += passed
                total_cases += len(results)
                print(
                    f"dtype={_dtype_name(value_dtype)} index_dtype={_dtype_name(index_dtype)} K={k_dim} Passed: {passed} / {len(results)}"
                )
    print(f"Total Passed: {total_passed} / {total_cases}")


if __name__ == "__main__":
    main()
