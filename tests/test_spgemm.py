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
SpGEMM tests: load SuiteSparse .mtx, run CSR SpGEMM(A@B), and report
error/performance in a SpMM-like table and CSV format.
"""

import argparse
import csv
import gc
import glob
import json
import os
import subprocess
import sys
import tempfile
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
import flagsparse.sparse_operations.spgemm_csr as ast_ops
from test_spmm import load_mtx_to_csr_torch

VALUE_DTYPES = [torch.float32, torch.float64]
INDEX_DTYPES = [torch.int32]
CSV_VALUE_DTYPES = [torch.float32, torch.float64]
CSV_INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 50
DEFAULT_INPUT_MODE = "auto"
TARGET_TIMED_WINDOW_SECONDS = 8.0
DEFAULT_REF_BLOCK_ROWS = 0  # auto
DEFAULT_COMPARE_BLOCK_ROWS = 2048
DEFAULT_COMPARE_DEVICE = "cpu"


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
    if value is None:
        return "N/A"
    return "PASS" if value else "FAIL"


def _append_error(current, message):
    msg = str(message)
    if not current:
        return msg
    return f"{current}; {msg}"


def _is_resource_error(message):
    text = str(message).lower()
    tokens = (
        "out of memory",
        "cudaerroroutofmemory",
        "cuda out of memory",
        "insufficient resources",
        "resource exhausted",
        "cusparsespgemm_workestimation",
        "cusparsespgemm_compute",
        "cusparse_status_insufficient_resources",
        "cublas_status_alloc_failed",
        "cannot allocate memory",
    )
    return any(tok in text for tok in tokens)


def _cleanup_reference_pools():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    if ast_ops.cp is not None:
        try:
            ast_ops.cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
        try:
            ast_ops.cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def _parse_ref_block_rows(raw_value):
    if raw_value is None:
        return DEFAULT_REF_BLOCK_ROWS
    s = str(raw_value).strip().lower()
    if s in ("auto", "0", ""):
        return 0
    v = int(s)
    if v <= 0:
        raise ValueError("--ref-block-rows must be positive or 'auto'")
    return v


def _candidate_block_rows(n_rows, requested):
    if requested and requested > 0:
        return [int(requested)]
    candidates = [8192, 4096, 2048, 1024, 512, 256]
    out = []
    for c in candidates:
        if c < n_rows:
            out.append(c)
    if not out:
        out = [max(1, n_rows)]
    return out


def _slice_csr_rows(data, indices, indptr, shape, row_start, row_end):
    ptr_start = int(indptr[row_start].item())
    ptr_end = int(indptr[row_end].item())
    sub_data = data[ptr_start:ptr_end]
    sub_indices = indices[ptr_start:ptr_end]
    sub_indptr = (indptr[row_start : row_end + 1] - ptr_start).to(torch.int64)
    sub_shape = (int(row_end - row_start), int(shape[1]))
    return sub_data, sub_indices, sub_indptr, sub_shape


def _concat_csr_row_blocks(blocks, n_rows, n_cols, device, data_dtype=torch.float32):
    if not blocks:
        return (
            torch.empty(0, dtype=data_dtype, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.zeros(n_rows + 1, dtype=torch.int64, device=device),
            (n_rows, n_cols),
        )
    data_dtype = blocks[0][0].dtype
    data_parts = []
    idx_parts = []
    indptr_parts = [torch.zeros(1, dtype=torch.int64, device=device)]
    nnz_acc = 0
    row_acc = 0
    for data_b, idx_b, indptr_b, shape_b in blocks:
        rows_b = int(shape_b[0])
        row_acc += rows_b
        data_parts.append(data_b)
        idx_parts.append(idx_b.to(torch.int32))
        if rows_b > 0:
            indptr_parts.append(indptr_b[1:].to(torch.int64) + nnz_acc)
        nnz_acc += int(data_b.numel())
    if row_acc != int(n_rows):
        raise RuntimeError(
            f"blocked CSR concat row mismatch: got {row_acc}, expected {n_rows}"
        )
    data = (
        torch.cat(data_parts)
        if data_parts
        else torch.empty(0, dtype=data_dtype, device=device)
    )
    indices = (
        torch.cat(idx_parts)
        if idx_parts
        else torch.empty(0, dtype=torch.int32, device=device)
    )
    indptr = torch.cat(indptr_parts)
    return data, indices, indptr, (int(n_rows), int(n_cols))


def _allclose_error_ratio(actual, reference, atol, rtol):
    if actual.numel() == 0:
        return 0.0
    diff = torch.abs(actual - reference).to(torch.float64)
    tol = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / tol).item())


def _csr_sorted_pairs_block(data, indices, indptr, n_cols, row_start, row_end):
    row_start = int(row_start)
    row_end = int(row_end)
    if row_end <= row_start:
        return (
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.empty(0, dtype=data.dtype, device=data.device),
        )
    ptr = indptr[row_start : row_end + 1].to(torch.int64)
    start = int(ptr[0].item())
    end = int(ptr[-1].item())
    if end <= start:
        return (
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.empty(0, dtype=data.dtype, device=data.device),
        )
    row_counts = ptr[1:] - ptr[:-1]
    rows = torch.repeat_interleave(
        torch.arange(row_start, row_end, device=data.device, dtype=torch.int64),
        row_counts,
    )
    cols = indices[start:end].to(torch.int64)
    vals = data[start:end]
    keys = rows * max(1, int(n_cols)) + cols
    order = torch.argsort(keys)
    return keys[order], vals[order]


def _spgemm_compare_metrics(candidate, reference, value_dtype):
    c_data, c_indices, c_indptr, c_shape = candidate
    r_data, r_indices, r_indptr, r_shape = reference
    if c_shape != r_shape:
        return {
            "pattern_ok": False,
            "pass": False,
            "err_ratio": float("inf"),
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "reason": f"shape mismatch {c_shape} vs {r_shape}",
        }

    c_nnz = int(c_indptr[-1].item()) if c_indptr.numel() > 0 else 0
    r_nnz = int(r_indptr[-1].item()) if r_indptr.numel() > 0 else 0
    if c_nnz != r_nnz:
        return {
            "pattern_ok": False,
            "pass": False,
            "err_ratio": float("inf"),
            "max_abs_error": float("inf"),
            "max_relative_error": float("inf"),
            "reason": f"nnz mismatch {c_nnz} vs {r_nnz}",
        }
    if c_nnz == 0:
        return {
            "pattern_ok": True,
            "pass": True,
            "err_ratio": 0.0,
            "max_abs_error": 0.0,
            "max_relative_error": 0.0,
            "reason": "ok",
        }

    atol, rtol = ast_ops._tolerance_for_dtype(value_dtype)
    n_rows = int(c_shape[0])
    compare_rows = max(1, int(DEFAULT_COMPARE_BLOCK_ROWS))
    err_ratio = 0.0
    max_abs = 0.0
    ref_max = 0.0
    row = 0
    while row < n_rows:
        chunk_rows = min(compare_rows, n_rows - row)
        while True:
            try:
                c_keys, c_vals = _csr_sorted_pairs_block(
                    c_data, c_indices, c_indptr, c_shape[1], row, row + chunk_rows
                )
                r_keys, r_vals = _csr_sorted_pairs_block(
                    r_data, r_indices, r_indptr, r_shape[1], row, row + chunk_rows
                )
                break
            except Exception as exc:
                if _is_resource_error(exc) and chunk_rows > 1:
                    chunk_rows = max(1, chunk_rows // 2)
                    continue
                raise

        if c_keys.numel() != r_keys.numel() or not torch.equal(c_keys, r_keys):
            return {
                "pattern_ok": False,
                "pass": False,
                "err_ratio": float("inf"),
                "max_abs_error": float("inf"),
                "max_relative_error": float("inf"),
                "reason": f"sparsity pattern mismatch in rows [{row}, {row + chunk_rows})",
            }
        if c_vals.numel() > 0:
            err_ratio = max(
                err_ratio, _allclose_error_ratio(c_vals, r_vals, atol, rtol)
            )
            abs_diff = torch.abs(c_vals - r_vals)
            max_abs = max(max_abs, float(torch.max(abs_diff).item()))
            ref_max = max(ref_max, float(torch.max(torch.abs(r_vals)).item()))
        row += chunk_rows

    max_rel = 0.0 if ref_max == 0.0 else max_abs / ref_max
    ok = (not torch.isnan(torch.tensor(err_ratio)).item()) and err_ratio <= 1.0
    return {
        "pattern_ok": True,
        "pass": bool(ok),
        "err_ratio": err_ratio,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "reason": "ok" if ok else "value mismatch",
    }


def _compare_spgemm_cpu(candidate, reference, value_dtype):
    return _spgemm_compare_metrics(candidate, reference, value_dtype)


def _classify_reference_reason(*messages):
    merged = " ".join(str(m).lower() for m in messages if m)
    if not merged:
        return "REF_UNAVAILABLE"
    if (
        "out of memory" in merged
        or "cudaerroroutofmemory" in merged
        or "cuda out of memory" in merged
        or "cannot allocate memory" in merged
        or "cublas_status_alloc_failed" in merged
    ):
        return "REF_OOM"
    if (
        "insufficient resources" in merged
        or "cusparsespgemm_workestimation" in merged
        or "cusparsespgemm_compute" in merged
        or "cusparse_status_insufficient_resources" in merged
        or "resource exhausted" in merged
        or "resource" in merged
    ):
        return "REF_RESOURCE"
    return "REF_UNAVAILABLE"


def _normalize_csv_path(csv_path):
    csv_path = str(csv_path)
    if not csv_path.lower().endswith(".csv"):
        csv_path = f"{csv_path}.csv"
    parent = os.path.dirname(os.path.abspath(csv_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    return csv_path


def _log_stage(path, stage, start_time):
    elapsed = time.perf_counter() - start_time
    print(
        f"[SpGEMM][{os.path.basename(path)}] {stage} (elapsed={elapsed:.2f}s)",
        flush=True,
    )


def _resolve_input_mode(requested_mode, shape):
    n_rows, n_cols = shape
    if requested_mode == "a_equals_b":
        if n_rows != n_cols:
            raise ValueError(
                f"input_mode=a_equals_b requires square matrix, got {n_rows}x{n_cols}"
            )
        return "A_EQUALS_B"
    if requested_mode == "a_at":
        return "A_AT"
    if requested_mode == "auto":
        return "A_EQUALS_B" if n_rows == n_cols else "A_AT"
    raise ValueError(f"unsupported input_mode: {requested_mode}")


def _build_spgemm_rhs(a_data, a_indices, a_indptr, a_shape, mode):
    if mode == "A_EQUALS_B":
        return a_data, a_indices, a_indptr, a_shape
    if mode != "A_AT":
        raise ValueError(f"unsupported resolved mode: {mode}")
    a_t = ast_ops._to_torch_csr(a_data, a_indices, a_indptr, a_shape)
    # CSR^T may materialize as CSC; convert through COO so downstream always receives CSR.
    b_t = a_t.transpose(0, 1).to_sparse_coo().coalesce()
    return ast_ops._torch_sparse_to_csr(b_t)


def _build_torch_spgemm_reference(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    a_csr = torch.sparse_csr_tensor(
        a_indptr.to(torch.int64),
        a_indices.to(torch.int64),
        a_data,
        size=a_shape,
        device=a_data.device,
    )
    b_csr = torch.sparse_csr_tensor(
        b_indptr.to(torch.int64),
        b_indices.to(torch.int64),
        b_data,
        size=b_shape,
        device=b_data.device,
    )
    ref_format = "CSR"
    try:
        op = lambda: torch.sparse.mm(a_csr, b_csr)
        ref_sparse = op()
    except Exception:
        ref_format = "COO"
        a_coo = a_csr.to_sparse_coo().coalesce()
        b_coo = b_csr.to_sparse_coo().coalesce()
        op = lambda: torch.sparse.mm(a_coo, b_coo)
        ref_sparse = op()
    if ref_sparse.layout not in (torch.sparse_coo, torch.sparse_csr):
        raise RuntimeError(
            f"Unexpected torch sparse.mm result layout: {ref_sparse.layout}"
        )
    return ast_ops._torch_sparse_to_csr(ref_sparse), ref_format, op


def _build_torch_spgemm_reference_blocked(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    block_rows,
):
    n_rows = int(a_shape[0])

    def _op():
        blocks = []
        formats = set()
        for row_start in range(0, n_rows, block_rows):
            row_end = min(row_start + block_rows, n_rows)
            a_blk = _slice_csr_rows(
                a_data, a_indices, a_indptr, a_shape, row_start, row_end
            )
            ref_blk, fmt_blk, _ = _build_torch_spgemm_reference(
                a_blk[0],
                a_blk[1],
                a_blk[2],
                a_blk[3],
                b_data,
                b_indices,
                b_indptr,
                b_shape,
            )
            blocks.append(ref_blk)
            formats.add(fmt_blk)
        fmt = "BLOCKED_CSR" if formats == {"CSR"} else "BLOCKED_MIXED"
        csr = _concat_csr_row_blocks(
            blocks,
            n_rows=n_rows,
            n_cols=int(b_shape[1]),
            device=a_data.device,
            data_dtype=a_data.dtype,
        )
        return csr, fmt

    csr_result, fmt_result = _op()

    def _bench_op():
        csr, _ = _op()
        return csr

    return csr_result, fmt_result, _bench_op


def _build_cupy_spgemm_reference(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    if ast_ops.cp is None or ast_ops.cpx_sparse is None:
        raise RuntimeError("CuPy/cuSPARSE is not available")
    a_cp = ast_ops.cpx_sparse.csr_matrix(
        (
            ast_ops._cupy_from_torch(a_data),
            ast_ops._cupy_from_torch(a_indices.to(torch.int64)),
            ast_ops._cupy_from_torch(a_indptr.to(torch.int64)),
        ),
        shape=a_shape,
    )
    b_cp = ast_ops.cpx_sparse.csr_matrix(
        (
            ast_ops._cupy_from_torch(b_data),
            ast_ops._cupy_from_torch(b_indices.to(torch.int64)),
            ast_ops._cupy_from_torch(b_indptr.to(torch.int64)),
        ),
        shape=b_shape,
    )

    # The timed window holds exactly the vendor SpGEMM call and nothing else.
    # cupy's ``@`` redoes the whole cusparseSpGEMM sequence (createDescr ->
    # workEstimation x2 -> compute x2 -> alloc -> copy) on every call with zero
    # cross-call caching, so no amortisable setup is being hoisted out of it -- what
    # stays outside is only the torch<->cupy marshalling and the
    # CSR->COO->coalesce->CSR round trip below, which exists solely so the result can
    # be compared against ours. FlagSparse never pays that, and ``coalesce()`` is a
    # full sort over nnz(C); leaving it inside the clock made this column 3-15x slower
    # than cuSPARSE actually is (roadNet 9.27ms vs 0.63ms, ecology1 13.50 vs 0.85) and
    # is what produced the bogus "4.0x cuSPARSE" headline.
    def _op():
        return a_cp @ b_cp

    def _to_torch_csr(c_cp):
        c_coo = c_cp.tocoo()
        rows = ast_ops._torch_from_cupy(c_coo.row).to(torch.int64)
        cols = ast_ops._torch_from_cupy(c_coo.col).to(torch.int64)
        vals = ast_ops._torch_from_cupy(c_coo.data).to(a_data.dtype)
        c_t = torch.sparse_coo_tensor(
            torch.stack([rows, cols]),
            vals,
            (a_shape[0], b_shape[1]),
            device=a_data.device,
        ).coalesce()
        return ast_ops._torch_sparse_to_csr(c_t)

    return _to_torch_csr(_op()), "CSR", _op


def _build_hipsparse_spgemm_reference(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
):
    """DCU/ROCm vendor SpGEMM reference, the hipSPARSE analogue of the CuPy path."""
    reason = ast_ops._hipsparse_spgemm_csr_skip_reason(
        a_data.dtype, a_indices.dtype, a_indptr.dtype, b_indices.dtype, b_indptr.dtype
    )
    if reason is not None:
        raise RuntimeError(reason)

    def _op():
        return ast_ops._spgemm_csr_ref_hipsparse(
            a_data, a_indices, a_indptr, a_shape, b_data, b_indices, b_indptr, b_shape
        )

    return _op(), "CSR", _op


def _build_cupy_spgemm_reference_blocked(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    block_rows,
):
    n_rows = int(a_shape[0])

    def _op():
        blocks = []
        for row_start in range(0, n_rows, block_rows):
            row_end = min(row_start + block_rows, n_rows)
            a_blk = _slice_csr_rows(
                a_data, a_indices, a_indptr, a_shape, row_start, row_end
            )
            ref_blk, _, _ = _build_cupy_spgemm_reference(
                a_blk[0],
                a_blk[1],
                a_blk[2],
                a_blk[3],
                b_data,
                b_indices,
                b_indptr,
                b_shape,
            )
            blocks.append(ref_blk)
        return _concat_csr_row_blocks(
            blocks,
            n_rows=n_rows,
            n_cols=int(b_shape[1]),
            device=a_data.device,
            data_dtype=a_data.dtype,
        )

    return _op(), "BLOCKED_CSR", _op


def _run_reference_worker_subprocess(
    backend,
    mtx_path,
    value_dtype,
    input_mode,
    warmup,
    iters,
    blocked_retry,
    block_rows,
    ref_cleanup,
):
    py = sys.executable
    if not py:
        return {
            "success": False,
            "reason": "isolated retry skipped: python executable is unavailable",
            "fail_stage": "isolated",
            "exec_mode": "isolated",
        }
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pt")
    tmp_path = tmp.name
    tmp.close()
    cmd = [
        py,
        str(Path(__file__).resolve()),
        "--_ref-worker",
        backend,
        "--_worker-mtx",
        str(mtx_path),
        "--_worker-output",
        tmp_path,
        "--dtype",
        _dtype_name(value_dtype),
        "--index-dtype",
        "int32",
        "--warmup",
        str(int(warmup)),
        "--iters",
        str(int(iters)),
        "--_worker-input-mode",
        str(input_mode).lower(),
    ]
    if block_rows > 0:
        cmd.extend(["--_worker-block-rows", str(int(block_rows))])
    if not blocked_retry:
        cmd.append("--_worker-no-blocked")
    if not ref_cleanup:
        cmd.append("--_worker-no-cleanup")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    payload = None
    try:
        if os.path.exists(tmp_path):
            payload = torch.load(tmp_path, map_location="cpu")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    if isinstance(payload, dict) and payload.get("success"):
        payload["exec_mode"] = "isolated"
        payload["retry_count"] = int(payload.get("retry_count", 0))
        return payload
    stderr = proc.stderr.strip() if proc.stderr else ""
    stdout = proc.stdout.strip() if proc.stdout else ""
    reason = None
    if isinstance(payload, dict):
        reason = payload.get("reason")
    if not reason:
        reason = (
            stderr or stdout or f"isolated worker failed with code {proc.returncode}"
        )
    return {
        "success": False,
        "reason": reason,
        "fail_stage": "isolated",
        "exec_mode": "isolated",
    }


def _run_reference_with_retries(
    backend,
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    warmup,
    iters,
    blocked_retry,
    block_rows,
    isolated_retry,
    ref_cleanup,
    mtx_path,
    value_dtype,
    input_mode,
    result_device="gpu",
):
    if backend == "torch":
        run_direct = _build_torch_spgemm_reference
        run_blocked = _build_torch_spgemm_reference_blocked
    elif backend == "hipsparse":
        run_direct = _build_hipsparse_spgemm_reference
        # hipSPARSE has no blocked variant; the torch builder is the memory-relief
        # retry on DCU just as it is when CuPy blocking is unavailable.
        run_blocked = _build_torch_spgemm_reference_blocked
    else:
        run_direct = _build_cupy_spgemm_reference
        run_blocked = _build_cupy_spgemm_reference_blocked
    attempted_modes = ["direct"]

    def _mark_mode(mode):
        if mode not in attempted_modes:
            attempted_modes.append(mode)

    def _finalize():
        state["attempted_modes"] = ">".join(attempted_modes)
        return state

    state = {
        "success": False,
        "result": None,
        "format": None,
        "ms": None,
        "exec_mode": "direct",
        "retry_count": 0,
        "peak_block_rows": None,
        "reason": None,
        "fail_stage": None,
        "attempted_modes": "direct",
    }
    try:
        ref_result, ref_format, ref_op = run_direct(
            a_data,
            a_indices,
            a_indptr,
            a_shape,
            b_data,
            b_indices,
            b_indptr,
            b_shape,
        )
        _, ref_ms = ast_ops._benchmark_cuda_op(ref_op, warmup=warmup, iters=iters)
        compare_result = _convert_result_for_compare(
            ref_result, result_device, a_data.device
        )
        ref_result = None
        if result_device == "cpu" and ref_cleanup:
            _cleanup_reference_pools()
        state.update(
            {
                "success": True,
                "result": compare_result,
                "format": ref_format,
                "ms": ref_ms,
                "exec_mode": "direct",
            }
        )
        return _finalize()
    except Exception as exc:
        state["reason"] = str(exc)
        state["fail_stage"] = "direct"
        if ref_cleanup:
            _cleanup_reference_pools()

    if blocked_retry and _is_resource_error(state["reason"]):
        _mark_mode("blocked")
        state["exec_mode"] = "blocked"
        for br in _candidate_block_rows(int(a_shape[0]), block_rows):
            try:
                state["retry_count"] += 1
                ref_result, ref_format, ref_op = run_blocked(
                    a_data,
                    a_indices,
                    a_indptr,
                    a_shape,
                    b_data,
                    b_indices,
                    b_indptr,
                    b_shape,
                    block_rows=int(br),
                )
                _, ref_ms = ast_ops._benchmark_cuda_op(
                    ref_op, warmup=warmup, iters=iters
                )
                compare_result = _convert_result_for_compare(
                    ref_result, result_device, a_data.device
                )
                ref_result = None
                if result_device == "cpu" and ref_cleanup:
                    _cleanup_reference_pools()
                state.update(
                    {
                        "success": True,
                        "result": compare_result,
                        "format": ref_format,
                        "ms": ref_ms,
                        "exec_mode": "blocked",
                        "peak_block_rows": int(br),
                        "reason": None,
                        "fail_stage": None,
                    }
                )
                return _finalize()
            except Exception as blk_exc:
                state["reason"] = str(blk_exc)
                state["fail_stage"] = "blocked"
                state["peak_block_rows"] = int(br)
                if ref_cleanup:
                    _cleanup_reference_pools()

    if isolated_retry and _is_resource_error(state["reason"]):
        _mark_mode("isolated")
        state["exec_mode"] = "isolated"
        state["retry_count"] += 1
        iso = _run_reference_worker_subprocess(
            backend=backend,
            mtx_path=mtx_path,
            value_dtype=value_dtype,
            input_mode=input_mode,
            warmup=warmup,
            iters=iters,
            blocked_retry=blocked_retry,
            block_rows=block_rows,
            ref_cleanup=ref_cleanup,
        )
        if iso.get("success"):
            ref_payload = iso.get("result")
            if not isinstance(ref_payload, (tuple, list)) or len(ref_payload) != 4:
                state["reason"] = "isolated worker produced invalid CSR payload"
                state["fail_stage"] = "isolated"
                return _finalize()
            state.update(
                {
                    "success": True,
                    "result": _convert_result_for_compare(
                        (
                            ref_payload[0],
                            ref_payload[1],
                            ref_payload[2],
                            tuple(ref_payload[3]),
                        ),
                        result_device,
                        a_data.device,
                    ),
                    "format": iso.get("format", "CSR"),
                    "ms": iso.get("ms"),
                    "exec_mode": "isolated",
                    "peak_block_rows": iso.get("peak_block_rows"),
                    "reason": None,
                    "fail_stage": None,
                }
            )
            return _finalize()
        state["reason"] = iso.get("reason")
        state["fail_stage"] = iso.get("fail_stage", "isolated")
    return _finalize()


def _pick_effective_benchmark_loops(
    warmup, iters, first_call_ms, target_window_seconds
):
    warmup = max(0, int(warmup))
    iters = max(1, int(iters))
    per_call_s = max(float(first_call_ms) / 1000.0, 1e-4)
    target_iters = max(1, int(float(target_window_seconds) / per_call_s))
    eff_iters = min(iters, target_iters)
    warmup_cap = max(1, eff_iters // 2)
    eff_warmup = min(warmup, warmup_cap)
    return eff_warmup, eff_iters


def _benchmark_flagsparse_spgemm(
    a_data,
    a_indices,
    a_indptr,
    a_shape,
    b_data,
    b_indices,
    b_indptr,
    b_shape,
    warmup,
    iters,
    adaptive_loops,
    target_window_seconds,
    start_time,
    mtx_path,
):
    _log_stage(mtx_path, "prepare", start_time)
    torch.cuda.synchronize()
    t_prepare0 = time.perf_counter()
    prepared = ast.prepare_spgemm_csr(
        a_data,
        a_indices,
        a_indptr,
        a_shape,
        b_data,
        b_indices,
        b_indptr,
        b_shape,
    )
    torch.cuda.synchronize()
    prepare_ms = (time.perf_counter() - t_prepare0) * 1000.0

    _log_stage(mtx_path, "first-call", start_time)
    torch.cuda.synchronize()
    t_first0 = time.perf_counter()
    first_result, first_meta = ast.flagsparse_spgemm_csr(
        prepared=prepared, return_meta=True
    )
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t_first0) * 1000.0
    first_meta = dict(first_meta)
    first_meta["prepare_ms"] = prepare_ms

    if adaptive_loops:
        eff_warmup, eff_iters = _pick_effective_benchmark_loops(
            warmup=warmup,
            iters=iters,
            first_call_ms=first_call_ms,
            target_window_seconds=target_window_seconds,
        )
    else:
        eff_warmup = max(0, int(warmup))
        eff_iters = max(1, int(iters))
    first_meta["effective_warmup"] = eff_warmup
    first_meta["effective_iters"] = eff_iters

    out_buffers = (first_result[0], first_result[1], first_result[2])
    _log_stage(mtx_path, f"timed-run warmup={eff_warmup} iters={eff_iters}", start_time)
    # ``prepare`` is inside the timed window. The cuSPARSE baseline has no
    # amortisable setup -- every cupy ``A @ B`` redoes createDescr ->
    # workEstimation -> compute -> copy with no cross-call caching -- so timing
    # only the compute half would compare unequal amounts of work, and would let
    # cost be "saved" merely by relocating it into prepare. ``prepare_ms`` is
    # still reported separately for diagnosis.
    triton_result, triton_ms = ast_ops._benchmark_cuda_op(
        lambda: ast.flagsparse_spgemm_csr(
            a_data,
            a_indices,
            a_indptr,
            a_shape,
            b_data,
            b_indices,
            b_indptr,
            b_shape,
            out=out_buffers,
        ),
        warmup=eff_warmup,
        iters=eff_iters,
    )
    return triton_result, triton_ms, first_call_ms, first_meta


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    input_mode=DEFAULT_INPUT_MODE,
    adaptive_loops=False,
    target_window_seconds=TARGET_TIMED_WINDOW_SECONDS,
    ref_blocked_retry=True,
    ref_block_rows=DEFAULT_REF_BLOCK_ROWS,
    ref_isolated_retry=True,
    ref_cleanup=True,
    compare_device=DEFAULT_COMPARE_DEVICE,
):
    case_start = time.perf_counter()
    device = torch.device("cuda")

    _log_stage(mtx_path, "load", case_start)
    a_data, a_indices, a_indptr, a_shape = load_mtx_to_csr_torch(
        mtx_path, dtype=value_dtype, device=device
    )
    a_indices = a_indices.to(index_dtype)
    resolved_mode = _resolve_input_mode(input_mode, a_shape)
    b_data, b_indices, b_indptr, b_shape = _build_spgemm_rhs(
        a_data, a_indices, a_indptr, a_shape, resolved_mode
    )
    b_indices = b_indices.to(index_dtype)

    nnz_a = int(a_data.numel())
    nnz_b = int(b_data.numel())
    print(
        f"[SpGEMM][{os.path.basename(mtx_path)}] preflight mode={resolved_mode} "
        f"shape_a={a_shape} shape_b={b_shape} nnz_a={nnz_a} nnz_b={nnz_b}",
        flush=True,
    )
    result = {
        "path": mtx_path,
        "shape": a_shape,
        "shape_a": a_shape,
        "shape_b": b_shape,
        "nnz": nnz_a,
        "nnz_a": nnz_a,
        "nnz_b": nnz_b,
        "nnz_c": None,
        "input_mode": resolved_mode,
        "error": None,
        "triton_started": False,
        "ref_started": False,
        "ref_reason_code": None,
        "triton_ms": None,
        "triton_first_call_ms": None,
        "prepare_ms": None,
        "count_ms": None,
        "fill_ms": None,
        "bucket_nrows_short": None,
        "bucket_nrows_medium": None,
        "bucket_nrows_long": None,
        "bucket_ms_short": None,
        "bucket_ms_medium": None,
        "bucket_ms_long": None,
        "long_row_sliced_count": None,
        "effective_warmup": None,
        "effective_iters": None,
        "pytorch_ms": None,
        "cusparse_ms": None,
        "err_pt": None,
        "err_cu": None,
        "max_abs_err_pt": None,
        "max_rel_err_pt": None,
        "max_abs_err_cu": None,
        "max_rel_err_cu": None,
        "triton_ok_pt": None,
        "triton_ok_cu": None,
        "pytorch_reason": None,
        "cusparse_reason": None,
        "pytorch_format": None,
        "pt_exec_mode": None,
        "cu_exec_mode": None,
        "attempted_modes_pt": None,
        "attempted_modes_cu": None,
        "compare_status": "OK",
        "compare_device": compare_device,
        "pt_retry_count": 0,
        "cu_retry_count": 0,
        "ref_peak_block_rows": None,
        "ref_fail_stage": None,
        "status": "UNKNOWN",
    }

    triton_result = None
    triton_compare_result = None
    _log_stage(mtx_path, "reference", case_start)
    result["ref_started"] = True
    pt_compared = False
    cu_compared = False
    pt_ref_success = False
    cu_ref_success = False
    pt_ref_result = None
    cu_ref_result = None
    ref_warmup = (
        result["effective_warmup"] if result["effective_warmup"] is not None else warmup
    )
    ref_iters = (
        result["effective_iters"] if result["effective_iters"] is not None else iters
    )
    ref_warmup = max(0, int(ref_warmup))
    ref_iters = max(1, int(ref_iters))

    pt_ref = _run_reference_with_retries(
        backend="torch",
        a_data=a_data,
        a_indices=a_indices,
        a_indptr=a_indptr,
        a_shape=a_shape,
        b_data=b_data,
        b_indices=b_indices,
        b_indptr=b_indptr,
        b_shape=b_shape,
        warmup=ref_warmup,
        iters=ref_iters,
        blocked_retry=ref_blocked_retry,
        block_rows=ref_block_rows,
        isolated_retry=ref_isolated_retry,
        ref_cleanup=ref_cleanup,
        mtx_path=mtx_path,
        value_dtype=value_dtype,
        input_mode=input_mode,
        result_device=compare_device,
    )
    result["pt_exec_mode"] = pt_ref.get("exec_mode")
    result["attempted_modes_pt"] = pt_ref.get("attempted_modes")
    result["pt_retry_count"] = int(pt_ref.get("retry_count", 0))
    if pt_ref.get("peak_block_rows") is not None:
        result["ref_peak_block_rows"] = int(pt_ref["peak_block_rows"])
    if pt_ref.get("success"):
        pt_ref_success = True
        pt_ref_result = pt_ref.get("result")
        result["pytorch_format"] = pt_ref.get("format")
        result["pytorch_ms"] = pt_ref.get("ms")
    else:
        result["pytorch_reason"] = pt_ref.get("reason")
        result["ref_fail_stage"] = pt_ref.get("fail_stage")
        result["error"] = _append_error(
            result["error"], f"pt_ref: {pt_ref.get('reason')}"
        )
    if ref_cleanup:
        _cleanup_reference_pools()

    if run_cusparse:
        # Vendor baseline per backend: hipSPARSE on DCU/ROCm, cuSPARSE via CuPy on CUDA.
        vendor_backend, _vendor_reason = ast_ops._spgemm_csr_sparse_ref_backend(
            value_dtype, a_indices.dtype, a_indptr.dtype, b_indices.dtype, b_indptr.dtype
        )
        cu_ref = _run_reference_with_retries(
            backend="hipsparse" if vendor_backend == "hipsparse" else "cupy",
            a_data=a_data,
            a_indices=a_indices,
            a_indptr=a_indptr,
            a_shape=a_shape,
            b_data=b_data,
            b_indices=b_indices,
            b_indptr=b_indptr,
            b_shape=b_shape,
            warmup=ref_warmup,
            iters=ref_iters,
            blocked_retry=ref_blocked_retry,
            block_rows=ref_block_rows,
            isolated_retry=ref_isolated_retry,
            ref_cleanup=ref_cleanup,
            mtx_path=mtx_path,
            value_dtype=value_dtype,
            input_mode=input_mode,
            result_device=compare_device,
        )
        result["cu_exec_mode"] = cu_ref.get("exec_mode")
        result["attempted_modes_cu"] = cu_ref.get("attempted_modes")
        result["cu_retry_count"] = int(cu_ref.get("retry_count", 0))
        if cu_ref.get("peak_block_rows") is not None:
            cur_peak = result.get("ref_peak_block_rows")
            result["ref_peak_block_rows"] = (
                int(cu_ref["peak_block_rows"])
                if cur_peak is None
                else max(int(cur_peak), int(cu_ref["peak_block_rows"]))
            )
        if cu_ref.get("success"):
            cu_ref_success = True
            result["cusparse_ms"] = cu_ref.get("ms")
            cu_ref_result = cu_ref.get("result")
        else:
            result["cusparse_reason"] = cu_ref.get("reason")
            if result.get("ref_fail_stage") is None:
                result["ref_fail_stage"] = cu_ref.get("fail_stage")
            result["error"] = _append_error(
                result["error"], f"cu_ref: {cu_ref.get('reason')}"
            )
    else:
        result["cusparse_reason"] = "CuPy/cuSPARSE reference is disabled"
        result["cu_exec_mode"] = "disabled"
        result["attempted_modes_cu"] = "disabled"
    # Ensure reference-side temporary allocations are released before Triton run.
    _cleanup_reference_pools()

    _log_stage(mtx_path, "triton", case_start)
    try:
        result["triton_started"] = True
        triton_result, triton_ms, triton_first_ms, meta = _benchmark_flagsparse_spgemm(
            a_data,
            a_indices,
            a_indptr,
            a_shape,
            b_data,
            b_indices,
            b_indptr,
            b_shape,
            warmup=warmup,
            iters=iters,
            adaptive_loops=adaptive_loops,
            target_window_seconds=target_window_seconds,
            start_time=case_start,
            mtx_path=mtx_path,
        )
        result["triton_ms"] = triton_ms
        result["triton_first_call_ms"] = triton_first_ms
        result["prepare_ms"] = meta.get("prepare_ms")
        result["count_ms"] = meta.get("count_ms")
        result["fill_ms"] = meta.get("fill_ms")
        result["bucket_nrows_short"] = meta.get("bucket_nrows_short")
        result["bucket_nrows_medium"] = meta.get("bucket_nrows_medium")
        result["bucket_nrows_long"] = meta.get("bucket_nrows_long")
        result["bucket_ms_short"] = meta.get("bucket_ms_short")
        result["bucket_ms_medium"] = meta.get("bucket_ms_medium")
        result["bucket_ms_long"] = meta.get("bucket_ms_long")
        result["long_row_sliced_count"] = meta.get("long_row_sliced_count")
        result["effective_warmup"] = meta.get("effective_warmup")
        result["effective_iters"] = meta.get("effective_iters")
        result["nnz_c"] = (
            int(triton_result[0].numel()) if triton_result is not None else None
        )
        triton_compare_result = _convert_result_for_compare(
            triton_result,
            compare_device,
            device=device,
        )
        if compare_device == "cpu":
            triton_result = None
            _cleanup_reference_pools()
    except Exception as exc:
        result["error"] = _append_error(result["error"], f"triton: {exc}")

    _log_stage(mtx_path, "compare", case_start)
    if triton_compare_result is not None and pt_ref_result is not None:
        try:
            compare_fn = (
                _compare_spgemm_cpu
                if compare_device == "cpu"
                else _spgemm_compare_metrics
            )
            pt_metrics = compare_fn(triton_compare_result, pt_ref_result, value_dtype)
            result["triton_ok_pt"] = pt_metrics["pass"]
            result["err_pt"] = pt_metrics["err_ratio"]
            result["max_abs_err_pt"] = pt_metrics["max_abs_error"]
            result["max_rel_err_pt"] = pt_metrics["max_relative_error"]
            if not pt_metrics["pattern_ok"]:
                result["error"] = _append_error(
                    result["error"], f"pt_ref: {pt_metrics['reason']}"
                )
            pt_compared = True
        except Exception as cmp_exc:
            cmp_msg = str(cmp_exc)
            result["error"] = _append_error(result["error"], f"pt_compare: {cmp_msg}")
            if _is_resource_error(cmp_msg):
                result["compare_status"] = "COMPARE_OOM"
            elif result.get("compare_status") == "OK":
                result["compare_status"] = "COMPARE_FAIL"
        finally:
            if compare_device == "cpu":
                pt_ref_result = None
                if ref_cleanup:
                    _cleanup_reference_pools()
    if triton_compare_result is not None and cu_ref_result is not None:
        try:
            compare_fn = (
                _compare_spgemm_cpu
                if compare_device == "cpu"
                else _spgemm_compare_metrics
            )
            cu_metrics = compare_fn(triton_compare_result, cu_ref_result, value_dtype)
            result["triton_ok_cu"] = cu_metrics["pass"]
            result["err_cu"] = cu_metrics["err_ratio"]
            result["max_abs_err_cu"] = cu_metrics["max_abs_error"]
            result["max_rel_err_cu"] = cu_metrics["max_relative_error"]
            if not cu_metrics["pattern_ok"]:
                result["error"] = _append_error(
                    result["error"], f"cu_ref: {cu_metrics['reason']}"
                )
            cu_compared = True
        except Exception as cmp_exc:
            cmp_msg = str(cmp_exc)
            result["error"] = _append_error(result["error"], f"cu_compare: {cmp_msg}")
            if _is_resource_error(cmp_msg):
                result["compare_status"] = "COMPARE_OOM"
            elif result.get("compare_status") == "OK":
                result["compare_status"] = "COMPARE_FAIL"
        finally:
            if compare_device == "cpu":
                cu_ref_result = None
                if ref_cleanup:
                    _cleanup_reference_pools()

    if triton_compare_result is None:
        result["status"] = "FAIL"
        result["ref_reason_code"] = _classify_reference_reason(
            result.get("pytorch_reason"),
            result.get("cusparse_reason"),
            result.get("error"),
        )
        if ref_cleanup:
            _cleanup_reference_pools()
        return result

    if pt_compared or cu_compared:
        result["status"] = (
            "PASS" if (result["triton_ok_pt"] or result["triton_ok_cu"]) else "FAIL"
        )
    else:
        had_ref_success = pt_ref_success or cu_ref_success
        if had_ref_success and result.get("compare_status") != "OK":
            result["status"] = "FAIL"
            result["ref_reason_code"] = result.get("compare_status")
        else:
            ref_code = _classify_reference_reason(
                result.get("pytorch_reason"),
                result.get("cusparse_reason"),
            )
            result["ref_reason_code"] = ref_code
            result["status"] = ref_code
    if ref_cleanup:
        _cleanup_reference_pools()
    return result


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    input_mode=DEFAULT_INPUT_MODE,
    adaptive_loops=False,
    target_window_seconds=TARGET_TIMED_WINDOW_SECONDS,
    ref_blocked_retry=True,
    ref_block_rows=DEFAULT_REF_BLOCK_ROWS,
    ref_isolated_retry=True,
    ref_cleanup=True,
    compare_device=DEFAULT_COMPARE_DEVICE,
    on_result=None,
    on_entry=None,
):
    results = []
    total = len(mtx_paths)
    for idx, path in enumerate(mtx_paths, start=1):
        print(f"[SpGEMM] ({idx}/{total}) {path}", flush=True)
        try:
            entry = run_one_mtx(
                path,
                value_dtype=value_dtype,
                index_dtype=index_dtype,
                warmup=warmup,
                iters=iters,
                run_cusparse=run_cusparse,
                input_mode=input_mode,
                adaptive_loops=adaptive_loops,
                target_window_seconds=target_window_seconds,
                ref_blocked_retry=ref_blocked_retry,
                ref_block_rows=ref_block_rows,
                ref_isolated_retry=ref_isolated_retry,
                ref_cleanup=ref_cleanup,
                compare_device=compare_device,
            )
        except BaseException as exc:  # noqa: BLE001
            # One matrix must not sink the whole sweep: an OOM (often in the
            # reference worker, whose result then fails to load) used to abort
            # the run before any CSV was written. Record the failure and go on.
            print(
                f"[SpGEMM][{os.path.basename(path)}] FAILED "
                f"{type(exc).__name__}: {str(exc)[:200]}",
                flush=True,
            )
            # keys the CSV builder dereferences directly must all be present
            entry = {
                "path": path,
                "shape": (None, None),
                "shape_a": (None, None),
                "shape_b": (None, None),
                "nnz": None,
                "nnz_a": None,
                "nnz_b": None,
                "nnz_c": None,
                "value_dtype": str(value_dtype),
                "index_dtype": str(index_dtype),
                "status": "ERROR",
                "compare_status": f"{type(exc).__name__}: {str(exc)[:180]}",
                "input_mode": input_mode,
                "compare_device": compare_device,
            }
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        results.append(entry)
        # the row printer formats numeric fields with widths, so it cannot take
        # the placeholder entry produced for a failed matrix
        if on_result is not None and entry.get("status") != "ERROR":
            on_result(entry)
        # on_entry persists the row, so it must see the failures too -- and it
        # runs per matrix rather than per batch because a vendor-library page
        # fault aborts the process outright, taking any buffered rows with it.
        if on_entry is not None:
            on_entry(entry)
    return results


def _print_spgemm_mtx_header(value_dtype, index_dtype):
    print(
        f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}"
    )
    print("Formats: FlagSparse=CSR SpGEMM(A@B), cuSPARSE=CSR@CSR, PyTorch=sparse.mm.")
    print("Err(PT/CU)=max(|diff|/(atol+rtol*|ref|)); MaxRel=max(|diff|)/max(|ref|).")
    print("-" * 320)
    print(
        f"{'Matrix':<28} {'Mode':<10} {'A_rows':>7} {'A_cols':>7} {'B_cols':>7} {'NNZ_A':>10} {'NNZ_B':>10} {'NNZ_C':>10} "
        f"{'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>13} {'PyTorch(ms)':>11} "
        f"{'FS/CU':>7} {'FS/PT':>7} {'PT':>6} {'CU':>6} {'Status':>13} {'RefCode':>14} "
        f"{'Err(PT)':>10} {'Err(CU)':>10} {'MaxAbs(PT)':>12} {'MaxRel(PT)':>12} {'MaxAbs(CU)':>12} {'MaxRel(CU)':>12} "
        f"{'Prep(ms)':>9} {'Count(ms)':>10} {'Fill(ms)':>9}"
    )
    print("-" * 320)


def _print_spgemm_mtx_row(entry):
    name = os.path.basename(entry["path"])[:27]
    a_rows, a_cols = entry["shape_a"]
    b_cols = entry["shape_b"][1]
    print(
        f"{name:<28} {entry.get('input_mode', 'N/A'):<10} {a_rows:>7} {a_cols:>7} {b_cols:>7} "
        f"{entry['nnz_a']:>10} {entry['nnz_b']:>10} {str(entry['nnz_c'] if entry['nnz_c'] is not None else 'N/A'):>10} "
        f"{_fmt_ms(entry.get('triton_ms')):>14} {_fmt_ms(entry.get('cusparse_ms')):>13} {_fmt_ms(entry.get('pytorch_ms')):>11} "
        f"{_fmt_speedup(entry.get('cusparse_ms'), entry.get('triton_ms')):>7} {_fmt_speedup(entry.get('pytorch_ms'), entry.get('triton_ms')):>7} "
        f"{_fmt_check(entry.get('triton_ok_pt')):>6} {_fmt_check(entry.get('triton_ok_cu')):>6} {entry.get('status', 'N/A'):>13} {str(entry.get('ref_reason_code') or 'N/A'):>14} "
        f"{_fmt_err(entry.get('err_pt')):>10} {_fmt_err(entry.get('err_cu')):>10} "
        f"{_fmt_err(entry.get('max_abs_err_pt')):>12} {_fmt_err(entry.get('max_rel_err_pt')):>12} {_fmt_err(entry.get('max_abs_err_cu')):>12} {_fmt_err(entry.get('max_rel_err_cu')):>12} "
        f"{_fmt_ms(entry.get('prepare_ms')):>9} {_fmt_ms(entry.get('count_ms')):>10} {_fmt_ms(entry.get('fill_ms')):>9}"
    )
    err = entry.get("error")
    if err:
        msg = str(err).replace("\n", " ")
        if len(msg) > 320:
            msg = msg[:317] + "..."
        print(f"  NOTE: {msg}")
    pt_mode = entry.get("pt_exec_mode")
    cu_mode = entry.get("cu_exec_mode")
    pt_retry = entry.get("pt_retry_count") or 0
    cu_retry = entry.get("cu_retry_count") or 0
    if (
        pt_mode not in (None, "direct")
        or cu_mode not in (None, "direct", "disabled")
        or int(pt_retry) > 0
        or int(cu_retry) > 0
    ):
        peak_rows = entry.get("ref_peak_block_rows")
        fail_stage = entry.get("ref_fail_stage")
        print(
            f"  REF: pt_mode={pt_mode or 'N/A'} cu_mode={cu_mode or 'N/A'} "
            f"pt_retry={pt_retry} cu_retry={cu_retry} "
            f"peak_block_rows={peak_rows if peak_rows is not None else 'N/A'} "
            f"fail_stage={fail_stage or 'N/A'} "
            f"pt_attempted={entry.get('attempted_modes_pt') or 'N/A'} "
            f"cu_attempted={entry.get('attempted_modes_cu') or 'N/A'}"
        )
    if entry.get("compare_status") not in (None, "OK"):
        print(f"  COMPARE: {entry.get('compare_status')}")
    b_rows = (
        entry.get("bucket_nrows_short"),
        entry.get("bucket_nrows_medium"),
        entry.get("bucket_nrows_long"),
    )
    b_ms = (
        entry.get("bucket_ms_short"),
        entry.get("bucket_ms_medium"),
        entry.get("bucket_ms_long"),
    )
    if any(
        v is not None for v in (*b_rows, *b_ms, (entry.get("long_row_sliced_count")))
    ):
        print(
            "  PERF: "
            f"bucket_nrows(s/m/l)={b_rows[0] if b_rows[0] is not None else 'N/A'}/"
            f"{b_rows[1] if b_rows[1] is not None else 'N/A'}/"
            f"{b_rows[2] if b_rows[2] is not None else 'N/A'} "
            f"bucket_ms(s/m/l)={_fmt_ms(b_ms[0])}/{_fmt_ms(b_ms[1])}/{_fmt_ms(b_ms[2])} "
            f"long_row_sliced={entry.get('long_row_sliced_count') if entry.get('long_row_sliced_count') is not None else 'N/A'}"
        )


def print_mtx_results(results, value_dtype, index_dtype):
    _print_spgemm_mtx_header(value_dtype, index_dtype)
    for entry in results:
        _print_spgemm_mtx_row(entry)
    print("-" * 320)


SPGEMM_CSV_FIELDNAMES = [
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
        "triton_speedup_vs_pytorch",
        "pt_status",
        "cu_status",
        "status",
        "ref_reason_code",
        "err_pt",
        "err_cu",
        "max_abs_err_pt",
        "max_rel_err_pt",
        "max_abs_err_cu",
        "max_rel_err_cu",
        "pytorch_reason",
        "cusparse_reason",
        "error",
        "pt_exec_mode",
        "cu_exec_mode",
        "attempted_modes_pt",
        "attempted_modes_cu",
        "compare_status",
        "pt_retry_count",
        "cu_retry_count",
        "ref_peak_block_rows",
        "ref_fail_stage",
        "nnz_a",
        "nnz_b",
        "nnz_c",
        "input_mode",
        "shape_a",
        "shape_b",
        "prepare_ms",
        "count_ms",
        "fill_ms",
        "bucket_nrows_short",
        "bucket_nrows_medium",
        "bucket_nrows_long",
        "bucket_ms_short",
        "bucket_ms_medium",
        "bucket_ms_long",
        "long_row_sliced_count",
        "triton_started",
        "ref_started",
        "effective_warmup",
        "effective_iters",
        "compare_device",
]


def _spgemm_csv_row(entry, value_dtype, index_dtype):
    n_rows, n_cols = entry["shape"]
    return (
                {
                    "matrix": os.path.basename(entry["path"]),
                    "value_dtype": _dtype_name(value_dtype),
                    "index_dtype": _dtype_name(index_dtype),
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    "nnz": entry["nnz"],
                    "triton_ms": entry.get("triton_ms"),
                    "cusparse_ms": entry.get("cusparse_ms"),
                    "pytorch_ms": entry.get("pytorch_ms"),
                    "triton_speedup_vs_cusparse": _speedup_ratio(
                        entry.get("cusparse_ms"), entry.get("triton_ms")
                    ),
                    "triton_speedup_vs_pytorch": _speedup_ratio(
                        entry.get("pytorch_ms"), entry.get("triton_ms")
                    ),
                    "pt_status": _status_label(entry.get("triton_ok_pt")),
                    "cu_status": _status_label(entry.get("triton_ok_cu")),
                    "status": entry.get("status"),
                    "ref_reason_code": entry.get("ref_reason_code"),
                    "err_pt": entry.get("err_pt"),
                    "err_cu": entry.get("err_cu"),
                    "max_abs_err_pt": entry.get("max_abs_err_pt"),
                    "max_rel_err_pt": entry.get("max_rel_err_pt"),
                    "max_abs_err_cu": entry.get("max_abs_err_cu"),
                    "max_rel_err_cu": entry.get("max_rel_err_cu"),
                    "pytorch_reason": entry.get("pytorch_reason"),
                    "cusparse_reason": entry.get("cusparse_reason"),
                    "error": entry.get("error"),
                    "pt_exec_mode": entry.get("pt_exec_mode"),
                    "cu_exec_mode": entry.get("cu_exec_mode"),
                    "attempted_modes_pt": entry.get("attempted_modes_pt"),
                    "attempted_modes_cu": entry.get("attempted_modes_cu"),
                    "compare_status": entry.get("compare_status"),
                    "pt_retry_count": entry.get("pt_retry_count"),
                    "cu_retry_count": entry.get("cu_retry_count"),
                    "ref_peak_block_rows": entry.get("ref_peak_block_rows"),
                    "ref_fail_stage": entry.get("ref_fail_stage"),
                    "nnz_a": entry.get("nnz_a"),
                    "nnz_b": entry.get("nnz_b"),
                    "nnz_c": entry.get("nnz_c"),
                    "input_mode": entry.get("input_mode"),
                    "shape_a": str(entry.get("shape_a")),
                    "shape_b": str(entry.get("shape_b")),
                    "prepare_ms": entry.get("prepare_ms"),
                    "count_ms": entry.get("count_ms"),
                    "fill_ms": entry.get("fill_ms"),
                    "bucket_nrows_short": entry.get("bucket_nrows_short"),
                    "bucket_nrows_medium": entry.get("bucket_nrows_medium"),
                    "bucket_nrows_long": entry.get("bucket_nrows_long"),
                    "bucket_ms_short": entry.get("bucket_ms_short"),
                    "bucket_ms_medium": entry.get("bucket_ms_medium"),
                    "bucket_ms_long": entry.get("bucket_ms_long"),
                    "long_row_sliced_count": entry.get("long_row_sliced_count"),
                    "triton_started": entry.get("triton_started"),
                    "ref_started": entry.get("ref_started"),
                    "effective_warmup": entry.get("effective_warmup"),
                    "effective_iters": entry.get("effective_iters"),
                    "compare_device": entry.get("compare_device"),
                }
    )


def _write_spgemm_csv_row(writer, handle, row):
    """Append one row and push it to disk immediately.

    SpGEMM buffered every row until the sweep ended, so a vendor-library
    VMFault (rocSPARSE csrgemm on DCU) killed the process with an empty CSV and
    the matrices that had already succeeded were lost.  Flushing per matrix
    keeps those results.
    """

    writer.writerow({key: ("" if value is None else value) for key, value in row.items()})
    handle.flush()
    os.fsync(handle.fileno())


def _spgemm_crash_witness_path(csv_path):
    return f"{os.path.splitext(csv_path)[0]}.inflight.json"


def _record_spgemm_inflight(csv_path, entry):
    """Name the item currently in flight so a hard abort leaves a culprit.

    A GPU page fault raises SIGABRT, which no Python handler can intercept, so
    the record has to be on disk *before* the work starts.  It is removed once
    the sweep finishes cleanly.
    """

    try:
        with open(_spgemm_crash_witness_path(csv_path), "w", encoding="utf-8") as fh:
            json.dump(entry, fh)
            fh.flush()
            os.fsync(fh.fileno())
    except OSError:
        pass


def _clear_spgemm_inflight(csv_path):
    try:
        os.remove(_spgemm_crash_witness_path(csv_path))
    except OSError:
        pass


def run_all_dtypes_export_csv(
    paths,
    csv_path,
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    input_mode=DEFAULT_INPUT_MODE,
    adaptive_loops=False,
    target_window_seconds=TARGET_TIMED_WINDOW_SECONDS,
    ref_blocked_retry=True,
    ref_block_rows=DEFAULT_REF_BLOCK_ROWS,
    ref_isolated_retry=True,
    ref_cleanup=True,
    compare_device=DEFAULT_COMPARE_DEVICE,
):
    csv_path = _normalize_csv_path(csv_path)
    written = 0
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SPGEMM_CSV_FIELDNAMES, extrasaction="ignore"
        )
        writer.writeheader()
        handle.flush()
        for value_dtype in CSV_VALUE_DTYPES:
            for index_dtype in CSV_INDEX_DTYPES:
                print("=" * 180)
                _print_spgemm_mtx_header(value_dtype, index_dtype)

                def _emit(entry, value_dtype=value_dtype, index_dtype=index_dtype):
                    nonlocal written
                    _write_spgemm_csv_row(
                        writer,
                        handle,
                        _spgemm_csv_row(entry, value_dtype, index_dtype),
                    )
                    written += 1
                    _record_spgemm_inflight(
                        csv_path,
                        {
                            "stage": "between-matrices",
                            "last_completed": os.path.basename(entry.get("path") or ""),
                            "value_dtype": _dtype_name(value_dtype),
                            "index_dtype": _dtype_name(index_dtype),
                        },
                    )

                _record_spgemm_inflight(
                    csv_path,
                    {
                        "stage": "batch-start",
                        "value_dtype": _dtype_name(value_dtype),
                        "index_dtype": _dtype_name(index_dtype),
                    },
                )
                run_mtx_batch(
                    paths,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=warmup,
                    iters=iters,
                    run_cusparse=run_cusparse,
                    input_mode=input_mode,
                    adaptive_loops=adaptive_loops,
                    target_window_seconds=target_window_seconds,
                    ref_blocked_retry=ref_blocked_retry,
                    ref_block_rows=ref_block_rows,
                    ref_isolated_retry=ref_isolated_retry,
                    ref_cleanup=ref_cleanup,
                    compare_device=compare_device,
                    on_result=_print_spgemm_mtx_row,
                    on_entry=_emit,

                )
                print("-" * 320)
    _clear_spgemm_inflight(csv_path)
    print(f"Wrote {written} rows to {csv_path}")


def run_api_validation_checks():
    if not torch.cuda.is_available():
        print("API checks skipped: CUDA is not available.")
        return 0
    device = torch.device("cuda")
    a_data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    a_indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    a_indptr = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    shape = (2, 2)
    c_data, c_indices, c_indptr, _ = ast.flagsparse_spgemm_csr(
        a_data,
        a_indices,
        a_indptr,
        shape,
        a_data,
        a_indices,
        a_indptr,
        shape,
    )
    negative_cases = [
        (
            "shape mismatch",
            lambda: ast.flagsparse_spgemm_csr(
                a_data,
                a_indices,
                a_indptr,
                (2, 3),
                a_data,
                a_indices,
                a_indptr,
                shape,
            ),
            ValueError,
        ),
        (
            "dtype mismatch",
            lambda: ast.flagsparse_spgemm_csr(
                a_data,
                a_indices,
                a_indptr,
                shape,
                a_data.to(torch.float64),
                a_indices,
                a_indptr,
                shape,
            ),
            TypeError,
        ),
        (
            "indices dtype must int32",
            lambda: ast.flagsparse_spgemm_csr(
                a_data,
                a_indices.to(torch.int64),
                a_indptr,
                shape,
                a_data,
                a_indices,
                a_indptr,
                shape,
            ),
            TypeError,
        ),
        (
            "out data must be CUDA",
            lambda: ast.flagsparse_spgemm_csr(
                a_data,
                a_indices,
                a_indptr,
                shape,
                a_data,
                a_indices,
                a_indptr,
                shape,
                out=(
                    torch.empty(c_data.shape, dtype=c_data.dtype),
                    torch.empty(c_indices.shape, dtype=c_indices.dtype, device=device),
                    torch.empty(c_indptr.shape, dtype=c_indptr.dtype, device=device),
                ),
            ),
            ValueError,
        ),
    ]
    failed = 0
    print("-" * 96)
    print("API validation checks (SpGEMM)")
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
        out = ast.flagsparse_spgemm_csr(
            a_data,
            a_indices,
            a_indptr,
            shape,
            a_data,
            a_indices,
            a_indptr,
            shape,
        )
        if len(out) != 4:
            raise AssertionError("unexpected result tuple length")
        print("PASS  positive path returned CSR tuple")
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


def _csr_to_cpu_payload(csr_tuple):
    data, indices, indptr, shape = csr_tuple
    return (
        data.detach().to("cpu"),
        indices.detach().to("cpu"),
        indptr.detach().to("cpu"),
        (int(shape[0]), int(shape[1])),
    )


def _csr_to_device_payload(csr_tuple, device):
    data, indices, indptr, shape = csr_tuple
    return (
        data.to(device),
        indices.to(device),
        indptr.to(device),
        (int(shape[0]), int(shape[1])),
    )


def _convert_result_for_compare(csr_tuple, compare_device, device=None):
    if csr_tuple is None:
        return None
    if compare_device == "cpu":
        if csr_tuple[0].device.type == "cpu":
            return csr_tuple
        return _csr_to_cpu_payload(csr_tuple)
    if compare_device == "gpu":
        if csr_tuple[0].device.type == "cuda":
            return csr_tuple
        if device is None:
            raise ValueError(
                "device is required when converting CPU CSR payload back to CUDA"
            )
        return _csr_to_device_payload(csr_tuple, device)
    raise ValueError(f"unsupported compare_device: {compare_device}")


def _run_reference_worker(args):
    if not torch.cuda.is_available():
        payload = {
            "success": False,
            "reason": "CUDA is not available in worker",
            "fail_stage": "direct",
            "exec_mode": "direct",
        }
        torch.save(payload, args._worker_output)
        return 1

    value_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device("cuda")
    a_data, a_indices, a_indptr, a_shape = load_mtx_to_csr_torch(
        args._worker_mtx, dtype=value_dtype, device=device
    )
    a_indices = a_indices.to(torch.int32)
    resolved_mode = _resolve_input_mode(args._worker_input_mode, a_shape)
    b_data, b_indices, b_indptr, b_shape = _build_spgemm_rhs(
        a_data, a_indices, a_indptr, a_shape, resolved_mode
    )
    b_indices = b_indices.to(torch.int32)
    ref_state = _run_reference_with_retries(
        backend=args._ref_worker,
        a_data=a_data,
        a_indices=a_indices,
        a_indptr=a_indptr,
        a_shape=a_shape,
        b_data=b_data,
        b_indices=b_indices,
        b_indptr=b_indptr,
        b_shape=b_shape,
        warmup=max(0, int(args.warmup)),
        iters=max(1, int(args.iters)),
        blocked_retry=not args._worker_no_blocked,
        block_rows=max(0, int(args._worker_block_rows)),
        isolated_retry=False,
        ref_cleanup=not args._worker_no_cleanup,
        mtx_path=args._worker_mtx,
        value_dtype=value_dtype,
        input_mode=resolved_mode,
    )
    payload = {
        "success": bool(ref_state.get("success")),
        "reason": ref_state.get("reason"),
        "fail_stage": ref_state.get("fail_stage"),
        "exec_mode": ref_state.get("exec_mode"),
        "format": ref_state.get("format"),
        "ms": ref_state.get("ms"),
        "retry_count": ref_state.get("retry_count", 0),
        "peak_block_rows": ref_state.get("peak_block_rows"),
        "result": None,
    }
    if ref_state.get("success") and ref_state.get("result") is not None:
        payload["result"] = _csr_to_cpu_payload(ref_state["result"])
    torch.save(payload, args._worker_output)
    return 0 if payload["success"] else 1


def main():
    parser = argparse.ArgumentParser(description="FlagSparse SpGEMM CSR tests")
    parser.add_argument("mtx", nargs="*", help=".mtx files or directories")
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=["float32", "float64"]
    )
    parser.add_argument("--index-dtype", type=str, default="int32", choices=["int32"])
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument(
        "--adaptive-loops",
        action="store_true",
        help="enable adaptive effective_warmup/effective_iters based on first-call runtime",
    )
    parser.add_argument(
        "--target-window-seconds",
        type=float,
        default=TARGET_TIMED_WINDOW_SECONDS,
        help="adaptive target runtime window per matrix (only used with --adaptive-loops)",
    )
    parser.add_argument("--no-cusparse", action="store_true")
    parser.add_argument(
        "--ref-blocked-retry",
        dest="ref_blocked_retry",
        action="store_true",
        default=True,
        help="enable blocked retry for torch/cupy references when direct call hits resource/OOM",
    )
    parser.add_argument(
        "--no-ref-blocked-retry",
        dest="ref_blocked_retry",
        action="store_false",
        help="disable blocked retry for references",
    )
    parser.add_argument(
        "--ref-block-rows",
        type=str,
        default="auto",
        help="row block size for blocked reference retry, integer or 'auto'",
    )
    parser.add_argument(
        "--ref-isolated-retry",
        dest="ref_isolated_retry",
        action="store_true",
        default=True,
        help="enable isolated subprocess retry for failed references",
    )
    parser.add_argument(
        "--no-ref-isolated-retry",
        dest="ref_isolated_retry",
        action="store_false",
        help="disable isolated subprocess retry for failed references",
    )
    parser.add_argument(
        "--ref-cleanup",
        dest="ref_cleanup",
        action="store_true",
        default=True,
        help="enable allocator cleanup between reference attempts",
    )
    parser.add_argument(
        "--no-ref-cleanup",
        dest="ref_cleanup",
        action="store_false",
        help="disable allocator cleanup between reference attempts",
    )
    parser.add_argument("--csv", type=str, default=None, metavar="FILE")
    parser.add_argument(
        "--run-api-checks",
        action="store_true",
        help="run API validation checks before matrix benchmark (disabled by default)",
    )
    parser.add_argument(
        "--skip-api-checks", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--input-mode",
        type=str,
        default=DEFAULT_INPUT_MODE,
        choices=["auto", "a_equals_b", "a_at"],
        help="extra option: auto(square->A@A, rectangular->A@A^T) to avoid shape mismatch on non-square matrices",
    )
    parser.add_argument(
        "--compare-device",
        type=str,
        default=DEFAULT_COMPARE_DEVICE,
        choices=["cpu", "gpu"],
        help="where to compare Triton/reference results; cpu mode offloads each result before compare",
    )
    parser.add_argument(
        "--_ref-worker",
        type=str,
        choices=["torch", "cupy"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--_worker-mtx", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--_worker-output", type=str, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_worker-block-rows", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_worker-input-mode",
        type=str,
        default=DEFAULT_INPUT_MODE,
        choices=["auto", "a_equals_b", "a_at"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--_worker-no-blocked", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--_worker-no-cleanup", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args._ref_worker is not None:
        if not args._worker_mtx or not args._worker_output:
            raise SystemExit("worker mode requires --_worker-mtx and --_worker-output")
        raise SystemExit(_run_reference_worker(args))

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    if args.run_api_checks and not args.skip_api_checks:
        failed = run_api_validation_checks()
        if failed > 0:
            raise SystemExit(1)

    value_dtype = torch.float32 if args.dtype == "float32" else torch.float64
    index_dtype = torch.int32
    ref_block_rows = _parse_ref_block_rows(args.ref_block_rows)
    paths = _expand_mtx_paths(args.mtx)
    if not paths and not args.csv:
        print(
            "No .mtx files given. Use: python test_spgemm.py <file.mtx> [file2.mtx ...] or <dir/>"
        )
        print(
            "Or run all dtypes and export CSV: python test_spgemm.py <dir/> --csv results.csv"
        )
        return

    if args.csv is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        csv_path = _normalize_csv_path(args.csv)
        print("=" * 120)
        print("FLAGSPARSE SpGEMM - f32/f64 with int32, export to CSV")
        print("=" * 120)
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  "
            f"input_mode: {args.input_mode}  |  adaptive_loops: {args.adaptive_loops}  |  "
            f"ref_blocked_retry: {args.ref_blocked_retry}  |  ref_isolated_retry: {args.ref_isolated_retry}  |  "
            f"ref_block_rows: {args.ref_block_rows}  |  compare_device: {args.compare_device}  |  CSV: {csv_path}"
        )
        run_all_dtypes_export_csv(
            paths,
            csv_path,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            input_mode=args.input_mode,
            adaptive_loops=args.adaptive_loops,
            target_window_seconds=args.target_window_seconds,
            ref_blocked_retry=args.ref_blocked_retry,
            ref_block_rows=ref_block_rows,
            ref_isolated_retry=args.ref_isolated_retry,
            ref_cleanup=args.ref_cleanup,
            compare_device=args.compare_device,
        )
        return

    print("=" * 160)
    print("FLAGSPARSE SpGEMM - SuiteSparse .mtx batch (CSR)")
    print("=" * 160)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(
        f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  warmup: {args.warmup}  "
        f"iters: {args.iters}  adaptive_loops: {args.adaptive_loops}  input_mode: {args.input_mode}  "
        f"ref_blocked_retry: {args.ref_blocked_retry}  ref_isolated_retry: {args.ref_isolated_retry}  "
        f"ref_block_rows: {args.ref_block_rows}  compare_device: {args.compare_device}"
    )
    print()
    results = run_mtx_batch(
        paths,
        value_dtype=value_dtype,
        index_dtype=index_dtype,
        warmup=args.warmup,
        iters=args.iters,
        run_cusparse=not args.no_cusparse,
        input_mode=args.input_mode,
        adaptive_loops=args.adaptive_loops,
        target_window_seconds=args.target_window_seconds,
        ref_blocked_retry=args.ref_blocked_retry,
        ref_block_rows=ref_block_rows,
        ref_isolated_retry=args.ref_isolated_retry,
        ref_cleanup=args.ref_cleanup,
        compare_device=args.compare_device,
    )
    print_mtx_results(results, value_dtype, index_dtype)


if __name__ == "__main__":
    main()
