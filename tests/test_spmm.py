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
SpMM tests: load SuiteSparse .mtx, batch run, output error and performance.
Supports: multi .mtx files, value_dtype / index_dtype, CSV export, synthetic cases,
API validation checks, and PyTorch / CuPy comparison baselines.

This test module targets the current FlagSparse CSR SpMM base implementation, which is
a Triton-native CSR path for CSR + dense with non/trans/conj operation modes.
It borrows part of the AlphaSparse CSR ALG1 dense-N heuristic, but the dedicated
experimental structural port lives in alpha_spmm_alg1.py.
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
import flagsparse.sparse_operations.spmm_csr as ast_ops

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
SPMM_OPS = ["non", "trans", "conj"]
TEST_CASES = [
    (512, 512, 4096, 16),
    (1024, 1024, 16384, 32),
    (2048, 2048, 65536, 64),
    (4096, 4096, 131072, 64),
]
ALG1_TILE_CASES = [
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
DEFAULT_BLOCK_NNZ = None
DEFAULT_MAX_SEGMENTS = None
LONG_ROW_NNZ = 1536
LONG_ROW_SHAPE = (2, 2048)
LONG_ROW_DENSE_COLS = 48


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
    tokens = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--ops must not be empty")
    invalid = [token for token in tokens if token not in SPMM_OPS]
    if invalid:
        raise ValueError(
            f"unsupported --ops: {', '.join(invalid)}; allowed: {', '.join(SPMM_OPS)}"
        )
    return tokens


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


def load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    """Load SuiteSparse / Matrix Market .mtx file into CSR as torch tensors.

    Backed by the C-accelerated scipy reader (see tests/mtx_fast.py); the former
    pure-Python parser took minutes on large SuiteSparse matrices.
    """
    from mtx_fast import load_csr

    return load_csr(file_path, dtype=dtype, device=device)


def _apply_torch_sparse_op(sparse_matrix, B, op):
    if op == "non":
        return torch.sparse.mm(sparse_matrix, B)
    if op == "trans":
        return torch.sparse.mm(sparse_matrix.transpose(0, 1), B)
    if op == "conj":
        mat = sparse_matrix.conj() if torch.is_complex(sparse_matrix) else sparse_matrix
        return torch.sparse.mm(mat.transpose(0, 1), B)
    raise ValueError(f"unsupported op: {op}")


def _build_pytorch_reference(data, indices, indptr, shape, B, op="non"):
    op = ast_ops._spmm_op_to_name(op)
    device = data.device
    n_rows = shape[0]
    indptr64 = indptr.to(torch.int64)
    indices64 = indices.to(torch.int64)
    row_ind = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )

    try:
        csr_pt = torch.sparse_csr_tensor(
            indptr64, indices64, data, size=shape, device=device
        )
        timing_op = lambda: _apply_torch_sparse_op(csr_pt, B, op)
        if data.dtype in (torch.float16, torch.bfloat16):
            csr_ref = torch.sparse_csr_tensor(
                indptr64, indices64, data.to(torch.float32), size=shape, device=device
            )
            ref = _apply_torch_sparse_op(csr_ref, B.to(torch.float32), op).to(
                data.dtype
            )
        elif data.dtype == torch.float32:
            csr_ref = torch.sparse_csr_tensor(
                indptr64, indices64, data.to(torch.float64), size=shape, device=device
            )
            ref = _apply_torch_sparse_op(csr_ref, B.to(torch.float64), op).to(
                data.dtype
            )
        elif data.dtype == torch.complex64:
            csr_ref = torch.sparse_csr_tensor(
                indptr64,
                indices64,
                data.to(torch.complex128),
                size=shape,
                device=device,
            )
            ref = _apply_torch_sparse_op(csr_ref, B.to(torch.complex128), op).to(
                data.dtype
            )
        else:
            ref = _apply_torch_sparse_op(csr_pt, B, op)
        return ref, timing_op, "CSR"
    except Exception:
        coo = torch.sparse_coo_tensor(
            torch.stack([row_ind, indices64]),
            data,
            shape,
            device=device,
        ).coalesce()
        timing_op = lambda: _apply_torch_sparse_op(coo, B, op)
        if data.dtype in (torch.float16, torch.bfloat16):
            ref = _apply_torch_sparse_op(
                coo.to(torch.float32), B.to(torch.float32), op
            ).to(data.dtype)
        elif data.dtype == torch.float32:
            ref = _apply_torch_sparse_op(
                coo.to(torch.float64), B.to(torch.float64), op
            ).to(data.dtype)
        elif data.dtype == torch.complex64:
            ref = _apply_torch_sparse_op(
                coo.to(torch.complex128), B.to(torch.complex128), op
            ).to(data.dtype)
        else:
            ref = _apply_torch_sparse_op(coo, B, op)
        return ref, timing_op, "COO"


def _benchmark_triton_spmm(
    data,
    indices,
    indptr,
    B,
    shape,
    warmup,
    iters,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    op="non",
):
    op = ast_ops._spmm_op_to_name(op)
    kwargs = {
        "data": data,
        "indices": indices,
        "indptr": indptr,
        "B": B,
        "shape": shape,
        "block_n": block_n,
        "block_nnz": block_nnz,
        "max_segments": max_segments,
        "op": op,
    }
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = ast.flagsparse_spmm_csr(**kwargs)
    torch.cuda.synchronize()
    first_call_ms = (time.perf_counter() - t0) * 1000.0
    result, steady_ms = ast_ops._benchmark_cuda_op(
        lambda: ast.flagsparse_spmm_csr(**kwargs),
        warmup=warmup,
        iters=iters,
    )
    return result, steady_ms, first_call_ms


def _assert_spmm_matches_reference(
    data,
    indices,
    indptr,
    B,
    shape,
    value_dtype,
    block_n=None,
    block_nnz=None,
    max_segments=None,
    out=None,
    op="non",
):
    op = ast_ops._spmm_op_to_name(op)
    result = ast.flagsparse_spmm_csr(
        data,
        indices,
        indptr,
        B,
        shape,
        block_n=block_n,
        block_nnz=block_nnz,
        max_segments=max_segments,
        out=out,
        op=op,
    )
    ref_C, _, _ = _build_pytorch_reference(data, indices, indptr, shape, B, op=op)
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
            "flagsparse_spmm_csr did not return the provided out tensor"
        )
    return result, ref_C


def _build_long_row_case(
    value_dtype, index_dtype, device, n_dense_cols=LONG_ROW_DENSE_COLS
):
    n_rows, n_cols = LONG_ROW_SHAPE
    row0_cols = torch.arange(LONG_ROW_NNZ, dtype=torch.int64, device=device)
    row1_cols = torch.tensor([7, 129, 511, 1024], dtype=torch.int64, device=device)
    indices = torch.cat([row0_cols, row1_cols]).to(index_dtype)
    data = _build_values(indices.numel(), value_dtype, device)
    indptr = torch.tensor(
        [0, LONG_ROW_NNZ, LONG_ROW_NNZ + row1_cols.numel()],
        dtype=torch.int64,
        device=device,
    )
    B = _build_dense_matrix(n_cols, n_dense_cols, value_dtype, device)
    return data, indices, indptr, B, (n_rows, n_cols)


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
    max_segments=DEFAULT_MAX_SEGMENTS,
    op="non",
):
    """Run SpMM on one .mtx and compare against PyTorch/CuPy baselines."""
    op = ast_ops._spmm_op_to_name(op)
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(
        mtx_path, dtype=value_dtype, device=device
    )
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz = data.numel()
    b_rows = n_rows if op in ("trans", "conj") else n_cols
    B = _build_dense_matrix(b_rows, n_dense_cols, value_dtype, device)
    atol, rtol = _tolerance_for_dtype(value_dtype)

    result = {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz,
        "dense_cols": n_dense_cols,
        "op": op,
        "error": None,
        "triton_ms": None,
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
    }

    try:
        ref_C, pytorch_op, pytorch_format = _build_pytorch_reference(
            data, indices, indptr, shape, B, op=op
        )
        result["pytorch_format"] = pytorch_format
    except Exception as exc:
        result["error"] = f"ref: {exc}"
        result["status"] = "REF_FAIL"
        return result

    triton_C = None
    try:
        triton_C, triton_ms, triton_first_call_ms = _benchmark_triton_spmm(
            data,
            indices,
            indptr,
            B,
            shape,
            warmup=warmup,
            iters=iters,
            block_n=block_n,
            block_nnz=block_nnz,
            max_segments=max_segments,
            op=op,
        )
        result["triton_ms"] = triton_ms
        result["triton_first_call_ms"] = triton_first_call_ms
    except Exception as exc:
        # Do not return: still time PyTorch / CuPy so CSV shows baseline ms when Triton fails.
        result["error"] = f"triton: {exc}"
        result["triton_ok_pt"] = False

    if triton_C is not None:
        triton_metrics = ast_ops._spmm_validation_metrics(triton_C, ref_C)
        result["triton_abs_err"] = triton_metrics["max_abs_error"]
        result["triton_relative_error_diag"] = triton_metrics["max_relative_error"]
        result["err_pt"] = _scaled_allclose_error(triton_C, ref_C, value_dtype)
        result["triton_ok_pt"] = torch.allclose(triton_C, ref_C, atol=atol, rtol=rtol)
    else:
        result["triton_ok_pt"] = False

    try:
        _, result["pytorch_ms"] = ast_ops._benchmark_cuda_op(
            pytorch_op,
            warmup=warmup,
            iters=iters,
        )
    except Exception as exc:
        result["pytorch_reason"] = str(exc)

    _cupy_supported_dtypes = (
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    )
    sparse_ref_backend, sparse_ref_reason = ast_ops._spmm_csr_sparse_ref_backend(
        value_dtype, indices.dtype, indptr.dtype
    )
    if run_cusparse and sparse_ref_backend == "hipsparse":
        # DCU/ROCm: hipSPARSE replaces cuSPARSE as the vendor baseline. Its SpMM
        # entry point is non-transpose only, so trans/conj are reported as skipped.
        if op != "non":
            result["cusparse_reason"] = (
                "hipSPARSE CSR SpMM reference covers op=non only; trans/conj skipped"
            )
        else:
            try:
                cs_C_t, result["cusparse_ms"] = ast_ops._benchmark_prepared_cuda_op(
                    lambda: ast_ops._prepare_spmm_csr_ref_hipsparse(
                        data, indices, indptr, B, shape
                    ),
                    ast_ops._run_spmm_csr_ref_hipsparse_prepared,
                    ast_ops._destroy_spmm_csr_ref_hipsparse_prepared,
                    warmup=warmup,
                    iters=iters,
                )
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

                data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
                ind_cp = cp.from_dlpack(
                    torch.utils.dlpack.to_dlpack(indices.to(torch.int64))
                )
                ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
                B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B))
                A_csr = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
                A_eff = A_csr
                if op == "trans":
                    A_eff = A_csr.transpose().tocsr()
                elif op == "conj":
                    A_eff = A_csr.transpose().conj().tocsr()

                torch.cuda.synchronize()
                for _ in range(warmup):
                    _ = A_eff @ B_cp
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    _ = A_eff @ B_cp
                end.record()
                torch.cuda.synchronize()
                result["cusparse_ms"] = start.elapsed_time(end) / iters

                cs_C = A_eff @ B_cp
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

    result["status"] = (
        "PASS" if (result["triton_ok_pt"] or result["triton_ok_cu"]) else "FAIL"
    )
    return result


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    n_dense_cols=32,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    max_segments=DEFAULT_MAX_SEGMENTS,
    op="non",
    on_result=None,
):
    results = []
    for path in mtx_paths:
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
            max_segments=max_segments,
            op=op,
        )
        results.append(entry)
        if on_result is not None:
            on_result(entry)
    return results


def _print_spmm_csr_mtx_header(value_dtype, index_dtype, op="non"):
    print(
        f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}  |  op: {op}"
    )
    print(
        "Formats: FlagSparse=CSR base (ALG1-inspired heuristic), cuSPARSE=CSR dense-mm, PyTorch=CSR or COO."
    )
    print(
        "Timing stays in native dtype. For float32, correctness references use float64 compute then cast."
    )
    print(
        "PT/CU show per-reference correctness. Err(PT)/Err(CU)=max(|diff| / (atol + rtol*|ref|))."
    )
    print(
        "For float32, PT checks the float64-based correctness reference while CU checks consistency with native cuSPARSE float32, so PT and CU may differ."
    )
    print("-" * 186)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} "
        f"{'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>13} {'PyTorch(ms)':>11} "
        f"{'FS/CU':>7} {'FS/PT':>7} {'PT':>6} {'CU':>6} {'Err(PT)':>10} {'Err(CU)':>10}"
    )
    print("-" * 186)


def _print_spmm_csr_mtx_row(entry):
    name = os.path.basename(entry["path"])[:27]
    n_rows, n_cols = entry["shape"]
    triton_ms = entry.get("triton_ms")
    cu_ms = entry.get("cusparse_ms")
    pt_ms = entry.get("pytorch_ms")
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {entry['nnz']:>10} {entry['dense_cols']:>8} "
        f"{_fmt_ms(triton_ms):>14} {_fmt_ms(cu_ms):>13} {_fmt_ms(pt_ms):>11} "
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


def print_mtx_results(results, value_dtype, index_dtype):
    op = results[0].get("op", "non") if results else "non"
    _print_spmm_csr_mtx_header(value_dtype, index_dtype, op=op)
    for entry in results:
        _print_spmm_csr_mtx_row(entry)
    print("-" * 186)


def run_all_dtypes_export_csv(
    paths,
    csv_path,
    warmup=10,
    iters=50,
    run_cusparse=True,
    n_dense_cols=32,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    max_segments=DEFAULT_MAX_SEGMENTS,
    value_dtypes=None,
    index_dtypes=None,
    ops=None,
):
    csv_path = _normalize_csv_path(csv_path)
    rows = []
    value_dtypes = CSV_VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = CSV_INDEX_DTYPES if index_dtypes is None else index_dtypes
    ops = SPMM_OPS if ops is None else ops
    for value_dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                print("=" * 150)
                _print_spmm_csr_mtx_header(value_dtype, index_dtype, op=op)
                results = run_mtx_batch(
                    paths,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=warmup,
                    iters=iters,
                    run_cusparse=run_cusparse,
                    n_dense_cols=n_dense_cols,
                    block_n=block_n,
                    block_nnz=block_nnz,
                    max_segments=max_segments,
                    op=op,
                    on_result=_print_spmm_csr_mtx_row,
                )
                print("-" * 186)
                for entry in results:
                    n_rows, n_cols = entry["shape"]
                    rows.append(
                        {
                            "matrix": os.path.basename(entry["path"]),
                            "value_dtype": _dtype_name(value_dtype),
                            "index_dtype": _dtype_name(index_dtype),
                            "op": op,
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
                            "status": (
                                "PASS"
                                if (
                                    entry.get("triton_ok_pt")
                                    or entry.get("triton_ok_cu")
                                )
                                else "FAIL"
                            ),
                            "err_pt": entry.get("err_pt"),
                            "err_cu": entry.get("err_cu"),
                            "error": entry.get("error"),
                        }
                    )
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
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
        "err_pt",
        "err_cu",
        "error",
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
    data = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, device=device)
    indices = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)
    indptr = torch.tensor([0, 2, 3], dtype=torch.int64, device=device)
    B = torch.randn((2, 4), dtype=torch.float32, device=device)
    long_data, long_indices, long_indptr, long_B, long_shape = _build_long_row_case(
        torch.float32, torch.int32, device
    )

    negative_cases = [
        (
            "shape must be 2D",
            lambda: ast.flagsparse_spmm_csr(data, indices, indptr, B, (2,)),
            ValueError,
        ),
        (
            "B must be 2D",
            lambda: ast.flagsparse_spmm_csr(data, indices, indptr, B[0], (2, 2)),
            ValueError,
        ),
        (
            "dtype mismatch",
            lambda: ast.flagsparse_spmm_csr(
                data, indices, indptr, B.to(torch.float64), (2, 2)
            ),
            TypeError,
        ),
        (
            "shape mismatch",
            lambda: ast.flagsparse_spmm_csr(
                data,
                indices,
                indptr,
                torch.randn((3, 4), dtype=torch.float32, device=device),
                (2, 2),
            ),
            ValueError,
        ),
        (
            "indptr length mismatch",
            lambda: ast.flagsparse_spmm_csr(data, indices, indptr[:-1], B, (2, 2)),
            ValueError,
        ),
        (
            "indptr must start at 0",
            lambda: ast.flagsparse_spmm_csr(
                data,
                indices,
                torch.tensor([1, 2, 3], dtype=torch.int64, device=device),
                B,
                (2, 2),
            ),
            ValueError,
        ),
        (
            "indptr last must equal nnz",
            lambda: ast.flagsparse_spmm_csr(
                data,
                indices,
                torch.tensor([0, 2, 2], dtype=torch.int64, device=device),
                B,
                (2, 2),
            ),
            ValueError,
        ),
        (
            "indptr monotonic",
            lambda: ast.flagsparse_spmm_csr(
                data,
                indices,
                torch.tensor([0, 3, 2], dtype=torch.int64, device=device),
                B,
                (2, 2),
            ),
            ValueError,
        ),
        (
            "indices out of range",
            lambda: ast.flagsparse_spmm_csr(
                data,
                torch.tensor([0, 3, 1], dtype=torch.int32, device=device),
                indptr,
                B,
                (2, 2),
            ),
            IndexError,
        ),
        (
            "block_n positive",
            lambda: ast.flagsparse_spmm_csr(
                data, indices, indptr, B, (2, 2), block_n=0
            ),
            ValueError,
        ),
        (
            "block_nnz positive",
            lambda: ast.flagsparse_spmm_csr(
                data, indices, indptr, B, (2, 2), block_nnz=0
            ),
            ValueError,
        ),
        (
            "max_segments positive",
            lambda: ast.flagsparse_spmm_csr(
                data, indices, indptr, B, (2, 2), max_segments=0
            ),
            ValueError,
        ),
        (
            "out shape mismatch",
            lambda: ast.flagsparse_spmm_csr(
                data,
                indices,
                indptr,
                B,
                (2, 2),
                out=torch.empty((3, 4), dtype=torch.float32, device=device),
            ),
            ValueError,
        ),
        (
            "out device mismatch",
            lambda: ast.flagsparse_spmm_csr(
                data,
                indices,
                indptr,
                B,
                (2, 2),
                out=torch.empty((2, 4), dtype=torch.float32),
            ),
            ValueError,
        ),
        (
            "bad op",
            lambda: ast.flagsparse_spmm_csr(data, indices, indptr, B, (2, 2), op="bad"),
            ValueError,
        ),
        (
            "op transpose conflict",
            lambda: ast.flagsparse_spmm_csr(
                data, indices, indptr, B, (2, 2), op="non", transpose=True
            ),
            ValueError,
        ),
        (
            "transpose B mismatch",
            lambda: ast.flagsparse_spmm_csr(
                data,
                indices,
                indptr,
                torch.randn((3, 4), dtype=torch.float32, device=device),
                (2, 2),
                op="trans",
            ),
            ValueError,
        ),
        (
            "transpose out shape mismatch",
            lambda: ast.flagsparse_spmm_csr(
                data,
                indices,
                indptr,
                torch.randn((2, 4), dtype=torch.float32, device=device),
                (2, 2),
                op="trans",
                out=torch.empty((2, 3), dtype=torch.float32, device=device),
            ),
            ValueError,
        ),
        (
            "segment overflow override",
            lambda: ast.flagsparse_spmm_csr(
                long_data,
                long_indices,
                long_indptr,
                long_B,
                long_shape,
                block_nnz=128,
                max_segments=4,
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
        _assert_spmm_matches_reference(
            data, indices, indptr, B, (2, 2), torch.float32, out=out
        )

    positive_checks.append(("out path success", _positive_out_path))

    def _positive_transpose_path():
        dense = torch.randn((2, 4), dtype=torch.float32, device=device)
        _assert_spmm_matches_reference(
            data, indices, indptr, dense, (2, 2), torch.float32, op="trans"
        )

    positive_checks.append(("transpose path success", _positive_transpose_path))

    def _positive_empty_matrix():
        empty_data = torch.tensor([], dtype=torch.float32, device=device)
        empty_indices = torch.tensor([], dtype=torch.int32, device=device)
        empty_indptr = torch.zeros(3, dtype=torch.int64, device=device)
        dense = torch.randn((2, 4), dtype=torch.float32, device=device)
        result, _ = _assert_spmm_matches_reference(
            empty_data,
            empty_indices,
            empty_indptr,
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
        result, _ = _assert_spmm_matches_reference(
            data,
            indices,
            indptr,
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
        _assert_spmm_matches_reference(
            data, indices, indptr, dense, (2, 2), torch.float32
        )

    positive_checks.append(("noncontiguous B success", _positive_noncontiguous_b))

    def _positive_long_row_default():
        _assert_spmm_matches_reference(
            long_data,
            long_indices,
            long_indptr,
            long_B,
            long_shape,
            torch.float32,
        )

    positive_checks.append(("long-row default success", _positive_long_row_default))

    for name, fn in positive_checks:
        try:
            fn()
            print(f"PASS  {name:<32} returned correct result")
        except Exception as exc:
            print(f"FAIL  {name:<32} raised {type(exc).__name__}: {exc}")
            failed += 1

    print("-" * 96)
    return failed


def run_alg1_tile_branch_coverage(warmup=WARMUP, iters=ITERS, run_cusparse=True):
    if not torch.cuda.is_available():
        print("ALG1 branch coverage skipped: CUDA is not available.")
        return 0

    print("=" * 132)
    print("ALG1 dense-column heuristic coverage")
    print("=" * 132)
    print(
        f"{'DenseN':>8} {'BLOCK_N':>8} {'NNZTile':>8} {'ReqSeg':>7} {'Warp':>6} {'Factor':>7} "
        f"{'PyTorch(ms)':>12} {'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>12} {'PT':>6} {'CU':>6} {'Err(FS)':>11}"
    )
    print("-" * 132)

    failed = 0
    note = None
    for n_rows, n_cols, nnz, n_dense_cols in ALG1_TILE_CASES:
        result = ast.benchmark_spmm_case(
            n_rows=n_rows,
            n_cols=n_cols,
            nnz=nnz,
            n_dense_cols=n_dense_cols,
            value_dtype=torch.float32,
            index_dtype=torch.int32,
            warmup=warmup,
            iters=iters,
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
        status = (
            "PASS" if triton_ok and (cusparse_ok is None or cusparse_ok) else "FAIL"
        )
        if status != "PASS":
            failed += 1
        if backend.get("cusparse_unavailable_reason"):
            note = backend["cusparse_unavailable_reason"]
        triton_err = _scaled_allclose_error(
            samples["triton"], samples["reference"], torch.float32
        )
        print(
            f"{n_dense_cols:>8} {params['block_n']:>8} {params['block_nnz']:>8} {params['required_segments']:>7} "
            f"{params['alg1_warp_size']:>6} {params['alg1_factor']:>7} "
            f"{_fmt_ms(perf.get('pytorch_ms')):>12} {_fmt_ms(perf.get('triton_ms')):>14} {_fmt_ms(perf.get('cusparse_ms')):>12} "
            f"{_fmt_check(triton_ok):>6} {_fmt_check(cusparse_ok):>6} {_fmt_err(triton_err):>11}"
        )
    print("-" * 132)
    if note:
        print(f"cuSPARSE note: {note}")
    print()
    return failed


def run_comprehensive_synthetic(
    warmup=WARMUP,
    iters=ITERS,
    run_cusparse=True,
    run_api_checks=True,
    run_alg1_coverage=True,
    block_n=DEFAULT_BLOCK_N,
    block_nnz=DEFAULT_BLOCK_NNZ,
    max_segments=DEFAULT_MAX_SEGMENTS,
    ops=None,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print("=" * 144)
    print("FLAGSPARSE SpMM BENCHMARK (synthetic CSR @ dense)")
    print("=" * 144)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  |  Warmup: {warmup}  Iters: {iters}  "
        f"BLOCK_N: {_fmt_launch_value(block_n)}  BLOCK_NNZ: {_fmt_launch_value(block_nnz)}  "
        f"MAX_SEGMENTS: {_fmt_launch_value(max_segments)}"
    )
    print(
        "Formats: FlagSparse=CSR base (ALG1-inspired heuristic), cuSPARSE=CSR dense-mm (when supported), PyTorch=CSR or COO."
    )
    print(
        "For float32, PT checks the float64-based correctness reference while CU reflects native cuSPARSE float32 consistency."
    )
    print()

    total = 0
    failed = 0
    ops = SPMM_OPS if ops is None else ops
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            for op in ops:
                print("-" * 144)
                print(
                    f"Value dtype: {_dtype_name(value_dtype):<12}  |  Index dtype: {_dtype_name(index_dtype):<6}  |  op: {op}"
                )
                print("-" * 144)
                print(
                    f"{'N_rows':>7} {'N_cols':>7} {'NNZ':>10} {'DenseN':>8} {'BN':>4} {'BNNZ':>6} {'Seg':>4} "
                    f"{'PyTorch(ms)':>12} {'FlagSparse(ms)':>14} {'cuSPARSE(ms)':>12} {'FS/PT':>8} {'FS/CU':>8} {'PT':>6} {'CU':>6} {'Err(FS)':>11} {'Err(CU)':>12}"
                )
                print("-" * 144)
                combo_reason = None
                for n_rows, n_cols, nnz, n_dense_cols in TEST_CASES:
                    result = ast.benchmark_spmm_case(
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
                        max_segments=max_segments,
                        run_cusparse=run_cusparse,
                        op=op,
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
                    status = (
                        "PASS"
                        if triton_ok and (cusparse_ok is None or cusparse_ok)
                        else "FAIL"
                    )
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
                    print(
                        f"{n_rows:>7} {n_cols:>7} {nnz:>10} {n_dense_cols:>8} {params['block_n']:>4} {params['block_nnz']:>6} {params['required_segments']:>4} "
                        f"{_fmt_ms(perf.get('pytorch_ms')):>12} {_fmt_ms(perf.get('triton_ms')):>14} {_fmt_ms(perf.get('cusparse_ms')):>12} "
                        f"{_fmt_speedup(perf.get('pytorch_ms'), perf.get('triton_ms')):>8} {_fmt_speedup(perf.get('cusparse_ms'), perf.get('triton_ms')):>8} "
                        f"{_fmt_check(triton_ok):>6} {_fmt_check(cusparse_ok):>6} {_fmt_err(triton_err):>11} {_fmt_err(cusparse_err):>12}"
                    )
                print("-" * 144)
                if combo_reason:
                    print(f"  cuSPARSE: {combo_reason}")
                print()

    alg1_failed = (
        run_alg1_tile_branch_coverage(
            warmup=warmup, iters=iters, run_cusparse=run_cusparse
        )
        if run_alg1_coverage
        else 0
    )
    api_failed = run_api_validation_checks() if run_api_checks else 0
    print("=" * 144)
    print(
        f"Total synthetic cases: {total}  Failed synthetic cases: {failed}  "
        f"Failed ALG1 branch cases: {alg1_failed}  Failed API checks: {api_failed}"
    )
    print("=" * 144)


def main():
    parser = argparse.ArgumentParser(
        description="SpMM test: SuiteSparse .mtx batch run, error and performance."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
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
        "--ops",
        default="non,trans,conj",
        help="Comma-separated SpMM ops to run: non,trans,conj",
    )
    parser.add_argument(
        "--dense-cols", type=int, default=32, help="Dense RHS column count"
    )
    parser.add_argument(
        "--block-n",
        type=int,
        default=DEFAULT_BLOCK_N,
        help="Output column tile override (default: auto from ALG1 heuristic)",
    )
    parser.add_argument(
        "--block-nnz",
        type=int,
        default=DEFAULT_BLOCK_NNZ,
        help="CSR segment width override (default: auto from ALG1 heuristic)",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=DEFAULT_MAX_SEGMENTS,
        help="CSR segment count override (default: auto from matrix max row nnz)",
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
        "--skip-alg1-coverage",
        action="store_true",
        help="Skip dense-column ALG1 heuristic coverage in synthetic mode",
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
        selected_ops = _parse_ops(args.ops)
    except ValueError as exc:
        parser.error(str(exc))

    if args.synthetic:
        run_comprehensive_synthetic(
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            run_api_checks=not args.skip_api_checks,
            run_alg1_coverage=not args.skip_alg1_coverage,
            block_n=args.block_n,
            block_nnz=args.block_nnz,
            max_segments=args.max_segments,
            ops=selected_ops,
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
            "No .mtx files given. Use: python test_spmm.py <file.mtx> [file2.mtx ...] or <dir/>"
        )
        print("Or run synthetic: python test_spmm.py --synthetic")
        print(
            "Or run all dtypes and export CSV: python test_spmm.py <dir/> --csv results.csv"
        )
        return

    if args.csv is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        csv_path = _normalize_csv_path(args.csv)
        print("=" * 100)
        print("FLAGSPARSE SpMM - selected dtype/index grid, export to CSV")
        print("=" * 100)
        try:
            csv_value_dtypes = _parse_csv_tokens(args.dtypes, dtype_map, "--dtypes")
            csv_index_dtypes = _parse_csv_tokens(
                args.index_dtypes, index_map, "--index-dtypes"
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  DenseN: {args.dense_cols}  |  CSV: {csv_path}"
        )
        print(
            f"dtypes: {args.dtypes}  |  index_dtypes: {args.index_dtypes}  |  ops: {args.ops}"
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
            max_segments=args.max_segments,
            value_dtypes=csv_value_dtypes,
            index_dtypes=csv_index_dtypes,
            ops=selected_ops,
        )
        return

    for op in selected_ops:
        print("=" * 140)
        print("FLAGSPARSE SpMM - SuiteSparse .mtx batch (error + performance)")
        print("=" * 140)
        print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
        print(
            f"dtype: {args.dtype}  index_dtype: {args.index_dtype}  op: {op}  dense_cols: {args.dense_cols}  "
            f"block_n: {_fmt_launch_value(args.block_n)}  block_nnz: {_fmt_launch_value(args.block_nnz)}  "
            f"max_segments: {_fmt_launch_value(args.max_segments)}  warmup: {args.warmup}  iters: {args.iters}"
        )
        print()
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
            max_segments=args.max_segments,
            op=op,
        )
        print_mtx_results(results, value_dtype, index_dtype)


if __name__ == "__main__":
    main()
