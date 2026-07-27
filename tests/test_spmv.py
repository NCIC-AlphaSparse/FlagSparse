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
SpMV tests (CSR): load SuiteSparse .mtx, batch run, output error and performance.
Supports: multi .mtx files, value_dtype / index_dtype, ops, --csv-csr export.
"""

import argparse
import csv
import glob
import math
import os

import torch
import flagsparse as ast
import flagsparse.sparse_operations.spmv_csr as spmv_csr_mod

VALUE_DTYPES = [
    torch.float32,
    torch.float64,
]
INDEX_DTYPES = [torch.int32, torch.int64]
CSV_VALUE_DTYPE_NAMES = ("float32", "float64", "complex64", "complex128")
OP_CHOICES = ("non", "trans", "conj")
OP_TO_CODE = {"non": 0, "trans": 1, "conj": 2}
TEST_CASES = [
    (512, 512, 4096),
    (1024, 1024, 16384),
    (2048, 2048, 65536),
    (4096, 4096, 131072),
]
WARMUP = 10
ITERS = 50


def _normalize_op(op):
    token = str(op or "non").strip().lower()
    if token not in OP_TO_CODE:
        raise ValueError("op must be one of: non, trans, conj")
    return token


def _parse_ops(value, default_ops):
    if value is None:
        return list(default_ops)
    tokens = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not tokens:
        raise ValueError("--ops must not be empty")
    invalid = [token for token in tokens if token not in OP_CHOICES]
    if invalid:
        raise ValueError(
            f"unsupported --ops value: {', '.join(invalid)}; allowed: {', '.join(OP_CHOICES)}"
        )
    return tokens


def _op_transposes(op):
    return _normalize_op(op) in ("trans", "conj")


def _apply_torch_sparse_op(matrix, x_2d, op):
    op = _normalize_op(op)
    if op == "non":
        return torch.sparse.mm(matrix, x_2d).squeeze(1)
    if op == "trans":
        return torch.sparse.mm(matrix.transpose(0, 1), x_2d).squeeze(1)
    return torch.sparse.mm(matrix.conj().transpose(0, 1), x_2d).squeeze(1)


def _cupy_sparse_op_matrix(matrix, op):
    op = _normalize_op(op)
    if op == "non":
        return matrix
    if op == "trans":
        return matrix.T
    matrix_conj = matrix.conj() if hasattr(matrix, "conj") else matrix.conjugate()
    return matrix_conj.T


def _apply_cupy_sparse_op(matrix, x, op):
    return _cupy_sparse_op_matrix(matrix, op) @ x


def _random_vector(size, dtype, device):
    return torch.randn(size, dtype=dtype, device=device)


def _dtype_map():
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
        "complex64": torch.complex64,
        "complex128": torch.complex128,
    }
    return mapping


def _csv_value_dtypes(dtype_map):
    return [dtype_map[name] for name in CSV_VALUE_DTYPE_NAMES if name in dtype_map]


def load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    """
    Load SuiteSparse / Matrix Market .mtx file into CSR as torch tensors.
    Correctly handles pattern matrices and symmetric/skew-symmetric expansions.
    Returns (data, indices, indptr, shape) on device.

    Backed by the C-accelerated scipy reader (see tests/mtx_fast.py); the former
    pure-Python parser took minutes on large SuiteSparse matrices.
    """
    from mtx_fast import load_csr

    return load_csr(file_path, dtype=dtype, device=device)


def _allclose_error_ratio(actual, reference, atol, rtol):
    if actual.numel() == 0:
        return 0.0
    diff = torch.abs(actual - reference).to(torch.float64)
    tol = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / tol).item())


def _has_non_finite(tensor):
    if tensor is None or tensor.numel() == 0:
        return False
    return not bool(torch.isfinite(tensor).all().item())


def _non_finite_error_reason(actual, reference, reference_name):
    actual_non_finite = _has_non_finite(actual)
    reference_non_finite = _has_non_finite(reference)
    if actual_non_finite and reference_non_finite:
        return f"non-finite FlagSparse output and {reference_name}"
    if actual_non_finite:
        return "non-finite FlagSparse output"
    if reference_non_finite:
        return f"non-finite {reference_name}"
    return None


def _has_numeric_error(error_value):
    return error_value is not None and not math.isnan(error_value)


def _benchmark_flagsparse_spmv(
    data,
    indices,
    indptr,
    x,
    shape,
    warmup,
    iters,
    block_nnz,
    max_segments,
    op="non",
):
    op = _normalize_op(op)
    prepared = ast.prepare_spmv_csr(
        data,
        indices,
        indptr,
        shape,
        block_nnz=block_nnz,
        max_segments=max_segments,
        op=OP_TO_CODE[op],
    )
    spmv_op = lambda: _run_flagsparse_spmv_timed_op(
        prepared,
        data,
        indices,
        indptr,
        x,
        shape,
        op,
    )
    y = spmv_op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = spmv_op()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        y = spmv_op()
    end_ev.record()
    torch.cuda.synchronize()
    return y, start_ev.elapsed_time(end_ev) / iters


def _clone_csr_prepared_for_timed_op(template, data, indices, indptr):
    return spmv_csr_mod.PreparedCsrSpmv(
        data=data,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=template.shape,
        n_rows=template.n_rows,
        n_cols=template.n_cols,
        block_nnz=template.block_nnz,
        max_segments=template.max_segments,
        opt_max_segments=template.opt_max_segments,
        max_row_nnz=template.max_row_nnz,
        opt_buckets=template.opt_buckets,
        transpose=template.transpose,
        op=template.op,
        index_fallback_policy=template.index_fallback_policy,
    )


def _materialize_csr_op_for_timing(data, indices, indptr, shape, op):
    if op == "non":
        return data, indices, indptr, shape
    data_op = data
    if op == "conj" and data.is_complex():
        data_op = data.conj()
        if hasattr(data_op, "resolve_conj"):
            data_op = data_op.resolve_conj()
    return spmv_csr_mod._transpose_csr_for_spmv(data_op, indices, indptr, shape)


def _run_flagsparse_spmv_timed_op(prepared, data, indices, indptr, x, shape, op):
    if op == "non":
        return ast.flagsparse_spmv_csr(
            x=x,
            prepared=prepared,
            return_time=False,
        )
    data_op, indices_op, indptr_op, _ = _materialize_csr_op_for_timing(
        data, indices, indptr, shape, op
    )
    timed_prepared = _clone_csr_prepared_for_timed_op(
        prepared, data_op, indices_op, indptr_op
    )
    return spmv_csr_mod._run_spmv_prepared_with_fallback(
        timed_prepared, x, use_opt=False
    )


def _reference_dtype(dtype):
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _cast_reference_output(y_ref, out_dtype):
    return y_ref.to(out_dtype) if y_ref.dtype != out_dtype else y_ref


def _pytorch_spmv_reference(
    data,
    indices,
    indptr,
    x,
    shape,
    out_dtype,
    op="non",
    return_compute=False,
):
    device = data.device
    op = _normalize_op(op)
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    try:
        csr_ref = torch.sparse_csr_tensor(
            indptr.to(torch.int64),
            indices.to(torch.int64),
            data_ref,
            size=shape,
            device=device,
        )
        y_ref = _apply_torch_sparse_op(csr_ref, x_ref.unsqueeze(1), op)
    except Exception:
        n_rows = int(shape[0])
        row_ind = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        coo_ref = torch.sparse_coo_tensor(
            torch.stack([row_ind, indices.to(torch.int64)]),
            data_ref,
            shape,
            device=device,
        ).coalesce()
        y_ref = _apply_torch_sparse_op(coo_ref, x_ref.unsqueeze(1), op)
    if return_compute:
        return y_ref
    return _cast_reference_output(y_ref, out_dtype)


def _build_pytorch_sparse_matrix(data, indices, indptr, shape):
    device = data.device
    try:
        return torch.sparse_csr_tensor(
            indptr.to(torch.int64),
            indices.to(torch.int64),
            data,
            size=shape,
            device=device,
        )
    except Exception:
        n_rows = int(shape[0])
        row_ind = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        return torch.sparse_coo_tensor(
            torch.stack([row_ind, indices.to(torch.int64)]),
            data,
            shape,
            device=device,
        ).coalesce()


def _run_pytorch_spmv_runtime_op(data, indices, indptr, x_2d, shape, op):
    data_op, indices_op, indptr_op, shape_op = _materialize_csr_op_for_timing(
        data, indices, indptr, shape, op
    )
    matrix = _build_pytorch_sparse_matrix(data_op, indices_op, indptr_op, shape_op)
    return torch.sparse.mm(matrix, x_2d).squeeze(1)


def _time_pytorch_spmv(data, indices, indptr, x, shape, warmup, iters, op="non"):
    op = _normalize_op(op)
    if data.numel() == 0:
        return 0.0
    x_2d = x.unsqueeze(1)
    if op == "non":
        matrix = _build_pytorch_sparse_matrix(data, indices, indptr, shape)
        spmv_op = lambda: torch.sparse.mm(matrix, x_2d).squeeze(1)
    else:
        spmv_op = lambda: _run_pytorch_spmv_runtime_op(
            data, indices, indptr, x_2d, shape, op
        )
    for _ in range(warmup):
        _ = spmv_op()
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        _ = spmv_op()
    end_ev.record()
    torch.cuda.synchronize()
    return start_ev.elapsed_time(end_ev) / iters


def _cupy_csr_reference(data, indices, indptr, x, shape, out_dtype, op="non"):
    import cupy as cp
    import cupyx.scipy.sparse as cpx

    op = _normalize_op(op)
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_ref))
    A_csr_ref = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
    y_ref = _apply_cupy_sparse_op(A_csr_ref, x_cp, op)
    y_ref_t = torch.utils.dlpack.from_dlpack(y_ref.toDlpack())
    return _cast_reference_output(y_ref_t, out_dtype)


def _run_cupy_spmv_runtime_op(data, indices, indptr, x, shape, op, fmt="csr"):
    import cupyx.scipy.sparse as cpx

    matrix = cpx.csr_matrix((data, indices, indptr), shape=shape)
    if fmt == "csc":
        matrix = matrix.tocsc()
    return _apply_cupy_sparse_op(matrix, x, op)


def _time_cupy_spmv(
    data, indices, indptr, x, shape, warmup, iters, op="non", fmt="csr"
):
    import cupy as cp
    import cupyx.scipy.sparse as cpx

    op = _normalize_op(op)
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    ind_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indices.to(torch.int64)))
    ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    if op == "non":
        matrix = cpx.csr_matrix((data_cp, ind_cp, ptr_cp), shape=shape)
        if fmt == "csc":
            matrix = matrix.tocsc()
        spmv_op = lambda: matrix @ x_cp
    else:
        spmv_op = lambda: _run_cupy_spmv_runtime_op(
            data_cp, ind_cp, ptr_cp, x_cp, shape, op, fmt=fmt
        )
    for _ in range(warmup):
        _ = spmv_op()
    cp.cuda.runtime.deviceSynchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = spmv_op()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _tolerance(value_dtype):
    if value_dtype == torch.float16:
        return 1e-3, 2e-3
    if value_dtype == torch.bfloat16:
        return 0.016, 1e-1
    if value_dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if value_dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    return 1e-6, 1e-5


def run_one_mtx(
    mtx_path,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    block_nnz=256,
    max_segments=None,
    op="non",
):
    """Run SpMV on one .mtx and return errors/timings."""
    op = _normalize_op(op)
    device = torch.device("cuda")
    data, indices, indptr, shape = load_mtx_to_csr_torch(
        mtx_path, dtype=value_dtype, device=device
    )
    indices = indices.to(index_dtype)
    n_rows, n_cols = shape
    nnz = data.numel()
    transpose = _op_transposes(op)
    x_size = n_rows if transpose else n_cols
    y_size = n_cols if transpose else n_rows
    x = _random_vector(x_size, value_dtype, device)
    atol, rtol = _tolerance(value_dtype)

    triton_y, triton_ms = _benchmark_flagsparse_spmv(
        data,
        indices,
        indptr,
        x,
        shape,
        warmup=warmup,
        iters=iters,
        block_nnz=block_nnz,
        max_segments=max_segments,
        op=op,
    )

    pt_ref_y = None
    pytorch_ms = None
    err_pt = None
    triton_ok_pt = False
    pt_error_reason = None
    try:
        pt_ref_y = _pytorch_spmv_reference(
            data,
            indices,
            indptr,
            x,
            shape,
            value_dtype,
            op=op,
        )
        pytorch_ms = _time_pytorch_spmv(
            data,
            indices,
            indptr,
            x,
            shape,
            warmup,
            iters,
            op=op,
        )
        if y_size:
            pt_error_reason = _non_finite_error_reason(
                triton_y, pt_ref_y, "PyTorch reference"
            )
            if pt_error_reason is None:
                err_pt = _allclose_error_ratio(triton_y, pt_ref_y, atol, rtol)
                triton_ok_pt = (not math.isnan(err_pt)) and err_pt <= 1.0
            else:
                err_pt = float("nan")
    except Exception as exc:
        pytorch_ms = None
        pt_error_reason = f"PyTorch reference error: {exc}"

    cusparse_ms = None
    err_cu = None
    triton_ok_cu = False
    cu_error_reason = None
    csc_ms = None
    if run_cusparse and value_dtype not in (torch.bfloat16,):
        try:
            cusparse_ms = _time_cupy_spmv(
                data,
                indices,
                indptr,
                x,
                shape,
                warmup,
                iters,
                op=op,
                fmt="csr",
            )
            cs_ref_t = _cupy_csr_reference(
                data, indices, indptr, x, shape, value_dtype, op=op
            )
            if y_size:
                cu_error_reason = _non_finite_error_reason(
                    triton_y, cs_ref_t, "cuSPARSE reference"
                )
                if cu_error_reason is None:
                    err_cu = _allclose_error_ratio(triton_y, cs_ref_t, atol, rtol)
                    triton_ok_cu = (not math.isnan(err_cu)) and err_cu <= 1.0
                else:
                    err_cu = float("nan")

            csc_ms = _time_cupy_spmv(
                data,
                indices,
                indptr,
                x,
                shape,
                warmup,
                iters,
                op=op,
                fmt="csc",
            )
        except Exception as exc:
            cusparse_ms = None
            err_cu = None
            csc_ms = None
            cu_error_reason = f"cuSPARSE reference error: {exc}"

    if pt_ref_y is None and err_cu is None:
        triton_non_finite = _has_non_finite(triton_y)
        error_reason = (
            "non-finite FlagSparse output"
            if triton_non_finite
            else (
                pt_error_reason
                or cu_error_reason
                or "ref: no PyTorch or cuSPARSE result"
            )
        )
        return {
            "path": mtx_path,
            "shape": shape,
            "nnz": nnz,
            "error": error_reason,
            "error_reason": error_reason,
            "op": op,
            "transpose": transpose,
            "triton_ms": triton_ms,
            "cusparse_ms": None,
            "pytorch_ms": None,
            "csc_ms": None,
            "err_pt": None,
            "err_cu": None,
            "triton_ok_pt": False,
            "triton_ok_cu": False,
            "status": "FAIL" if triton_non_finite else "REF_FAIL",
        }
    if _has_non_finite(triton_y):
        status = "FAIL"
        if pt_error_reason and pt_error_reason.startswith(
            "non-finite FlagSparse output"
        ):
            error_reason = pt_error_reason
        else:
            error_reason = "non-finite FlagSparse output"
    elif triton_ok_pt or triton_ok_cu:
        status = "PASS"
        error_reason = None
    elif pt_error_reason == "non-finite PyTorch reference" and err_cu is None:
        status = "REF_FAIL"
        error_reason = pt_error_reason
    else:
        status = "FAIL"
        error_reason = (
            "tolerance exceeded"
            if (_has_numeric_error(err_pt) or _has_numeric_error(err_cu))
            else (pt_error_reason or cu_error_reason or "tolerance exceeded")
        )
    return {
        "path": mtx_path,
        "shape": shape,
        "nnz": nnz,
        "error": None,
        "error_reason": error_reason,
        "op": op,
        "transpose": transpose,
        "triton_ms": triton_ms,
        "cusparse_ms": cusparse_ms,
        "pytorch_ms": pytorch_ms,
        "csc_ms": csc_ms,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "triton_ok_pt": triton_ok_pt,
        "triton_ok_cu": triton_ok_cu,
        "status": status,
    }


def run_mtx_batch(
    mtx_paths,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    warmup=10,
    iters=50,
    run_cusparse=True,
    on_result=None,
    op="non",
):
    """Batch run SpMV on multiple .mtx files; return list of result dicts."""
    op = _normalize_op(op)
    results = []
    for path in mtx_paths:
        try:
            r = run_one_mtx(
                path,
                value_dtype=value_dtype,
                index_dtype=index_dtype,
                warmup=warmup,
                iters=iters,
                run_cusparse=run_cusparse,
                op=op,
            )
        except Exception as exc:
            r = {
                "path": path,
                "shape": (0, 0),
                "nnz": 0,
                "error": str(exc),
                "error_reason": str(exc),
                "op": op,
                "transpose": _op_transposes(op),
                "triton_ms": None,
                "cusparse_ms": None,
                "pytorch_ms": None,
                "csc_ms": None,
                "err_pt": None,
                "err_cu": None,
                "triton_ok_pt": False,
                "triton_ok_cu": False,
                "status": "ERROR",
            }
        results.append(r)
        if on_result is not None:
            on_result(r)
    return results


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_speedup(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return "N/A"
    return f"{other_ms / triton_ms:.2f}x"


def _speedup_ratio(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return None
    return other_ms / triton_ms


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _status_str(ok, available):
    if not available:
        return "N/A"
    return "PASS" if ok else "FAIL"


def _print_mtx_header(value_dtype, index_dtype, op="non"):
    op = _normalize_op(op)
    transpose = _op_transposes(op)
    print(
        f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}  |  op: {op}  |  transpose: {bool(transpose)}"
    )
    print("Formats: FlagSparse=CSR, cuSPARSE=CSR/CSC, PyTorch=CSR or COO.")
    print(
        "Timing stays in native dtype. For float32, correctness references use float64 compute then cast."
    )
    print(
        "Timing policy: non = compute only; trans/conj = raw op materialization + compute."
    )
    print(
        "PT/CU show per-reference correctness. Err(PT)/Err(CU)=max(|diff| / (atol + rtol*|ref|))."
    )
    print("-" * 150)
    print(
        f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
        f"{'FlagSparse(ms)':>10} {'CSR(ms)':>10} {'CSC(ms)':>10} {'PyTorch(ms)':>11} "
        f"{'FS/CSR':>7} {'FS/PT':>7} {'PT':>6} {'CU':>6} {'Err(PT)':>10} {'Err(CU)':>10}"
    )
    print("-" * 150)


def _print_mtx_row(r):
    basename = os.path.basename(r["path"])
    name = basename[:27] + ("..." if len(basename) > 27 else "")
    n_rows, n_cols = r["shape"]
    if r.get("status") == "ERROR":
        print(
            f"{name:<28} {n_rows:>7} {n_cols:>7} {r['nnz']:>10} "
            f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>11} "
            f"{'N/A':>7} {'N/A':>7} {'ERROR':>6} {'ERROR':>6} "
            f"{'N/A':>10} {'N/A':>10}  {r.get('error_reason', r.get('error', ''))}"
        )
        return
    triton_ms = r.get("triton_ms")
    csr_ms = r.get("cusparse_ms")
    csc_ms = r.get("csc_ms")
    pt_ms = r.get("pytorch_ms")
    err_pt_str = _fmt_err(r.get("err_pt"))
    err_cu_str = _fmt_err(r.get("err_cu"))
    pt_status = _status_str(r.get("triton_ok_pt", False), r.get("err_pt") is not None)
    cu_status = _status_str(r.get("triton_ok_cu", False), r.get("err_cu") is not None)
    print(
        f"{name:<28} {n_rows:>7} {n_cols:>7} {r['nnz']:>10} "
        f"{_fmt_ms(triton_ms):>10} {_fmt_ms(csr_ms):>10} {_fmt_ms(csc_ms):>10} {_fmt_ms(pt_ms):>11} "
        f"{_fmt_speedup(csr_ms, triton_ms):>7} {_fmt_speedup(pt_ms, triton_ms):>7} "
        f"{pt_status:>6} {cu_status:>6} {err_pt_str:>10} {err_cu_str:>10}"
    )


def print_mtx_results(results, value_dtype, index_dtype):
    op = results[0].get("op", "non") if results else "non"
    _print_mtx_header(value_dtype, index_dtype, op=op)
    for r in results:
        _print_mtx_row(r)
    print("-" * 150)


def _dtype_str(d):
    return str(d).replace("torch.", "")


def run_all_dtypes_export_csv(
    paths,
    csv_path,
    warmup=10,
    iters=50,
    run_cusparse=True,
    ops=None,
    value_dtypes=None,
    index_dtypes=None,
):
    """Run SpMV for all VALUE_DTYPES x INDEX_DTYPES on each .mtx and write results to CSV."""
    ops = [_normalize_op(op) for op in (OP_CHOICES if ops is None else ops)]
    rows = []
    value_dtypes = (
        _csv_value_dtypes(_dtype_map()) if value_dtypes is None else value_dtypes
    )
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    for op in ops:
        transpose = _op_transposes(op)
        for value_dtype in value_dtypes:
            for index_dtype in index_dtypes:
                print("=" * 150)
                _print_mtx_header(value_dtype, index_dtype, op=op)
                results = run_mtx_batch(
                    paths,
                    value_dtype=value_dtype,
                    index_dtype=index_dtype,
                    warmup=warmup,
                    iters=iters,
                    run_cusparse=run_cusparse,
                    on_result=_print_mtx_row,
                    op=op,
                )
                print("-" * 150)
                for r in results:
                    n_rows, n_cols = r["shape"]
                    rows.append(
                        {
                            "matrix": os.path.basename(r["path"]),
                            "value_dtype": _dtype_str(value_dtype),
                            "index_dtype": _dtype_str(index_dtype),
                            "op": op,
                            "transpose": bool(transpose),
                            "n_rows": n_rows,
                            "n_cols": n_cols,
                            "nnz": r["nnz"],
                            "triton_ms": r.get("triton_ms"),
                            "cusparse_ms": r.get("cusparse_ms"),
                            "pytorch_ms": r.get("pytorch_ms"),
                            "csc_ms": r.get("csc_ms"),
                            "triton_speedup_vs_cusparse": _speedup_ratio(
                                r.get("cusparse_ms"), r.get("triton_ms")
                            ),
                            "triton_speedup_vs_pytorch": _speedup_ratio(
                                r.get("pytorch_ms"), r.get("triton_ms")
                            ),
                            "pt_status": _status_str(
                                r.get("triton_ok_pt", False),
                                r.get("err_pt") is not None,
                            ),
                            "cu_status": _status_str(
                                r.get("triton_ok_cu", False),
                                r.get("err_cu") is not None,
                            ),
                            "status": r.get("status", r.get("error", "")),
                            "error_reason": r.get("error_reason", r.get("error")),
                            "err_pt": r.get("err_pt"),
                            "err_cu": r.get("err_cu"),
                        }
                    )
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
        "transpose",
        "n_rows",
        "n_cols",
        "nnz",
        "triton_ms",
        "cusparse_ms",
        "pytorch_ms",
        "csc_ms",
        "triton_speedup_vs_cusparse",
        "triton_speedup_vs_pytorch",
        "pt_status",
        "cu_status",
        "status",
        "error_reason",
        "err_pt",
        "err_cu",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})
    print(f"Wrote {len(rows)} rows to {csv_path}")


def run_comprehensive_synthetic(op="non"):
    """Synthetic benchmark with per-case table (like test_gather)."""
    op = _normalize_op(op)
    transpose = _op_transposes(op)
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    print("=" * 110)
    print("FLAGSPARSE SpMV BENCHMARK (synthetic CSR)")
    print("=" * 110)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}  |  Warmup: {WARMUP}  Iters: {ITERS}  |  op: {op}  |  transpose: {bool(transpose)}"
    )
    print(
        "Formats: FlagSparse=CSR, cuSPARSE=CSR (when supported), Reference=CuPy CSR or PyTorch COO"
    )
    print(
        "When CuPy does not support dtype (e.g. bfloat16/float16), reference = PyTorch (float32 then cast)."
    )
    print()
    total = 0
    failed = 0
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("-" * 110)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<12}  |  Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * 110)
            print(
                f"{'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                f"{'FlagSparse(ms)':>11} {'cuSPARSE(ms)':>12} {'FS/CS':>8} "
                f"{'Status':>6} {'Err(FS)':>10} {'Err(CS)':>10}"
            )
            print("-" * 110)
            for n_rows, n_cols, nnz in TEST_CASES:
                total += 1
                try:
                    result = ast.benchmark_spmv_case(
                        n_rows=n_rows,
                        n_cols=n_cols,
                        nnz=nnz,
                        value_dtype=value_dtype,
                        index_dtype=index_dtype,
                        warmup=WARMUP,
                        iters=ITERS,
                        run_cusparse=True,
                        op=op,
                    )
                    perf = result["performance"]
                    verify = result["verification"]
                    ok = verify["triton_match_reference"]
                    cs_ok = verify.get("cusparse_match_reference")
                    status = "PASS" if (ok and (cs_ok is None or cs_ok)) else "FAIL"
                    if status == "FAIL":
                        failed += 1
                    triton_ms = perf["triton_ms"]
                    cusparse_ms = perf["cusparse_ms"]
                    speedup = perf.get("triton_speedup_vs_cusparse")
                    speedup_str = f"{speedup:.2f}x" if speedup is not None else "N/A"
                    print(
                        f"{n_rows:>7} {n_cols:>7} {nnz:>10} "
                        f"{_fmt_ms(triton_ms):>11} {_fmt_ms(cusparse_ms):>12} {speedup_str:>8} "
                        f"{status:>6} {_fmt_err(verify.get('triton_max_error')):>10} "
                        f"{_fmt_err(verify.get('cusparse_max_error')):>10}"
                    )
                except Exception as exc:
                    failed += 1
                    print(
                        f"{n_rows:>7} {n_cols:>7} {nnz:>10} "
                        f"{'N/A':>11} {'N/A':>12} {'N/A':>8} "
                        f"{'ERROR':>6} {'N/A':>10} {'N/A':>10}  {exc}"
                    )
            print("-" * 110)
            print()
    print("=" * 110)
    print(f"Total: {total}  Failed: {failed}")
    print("=" * 110)


def main():
    dtype_map = _dtype_map()
    parser = argparse.ArgumentParser(
        description="SpMV test: SuiteSparse .mtx batch run, error and performance."
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
        "--ops",
        default=None,
        help="Comma-separated CSR SpMV ops: non,trans,conj. CSV default: all; non-CSV default: non.",
    )
    parser.add_argument(
        "--dtype",
        default=None,
        choices=sorted(dtype_map.keys()),
        help="Value dtype filter (default: float32 for single run; CSV runs float32/float64/complex64/complex128)",
    )
    parser.add_argument(
        "--index-dtype",
        default=None,
        choices=["int32", "int64"],
        help="Index dtype filter (default: int32 for single run; CSV runs int32/int64)",
    )
    parser.add_argument("--warmup", type=int, default=10, help="Warmup runs")
    parser.add_argument("--iters", type=int, default=50, help="Timing iterations")
    parser.add_argument(
        "--no-cusparse", action="store_true", help="Skip cuSPARSE baseline"
    )
    parser.add_argument(
        "--csv-csr",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all value_dtype x index_dtype on all .mtx (CSR) and write results to CSV",
    )
    args = parser.parse_args()
    index_map = {"int32": torch.int32, "int64": torch.int64}
    value_dtype_name = args.dtype or "float32"
    index_dtype_name = args.index_dtype or "int32"
    value_dtype = dtype_map[value_dtype_name]
    index_dtype = index_map[index_dtype_name]
    try:
        ops = _parse_ops(
            args.ops,
            default_ops=OP_CHOICES if args.csv_csr is not None else ("non",),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.synthetic:
        for op in ops:
            run_comprehensive_synthetic(op=op)
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if not paths and not args.csv_csr:
        print(
            "No .mtx files given. Use: python test_spmv.py <file.mtx> [file2.mtx ...] or <dir/>"
        )
        print("Or run synthetic: python test_spmv.py --synthetic")
        print(
            "Or run all dtypes and export CSR CSV: python test_spmv.py <dir/> --csv-csr results.csv"
        )
        return
    if args.csv_csr is not None:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found. Specify files or a directory.")
            return
        print("=" * 80)
        print("FLAGSPARSE SpMV (CSR) all dtypes, export to CSV")
        print("=" * 80)
        print(
            f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}  |  CSV: {args.csv_csr}  |  ops: {','.join(ops)}"
        )
        run_all_dtypes_export_csv(
            paths,
            args.csv_csr,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            ops=ops,
            value_dtypes=(
                [dtype_map[args.dtype]] if args.dtype else _csv_value_dtypes(dtype_map)
            ),
            index_dtypes=(
                [index_map[args.index_dtype]] if args.index_dtype else INDEX_DTYPES
            ),
        )
        return
    print("=" * 120)
    print("FLAGSPARSE SpMV SuiteSparse .mtx batch (error + performance)")
    print("=" * 120)
    print(f"GPU: {torch.cuda.get_device_name(0)}  |  Files: {len(paths)}")
    print(
        f"dtype: {value_dtype_name}  index_dtype: {index_dtype_name}  ops: {','.join(ops)}  warmup: {args.warmup}  iters: {args.iters}"
    )
    print()
    total_passed = 0
    total_cases = 0
    for op in ops:
        results = run_mtx_batch(
            paths,
            value_dtype=value_dtype,
            index_dtype=index_dtype,
            warmup=args.warmup,
            iters=args.iters,
            run_cusparse=not args.no_cusparse,
            op=op,
        )
        print_mtx_results(results, value_dtype, index_dtype)
        passed = sum(1 for r in results if r.get("status") == "PASS")
        total_passed += passed
        total_cases += len(results)
        print(f"op={op} Passed: {passed} / {len(results)}")
    print(f"Total Passed: {total_passed} / {total_cases}")


if __name__ == "__main__":
    main()
