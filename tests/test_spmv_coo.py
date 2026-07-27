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

"""SpMV COO tests: dtype/index/op grids, synthetic + optional .mtx, CSV export."""

import argparse
import glob
import csv
import math
import os
import time

import torch
import flagsparse as fs
import flagsparse.sparse_operations.spmv_coo as spmv_coo_mod

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    cp = None
    cpx_sparse = None

VALUE_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
INDEX_DTYPES = [torch.int32, torch.int64]
OPS = ["non", "trans", "conj"]
TEST_SIZES = [(512, 512), (1024, 1024), (2048, 2048)]
WARMUP = 10
ITERS = 50

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}
INDEX_DTYPE_MAP = {
    "int32": torch.int32,
    "int64": torch.int64,
}
OP_NAMES = ("non", "trans", "conj")


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _parse_csv_tokens(value, mapping, name):
    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError(f"{name} must not be empty")
    unknown = [token for token in tokens if token not in mapping]
    if unknown:
        allowed = ", ".join(mapping)
        raise ValueError(
            f"unsupported {name}: {', '.join(unknown)}; allowed: {allowed}"
        )
    return [mapping[token] for token in tokens]


def _parse_ops(value):
    tokens = [token.strip().lower() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("--ops must not be empty")
    unknown = [token for token in tokens if token not in OP_NAMES]
    if unknown:
        raise ValueError(
            f"unsupported op: {', '.join(unknown)}; allowed: {', '.join(OP_NAMES)}"
        )
    return tokens


def _random_values(shape, dtype, device):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.complex64:
        real = torch.randn(shape, dtype=torch.float32, device=device)
        imag = torch.randn(shape, dtype=torch.float32, device=device)
        return torch.complex(real, imag)
    if dtype == torch.complex128:
        real = torch.randn(shape, dtype=torch.float64, device=device)
        imag = torch.randn(shape, dtype=torch.float64, device=device)
        return torch.complex(real, imag)
    raise TypeError(f"unsupported dtype: {dtype}")


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


def _allclose_error_ratio(actual, reference, atol, rtol):
    if actual.numel() == 0:
        return 0.0
    diff = torch.abs(actual - reference).to(torch.float64)
    tol = (atol + rtol * torch.abs(reference)).to(torch.float64)
    return float(torch.max(diff / tol).item())


def _reference_dtype(dtype):
    if dtype == torch.float32:
        return torch.float64
    if dtype == torch.complex64:
        return torch.complex128
    return dtype


def _tol_for_dtype(dtype):
    if dtype == torch.float16:
        return 1e-3, 2e-3
    if dtype == torch.bfloat16:
        return 0.016, 1e-1
    if dtype in (torch.float32, torch.complex64):
        return 1.3e-6, 1e-3
    if dtype in (torch.float64, torch.complex128):
        return 1e-7, 1e-5
    return 1e-6, 1e-5


def _op_transposes(op):
    return op in ("trans", "conj")


def _out_size_for_op(shape, op):
    n_rows, n_cols = shape
    return n_cols if _op_transposes(op) else n_rows


def _x_size_for_op(shape, op):
    n_rows, n_cols = shape
    return n_rows if _op_transposes(op) else n_cols


def _apply_coo_op(data, row, col, shape, op):
    if op == "non":
        return data, row, col, shape
    n_rows, n_cols = shape
    data_op = data
    if op == "conj":
        data_op = data.conj()
        if hasattr(data_op, "resolve_conj"):
            data_op = data_op.resolve_conj()
    elif op != "trans":
        raise ValueError(f"unsupported op: {op}")
    return data_op, col, row, (n_cols, n_rows)


def _apply_torch_sparse_op(matrix, x_2d, op):
    if op == "non":
        return torch.sparse.mm(matrix, x_2d).squeeze(1)
    if op == "trans":
        return torch.sparse.mm(matrix.transpose(0, 1), x_2d).squeeze(1)
    if op == "conj":
        return torch.sparse.mm(matrix.conj().transpose(0, 1), x_2d).squeeze(1)
    raise ValueError(f"unsupported op: {op}")


def _cupy_sparse_op_matrix(matrix, op):
    if op == "non":
        return matrix
    if op == "trans":
        return matrix.T
    if op == "conj":
        matrix_conj = matrix.conj() if hasattr(matrix, "conj") else matrix.conjugate()
        return matrix_conj.T
    raise ValueError(f"unsupported op: {op}")


def _apply_cupy_sparse_op(matrix, x, op):
    return _cupy_sparse_op_matrix(matrix, op) @ x


def _apply_cupy_coo_op(data, row, col, shape, op):
    if op == "non":
        return data, row, col, shape
    n_rows, n_cols = shape
    data_op = data
    if op == "conj":
        data_op = data.conj() if hasattr(data, "conj") else data.conjugate()
    elif op != "trans":
        raise ValueError(f"unsupported op: {op}")
    return data_op, col, row, (n_cols, n_rows)


def _pytorch_coo_reference(data, row, col, x, shape, out_dtype, op="non"):
    data, row, col, shape = _apply_coo_op(data, row, col, shape, op)
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    coo_ref = torch.sparse_coo_tensor(
        torch.stack([row.to(torch.int64), col.to(torch.int64)]),
        data_ref,
        shape,
        device=data.device,
    ).coalesce()
    y_ref = torch.sparse.mm(coo_ref, x_ref.unsqueeze(1)).squeeze(1)
    return y_ref.to(out_dtype) if ref_dtype != out_dtype else y_ref


def _cupy_coo_reference(data, row, col, x, shape, out_dtype, op="non"):
    data, row, col, shape = _apply_coo_op(data, row, col, shape, op)
    ref_dtype = _reference_dtype(out_dtype)
    data_ref = data.to(ref_dtype)
    x_ref = x.to(ref_dtype)
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref))
    row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.to(torch.int64)))
    col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.to(torch.int64)))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x_ref))
    A_cp_ref = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
    y_ref = A_cp_ref @ x_cp
    y_ref_t = torch.utils.dlpack.from_dlpack(y_ref.toDlpack())
    return y_ref_t.to(out_dtype) if ref_dtype != out_dtype else y_ref_t


def _dense_to_coo(A):
    rows, cols = A.nonzero(as_tuple=True)
    data = A[rows, cols]
    return data, rows, cols


COO_SEP = "-" * 200


def _coo_header(timing=False):
    split = (
        f" {'BasePGPU':>9} {'BaseComp':>9} {'OptPGPU':>9} {'OptComp':>9}"
        if timing
        else ""
    )
    base = (
        f"{'Matrix':<28} {'Op':>5} {'Out':>7} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10}  "
        f"{'Base(ms)':>9} {'BaseGPU':>9} {'BaseCPU':>9} "
        f"{'Opt(ms)':>9} {'OptGPU':>9} {'OptCPU':>9}{split}"
    )
    return (
        base + f" {'PT(ms)':>9} {'CU(ms)':>9}  "
        f"{'Opt/Base':>8} {'Opt/PT':>8} {'Opt/CU':>8}  "
        f"{'Err(Base)':>10} {'Err(Opt)':>10} {'Status':>6}"
    )


def _spd(num, den):
    if num is None or den is None or den <= 0:
        return "N/A"
    return f"{num / den:.2f}x"


# FlagSparse native COO SpMV: see sparse_operations.spmv_coo
COO_ATOMIC_BLOCK = 256
COO_ATOMIC_WARPS = 4
COO_SEG_BLOCK_INNER = 128


def _cuda_event_benchmark(op, warmup, iters):
    out = None
    count = max(1, int(iters))
    for _ in range(max(0, int(warmup))):
        out = op()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(count):
        out = op()
    e1.record()
    torch.cuda.synchronize()
    return out, e0.elapsed_time(e1) / count


def _run_flagsparse_coo_launch(
    launch,
    x,
):
    x = spmv_coo_mod._validate_x_coo(x, launch)
    return spmv_coo_mod._run_spmv_coo_prepared_with_fallback(
        launch,
        x,
        block_size=COO_ATOMIC_BLOCK,
        num_warps=COO_ATOMIC_WARPS,
        block_inner=COO_SEG_BLOCK_INNER,
    )


def _build_flagsparse_coo_launch(
    data,
    row,
    col,
    shape,
    sort_by_row,
    op,
):
    return spmv_coo_mod._prepare_spmv_coo_launch_from_raw(
        data=data,
        row=row,
        col=col,
        shape=shape,
        sort_by_row=sort_by_row,
        op=op,
    )


def _time_flagsparse_coo_row_run(
    data,
    row,
    col,
    x,
    shape,
    op,
    warmup,
    iters,
    timing=False,
):
    op = str(op).lower()

    def prepare():
        return _build_flagsparse_coo_launch(
            data,
            row,
            col,
            shape,
            True,
            op,
        )

    def full_op():
        launch = prepare()
        return _run_flagsparse_coo_launch(launch, x)

    y, gpu_ms = _cuda_event_benchmark(full_op, warmup, iters)
    process_gpu_ms = None
    compute_ms = None
    total_ms = gpu_ms
    if timing:
        launch, process_gpu_ms = _cuda_event_benchmark(prepare, warmup, iters)
        y, compute_ms = _cuda_event_benchmark(
            lambda: _run_flagsparse_coo_launch(launch, x),
            warmup,
            iters,
        )
        total_ms = process_gpu_ms + compute_ms
    return {
        "out": y,
        "ms": total_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": process_gpu_ms,
        "compute_ms": compute_ms,
    }


def _time_flagsparse_coo_atomic(
    data,
    row,
    col,
    x,
    shape,
    op,
    warmup,
    iters,
    timing=False,
):
    launch = _build_flagsparse_coo_launch(
        data,
        row,
        col,
        shape,
        False,
        str(op).lower(),
    )
    y, gpu_ms = _cuda_event_benchmark(
        lambda: _run_flagsparse_coo_launch(launch, x),
        warmup,
        iters,
    )
    return {
        "out": y,
        "ms": gpu_ms,
        "gpu_ms": gpu_ms,
        "process_cpu_ms": 0.0,
        "process_gpu_ms": 0.0 if timing else None,
        "compute_ms": gpu_ms if timing else None,
    }


def _timed_flagsparse_coo_tocsr_runtime(
    data,
    row,
    col,
    x,
    shape,
    warmup,
    iters,
):
    spmv_op = lambda: fs.flagsparse_spmv_coo_tocsr(
        data,
        row,
        col,
        x,
        shape=shape,
        assume_sorted=False,
    )
    y = spmv_op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        y = spmv_op()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        y = spmv_op()
    e1.record()
    torch.cuda.synchronize()
    return y, e0.elapsed_time(e1) / iters


def _timed_flagsparse_coo_tocsr_prepared(
    prepared,
    x,
    warmup,
    iters,
):
    spmv_op = lambda: fs.flagsparse_spmv_coo_tocsr(x=x, prepared=prepared)
    y = spmv_op()
    torch.cuda.synchronize()
    for _ in range(warmup):
        y = spmv_op()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        y = spmv_op()
    e1.record()
    torch.cuda.synchronize()
    return y, e0.elapsed_time(e1) / iters


def run_synthetic(
    value_dtypes=None,
    index_dtypes=None,
    ops=None,
    warmup=WARMUP,
    iters=ITERS,
    timing=False,
):
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device("cuda")
    print("=" * 172)
    print(
        "FLAGSPARSE SpMV COO BENCHMARK (synthetic dense -> COO). All backends stay COO."
    )
    print("=" * 172)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warmup: {warmup} | Iters: {iters}")
    print()

    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    ops = OPS if ops is None else ops

    for dtype in value_dtypes:
        atol, rtol = _tol_for_dtype(dtype)
        for index_dtype in index_dtypes:
            for op in ops:
                print(COO_SEP)
                print(
                    f"dtype: {_dtype_name(dtype)}  index_dtype: {_dtype_name(index_dtype)}  op: {op}"
                )
                print(COO_SEP)
                print("FlagSparse: native COO Triton only (no CSR).")
                print(
                    "Base(ms)=BaseCPU+BaseGPU; BaseGPU wraps row-run sort + seg_starts + segmented kernel."
                )
                print(
                    "Opt(ms)=OptCPU+OptGPU; atomic has no segment/bucket preprocessing."
                )
                print(
                    "--timing splits row-run GPU work into BasePGPU + BaseComp; atomic OptPGPU is zero."
                )
                print(
                    "Speedups use Opt(ms) as the Triton comparison path; Base(ms) is reported separately."
                )
                print(COO_SEP)
                print(_coo_header(timing=timing))
                print(COO_SEP)
                for m, n in TEST_SIZES:
                    A = _random_values((m, n), dtype, device)
                    A *= (torch.rand(m, n, device=device) < 0.1).to(dtype=dtype)
                    data, row, col = _dense_to_coo(A)
                    row = row.to(index_dtype).contiguous()
                    col = col.to(index_dtype).contiguous()
                    result = _run_one_coo_case(
                        data=data,
                        row=row,
                        col=col,
                        shape=(m, n),
                        dtype=dtype,
                        index_dtype=index_dtype,
                        op=op,
                        matrix_name=f"{m}x{n}",
                        warmup=warmup,
                        iters=iters,
                        timing=timing,
                    )
                    _print_coo_result(result, timing=timing)
                print(COO_SEP)
                print()


def _run_one_coo_case(
    data,
    row,
    col,
    shape,
    dtype,
    index_dtype,
    op,
    matrix_name,
    warmup,
    iters,
    timing=False,
):
    row = row.to(index_dtype).contiguous()
    col = col.to(index_dtype).contiguous()
    x = _random_values((_x_size_for_op(shape, op),), dtype, data.device)
    atol, rtol = _tol_for_dtype(dtype)
    base = _time_flagsparse_coo_row_run(
        data,
        row,
        col,
        x,
        shape,
        op,
        warmup,
        iters,
        timing=timing,
    )
    opt = _time_flagsparse_coo_atomic(
        data,
        row,
        col,
        x,
        shape,
        op,
        warmup,
        iters,
        timing=timing,
    )
    y_base = base["out"]
    y_opt = opt["out"]
    y_ref = _pytorch_coo_reference(data, row, col, x, shape, dtype, op=op)
    err_base = _allclose_error_ratio(y_base, y_ref, atol, rtol)
    err_opt = _allclose_error_ratio(y_opt, y_ref, atol, rtol)
    err_pt = None
    err_cu = None
    pt_ms = _time_pytorch_coo(data, row, col, x, shape, op, warmup, iters)
    y_pt = _pytorch_coo_reference(data, row, col, x, shape, dtype, op=op)
    err_pt = _allclose_error_ratio(y_opt, y_pt, atol, rtol)
    cu_ms = None
    triton_ok_cu = False
    if cp is not None and cpx_sparse is not None:
        try:
            cu_ms = _time_cupy_coo(data, row, col, x, shape, op, warmup, iters)
            y_cu = _cupy_coo_reference(data, row, col, x, shape, dtype, op=op)
            err_cu = _allclose_error_ratio(y_opt, y_cu, atol, rtol)
            triton_ok_cu = (not math.isnan(err_cu)) and err_cu <= 1.0
        except Exception:
            cu_ms = None
            err_cu = None
    triton_ok_pt = (not math.isnan(err_pt)) and err_pt <= 1.0
    status = (
        "PASS"
        if (
            (not math.isnan(err_base))
            and (not math.isnan(err_opt))
            and err_base <= 1.0
            and err_opt <= 1.0
        )
        else "FAIL"
    )
    n_rows, n_cols = shape
    return {
        "matrix": matrix_name,
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "out_size": _out_size_for_op(shape, op),
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(data.numel()),
        "base_ms": base["ms"],
        "opt_ms": opt["ms"],
        "base_gpu_ms": base["gpu_ms"],
        "base_process_cpu_ms": base["process_cpu_ms"],
        "base_process_gpu_ms": base["process_gpu_ms"],
        "base_compute_ms": base["compute_ms"],
        "opt_gpu_ms": opt["gpu_ms"],
        "opt_process_cpu_ms": opt["process_cpu_ms"],
        "opt_process_gpu_ms": opt["process_gpu_ms"],
        "opt_compute_ms": opt["compute_ms"],
        "cusparse_ms": cu_ms,
        "pytorch_ms": pt_ms,
        "opt_speedup_vs_cusparse": _speedup_ratio(cu_ms, opt["ms"]),
        "opt_speedup_vs_pytorch": _speedup_ratio(pt_ms, opt["ms"]),
        "pt_status": _status_str(triton_ok_pt, err_pt is not None),
        "cu_status": _status_str(triton_ok_cu, err_cu is not None),
        "status": status,
        "err_base": err_base,
        "err_opt": err_opt,
        "err_pt": err_pt,
        "err_cu": err_cu,
    }


def _run_one_tocsr_case(
    data,
    row,
    col,
    shape,
    dtype,
    index_dtype,
    matrix_name,
    warmup,
    iters,
):
    row = row.to(index_dtype).contiguous()
    col = col.to(index_dtype).contiguous()
    x = _random_values((shape[1],), dtype, data.device)
    atol, rtol = _tol_for_dtype(dtype)
    y_runtime, runtime_ms = _timed_flagsparse_coo_tocsr_runtime(
        data,
        row,
        col,
        x,
        shape,
        warmup,
        iters,
    )
    prepared = fs.prepare_spmv_coo_tocsr(data, row, col, shape)
    y_prepared, prepared_ms = _timed_flagsparse_coo_tocsr_prepared(
        prepared,
        x,
        warmup,
        iters,
    )
    y_ref = _pytorch_coo_reference(data, row, col, x, shape, dtype, op="non")
    err_runtime = _allclose_error_ratio(y_runtime, y_ref, atol, rtol)
    err_prepared = _allclose_error_ratio(y_prepared, y_ref, atol, rtol)
    pt_ms = _time_pytorch_coo(data, row, col, x, shape, "non", warmup, iters)
    cu_ms = None
    err_cu = None
    triton_ok_cu = False
    if cp is not None and cpx_sparse is not None:
        try:
            cu_ms = _time_cupy_coo(data, row, col, x, shape, "non", warmup, iters)
            y_cu = _cupy_coo_reference(data, row, col, x, shape, dtype, op="non")
            err_cu = _allclose_error_ratio(y_prepared, y_cu, atol, rtol)
            triton_ok_cu = (not math.isnan(err_cu)) and err_cu <= 1.0
        except Exception:
            cu_ms = None
            err_cu = None
    triton_ok_pt = (not math.isnan(err_prepared)) and err_prepared <= 1.0
    status = (
        "PASS"
        if (
            (not math.isnan(err_runtime))
            and (not math.isnan(err_prepared))
            and err_runtime <= 1.0
            and err_prepared <= 1.0
        )
        else "FAIL"
    )
    n_rows, n_cols = shape
    return {
        "matrix": matrix_name,
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": "non",
        "out_size": n_rows,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(data.numel()),
        "runtime_ms": runtime_ms,
        "prepared_ms": prepared_ms,
        "cusparse_ms": cu_ms,
        "pytorch_ms": pt_ms,
        "prepared_speedup_vs_runtime": _speedup_ratio(runtime_ms, prepared_ms),
        "prepared_speedup_vs_cusparse": _speedup_ratio(cu_ms, prepared_ms),
        "prepared_speedup_vs_pytorch": _speedup_ratio(pt_ms, prepared_ms),
        "pt_status": _status_str(triton_ok_pt, True),
        "cu_status": _status_str(triton_ok_cu, err_cu is not None),
        "status": status,
        "err_runtime": err_runtime,
        "err_prepared": err_prepared,
        "err_pt": err_prepared,
        "err_cu": err_cu,
    }


def _build_torch_sparse_coo(data, row, col, shape):
    return torch.sparse_coo_tensor(
        torch.stack([row, col]),
        data,
        shape,
        device=data.device,
    ).coalesce()


def _time_pytorch_coo(data, row, col, x, shape, op, warmup, iters):
    row_i64 = row.to(torch.int64)
    col_i64 = col.to(torch.int64)
    x_ref_2d = x.unsqueeze(1)
    if data.numel() == 0:
        return 0.0
    if op == "non":
        coo = _build_torch_sparse_coo(data, row_i64, col_i64, shape)
        spmv_op = lambda: torch.sparse.mm(coo, x_ref_2d).squeeze(1)
    else:
        spmv_op = lambda: _run_torch_runtime_op(
            data,
            row_i64,
            col_i64,
            x_ref_2d,
            shape,
            op,
        )
    torch.cuda.synchronize()
    for _ in range(warmup):
        _ = spmv_op()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        _ = spmv_op()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def _run_torch_runtime_op(data, row, col, x_2d, shape, op):
    data_op, row_op, col_op, shape_op = _apply_coo_op(data, row, col, shape, op)
    coo = _build_torch_sparse_coo(data_op, row_op, col_op, shape_op)
    return torch.sparse.mm(coo, x_2d).squeeze(1)


def _time_cupy_coo(data, row, col, x, shape, op, warmup, iters):
    data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data))
    row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.to(torch.int64)))
    col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.to(torch.int64)))
    x_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
    if op == "non":
        A_cp = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
        spmv_op = lambda: A_cp @ x_cp
    else:
        spmv_op = lambda: _run_cupy_runtime_op(
            data_cp,
            row_cp,
            col_cp,
            x_cp,
            shape,
            op,
        )
    for _ in range(warmup):
        _ = spmv_op()
    cp.cuda.runtime.deviceSynchronize()
    c0 = cp.cuda.Event()
    c1 = cp.cuda.Event()
    c0.record()
    for _ in range(iters):
        _ = spmv_op()
    c1.record()
    c1.synchronize()
    return cp.cuda.get_elapsed_time(c0, c1) / iters


def _run_cupy_runtime_op(data, row, col, x, shape, op):
    data_op, row_op, col_op, shape_op = _apply_cupy_coo_op(data, row, col, shape, op)
    A_cp = cpx_sparse.coo_matrix((data_op, (row_op, col_op)), shape=shape_op)
    return A_cp @ x


def _print_coo_result(row, timing=False):
    name = str(row["matrix"])[:27]
    if len(str(row["matrix"])) > 27:
        name += "…"
    split_text = ""
    if timing:
        split_text = (
            f" {_fmt_ms(row.get('base_process_gpu_ms')):>9}"
            f" {_fmt_ms(row.get('base_compute_ms')):>9}"
            f" {_fmt_ms(row.get('opt_process_gpu_ms')):>9}"
            f" {_fmt_ms(row.get('opt_compute_ms')):>9}"
        )
    print(
        f"{name:<28} {row['op']:>5} {row['out_size']:>7} "
        f"{row['n_rows']:>7} {row['n_cols']:>7} {row['nnz']:>10}  "
        f"{_fmt_ms(row['base_ms']):>9} {_fmt_ms(row.get('base_gpu_ms')):>9} {_fmt_ms(row.get('base_process_cpu_ms')):>9} "
        f"{_fmt_ms(row['opt_ms']):>9} {_fmt_ms(row.get('opt_gpu_ms')):>9} {_fmt_ms(row.get('opt_process_cpu_ms')):>9}"
        f"{split_text} "
        f"{_fmt_ms(row['pytorch_ms']):>9} {_fmt_ms(row['cusparse_ms']):>9}  "
        f"{_spd(row['base_ms'], row['opt_ms']):>8} "
        f"{_spd(row['pytorch_ms'], row['opt_ms']):>8} "
        f"{_spd(row['cusparse_ms'], row['opt_ms']):>8}  "
        f"{_fmt_err(row['err_base']):>10} {_fmt_err(row['err_opt']):>10} "
        f"{row['status']:>6}"
    )


TOCSR_SEP = "-" * 200
TOCSR_HEADER = (
    f"{'Matrix':<28} {'Out':>7} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10}  "
    f"{'Runtime(ms)':>11} {'Prepared(ms)':>12} {'PT(ms)':>9} {'CU(ms)':>9}  "
    f"{'Prep/Run':>9} {'Prep/PT':>8} {'Prep/CU':>8}  "
    f"{'Err(Runtime)':>12} {'Err(Prepared)':>13} {'Status':>6}"
)


def _print_tocsr_result(row):
    name = str(row["matrix"])[:27]
    if len(str(row["matrix"])) > 27:
        name += "..."
    print(
        f"{name:<28} {row['out_size']:>7} "
        f"{row['n_rows']:>7} {row['n_cols']:>7} {row['nnz']:>10}  "
        f"{_fmt_ms(row['runtime_ms']):>11} {_fmt_ms(row['prepared_ms']):>12} "
        f"{_fmt_ms(row['pytorch_ms']):>9} {_fmt_ms(row['cusparse_ms']):>9}  "
        f"{_spd(row['runtime_ms'], row['prepared_ms']):>9} "
        f"{_spd(row['pytorch_ms'], row['prepared_ms']):>8} "
        f"{_spd(row['cusparse_ms'], row['prepared_ms']):>8}  "
        f"{_fmt_err(row['err_runtime']):>12} {_fmt_err(row['err_prepared']):>13} "
        f"{row['status']:>6}"
    )


def _mtx_value_for_dtype(raw_value, dtype):
    if dtype in (torch.complex64, torch.complex128):
        if isinstance(raw_value, complex):
            return raw_value
        return complex(float(raw_value), 0.5 * float(raw_value))
    if isinstance(raw_value, complex):
        return float(raw_value.real)
    return float(raw_value)


def _load_mtx_to_coo_torch(file_path, dtype=torch.float32, device=None):
    """Load a .mtx into COO torch tensors via the C-accelerated scipy reader
    (see tests/mtx_fast.py); the former pure-Python parser took minutes on
    large SuiteSparse matrices. Returns (vals, rows, cols, shape)."""
    from mtx_fast import load_coo

    data, rows, cols, shape = load_coo(file_path, dtype=dtype, device=device)
    return data, rows, cols, shape


# Dense PyTorch reference for SpSV can OOM on large matrices.
DENSE_REF_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB


def _allow_dense_pytorch_ref(shape, dtype):
    n_rows, n_cols = shape
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    dense_bytes = int(n_rows) * int(n_cols) * int(elem_bytes)
    return dense_bytes <= DENSE_REF_MAX_BYTES


def _error_row(path, dtype, index_dtype, op):
    return {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": op,
        "out_size": "ERR",
        "n_rows": "ERR",
        "n_cols": "ERR",
        "nnz": "ERR",
        "base_ms": None,
        "opt_ms": None,
        "base_gpu_ms": None,
        "base_process_cpu_ms": None,
        "base_process_gpu_ms": None,
        "base_compute_ms": None,
        "opt_gpu_ms": None,
        "opt_process_cpu_ms": None,
        "opt_process_gpu_ms": None,
        "opt_compute_ms": None,
        "cusparse_ms": None,
        "pytorch_ms": None,
        "opt_speedup_vs_cusparse": None,
        "opt_speedup_vs_pytorch": None,
        "status": "ERROR",
        "err_base": None,
        "err_opt": None,
        "err_pt": None,
        "err_cu": None,
        "pt_status": "N/A",
        "cu_status": "N/A",
    }


def _error_row_tocsr(path, dtype, index_dtype):
    return {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(dtype),
        "index_dtype": _dtype_name(index_dtype),
        "op": "non",
        "out_size": "ERR",
        "n_rows": "ERR",
        "n_cols": "ERR",
        "nnz": "ERR",
        "runtime_ms": None,
        "prepared_ms": None,
        "cusparse_ms": None,
        "pytorch_ms": None,
        "prepared_speedup_vs_runtime": None,
        "prepared_speedup_vs_cusparse": None,
        "prepared_speedup_vs_pytorch": None,
        "pt_status": "N/A",
        "cu_status": "N/A",
        "status": "ERROR",
        "err_runtime": None,
        "err_prepared": None,
        "err_pt": None,
        "err_cu": None,
    }


def run_all_dtypes_coo_csv(
    mtx_paths,
    csv_path,
    value_dtypes=None,
    index_dtypes=None,
    ops=None,
    warmup=WARMUP,
    iters=ITERS,
    timing=False,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    value_dtypes = VALUE_DTYPES if value_dtypes is None else value_dtypes
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    ops = OPS if ops is None else ops
    print("=" * 200)
    print(
        "Input: MatrixMarket -> COO. FlagSparse: native COO Triton only (seg + atomic), no CSR."
    )
    print("PyTorch = COO sparse.mm; CuPy = COO matvec (coo_matrix @ x, no tocsr).")
    print(
        "Timing policy: Base/Opt ms = process_cpu_ms + GPU event time. "
        "Row-run sort + seg_starts are GPU process; atomic has no process. "
        "PyTorch/CuPy timings use original dtype."
    )
    print(
        f"{warmup} warmup + {iters} averaged iterations. "
        "--timing splits process_gpu_ms and compute_ms for native COO."
    )
    print("=" * 200)
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            for op in ops:
                print(COO_SEP)
                print(
                    f"Value dtype: {_dtype_name(dtype)} | Index dtype: {_dtype_name(index_dtype)} | op: {op}"
                )
                print(COO_SEP)
                print(_coo_header(timing=timing))
                print(COO_SEP)
                for path in mtx_paths:
                    try:
                        data, row, col, shape = _load_mtx_to_coo_torch(
                            path, dtype=dtype, device=device
                        )
                        result = _run_one_coo_case(
                            data=data,
                            row=row,
                            col=col,
                            shape=shape,
                            dtype=dtype,
                            index_dtype=index_dtype,
                            op=op,
                            matrix_name=os.path.basename(path),
                            warmup=warmup,
                            iters=iters,
                            timing=timing,
                        )
                        rows_out.append(result)
                        _print_coo_result(result, timing=timing)
                    except Exception as e:
                        row_out = _error_row(path, dtype, index_dtype, op)
                        rows_out.append(row_out)
                        _print_coo_result(row_out, timing=timing)
                        print(f"  ERROR: {e}")
                print(COO_SEP)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
        "out_size",
        "n_rows",
        "n_cols",
        "nnz",
        "base_ms",
        "base_gpu_ms",
        "base_process_cpu_ms",
        "opt_ms",
        "opt_gpu_ms",
        "opt_process_cpu_ms",
        "base_process_gpu_ms",
        "base_compute_ms",
        "opt_process_gpu_ms",
        "opt_compute_ms",
        "cusparse_ms",
        "pytorch_ms",
        "opt_speedup_vs_cusparse",
        "opt_speedup_vs_pytorch",
        "pt_status",
        "cu_status",
        "status",
        "err_base",
        "err_opt",
        "err_pt",
        "err_cu",
    ]
    if not timing:
        fieldnames = [
            field
            for field in fieldnames
            if field
            not in (
                "base_process_gpu_ms",
                "base_compute_ms",
                "opt_process_gpu_ms",
                "opt_compute_ms",
            )
        ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def run_all_dtypes_tocsr_csv(
    mtx_paths,
    csv_path,
    value_dtypes=None,
    index_dtypes=None,
    warmup=WARMUP,
    iters=ITERS,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    value_dtypes = (
        (torch.float32, torch.float64) if value_dtypes is None else value_dtypes
    )
    index_dtypes = INDEX_DTYPES if index_dtypes is None else index_dtypes
    print("=" * 200)
    print("Input: MatrixMarket -> COO. FlagSparse: COO-to-CSR preparation path.")
    print(
        "Runtime(ms) includes COO->CSR conversion; Prepared(ms) measures repeated CSR steady-state calls."
    )
    print(
        f"prepare_spmv_coo_tocsr once per variant + {warmup} warmup + "
        f"{iters} CUDA-event-averaged SpMV per backend."
    )
    print("=" * 200)
    for dtype in value_dtypes:
        for index_dtype in index_dtypes:
            print(TOCSR_SEP)
            print(
                f"Value dtype: {_dtype_name(dtype)} | Index dtype: {_dtype_name(index_dtype)} | op: non"
            )
            print(TOCSR_SEP)
            print(TOCSR_HEADER)
            print(TOCSR_SEP)
            for path in mtx_paths:
                try:
                    data, row, col, shape = _load_mtx_to_coo_torch(
                        path, dtype=dtype, device=device
                    )
                    result = _run_one_tocsr_case(
                        data=data,
                        row=row,
                        col=col,
                        shape=shape,
                        dtype=dtype,
                        index_dtype=index_dtype,
                        matrix_name=os.path.basename(path),
                        warmup=warmup,
                        iters=iters,
                    )
                    rows_out.append(result)
                    _print_tocsr_result(result)
                except Exception as e:
                    row_out = _error_row_tocsr(path, dtype, index_dtype)
                    rows_out.append(row_out)
                    _print_tocsr_result(row_out)
                    print(f"  ERROR: {e}")
            print(TOCSR_SEP)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "op",
        "out_size",
        "n_rows",
        "n_cols",
        "nnz",
        "runtime_ms",
        "prepared_ms",
        "cusparse_ms",
        "pytorch_ms",
        "prepared_speedup_vs_runtime",
        "prepared_speedup_vs_cusparse",
        "prepared_speedup_vs_pytorch",
        "pt_status",
        "cu_status",
        "status",
        "err_runtime",
        "err_prepared",
        "err_pt",
        "err_cu",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="SpMV COO test: synthetic dense->COO and optional .mtx, export CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic dense->COO tests"
    )
    parser.add_argument(
        "--csv-coo",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all dtypes on given .mtx and export COO SpMV results to CSV",
    )
    parser.add_argument(
        "--csv-tocsr",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all dtypes on given .mtx and export COO-to-CSR SpMV results to CSV",
    )
    parser.add_argument(
        "--dtypes",
        type=str,
        default="float32,float64,complex64,complex128",
        help="Comma-separated value dtypes: float32,float64,complex64,complex128",
    )
    parser.add_argument(
        "--index-dtypes",
        type=str,
        default="int32,int64",
        help="Comma-separated index dtypes: int32,int64",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated matrix ops: non,trans,conj; defaults to all COO ops, or non for --csv-tocsr",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Show/export native COO timing breakdown columns",
    )
    args = parser.parse_args()
    value_dtypes = _parse_csv_tokens(args.dtypes, DTYPE_MAP, "--dtypes")
    index_dtypes = _parse_csv_tokens(
        args.index_dtypes, INDEX_DTYPE_MAP, "--index-dtypes"
    )
    ops_arg = args.ops or ("non" if args.csv_tocsr else "non,trans,conj")
    ops = _parse_ops(ops_arg)

    if args.synthetic:
        run_synthetic(
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            ops=ops,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
        )
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_coo_csv(
            paths,
            args.csv_coo,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            ops=ops,
            warmup=args.warmup,
            iters=args.iters,
            timing=args.timing,
        )
        return

    if args.csv_tocsr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-tocsr")
            return
        if any(op != "non" for op in ops):
            raise ValueError("spmv_coo_tocsr performance only supports --ops non")
        run_all_dtypes_tocsr_csv(
            paths,
            args.csv_tocsr,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            warmup=args.warmup,
            iters=args.iters,
        )
        return

    print("Use --synthetic, --csv-coo, or --csv-tocsr to run COO SpMV tests.")


if __name__ == "__main__":
    main()
