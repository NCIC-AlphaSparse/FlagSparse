"""SpSM tests: synthetic triangular systems and optional .mtx batch CSV."""

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
import flagsparse.sparse_operations.spsm as fs_spsm_impl

try:
    import cupy as cp
    import cupyx.cusparse as cpx_cusparse
    import cupyx.scipy.sparse as cpx_sparse
except Exception:
    cp = None
    cpx_cusparse = None
    cpx_sparse = None


FORMATS = ("csr", "coo")
VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = [torch.int32]
CSV_VALUE_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
CSV_INDEX_DTYPES = [torch.int32]
WARMUP = 1
ITERS = 1
SPSM_OP_MODES = ["NON", "NON_TRANS"]


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _tol(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-3
    return 1e-12, 1e-10


def _reference_check_threshold(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-6
    return 1e-12


def _reference_max_relative_error(answer, result, dtype):
    if answer is None or result is None:
        return None
    if answer.numel() != result.numel():
        return float("inf")
    if answer.numel() == 0:
        return 0.0
    if dtype in (torch.complex64, torch.complex128):
        answer_cmp = torch.abs(answer)
        result_cmp = torch.abs(result)
        diff = torch.abs(answer_cmp - result_cmp)
    else:
        diff = torch.abs(answer - result)
        result_cmp = torch.abs(result)
    if not bool(torch.isfinite(diff).all().item()) or not bool(torch.isfinite(result_cmp).all().item()):
        return float("inf")
    max_error = torch.max(diff)
    max_result = torch.max(result_cmp)
    if float(max_result.item()) == 0.0:
        return 0.0 if float(max_error.item()) == 0.0 else float("inf")
    return float((max_error / max_result).item())


def _is_fatal_cuda_error(exc):
    msg = str(exc).lower()
    return (
        "illegal memory access" in msg
        or "device-side assert" in msg
        or "unspecified launch failure" in msg
    )


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_ratio(v):
    return "N/A" if v is None else f"{v:.2f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _safe_ratio(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return None
    return other_ms / triton_ms


def _sum_ms(*values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return sum(values)


def _spsm_benchmark_schedule(nnz, n_rhs, value_dtype, fmt="csr"):
    del nnz, n_rhs, value_dtype, fmt
    return int(WARMUP), int(ITERS)


def _csv_export_row_spsm(row):
    return {
        "matrix": row.get("matrix"),
        "value_dtype": row.get("value_dtype"),
        "index_dtype": row.get("index_dtype"),
        "format": row.get("format"),
        "n_rows": row.get("n_rows"),
        "n_cols": row.get("n_cols"),
        "nnz": row.get("nnz"),
        "n_rhs": row.get("n_rhs"),
        "analysis_ms": row.get("analysis_ms"),
        "triton_ms": row.get("triton_ms"),
        "triton_total_ms": row.get("triton_total_ms"),
        "cusparse_ms": row.get("cusparse_ms"),
        "pytorch_ms": row.get("pytorch_ms"),
        "cusparse_speedup_solve": row.get("cusparse_speedup_solve"),
        "pytorch_speedup_solve": row.get("pytorch_speedup_solve"),
        "cusparse_speedup_total": row.get("cusparse_speedup_total"),
        "pytorch_speedup_total": row.get("pytorch_speedup_total"),
        "status": row.get("status"),
        "err_ref": row.get("err_ref"),
        "err_res": row.get("err_res"),
        "err_pt": row.get("err_pt"),
        "err_cu": row.get("err_cu"),
        "pytorch_reason": row.get("pytorch_reason"),
        "error": row.get("error"),
    }


def _parse_csv_tokens(raw):
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_ops_filter(raw):
    tokens = [tok.strip().upper() for tok in _parse_csv_tokens(raw)]
    if not tokens:
        return ["NON"]
    invalid = [tok for tok in tokens if tok not in SPSM_OP_MODES]
    if invalid:
        raise ValueError(
            f"unsupported spsm ops: {invalid}; current SpSM test only supports NON/NON_TRANS"
        )
    normalized = []
    for tok in tokens:
        normalized.append("NON" if tok == "NON_TRANS" else tok)
    return normalized


def _build_triangular_case(n=512, n_rhs=1024, value_dtype=torch.float32):
    device = torch.device("cuda")
    A = torch.tril(torch.randn((n, n), dtype=value_dtype, device=device) * 0.02)
    diag_base_dtype = torch.float32 if value_dtype == torch.complex64 else torch.float64
    diag = (torch.rand((n,), dtype=diag_base_dtype, device=device) + 2.0).to(value_dtype)
    A = A + torch.diag(diag)
    coo = A.to_sparse().coalesce()
    row = coo.indices()[0].to(torch.int64)
    col = coo.indices()[1].to(torch.int64)
    data = coo.values().to(value_dtype)
    _, order = torch.sort(row * n + col)
    row = row[order]
    col = col[order]
    data = data[order]
    nnz_per_row = torch.bincount(row, minlength=n)
    indptr = torch.zeros(n + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    B = torch.randn((n, n_rhs), dtype=value_dtype, device=device).contiguous()
    return data, row, col, indptr, B, (n, n)


def _csr_to_coo(indices, indptr, n_rows):
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=indptr.device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    return row, indices.to(torch.int64)


def _benchmark_pytorch_reference(data, indices, indptr, shape, B):
    try:
        sparse_spsolve = getattr(torch.sparse, "spsolve", None)
        if sparse_spsolve is None:
            raise NotImplementedError("torch.sparse.spsolve is unavailable")
        A_csr = torch.sparse_csr_tensor(
            indptr.to(torch.int64),
            indices.to(torch.int64),
            data,
            size=shape,
            device=data.device,
        )
        if not A_csr.is_cuda:
            raise RuntimeError("torch.sparse.spsolve CUDA path is unavailable")
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True)
        e1 = torch.cuda.Event(True)
        e0.record()
        cols = []
        for bj in torch.unbind(B, dim=1):
            cols.append(sparse_spsolve(A_csr, bj))
        X_ref = torch.stack(cols, dim=1) if cols else B.new_empty(B.shape)
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        return X_ref.to(B.dtype), ms, "gpu_sparse", None
    except Exception as exc:
        if "out of memory" in str(exc).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, "unavailable", f"PyTorch sparse solve unavailable ({exc})"


def _benchmark_cusparse_reference(data, row, col, indptr, B, shape, fmt, warmup, iters):
    if cp is None or cpx_sparse is None or cpx_cusparse is None:
        return None, None, "cusparse unavailable"
    try:
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data.contiguous()))
        B_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(B.contiguous()))
        if fmt == "coo":
            row_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(row.contiguous()))
            col_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.contiguous()))
            A_cp = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
        else:
            idx_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(col.contiguous()))
            ptr_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(indptr.contiguous()))
            A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        A_cp.sum_duplicates()
        for _ in range(warmup):
            _ = cpx_cusparse.spsm(A_cp, B_cp, lower=True, unit_diag=False, transa=False)
        cp.cuda.runtime.deviceSynchronize()
        c0 = cp.cuda.Event()
        c1 = cp.cuda.Event()
        c0.record()
        for _ in range(iters):
            X_cp = cpx_cusparse.spsm(A_cp, B_cp, lower=True, unit_diag=False, transa=False)
        c1.record()
        c1.synchronize()
        ms = cp.cuda.get_elapsed_time(c0, c1) / iters
        X_t = torch.utils.dlpack.from_dlpack(X_cp.toDlpack()).to(B.dtype)
        return X_t, ms, None
    except Exception as exc:
        return None, None, str(exc)


def _apply_csr_to_dense_rhs(data, indices, indptr, X, shape):
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    out = torch.zeros((n_rows, X.shape[1]), dtype=X.dtype, device=X.device)
    out.index_add_(0, row, data[:, None] * X[col])
    return out


def _solution_residual_metrics(data, indices, indptr, shape, X, B, value_dtype):
    atol, rtol = _tol(value_dtype)
    B_recon = _apply_csr_to_dense_rhs(data, indices, indptr, X, shape)
    err = float(torch.max(torch.abs(B_recon - B)).item()) if B.numel() else 0.0
    ok = torch.allclose(B_recon, B, atol=atol, rtol=rtol)
    return err, ok


def _benchmark_flagsparse(call, warmup, iters):
    X = None
    for _ in range(warmup):
        X = call()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        X = call()
    e1.record()
    torch.cuda.synchronize()
    return X, e0.elapsed_time(e1) / iters


def _benchmark_flagsparse_spsm_csr_split(data, indices, indptr, B, shape):
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), B.shape[1], data.dtype, fmt="csr"
    )
    analysis_ms = fs_spsm_impl._analyze_spsm_csr(
        data,
        indices,
        indptr,
        B,
        shape,
        lower=True,
        unit_diagonal=False,
        clear_cache=True,
        return_time=True,
    )
    X, solve_ms = _benchmark_flagsparse(
        lambda: fs.flagsparse_spsm_csr(
            data,
            indices,
            indptr,
            B,
            shape,
            lower=True,
            unit_diagonal=False,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        ),
        warmup,
        iters,
    )
    return X, analysis_ms, solve_ms


def _benchmark_flagsparse_spsm_coo_split(data, row, col, B, shape):
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), B.shape[1], data.dtype, fmt="coo"
    )
    analysis_ms = fs_spsm_impl._analyze_spsm_coo(
        data,
        row,
        col,
        B,
        shape,
        lower=True,
        unit_diagonal=False,
        clear_cache=True,
        return_time=True,
    )
    X, solve_ms = _benchmark_flagsparse(
        lambda: fs.flagsparse_spsm_coo(
            data,
            row,
            col,
            B,
            shape,
            lower=True,
            unit_diagonal=False,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        ),
        warmup,
        iters,
    )
    return X, analysis_ms, solve_ms


def _load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = []
    header_info = None
    mm_field = "real"
    mm_symmetry = "general"
    for line in lines:
        line = line.strip()
        if line.startswith("%%MatrixMarket"):
            parts = line.split()
            if len(parts) >= 5:
                mm_field = parts[3].lower()
                mm_symmetry = parts[4].lower()
            continue
        if line.startswith("%"):
            continue
        if not header_info and line:
            parts = line.split()
            n_rows = int(parts[0])
            n_cols = int(parts[1])
            nnz = int(parts[2]) if len(parts) > 2 else 0
            header_info = (n_rows, n_cols, nnz)
            continue
        if line:
            data_lines.append(line)
    if header_info is None:
        raise ValueError(f"Cannot parse .mtx header: {file_path}")
    n_rows, n_cols, nnz = header_info
    if n_rows != n_cols:
        raise ValueError("SpSM requires square matrices")

    row_maps = [dict() for _ in range(n_rows)]

    def _accum(r, c, v):
        row = row_maps[r]
        row[c] = row.get(c, 0.0) + v

    for line in data_lines[:nnz]:
        parts = line.split()
        if len(parts) < 2:
            continue
        r = int(parts[0]) - 1
        c = int(parts[1]) - 1
        if mm_field == "complex":
            if len(parts) < 4:
                raise ValueError("MatrixMarket complex entry requires real and imag parts")
            v = complex(float(parts[2]), float(parts[3]))
        elif len(parts) >= 3:
            v = float(parts[2])
        elif mm_field == "pattern":
            v = 1.0
        else:
            continue
        _accum(r, c, v)
        if mm_symmetry in ("symmetric", "hermitian") and r != c:
            _accum(c, r, v)
        elif mm_symmetry == "skew-symmetric" and r != c:
            _accum(c, r, -v)

    for r in range(n_rows):
        row = row_maps[r]
        lower_row = {}
        off_abs_sum = 0.0
        for c, v in row.items():
            if c < r:
                lower_row[c] = lower_row.get(c, 0.0) + v
                off_abs_sum += abs(v)
        lower_row[r] = off_abs_sum + 2.0
        row_maps[r] = lower_row

    cols_s = []
    vals_s = []
    indptr_list = [0]
    for r in range(n_rows):
        row = row_maps[r]
        for c in sorted(row.keys()):
            cols_s.append(c)
            vals_s.append(row[c])
        indptr_list.append(len(cols_s))

    data = torch.tensor(vals_s, dtype=dtype, device=device)
    indices = torch.tensor(cols_s, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def _run_one_spsm_case(data, indices, indptr, shape, value_dtype, index_dtype, n_rhs, fmt):
    n_rows = int(shape[0])
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=data.device).contiguous()
    atol, rtol = _tol(value_dtype)
    row, col = _csr_to_coo(indices, indptr, n_rows)
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), n_rhs, value_dtype, fmt=fmt
    )

    if fmt == "csr":
        X_fs, analysis_ms, solve_ms = _benchmark_flagsparse_spsm_csr_split(
            data,
            indices.to(index_dtype),
            indptr.to(index_dtype),
            B,
            shape,
        )
    else:
        X_fs, analysis_ms, solve_ms = _benchmark_flagsparse_spsm_coo_split(
            data,
            row.to(index_dtype),
            col.to(index_dtype),
            B,
            shape,
        )
    X_cu, cusparse_ms, _cusparse_reason = _benchmark_cusparse_reference(
        data, row, col, indptr, B, shape, fmt, warmup, iters
    )
    X_pt, pytorch_ms, _pt_backend, pytorch_reason = _benchmark_pytorch_reference(
        data, indices, indptr, shape, B
    )

    err_cu = None
    ok_cu = None
    rel_cu = None
    if X_cu is not None:
        err_cu = float(torch.max(torch.abs(X_fs - X_cu)).item()) if X_fs.numel() else 0.0
        rel_cu = _reference_max_relative_error(X_cu, X_fs, value_dtype)
        ok_cu = rel_cu <= _reference_check_threshold(value_dtype)

    err_pt = None
    ok_pt = None
    rel_pt = None
    if X_pt is not None:
        err_pt = float(torch.max(torch.abs(X_fs - X_pt)).item()) if X_fs.numel() else 0.0
        rel_pt = _reference_max_relative_error(X_pt, X_fs, value_dtype)
        ok_pt = rel_pt <= _reference_check_threshold(value_dtype)

    err_res, ok_res = _solution_residual_metrics(data, indices, indptr, shape, X_fs, B, value_dtype)
    ref_errors = [v for v in (err_pt, err_cu) if v is not None]
    err_ref = min(ref_errors) if ref_errors else None

    if ok_cu is not None:
        status = "PASS" if ok_cu else "FAIL"
    elif ok_pt is not None:
        status = "PASS" if ok_pt else "FAIL"
    elif X_pt is None and X_cu is None:
        status = "REF_FAIL"
    else:
        status = "FAIL"

    return {
        "format": fmt,
        "n_rows": n_rows,
        "n_cols": int(shape[1]),
        "nnz": int(data.numel()),
        "n_rhs": int(n_rhs),
        "analysis_ms": analysis_ms,
        "triton_ms": solve_ms,
        "triton_total_ms": _sum_ms(analysis_ms, solve_ms),
        "cusparse_ms": cusparse_ms,
        "pytorch_ms": pytorch_ms,
        "cusparse_speedup_solve": _safe_ratio(cusparse_ms, solve_ms),
        "pytorch_speedup_solve": _safe_ratio(pytorch_ms, solve_ms),
        "cusparse_speedup_total": _safe_ratio(cusparse_ms, _sum_ms(analysis_ms, solve_ms)),
        "pytorch_speedup_total": _safe_ratio(pytorch_ms, _sum_ms(analysis_ms, solve_ms)),
        "status": status,
        "err_ref": err_ref,
        "err_res": err_res,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "pytorch_reason": pytorch_reason,
        "error": None,
    }


def run_spsm_synthetic_all(n=512, n_rhs=1024):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    total = 0
    failed = 0
    print("=" * 160)
    print("FLAGSPARSE SpSM synthetic test")
    print("=" * 160)
    print(
        "Baselines: cuSPARSE sparse triangular solve + PyTorch official sparse solve "
        "(PyTorch aggregates one torch.sparse.spsolve call per RHS column; "
        "cuSPARSE solves the full matrix RHS in one interface call)."
    )
    print(
        f"{'Fmt':>5} {'dtype':>9} {'index':>7} {'N':>6} {'RHS':>6} {'NNZ':>10} "
        f"{'FS.analysis':>11} {'FS.solve':>10} {'FS.total':>10} "
        f"{'CU.total':>10} {'PT.total':>10} {'CU.spdS':>10} {'PT.spdS':>10} {'CU.spdT':>10} {'PT.spdT':>10} "
        f"{'Status':>10} {'Err(Ref)':>12} {'Err(Res)':>12} {'Err(PT)':>12} {'Err(CU)':>12}"
    )
    print("-" * 160)

    for fmt in FORMATS:
        for value_dtype in VALUE_DTYPES:
            for index_dtype in INDEX_DTYPES:
                data, row, col, indptr, _B, shape = _build_triangular_case(
                    n=n,
                    n_rhs=n_rhs,
                    value_dtype=value_dtype,
                )
                one = _run_one_spsm_case(
                    data,
                    col,
                    indptr,
                    shape,
                    value_dtype,
                    index_dtype,
                    n_rhs,
                    fmt,
                )
                total += 1
                if one["status"] != "PASS":
                    failed += 1
                print(
                    f"{fmt:>5} {_dtype_name(value_dtype):>9} {_dtype_name(index_dtype):>7} "
                    f"{shape[0]:>6} {n_rhs:>6} {one['nnz']:>10} "
                    f"{_fmt_ms(one['analysis_ms']):>11} {_fmt_ms(one['triton_ms']):>10} {_fmt_ms(one['triton_total_ms']):>10} "
                    f"{_fmt_ms(one['cusparse_ms']):>10} {_fmt_ms(one['pytorch_ms']):>10} "
                    f"{_fmt_ratio(one['cusparse_speedup_solve']):>10} {_fmt_ratio(one['pytorch_speedup_solve']):>10} "
                    f"{_fmt_ratio(one['cusparse_speedup_total']):>10} {_fmt_ratio(one['pytorch_speedup_total']):>10} "
                    f"{one['status']:>10} {_fmt_err(one['err_ref']):>12} {_fmt_err(one['err_res']):>12} "
                    f"{_fmt_err(one['err_pt']):>12} {_fmt_err(one['err_cu']):>12}"
                )
                if one["status"] in ("FAIL", "REF_FAIL"):
                    if one["pytorch_reason"]:
                        print(f"  NOTE: {one['pytorch_reason']}")
    print("-" * 160)
    print(f"Total cases: {total}  Failed: {failed}")
    print("=" * 160)


def run_all_dtypes_spsm_csv(mtx_paths, csv_path, use_coo=False, n_rhs=1024):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    fmt = "coo" if use_coo else "csr"

    print("=" * 176)
    print(
        f"FLAGSPARSE SpSM .mtx batch ({fmt.upper()}) | "
        "baselines: cuSPARSE sparse triangular solve + PyTorch official sparse solve "
        "(PyTorch aggregates one torch.sparse.spsolve call per RHS column; "
        "cuSPARSE solves the full matrix RHS in one interface call)"
    )
    print("=" * 176)
    print(
        f"Benchmark schedule: warmup={WARMUP}, iter={ITERS} "
        "(Library-main style defaults; override with --warmup/--iters)"
    )
    print(
        "PT.total is the aggregated time of one torch.sparse.spsolve call per RHS column; "
        "CU.total is one official sparse triangular solve call on the full matrix RHS. "
        "PT.spdS and CU.spdS compare against FS.solve; PT.spdT and CU.spdT compare against FS.total."
    )
    print(
        f"{'Matrix':<28} {'dtype':>9} {'index':>7} {'N':>7} {'RHS':>6} {'NNZ':>10} "
        f"{'FS.analysis':>11} {'FS.solve':>10} {'FS.total':>10} "
        f"{'CU.total':>10} {'PT.total':>10} {'CU.spdS':>10} {'PT.spdS':>10} {'CU.spdT':>10} {'PT.spdT':>10} "
        f"{'Status':>10} {'Err(Ref)':>12} {'Err(Res)':>12} {'Err(PT)':>12} {'Err(CU)':>12}"
    )
    print("-" * 176)

    for value_dtype in CSV_VALUE_DTYPES:
        for index_dtype in CSV_INDEX_DTYPES:
            for path in mtx_paths:
                base = {
                    "matrix": os.path.basename(path),
                    "value_dtype": _dtype_name(value_dtype),
                    "index_dtype": _dtype_name(index_dtype),
                }
                try:
                    print(
                        f"RUNNING: {base['matrix']} | dtype={base['value_dtype']} | "
                        f"index={base['index_dtype']} | fmt={fmt}",
                        flush=True,
                    )
                    data, indices, indptr, shape = _load_mtx_to_csr_torch(
                        path,
                        dtype=value_dtype,
                        device=device,
                    )
                    row = _run_one_spsm_case(
                        data,
                        indices,
                        indptr,
                        shape,
                        value_dtype,
                        index_dtype,
                        n_rhs,
                        fmt,
                    )
                    row = {**base, **row}
                    rows_out.append(row)
                    short = base["matrix"][:27] + ("…" if len(base["matrix"]) > 27 else "")
                    print(
                        f"{short:<28} {base['value_dtype']:>9} {base['index_dtype']:>7} "
                        f"{row['n_rows']:>7} {row['n_rhs']:>6} {row['nnz']:>10} "
                        f"{_fmt_ms(row['analysis_ms']):>11} {_fmt_ms(row['triton_ms']):>10} {_fmt_ms(row['triton_total_ms']):>10} "
                        f"{_fmt_ms(row['cusparse_ms']):>10} {_fmt_ms(row['pytorch_ms']):>10} "
                        f"{_fmt_ratio(row['cusparse_speedup_solve']):>10} {_fmt_ratio(row['pytorch_speedup_solve']):>10} "
                        f"{_fmt_ratio(row['cusparse_speedup_total']):>10} {_fmt_ratio(row['pytorch_speedup_total']):>10} "
                        f"{row['status']:>10} {_fmt_err(row['err_ref']):>12} {_fmt_err(row['err_res']):>12} "
                        f"{_fmt_err(row['err_pt']):>12} {_fmt_err(row['err_cu']):>12}"
                    )
                    if row["status"] in ("FAIL", "REF_FAIL"):
                        if row["pytorch_reason"]:
                            print(f"  NOTE: {row['pytorch_reason']}")
                except Exception as exc:
                    err_msg = str(exc)
                    if _is_fatal_cuda_error(exc):
                        print(
                            f"  FATAL CUDA ERROR: {exc}\n"
                            "  CUDA context is no longer reliable. Restart the Python/Singularity session "
                            "and rerun the single failing matrix with CUDA_LAUNCH_BLOCKING=1."
                        )
                        raise
                    status = "SKIP" if "SpSM requires square matrices" in err_msg else "ERROR"
                    row = {
                        **base,
                        "format": fmt,
                        "n_rows": "ERR",
                        "n_cols": "ERR",
                        "nnz": "ERR",
                        "n_rhs": int(n_rhs),
                        "analysis_ms": None,
                        "triton_ms": None,
                        "triton_total_ms": None,
                        "cusparse_ms": None,
                        "pytorch_ms": None,
                        "cusparse_speedup_solve": None,
                        "pytorch_speedup_solve": None,
                        "cusparse_speedup_total": None,
                        "pytorch_speedup_total": None,
                        "status": status,
                        "err_ref": None,
                        "err_res": None,
                        "err_pt": None,
                        "err_cu": None,
                        "pytorch_reason": None,
                        "error": err_msg,
                    }
                    rows_out.append(row)
                    short = base["matrix"][:27] + ("…" if len(base["matrix"]) > 27 else "")
                    print(
                        f"{short:<28} {base['value_dtype']:>9} {base['index_dtype']:>7} "
                        f"{'ERR':>7} {int(n_rhs):>6} {'ERR':>10} "
                        f"{_fmt_ms(None):>11} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} "
                        f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {status:>10} "
                        f"{_fmt_err(None):>12} {_fmt_err(None):>12} {_fmt_err(None):>12} {_fmt_err(None):>12}"
                    )
                    print(f"  {status}: {exc}")

    print("-" * 176)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "format",
        "n_rows",
        "n_cols",
        "nnz",
        "n_rhs",
        "analysis_ms",
        "triton_ms",
        "triton_total_ms",
        "cusparse_ms",
        "pytorch_ms",
        "cusparse_speedup_solve",
        "pytorch_speedup_solve",
        "cusparse_speedup_total",
        "pytorch_speedup_total",
        "status",
        "err_ref",
        "err_res",
        "err_pt",
        "err_cu",
        "pytorch_reason",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_out:
            w.writerow({k: ("" if v is None else v) for k, v in _csv_export_row_spsm(row).items()})
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def main():
    global WARMUP, ITERS
    parser = argparse.ArgumentParser(
        description="SpSM test: synthetic triangular systems and optional .mtx batch CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic triangular tests")
    parser.add_argument("--n", type=int, default=512, help="matrix size (synthetic)")
    parser.add_argument(
        "--rhs",
        type=int,
        default=1024,
        help="number of RHS columns (default: 1024, matching all-in-one SpSM)",
    )
    parser.add_argument("--csv-csr", type=str, default=None, metavar="FILE")
    parser.add_argument("--csv-coo", type=str, default=None, metavar="FILE")
    parser.add_argument(
        "--ops",
        type=str,
        default="NON",
        help="comma-separated op(A) modes; currently only NON/NON_TRANS is supported",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="Benchmark warmup iterations (Library-main style default: 1)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=ITERS,
        help="Benchmark timed iterations (Library-main style default: 1)",
    )
    args = parser.parse_args()
    WARMUP = max(0, int(args.warmup))
    ITERS = max(1, int(args.iters))

    ops = _parse_ops_filter(args.ops)
    if any(op != "NON" for op in ops):
        raise ValueError("SpSM test currently supports only --ops NON/NON_TRANS")

    ops = _parse_ops_filter(args.ops)
    if any(op != "NON" for op in ops):
        raise ValueError("SpSM test currently supports only --ops NON/NON_TRANS")

    if args.synthetic:
        run_spsm_synthetic_all(n=args.n, n_rhs=args.rhs)
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))

    if args.csv_csr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-csr")
            return
        run_all_dtypes_spsm_csv(paths, args.csv_csr, use_coo=False, n_rhs=args.rhs)
        return

    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        run_all_dtypes_spsm_csv(paths, args.csv_coo, use_coo=True, n_rhs=args.rhs)
        return

    print("Use --synthetic, --csv-csr, or --csv-coo to run SpSM tests.")


if __name__ == "__main__":
    main()
