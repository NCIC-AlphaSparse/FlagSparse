"""SpSV tests: synthetic triangular systems and optional .mtx (CSR/COO)."""

import argparse
import csv
import glob
import hashlib
import os
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
import flagsparse.sparse_operations.spsv as fs_spsv_impl

try:
    import cupy as cp
    import cupyx.scipy.sparse as cpx_sparse
    from cupyx.scipy.sparse.linalg import spsolve_triangular as cpx_spsolve_triangular
except Exception:
    cp = None
    cpx_sparse = None
    cpx_spsolve_triangular = None

VALUE_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
INDEX_DTYPES = [torch.int32, torch.int64]
TEST_SIZES = [256, 512, 1024, 2048]
WARMUP = 1
ITERS = 1

SPSV_TRIANGULAR_DIAG_DOMINANCE = 4.0
# CSR 完整组合覆盖（在原 csv-csr 逻辑外新增，不影响原入口）
CSR_FULL_VALUE_DTYPES = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
]
CSR_FULL_INDEX_DTYPES = [torch.int32, torch.int64]
SPSV_OP_MODES = ["NON", "TRANS", "CONJ"]


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


VALUE_DTYPE_NAME_MAP = {
    _dtype_name(dtype): dtype for dtype in CSR_FULL_VALUE_DTYPES
}
VALUE_DTYPE_NAME_MAP.update({
    "float": torch.float32,
    "double": torch.float64,
})
INDEX_DTYPE_NAME_MAP = {
    _dtype_name(dtype): dtype for dtype in CSR_FULL_INDEX_DTYPES
}
SPSV_ALG_NUM_TO_SOLVE_KIND = {
    1: "csr_cw",
    2: "csr_cw_levelschd",
    3: "csr_nnz_balance",
    8: "csr_nnz_balance",
}


def _parse_csv_tokens(raw):
    return [tok.strip() for tok in str(raw).split(",") if tok.strip()]


def _parse_value_dtypes_filter(raw):
    tokens = [tok.lower() for tok in _parse_csv_tokens(raw)]
    invalid = [tok for tok in tokens if tok not in VALUE_DTYPE_NAME_MAP]
    if invalid:
        raise ValueError(f"unsupported value dtypes: {invalid}")
    return [VALUE_DTYPE_NAME_MAP[tok] for tok in tokens]


def _parse_index_dtypes_filter(raw):
    tokens = [tok.lower() for tok in _parse_csv_tokens(raw)]
    invalid = [tok for tok in tokens if tok not in INDEX_DTYPE_NAME_MAP]
    if invalid:
        raise ValueError(f"unsupported index dtypes: {invalid}")
    return [INDEX_DTYPE_NAME_MAP[tok] for tok in tokens]


def _parse_op_modes_filter(raw):
    tokens = [tok.upper() for tok in _parse_csv_tokens(raw)]
    invalid = [tok for tok in tokens if tok not in SPSV_OP_MODES]
    if invalid:
        raise ValueError(f"unsupported ops: {invalid}")
    return tokens


def _parse_alg_num(raw):
    value = int(raw)
    if value not in SPSV_ALG_NUM_TO_SOLVE_KIND:
        raise ValueError(
            "unsupported alg_num: "
            f"{value}. Supported values: {sorted(SPSV_ALG_NUM_TO_SOLVE_KIND)}"
        )
    return value


def _solve_kind_from_alg_num(alg_num):
    if alg_num is None:
        return None
    return SPSV_ALG_NUM_TO_SOLVE_KIND[int(alg_num)]


def _alg_label(alg_num):
    return "AUTO" if alg_num is None else f"ALG{int(alg_num)}"


def _alg_num_supports_case(alg_num, fmt, op_mode, lower, value_dtype):
    if alg_num is None:
        return True
    alg_num = int(alg_num)
    if alg_num == 1:
        return True
    if alg_num in (2, 3, 8):
        return (
            fmt in ("CSR", "COO")
            and op_mode == "NON"
            and bool(lower)
        )
    return False


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_speedup(other_ms, triton_ms):
    if other_ms is None or triton_ms is None or triton_ms <= 0:
        return "N/A"
    return f"{other_ms / triton_ms:.2f}"


def _fmt_ratio(v):
    return "N/A" if v is None else f"{v:.2f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _safe_ratio(other_ms, base_ms):
    if other_ms is None or base_ms is None or base_ms <= 0:
        return None
    return other_ms / base_ms


def _sum_ms(*values):
    if any(v is None for v in values):
        return None
    return sum(values)


def _spsv_benchmark_schedule(nnz, op_mode, value_dtype, fmt="CSR"):
    del nnz, op_mode, value_dtype, fmt
    return int(WARMUP), int(ITERS)


def _status_str(ok_flag, has_value):
    if ok_flag:
        return "PASS"
    return "FAIL" if has_value else "N/A"


def _csv_export_row_spsv(row):
    return {
        "matrix": row.get("matrix"),
        "value_dtype": row.get("value_dtype"),
        "index_dtype": row.get("index_dtype"),
        "opA": row.get("opA"),
        "n_rows": row.get("n_rows"),
        "n_cols": row.get("n_cols"),
        "nnz": row.get("nnz"),
        "analysis_ms": row.get("analysis_ms"),
        "triton_ms": row.get("triton_ms"),
        "triton_total_ms": row.get("triton_total_ms"),
        "cusparse_ms": row.get("cusparse_ms"),
        "pytorch_ms": row.get("pytorch_ms"),
        "cusparse_speedup_solve": row.get("cusparse_speedup_solve"),
        "pytorch_speedup_solve": row.get("pytorch_speedup_solve"),
        "cusparse_speedup_total": row.get("cusparse_speedup_total"),
        "pytorch_speedup_total": row.get("pytorch_speedup_total"),
        "pt_status": row.get("pt_status"),
        "cu_status": row.get("cu_status"),
        "status": row.get("status"),
        "err_ref": row.get("err_ref"),
        "err_res": row.get("err_res"),
        "err_pt": row.get("err_pt"),
        "err_cu": row.get("err_cu"),
        "pytorch_reason": row.get("pytorch_reason"),
        "error": row.get("error"),
    }


def _tol_for_dtype(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4, 1e-2
    return 1e-12, 1e-10


def _stable_case_seed(*parts):
    raw = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63)


def _generator_for_seed(seed):
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(int(seed))
    return gen


def _randn_by_dtype(n, dtype, device, generator=None):
    if dtype in (torch.float32, torch.float64):
        return torch.randn(n, dtype=dtype, device=device, generator=generator)
    base = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(n, dtype=base, device=device, generator=generator)
    imag = torch.randn(n, dtype=base, device=device, generator=generator)
    return torch.complex(real, imag)


def _tensor_from_scalar_values(values, dtype, device):
    return torch.tensor(values, dtype=dtype, device=device)


def _matrix_market_value(parts, mm_field):
    if mm_field == "complex":
        if len(parts) < 4:
            raise ValueError("MatrixMarket complex entry requires real and imag parts")
        return complex(float(parts[2]), float(parts[3]))
    if len(parts) >= 3:
        return float(parts[2])
    if mm_field == "pattern":
        return 1.0
    raise ValueError("MatrixMarket entry is missing a numeric value")


def _build_csr_tensor_for_op(data, indices, indptr, shape, op_mode):
    if op_mode == "TRANS":
        data_eff, indices_eff, indptr_eff = _csr_transpose(
            data,
            indices.to(torch.int64),
            indptr.to(torch.int64),
            shape,
            conjugate=False,
        )
    elif op_mode == "CONJ":
        data_eff, indices_eff, indptr_eff = _csr_transpose(
            data,
            indices.to(torch.int64),
            indptr.to(torch.int64),
            shape,
            conjugate=True,
        )
    else:
        data_eff = data
        indices_eff = indices.to(torch.int64)
        indptr_eff = indptr.to(torch.int64)
    return torch.sparse_csr_tensor(
        indptr_eff,
        indices_eff,
        data_eff,
        size=shape,
        device=data.device,
    )


def _benchmark_pytorch_reference(data, indices, indptr, shape, b, *, lower, op_mode):
    try:
        sparse_spsolve = getattr(torch.sparse, "spsolve", None)
        if sparse_spsolve is None:
            raise NotImplementedError("torch.sparse.spsolve is unavailable")
        A_csr = _build_csr_tensor_for_op(data, indices, indptr, shape, op_mode)
        if not A_csr.is_cuda:
            raise RuntimeError("torch.sparse.spsolve CUDA path is unavailable")
        torch.cuda.synchronize()
        e0 = torch.cuda.Event(True)
        e1 = torch.cuda.Event(True)
        e0.record()
        x_ref = sparse_spsolve(A_csr, b)
        e1.record()
        torch.cuda.synchronize()
        ms = e0.elapsed_time(e1)
        return x_ref.to(b.dtype), ms, "gpu_sparse", None
    except Exception as sparse_err:
        if "out of memory" in str(sparse_err).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None, "unavailable", f"PyTorch sparse solve unavailable ({sparse_err})"


def _cupy_ref_inputs(data, b):
    return data, b


def _supported_csr_full_ops(value_dtype, index_dtype):
    if value_dtype not in CSR_FULL_VALUE_DTYPES:
        return []
    if index_dtype == torch.int32:
        return ["NON", "TRANS", "CONJ"]
    if index_dtype == torch.int64:
        return ["NON", "TRANS", "CONJ"]
    return []


def _build_random_triangular_csr(n, value_dtype, index_dtype, device, lower=True):
    """Build a well-conditioned triangular CSR for real and complex dtypes."""
    max_bandwidth = max(4, min(n, 16))
    rows_host = []
    cols_host = []
    vals_host = []
    row_off_abs = [0.0] * n
    col_off_abs = [0.0] * n
    if value_dtype == torch.float32:
        base_real_dtype = torch.float32
    elif value_dtype == torch.float64:
        base_real_dtype = torch.float64
    elif value_dtype == torch.complex64:
        base_real_dtype = torch.float32
    else:
        base_real_dtype = torch.float64

    for i in range(n):
        if lower:
            cand_cols = list(range(0, i + 1))
        else:
            cand_cols = list(range(i, n))
        if not cand_cols:
            cand_cols = [i]
        diag_col = i
        off_cand = [c for c in cand_cols if c != diag_col]
        k_off = min(len(off_cand), max_bandwidth - 1)
        if k_off > 0:
            perm = torch.randperm(len(off_cand))[:k_off].tolist()
            off_cols = [off_cand[j] for j in perm]
        else:
            off_cols = []
        if value_dtype in (torch.complex64, torch.complex128):
            off_vals = torch.complex(
                torch.randn(len(off_cols), dtype=base_real_dtype, device=device).mul_(0.01),
                torch.randn(len(off_cols), dtype=base_real_dtype, device=device).mul_(0.01),
            )
            off_vals_host = [complex(v) for v in off_vals.cpu().tolist()]
        else:
            off_vals = torch.randn(len(off_cols), dtype=base_real_dtype, device=device).mul_(0.01)
            off_vals_host = off_vals.cpu().tolist()
        for c, v in zip(off_cols, off_vals_host):
            rows_host.append(i)
            cols_host.append(int(c))
            vals_host.append(v)
            mag = abs(v)
            row_off_abs[i] += mag
            col_off_abs[int(c)] += mag

    for i in range(n):
        diag_mag = (
            SPSV_TRIANGULAR_DIAG_DOMINANCE * max(row_off_abs[i], col_off_abs[i]) + 1.0
        )
        diag_val = (
            complex(diag_mag, 0.0)
            if value_dtype in (torch.complex64, torch.complex128)
            else diag_mag
        )
        rows_host.append(i)
        cols_host.append(i)
        vals_host.append(diag_val)

    rows_t = torch.tensor(rows_host, dtype=torch.int64, device=device)
    cols_t = torch.tensor(cols_host, dtype=torch.int64, device=device)
    vals_t = torch.tensor(vals_host, dtype=value_dtype, device=device)
    order = torch.argsort(rows_t * max(1, n) + cols_t)
    rows_t = rows_t[order]
    cols_t = cols_t[order]
    vals_t = vals_t[order]
    nnz_per_row = torch.bincount(rows_t, minlength=n)
    indptr = torch.zeros(n + 1, dtype=torch.int64, device=device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    indices = cols_t.to(index_dtype)
    return vals_t, indices, indptr, (n, n)

def _csr_to_coo(data, indices, indptr, shape, index_dtype=torch.int64):
    n_rows = int(shape[0])
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=index_dtype),
        indptr[1:] - indptr[:-1],
    )
    col = indices.to(index_dtype)
    return data, row, col


def _csr_transpose(data, indices, indptr, shape, conjugate=False):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() == 0:
        return (
            data,
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.zeros(n_cols + 1, dtype=torch.int64, device=data.device),
        )

    row, col = _csr_to_coo(data, indices, indptr, shape)[1:]
    row_t = col
    col_t = row
    key = row_t * max(1, n_rows) + col_t
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)

    row_t = row_t[order]
    col_t = col_t[order]
    data_eff = data.conj() if conjugate and torch.is_complex(data) else data
    data_t = data_eff[order]
    nnz_per_row = torch.bincount(row_t, minlength=n_cols)
    indptr_t = torch.zeros(n_cols + 1, dtype=torch.int64, device=data.device)
    indptr_t[1:] = torch.cumsum(nnz_per_row, dim=0)
    return data_t, col_t.to(torch.int64), indptr_t


def _load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None, lower=True):
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
        raise ValueError("SpSV requires square matrices")
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
        v = _matrix_market_value(parts, mm_field)
        _accum(r, c, v)
        if mm_symmetry == "symmetric" and r != c:
            _accum(c, r, v)
        elif mm_symmetry == "hermitian" and r != c:
            _accum(c, r, v.conjugate() if isinstance(v, complex) else v)
        elif mm_symmetry == "skew-symmetric" and r != c:
            _accum(c, r, -v)

    tri_rows = [dict() for _ in range(n_rows)]
    row_off_abs = [0.0] * n_rows
    col_off_abs = [0.0] * n_cols
    for r in range(n_rows):
        for c, v in row_maps[r].items():
            keep = c < r if lower else c > r
            if keep:
                tri_rows[r][c] = tri_rows[r].get(c, 0.0) + v

    for r, row in enumerate(tri_rows):
        for c, v in row.items():
            mag = abs(v)
            row_off_abs[r] += mag
            col_off_abs[c] += mag

    for r in range(n_rows):
        # Make the generated triangular system stable for both A and op(A).
        tri_rows[r][r] = (
            SPSV_TRIANGULAR_DIAG_DOMINANCE * max(row_off_abs[r], col_off_abs[r]) + 1.0
        )
    row_maps = tri_rows

    cols_s = []
    vals_s = []
    indptr_list = [0]
    for r in range(n_rows):
        row = row_maps[r]
        for c in sorted(row.keys()):
            cols_s.append(c)
            vals_s.append(row[c])
        indptr_list.append(len(cols_s))
    data = _tensor_from_scalar_values(vals_s, dtype, device)
    indices = torch.tensor(cols_s, dtype=torch.int64, device=device)
    indptr = torch.tensor(indptr_list, dtype=torch.int64, device=device)
    return data, indices, indptr, (n_rows, n_cols)


def _coo_inputs_for_csv(data, indices, indptr, shape, index_dtype=torch.int64):
    """Build normal COO inputs from canonical CSR while preserving matrix metadata."""
    data_c, row_c, col_c = _csr_to_coo(
        data, indices, indptr, shape, index_dtype=index_dtype
    )
    return data_c, row_c, col_c


def _random_rhs_for_spsv(shape, value_dtype, device, op_mode="NON", seed=None):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    rhs_size = n_rows if op_mode == "NON" else n_cols
    if seed is None:
        return _randn_by_dtype(rhs_size, value_dtype, device)
    rhs = _randn_by_dtype(
        rhs_size,
        value_dtype,
        torch.device("cpu"),
        generator=_generator_for_seed(seed),
    )
    return rhs.to(device)


def _apply_csr_op(data, indices, indptr, x, shape, op_mode):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr.to(torch.int64)[1:] - indptr.to(torch.int64)[:-1],
    )
    col = indices.to(torch.int64)
    if op_mode == "NON":
        b = torch.zeros(n_rows, dtype=data.dtype, device=data.device)
        b.scatter_add_(0, row, data * x[col])
        return b
    if op_mode == "TRANS":
        b = torch.zeros(n_cols, dtype=data.dtype, device=data.device)
        b.scatter_add_(0, col, data * x[row])
        return b
    if op_mode == "CONJ":
        b = torch.zeros(n_cols, dtype=data.dtype, device=data.device)
        data_eff = data.conj() if torch.is_complex(data) else data
        b.scatter_add_(0, col, data_eff * x[row])
        return b
    raise ValueError("op_mode must be 'NON', 'TRANS', or 'CONJ'")


def _solution_residual_metrics(data, indices, indptr, shape, x, b, value_dtype, op_mode):
    atol, rtol = _tol_for_dtype(value_dtype)
    b_recon = _apply_csr_op(data, indices, indptr, x, shape, op_mode)
    err_res = (
        float(torch.max(torch.abs(b_recon - b)).item())
        if b.numel() > 0
        else 0.0
    )
    ok_res = torch.allclose(b_recon, b, atol=atol, rtol=rtol)
    return err_res, ok_res


def _benchmark_flagsparse(call, *, warmup=WARMUP, iters=ITERS):
    x = None
    for _ in range(warmup):
        x = call()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(True)
    e1 = torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        x = call()
    e1.record()
    torch.cuda.synchronize()
    return x, e0.elapsed_time(e1) / iters


def _benchmark_flagsparse_spsv_csr(
    data,
    indices,
    indptr,
    b,
    shape,
    *,
    lower=True,
    transpose=False,
    solve_kind=None,
    warmup=WARMUP,
    iters=ITERS,
):
    return _benchmark_flagsparse(
        lambda: fs.flagsparse_spsv_csr(
            data,
            indices,
            indptr,
            b,
            shape,
            lower=lower,
            transpose=transpose,
            solve_kind=solve_kind,
        ),
        warmup=warmup,
        iters=iters,
    )


def _benchmark_flagsparse_spsv_csr_split(
    data,
    indices,
    indptr,
    b,
    shape,
    *,
    lower=True,
    transpose=False,
    solve_kind=None,
):
    op_mode = fs_spsv_impl._normalize_spsv_transpose_mode(transpose)
    warmup, iters = _spsv_benchmark_schedule(
        int(data.numel()),
        "NON" if op_mode == "N" else ("TRANS" if op_mode == "T" else "CONJ"),
        data.dtype,
        fmt="CSR",
    )
    analysis_ms = fs_spsv_impl._analyze_spsv_csr(
        data,
        indices,
        indptr,
        b,
        shape,
        lower=lower,
        transpose=transpose,
        solve_kind=solve_kind,
        clear_cache=True,
        return_time=True,
    )
    x, solve_ms = _benchmark_flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        shape,
        lower=lower,
        transpose=transpose,
        solve_kind=solve_kind,
        warmup=warmup,
        iters=iters,
    )
    return x, analysis_ms, solve_ms


def _benchmark_flagsparse_spsv_coo_split(
    data,
    row,
    col,
    b,
    shape,
    *,
    lower=True,
    transpose=False,
    solve_kind=None,
):
    data, input_index_dtype, row64, col64, b, n_rows, n_cols = fs_spsv_impl._prepare_spsv_coo_inputs(
        data, row, col, b, shape
    )
    trans_mode = fs_spsv_impl._normalize_spsv_transpose_mode(transpose)
    if trans_mode == "N":
        fs_spsv_impl._validate_spsv_non_trans_combo(data.dtype, input_index_dtype, "COO")
    else:
        fs_spsv_impl._validate_spsv_trans_combo(data.dtype, input_index_dtype, "COO")
    data_csr, indices_csr, indptr_csr = fs_spsv_impl._coo2csr_for_spsv(
        data, row64, col64, n_rows, assume_ordered=False
    )
    warmup, iters = _spsv_benchmark_schedule(
        int(data_csr.numel()),
        "NON" if trans_mode == "N" else ("TRANS" if trans_mode == "T" else "CONJ"),
        data.dtype,
        fmt="COO",
    )
    analysis_ms = fs_spsv_impl._analyze_spsv_csr(
        data_csr,
        indices_csr,
        indptr_csr,
        b,
        (n_rows, n_cols),
        lower=lower,
        transpose=transpose,
        solve_kind=solve_kind,
        clear_cache=True,
        return_time=True,
    )
    x, solve_ms = _benchmark_flagsparse_spsv_csr(
        data_csr,
        indices_csr,
        indptr_csr,
        b,
        (n_rows, n_cols),
        lower=lower,
        transpose=transpose,
        solve_kind=solve_kind,
        warmup=warmup,
        iters=iters,
    )
    return x, analysis_ms, solve_ms


def _cupy_spsolve_lower_csr_or_coo(
    fmt,
    data,
    indices,
    indptr,
    shape,
    b,
    warmup,
    iters,
    lower,
):
    """Triangular solve via CuPy: CSR or COO storage. Returns (ms, x_torch) or (None, None)."""
    if (
        cp is None
        or cpx_sparse is None
        or cpx_spsolve_triangular is None
    ):
        return None, None
    try:
        b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b.contiguous()))
        if fmt == "COO":
            dc, rr, cc = _csr_to_coo(data, indices, indptr, shape)
            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(dc.contiguous()))
            row_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(rr.to(torch.int64).contiguous())
            )
            col_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(cc.to(torch.int64).contiguous())
            )
            A_cp = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
        else:
            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data.contiguous()))
            idx_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(indices.to(torch.int64).contiguous())
            )
            ptr_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(indptr.contiguous())
            )
            A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        for _ in range(warmup):
            _ = cpx_spsolve_triangular(
                A_cp, b_cp, lower=lower, unit_diagonal=False
            )
        cp.cuda.runtime.deviceSynchronize()
        t0 = cp.cuda.Event()
        t1 = cp.cuda.Event()
        t0.record()
        for _ in range(iters):
            x_cu = cpx_spsolve_triangular(
                A_cp, b_cp, lower=lower, unit_diagonal=False
            )
        t1.record()
        t1.synchronize()
        cupy_ms = cp.cuda.get_elapsed_time(t0, t1) / iters
        x_cu_t = torch.utils.dlpack.from_dlpack(x_cu.toDlpack())
        x_cu_t = x_cu_t.to(b.dtype)
        return cupy_ms, x_cu_t
    except Exception:
        return None, None


def _cupy_spsolve_csr_with_op(data, indices, indptr, shape, b, op_mode, lower):
    if (
        cp is None
        or cpx_sparse is None
        or cpx_spsolve_triangular is None
    ):
        return None, None
    try:
        warmup, iters = _spsv_benchmark_schedule(int(data.numel()), op_mode, data.dtype, fmt="CSR")
        data_ref, b_ref = _cupy_ref_inputs(data, b)
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref.contiguous()))
        idx_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(indices.to(torch.int64).contiguous())
        )
        ptr_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(indptr.to(torch.int64).contiguous())
        )
        b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b_ref.contiguous()))
        A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        if op_mode == "TRANS":
            A_eff = A_cp.transpose().tocsr()
            lower_eff = not lower
        elif op_mode == "CONJ":
            A_eff = A_cp.transpose().conj().tocsr()
            lower_eff = not lower
        else:
            A_eff = A_cp
            lower_eff = lower

        for _ in range(warmup):
            _ = cpx_spsolve_triangular(
                A_eff, b_cp, lower=lower_eff, unit_diagonal=False
            )
        cp.cuda.runtime.deviceSynchronize()
        c0 = cp.cuda.Event()
        c1 = cp.cuda.Event()
        c0.record()
        for _ in range(iters):
            x_cp = cpx_spsolve_triangular(
                A_eff, b_cp, lower=lower_eff, unit_diagonal=False
            )
        c1.record()
        c1.synchronize()
        ms = cp.cuda.get_elapsed_time(c0, c1) / iters
        x_t = torch.utils.dlpack.from_dlpack(x_cp.toDlpack()).to(b.dtype)
        return ms, x_t
    except Exception:
        return None, None


def run_spsv_synthetic_all(lower=True, alg_num=None):
    if not torch.cuda.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device("cuda")
    sep = "=" * 110
    print(sep)
    print("FLAGSPARSE SpSV BENCHMARK (synthetic triangular systems, CSR + COO)")
    print(sep)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Benchmark schedule: warmup={WARMUP}, iter={ITERS} "
        "(Library-main style defaults; override with --warmup/--iters)"
    )
    print(f"Triangle: {'LOWER' if lower else 'UPPER'}")
    print(f"Algorithm: {_alg_label(alg_num)}")
    print()

    hdr = (
        f"{'Fmt':>5} {'opA':>5} {'N':>6} {'FS.analysis':>12} {'FS.solve':>10} {'FS.total':>10} "
        f"{'PT.total':>10} {'CU.total':>10} {'PT.spdS':>10} {'CU.spdS':>10} {'PT.spdT':>10} {'CU.spdT':>10} "
        f"{'Status':>8} {'Err(PT)':>12} {'Err(CU)':>12}"
    )

    total = 0
    failed = 0
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("-" * 110)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<12} | "
                f"Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * 110)
            print(hdr)
            print("-" * 110)
            for n in TEST_SIZES:
                for fmt in ("CSR", "COO"):
                    op_modes = (
                        _supported_csr_full_ops(value_dtype, index_dtype)
                        if fmt == "CSR"
                        else ["NON"]
                    )
                    for op_mode in op_modes:
                        if not _alg_num_supports_case(
                            alg_num, fmt, op_mode, lower, value_dtype
                        ):
                            continue
                        data, indices, indptr, shape = _build_random_triangular_csr(
                            n, value_dtype, index_dtype, device, lower=lower
                        )
                        rhs_op = op_mode if fmt == "CSR" else "NON"
                        b = _random_rhs_for_spsv(
                            shape,
                            value_dtype,
                            device,
                            op_mode=rhs_op,
                            seed=_stable_case_seed(
                                "synthetic",
                                "LOWER" if lower else "UPPER",
                                fmt,
                                op_mode,
                                n,
                                _dtype_name(value_dtype),
                                _dtype_name(index_dtype),
                            ),
                        )

                        torch.cuda.synchronize()
                        if fmt == "CSR":
                            x, analysis_ms, t_ms = _benchmark_flagsparse_spsv_csr_split(
                                data,
                                indices,
                                indptr,
                                b,
                                shape,
                                lower=lower,
                                transpose=op_mode,
                                solve_kind=_solve_kind_from_alg_num(alg_num),
                            )
                        else:
                            dc, rr, cc = _csr_to_coo(
                                data, indices, indptr, shape, index_dtype=index_dtype
                            )
                            x, analysis_ms, t_ms = _benchmark_flagsparse_spsv_coo_split(
                                dc,
                                rr,
                                cc,
                                b,
                                shape,
                                lower=lower,
                                transpose=op_mode,
                                solve_kind=_solve_kind_from_alg_num(alg_num),
                            )
                        torch.cuda.synchronize()

                        x_pt, pytorch_ms, _pt_backend, pt_skip_reason = _benchmark_pytorch_reference(
                            data,
                            indices,
                            indptr,
                            shape,
                            b,
                            lower=lower,
                            op_mode=op_mode,
                        )
                        err_pt = (
                            float(torch.max(torch.abs(x - x_pt)).item())
                            if (x_pt is not None and n > 0)
                            else None
                        )

                        cupy_ms = None
                        err_cu = None
                        x_cu_t = None
                        if fmt == "CSR":
                            cupy_ms, x_cu_t = _cupy_spsolve_csr_with_op(
                                data, indices, indptr, shape, b, op_mode, lower
                            )
                        elif value_dtype in (
                            torch.float32,
                            torch.float64,
                            torch.complex64,
                            torch.complex128,
                        ):
                            cupy_ms, x_cu_t = _cupy_spsolve_lower_csr_or_coo(
                                fmt,
                                data,
                                indices,
                                indptr,
                                shape,
                                b,
                                WARMUP,
                                ITERS,
                                lower,
                            )
                        if x_cu_t is not None and n > 0:
                            err_cu = float(
                                torch.max(torch.abs(x - x_cu_t)).item()
                            )

                        atol, rtol = _tol_for_dtype(value_dtype)
                        ok_pt = (
                            torch.allclose(x, x_pt, atol=atol, rtol=rtol)
                            if x_pt is not None
                            else False
                        )
                        ok_cu = (
                            True
                            if x_cu_t is None
                            else torch.allclose(x, x_cu_t, atol=atol, rtol=rtol)
                        )
                        ok = ok_pt or ok_cu
                        status = "PASS" if ok else "FAIL"
                        if not ok:
                            failed += 1
                        total += 1

                        total_ms = _sum_ms(analysis_ms, t_ms)
                        pt_vs_solve = _safe_ratio(pytorch_ms, t_ms)
                        cu_vs_solve = _safe_ratio(cupy_ms, t_ms)
                        pt_vs_total = _safe_ratio(pytorch_ms, total_ms)
                        cu_vs_total = _safe_ratio(cupy_ms, total_ms)
                        print(
                            f"{fmt:>5} {op_mode:>5} {n:>6} {_fmt_ms(analysis_ms):>12} {_fmt_ms(t_ms):>10} {_fmt_ms(total_ms):>10} {_fmt_ms(pytorch_ms):>10} "
                            f"{_fmt_ms(cupy_ms):>10} "
                            f"{_fmt_ratio(pt_vs_solve):>10} {_fmt_ratio(cu_vs_solve):>10} {_fmt_ratio(pt_vs_total):>10} {_fmt_ratio(cu_vs_total):>10} "
                            f"{status:>8} {_fmt_err(err_pt):>12} {_fmt_err(err_cu):>12}"
                        )
                        # Synthetic benchmark keeps the main row compact; PyTorch fallback notes
                        # are only emitted in matrix CSV runs where failed reference checks matter.
            print("-" * 110)
            print()

    print(sep)
    print(f"Total cases: {total}  Failed: {failed}")
    print(sep)


def _run_one_csv_row_coo(path, value_dtype, index_dtype, op_mode, device, lower=True, alg_num=None):
    data, indices, indptr, shape = _load_mtx_to_csr_torch(
        path, dtype=value_dtype, device=device, lower=lower
    )
    indices = indices.to(index_dtype)
    indptr = indptr.to(index_dtype)
    n_rows, n_cols = shape
    b = _random_rhs_for_spsv(
        shape,
        value_dtype,
        device,
        op_mode=op_mode,
        seed=_stable_case_seed(
            "csv-coo",
            os.path.basename(path),
            "LOWER" if lower else "UPPER",
            op_mode,
            _dtype_name(value_dtype),
            _dtype_name(index_dtype),
        ),
    )
    d_in, r_in, c_in = _coo_inputs_for_csv(
        data, indices, indptr, shape, index_dtype=index_dtype
    )
    x, analysis_ms, t_ms = _benchmark_flagsparse_spsv_coo_split(
        d_in,
        r_in,
        c_in,
        b,
        shape,
        lower=lower,
        transpose=op_mode,
        solve_kind=_solve_kind_from_alg_num(alg_num),
    )
    return _finalize_csv_row(
        path,
        value_dtype,
        index_dtype,
        op_mode,
        data,
        indices,
        indptr,
        shape,
        x,
        analysis_ms,
        t_ms,
        b,
        n_rows,
        n_cols,
        lower=lower,
    )


def _finalize_csv_row(
    path,
    value_dtype,
    index_dtype,
    op_mode,
    data,
    indices,
    indptr,
    shape,
    x,
    analysis_ms,
    t_ms,
    b,
    n_rows,
    n_cols,
    *,
    lower=True,
):
    atol, rtol = _tol_for_dtype(value_dtype)
    err_res, _ = _solution_residual_metrics(
        data, indices, indptr, shape, x, b, value_dtype, op_mode
    )
    pytorch_ms = None
    err_pt = None
    ok_pt = False
    pt_skip_reason = None
    x_ref, pytorch_ms, _pt_backend, pt_skip_reason = _benchmark_pytorch_reference(
        data,
        indices,
        indptr,
        shape,
        b,
        lower=lower,
        op_mode=op_mode,
    )
    if x_ref is not None:
        x_cmp = x
        x_ref_cmp = x_ref
        err_pt = (
            float(torch.max(torch.abs(x_cmp - x_ref_cmp)).item())
            if n_rows > 0
            else 0.0
        )
        ok_pt = torch.allclose(x_cmp, x_ref_cmp, atol=atol, rtol=rtol)

    cupy_ms = None
    err_cu = None
    ok_cu = False
    x_cu_t = None
    cupy_ms, x_cu_t = _cupy_spsolve_csr_with_op(
        data, indices, indptr, shape, b, op_mode, lower
    )
    if x_cu_t is not None:
        x_cmp = x
        x_cu_cmp = x_cu_t
        err_cu = (
            float(torch.max(torch.abs(x_cmp - x_cu_cmp)).item())
            if n_rows > 0
            else 0.0
        )
        ok_cu = torch.allclose(x_cmp, x_cu_cmp, atol=atol, rtol=rtol)

    status = "PASS" if (ok_pt or ok_cu) else "FAIL"
    if (not ok_pt) and (not ok_cu) and (err_pt is None and err_cu is None):
        status = "REF_FAIL"
    ref_errors = [err for err in (err_pt, err_cu) if err is not None]
    err_ref = min(ref_errors) if ref_errors else None

    row = {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(value_dtype),
        "index_dtype": _dtype_name(index_dtype),
        "opA": op_mode,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(data.numel()),
        "analysis_ms": analysis_ms,
        "triton_ms": t_ms,
        "triton_total_ms": _sum_ms(analysis_ms, t_ms),
        "cusparse_ms": cupy_ms,
        "pytorch_ms": pytorch_ms,
        "cusparse_speedup_solve": _safe_ratio(cupy_ms, t_ms),
        "pytorch_speedup_solve": _safe_ratio(pytorch_ms, t_ms),
        "cusparse_speedup_total": _safe_ratio(cupy_ms, _sum_ms(analysis_ms, t_ms)),
        "pytorch_speedup_total": _safe_ratio(pytorch_ms, _sum_ms(analysis_ms, t_ms)),
        "pt_status": _status_str(ok_pt, err_pt is not None),
        "cu_status": _status_str(ok_cu, err_cu is not None),
        "status": status,
        "err_ref": err_ref,
        "err_res": err_res,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "pytorch_reason": pt_skip_reason,
        "error": None,
    }
    return row, pt_skip_reason


def _run_one_csv_row_csr_full(path, value_dtype, index_dtype, op_mode, device, lower=True, alg_num=None):
    data, indices, indptr, shape = _load_mtx_to_csr_torch(
        path, dtype=value_dtype, device=device, lower=lower
    )
    indices = indices.to(index_dtype)
    indptr = indptr.to(index_dtype)
    n_rows, n_cols = shape
    b = _random_rhs_for_spsv(
        shape,
        value_dtype,
        device,
        op_mode=op_mode,
        seed=_stable_case_seed(
            "csv-csr",
            os.path.basename(path),
            "LOWER" if lower else "UPPER",
            op_mode,
            _dtype_name(value_dtype),
            _dtype_name(index_dtype),
        ),
    )
    x, analysis_ms, t_ms = _benchmark_flagsparse_spsv_csr_split(
        data,
        indices,
        indptr,
        b,
        shape,
        lower=lower,
        transpose=op_mode,
        solve_kind=_solve_kind_from_alg_num(alg_num),
    )
    return _finalize_csv_row_csr_full(
        path,
        value_dtype,
        index_dtype,
        op_mode,
        data,
        indices,
        indptr,
        shape,
        x,
        analysis_ms,
        t_ms,
        b,
        n_rows,
        n_cols,
        lower=lower,
    )


def _finalize_csv_row_csr_full(
    path,
    value_dtype,
    index_dtype,
    op_mode,
    data,
    indices,
    indptr,
    shape,
    x,
    analysis_ms,
    t_ms,
    b,
    n_rows,
    n_cols,
    lower=True,
):
    atol, rtol = _tol_for_dtype(value_dtype)
    err_res, _ = _solution_residual_metrics(
        data, indices, indptr, shape, x, b, value_dtype, op_mode
    )

    pytorch_ms = None
    err_pt = None
    ok_pt = False
    pt_skip_reason = None
    x_ref, pytorch_ms, _pt_backend, pt_skip_reason = _benchmark_pytorch_reference(
        data,
        indices,
        indptr,
        shape,
        b,
        lower=lower,
        op_mode=op_mode,
    )
    if x_ref is not None:
        x_cmp = x
        x_ref_cmp = x_ref
        err_pt = (
            float(torch.max(torch.abs(x_cmp - x_ref_cmp)).item())
            if n_rows > 0
            else 0.0
        )
        ok_pt = torch.allclose(x_cmp, x_ref_cmp, atol=atol, rtol=rtol)

    cupy_ms = None
    err_cu = None
    ok_cu = False
    x_cu_t = None
    cupy_ms, x_cu_t = _cupy_spsolve_csr_with_op(
        data, indices, indptr, shape, b, op_mode, lower
    )
    if x_cu_t is not None:
        x_cmp = x
        x_cu_cmp = x_cu_t
        err_cu = (
            float(torch.max(torch.abs(x_cmp - x_cu_cmp)).item())
            if n_rows > 0
            else 0.0
        )
        ok_cu = torch.allclose(x_cmp, x_cu_cmp, atol=atol, rtol=rtol)

    status = "PASS" if (ok_pt or ok_cu) else "FAIL"
    if (not ok_pt) and (not ok_cu) and (err_pt is None and err_cu is None):
        status = "REF_FAIL"
    ref_errors = [err for err in (err_pt, err_cu) if err is not None]
    err_ref = min(ref_errors) if ref_errors else None

    row = {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(value_dtype),
        "index_dtype": _dtype_name(index_dtype),
        "opA": op_mode,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(data.numel()),
        "analysis_ms": analysis_ms,
        "triton_ms": t_ms,
        "triton_total_ms": _sum_ms(analysis_ms, t_ms),
        "cusparse_ms": cupy_ms,
        "pytorch_ms": pytorch_ms,
        "cusparse_speedup_solve": _safe_ratio(cupy_ms, t_ms),
        "pytorch_speedup_solve": _safe_ratio(pytorch_ms, t_ms),
        "cusparse_speedup_total": _safe_ratio(cupy_ms, _sum_ms(analysis_ms, t_ms)),
        "pytorch_speedup_total": _safe_ratio(pytorch_ms, _sum_ms(analysis_ms, t_ms)),
        "pt_status": _status_str(ok_pt, err_pt is not None),
        "cu_status": _status_str(ok_cu, err_cu is not None),
        "status": status,
        "err_ref": err_ref,
        "err_res": err_res,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "pytorch_reason": pt_skip_reason,
        "error": None,
    }
    return row, pt_skip_reason


def run_all_supported_spsv_csr_csv(
    mtx_paths,
    csv_path,
    lower=True,
    value_dtypes=None,
    index_dtypes=None,
    op_modes=None,
    alg_num=None,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    selected_value_dtypes = value_dtypes or CSR_FULL_VALUE_DTYPES
    selected_index_dtypes = index_dtypes or CSR_FULL_INDEX_DTYPES
    selected_op_modes = op_modes or SPSV_OP_MODES
    for value_dtype in selected_value_dtypes:
        for index_dtype in selected_index_dtypes:
            supported_op_modes = [
                op for op in _supported_csr_full_ops(value_dtype, index_dtype)
                if op in selected_op_modes
            ]
            for op_mode in supported_op_modes:
                if not _alg_num_supports_case(
                    alg_num, "CSR", op_mode, lower, value_dtype
                ):
                    continue
                print("=" * 150)
                print(
                    f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}  |  CSR  |  triA={'LOWER' if lower else 'UPPER'}  |  opA={op_mode}"
                )
                print(f"Algorithm: {_alg_label(alg_num)}")
                print(
                    "Formats: FlagSparse=CSR, cuSPARSE=CSR ref, "
                    "PyTorch(ms)=official sparse solve reference"
                )
                print(
                    f"Benchmark schedule: warmup={WARMUP}, iter={ITERS} "
                    "(Library-main style defaults; override with --warmup/--iters)"
                )
                print(
                    "RHS is generated directly, matching Library-main's SpSV test style. "
                    "PT.total / CU.total are single official interface call times. "
                    "PT.spdS and CU.spdS compare against FS.solve; PT.spdT and CU.spdT compare against FS.total. "
                    "Err(Ref)=best |FlagSparse-reference|, Err(Res)=|op(A)*x-b|, "
                    "Err(PT)=|FlagSparse-PyTorch|, Err(CU)=|FlagSparse-cuSPARSE|. "
                    "PASS if PyTorch / cuSPARSE reference passes. Residual is diagnostic only."
                )
                print("-" * 150)
                print(
                    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                    f"{'FS.analysis':>11} {'FS.solve':>10} {'FS.total':>10} {'CU.total':>10} {'PT.total':>10} "
                    f"{'CU.spdS':>10} {'PT.spdS':>10} {'CU.spdT':>10} {'PT.spdT':>10} "
                    f"{'Status':>6} {'Err(Ref)':>10} {'Err(Res)':>10} {'Err(PT)':>10} {'Err(CU)':>10}"
                )
                print("-" * 150)
                for path in mtx_paths:
                    try:
                        row, pt_skip = _run_one_csv_row_csr_full(
                            path, value_dtype, index_dtype, op_mode, device, lower=lower, alg_num=alg_num
                        )
                        rows_out.append(row)
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        n_rows, n_cols = row["n_rows"], row["n_cols"]
                        nnz = row["nnz"]
                        analysis_ms = row["analysis_ms"]
                        t_ms = row["triton_ms"]
                        total_ms = row["triton_total_ms"]
                        cupy_ms = row["cusparse_ms"]
                        pytorch_ms = row["pytorch_ms"]
                        err_ref, err_res = row["err_ref"], row["err_res"]
                        err_pt, err_cu = row["err_pt"], row["err_cu"]
                        status = row["status"]
                        print(
                            f"{name:<28} {n_rows:>7} {n_cols:>7} {nnz:>10} "
                            f"{_fmt_ms(analysis_ms):>11} {_fmt_ms(t_ms):>10} {_fmt_ms(total_ms):>10} {_fmt_ms(cupy_ms):>10} {_fmt_ms(pytorch_ms):>10} "
                            f"{_fmt_ratio(row['cusparse_speedup_solve']):>10} {_fmt_ratio(row['pytorch_speedup_solve']):>10} {_fmt_ratio(row['cusparse_speedup_total']):>10} {_fmt_ratio(row['pytorch_speedup_total']):>10} "
                            f"{status:>6} {_fmt_err(err_ref):>10} {_fmt_err(err_res):>10} {_fmt_err(err_pt):>10} {_fmt_err(err_cu):>10}"
                        )
                        if status in ("FAIL", "REF_FAIL"):
                            if pt_skip:
                                print(f"  NOTE: {pt_skip}")
                    except Exception as e:
                        err_msg = str(e)
                        status = "SKIP" if "SpSV requires square matrices" in err_msg else "ERROR"
                        rows_out.append(
                            {
                                "matrix": os.path.basename(path),
                                "value_dtype": _dtype_name(value_dtype),
                                "index_dtype": _dtype_name(index_dtype),
                                "opA": op_mode,
                                "n_rows": "ERR",
                                "n_cols": "ERR",
                                "nnz": "ERR",
                                "analysis_ms": None,
                                "triton_ms": None,
                                "triton_total_ms": None,
                                "cusparse_ms": None,
                                "pytorch_ms": None,
                                "cusparse_speedup_solve": None,
                                "pytorch_speedup_solve": None,
                                "cusparse_speedup_total": None,
                                "pytorch_speedup_total": None,
                                "pt_status": "N/A",
                                "cu_status": "N/A",
                                "status": status,
                                "err_ref": None,
                                "err_res": None,
                                "err_pt": None,
                                "err_cu": None,
                                "pytorch_reason": None,
                                "error": err_msg,
                            }
                        )
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        print(
                            f"{name:<28} {'ERR':>7} {'ERR':>7} {'ERR':>10} "
                            f"{_fmt_ms(None):>11} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} "
                            f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} "
                            f"{status:>6} {_fmt_err(None):>10} {_fmt_err(None):>10} {_fmt_err(None):>10} {_fmt_err(None):>10}"
                        )
                        print(f"  {status}: {e}")
                print("-" * 150)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "opA",
        "n_rows",
        "n_cols",
        "nnz",
        "analysis_ms",
        "triton_ms",
        "triton_total_ms",
        "cusparse_ms",
        "pytorch_ms",
        "cusparse_speedup_solve",
        "pytorch_speedup_solve",
        "cusparse_speedup_total",
        "pytorch_speedup_total",
        "pt_status",
        "cu_status",
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
        for r in rows_out:
            w.writerow({k: ("" if v is None else v) for k, v in _csv_export_row_spsv(r).items()})
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def run_all_dtypes_spsv_coo_csv(
    mtx_paths,
    csv_path,
    lower=True,
    value_dtypes=None,
    index_dtypes=None,
    op_modes=None,
    alg_num=None,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    rows_out = []
    selected_value_dtypes = value_dtypes or VALUE_DTYPES
    selected_index_dtypes = index_dtypes or INDEX_DTYPES
    for value_dtype in selected_value_dtypes:
        for index_dtype in selected_index_dtypes:
            supported_op_modes = [
                op for op in _supported_csr_full_ops(value_dtype, index_dtype)
                if op in (op_modes or SPSV_OP_MODES)
            ]
            for op_mode in supported_op_modes:
                if not _alg_num_supports_case(
                    alg_num, "COO", op_mode, lower, value_dtype
                ):
                    continue
                print("=" * 150)
                print(
                    f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}  |  COO"
                    f"  triA={'LOWER' if lower else 'UPPER'}  |  opA={op_mode}"
                )
                print(f"Algorithm: {_alg_label(alg_num)}")
                print(
                    "Formats: FlagSparse=COO input routed through CSR SpSV, cuSPARSE=CSR ref, "
                    "PyTorch(ms)=official sparse solve reference. "
                    "RHS is generated directly, matching Library-main's SpSV test style."
                )
                print(
                    f"Benchmark schedule: warmup={WARMUP}, iter={ITERS} "
                    "(Library-main style defaults; override with --warmup/--iters)"
                )
                print(
                    "PT.total / CU.total are single official interface call times. "
                    "PT.spdS and CU.spdS compare against FS.solve; PT.spdT and CU.spdT compare against FS.total."
                )
                print(
                    "Matrix metadata fields reuse the canonical triangular matrix, matching CSR CSV output."
                )
                print(
                    "Err(Ref)=best |FlagSparse-reference|, Err(Res)=|op(A)*x-b|, "
                    "Err(PT)=|FlagSparse-PyTorch|, Err(CU)=|FlagSparse-cuSPARSE|. "
                    "PASS if PyTorch / cuSPARSE reference passes. Residual is diagnostic only."
                )
                print("-" * 150)
                print(
                    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                    f"{'FS.analysis':>11} {'FS.solve':>10} {'FS.total':>10} {'CU.total':>10} {'PT.total':>10} "
                    f"{'CU.spdS':>10} {'PT.spdS':>10} {'CU.spdT':>10} {'PT.spdT':>10} "
                    f"{'Status':>6} {'Err(Ref)':>10} {'Err(Res)':>10} {'Err(PT)':>10} {'Err(CU)':>10}"
                )
                print("-" * 150)
                for path in mtx_paths:
                    try:
                        row, pt_skip = _run_one_csv_row_coo(
                            path, value_dtype, index_dtype, op_mode, device, lower=lower, alg_num=alg_num
                        )
                        rows_out.append(row)
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        n_rows, n_cols = row["n_rows"], row["n_cols"]
                        nnz = row["nnz"]
                        analysis_ms = row["analysis_ms"]
                        t_ms = row["triton_ms"]
                        total_ms = row["triton_total_ms"]
                        cupy_ms = row["cusparse_ms"]
                        pytorch_ms = row["pytorch_ms"]
                        err_ref, err_res = row["err_ref"], row["err_res"]
                        err_pt, err_cu = row["err_pt"], row["err_cu"]
                        status = row["status"]
                        print(
                            f"{name:<28} {n_rows:>7} {n_cols:>7} {nnz:>10} "
                            f"{_fmt_ms(analysis_ms):>11} {_fmt_ms(t_ms):>10} {_fmt_ms(total_ms):>10} {_fmt_ms(cupy_ms):>10} {_fmt_ms(pytorch_ms):>10} "
                            f"{_fmt_ratio(row['cusparse_speedup_solve']):>10} {_fmt_ratio(row['pytorch_speedup_solve']):>10} {_fmt_ratio(row['cusparse_speedup_total']):>10} {_fmt_ratio(row['pytorch_speedup_total']):>10} "
                            f"{status:>6} {_fmt_err(err_ref):>10} {_fmt_err(err_res):>10} {_fmt_err(err_pt):>10} {_fmt_err(err_cu):>10}"
                        )
                        if status in ("FAIL", "REF_FAIL"):
                            if pt_skip:
                                print(f"  NOTE: {pt_skip}")
                    except Exception as e:
                        err_msg = str(e)
                        status = "SKIP" if "SpSV requires square matrices" in err_msg else "ERROR"
                        rows_out.append(
                            {
                                "matrix": os.path.basename(path),
                                "value_dtype": _dtype_name(value_dtype),
                                "index_dtype": _dtype_name(index_dtype),
                                "opA": op_mode,
                                "n_rows": "ERR",
                                "n_cols": "ERR",
                                "nnz": "ERR",
                                "analysis_ms": None,
                                "triton_ms": None,
                                "triton_total_ms": None,
                                "cusparse_ms": None,
                                "pytorch_ms": None,
                                "cusparse_speedup_solve": None,
                                "pytorch_speedup_solve": None,
                                "cusparse_speedup_total": None,
                                "pytorch_speedup_total": None,
                                "pt_status": "N/A",
                                "cu_status": "N/A",
                                "status": status,
                                "err_ref": None,
                                "err_res": None,
                                "err_pt": None,
                                "err_cu": None,
                                "pytorch_reason": None,
                                "error": err_msg,
                            }
                        )
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        print(
                            f"{name:<28} {'ERR':>7} {'ERR':>7} {'ERR':>10} "
                            f"{_fmt_ms(None):>11} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} "
                            f"{'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {status:>6} {_fmt_err(None):>10} {_fmt_err(None):>10} {_fmt_err(None):>10} {_fmt_err(None):>10}"
                        )
                        print(f"  {status}: {e}")
                print("-" * 150)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "opA",
        "n_rows",
        "n_cols",
        "nnz",
        "analysis_ms",
        "triton_ms",
        "triton_total_ms",
        "cusparse_ms",
        "pytorch_ms",
        "cusparse_speedup_solve",
        "pytorch_speedup_solve",
        "cusparse_speedup_total",
        "pytorch_speedup_total",
        "pt_status",
        "cu_status",
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
        for r in rows_out:
            w.writerow({k: ("" if v is None else v) for k, v in _csv_export_row_spsv(r).items()})
    print(f"Wrote {len(rows_out)} rows to {csv_path}")


def _check_one_csr_transpose_case(path, value_dtype, index_dtype, op_mode, device, lower=True):
    data, indices, indptr, shape = _load_mtx_to_csr_torch(
        path, dtype=value_dtype, device=device, lower=lower
    )
    indices = indices.to(index_dtype)
    indptr = indptr.to(index_dtype)
    n_rows, n_cols = shape
    trans_data, trans_indices64, trans_indptr64 = fs_spsv_impl._csr_transpose(
        data,
        indices.to(torch.int64),
        indptr.to(torch.int64),
        n_rows,
        n_cols,
        conjugate=(op_mode == "CONJ"),
    )
    trans_shape = (n_cols, n_rows)
    trans_indices = trans_indices64.to(index_dtype)
    trans_indptr = trans_indptr64.to(index_dtype)

    probe = _random_rhs_for_spsv(
        trans_shape,
        value_dtype,
        device,
        op_mode="NON",
        seed=_stable_case_seed(
            "check-transpose-action",
            os.path.basename(path),
            "LOWER" if lower else "UPPER",
            op_mode,
            _dtype_name(value_dtype),
            _dtype_name(index_dtype),
        ),
    )
    action_ref = _apply_csr_op(data, indices, indptr, probe, shape, op_mode)
    action_trans = _apply_csr_op(
        trans_data, trans_indices, trans_indptr, probe, trans_shape, "NON"
    )
    action_err = (
        float(torch.max(torch.abs(action_trans - action_ref)).item())
        if action_ref.numel() > 0
        else 0.0
    )
    atol, rtol = _tol_for_dtype(value_dtype)
    action_ok = torch.allclose(action_trans, action_ref, atol=atol, rtol=rtol)

    b = _random_rhs_for_spsv(
        shape,
        value_dtype,
        device,
        op_mode=op_mode,
        seed=_stable_case_seed(
            "check-transpose-solve",
            os.path.basename(path),
            "LOWER" if lower else "UPPER",
            op_mode,
            _dtype_name(value_dtype),
            _dtype_name(index_dtype),
        ),
    )
    x_op = fs.flagsparse_spsv_csr(
        data,
        indices,
        indptr,
        b,
        shape,
        lower=lower,
        transpose=op_mode,
    )
    x_mat = fs.flagsparse_spsv_csr(
        trans_data,
        trans_indices,
        trans_indptr,
        b,
        trans_shape,
        lower=not lower,
        transpose="NON",
    )
    solve_err = (
        float(torch.max(torch.abs(x_op - x_mat)).item()) if x_op.numel() > 0 else 0.0
    )
    solve_ok = torch.allclose(x_op, x_mat, atol=atol, rtol=rtol)

    ref_err = None
    ref_ok = None
    x_ref, _, _, _ = _benchmark_pytorch_reference(
        data,
        indices,
        indptr,
        shape,
        b,
        lower=lower,
        op_mode=op_mode,
    )
    if x_ref is not None:
        ref_err = (
            float(torch.max(torch.abs(x_op - x_ref)).item()) if x_op.numel() > 0 else 0.0
        )
        ref_ok = torch.allclose(x_op, x_ref, atol=atol, rtol=rtol)

    status = "PASS" if action_ok and solve_ok and (ref_ok is not False) else "FAIL"
    return {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(value_dtype),
        "index_dtype": _dtype_name(index_dtype),
        "opA": op_mode,
        "n_rows": n_rows,
        "nnz": int(data.numel()),
        "action_err": action_err,
        "solve_err": solve_err,
        "ref_err": ref_err,
        "status": status,
    }


def run_csr_transpose_check(
    mtx_paths,
    lower=True,
    value_dtypes=None,
    index_dtypes=None,
    op_modes=None,
):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    selected_value_dtypes = value_dtypes or CSR_FULL_VALUE_DTYPES
    selected_index_dtypes = index_dtypes or CSR_FULL_INDEX_DTYPES
    selected_op_modes = [op for op in (op_modes or ("TRANS", "CONJ")) if op in ("TRANS", "CONJ")]
    if not selected_op_modes:
        print("--check-transpose only checks TRANS/CONJ; no matching op selected.")
        return

    print("=" * 150)
    print(
        "CSR TRANS/CONJ preprocessing check: "
        "ActionErr compares materialized op(A) against direct CSR scatter; "
        "SolveErr compares transpose path against materialized NON path."
    )
    print("-" * 150)
    print(
        f"{'Matrix':<28} {'dtype':>10} {'index':>7} {'opA':>5} "
        f"{'N':>7} {'NNZ':>10} {'Status':>6} {'ActionErr':>10} {'SolveErr':>10} {'RefErr':>10}"
    )
    print("-" * 150)
    total = 0
    failed = 0
    for value_dtype in selected_value_dtypes:
        for index_dtype in selected_index_dtypes:
            for op_mode in selected_op_modes:
                for path in mtx_paths:
                    try:
                        row = _check_one_csr_transpose_case(
                            path,
                            value_dtype,
                            index_dtype,
                            op_mode,
                            device,
                            lower=lower,
                        )
                        total += 1
                        failed += int(row["status"] != "PASS")
                        name = row["matrix"][:27]
                        if len(row["matrix"]) > 27:
                            name += "..."
                        print(
                            f"{name:<28} {row['value_dtype']:>10} {row['index_dtype']:>7} {row['opA']:>5} "
                            f"{row['n_rows']:>7} {row['nnz']:>10} {row['status']:>6} "
                            f"{_fmt_err(row['action_err']):>10} {_fmt_err(row['solve_err']):>10} {_fmt_err(row['ref_err']):>10}"
                        )
                    except Exception as e:
                        total += 1
                        failed += 1
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name += "..."
                        print(
                            f"{name:<28} {_dtype_name(value_dtype):>10} {_dtype_name(index_dtype):>7} {op_mode:>5} "
                            f"{'ERR':>7} {'ERR':>10} {'ERROR':>6} "
                            f"{_fmt_err(None):>10} {_fmt_err(None):>10} {_fmt_err(None):>10}"
                        )
                        print(f"  ERROR: {e}")
    print("-" * 150)
    print(f"Total cases: {total}  Failed: {failed}")


def main():
    global WARMUP, ITERS
    parser = argparse.ArgumentParser(
        description="SpSV test: synthetic triangular systems and optional .mtx (CSR/COO), same baselines as CSR."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic triangular tests"
    )
    parser.add_argument(
        "--csv-csr",
        type=str,
        default=None,
        metavar="FILE",
        help="Run full supported CSR SpSV combinations (dtype/index/opA) on .mtx and export CSV",
    )
    parser.add_argument(
        "--csv-coo",
        type=str,
        default=None,
        metavar="FILE",
        help="Run all dtypes on .mtx (COO SpSV), same CSV columns as --csv-csr",
    )
    parser.add_argument(
        "--check-transpose",
        action="store_true",
        help="Check CSR TRANS/CONJ preprocessing against direct CSR scatter and materialized NON solve",
    )
    parser.add_argument(
        "--upper",
        action="store_true",
        help="Use upper-triangular inputs instead of the default lower-triangular inputs",
    )
    parser.add_argument(
        "--ops",
        type=str,
        default=None,
        help="Comma-separated opA filter for CSR/COO CSV, e.g. NON,TRANS,CONJ",
    )
    parser.add_argument(
        "--alg-num",
        "--alg_num",
        dest="alg_num",
        type=_parse_alg_num,
        default=None,
        help=(
            "Algorithm selection compatible with allinone style. "
            "Supported: 1=ALG1(csr_cw), 2=ALG2(csr_cw_levelschd), 3=ALG3(csr_nnz_balance), 8=ALG8(csr_nnz_balance). "
            "Omit to use AUTO routing."
        ),
    )
    parser.add_argument(
        "--value-dtypes",
        type=str,
        default=None,
        help="Comma-separated value dtype filter for CSR CSV, e.g. float,double,complex64,complex128",
    )
    parser.add_argument(
        "--index-dtypes",
        type=str,
        default=None,
        help="Comma-separated index dtype filter for CSR CSV, e.g. int32,int64",
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
    lower = not args.upper
    if args.alg_num in (2, 3, 8):
        if args.check_transpose:
            raise ValueError(
                f"ALG{args.alg_num} matches allinone's NON-only path; --check-transpose is not supported"
            )
        if args.upper:
            raise ValueError(
                f"ALG{args.alg_num} matches allinone's lower-triangular path; --upper is not supported"
            )
        if args.ops:
            op_modes_cli = _parse_op_modes_filter(args.ops)
            if any(op != "NON" for op in op_modes_cli):
                raise ValueError(
                    f"ALG{args.alg_num} matches allinone's NON-only path; use --ops NON"
                )

    if args.synthetic:
        run_spsv_synthetic_all(lower=lower, alg_num=args.alg_num)
        return

    paths = []
    for p in args.mtx:
        if os.path.isfile(p) and p.endswith(".mtx"):
            paths.append(p)
        elif os.path.isdir(p):
            paths.extend(sorted(glob.glob(os.path.join(p, "*.mtx"))))
    if args.check_transpose:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --check-transpose")
            return
        value_dtypes = (
            _parse_value_dtypes_filter(args.value_dtypes)
            if args.value_dtypes
            else None
        )
        index_dtypes = (
            _parse_index_dtypes_filter(args.index_dtypes)
            if args.index_dtypes
            else None
        )
        op_modes = (
            _parse_op_modes_filter(args.ops)
            if args.ops
            else None
        )
        run_csr_transpose_check(
            paths,
            lower=lower,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            op_modes=op_modes,
        )
        return
    if args.csv_csr:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-csr")
            return
        value_dtypes = (
            _parse_value_dtypes_filter(args.value_dtypes)
            if args.value_dtypes
            else None
        )
        index_dtypes = (
            _parse_index_dtypes_filter(args.index_dtypes)
            if args.index_dtypes
            else None
        )
        op_modes = (
            _parse_op_modes_filter(args.ops)
            if args.ops
            else None
        )
        run_all_supported_spsv_csr_csv(
            paths,
            args.csv_csr,
            lower=lower,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            op_modes=op_modes,
            alg_num=args.alg_num,
        )
        return
    if args.csv_coo:
        if not paths:
            paths = sorted(glob.glob("*.mtx"))
        if not paths:
            print("No .mtx files found for --csv-coo")
            return
        value_dtypes = (
            _parse_value_dtypes_filter(args.value_dtypes)
            if args.value_dtypes
            else None
        )
        index_dtypes = (
            _parse_index_dtypes_filter(args.index_dtypes)
            if args.index_dtypes
            else None
        )
        op_modes = (
            _parse_op_modes_filter(args.ops)
            if args.ops
            else None
        )
        run_all_dtypes_spsv_coo_csv(
            paths,
            args.csv_coo,
            lower=lower,
            value_dtypes=value_dtypes,
            index_dtypes=index_dtypes,
            op_modes=op_modes,
            alg_num=args.alg_num,
        )
        return

    print("Use --synthetic, --csv-csr, or --csv-coo to run SpSV tests.")


if __name__ == "__main__":
    main()
