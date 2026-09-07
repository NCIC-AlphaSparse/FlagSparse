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

"""SpSV tests: synthetic triangular systems and optional .mtx (CSR/COO)."""

import argparse
import csv
import glob
import hashlib
import os
import sys
import time
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import flagsparse as fs
from flagsparse.sparse_operations import _common as fs_common
import flagsparse.sparse_operations.spsv as fs_spsv_impl
from mtx_fast import NonSquareMatrixError

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
WARMUP = 10
ITERS = 20

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


VALUE_DTYPE_NAME_MAP = {_dtype_name(dtype): dtype for dtype in CSR_FULL_VALUE_DTYPES}
VALUE_DTYPE_NAME_MAP.update(
    {
        "float": torch.float32,
        "double": torch.float64,
    }
)
INDEX_DTYPE_NAME_MAP = {_dtype_name(dtype): dtype for dtype in CSR_FULL_INDEX_DTYPES}
CUDA_SPSV_ALG_NUM_TO_SOLVE_KIND = {
    1: "csr_cw",
    2: "csr_cw_levelschd",
    3: "csr_roc",
    4: "csr_smblk",
    8: "csr_nnz_balance",
}
ROCM_SPSV_ALG_NUM_TO_SOLVE_KIND = {
    1: "csr_cw",
    2: "csr_cw_levelschd",
    3: "csr_nnz_balance",
}


def _active_spsv_alg_num_to_solve_kind():
    if fs_spsv_impl._is_rocm_runtime():
        return ROCM_SPSV_ALG_NUM_TO_SOLVE_KIND
    if fs_spsv_impl._is_mthreads_runtime() or fs_spsv_impl._is_ascend_runtime():
        return {1: "csr_cw"}
    return CUDA_SPSV_ALG_NUM_TO_SOLVE_KIND


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
    active_map = _active_spsv_alg_num_to_solve_kind()
    if value not in active_map:
        raise ValueError(
            "unsupported alg_num: "
            f"{value}. Supported values on the current backend: {sorted(active_map)}"
        )
    return value


def _solve_kind_from_alg_num(alg_num):
    if alg_num is None:
        return None
    active_map = _active_spsv_alg_num_to_solve_kind()
    value = int(alg_num)
    if value not in active_map:
        raise ValueError(
            f"ALG{value} is unavailable on the current backend; "
            f"supported values are {sorted(active_map)}"
        )
    return active_map[value]


def _alg_label(alg_num):
    return "AUTO" if alg_num is None else f"ALG{int(alg_num)}"


def _print_rocm_alg3_launch_config(alg_num):
    if not fs_spsv_impl._is_rocm_runtime():
        return
    if _solve_kind_from_alg_num(alg_num) != "csr_nnz_balance":
        return
    cu_count = int(fs_spsv_impl._ACCEL.get_device_properties(0).multi_processor_count)
    workgroups_per_cu = fs_spsv_impl.SPSV_ROCM_ALG3_WORKGROUPS_PER_CU
    worker_cap = cu_count * workgroups_per_cu
    print(
        "DCU ALG3 launch: "
        f"BLOCK_NNZ={fs_spsv_impl.SPSV_ROCM_ALG3_BLOCK_NNZ}, "
        f"workgroups/CU={workgroups_per_cu}, CU={cu_count}, "
        f"worker_cap={worker_cap}"
    )


def _alg_num_supports_case(alg_num, fmt, op_mode, lower, value_dtype):
    if alg_num is None:
        return True
    alg_num = int(alg_num)
    if alg_num not in _active_spsv_alg_num_to_solve_kind():
        return False
    if alg_num == 1:
        return True
    if fs_spsv_impl._is_rocm_runtime() and alg_num == 3:
        return fmt in ("CSR", "COO") and op_mode == "NON" and bool(lower)
    if alg_num in (2, 3, 4, 8):
        return fmt in ("CSR", "COO") and op_mode == "NON"
    return False


def _fmt_ms(v):
    return "N/A" if v is None else f"{v:.4f}"


def _fmt_ratio(v):
    return "N/A" if v is None else f"{v:.2f}"


def _fmt_err(v):
    return "N/A" if v is None else f"{v:.2e}"


def _safe_ratio(other_ms, base_ms):
    if other_ms is None or base_ms is None or base_ms <= 0:
        return None
    return other_ms / base_ms


def _vendor_backend_name():
    return fs_common._expected_vendor_sparse_label()


def _vendor_short_name():
    return fs_common._expected_vendor_sparse_short()


def _backend_error_key():
    backend = fs_common._expected_vendor_sparse_backend()
    if backend == "hipsparse":
        return "err_hip"
    if backend in ("cupy_cusparse", "native_cusparse"):
        return "err_cu"
    return f"err_{str(backend or 'vendor').replace('-', '_')}"


def _spsv_csv_fieldnames():
    """Return one CSV schema named for the active sparse-library backend."""

    backend_name = _vendor_backend_name()
    return [
        "matrix",
        "value_dtype",
        "index_dtype",
        "opA",
        "n_rows",
        "n_cols",
        "nnz",
        "FlagSparse_ms",
        f"{backend_name}_route",
        f"{backend_name}_ms",
        "PyTorch_ms",
        f"FlagSparse_vs_{backend_name}_speedup",
        "FlagSparse_vs_PyTorch_speedup",
        "status",
        "err_pt",
        _backend_error_key(),
        f"{backend_name}_reason",
        "pytorch_reason",
        "error",
    ]


def _vendor_reference_route():
    backend = fs_common._expected_vendor_sparse_backend()
    if backend == "hipsparse":
        return "hipSPARSE direct API"
    if backend == "cupy_cusparse":
        return "cuSPARSE via CuPy spsolve_triangular"
    return f"{fs_common._expected_vendor_sparse_label()} route"


def _spsv_benchmark_schedule(nnz, op_mode, value_dtype, fmt="CSR"):
    del nnz, op_mode, value_dtype, fmt
    return int(WARMUP), int(ITERS)


def _allinone_filtered_avg_ms(times, fmt="CSR"):
    if not times:
        return None
    times = [float(t) for t in times]
    if len(times) == 1:
        return times[0]
    if fmt.upper() == "COO":
        avg = sum(times) / len(times)
        kept = [t for t in times if t < 2.0 * avg]
        return sum(kept) / len(kept) if kept else avg
    ordered = sorted(times)
    n = len(ordered)
    if n % 2 == 0:
        median = (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0
    else:
        median = ordered[n // 2]
    lo = median * 0.9
    hi = median * 1.1
    kept = [t for t in ordered if lo <= t <= hi]
    return sum(kept) / len(kept) if kept else median


def _tol_for_dtype(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-6, 1e-5
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


def _extract_triangular_csr(data, indices, indptr, shape, *, lower):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() == 0:
        return (
            data,
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device),
        )
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr.to(torch.int64)[1:] - indptr.to(torch.int64)[:-1],
    )
    col = indices.to(torch.int64)
    keep = col <= row if lower else col >= row
    row = row[keep]
    col = col[keep]
    data_eff = data[keep]
    key = row * max(1, n_cols) + col
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    row = row[order]
    col = col[order]
    data_eff = data_eff[order]
    nnz_per_row = torch.bincount(row, minlength=n_rows)
    indptr_eff = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    indptr_eff[1:] = torch.cumsum(nnz_per_row, dim=0)
    return data_eff.contiguous(), col.contiguous(), indptr_eff


def _effective_csr_for_op(data, indices, indptr, shape, *, lower, op_mode):
    data_tri, indices_tri, indptr_tri = _extract_triangular_csr(
        data, indices, indptr, shape, lower=lower
    )
    if op_mode == "TRANS":
        data_eff, indices_eff, indptr_eff = _csr_transpose(
            data_tri,
            indices_tri,
            indptr_tri,
            shape,
            conjugate=False,
        )
    elif op_mode == "CONJ":
        data_eff, indices_eff, indptr_eff = _csr_transpose(
            data_tri,
            indices_tri,
            indptr_tri,
            shape,
            conjugate=True,
        )
    else:
        data_eff = data_tri
        indices_eff = indices_tri
        indptr_eff = indptr_tri
    return data_eff, indices_eff, indptr_eff


def _build_csr_tensor_for_op(data, indices, indptr, shape, op_mode, *, lower):
    data_eff, indices_eff, indptr_eff = _effective_csr_for_op(
        data, indices, indptr, shape, lower=lower, op_mode=op_mode
    )
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
        A_csr = _build_csr_tensor_for_op(
            data, indices, indptr, shape, op_mode, lower=lower
        )
        if not fs_spsv_impl._is_accel_tensor(A_csr):
            raise RuntimeError("torch.sparse.spsolve CUDA path is unavailable")
        fs_spsv_impl._ACCEL.synchronize()
        e0 = fs_spsv_impl._ACCEL.Event(True)
        e1 = fs_spsv_impl._ACCEL.Event(True)
        e0.record()
        x_ref = sparse_spsolve(A_csr, b)
        e1.record()
        fs_spsv_impl._ACCEL.synchronize()
        ms = e0.elapsed_time(e1)
        return x_ref.to(b.dtype), ms, "gpu_sparse", None
    except Exception as sparse_err:
        if "out of memory" in str(sparse_err).lower() and fs_spsv_impl._ACCEL.is_available():
            fs_spsv_impl._ACCEL.empty_cache()
        return (
            None,
            None,
            "unavailable",
            f"PyTorch sparse solve unavailable ({sparse_err})",
        )


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
                torch.randn(len(off_cols), dtype=base_real_dtype, device=device).mul_(
                    0.01
                ),
                torch.randn(len(off_cols), dtype=base_real_dtype, device=device).mul_(
                    0.01
                ),
            )
            off_vals_host = [complex(v) for v in off_vals.cpu().tolist()]
        else:
            off_vals = torch.randn(
                len(off_cols), dtype=base_real_dtype, device=device
            ).mul_(0.01)
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
    """Load a .mtx into a well-conditioned triangular-ready CSR via the fast
    scipy reader (see tests/mtx_fast.py). Ensures a structural diagonal and
    row-normalizes so the triangular solve stays diagonally dominant; the former
    pure-Python parser took minutes on large SuiteSparse matrices."""
    from mtx_fast import load_csr_spsv

    return load_csr_spsv(file_path, dtype=dtype, device=device, lower=lower)


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


def _apply_csr_op(data, indices, indptr, x, shape, op_mode, *, lower):
    n_rows, n_cols = int(shape[0]), int(shape[1])
    data_eff, indices_eff, indptr_eff = _effective_csr_for_op(
        data, indices, indptr, shape, lower=lower, op_mode=op_mode
    )
    row = torch.repeat_interleave(
        torch.arange(n_rows, device=data.device, dtype=torch.int64),
        indptr_eff[1:] - indptr_eff[:-1],
    )
    col = indices_eff.to(torch.int64)
    if op_mode == "NON":
        b = torch.zeros(n_rows, dtype=data.dtype, device=data.device)
        b.scatter_add_(0, row, data_eff * x[col])
        return b
    if op_mode == "TRANS":
        b = torch.zeros(n_cols, dtype=data.dtype, device=data.device)
        b.scatter_add_(0, row, data_eff * x[col])
        return b
    if op_mode == "CONJ":
        b = torch.zeros(n_cols, dtype=data.dtype, device=data.device)
        b.scatter_add_(0, row, data_eff * x[col])
        return b
    raise ValueError("op_mode must be 'NON', 'TRANS', or 'CONJ'")


def _solution_residual_metrics(
    data, indices, indptr, shape, x, b, value_dtype, op_mode, *, lower
):
    atol, rtol = _tol_for_dtype(value_dtype)
    b_recon = _apply_csr_op(data, indices, indptr, x, shape, op_mode, lower=lower)
    err_res = float(torch.max(torch.abs(b_recon - b)).item()) if b.numel() > 0 else 0.0
    ok_res = torch.allclose(b_recon, b, atol=atol, rtol=rtol)
    return err_res, ok_res


def _benchmark_flagsparse_spsv_full_rounds(
    reset_call,
    analyze_call,
    solve_call,
    *,
    warmup,
    iters,
):
    """Measure one fresh FlagSparse analysis plus one solve per round."""

    warmup = max(0, int(warmup))
    iters = max(1, int(iters))

    def run_round(record):
        reset_call()
        fs_spsv_impl._ACCEL.synchronize()
        total_start = time.perf_counter()
        state = analyze_call()
        fs_spsv_impl._ACCEL.synchronize()
        analysis_end = time.perf_counter()
        x = solve_call(state)
        fs_spsv_impl._ACCEL.synchronize()
        solve_end = time.perf_counter()
        if record:
            analysis_times.append((analysis_end - total_start) * 1000.0)
            solve_times.append((solve_end - analysis_end) * 1000.0)
            total_times.append((solve_end - total_start) * 1000.0)
        return x, state

    x = None
    state = None
    analysis_times = []
    solve_times = []
    total_times = []
    for _ in range(warmup):
        x, state = run_round(False)
    for _ in range(iters):
        x, state = run_round(True)
    return (
        x,
        state,
        _allinone_filtered_avg_ms(analysis_times),
        _allinone_filtered_avg_ms(solve_times),
        _allinone_filtered_avg_ms(total_times),
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
    data_tri, indices_tri, indptr_tri = _extract_triangular_csr(
        data, indices, indptr, shape, lower=lower
    )
    warmup, iters = _spsv_benchmark_schedule(
        int(data_tri.numel()),
        "NON" if op_mode == "N" else ("TRANS" if op_mode == "T" else "CONJ"),
        data.dtype,
        fmt="CSR",
    )
    if op_mode != "N":
        analyze_call = lambda: fs_spsv_impl._analyze_spsv_csr(
            data_tri,
            indices_tri,
            indptr_tri,
            b,
            shape,
            lower=lower,
            transpose=transpose,
            solve_kind=solve_kind,
            clear_cache=False,
            return_time=False,
        )
        solve_call = lambda _state: fs.flagsparse_spsv_csr(
            data_tri,
            indices_tri,
            indptr_tri,
            b,
            shape,
            lower=lower,
            transpose=transpose,
            solve_kind=solve_kind,
        )
        x, _state, analysis_ms, solve_ms, total_ms = (
            _benchmark_flagsparse_spsv_full_rounds(
                fs_spsv_impl._clear_spsv_csr_preprocess_cache,
                analyze_call,
                solve_call,
                warmup=warmup,
                iters=iters,
            )
        )
        return x, analysis_ms, solve_ms, total_ms, "transpose_cw"

    def analyze_call():
        descr = fs_spsv_impl.flagsparse_spsv_analysis_csr(
            data_tri,
            indices_tri,
            indptr_tri,
            shape,
            lower=lower,
            transpose=transpose,
            solve_kind=solve_kind,
            clear_cache=False,
        )
        workspace = fs_spsv_impl.flagsparse_spsv_create_workspace(descr)
        if descr.solve_kind == "transpose_cw":
            fs_spsv_impl.flagsparse_spsv_preprocess_csr(
                descr, workspace=workspace
            )
        return descr, workspace

    def solve_call(state):
        descr, workspace = state
        return fs_spsv_impl.flagsparse_spsv_solve_csr(
            descr,
            b,
            workspace=workspace,
        )

    x, state, analysis_ms, solve_ms, total_ms = (
        _benchmark_flagsparse_spsv_full_rounds(
            fs_spsv_impl._clear_spsv_csr_preprocess_cache,
            analyze_call,
            solve_call,
            warmup=warmup,
            iters=iters,
        )
    )
    descr, _workspace = state
    return x, analysis_ms, solve_ms, total_ms, descr.route_name


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
    data, input_index_dtype, row64, col64, b, n_rows, n_cols = (
        fs_spsv_impl._prepare_spsv_coo_inputs(data, row, col, b, shape)
    )
    trans_mode = fs_spsv_impl._normalize_spsv_transpose_mode(transpose)
    if trans_mode == "N":
        fs_spsv_impl._validate_spsv_non_trans_combo(
            data.dtype, input_index_dtype, "COO"
        )
    else:
        fs_spsv_impl._validate_spsv_trans_combo(data.dtype, input_index_dtype, "COO")
    data_csr, indices_csr, indptr_csr = fs_spsv_impl._coo2csr_for_spsv(
        data, row64, col64, n_rows, assume_ordered=False
    )
    data_tri, indices_tri, indptr_tri = _extract_triangular_csr(
        data_csr, indices_csr, indptr_csr, (n_rows, n_cols), lower=lower
    )
    warmup, iters = _spsv_benchmark_schedule(
        int(data_tri.numel()),
        "NON" if trans_mode == "N" else ("TRANS" if trans_mode == "T" else "CONJ"),
        data.dtype,
        fmt="COO",
    )
    if trans_mode != "N":
        analyze_call = lambda: fs_spsv_impl._analyze_spsv_csr(
            data_tri,
            indices_tri,
            indptr_tri,
            b,
            (n_rows, n_cols),
            lower=lower,
            transpose=transpose,
            solve_kind=solve_kind,
            clear_cache=False,
            return_time=False,
        )
        solve_call = lambda _state: fs.flagsparse_spsv_csr(
            data_tri,
            indices_tri,
            indptr_tri,
            b,
            (n_rows, n_cols),
            lower=lower,
            transpose=transpose,
            solve_kind=solve_kind,
        )
        x, _state, analysis_ms, solve_ms, total_ms = (
            _benchmark_flagsparse_spsv_full_rounds(
                fs_spsv_impl._clear_spsv_csr_preprocess_cache,
                analyze_call,
                solve_call,
                warmup=warmup,
                iters=iters,
            )
        )
        return x, analysis_ms, solve_ms, total_ms, "transpose_cw"

    def analyze_call():
        descr = fs_spsv_impl.flagsparse_spsv_analysis_csr(
            data_tri,
            indices_tri,
            indptr_tri,
            (n_rows, n_cols),
            lower=lower,
            transpose=transpose,
            solve_kind=solve_kind,
            clear_cache=False,
        )
        workspace = fs_spsv_impl.flagsparse_spsv_create_workspace(descr)
        return descr, workspace

    def solve_call(state):
        descr, workspace = state
        return fs_spsv_impl.flagsparse_spsv_solve_csr(
            descr,
            b,
            workspace=workspace,
        )

    x, state, analysis_ms, solve_ms, total_ms = (
        _benchmark_flagsparse_spsv_full_rounds(
            fs_spsv_impl._clear_spsv_csr_preprocess_cache,
            analyze_call,
            solve_call,
            warmup=warmup,
            iters=iters,
        )
    )
    descr, _workspace = state
    return x, analysis_ms, solve_ms, total_ms, descr.route_name


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
    """Return vendor total time, solution, failure reason, and actual route."""
    vendor_backend, vendor_reason = fs_spsv_impl._spsv_csr_sparse_ref_backend(
        data.dtype, indices.dtype, indptr.dtype, op="non"
    )
    if vendor_backend is None:
        return None, None, vendor_reason, None
    if vendor_backend == "hipsparse":
        # The DCU vendor reference is hipSPARSE CSR SpSV.  COO cases use the
        # same mathematically equivalent CSR reference after input conversion.
        return _cupy_spsolve_csr_with_op(
            data, indices, indptr, shape, b, "NON", lower
        )
    if cp is None or cpx_sparse is None or cpx_spsolve_triangular is None:
        return None, None, "CuPy spsolve_triangular is unavailable", None
    try:
        data_eff, indices_eff, indptr_eff = _effective_csr_for_op(
            data, indices, indptr, shape, lower=lower, op_mode="NON"
        )
        b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b.contiguous()))
        if fmt == "COO":
            dc, rr, cc = _csr_to_coo(data_eff, indices_eff, indptr_eff, shape)
            data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(dc.contiguous()))
            row_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(rr.to(torch.int64).contiguous())
            )
            col_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(cc.to(torch.int64).contiguous())
            )
            A_cp = cpx_sparse.coo_matrix((data_cp, (row_cp, col_cp)), shape=shape)
        else:
            data_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(data_eff.contiguous())
            )
            idx_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(indices_eff.to(torch.int64).contiguous())
            )
            ptr_cp = cp.from_dlpack(
                torch.utils.dlpack.to_dlpack(indptr_eff.contiguous())
            )
            A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        for _ in range(warmup):
            _ = cpx_spsolve_triangular(A_cp, b_cp, lower=lower, unit_diagonal=False)
        cp.cuda.runtime.deviceSynchronize()
        times = []
        for _ in range(iters):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            x_cu = cpx_spsolve_triangular(A_cp, b_cp, lower=lower, unit_diagonal=False)
            cp.cuda.runtime.deviceSynchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        cupy_ms = _allinone_filtered_avg_ms(times, fmt=fmt)
        x_cu_t = torch.utils.dlpack.from_dlpack(x_cu.toDlpack())
        x_cu_t = x_cu_t.to(b.dtype)
        return cupy_ms, x_cu_t, None, "cuSPARSE via CuPy spsolve_triangular"
    except Exception as exc:
        return None, None, str(exc), None


def _cupy_spsolve_csr_with_op(data, indices, indptr, shape, b, op_mode, lower):
    # Vendor triangular-solve baseline, dispatched per backend: hipSPARSE SpSV on
    # DCU/ROCm, CuPy's spsolve_triangular (cuSPARSE-backed) on CUDA.
    vendor_backend, selector_reason = fs_spsv_impl._spsv_csr_sparse_ref_backend(
        data.dtype,
        indices.dtype,
        indptr.dtype,
        op=str(op_mode).lower(),
    )
    if vendor_backend is None:
        return None, None, selector_reason, None
    if vendor_backend == "hipsparse":
        warmup, iters = _spsv_benchmark_schedule(
            int(data.numel()), op_mode, data.dtype, fmt="CSR"
        )
        sparse_ref = fs_spsv_impl._benchmark_spsv_csr_sparse_ref(
            data,
            indices,
            indptr,
            b,
            shape,
            lower=lower,
            unit_diagonal=False,
            op=str(op_mode).lower(),
            warmup=warmup,
            iters=iters,
            # Match the CUDA reference scope below: every measured vendor call
            # includes triangular-solve analysis plus solve. Caller-visible
            # sparse-input construction remains outside the timed loop.
            fresh_each_iter=True,
        )
        if sparse_ref.get("backend") != "hipsparse":
            reason = sparse_ref.get("reason") or "backend selector returned no reason"
            return (
                None,
                None,
                f"ROCm/DCU vendor dispatch did not select hipSPARSE: {reason}",
                None,
            )
        return (
            sparse_ref["ms"],
            sparse_ref["values"],
            sparse_ref.get("reason"),
            "hipSPARSE direct API",
        )
    if vendor_backend != "cupy_cusparse":
        return None, None, f"unsupported vendor backend: {vendor_backend}", None
    if cp is None or cpx_sparse is None or cpx_spsolve_triangular is None:
        return None, None, "CuPy spsolve_triangular is unavailable", None
    try:
        warmup, iters = _spsv_benchmark_schedule(
            int(data.numel()), op_mode, data.dtype, fmt="CSR"
        )
        data_ref, indices_ref, indptr_ref = _effective_csr_for_op(
            data, indices, indptr, shape, lower=lower, op_mode=op_mode
        )
        data_ref, b_ref = _cupy_ref_inputs(data_ref, b)
        data_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(data_ref.contiguous()))
        idx_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(indices_ref.to(torch.int64).contiguous())
        )
        ptr_cp = cp.from_dlpack(
            torch.utils.dlpack.to_dlpack(indptr_ref.to(torch.int64).contiguous())
        )
        b_cp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(b_ref.contiguous()))
        A_cp = cpx_sparse.csr_matrix((data_cp, idx_cp, ptr_cp), shape=shape)
        if op_mode == "TRANS":
            A_eff = A_cp
            lower_eff = not lower
        elif op_mode == "CONJ":
            A_eff = A_cp
            lower_eff = not lower
        else:
            A_eff = A_cp
            lower_eff = lower

        for _ in range(warmup):
            _ = cpx_spsolve_triangular(
                A_eff, b_cp, lower=lower_eff, unit_diagonal=False
            )
        cp.cuda.runtime.deviceSynchronize()
        times = []
        for _ in range(iters):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            x_cp = cpx_spsolve_triangular(
                A_eff, b_cp, lower=lower_eff, unit_diagonal=False
            )
            cp.cuda.runtime.deviceSynchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = _allinone_filtered_avg_ms(times, fmt="CSR")
        x_t = torch.utils.dlpack.from_dlpack(x_cp.toDlpack()).to(b.dtype)
        return ms, x_t, None, "cuSPARSE via CuPy spsolve_triangular"
    except Exception as exc:
        return None, None, str(exc), None


def run_spsv_synthetic_all(lower=True, alg_num=None):
    if not fs_spsv_impl._ACCEL.is_available():
        print("CUDA is not available. Please run on a GPU-enabled system.")
        return
    device = torch.device(fs_spsv_impl._ACCEL_DEVICE_TYPE)
    sep = "=" * 124
    print(sep)
    print("FLAGSPARSE SpSV BENCHMARK (synthetic triangular systems, CSR + COO)")
    print(sep)
    print(f"GPU: {fs_spsv_impl._ACCEL.get_device_name(0)}")
    print(
        f"Benchmark schedule: warmup={WARMUP}, iter={ITERS} "
        "(each timed round is one fresh analysis plus one solve; override with --warmup/--iters)"
    )
    print(f"Triangle: {'LOWER' if lower else 'UPPER'}")
    print(f"Algorithm: {_alg_label(alg_num)}")
    print(f"FlagSparse route: {_solve_kind_from_alg_num(alg_num) or 'AUTO'}")
    _print_rocm_alg3_launch_config(alg_num)
    vendor_name = _vendor_backend_name()
    vendor_short = _vendor_short_name()
    print(
        f"FS.ms and {vendor_name}.ms are average complete "
        "analysis/preparation + solve rounds; speedup = vendor_ms / FS.ms."
    )
    print()

    hdr = (
        f"{'Fmt':>5} {'opA':>5} {'N':>6} {'FS.ms':>10} "
        f"{(vendor_short + '.ms'):>10} {'PT.ms':>10} "
        f"{(vendor_short + '.spdT'):>10} {'PT.spdT':>10} "
        f"{'Status':>8} {'Err(PT)':>12} {('Err(' + vendor_short + ')'):>12}"
    )

    total = 0
    failed = 0
    for value_dtype in VALUE_DTYPES:
        for index_dtype in INDEX_DTYPES:
            print("-" * 124)
            print(
                f"Value dtype: {_dtype_name(value_dtype):<12} | "
                f"Index dtype: {_dtype_name(index_dtype):<6}"
            )
            print("-" * 124)
            print(hdr)
            print("-" * 124)
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

                        fs_spsv_impl._ACCEL.synchronize()
                        if fmt == "CSR":
                            x, analysis_ms, t_ms, flagsparse_ms, _route_name = (
                                _benchmark_flagsparse_spsv_csr_split(
                                    data,
                                    indices,
                                    indptr,
                                    b,
                                    shape,
                                    lower=lower,
                                    transpose=op_mode,
                                    solve_kind=_solve_kind_from_alg_num(alg_num),
                                )
                            )
                        else:
                            dc, rr, cc = _csr_to_coo(
                                data, indices, indptr, shape, index_dtype=index_dtype
                            )
                            x, analysis_ms, t_ms, flagsparse_ms, _route_name = (
                                _benchmark_flagsparse_spsv_coo_split(
                                    dc,
                                    rr,
                                    cc,
                                    b,
                                    shape,
                                    lower=lower,
                                    transpose=op_mode,
                                    solve_kind=_solve_kind_from_alg_num(alg_num),
                                )
                            )
                        fs_spsv_impl._ACCEL.synchronize()

                        x_pt, pytorch_ms, _pt_backend, _pt_skip_reason = (
                            _benchmark_pytorch_reference(
                                data,
                                indices,
                                indptr,
                                shape,
                                b,
                                lower=lower,
                                op_mode=op_mode,
                            )
                        )
                        err_pt = (
                            float(torch.max(torch.abs(x - x_pt)).item())
                            if (x_pt is not None and n > 0)
                            else None
                        )

                        cupy_ms = None
                        err_cu = None
                        x_cu_t = None
                        vendor_reason = None
                        vendor_route = None
                        if fmt == "CSR":
                            cupy_ms, x_cu_t, vendor_reason, vendor_route = (
                                _cupy_spsolve_csr_with_op(
                                    data, indices, indptr, shape, b, op_mode, lower
                                )
                            )
                        elif value_dtype in (
                            torch.float32,
                            torch.float64,
                            torch.complex64,
                            torch.complex128,
                        ):
                            cupy_ms, x_cu_t, vendor_reason, vendor_route = (
                                _cupy_spsolve_lower_csr_or_coo(
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
                            )
                        if x_cu_t is not None and n > 0:
                            err_cu = float(torch.max(torch.abs(x - x_cu_t)).item())

                        atol, rtol = _tol_for_dtype(value_dtype)
                        ok_pt = (
                            torch.allclose(x, x_pt, atol=atol, rtol=rtol)
                            if x_pt is not None
                            else False
                        )
                        ok_cu = (
                            torch.allclose(x, x_cu_t, atol=atol, rtol=rtol)
                            if x_cu_t is not None
                            else False
                        )
                        ok = ok_pt or ok_cu
                        status = "PASS" if ok else "FAIL"
                        if not ok:
                            failed += 1
                        total += 1

                        pt_vs_total = _safe_ratio(pytorch_ms, flagsparse_ms)
                        cu_vs_total = _safe_ratio(cupy_ms, flagsparse_ms)
                        print(
                            f"{fmt:>5} {op_mode:>5} {n:>6} {_fmt_ms(flagsparse_ms):>10} {_fmt_ms(cupy_ms):>10} "
                            f"{_fmt_ms(pytorch_ms):>10} {_fmt_ratio(cu_vs_total):>10} {_fmt_ratio(pt_vs_total):>10} "
                            f"{status:>8} {_fmt_err(err_pt):>12} {_fmt_err(err_cu):>12}"
                        )
                        if vendor_reason:
                            print(f"  {vendor_name} reference unavailable: {vendor_reason}")
                        elif vendor_route:
                            print(f"  {vendor_name} route used: {vendor_route}")
                        # Synthetic benchmark keeps the main row compact; PyTorch fallback notes
                        # are only emitted in matrix CSV runs where failed reference checks matter.
            print("-" * 124)
            print()

    print(sep)
    print(f"Total cases: {total}  Failed: {failed}")
    print(sep)


def _run_one_csv_row_coo(
    path, value_dtype, index_dtype, op_mode, device, lower=True, alg_num=None
):
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
    data_tri, _indices_tri, _indptr_tri = _extract_triangular_csr(
        data, indices, indptr, shape, lower=lower
    )
    x, analysis_ms, t_ms, flagsparse_ms, _route_name = (
        _benchmark_flagsparse_spsv_coo_split(
            d_in,
            r_in,
            c_in,
            b,
            shape,
            lower=lower,
            transpose=op_mode,
            solve_kind=_solve_kind_from_alg_num(alg_num),
        )
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
        flagsparse_ms,
        b,
        n_rows,
        n_cols,
        int(data_tri.numel()),
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
    flagsparse_ms,
    b,
    n_rows,
    n_cols,
    nnz_effective,
    *,
    lower=True,
):
    atol, rtol = _tol_for_dtype(value_dtype)
    err_res, _ = _solution_residual_metrics(
        data, indices, indptr, shape, x, b, value_dtype, op_mode, lower=lower
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
            float(torch.max(torch.abs(x_cmp - x_ref_cmp)).item()) if n_rows > 0 else 0.0
        )
        ok_pt = torch.allclose(x_cmp, x_ref_cmp, atol=atol, rtol=rtol)

    vendor_ms = None
    err_vendor = None
    ok_vendor = False
    x_vendor = None
    vendor_ms, x_vendor, vendor_reason, vendor_route = (
        _cupy_spsolve_csr_with_op(
            data, indices, indptr, shape, b, op_mode, lower
        )
    )
    if x_vendor is not None:
        x_cmp = x
        x_vendor_cmp = x_vendor
        err_vendor = (
            float(torch.max(torch.abs(x_cmp - x_vendor_cmp)).item())
            if n_rows > 0
            else 0.0
        )
        ok_vendor = torch.allclose(x_cmp, x_vendor_cmp, atol=atol, rtol=rtol)

    status = "PASS" if (ok_pt or ok_vendor) else "FAIL"
    if (not ok_pt) and (not ok_vendor) and (err_pt is None and err_vendor is None):
        status = "REF_FAIL"
    vendor_backend = _vendor_backend_name()
    backend_error_key = _backend_error_key()

    record = {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(value_dtype),
        "index_dtype": _dtype_name(index_dtype),
        "opA": op_mode,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(nnz_effective),
        "FlagSparse_ms": flagsparse_ms,
        f"{vendor_backend}_route": vendor_route,
        f"{vendor_backend}_ms": vendor_ms,
        "PyTorch_ms": pytorch_ms,
        f"FlagSparse_vs_{vendor_backend}_speedup": _safe_ratio(
            vendor_ms, flagsparse_ms
        ),
        "FlagSparse_vs_PyTorch_speedup": _safe_ratio(
            pytorch_ms, flagsparse_ms
        ),
        "status": status,
        "err_pt": err_pt,
        backend_error_key: err_vendor,
        f"{vendor_backend}_reason": vendor_reason,
        "pytorch_reason": pt_skip_reason,
        "error": None,
        "_err_res": err_res,
    }
    return record, pt_skip_reason


def _run_one_csv_row_csr_full(
    path, value_dtype, index_dtype, op_mode, device, lower=True, alg_num=None
):
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
    data_tri, _indices_tri, _indptr_tri = _extract_triangular_csr(
        data, indices, indptr, shape, lower=lower
    )
    x, analysis_ms, t_ms, flagsparse_ms, _route_name = (
        _benchmark_flagsparse_spsv_csr_split(
            data,
            indices,
            indptr,
            b,
            shape,
            lower=lower,
            transpose=op_mode,
            solve_kind=_solve_kind_from_alg_num(alg_num),
        )
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
        flagsparse_ms,
        b,
        n_rows,
        n_cols,
        int(data_tri.numel()),
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
    flagsparse_ms,
    b,
    n_rows,
    n_cols,
    nnz_effective,
    lower=True,
):
    atol, rtol = _tol_for_dtype(value_dtype)
    err_res, _ = _solution_residual_metrics(
        data, indices, indptr, shape, x, b, value_dtype, op_mode, lower=lower
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
            float(torch.max(torch.abs(x_cmp - x_ref_cmp)).item()) if n_rows > 0 else 0.0
        )
        ok_pt = torch.allclose(x_cmp, x_ref_cmp, atol=atol, rtol=rtol)

    vendor_ms = None
    err_vendor = None
    ok_vendor = False
    x_vendor = None
    vendor_ms, x_vendor, vendor_reason, vendor_route = (
        _cupy_spsolve_csr_with_op(
            data, indices, indptr, shape, b, op_mode, lower
        )
    )
    if x_vendor is not None:
        x_cmp = x
        x_vendor_cmp = x_vendor
        err_vendor = (
            float(torch.max(torch.abs(x_cmp - x_vendor_cmp)).item())
            if n_rows > 0
            else 0.0
        )
        ok_vendor = torch.allclose(x_cmp, x_vendor_cmp, atol=atol, rtol=rtol)

    status = "PASS" if (ok_pt or ok_vendor) else "FAIL"
    if (not ok_pt) and (not ok_vendor) and (err_pt is None and err_vendor is None):
        status = "REF_FAIL"
    vendor_backend = _vendor_backend_name()
    backend_error_key = _backend_error_key()

    record = {
        "matrix": os.path.basename(path),
        "value_dtype": _dtype_name(value_dtype),
        "index_dtype": _dtype_name(index_dtype),
        "opA": op_mode,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "nnz": int(nnz_effective),
        "FlagSparse_ms": flagsparse_ms,
        f"{vendor_backend}_route": vendor_route,
        f"{vendor_backend}_ms": vendor_ms,
        "PyTorch_ms": pytorch_ms,
        f"FlagSparse_vs_{vendor_backend}_speedup": _safe_ratio(
            vendor_ms, flagsparse_ms
        ),
        "FlagSparse_vs_PyTorch_speedup": _safe_ratio(
            pytorch_ms, flagsparse_ms
        ),
        "status": status,
        "err_pt": err_pt,
        backend_error_key: err_vendor,
        f"{vendor_backend}_reason": vendor_reason,
        "pytorch_reason": pt_skip_reason,
        "error": None,
        "_err_res": err_res,
    }
    return record, pt_skip_reason


def run_all_supported_spsv_csr_csv(
    mtx_paths,
    csv_path,
    lower=True,
    value_dtypes=None,
    index_dtypes=None,
    op_modes=None,
    alg_num=None,
):
    if not fs_spsv_impl._ACCEL.is_available():
        print("GPU runtime is not available.")
        return
    device = torch.device(fs_spsv_impl._ACCEL_DEVICE_TYPE)
    records_out = []
    vendor_name = _vendor_backend_name()
    vendor_short = _vendor_short_name()
    vendor_route_key = f"{vendor_name}_route"
    vendor_speedup_key = f"FlagSparse_vs_{vendor_name}_speedup"
    backend_error_key = _backend_error_key()
    vendor_reason_key = f"{vendor_name}_reason"
    selected_value_dtypes = value_dtypes or CSR_FULL_VALUE_DTYPES
    selected_index_dtypes = index_dtypes or CSR_FULL_INDEX_DTYPES
    selected_op_modes = op_modes or SPSV_OP_MODES
    for value_dtype in selected_value_dtypes:
        for index_dtype in selected_index_dtypes:
            supported_op_modes = [
                op
                for op in _supported_csr_full_ops(value_dtype, index_dtype)
                if op in selected_op_modes
            ]
            for op_mode in supported_op_modes:
                if not _alg_num_supports_case(
                    alg_num, "CSR", op_mode, lower, value_dtype
                ):
                    continue
                print("=" * 126)
                print(
                    f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}  |  CSR  |  triA={'LOWER' if lower else 'UPPER'}  |  opA={op_mode}"
                )
                print(f"Algorithm: {_alg_label(alg_num)}")
                print(
                    f"FlagSparse route: "
                    f"{_solve_kind_from_alg_num(alg_num) or 'AUTO'}"
                )
                _print_rocm_alg3_launch_config(alg_num)
                print(
                    f"Formats: FlagSparse=CSR, {vendor_name}=CSR reference, "
                    "PT=official sparse solve reference"
                )
                print(f"{vendor_name} route: {_vendor_reference_route()}")
                print(
                    f"Benchmark schedule: warmup={WARMUP}, iter={ITERS} "
                    "(each timed round is one fresh analysis plus one solve; override with --warmup/--iters)"
                )
                print(
                    f"FS.ms and {vendor_short}.ms are average complete "
                    "analysis/preparation + solve rounds; "
                    f"{vendor_short}.spdT={vendor_name}_ms/FS.ms. "
                    f"Ept=|FS-PT|, E{vendor_short}=|FS-{vendor_name}|."
                )
                print("-" * 126)
                print(
                    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                    f"{'FS.ms':>10} {(vendor_short + '.ms'):>10} {'PT.ms':>10} "
                    f"{(vendor_short + '.spdT'):>10} {'PT.spdT':>10} "
                    f"{'Status':>8} {'Ept':>10} {('E' + vendor_short):>10}"
                )
                print("-" * 126)
                for path in mtx_paths:
                    try:
                        record, pt_skip = _run_one_csv_row_csr_full(
                            path,
                            value_dtype,
                            index_dtype,
                            op_mode,
                            device,
                            lower=lower,
                            alg_num=alg_num,
                        )
                        records_out.append(record)
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        n_rows, n_cols = record["n_rows"], record["n_cols"]
                        nnz = record["nnz"]
                        flagsparse_ms = record["FlagSparse_ms"]
                        vendor_ms = record[f"{vendor_name}_ms"]
                        pytorch_ms = record["PyTorch_ms"]
                        err_pt = record["err_pt"]
                        err_backend = record[backend_error_key]
                        status = record["status"]
                        print(
                            f"{name:<28} {n_rows:>7} {n_cols:>7} {nnz:>10} "
                            f"{_fmt_ms(flagsparse_ms):>10} {_fmt_ms(vendor_ms):>10} {_fmt_ms(pytorch_ms):>10} "
                            f"{_fmt_ratio(record[vendor_speedup_key]):>10} "
                            f"{_fmt_ratio(record['FlagSparse_vs_PyTorch_speedup']):>10} "
                            f"{status:>8} {_fmt_err(err_pt):>10} "
                            f"{_fmt_err(err_backend):>10}"
                        )
                        if record[vendor_reason_key]:
                            print(
                                f"  NOTE: {vendor_name} reference unavailable: "
                                f"{record[vendor_reason_key]}"
                            )
                        elif record[vendor_route_key]:
                            print(
                                f"  {vendor_name} route used: "
                                f"{record[vendor_route_key]}"
                            )
                        if status in ("FAIL", "REF_FAIL") and pt_skip:
                            print(f"  NOTE: {pt_skip}")
                        if status in ("FAIL", "REF_FAIL"):
                            print(
                                "  Diagnostic residual |op(A)*x-b|: "
                                f"{_fmt_err(record.get('_err_res'))}"
                            )
                    except Exception as e:
                        err_msg = str(e)
                        is_skip = isinstance(e, NonSquareMatrixError)
                        status = "SKIP" if is_skip else "ERROR"
                        n_rows, n_cols = e.shape if is_skip else ("ERR", "ERR")
                        nnz = "N/A" if is_skip else "ERR"
                        records_out.append(
                            {
                                "matrix": os.path.basename(path),
                                "value_dtype": _dtype_name(value_dtype),
                                "index_dtype": _dtype_name(index_dtype),
                                "opA": op_mode,
                                "n_rows": n_rows,
                                "n_cols": n_cols,
                                "nnz": nnz,
                                "FlagSparse_ms": None,
                                vendor_route_key: None,
                                f"{vendor_name}_ms": None,
                                "PyTorch_ms": None,
                                vendor_speedup_key: None,
                                "FlagSparse_vs_PyTorch_speedup": None,
                                "status": status,
                                "err_pt": None,
                                backend_error_key: None,
                                vendor_reason_key: None,
                                "pytorch_reason": None,
                                "error": err_msg,
                            }
                        )
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        print(
                            f"{name:<28} {str(n_rows):>7} {str(n_cols):>7} {str(nnz):>10} "
                            f"{_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} "
                            f"{'N/A':>10} {'N/A':>10} {status:>8} "
                            f"{_fmt_err(None):>10} {_fmt_err(None):>10}"
                        )
                        print(f"  {status}: {e}")
                print("-" * 126)
    fieldnames = _spsv_csv_fieldnames()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for record in records_out:
            w.writerow(
                {k: ("" if record.get(k) is None else record.get(k)) for k in fieldnames}
            )
    print(f"Wrote {len(records_out)} rows to {csv_path}")


def run_all_dtypes_spsv_coo_csv(
    mtx_paths,
    csv_path,
    lower=True,
    value_dtypes=None,
    index_dtypes=None,
    op_modes=None,
    alg_num=None,
):
    if not fs_spsv_impl._ACCEL.is_available():
        print("GPU runtime is not available.")
        return
    device = torch.device(fs_spsv_impl._ACCEL_DEVICE_TYPE)
    records_out = []
    vendor_name = _vendor_backend_name()
    vendor_short = _vendor_short_name()
    vendor_route_key = f"{vendor_name}_route"
    vendor_speedup_key = f"FlagSparse_vs_{vendor_name}_speedup"
    backend_error_key = _backend_error_key()
    vendor_reason_key = f"{vendor_name}_reason"
    selected_value_dtypes = value_dtypes or VALUE_DTYPES
    selected_index_dtypes = index_dtypes or INDEX_DTYPES
    for value_dtype in selected_value_dtypes:
        for index_dtype in selected_index_dtypes:
            supported_op_modes = [
                op
                for op in _supported_csr_full_ops(value_dtype, index_dtype)
                if op in (op_modes or SPSV_OP_MODES)
            ]
            for op_mode in supported_op_modes:
                if not _alg_num_supports_case(
                    alg_num, "COO", op_mode, lower, value_dtype
                ):
                    continue
                print("=" * 126)
                print(
                    f"Value dtype: {_dtype_name(value_dtype)}  |  Index dtype: {_dtype_name(index_dtype)}  |  COO"
                    f"  triA={'LOWER' if lower else 'UPPER'}  |  opA={op_mode}"
                )
                print(f"Algorithm: {_alg_label(alg_num)}")
                print(
                    f"FlagSparse route: "
                    f"{_solve_kind_from_alg_num(alg_num) or 'AUTO'}"
                )
                _print_rocm_alg3_launch_config(alg_num)
                print(
                    f"Formats: FlagSparse=COO via CSR SpSV, {vendor_name}=CSR "
                    "reference, PT=official sparse solve reference."
                )
                print(f"{vendor_name} route: {_vendor_reference_route()}")
                print(
                    f"Benchmark schedule: warmup={WARMUP}, iter={ITERS} "
                    "(each timed round is one fresh analysis plus one solve; override with --warmup/--iters)"
                )
                print(
                    f"FS.ms and {vendor_short}.ms are average complete "
                    "analysis/preparation + solve rounds; "
                    f"{vendor_short}.spdT={vendor_name}_ms/FS.ms."
                )
                print(
                    "Matrix metadata reuse the canonical triangular matrix, matching CSR CSV output."
                )
                print(
                    f"Ept=|FS-PT|, E{vendor_short}=|FS-{vendor_name}|."
                )
                print("-" * 126)
                print(
                    f"{'Matrix':<28} {'N_rows':>7} {'N_cols':>7} {'NNZ':>10} "
                    f"{'FS.ms':>10} {(vendor_short + '.ms'):>10} {'PT.ms':>10} "
                    f"{(vendor_short + '.spdT'):>10} {'PT.spdT':>10} "
                    f"{'Status':>8} {'Ept':>10} {('E' + vendor_short):>10}"
                )
                print("-" * 126)
                for path in mtx_paths:
                    try:
                        record, pt_skip = _run_one_csv_row_coo(
                            path,
                            value_dtype,
                            index_dtype,
                            op_mode,
                            device,
                            lower=lower,
                            alg_num=alg_num,
                        )
                        records_out.append(record)
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        n_rows, n_cols = record["n_rows"], record["n_cols"]
                        nnz = record["nnz"]
                        flagsparse_ms = record["FlagSparse_ms"]
                        vendor_ms = record[f"{vendor_name}_ms"]
                        pytorch_ms = record["PyTorch_ms"]
                        err_pt = record["err_pt"]
                        err_backend = record[backend_error_key]
                        status = record["status"]
                        print(
                            f"{name:<28} {n_rows:>7} {n_cols:>7} {nnz:>10} "
                            f"{_fmt_ms(flagsparse_ms):>10} {_fmt_ms(vendor_ms):>10} {_fmt_ms(pytorch_ms):>10} "
                            f"{_fmt_ratio(record[vendor_speedup_key]):>10} "
                            f"{_fmt_ratio(record['FlagSparse_vs_PyTorch_speedup']):>10} "
                            f"{status:>8} {_fmt_err(err_pt):>10} "
                            f"{_fmt_err(err_backend):>10}"
                        )
                        if record[vendor_reason_key]:
                            print(
                                f"  NOTE: {vendor_name} reference unavailable: "
                                f"{record[vendor_reason_key]}"
                            )
                        elif record[vendor_route_key]:
                            print(
                                f"  {vendor_name} route used: "
                                f"{record[vendor_route_key]}"
                            )
                        if status in ("FAIL", "REF_FAIL") and pt_skip:
                            print(f"  NOTE: {pt_skip}")
                        if status in ("FAIL", "REF_FAIL"):
                            print(
                                "  Diagnostic residual |op(A)*x-b|: "
                                f"{_fmt_err(record.get('_err_res'))}"
                            )
                    except Exception as e:
                        err_msg = str(e)
                        is_skip = isinstance(e, NonSquareMatrixError)
                        status = "SKIP" if is_skip else "ERROR"
                        n_rows, n_cols = e.shape if is_skip else ("ERR", "ERR")
                        nnz = "N/A" if is_skip else "ERR"
                        records_out.append(
                            {
                                "matrix": os.path.basename(path),
                                "value_dtype": _dtype_name(value_dtype),
                                "index_dtype": _dtype_name(index_dtype),
                                "opA": op_mode,
                                "n_rows": n_rows,
                                "n_cols": n_cols,
                                "nnz": nnz,
                                "FlagSparse_ms": None,
                                vendor_route_key: None,
                                f"{vendor_name}_ms": None,
                                "PyTorch_ms": None,
                                vendor_speedup_key: None,
                                "FlagSparse_vs_PyTorch_speedup": None,
                                "status": status,
                                "err_pt": None,
                                backend_error_key: None,
                                vendor_reason_key: None,
                                "pytorch_reason": None,
                                "error": err_msg,
                            }
                        )
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name = name + "…"
                        print(
                            f"{name:<28} {str(n_rows):>7} {str(n_cols):>7} {str(nnz):>10} "
                            f"{_fmt_ms(None):>10} {_fmt_ms(None):>10} {_fmt_ms(None):>10} "
                            f"{'N/A':>10} {'N/A':>10} {status:>8} "
                            f"{_fmt_err(None):>10} {_fmt_err(None):>10}"
                        )
                        print(f"  {status}: {e}")
                print("-" * 126)
    fieldnames = _spsv_csv_fieldnames()
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for record in records_out:
            w.writerow(
                {k: ("" if record.get(k) is None else record.get(k)) for k in fieldnames}
            )
    print(f"Wrote {len(records_out)} rows to {csv_path}")


def _check_one_csr_transpose_case(
    path, value_dtype, index_dtype, op_mode, device, lower=True
):
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
    action_ref = _apply_csr_op(
        data, indices, indptr, probe, shape, op_mode, lower=lower
    )
    action_trans = _apply_csr_op(
        trans_data,
        trans_indices,
        trans_indptr,
        probe,
        trans_shape,
        "NON",
        lower=not lower,
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
            float(torch.max(torch.abs(x_op - x_ref)).item())
            if x_op.numel() > 0
            else 0.0
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
    if not fs_spsv_impl._ACCEL.is_available():
        print("CUDA is not available.")
        return
    device = torch.device(fs_spsv_impl._ACCEL_DEVICE_TYPE)
    selected_value_dtypes = value_dtypes or CSR_FULL_VALUE_DTYPES
    selected_index_dtypes = index_dtypes or CSR_FULL_INDEX_DTYPES
    selected_op_modes = [
        op for op in (op_modes or ("TRANS", "CONJ")) if op in ("TRANS", "CONJ")
    ]
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
                        record = _check_one_csr_transpose_case(
                            path,
                            value_dtype,
                            index_dtype,
                            op_mode,
                            device,
                            lower=lower,
                        )
                        total += 1
                        failed += int(record["status"] != "PASS")
                        name = record["matrix"][:27]
                        if len(record["matrix"]) > 27:
                            name += "..."
                        print(
                            f"{name:<28} {record['value_dtype']:>10} {record['index_dtype']:>7} {record['opA']:>5} "
                            f"{record['n_rows']:>7} {record['nnz']:>10} {record['status']:>6} "
                            f"{_fmt_err(record['action_err']):>10} {_fmt_err(record['solve_err']):>10} {_fmt_err(record['ref_err']):>10}"
                        )
                    except Exception as e:
                        total += 1
                        is_skip = isinstance(e, NonSquareMatrixError)
                        status = "SKIP" if is_skip else "ERROR"
                        failed += int(not is_skip)
                        name = os.path.basename(path)[:27]
                        if len(os.path.basename(path)) > 27:
                            name += "..."
                        print(
                            f"{name:<28} {_dtype_name(value_dtype):>10} {_dtype_name(index_dtype):>7} {op_mode:>5} "
                            f"{'N/A' if is_skip else 'ERR':>7} {'N/A' if is_skip else 'ERR':>10} {status:>6} "
                            f"{_fmt_err(None):>10} {_fmt_err(None):>10} {_fmt_err(None):>10}"
                        )
                        print(f"  {status}: {e}")
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
            "DCU: 1=ALG1(csr_cw), 2=ALG2(csr_cw_levelschd), "
            "3=ALG3(csr_nnz_balance). CUDA remains: 1=csr_cw, "
            "2=csr_cw_levelschd, 3=csr_roc, 4=csr_smblk, "
            "8=csr_nnz_balance. "
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
        help=(
            "Warmup analysis/preparation + solve rounds "
            "(default: 10, matching SpSM)"
        ),
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=ITERS,
        help=(
            "Timed analysis/preparation + solve rounds "
            "(default: 20, matching SpSM)"
        ),
    )
    args = parser.parse_args()
    WARMUP = max(0, int(args.warmup))
    ITERS = max(1, int(args.iters))
    lower = not args.upper
    if args.alg_num in (2, 3, 4, 8):
        if args.check_transpose:
            raise ValueError(
                f"ALG{args.alg_num} matches allinone's NON-only path; --check-transpose is not supported"
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
            _parse_value_dtypes_filter(args.value_dtypes) if args.value_dtypes else None
        )
        index_dtypes = (
            _parse_index_dtypes_filter(args.index_dtypes) if args.index_dtypes else None
        )
        op_modes = _parse_op_modes_filter(args.ops) if args.ops else None
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
            _parse_value_dtypes_filter(args.value_dtypes) if args.value_dtypes else None
        )
        index_dtypes = (
            _parse_index_dtypes_filter(args.index_dtypes) if args.index_dtypes else None
        )
        op_modes = _parse_op_modes_filter(args.ops) if args.ops else None
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
            _parse_value_dtypes_filter(args.value_dtypes) if args.value_dtypes else None
        )
        index_dtypes = (
            _parse_index_dtypes_filter(args.index_dtypes) if args.index_dtypes else None
        )
        op_modes = _parse_op_modes_filter(args.ops) if args.ops else None
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
