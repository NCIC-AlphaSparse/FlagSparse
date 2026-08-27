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

"""SpSM tests: synthetic triangular systems and optional .mtx batch CSV."""

import argparse
import ctypes
import ctypes.util
import csv
import glob
import math
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
import flagsparse.sparse_operations.spsm as fs_spsm_impl
from mtx_fast import NonSquareMatrixError

FORMATS = ("csr", "coo")
VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)
INDEX_DTYPES = [torch.int32]
CSV_VALUE_DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]
CSV_INDEX_DTYPES = [torch.int32]
WARMUP = 10
ITERS = 20
TRACE_CUSPARSE = False
SPSM_OP_MODES = ["NON", "NON_TRANS"]
_CUSPARSE_MEASUREMENT_SCOPE = "native_cusparse_prepare_plus_solve"

_CUSPARSE_STATUS_SUCCESS = 0
_CUSPARSE_OPERATION_NON_TRANSPOSE = 0
_CUSPARSE_OPERATION_TRANSPOSE = 1
_CUSPARSE_INDEX_BASE_ZERO = 0
_CUSPARSE_INDEX_32I = 2
_CUSPARSE_INDEX_64I = 3
_CUSPARSE_ORDER_COL = 1
_CUSPARSE_SPMAT_FILL_MODE = 0
_CUSPARSE_SPMAT_DIAG_TYPE = 1
_CUSPARSE_FILL_MODE_LOWER = 0
_CUSPARSE_DIAG_TYPE_NON_UNIT = 0
_CUSPARSE_SPSM_ALG_DEFAULT = 0
_CUDA_R_32F = 0
_CUDA_R_64F = 1
_CUDA_C_32F = 4
_CUDA_C_64F = 5
_CUSPARSE_LIB = None
_CUSPARSE_LIB_LOAD_ERROR = None


def _dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def _reference_check_threshold(dtype):
    if dtype in (torch.float32, torch.complex64):
        return 1e-4
    return 1e-10


def _reference_max_relative_error(answer, result):
    if answer is None or result is None:
        return None
    if answer.numel() != result.numel():
        return float("inf")
    if answer.numel() == 0:
        return 0.0
    diff = torch.abs(answer - result)
    result_cmp = torch.abs(answer)
    if not bool(torch.isfinite(diff).all().item()) or not bool(
        torch.isfinite(result_cmp).all().item()
    ):
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


def _fmt_trace_times(times):
    if not times:
        return "[]"
    return "[" + ", ".join(f"{float(t):.6f}" for t in times) + "]"


def _emit_cusparse_trace(tag, warmup_times, timed_times, reduced_ms):
    if not TRACE_CUSPARSE:
        return
    print(
        f"[trace][{tag}] scope={_CUSPARSE_MEASUREMENT_SCOPE} "
        "(descriptor/input construction stays outside the timed region)"
    )
    if warmup_times:
        print(f"[trace][{tag}] warmup_ms={_fmt_trace_times(warmup_times)}")
    print(f"[trace][{tag}] timed_ms={_fmt_trace_times(timed_times)}")
    if timed_times:
        print(
            f"[trace][{tag}] first_timed_ms={float(timed_times[0]):.6f} "
            f"reduced_ms={float(reduced_ms):.6f}"
        )


def _spsm_benchmark_schedule(nnz, n_rhs, value_dtype, fmt="csr"):
    del nnz, n_rhs, value_dtype, fmt
    return int(WARMUP), int(ITERS)


def _allinone_filtered_avg_ms(times):
    if not times:
        return None
    times = [float(t) for t in times]
    if len(times) == 1:
        return times[0]
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
    diag = (torch.rand((n,), dtype=diag_base_dtype, device=device) + 2.0).to(
        value_dtype
    )
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


def _extract_effective_lower_csr(data, indices, indptr, shape):
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    tri_mask = col <= row
    data_tri = data[tri_mask]
    row_tri = row[tri_mask]
    col_tri = col[tri_mask]
    if row_tri.numel() == 0:
        empty_indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr.device)
        empty_indices = torch.empty((0,), dtype=torch.int64, device=indices.device)
        empty_data = data.new_empty((0,))
        return empty_data, empty_indices, empty_indptr
    order = torch.argsort(row_tri * n_rows + col_tri)
    row_tri = row_tri[order]
    col_tri = col_tri[order]
    data_tri = data_tri[order]
    counts = torch.bincount(row_tri, minlength=n_rows)
    indptr_tri = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr.device)
    indptr_tri[1:] = torch.cumsum(counts, dim=0)
    return data_tri, col_tri.to(torch.int64), indptr_tri


def _stabilize_lower_triangular_csr(data, indices, indptr, shape):
    """Make an arbitrary lower-triangular pattern safe for NON_UNIT SpSM.

    Matrix Market inputs are general sparse matrices, not guaranteed triangular
    systems. Preserve their lower-triangular off-diagonal values and sparsity,
    but insert/replace the diagonal with a strictly row-diagonally-dominant
    value. This preprocessing is shared by every backend and stays outside all
    analysis/solve timing.
    """
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    diag_mask = row == col
    offdiag_mask = ~diag_mask

    real_dtype = (
        torch.float32
        if data.dtype in (torch.float32, torch.complex64)
        else torch.float64
    )
    offdiag_abs_sum = torch.zeros(n_rows, dtype=real_dtype, device=data.device)
    if bool(torch.any(offdiag_mask).item()):
        offdiag_abs_sum.index_add_(
            0,
            row[offdiag_mask],
            torch.abs(data[offdiag_mask]).to(real_dtype),
        )
    stable_diag = (offdiag_abs_sum + 1.0).to(data.dtype)

    data_stable = data.clone()
    diag_present = torch.zeros(n_rows, dtype=torch.bool, device=data.device)
    if bool(torch.any(diag_mask).item()):
        diag_rows = row[diag_mask]
        data_stable[diag_mask] = stable_diag[diag_rows]
        diag_present[diag_rows] = True

    missing_diag = (
        torch.nonzero(~diag_present, as_tuple=False).reshape(-1).to(torch.int64)
    )
    if missing_diag.numel() > 0:
        row = torch.cat((row, missing_diag))
        col = torch.cat((col, missing_diag))
        data_stable = torch.cat((data_stable, stable_diag[missing_diag]))

    order = torch.argsort(row * max(1, n_rows) + col)
    row = row[order]
    col = col[order]
    data_stable = data_stable[order]
    counts = torch.bincount(row, minlength=n_rows)
    indptr_stable = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    indptr_stable[1:] = torch.cumsum(counts, dim=0)
    return data_stable, col.to(torch.int64), indptr_stable


def _benchmark_pytorch_reference(data, indices, indptr, shape, B):
    try:
        sparse_spsolve = getattr(torch.sparse, "spsolve", None)
        if sparse_spsolve is None:
            raise NotImplementedError("torch.sparse.spsolve is unavailable")
        data_eff, indices_eff, indptr_eff = _extract_effective_lower_csr(
            data, indices, indptr, shape
        )
        A_csr = torch.sparse_csr_tensor(
            indptr_eff,
            indices_eff,
            data_eff,
            size=shape,
            device=data.device,
        )
        if not A_csr.is_cuda:
            raise RuntimeError("torch.sparse.spsolve CUDA path is unavailable")
        cols = []
        for bj in torch.unbind(B, dim=1):
            cols.append(sparse_spsolve(A_csr, bj))
        X_ref = torch.stack(cols, dim=1) if cols else B.new_empty(B.shape)
        torch.cuda.synchronize()
        return X_ref.to(B.dtype), None
    except Exception as exc:
        if "out of memory" in str(exc).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, f"PyTorch sparse solve unavailable ({exc})"


class _CuComplexFloat(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class _CuComplexDouble(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


def _check_cusparse_status(status, op_name):
    if int(status) != _CUSPARSE_STATUS_SUCCESS:
        raise RuntimeError(f"{op_name} failed with cuSPARSE status {int(status)}")


def _cuda_data_type_from_torch(dtype):
    mapping = {
        torch.float32: _CUDA_R_32F,
        torch.float64: _CUDA_R_64F,
        torch.complex64: _CUDA_C_32F,
        torch.complex128: _CUDA_C_64F,
    }
    if dtype not in mapping:
        raise TypeError(f"Unsupported native cuSPARSE SpSM dtype: {dtype}")
    return mapping[dtype]


def _cusparse_index_type_from_torch(dtype):
    if dtype == torch.int32:
        return _CUSPARSE_INDEX_32I
    if dtype == torch.int64:
        return _CUSPARSE_INDEX_64I
    raise TypeError(f"Unsupported native cuSPARSE SpSM index dtype: {dtype}")


def _native_alpha_one(dtype):
    if dtype == torch.float32:
        return ctypes.c_float(1.0)
    if dtype == torch.float64:
        return ctypes.c_double(1.0)
    if dtype == torch.complex64:
        return _CuComplexFloat(1.0, 0.0)
    if dtype == torch.complex128:
        return _CuComplexDouble(1.0, 0.0)
    raise TypeError(f"Unsupported native cuSPARSE SpSM dtype: {dtype}")


def _configure_cusparse_spsm_api(lib):
    void_p = ctypes.c_void_p
    void_pp = ctypes.POINTER(void_p)
    int64 = ctypes.c_int64
    cint = ctypes.c_int

    lib.cusparseCreate.argtypes = [void_pp]
    lib.cusparseCreate.restype = cint
    lib.cusparseDestroy.argtypes = [void_p]
    lib.cusparseDestroy.restype = cint
    lib.cusparseSetStream.argtypes = [void_p, void_p]
    lib.cusparseSetStream.restype = cint
    lib.cusparseCreateCsr.argtypes = [
        void_pp,
        int64,
        int64,
        int64,
        void_p,
        void_p,
        void_p,
        cint,
        cint,
        cint,
        cint,
    ]
    lib.cusparseCreateCsr.restype = cint
    lib.cusparseCreateCoo.argtypes = [
        void_pp,
        int64,
        int64,
        int64,
        void_p,
        void_p,
        void_p,
        cint,
        cint,
        cint,
    ]
    lib.cusparseCreateCoo.restype = cint
    lib.cusparseDestroySpMat.argtypes = [void_p]
    lib.cusparseDestroySpMat.restype = cint
    lib.cusparseSpMatSetAttribute.argtypes = [
        void_p,
        cint,
        void_p,
        ctypes.c_size_t,
    ]
    lib.cusparseSpMatSetAttribute.restype = cint
    lib.cusparseCreateDnMat.argtypes = [
        void_pp,
        int64,
        int64,
        int64,
        void_p,
        cint,
        cint,
    ]
    lib.cusparseCreateDnMat.restype = cint
    lib.cusparseDestroyDnMat.argtypes = [void_p]
    lib.cusparseDestroyDnMat.restype = cint
    lib.cusparseSpSM_createDescr.argtypes = [void_pp]
    lib.cusparseSpSM_createDescr.restype = cint
    lib.cusparseSpSM_destroyDescr.argtypes = [void_p]
    lib.cusparseSpSM_destroyDescr.restype = cint
    lib.cusparseSpSM_bufferSize.argtypes = [
        void_p,
        cint,
        cint,
        void_p,
        void_p,
        void_p,
        void_p,
        cint,
        cint,
        void_p,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    lib.cusparseSpSM_bufferSize.restype = cint
    lib.cusparseSpSM_analysis.argtypes = [
        void_p,
        cint,
        cint,
        void_p,
        void_p,
        void_p,
        void_p,
        cint,
        cint,
        void_p,
        void_p,
    ]
    lib.cusparseSpSM_analysis.restype = cint
    lib.cusparseSpSM_solve.argtypes = [
        void_p,
        cint,
        cint,
        void_p,
        void_p,
        void_p,
        void_p,
        cint,
        cint,
        void_p,
    ]
    lib.cusparseSpSM_solve.restype = cint


def _load_cusparse_spsm_library():
    global _CUSPARSE_LIB, _CUSPARSE_LIB_LOAD_ERROR
    if _CUSPARSE_LIB is not None:
        return _CUSPARSE_LIB
    if _CUSPARSE_LIB_LOAD_ERROR is not None:
        raise RuntimeError(_CUSPARSE_LIB_LOAD_ERROR)

    candidates = []
    found_name = ctypes.util.find_library("cusparse")
    if found_name:
        candidates.append(found_name)
    candidates.extend(["libcusparse.so", "libcusparse.so.12", "libcusparse.so.11"])
    load_error = None
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            _configure_cusparse_spsm_api(lib)
            _CUSPARSE_LIB = lib
            return lib
        except Exception as exc:
            load_error = exc
    _CUSPARSE_LIB_LOAD_ERROR = "Failed to load native cuSPARSE SpSM API" + (
        f": {load_error}" if load_error is not None else ""
    )
    raise RuntimeError(_CUSPARSE_LIB_LOAD_ERROR)


def _current_cuda_stream_ptr():
    stream_ptr = getattr(torch.cuda.current_stream(), "cuda_stream", None)
    if stream_ptr is None:
        raise RuntimeError("Could not obtain the current CUDA stream pointer")
    return ctypes.c_void_p(int(stream_ptr))


class _PreparedCusparseNativeSpSM:
    def __init__(self, data, row, col, indptr, B, shape, fmt):
        if fmt not in FORMATS:
            raise ValueError(f"Unsupported native cuSPARSE SpSM format: {fmt}")
        if not all(t.is_cuda for t in (data, row, col, indptr, B)):
            raise ValueError("Native cuSPARSE SpSM inputs must be CUDA tensors")

        self.data = data.contiguous()
        self.row = row.contiguous()
        self.col = col.contiguous()
        self.indptr = indptr.contiguous()
        self.B = B.contiguous()
        rhs_cols = int(self.B.shape[1])
        rows = int(shape[0])
        # Reinterpret row-major Torch B as a column-major transposed matrix.
        # This matches the native cuSPARSE/CuPy SpSM path without copying B.
        self.X_storage = torch.empty(
            (rhs_cols, rows),
            dtype=self.B.dtype,
            device=self.B.device,
        )
        self.X = self.X_storage.transpose(0, 1)
        self.shape = (int(shape[0]), int(shape[1]))
        self.fmt = fmt
        self.op_b = _CUSPARSE_OPERATION_TRANSPOSE
        self.compute_type = _cuda_data_type_from_torch(self.data.dtype)
        self.index_type = _cusparse_index_type_from_torch(self.col.dtype)
        self.alpha = _native_alpha_one(self.data.dtype)
        self.lib = _load_cusparse_spsm_library()
        self.handle = ctypes.c_void_p()
        self.mat_a = ctypes.c_void_p()
        self.mat_b = ctypes.c_void_p()
        self.mat_x = ctypes.c_void_p()
        self.spsm_descr = ctypes.c_void_p()
        self.workspace = None
        self.closed = False

        try:
            self._create_descriptors()
        except Exception:
            self.close()
            raise

    def _create_descriptors(self):
        rows, cols = self.shape
        nnz = int(self.data.numel())
        _check_cusparse_status(
            self.lib.cusparseCreate(ctypes.byref(self.handle)),
            "cusparseCreate",
        )
        _check_cusparse_status(
            self.lib.cusparseSetStream(self.handle, _current_cuda_stream_ptr()),
            "cusparseSetStream",
        )
        if self.fmt == "csr":
            row_type = _cusparse_index_type_from_torch(self.indptr.dtype)
            _check_cusparse_status(
                self.lib.cusparseCreateCsr(
                    ctypes.byref(self.mat_a),
                    ctypes.c_int64(rows),
                    ctypes.c_int64(cols),
                    ctypes.c_int64(nnz),
                    ctypes.c_void_p(int(self.indptr.data_ptr())),
                    ctypes.c_void_p(int(self.col.data_ptr())),
                    ctypes.c_void_p(int(self.data.data_ptr())),
                    ctypes.c_int(row_type),
                    ctypes.c_int(self.index_type),
                    ctypes.c_int(_CUSPARSE_INDEX_BASE_ZERO),
                    ctypes.c_int(self.compute_type),
                ),
                "cusparseCreateCsr",
            )
        else:
            row_type = _cusparse_index_type_from_torch(self.row.dtype)
            if row_type != self.index_type:
                raise TypeError("COO row and column index dtypes must match")
            _check_cusparse_status(
                self.lib.cusparseCreateCoo(
                    ctypes.byref(self.mat_a),
                    ctypes.c_int64(rows),
                    ctypes.c_int64(cols),
                    ctypes.c_int64(nnz),
                    ctypes.c_void_p(int(self.row.data_ptr())),
                    ctypes.c_void_p(int(self.col.data_ptr())),
                    ctypes.c_void_p(int(self.data.data_ptr())),
                    ctypes.c_int(self.index_type),
                    ctypes.c_int(_CUSPARSE_INDEX_BASE_ZERO),
                    ctypes.c_int(self.compute_type),
                ),
                "cusparseCreateCoo",
            )

        fill_mode = ctypes.c_int(_CUSPARSE_FILL_MODE_LOWER)
        diag_type = ctypes.c_int(_CUSPARSE_DIAG_TYPE_NON_UNIT)
        _check_cusparse_status(
            self.lib.cusparseSpMatSetAttribute(
                self.mat_a,
                ctypes.c_int(_CUSPARSE_SPMAT_FILL_MODE),
                ctypes.byref(fill_mode),
                ctypes.sizeof(fill_mode),
            ),
            "cusparseSpMatSetAttribute(fill)",
        )
        _check_cusparse_status(
            self.lib.cusparseSpMatSetAttribute(
                self.mat_a,
                ctypes.c_int(_CUSPARSE_SPMAT_DIAG_TYPE),
                ctypes.byref(diag_type),
                ctypes.sizeof(diag_type),
            ),
            "cusparseSpMatSetAttribute(diag)",
        )

        rhs_cols = int(self.B.shape[1])
        _check_cusparse_status(
            self.lib.cusparseCreateDnMat(
                ctypes.byref(self.mat_b),
                ctypes.c_int64(rhs_cols),
                ctypes.c_int64(rows),
                ctypes.c_int64(rhs_cols),
                ctypes.c_void_p(int(self.B.data_ptr())),
                ctypes.c_int(self.compute_type),
                ctypes.c_int(_CUSPARSE_ORDER_COL),
            ),
            "cusparseCreateDnMat(B)",
        )
        _check_cusparse_status(
            self.lib.cusparseCreateDnMat(
                ctypes.byref(self.mat_x),
                ctypes.c_int64(rows),
                ctypes.c_int64(rhs_cols),
                ctypes.c_int64(rows),
                ctypes.c_void_p(int(self.X_storage.data_ptr())),
                ctypes.c_int(self.compute_type),
                ctypes.c_int(_CUSPARSE_ORDER_COL),
            ),
            "cusparseCreateDnMat(X)",
        )
        _check_cusparse_status(
            self.lib.cusparseSpSM_createDescr(ctypes.byref(self.spsm_descr)),
            "cusparseSpSM_createDescr",
        )

    def prepare_and_solve(self):
        if self.closed:
            raise RuntimeError("Prepared native cuSPARSE SpSM plan is closed")
        buffer_size = ctypes.c_size_t()
        _check_cusparse_status(
            self.lib.cusparseSpSM_bufferSize(
                self.handle,
                ctypes.c_int(_CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(self.op_b),
                ctypes.byref(self.alpha),
                self.mat_a,
                self.mat_b,
                self.mat_x,
                ctypes.c_int(self.compute_type),
                ctypes.c_int(_CUSPARSE_SPSM_ALG_DEFAULT),
                self.spsm_descr,
                ctypes.byref(buffer_size),
            ),
            "cusparseSpSM_bufferSize",
        )
        self.workspace = torch.empty(
            max(1, int(buffer_size.value)),
            dtype=torch.uint8,
            device=self.data.device,
        )
        workspace_ptr = (
            ctypes.c_void_p(int(self.workspace.data_ptr()))
            if int(buffer_size.value) > 0
            else ctypes.c_void_p()
        )
        _check_cusparse_status(
            self.lib.cusparseSpSM_analysis(
                self.handle,
                ctypes.c_int(_CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(self.op_b),
                ctypes.byref(self.alpha),
                self.mat_a,
                self.mat_b,
                self.mat_x,
                ctypes.c_int(self.compute_type),
                ctypes.c_int(_CUSPARSE_SPSM_ALG_DEFAULT),
                self.spsm_descr,
                workspace_ptr,
            ),
            "cusparseSpSM_analysis",
        )
        _check_cusparse_status(
            self.lib.cusparseSpSM_solve(
                self.handle,
                ctypes.c_int(_CUSPARSE_OPERATION_NON_TRANSPOSE),
                ctypes.c_int(self.op_b),
                ctypes.byref(self.alpha),
                self.mat_a,
                self.mat_b,
                self.mat_x,
                ctypes.c_int(self.compute_type),
                ctypes.c_int(_CUSPARSE_SPSM_ALG_DEFAULT),
                self.spsm_descr,
            ),
            "cusparseSpSM_solve",
        )
        return self.X

    def reset_analysis_state(self):
        if self.closed:
            raise RuntimeError("Prepared native cuSPARSE SpSM plan is closed")
        self.workspace = None
        if self.spsm_descr.value:
            _check_cusparse_status(
                self.lib.cusparseSpSM_destroyDescr(self.spsm_descr),
                "cusparseSpSM_destroyDescr",
            )
            self.spsm_descr = ctypes.c_void_p()
        _check_cusparse_status(
            self.lib.cusparseSpSM_createDescr(ctypes.byref(self.spsm_descr)),
            "cusparseSpSM_createDescr",
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.spsm_descr.value:
            try:
                self.lib.cusparseSpSM_destroyDescr(self.spsm_descr)
            except Exception:
                pass
            self.spsm_descr = ctypes.c_void_p()
        for attr in ("mat_x", "mat_b"):
            desc = getattr(self, attr)
            if desc.value:
                try:
                    self.lib.cusparseDestroyDnMat(desc)
                except Exception:
                    pass
                setattr(self, attr, ctypes.c_void_p())
        if self.mat_a.value:
            try:
                self.lib.cusparseDestroySpMat(self.mat_a)
            except Exception:
                pass
            self.mat_a = ctypes.c_void_p()
        if self.handle.value:
            try:
                self.lib.cusparseDestroy(self.handle)
            except Exception:
                pass
            self.handle = ctypes.c_void_p()


def _benchmark_cusparse_reference(data, row, col, indptr, B, shape, fmt, warmup, iters):
    # Vendor triangular-solve baseline per backend: hipSPARSE csrsm2 on DCU/ROCm,
    # native cuSPARSE SpSM on CUDA.
    if fs_spsm_impl._is_rocm_runtime() and indptr is not None:
        sparse_ref = fs_spsm_impl._benchmark_spsm_csr_sparse_ref(
            data,
            col if fmt == "csr" else row,
            indptr,
            B,
            shape,
            warmup=warmup,
            iters=iters,
        )
        if sparse_ref["backend"] is None:
            return None, None, sparse_ref["reason"]
        return sparse_ref["values"], sparse_ref["ms"], None
    plan = None
    try:
        plan = _PreparedCusparseNativeSpSM(data, row, col, indptr, B, shape, fmt)
        warmup_times = []
        for _ in range(warmup):
            plan.reset_analysis_state()
            torch.cuda.synchronize()
            start = time.perf_counter()
            X_t = plan.prepare_and_solve()
            torch.cuda.synchronize()
            if TRACE_CUSPARSE:
                warmup_times.append((time.perf_counter() - start) * 1000.0)
        times = []
        for _ in range(iters):
            plan.reset_analysis_state()
            torch.cuda.synchronize()
            start = time.perf_counter()
            X_t = plan.prepare_and_solve()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000.0)
        total_ms = _allinone_filtered_avg_ms(times)
        _emit_cusparse_trace(
            f"SpSM fmt={fmt.upper()} shape={shape} rhs={B.shape[1]}",
            warmup_times,
            times,
            total_ms,
        )
        return X_t, total_ms, None
    except Exception as exc:
        return None, None, str(exc)
    finally:
        if plan is not None:
            plan.close()


def _normalized_solution_residual(data, indices, indptr, shape, X, B):
    n_rows = int(shape[0])
    row, col = _csr_to_coo(indices, indptr, n_rows)
    if n_rows == 0 or B.numel() == 0:
        return 0.0

    check_dtype = (
        torch.complex128
        if data.dtype in (torch.complex64, torch.complex128)
        else torch.float64
    )
    data_check = data.to(check_dtype)
    row_abs_sum = torch.zeros(n_rows, dtype=torch.float64, device=data.device)
    row_abs_sum.index_add_(0, row, torch.abs(data_check).to(torch.float64))
    a_norm = torch.max(row_abs_sum)

    residual_row_sum = torch.zeros_like(row_abs_sum)
    x_row_sum = torch.zeros_like(row_abs_sum)
    b_row_sum = torch.zeros_like(row_abs_sum)
    rhs_chunk = 8
    for start in range(0, int(B.shape[1]), rhs_chunk):
        end = min(start + rhs_chunk, int(B.shape[1]))
        X_chunk = X[:, start:end].to(check_dtype)
        B_chunk = B[:, start:end].to(check_dtype)
        B_recon = torch.zeros((n_rows, end - start), dtype=check_dtype, device=X.device)
        B_recon.index_add_(0, row, data_check[:, None] * X_chunk[col])
        residual_row_sum += torch.sum(torch.abs(B_recon - B_chunk), dim=1).to(
            torch.float64
        )
        x_row_sum += torch.sum(torch.abs(X_chunk), dim=1).to(torch.float64)
        b_row_sum += torch.sum(torch.abs(B_chunk), dim=1).to(torch.float64)

    residual_norm = torch.max(residual_row_sum)
    x_norm = torch.max(x_row_sum)
    b_norm = torch.max(b_row_sum)
    scale = a_norm * x_norm + b_norm
    scale_value = float(scale.item())
    if scale_value == 0.0:
        return 0.0 if float(residual_norm.item()) == 0.0 else float("inf")
    return float((residual_norm / scale).item())


def _benchmark_flagsparse_full_round(
    reset_call, analyze_call, solve_call, warmup, iters
):
    X = None
    for _ in range(warmup):
        reset_call()
        torch.cuda.synchronize()
        analyze_call()
        X = solve_call()
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        reset_call()
        torch.cuda.synchronize()
        start = time.perf_counter()
        analyze_call()
        X = solve_call()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return X, _allinone_filtered_avg_ms(times)


def _benchmark_flagsparse_spsm_csr_total(data, indices, indptr, B, shape):
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), B.shape[1], data.dtype, fmt="csr"
    )
    analyze_call = lambda: fs_spsm_impl._analyze_spsm_csr(
        data,
        indices,
        indptr,
        B,
        shape,
        lower=True,
        unit_diagonal=False,
        clear_cache=False,
        return_time=False,
    )
    solve_call = lambda: fs.flagsparse_spsm_csr(
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
    )
    return _benchmark_flagsparse_full_round(
        fs_spsm_impl._clear_spsm_preprocess_cache,
        analyze_call,
        solve_call,
        warmup,
        iters,
    )


def _benchmark_flagsparse_spsm_coo_total(data, row, col, B, shape):
    warmup, iters = _spsm_benchmark_schedule(
        data.numel(), B.shape[1], data.dtype, fmt="coo"
    )
    analyze_call = lambda: fs_spsm_impl._analyze_spsm_coo(
        data,
        row,
        col,
        B,
        shape,
        lower=True,
        unit_diagonal=False,
        clear_cache=False,
        return_time=False,
    )
    solve_call = lambda: fs.flagsparse_spsm_coo(
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
    )
    return _benchmark_flagsparse_full_round(
        fs_spsm_impl._clear_spsm_preprocess_cache,
        analyze_call,
        solve_call,
        warmup,
        iters,
    )


def _load_mtx_to_csr_torch(file_path, dtype=torch.float32, device=None):
    """Load a square .mtx into CSR torch tensors via the C-accelerated scipy
    reader (see tests/mtx_fast.py). Triangular stabilization happens downstream
    in _stabilize_lower_triangular_csr, so this only needs the raw CSR."""
    from mtx_fast import load_csr

    data, indices, indptr, shape = load_csr(file_path, dtype=dtype, device=device)
    if shape[0] != shape[1]:
        raise NonSquareMatrixError("SpSM", shape)
    return data, indices, indptr, shape

def _run_one_spsm_case(
    data, indices, indptr, shape, value_dtype, index_dtype, n_rhs, fmt
):
    n_rows = int(shape[0])
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=data.device).contiguous()
    data_eff, indices_eff, indptr_eff = _extract_effective_lower_csr(
        data, indices, indptr, shape
    )
    data_eff, indices_eff, indptr_eff = _stabilize_lower_triangular_csr(
        data_eff, indices_eff, indptr_eff, shape
    )
    row, col = _csr_to_coo(indices_eff, indptr_eff, n_rows)
    warmup, iters = _spsm_benchmark_schedule(
        data_eff.numel(), n_rhs, value_dtype, fmt=fmt
    )

    if fmt == "csr":
        X_fs, flagsparse_ms = _benchmark_flagsparse_spsm_csr_total(
            data_eff,
            indices_eff.to(index_dtype),
            indptr_eff.to(index_dtype),
            B,
            shape,
        )
    else:
        X_fs, flagsparse_ms = _benchmark_flagsparse_spsm_coo_total(
            data_eff,
            row.to(index_dtype),
            col.to(index_dtype),
            B,
            shape,
        )
    (
        X_cu,
        cusparse_ms,
        _cusparse_reason,
    ) = _benchmark_cusparse_reference(
        data_eff, row, col, indptr_eff, B, shape, fmt, warmup, iters
    )
    X_pt, pytorch_reason = _benchmark_pytorch_reference(
        data_eff, indices_eff, indptr_eff, shape, B
    )

    err_cu = None
    ok_cu = None
    if X_cu is not None:
        err_cu = _reference_max_relative_error(X_cu, X_fs)
        ok_cu = err_cu <= _reference_check_threshold(value_dtype)

    err_pt = None
    ok_pt = None
    if X_pt is not None:
        err_pt = _reference_max_relative_error(X_pt, X_fs)
        ok_pt = err_pt <= _reference_check_threshold(value_dtype)

    err_res = _normalized_solution_residual(
        data_eff, indices_eff, indptr_eff, shape, X_fs, B
    )
    residual_ok = math.isfinite(err_res) and (
        err_res <= _reference_check_threshold(value_dtype)
    )
    err_ref = err_cu if err_cu is not None else err_pt

    if ok_cu is not None:
        status = "PASS" if ok_cu and residual_ok else "FAIL"
    elif ok_pt is not None:
        status = "PASS" if ok_pt and residual_ok else "FAIL"
    elif X_pt is None and X_cu is None:
        status = "REF_FAIL"
    else:
        status = "FAIL"

    return {
        "format": fmt,
        "n_rows": n_rows,
        "n_cols": int(shape[1]),
        "nnz": int(data_eff.numel()),
        "n_rhs": int(n_rhs),
        "FlagSparse_ms": flagsparse_ms,
        "cuSPARSE_ms": cusparse_ms,
        "FlagSparse_vs_cuSPARSE_speedup": _safe_ratio(cusparse_ms, flagsparse_ms),
        "status": status,
        "err_ref": err_ref,
        "err_res": err_res,
        "err_pt": err_pt,
        "err_cu": err_cu,
        "cusparse_reason": _cusparse_reason,
        "pytorch_reason": pytorch_reason,
        "error": None,
    }


def run_spsm_synthetic_all(n=512, n_rhs=1024):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    total = 0
    failed = 0
    print("=" * 138)
    print("FLAGSPARSE SpSM synthetic test")
    print("=" * 138)
    print("Every timed round performs analysis/preparation + solve.")
    print(
        f"{'Fmt':>5} {'dtype':>9} {'index':>7} {'N':>6} {'RHS':>6} {'NNZ':>10} "
        f"{'FS(ms)':>10} {'CU(ms)':>10} {'FS/CU':>10} "
        f"{'Status':>10} {'Err(Ref)':>12} {'Err(Res)':>12} {'Err(PT)':>12} {'Err(CU)':>12}"
    )
    print("-" * 138)

    for fmt in FORMATS:
        for value_dtype in VALUE_DTYPES:
            for index_dtype in INDEX_DTYPES:
                data, _, col, indptr, _, shape = _build_triangular_case(
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
                    f"{_fmt_ms(one['FlagSparse_ms']):>10} {_fmt_ms(one['cuSPARSE_ms']):>10} "
                    f"{_fmt_ratio(one['FlagSparse_vs_cuSPARSE_speedup']):>10} "
                    f"{one['status']:>10} {_fmt_err(one['err_ref']):>12} {_fmt_err(one['err_res']):>12} "
                    f"{_fmt_err(one['err_pt']):>12} {_fmt_err(one['err_cu']):>12}"
                )
                if one["status"] in ("FAIL", "REF_FAIL"):
                    if one["cusparse_reason"]:
                        print(
                            f"  NOTE: native cuSPARSE unavailable: {one['cusparse_reason']}"
                        )
                    if one["pytorch_reason"]:
                        print(f"  NOTE: {one['pytorch_reason']}")
    print("-" * 138)
    print(f"Total cases: {total}  Failed: {failed}")
    print("=" * 138)


def run_all_dtypes_spsm_csv(mtx_paths, csv_path, use_coo=False, n_rhs=1024):
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return
    device = torch.device("cuda")
    records_out = []
    fmt = "coo" if use_coo else "csr"

    print("=" * 146)
    print(
        f"FLAGSPARSE SpSM .mtx batch ({fmt.upper()}) | "
        "FS/CU: every round analysis/preparation + solve"
    )
    print("=" * 146)
    print(
        f"Benchmark schedule: warmup={WARMUP}, timed_iters={ITERS} "
        "(filtered per-round averages; override with --warmup/--iters)"
    )
    print(
        "FS(ms) and CU(ms) each include one fresh analysis/preparation plus one solve. "
        "FS/CU = CU(ms) / FS(ms)."
    )
    print(
        f"{'Matrix':<28} {'dtype':>9} {'index':>7} {'N':>7} {'RHS':>6} {'NNZ':>10} "
        f"{'FS(ms)':>10} {'CU(ms)':>10} {'FS/CU':>10} "
        f"{'Status':>10} {'Eref':>12} {'Eres':>12} {'Ept':>12} {'Ecu':>12}"
    )
    print("-" * 146)

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
                    record = _run_one_spsm_case(
                        data,
                        indices,
                        indptr,
                        shape,
                        value_dtype,
                        index_dtype,
                        n_rhs,
                        fmt,
                    )
                    record = {**base, **record}
                    records_out.append(record)
                    short = base["matrix"][:27] + (
                        "…" if len(base["matrix"]) > 27 else ""
                    )
                    print(
                        f"{short:<28} {base['value_dtype']:>9} {base['index_dtype']:>7} "
                        f"{record['n_rows']:>7} {record['n_rhs']:>6} {record['nnz']:>10} "
                        f"{_fmt_ms(record['FlagSparse_ms']):>10} {_fmt_ms(record['cuSPARSE_ms']):>10} "
                        f"{_fmt_ratio(record['FlagSparse_vs_cuSPARSE_speedup']):>10} "
                        f"{record['status']:>10} {_fmt_err(record['err_ref']):>12} {_fmt_err(record['err_res']):>12} "
                        f"{_fmt_err(record['err_pt']):>12} {_fmt_err(record['err_cu']):>12}"
                    )
                    if record["status"] in ("FAIL", "REF_FAIL"):
                        if record["cusparse_reason"]:
                            print(
                                "  NOTE: native cuSPARSE unavailable: "
                                f"{record['cusparse_reason']}"
                            )
                        if record["pytorch_reason"]:
                            print(f"  NOTE: {record['pytorch_reason']}")
                except Exception as exc:
                    err_msg = str(exc)
                    if _is_fatal_cuda_error(exc):
                        print(
                            f"  FATAL CUDA ERROR: {exc}\n"
                            "  CUDA context is no longer reliable. Restart the Python/Singularity session "
                            "and rerun the single failing matrix with CUDA_LAUNCH_BLOCKING=1."
                        )
                        raise
                    is_skip = isinstance(exc, NonSquareMatrixError)
                    status = "SKIP" if is_skip else "ERROR"
                    n_rows, n_cols = exc.shape if is_skip else ("ERR", "ERR")
                    nnz = "N/A" if is_skip else "ERR"
                    record = {
                        **base,
                        "format": fmt,
                        "n_rows": n_rows,
                        "n_cols": n_cols,
                        "nnz": nnz,
                        "n_rhs": int(n_rhs),
                        "FlagSparse_ms": None,
                        "cuSPARSE_ms": None,
                        "FlagSparse_vs_cuSPARSE_speedup": None,
                        "status": status,
                        "err_ref": None,
                        "err_res": None,
                        "err_pt": None,
                        "err_cu": None,
                        "cusparse_reason": None,
                        "pytorch_reason": None,
                        "error": err_msg,
                    }
                    records_out.append(record)
                    short = base["matrix"][:27] + (
                        "…" if len(base["matrix"]) > 27 else ""
                    )
                    print(
                        f"{short:<28} {base['value_dtype']:>9} {base['index_dtype']:>7} "
                        f"{str(n_rows):>7} {int(n_rhs):>6} {str(nnz):>10} "
                        f"{_fmt_ms(None):>10} {_fmt_ms(None):>10} "
                        f"{'N/A':>10} {status:>10} "
                        f"{_fmt_err(None):>12} {_fmt_err(None):>12} {_fmt_err(None):>12} {_fmt_err(None):>12}"
                    )
                    print(f"  {status}: {exc}")

    print("-" * 146)
    fieldnames = [
        "matrix",
        "value_dtype",
        "index_dtype",
        "format",
        "n_rows",
        "n_cols",
        "nnz",
        "n_rhs",
        "FlagSparse_ms",
        "cuSPARSE_ms",
        "FlagSparse_vs_cuSPARSE_speedup",
        "status",
        "err_ref",
        "err_res",
        "err_pt",
        "err_cu",
        "cusparse_reason",
        "pytorch_reason",
        "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for record in records_out:
            w.writerow({k: ("" if v is None else v) for k, v in record.items()})
    print(f"Wrote {len(records_out)} rows to {csv_path}")


def main():
    global WARMUP, ITERS, TRACE_CUSPARSE
    parser = argparse.ArgumentParser(
        description="SpSM test: synthetic triangular systems and optional .mtx batch CSV."
    )
    parser.add_argument(
        "mtx",
        nargs="*",
        help=".mtx file path(s), or directory(ies) to glob for *.mtx",
    )
    parser.add_argument(
        "--synthetic", action="store_true", help="Run synthetic triangular tests"
    )
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
        help="Warmup rounds; each round performs analysis/preparation + solve",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=ITERS,
        help="Timed analysis/preparation + solve rounds (default: 20)",
    )
    parser.add_argument(
        "--print-cusparse-times",
        action="store_true",
        help="Print native cuSPARSE preparation + solve time for every round.",
    )
    args = parser.parse_args()
    WARMUP = max(0, int(args.warmup))
    ITERS = max(1, int(args.iters))
    TRACE_CUSPARSE = bool(args.print_cusparse_times)

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
