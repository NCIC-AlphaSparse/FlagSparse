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

"""Native cuSPARSE baselines for the benchmark scripts.

SpVV, Axpby and SELL SpMV for the q4 operators, plus CSR/COO/CSC SpMV, CSR/COO SpMM
and CSR SDDMM for the q4 *variants* of existing operators: the mixed-precision and
integer type combinations (f16 -> f32, i8 -> i32, real A x complex x, ...) that
the CuPy-based baselines of the existing scripts cannot express.

ctypes over the generic API, so no CuPy build is needed (CuPy wraps none of these).
Descriptors are created once in ``prepare``; ``run`` issues only the library call,
so the timed window holds the same work as FlagSparse's run step.

Only CUDA: on other backends ``skip_reason()`` names why, and the benchmark writes
an empty ``cusparse_ms`` -- the row then has no vendor baseline, which is what
``tools/baseline_bound.py`` is for.
"""

import ctypes
import ctypes.util

import torch

_CUDA_TYPES = {
    torch.float16: 2,
    torch.bfloat16: 14,
    torch.float32: 0,
    torch.float64: 1,
    torch.complex64: 4,
    torch.complex128: 5,
    torch.int8: 3,
    torch.int32: 10,
}
_INDEX_32I = 2
_INDEX_64I = 3
_BASE_ZERO = 0
_OP = {"non": 0, "trans": 1, "conj": 2}
_SPMV_SELL_ALG1 = 5
_ALG_DEFAULT = 0
_ORDER_COL, _ORDER_ROW = 1, 2

_LIB = None
_LIB_ERROR = None
_HANDLE = None

_P = ctypes.c_void_p
_PP = ctypes.POINTER(ctypes.c_void_p)
_I = ctypes.c_int
_I64 = ctypes.c_int64
_SIGNATURES = {
    "cusparseCreate": [_PP],
    "cusparseSetStream": [_P, _P],
    "cusparseCreateSpVec": [_PP, _I64, _I64, _P, _P, _I, _I, _I],
    "cusparseDestroySpVec": [_P],
    "cusparseCreateDnVec": [_PP, _I64, _P, _I],
    "cusparseDestroyDnVec": [_P],
    "cusparseCreateSlicedEll": [_PP, _I64, _I64, _I64, _I64, _I64, _P, _P, _P, _I, _I, _I, _I],
    "cusparseDestroySpMat": [_P],
    "cusparseSpVV_bufferSize": [_P, _I, _P, _P, _P, _I, ctypes.POINTER(ctypes.c_size_t)],
    "cusparseSpVV": [_P, _I, _P, _P, _P, _I, _P],
    "cusparseAxpby": [_P, _P, _P, _P, _P],
    "cusparseSpMV_bufferSize": [_P, _I, _P, _P, _P, _P, _P, _I, _I, ctypes.POINTER(ctypes.c_size_t)],
    "cusparseSpMV": [_P, _I, _P, _P, _P, _P, _P, _I, _I, _P],
    "cusparseCreateCsr": [_PP, _I64, _I64, _I64, _P, _P, _P, _I, _I, _I, _I],
    "cusparseCreateCsc": [_PP, _I64, _I64, _I64, _P, _P, _P, _I, _I, _I, _I],
    "cusparseCreateCoo": [_PP, _I64, _I64, _I64, _P, _P, _P, _I, _I, _I],
    "cusparseCreateDnMat": [_PP, _I64, _I64, _I64, _P, _I, _I],
    "cusparseCreateBlockedEll": [_PP, _I64, _I64, _I64, _I64, _P, _P, _I, _I, _I],
    "cusparseDestroyDnMat": [_P],
    "cusparseSpMM_bufferSize": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, ctypes.POINTER(ctypes.c_size_t)],
    "cusparseSpMM": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P],
    "cusparseSDDMM_bufferSize": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, ctypes.POINTER(ctypes.c_size_t)],
    "cusparseSDDMM_preprocess": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P],
    "cusparseSDDMM": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P],
}


def _lib():
    global _LIB, _LIB_ERROR
    if _LIB is not None:
        return _LIB
    if _LIB_ERROR is not None:
        raise RuntimeError(_LIB_ERROR)
    names = [ctypes.util.find_library("cusparse"), "libcusparse.so.12", "libcusparse.so"]
    for name in filter(None, names):
        try:
            lib = ctypes.CDLL(name)
            for fn, args in _SIGNATURES.items():
                getattr(lib, fn).argtypes = args
                getattr(lib, fn).restype = _I
            _LIB = lib
            return lib
        except (OSError, AttributeError) as exc:
            _LIB_ERROR = f"cannot load cuSPARSE generic API from {name}: {exc}"
    raise RuntimeError(_LIB_ERROR or "libcusparse not found")


def skip_reason():
    """None when the cuSPARSE baseline can run here, otherwise why not."""
    if getattr(torch.version, "hip", None) is not None:
        return "no hipSPARSE baseline wired for this operator (ROCm)"
    try:
        from flagsparse.sparse_operations._common import _backend_name

        backend = _backend_name()
    except Exception:
        backend = "cuda"
    if backend != "cuda":
        return f"no vendor sparse baseline wired for this operator on {backend}"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    try:
        _lib()
    except RuntimeError as exc:
        return str(exc)
    return None


def _check(status, what):
    if status != 0:
        raise RuntimeError(f"{what} failed with cuSPARSE status {status}")


def _handle():
    global _HANDLE
    if _HANDLE is None:
        h = _P()
        _check(_lib().cusparseCreate(ctypes.byref(h)), "cusparseCreate")
        _HANDLE = h
    stream = torch.cuda.current_stream().cuda_stream
    _check(_lib().cusparseSetStream(_HANDLE, _P(stream)), "cusparseSetStream")
    return _HANDLE


def _index_type(t):
    return _INDEX_64I if t.dtype == torch.int64 else _INDEX_32I


def _scalar(value, dtype):
    """Host scalar in the compute type (pointer mode is HOST by default)."""
    if dtype in (torch.complex64,):
        return (ctypes.c_float * 2)(complex(value).real, complex(value).imag)
    if dtype == torch.complex128:
        return (ctypes.c_double * 2)(complex(value).real, complex(value).imag)
    if dtype == torch.float64:
        return ctypes.c_double(float(value))
    if dtype == torch.int32:
        return ctypes.c_int32(int(value))
    return ctypes.c_float(float(value))


class _Plan:
    """Descriptors, workspace and the call; ``close`` releases the descriptors."""

    def __init__(self, run, owners, destroy):
        self._run = run
        self._owners = owners  # keeps tensors / scalars alive while descriptors point at them
        self._destroy = destroy

    def run(self):
        return self._run()

    def close(self):
        for fn, handle in self._destroy:
            fn(handle)
        self._destroy = []


def _spvec(values, indices, size):
    h = _P()
    _check(
        _lib().cusparseCreateSpVec(
            ctypes.byref(h), size, values.numel(), _P(indices.data_ptr()), _P(values.data_ptr()),
            _index_type(indices), _BASE_ZERO, _CUDA_TYPES[values.dtype],
        ),
        "cusparseCreateSpVec",
    )
    return h


def _dnvec(t):
    h = _P()
    _check(
        _lib().cusparseCreateDnVec(ctypes.byref(h), t.numel(), _P(t.data_ptr()), _CUDA_TYPES[t.dtype]),
        "cusparseCreateDnVec",
    )
    return h


def prepare_spvv(values, indices, y, op, compute_dtype):
    lib, handle = _lib(), _handle()
    x_d, y_d = _spvec(values, indices, y.numel()), _dnvec(y)
    ctype = _CUDA_TYPES[compute_dtype]
    result = _scalar(0, compute_dtype)
    size = ctypes.c_size_t(0)
    _check(
        lib.cusparseSpVV_bufferSize(handle, _OP[op], x_d, y_d, ctypes.byref(result), ctype, ctypes.byref(size)),
        "cusparseSpVV_bufferSize",
    )
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=y.device)

    def run():
        _check(
            lib.cusparseSpVV(handle, _OP[op], x_d, y_d, ctypes.byref(result), ctype, _P(work.data_ptr())),
            "cusparseSpVV",
        )
        if compute_dtype.is_complex:
            return complex(result[0], result[1])
        return result.value

    destroy = [(lib.cusparseDestroySpVec, x_d), (lib.cusparseDestroyDnVec, y_d)]
    return _Plan(run, (values, indices, y, work, result), destroy)


def prepare_axpby(values, indices, y, alpha, beta, compute_dtype):
    lib, handle = _lib(), _handle()
    x_d, y_d = _spvec(values, indices, y.numel()), _dnvec(y)
    a, b = _scalar(alpha, compute_dtype), _scalar(beta, compute_dtype)

    def run():
        _check(lib.cusparseAxpby(handle, ctypes.byref(a), x_d, ctypes.byref(b), y_d), "cusparseAxpby")
        return y

    destroy = [(lib.cusparseDestroySpVec, x_d), (lib.cusparseDestroyDnVec, y_d)]
    return _Plan(run, (values, indices, y, a, b), destroy)


def prepare_spmv_sell(values, cols, offsets, x, y, shape, slice_size, compute_dtype):
    lib, handle = _lib(), _handle()
    m = _P()
    nnz = int((cols >= 0).sum().item())
    _check(
        lib.cusparseCreateSlicedEll(
            ctypes.byref(m), int(shape[0]), int(shape[1]), nnz, values.numel(), int(slice_size),
            _P(offsets.data_ptr()), _P(cols.data_ptr()), _P(values.data_ptr()),
            _index_type(offsets), _index_type(cols), _BASE_ZERO, _CUDA_TYPES[values.dtype],
        ),
        "cusparseCreateSlicedEll",
    )
    x_d, y_d = _dnvec(x), _dnvec(y)
    ctype = _CUDA_TYPES[compute_dtype]
    one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
    size = ctypes.c_size_t(0)
    _check(
        lib.cusparseSpMV_bufferSize(
            handle, _OP["non"], ctypes.byref(one), m, x_d, ctypes.byref(zero), y_d, ctype,
            _SPMV_SELL_ALG1, ctypes.byref(size),
        ),
        "cusparseSpMV_bufferSize",
    )
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=x.device)

    def run():
        _check(
            lib.cusparseSpMV(
                handle, _OP["non"], ctypes.byref(one), m, x_d, ctypes.byref(zero), y_d, ctype,
                _SPMV_SELL_ALG1, _P(work.data_ptr()),
            ),
            "cusparseSpMV",
        )
        return y

    destroy = [(lib.cusparseDestroySpMat, m), (lib.cusparseDestroyDnVec, x_d), (lib.cusparseDestroyDnVec, y_d)]
    return _Plan(run, (values, cols, offsets, x, y, work, one, zero), destroy)


# ---------------------------------------------------------------------------
# Sparse-matrix operators for the q4 variants of existing operators
# ---------------------------------------------------------------------------


def _spmat(fmt, arrays, shape, value_dtype):
    """``arrays``: csr (values, col_indices, row_offsets), csc (values, row_indices,
    col_offsets), coo (values, row_indices, col_indices), bell (values as a
    rows x ellCols matrix, block column indices, block size)."""
    lib = _lib()
    h = _P()
    values, a, b = arrays
    rows, cols = int(shape[0]), int(shape[1])
    ctype = _CUDA_TYPES[value_dtype]
    if fmt == "csr":
        status = lib.cusparseCreateCsr(
            ctypes.byref(h), rows, cols, values.numel(), _P(b.data_ptr()), _P(a.data_ptr()),
            _P(values.data_ptr()), _index_type(b), _index_type(a), _BASE_ZERO, ctype,
        )
    elif fmt == "csc":
        status = lib.cusparseCreateCsc(
            ctypes.byref(h), rows, cols, values.numel(), _P(b.data_ptr()), _P(a.data_ptr()),
            _P(values.data_ptr()), _index_type(b), _index_type(a), _BASE_ZERO, ctype,
        )
    elif fmt == "bell":
        # values: rows x ellCols dense block storage; a: ellColInd (block columns, -1 pad)
        block = int(b)
        status = lib.cusparseCreateBlockedEll(
            ctypes.byref(h), rows, cols, block, int(values.shape[1]), _P(a.data_ptr()),
            _P(values.data_ptr()), _index_type(a), _BASE_ZERO, ctype,
        )
    elif fmt == "coo":
        status = lib.cusparseCreateCoo(
            ctypes.byref(h), rows, cols, values.numel(), _P(a.data_ptr()), _P(b.data_ptr()),
            _P(values.data_ptr()), _index_type(a), _BASE_ZERO, ctype,
        )
    else:
        raise ValueError(f"unsupported format {fmt}")
    _check(status, f"cusparseCreate{fmt.capitalize()}")
    return h


def _dnmat(t):
    """Descriptor for a 2D tensor stored row- or column-major (any leading dimension)."""
    rows, cols = int(t.shape[0]), int(t.shape[1])
    if t.stride(1) == 1:
        order, ld = _ORDER_ROW, max(1, t.stride(0))
    elif t.stride(0) == 1:
        order, ld = _ORDER_COL, max(1, t.stride(1))
    else:
        raise ValueError("dense operand must be row- or column-major")
    h = _P()
    _check(
        _lib().cusparseCreateDnMat(ctypes.byref(h), rows, cols, ld, _P(t.data_ptr()), _CUDA_TYPES[t.dtype], order),
        "cusparseCreateDnMat",
    )
    return h


def prepare_spmv(fmt, arrays, shape, x, y, op, compute_dtype):
    """y = op(A) @ x; x / y dtypes may differ from A's (mixed precision)."""
    lib, handle = _lib(), _handle()
    values = arrays[0]
    m = _spmat(fmt, arrays, shape, values.dtype)
    x_d, y_d = _dnvec(x), _dnvec(y)
    ctype = _CUDA_TYPES[compute_dtype]
    one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
    size = ctypes.c_size_t(0)
    _check(
        lib.cusparseSpMV_bufferSize(
            handle, _OP[op], ctypes.byref(one), m, x_d, ctypes.byref(zero), y_d, ctype,
            _ALG_DEFAULT, ctypes.byref(size),
        ),
        "cusparseSpMV_bufferSize",
    )
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=x.device)

    def run():
        _check(
            lib.cusparseSpMV(
                handle, _OP[op], ctypes.byref(one), m, x_d, ctypes.byref(zero), y_d, ctype,
                _ALG_DEFAULT, _P(work.data_ptr()),
            ),
            "cusparseSpMV",
        )
        return y

    destroy = [(lib.cusparseDestroySpMat, m), (lib.cusparseDestroyDnVec, x_d), (lib.cusparseDestroyDnVec, y_d)]
    return _Plan(run, (arrays, x, y, work, one, zero), destroy)


def prepare_spmm(fmt, arrays, shape, B, C, op_a, op_b, compute_dtype):
    """C = op(A) @ op(B); B / C may be row- or column-major views."""
    lib, handle = _lib(), _handle()
    values = arrays[0]
    m = _spmat(fmt, arrays, shape, values.dtype)
    b_d, c_d = _dnmat(B), _dnmat(C)
    ctype = _CUDA_TYPES[compute_dtype]
    one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
    size = ctypes.c_size_t(0)
    _check(
        lib.cusparseSpMM_bufferSize(
            handle, _OP[op_a], _OP[op_b], ctypes.byref(one), m, b_d, ctypes.byref(zero), c_d,
            ctype, _ALG_DEFAULT, ctypes.byref(size),
        ),
        "cusparseSpMM_bufferSize",
    )
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=B.device)

    def run():
        _check(
            lib.cusparseSpMM(
                handle, _OP[op_a], _OP[op_b], ctypes.byref(one), m, b_d, ctypes.byref(zero), c_d,
                ctype, _ALG_DEFAULT, _P(work.data_ptr()),
            ),
            "cusparseSpMM",
        )
        return C

    destroy = [(lib.cusparseDestroySpMat, m), (lib.cusparseDestroyDnMat, b_d), (lib.cusparseDestroyDnMat, c_d)]
    return _Plan(run, (arrays, B, C, work, one, zero), destroy)


def prepare_sddmm(arrays, shape, A, B, out_values, op_a, op_b, compute_dtype):
    """out_values = op(A) @ op(B) sampled on the CSR pattern ``arrays`` (values unused)."""
    lib, handle = _lib(), _handle()
    _, cols, ptr = arrays
    pattern = (out_values, cols, ptr)
    m = _spmat("csr", pattern, shape, out_values.dtype)
    a_d, b_d = _dnmat(A), _dnmat(B)
    ctype = _CUDA_TYPES[compute_dtype]
    one, zero = _scalar(1, compute_dtype), _scalar(0, compute_dtype)
    args = (handle, _OP[op_a], _OP[op_b], ctypes.byref(one), a_d, b_d, ctypes.byref(zero), m, ctype, _ALG_DEFAULT)
    size = ctypes.c_size_t(0)
    _check(lib.cusparseSDDMM_bufferSize(*args, ctypes.byref(size)), "cusparseSDDMM_bufferSize")
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=A.device)
    _check(lib.cusparseSDDMM_preprocess(*args, _P(work.data_ptr())), "cusparseSDDMM_preprocess")

    def run():
        _check(lib.cusparseSDDMM(*args, _P(work.data_ptr())), "cusparseSDDMM")
        return out_values

    destroy = [(lib.cusparseDestroySpMat, m), (lib.cusparseDestroyDnMat, a_d), (lib.cusparseDestroyDnMat, b_d)]
    return _Plan(run, (pattern, A, B, work, one, zero), destroy)
