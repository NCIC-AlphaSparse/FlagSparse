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
the CuPy-based baselines of the existing scripts cannot express. Batch 3 adds
gemvi and csrgeam2 (legacy API) and complex CSR SpGEMM.

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
_OPTIONAL_READY = set()

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


# Resolved on first use rather than at load: gemvi is deprecated since CUDA 12.8 and
# csrgeam2 is legacy API, so a cuSPARSE without them must not break every other
# baseline in this module -- only the rows that need them.
_OPTIONAL_SIGNATURES = {
    "cusparseSgemvi_bufferSize": [_P, _I, _I, _I, _I, ctypes.POINTER(_I)],
    "cusparseCgemvi_bufferSize": [_P, _I, _I, _I, _I, ctypes.POINTER(_I)],
    "cusparseSgemvi": [_P, _I, _I, _I, _P, _P, _I, _I, _P, _P, _P, _P, _I, _P],
    "cusparseCgemvi": [_P, _I, _I, _I, _P, _P, _I, _I, _P, _P, _P, _P, _I, _P],
    "cusparseCreateMatDescr": [_PP],
    "cusparseDestroyMatDescr": [_P],
    "cusparseScsrgeam2_bufferSizeExt": [
        _P, _I, _I, _P, _P, _I, _P, _P, _P, _P, _P, _I, _P, _P, _P, _P, _P, _P, _P,
        ctypes.POINTER(ctypes.c_size_t),
    ],
    "cusparseCcsrgeam2_bufferSizeExt": [
        _P, _I, _I, _P, _P, _I, _P, _P, _P, _P, _P, _I, _P, _P, _P, _P, _P, _P, _P,
        ctypes.POINTER(ctypes.c_size_t),
    ],
    "cusparseXcsrgeam2Nnz": [_P, _I, _I, _P, _I, _P, _P, _P, _I, _P, _P, _P, _P, ctypes.POINTER(_I), _P],
    "cusparseScsrgeam2": [_P, _I, _I, _P, _P, _I, _P, _P, _P, _P, _P, _I, _P, _P, _P, _P, _P, _P, _P, _P],
    "cusparseCcsrgeam2": [_P, _I, _I, _P, _P, _I, _P, _P, _P, _P, _P, _I, _P, _P, _P, _P, _P, _P, _P, _P],
    "cusparseSpGEMM_createDescr": [_PP],
    "cusparseSpGEMM_destroyDescr": [_P],
    "cusparseSpGEMM_workEstimation": [
        _P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P, ctypes.POINTER(ctypes.c_size_t), _P,
    ],
    "cusparseSpGEMM_compute": [
        _P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P, ctypes.POINTER(ctypes.c_size_t), _P,
    ],
    "cusparseSpGEMM_copy": [_P, _I, _I, _P, _P, _P, _P, _P, _I, _I, _P],
    "cusparseSpMatGetSize": [_P, ctypes.POINTER(_I64), ctypes.POINTER(_I64), ctypes.POINTER(_I64)],
    "cusparseCsrSetPointers": [_P, _P, _P, _P],
}
_SPGEMM_ALG_DEFAULT = 0


def _optional(*names):
    """The library, after checking it exports ``names`` (see _OPTIONAL_SIGNATURES)."""
    lib = _lib()
    for name in names:
        if name in _OPTIONAL_READY:
            continue
        fn = getattr(lib, name, None)
        if fn is None:
            raise RuntimeError(f"this cuSPARSE has no {name}")
        fn.argtypes = _OPTIONAL_SIGNATURES[name]
        fn.restype = _I
        _OPTIONAL_READY.add(name)
    return lib


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


def _legacy_prefix(dtype):
    prefix = {torch.float32: "S", torch.complex64: "C"}.get(dtype)
    if prefix is None:
        raise TypeError(f"no legacy cuSPARSE routine wired for {dtype}")
    return prefix


def prepare_gemvi(A, values, indices, y, op, alpha, beta):
    """cusparse<t>gemvi: y = alpha * op(A) @ x + beta * y, A column-major."""
    t = _legacy_prefix(A.dtype)
    lib = _optional(f"cusparse{t}gemvi_bufferSize", f"cusparse{t}gemvi")
    handle = _handle()
    if A.stride(0) != 1:
        raise ValueError("cusparse<t>gemvi needs a column-major A")
    m, n = int(A.shape[0]), int(A.shape[1])
    lda = max(1, A.stride(1))
    nnz = int(values.numel())
    idx = indices.to(torch.int32).contiguous()
    size = _I(0)
    _check(getattr(lib, f"cusparse{t}gemvi_bufferSize")(handle, _OP[op], m, n, nnz, ctypes.byref(size)),
           f"cusparse{t}gemvi_bufferSize")
    work = torch.empty(max(1, size.value), dtype=torch.uint8, device=A.device)
    a, b = _scalar(alpha, A.dtype), _scalar(beta, A.dtype)
    call = getattr(lib, f"cusparse{t}gemvi")

    def run():
        _check(
            call(handle, _OP[op], m, n, ctypes.byref(a), _P(A.data_ptr()), lda, nnz,
                 _P(values.data_ptr()), _P(idx.data_ptr()), ctypes.byref(b), _P(y.data_ptr()),
                 _BASE_ZERO, _P(work.data_ptr())),
            f"cusparse{t}gemvi",
        )
        return y

    return _Plan(run, (A, values, idx, y, work, a, b), [])


def prepare_geam_csr(a_arrays, b_arrays, shape, alpha, beta):
    """cusparse<t>csrgeam2, C allocated per call: bufferSize, Nnz, then the sum.

    ``run`` returns (values, col_indices, row_offsets) of C. The whole sequence is
    in the timed call, as it is for flagsparse_geam_csr, which also counts,
    allocates and fills on every call.
    """
    a_vals, a_cols, a_ptr = a_arrays
    b_vals, b_cols, b_ptr = b_arrays
    t = _legacy_prefix(a_vals.dtype)
    lib = _optional(
        "cusparseCreateMatDescr", "cusparseDestroyMatDescr", f"cusparse{t}csrgeam2_bufferSizeExt",
        "cusparseXcsrgeam2Nnz", f"cusparse{t}csrgeam2",
    )
    handle = _handle()
    m, n = int(shape[0]), int(shape[1])
    a_cols, b_cols = a_cols.to(torch.int32).contiguous(), b_cols.to(torch.int32).contiguous()
    a_ptr, b_ptr = a_ptr.to(torch.int32).contiguous(), b_ptr.to(torch.int32).contiguous()
    nnz_a, nnz_b = int(a_vals.numel()), int(b_vals.numel())
    descrs = []
    for _ in range(3):
        d = _P()
        _check(lib.cusparseCreateMatDescr(ctypes.byref(d)), "cusparseCreateMatDescr")
        descrs.append(d)
    da, db, dc = descrs
    al, be = _scalar(alpha, a_vals.dtype), _scalar(beta, a_vals.dtype)
    size_fn = getattr(lib, f"cusparse{t}csrgeam2_bufferSizeExt")
    geam_fn = getattr(lib, f"cusparse{t}csrgeam2")
    dev = a_vals.device

    def run():
        c_ptr = torch.empty(m + 1, dtype=torch.int32, device=dev)
        size = ctypes.c_size_t(0)
        _check(
            size_fn(handle, m, n, ctypes.byref(al), da, nnz_a, _P(a_vals.data_ptr()), _P(a_ptr.data_ptr()),
                    _P(a_cols.data_ptr()), ctypes.byref(be), db, nnz_b, _P(b_vals.data_ptr()),
                    _P(b_ptr.data_ptr()), _P(b_cols.data_ptr()), dc, _P(0), _P(c_ptr.data_ptr()), _P(0),
                    ctypes.byref(size)),
            f"cusparse{t}csrgeam2_bufferSizeExt",
        )
        work = torch.empty(max(1, size.value), dtype=torch.uint8, device=dev)
        nnz_c = _I(0)
        _check(
            lib.cusparseXcsrgeam2Nnz(handle, m, n, da, nnz_a, _P(a_ptr.data_ptr()), _P(a_cols.data_ptr()),
                                     db, nnz_b, _P(b_ptr.data_ptr()), _P(b_cols.data_ptr()), dc,
                                     _P(c_ptr.data_ptr()), ctypes.byref(nnz_c), _P(work.data_ptr())),
            "cusparseXcsrgeam2Nnz",
        )
        c_cols = torch.empty(max(1, nnz_c.value), dtype=torch.int32, device=dev)
        c_vals = torch.empty(max(1, nnz_c.value), dtype=a_vals.dtype, device=dev)
        _check(
            geam_fn(handle, m, n, ctypes.byref(al), da, nnz_a, _P(a_vals.data_ptr()), _P(a_ptr.data_ptr()),
                    _P(a_cols.data_ptr()), ctypes.byref(be), db, nnz_b, _P(b_vals.data_ptr()),
                    _P(b_ptr.data_ptr()), _P(b_cols.data_ptr()), dc, _P(c_vals.data_ptr()),
                    _P(c_ptr.data_ptr()), _P(c_cols.data_ptr()), _P(work.data_ptr())),
            f"cusparse{t}csrgeam2",
        )
        return c_vals[: nnz_c.value], c_cols[: nnz_c.value], c_ptr

    destroy = [(lib.cusparseDestroyMatDescr, d) for d in descrs]
    return _Plan(run, (a_vals, a_cols, a_ptr, b_vals, b_cols, b_ptr, al, be), destroy)


def prepare_spgemm_csr(a_arrays, a_shape, b_arrays, b_shape):
    """Generic cusparseSpGEMM (ALG_DEFAULT), the whole sequence per call.

    createDescr -> workEstimation x2 -> compute x2 -> getSize -> setPointers ->
    copy, allocating C each time: what CuPy's ``A @ B`` runs for the real dtypes in
    tests/test_spgemm.py, and what one flagsparse_spgemm_csr call does. int32
    indices only (cuSPARSE SpGEMM has no 64-bit index path).
    """
    lib = _optional(
        "cusparseSpGEMM_createDescr", "cusparseSpGEMM_destroyDescr", "cusparseSpGEMM_workEstimation",
        "cusparseSpGEMM_compute", "cusparseSpGEMM_copy", "cusparseSpMatGetSize", "cusparseCsrSetPointers",
    )
    handle = _handle()
    a_vals, a_cols, a_ptr = a_arrays[0], a_arrays[1].to(torch.int32), a_arrays[2].to(torch.int32)
    b_vals, b_cols, b_ptr = b_arrays[0], b_arrays[1].to(torch.int32), b_arrays[2].to(torch.int32)
    dtype, dev = a_vals.dtype, a_vals.device
    ctype = _CUDA_TYPES[dtype]
    mat_a = _spmat("csr", (a_vals, a_cols, a_ptr), a_shape, dtype)
    mat_b = _spmat("csr", (b_vals, b_cols, b_ptr), b_shape, dtype)
    m, n = int(a_shape[0]), int(b_shape[1])
    one, zero = _scalar(1, dtype), _scalar(0, dtype)

    def run():
        mat_c = _P()
        _check(lib.cusparseCreateCsr(ctypes.byref(mat_c), m, n, 0, _P(0), _P(0), _P(0),
                                     _INDEX_32I, _INDEX_32I, _BASE_ZERO, ctype), "cusparseCreateCsr")
        descr = _P()
        _check(lib.cusparseSpGEMM_createDescr(ctypes.byref(descr)), "cusparseSpGEMM_createDescr")
        try:
            args = (handle, _OP["non"], _OP["non"], ctypes.byref(one), mat_a, mat_b, ctypes.byref(zero),
                    mat_c, ctype, _SPGEMM_ALG_DEFAULT, descr)
            buffers = []
            for step in (lib.cusparseSpGEMM_workEstimation, lib.cusparseSpGEMM_compute):
                size = ctypes.c_size_t(0)
                _check(step(*args, ctypes.byref(size), _P(0)), step.__name__)
                buf = torch.empty(max(1, size.value), dtype=torch.uint8, device=dev)
                _check(step(*args, ctypes.byref(size), _P(buf.data_ptr())), step.__name__)
                buffers.append(buf)
            rows, cols, nnz = _I64(0), _I64(0), _I64(0)
            _check(lib.cusparseSpMatGetSize(mat_c, ctypes.byref(rows), ctypes.byref(cols), ctypes.byref(nnz)),
                   "cusparseSpMatGetSize")
            c_ptr = torch.empty(m + 1, dtype=torch.int32, device=dev)
            c_cols = torch.empty(max(1, nnz.value), dtype=torch.int32, device=dev)
            c_vals = torch.empty(max(1, nnz.value), dtype=dtype, device=dev)
            _check(lib.cusparseCsrSetPointers(mat_c, _P(c_ptr.data_ptr()), _P(c_cols.data_ptr()),
                                              _P(c_vals.data_ptr())), "cusparseCsrSetPointers")
            _check(lib.cusparseSpGEMM_copy(*args), "cusparseSpGEMM_copy")
            return c_vals[: nnz.value], c_cols[: nnz.value], c_ptr
        finally:
            lib.cusparseSpGEMM_destroyDescr(descr)
            lib.cusparseDestroySpMat(mat_c)

    destroy = [(lib.cusparseDestroySpMat, mat_a), (lib.cusparseDestroySpMat, mat_b)]
    return _Plan(run, (a_vals, a_cols, a_ptr, b_vals, b_cols, b_ptr, one, zero), destroy)
