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

"""Mixed-precision and integer SpMV / SpMM (the cuSPARSE 12.5 type combinations).

The public entry points (``flagsparse_spmv_csr``, ``flagsparse_spmv_coo``,
``flagsparse_spmm_csr``) route here only for combinations their native paths do
not take, so every previously supported call keeps its exact code path:

=========================  ==============================  ===========
matrix (and x / B)         output (``out_dtype``/``out``)  accumulate
=========================  ==============================  ===========
float16 / bfloat16         float32                         float32
int8                       int32 (default) or float32      int32 / f32
float32 A, complex64 x     complex64 (SpMV only)           float32 x2
float16 / bfloat16 (COO)   same half type                  float32
=========================  ==============================  ===========

Backend notes:
- COO accumulates with ``tl.atomic_add`` on a float32 / int32 buffer and casts at
  the end; fp16 and complex atomics are never needed (not every backend has them).
- CSR kernels loop over a row's entries at runtime; nothing is unrolled, so the
  per-thread private memory stays small (MetaX caps it at 4 KB).
- Complex vectors are read as interleaved real/imag planes: no complex indexing,
  which Moore Threads lacks.
- Ascend's Triton lacks the pieces these kernels use, so it takes a torch_npu
  ``index_add_`` path (complex added as two real planes).
"""

from ._common import *

import time

import triton
import triton.language as tl

_HALF = (torch.float16, torch.bfloat16)
_TL_ACC = {torch.float32: tl.float32, torch.float64: tl.float64, torch.int32: tl.int32}

# value dtype -> widened output dtypes (the first is the default for int8).
_WIDE_OUT = {
    torch.float16: (torch.float32,),
    torch.bfloat16: (torch.float32,),
    torch.int8: (torch.int32, torch.float32),
}
# real matrix dtype -> complex vector dtype it may multiply (SpMV only).
_REAL_TIMES_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def _requested_out_dtype(out, out_dtype):
    if out is not None and out_dtype is not None and out.dtype != out_dtype:
        raise ValueError("out.dtype conflicts with out_dtype")
    return out_dtype if out_dtype is not None else (None if out is None else out.dtype)


def spmv_needs_mixed(data_dtype, x_dtype, out, out_dtype, *, coo=False):
    """Whether an SpMV call is one of the combinations handled in this module."""
    requested = _requested_out_dtype(out, out_dtype)
    if data_dtype == torch.int8:
        return True
    if coo and data_dtype in _HALF:
        return True  # COO's native path has no half-precision kernel
    if x_dtype is not None and x_dtype != data_dtype and data_dtype in _REAL_TIMES_COMPLEX:
        return True
    return requested is not None and requested != data_dtype and data_dtype in _WIDE_OUT


def spmm_needs_mixed(data_dtype, out, out_dtype):
    requested = _requested_out_dtype(out, out_dtype)
    if data_dtype == torch.int8:
        return True
    return requested is not None and requested != data_dtype and data_dtype in _WIDE_OUT


def _resolve_types(data_dtype, rhs_dtype, out, out_dtype, allow_complex_rhs):
    """-> (output dtype, accumulate dtype); raises for combinations cuSPARSE lacks."""
    requested = _requested_out_dtype(out, out_dtype)
    if rhs_dtype != data_dtype:
        if not (allow_complex_rhs and _REAL_TIMES_COMPLEX.get(data_dtype) == rhs_dtype):
            raise TypeError(
                f"{data_dtype} matrix with {rhs_dtype} operand is not a supported combination"
            )
        if requested not in (None, rhs_dtype):
            raise TypeError(f"real-by-complex SpMV writes {rhs_dtype}")
        return rhs_dtype, data_dtype
    if data_dtype == torch.int8:
        result = requested or torch.int32
        if result not in _WIDE_OUT[torch.int8]:
            raise TypeError("int8 SpMV/SpMM writes int32 or float32")
        return result, (torch.int32 if result == torch.int32 else torch.float32)
    if data_dtype in _HALF:
        result = requested or data_dtype
        if result not in (data_dtype, torch.float32):
            raise TypeError(f"{data_dtype} SpMV/SpMM writes {data_dtype} or float32")
        return result, torch.float32
    if requested not in (None, data_dtype):
        raise TypeError(f"{data_dtype} SpMV/SpMM has no {requested} output")
    return data_dtype, data_dtype


def _normalize_non_op(op, transpose):
    token = "non" if op is None else str(op).strip().lower()
    if token in ("0", "n", "non_trans"):
        token = "non"
    if token != "non" or bool(transpose):
        raise NotImplementedError("mixed-precision/int8 paths support op='non' only")


def _check_1d(*pairs):
    for tensor, name in pairs:
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor")
        if not _is_accel_tensor(tensor):
            raise ValueError(f"{name} must be an accelerator tensor")


def _index32(t):
    return t.to(torch.int32) if t.dtype == torch.int64 else t


def _csr_row_ids(indptr, n_rows, nnz):
    lengths = (indptr[1:] - indptr[:-1]).to(torch.int64)
    return torch.repeat_interleave(
        torch.arange(n_rows, device=indptr.device, dtype=torch.int64), lengths, output_size=nnz
    )


def _finish(y, out, t0, return_time):
    if out is not None:
        out.copy_(y)
        y = out
    if return_time:
        _ACCEL.synchronize()
        return y, (time.perf_counter() - t0) * 1000.0
    return y


def _start(return_time):
    if not return_time:
        return None
    _ACCEL.synchronize()
    return time.perf_counter()


# ---------------------------------------------------------------------------
# CSR SpMV
# ---------------------------------------------------------------------------


# CSR SpMV walks a tile of ROWS rows per program, BLOCK entries of each row at a time.
# One program per row was 3-4x slower (cage12: 63 us vs 15 us, cuSPARSE fp32 21 us):
# 130k programs of ~16 entries each spend their time launching, not loading.
@triton.jit
def _csr_spmv_mixed_kernel(
    y_ptr, data_ptr, cols_ptr, indptr_ptr, x_ptr, n_rows,
    ACC: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    start = tl.load(indptr_ptr + rows, mask=rmask, other=0)
    end = tl.load(indptr_ptr + rows + 1, mask=rmask, other=0)
    max_len = tl.max(end - start, axis=0)
    acc = tl.zeros((ROWS, BLOCK), dtype=ACC)
    for j in range(0, max_len, BLOCK):
        offs = start[:, None] + j + tl.arange(0, BLOCK)[None, :]
        mask = offs < end[:, None]
        col = tl.load(cols_ptr + offs, mask=mask, other=0)
        v = tl.load(data_ptr + offs, mask=mask, other=0).to(ACC)
        xv = tl.load(x_ptr + col, mask=mask, other=0).to(ACC)
        acc += v * xv
    tl.store(y_ptr + rows, tl.sum(acc, axis=1).to(y_ptr.dtype.element_ty), mask=rmask)


@triton.jit
def _csr_spmv_real_by_complex_kernel(
    y_ri_ptr, data_ptr, cols_ptr, indptr_ptr, x_ri_ptr, n_rows,
    ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    start = tl.load(indptr_ptr + rows, mask=rmask, other=0)
    end = tl.load(indptr_ptr + rows + 1, mask=rmask, other=0)
    max_len = tl.max(end - start, axis=0)
    acc_re = tl.zeros((ROWS, BLOCK), dtype=y_ri_ptr.dtype.element_ty)
    acc_im = tl.zeros((ROWS, BLOCK), dtype=y_ri_ptr.dtype.element_ty)
    for j in range(0, max_len, BLOCK):
        offs = start[:, None] + j + tl.arange(0, BLOCK)[None, :]
        mask = offs < end[:, None]
        col = tl.load(cols_ptr + offs, mask=mask, other=0)
        v = tl.load(data_ptr + offs, mask=mask, other=0.0)
        acc_re += v * tl.load(x_ri_ptr + 2 * col, mask=mask, other=0.0)
        acc_im += v * tl.load(x_ri_ptr + 2 * col + 1, mask=mask, other=0.0)
    tl.store(y_ri_ptr + 2 * rows, tl.sum(acc_re, axis=1), mask=rmask)
    tl.store(y_ri_ptr + 2 * rows + 1, tl.sum(acc_im, axis=1), mask=rmask)


def _csr_row_tile(nnz, n_rows):
    """(ROWS, BLOCK) from the mean row length: one BLOCK covers a typical row.

    Measured on a 5090 over cage12 / GL7d14 / roadNet-TX / amazon0601 / wave; a
    BLOCK much wider than the rows wastes the tile (roadNet-TX, 2.8 nnz/row:
    16x32 took 72 us, 128x4 took 14 us). ROWS * BLOCK stays 512 lanes, which
    four warps cover at warp size 32 or 64.
    """
    mean = nnz / max(1, n_rows)
    if mean <= 4:
        return 128, 4
    if mean <= 12:
        return 64, 8
    if mean <= 24:
        return 32, 16
    return 16, 32


def _torch_spmv_rows(data, rows, cols, x, n_out, acc_dtype):
    """Ascend path: gather, multiply in the accumulate type, index_add by output row."""
    cols = cols.to(torch.int64)
    if _is_complex_dtype(x.dtype):
        xv = torch.view_as_real(x).index_select(0, cols)
        d = data.to(xv.dtype).unsqueeze(-1)
        y = torch.zeros(n_out, dtype=x.dtype, device=x.device)
        torch.view_as_real(y).index_add_(0, rows, d * xv)
        return y
    y = torch.zeros(n_out, dtype=acc_dtype, device=x.device)
    y.index_add_(0, rows, data.to(acc_dtype) * x.index_select(0, cols).to(acc_dtype))
    return y


def spmv_csr_mixed(data, indices, indptr, x, shape, *, op=None, transpose=None,
                   out=None, out_dtype=None, return_time=False):
    _normalize_non_op(op, transpose)
    _check_1d((data, "data"), (indices, "indices"), (indptr, "indptr"), (x, "x"))
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != indices.numel() or indptr.numel() != n_rows + 1:
        raise ValueError("invalid CSR dimensions")
    if x.numel() != n_cols:
        raise ValueError("x shape does not match CSR operation")
    result_dtype, acc_dtype = _resolve_types(data.dtype, x.dtype, out, out_dtype, True)
    if out is not None and (out.shape != (n_rows,) or out.device != x.device):
        raise ValueError("out shape/device must match the CSR SpMV result")
    data, x = data.contiguous(), x.contiguous()
    cols, ptr = _index32(indices.contiguous()), _index32(indptr.contiguous())
    t0 = _start(return_time)
    if _is_ascend_runtime():
        rows = _csr_row_ids(ptr, n_rows, data.numel())
        y = _torch_spmv_rows(data, rows, cols, x, n_rows, acc_dtype).to(result_dtype)
        return _finish(y, out, t0, return_time)
    y = torch.empty(n_rows, dtype=result_dtype, device=x.device)
    if n_rows:
        rows_per, block = _csr_row_tile(data.numel(), n_rows)
        grid = (triton.cdiv(n_rows, rows_per),)
        if _is_complex_dtype(result_dtype):
            _csr_spmv_real_by_complex_kernel[grid](
                torch.view_as_real(y).reshape(-1), data, cols, ptr,
                torch.view_as_real(x).reshape(-1), n_rows,
                ROWS=rows_per, BLOCK=block, num_warps=4,
            )
        else:
            _csr_spmv_mixed_kernel[grid](
                y, data, cols, ptr, x, n_rows,
                ACC=_TL_ACC[acc_dtype], ROWS=rows_per, BLOCK=block, num_warps=4,
            )
    return _finish(y, out, t0, return_time)


# ---------------------------------------------------------------------------
# COO SpMV
# ---------------------------------------------------------------------------


@triton.jit
def _coo_spmv_atomic_kernel(
    acc_ptr, data_ptr, rows_ptr, cols_ptr, x_ptr, nnz,
    ACC: tl.constexpr, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < nnz
    row = tl.load(rows_ptr + offs, mask=mask, other=0)
    col = tl.load(cols_ptr + offs, mask=mask, other=0)
    v = tl.load(data_ptr + offs, mask=mask, other=0).to(ACC)
    xv = tl.load(x_ptr + col, mask=mask, other=0).to(ACC)
    tl.atomic_add(acc_ptr + row, v * xv, mask=mask)


def spmv_coo_mixed(data, row, col, x, shape, *, op=None, transpose=None,
                   out=None, out_dtype=None, return_time=False):
    _normalize_non_op(op, transpose)
    _check_1d((data, "data"), (row, "row"), (col, "col"), (x, "x"))
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if not (data.numel() == row.numel() == col.numel()):
        raise ValueError("data, row and col must have the same length")
    if x.numel() != n_cols:
        raise ValueError("x shape does not match COO operation")
    result_dtype, acc_dtype = _resolve_types(data.dtype, x.dtype, out, out_dtype, False)
    if out is not None and (out.shape != (n_rows,) or out.device != x.device):
        raise ValueError("out shape/device must match the COO SpMV result")
    data, x = data.contiguous(), x.contiguous()
    rows, cols = _index32(row.contiguous()), _index32(col.contiguous())
    t0 = _start(return_time)
    if _is_ascend_runtime():
        y = _torch_spmv_rows(data, rows.to(torch.int64), cols, x, n_rows, acc_dtype)
        return _finish(y.to(result_dtype), out, t0, return_time)
    acc = torch.zeros(n_rows, dtype=acc_dtype, device=x.device)
    nnz = data.numel()
    if nnz:
        block = 256
        _coo_spmv_atomic_kernel[(triton.cdiv(nnz, block),)](
            acc, data, rows, cols, x, nnz, ACC=_TL_ACC[acc_dtype], BLOCK=block, num_warps=4,
        )
    y = acc if acc_dtype == result_dtype else acc.to(result_dtype)
    return _finish(y, out, t0, return_time)


# ---------------------------------------------------------------------------
# CSR SpMM (C = A @ B, B/C row- or column-major through their strides)
# ---------------------------------------------------------------------------


@triton.jit
def _csr_spmm_mixed_kernel(
    c_ptr, data_ptr, cols_ptr, indptr_ptr, b_ptr, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    ACC: tl.constexpr, BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < n_dense
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    acc = tl.zeros((BLOCK_N,), dtype=ACC)
    for p in range(start, end):
        col = tl.load(cols_ptr + p)
        a = tl.load(data_ptr + p).to(ACC)
        b = tl.load(b_ptr + col * stride_bk + n_offs * stride_bn, mask=n_mask, other=0).to(ACC)
        acc += a * b
    tl.store(
        c_ptr + row * stride_cm + n_offs * stride_cn,
        acc.to(c_ptr.dtype.element_ty),
        mask=n_mask,
    )


def spmm_csr_mixed(data, indices, indptr, B, shape, *, op=None, transpose=None,
                   out=None, out_dtype=None, return_time=False):
    """``B`` may be any 2D strided view (column-major, or a transposed ``op_b``)."""
    _normalize_non_op(op, transpose)
    _check_1d((data, "data"), (indices, "indices"), (indptr, "indptr"))
    if not torch.is_tensor(B) or B.ndim != 2 or not _is_accel_tensor(B):
        raise ValueError("B must be a 2D accelerator tensor")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != indices.numel() or indptr.numel() != n_rows + 1:
        raise ValueError("invalid CSR dimensions")
    if B.shape[0] != n_cols:
        raise ValueError(f"B has {B.shape[0]} rows, expected {n_cols}")
    result_dtype, acc_dtype = _resolve_types(data.dtype, B.dtype, out, out_dtype, False)
    n_dense = int(B.shape[1])
    if out is not None and (tuple(out.shape) != (n_rows, n_dense) or out.device != B.device):
        raise ValueError("out shape/device must match the CSR SpMM result")
    data = data.contiguous()
    cols, ptr = _index32(indices.contiguous()), _index32(indptr.contiguous())
    t0 = _start(return_time)
    if _is_ascend_runtime():
        rows = _csr_row_ids(ptr, n_rows, data.numel())
        C = torch.zeros(n_rows, n_dense, dtype=acc_dtype, device=B.device)
        C.index_add_(0, rows, data.to(acc_dtype).unsqueeze(1) * B.index_select(0, cols.to(torch.int64)).to(acc_dtype))
        return _finish(C.to(result_dtype), out, t0, return_time)
    C = out if out is not None else torch.empty(n_rows, n_dense, dtype=result_dtype, device=B.device)
    if n_rows and n_dense:
        block_n = min(128, max(16, triton.next_power_of_2(n_dense)))
        _csr_spmm_mixed_kernel[(n_rows, triton.cdiv(n_dense, block_n))](
            C, data, cols, ptr, B, n_dense,
            B.stride(0), B.stride(1), C.stride(0), C.stride(1),
            ACC=_TL_ACC[acc_dtype], BLOCK_N=block_n, num_warps=2,
        )
    if return_time:
        _ACCEL.synchronize()
        return C, (time.perf_counter() - t0) * 1000.0
    return C


# ---------------------------------------------------------------------------
# COO SpMM (C = A @ B, B/C row- or column-major through their strides)
# ---------------------------------------------------------------------------


@triton.jit
def _coo_spmm_mixed_kernel(
    c_ptr, data_ptr, rows_ptr, cols_ptr, b_ptr, nnz, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    ACC: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One program per (nonzero, dense-column tile); every nonzero's contribution
    to its output row is atomic_add'd independently, same shape as
    _coo_spmv_atomic_kernel above with a dense-N axis added (mirroring
    _csr_spmm_mixed_kernel's per-row accumulator, but COO has no contiguous
    per-row range to loop over without a CSR-style indptr)."""
    idx = tl.program_id(0)
    if idx >= nnz:
        return
    n_offs = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offs < n_dense
    row = tl.load(rows_ptr + idx)
    col = tl.load(cols_ptr + idx)
    a = tl.load(data_ptr + idx).to(ACC)
    b = tl.load(b_ptr + col * stride_bk + n_offs * stride_bn, mask=n_mask, other=0).to(ACC)
    tl.atomic_add(c_ptr + row * stride_cm + n_offs * stride_cn, a * b, mask=n_mask)


def spmm_coo_mixed(data, row, col, B, shape, *, op=None, transpose=None,
                   out=None, out_dtype=None, return_time=False):
    """``B`` may be any 2D strided view (column-major, or a transposed ``op_b``)."""
    _normalize_non_op(op, transpose)
    _check_1d((data, "data"), (row, "row"), (col, "col"))
    if not torch.is_tensor(B) or B.ndim != 2 or not _is_accel_tensor(B):
        raise ValueError("B must be a 2D accelerator tensor")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if not (data.numel() == row.numel() == col.numel()):
        raise ValueError("data, row and col must have the same length")
    if B.shape[0] != n_cols:
        raise ValueError(f"B has {B.shape[0]} rows, expected {n_cols}")
    result_dtype, acc_dtype = _resolve_types(data.dtype, B.dtype, out, out_dtype, False)
    n_dense = int(B.shape[1])
    if out is not None and (tuple(out.shape) != (n_rows, n_dense) or out.device != B.device):
        raise ValueError("out shape/device must match the COO SpMM result")
    data = data.contiguous()
    rows, cols = _index32(row.contiguous()), _index32(col.contiguous())
    t0 = _start(return_time)
    if _is_ascend_runtime():
        C = torch.zeros(n_rows, n_dense, dtype=acc_dtype, device=B.device)
        C.index_add_(
            0, rows.to(torch.int64),
            data.to(acc_dtype).unsqueeze(1) * B.index_select(0, cols.to(torch.int64)).to(acc_dtype),
        )
        return _finish(C.to(result_dtype), out, t0, return_time)
    acc = torch.zeros(n_rows, n_dense, dtype=acc_dtype, device=B.device)
    nnz = data.numel()
    if nnz and n_dense:
        block_n = min(128, max(16, triton.next_power_of_2(n_dense)))
        _coo_spmm_mixed_kernel[(nnz, triton.cdiv(n_dense, block_n))](
            acc, data, rows, cols, B, nnz, n_dense,
            B.stride(0), B.stride(1), acc.stride(0), acc.stride(1),
            ACC=_TL_ACC[acc_dtype], BLOCK_N=block_n, num_warps=2,
        )
    y = acc if acc_dtype == result_dtype else acc.to(result_dtype)
    return _finish(y, out, t0, return_time)
