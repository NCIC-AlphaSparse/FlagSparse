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

"""Native CSC SpMV kernels and public helpers."""

from ._common import *

import triton
import triton.language as tl

SUPPORTED_SPMV_CSC_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMV_CSC_OP_NON = 0
SPMV_CSC_OP_TRANS = 1
SPMV_CSC_OP_CONJ_TRANS = 2
SPMV_CSC_OP_NAMES = {
    SPMV_CSC_OP_NON: "non",
    SPMV_CSC_OP_TRANS: "trans",
    SPMV_CSC_OP_CONJ_TRANS: "conj",
}
_SPMV_CSC_OP_NAME_TO_CODE = {name: code for code, name in SPMV_CSC_OP_NAMES.items()}


def _spmv_csc_dtype_error_message():
    return "CSC SpMV supports float32, float64, complex64, and complex128"


def _normalize_spmv_csc_op(op=None, transpose=False):
    if op is None:
        return SPMV_CSC_OP_TRANS if bool(transpose) else SPMV_CSC_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMV_CSC_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMV_CSC_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMV_CSC_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmv_csc_op_to_name(op):
    return SPMV_CSC_OP_NAMES[_normalize_spmv_csc_op(op)]


def _spmv_csc_op_transposes(op):
    return _normalize_spmv_csc_op(op) in (
        SPMV_CSC_OP_TRANS,
        SPMV_CSC_OP_CONJ_TRANS,
    )


def _normalize_spmv_csc_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


class PreparedCscSpmv:
    """Prepared CSC metadata for repeated SpMV calls."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "block_nnz",
        "max_segments",
        "col_lengths",
        "max_col_nnz",
        "op",
        "transpose",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
    )

    def __init__(
        self,
        data,
        kernel_indices,
        kernel_indptr,
        shape,
        n_rows,
        n_cols,
        block_nnz,
        max_segments,
        max_col_nnz,
        col_lengths=None,
        op=None,
        transpose=False,
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
    ):
        self.data = data
        self.kernel_indices = kernel_indices
        self.kernel_indptr = kernel_indptr
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        self.nnz = int(data.numel())
        self.block_nnz = int(block_nnz)
        self.max_segments = int(max_segments)
        if col_lengths is None:
            col_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
        self.col_lengths = col_lengths
        self.max_col_nnz = int(max_col_nnz)
        self.op = _normalize_spmv_csc_op(op, transpose=transpose)
        self.transpose = _spmv_csc_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason


@triton.jit
def _spmv_csc_non_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
):
    col = tl.program_id(0)
    seg = tl.program_id(1)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
    x_val = tl.load(x_ptr + col)
    tl.atomic_add(y_ptr + rows, vals * x_val, mask=mask, sem="relaxed")


@triton.jit
def _spmv_csc_non_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
):
    col = tl.program_id(0)
    seg = tl.program_id(1)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    rows = tl.load(indices_ptr + offs, mask=mask, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
    x_re = tl.load(x_ri_ptr + col * 2)
    x_im = tl.load(x_ri_ptr + col * 2 + 1)
    prod_re = a_re * x_re - a_im * x_im
    prod_im = a_re * x_im + a_im * x_re
    tl.atomic_add(y_ri_ptr + rows * 2, prod_re, mask=mask, sem="relaxed")
    tl.atomic_add(y_ri_ptr + rows * 2 + 1, prod_im, mask=mask, sem="relaxed")


@triton.jit
def _spmv_csc_trans_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    acc = tl.load(data_ptr + start, mask=start < end, other=0.0) * 0
    for seg in range(MAX_SEGMENTS):
        offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        mask = offs < end
        rows = tl.load(indices_ptr + offs, mask=mask, other=0)
        vals = tl.load(data_ptr + offs, mask=mask, other=0.0)
        x_vals = tl.load(x_ptr + rows, mask=mask, other=0.0)
        acc = acc + tl.sum(tl.where(mask, vals * x_vals, 0.0))
    tl.store(y_ptr + col, acc)


@triton.jit
def _spmv_csc_trans_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_cols,
    BLOCK_NNZ: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    CONJ: tl.constexpr,
):
    col = tl.program_id(0)
    if col >= n_cols:
        return
    start = tl.load(indptr_ptr + col)
    end = tl.load(indptr_ptr + col + 1)
    acc_re = tl.load(data_ri_ptr + start * 2, mask=start < end, other=0.0) * 0
    acc_im = tl.load(data_ri_ptr + start * 2 + 1, mask=start < end, other=0.0) * 0
    for seg in range(MAX_SEGMENTS):
        offs = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        mask = offs < end
        rows = tl.load(indices_ptr + offs, mask=mask, other=0)
        a_re = tl.load(data_ri_ptr + offs * 2, mask=mask, other=0.0)
        a_im_raw = tl.load(data_ri_ptr + offs * 2 + 1, mask=mask, other=0.0)
        if CONJ:
            a_im = -a_im_raw
        else:
            a_im = a_im_raw
        x_re = tl.load(x_ri_ptr + rows * 2, mask=mask, other=0.0)
        x_im = tl.load(x_ri_ptr + rows * 2 + 1, mask=mask, other=0.0)
        prod_re = a_re * x_re - a_im * x_im
        prod_im = a_re * x_im + a_im * x_re
        acc_re = acc_re + tl.sum(tl.where(mask, prod_re, 0.0))
        acc_im = acc_im + tl.sum(tl.where(mask, prod_im, 0.0))
    tl.store(y_ri_ptr + col * 2, acc_re)
    tl.store(y_ri_ptr + col * 2 + 1, acc_im)


def _prepare_spmv_csc_matrix(data, indices, indptr, shape):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr)):
        raise TypeError("data, indices, indptr must all be torch.Tensor")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_cols + 1:
        raise ValueError(
            f"indptr length must be n_cols+1={n_cols + 1}, got {indptr.numel()}"
        )
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if not all(_is_accel_tensor(t) for t in (data, indices, indptr)):
        raise ValueError("data, indices, indptr must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr)):
        raise ValueError("data, indices, indptr must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMV_CSC_VALUE_DTYPES:
        raise TypeError(_spmv_csc_dtype_error_message())
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()
    if indptr.numel() > 0:
        if int(indptr[0].item()) != 0:
            raise ValueError("indptr must start at zero")
        if int(indptr[-1].item()) != data.numel():
            raise ValueError("indptr[-1] must equal nnz")
        if indptr.numel() > 1 and torch.any(indptr[1:] < indptr[:-1]).item():
            raise ValueError("indptr must be non-decreasing")
    if data.numel() > 0:
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= n_rows:
            raise IndexError("indices out of range for n_rows")
    col_lengths = indptr[1:] - indptr[:-1]
    max_col_nnz = int(col_lengths.max().item()) if n_cols > 0 else 0
    return data, indices, indptr, n_rows, n_cols, col_lengths, max_col_nnz


def prepare_spmv_csc(
    data,
    indices,
    indptr,
    shape,
    block_nnz=256,
    max_segments=None,
    transpose=False,
    op=None,
    index_fallback_policy="auto",
):
    index_fallback_policy = _normalize_spmv_csc_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_csc_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_CSC_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    data, indices, indptr, n_rows, n_cols, col_lengths, max_col_nnz = (
        _prepare_spmv_csc_matrix(data, indices, indptr, shape)
    )
    block_nnz_use = int(block_nnz)
    if block_nnz_use <= 0:
        raise ValueError("block_nnz must be positive")
    if max_segments is None:
        max_segments_use = max((max_col_nnz + block_nnz_use - 1) // block_nnz_use, 1)
        while max_segments_use > 2048 and block_nnz_use < 65536:
            block_nnz_use *= 2
            max_segments_use = max(
                (max_col_nnz + block_nnz_use - 1) // block_nnz_use,
                1,
            )
    else:
        max_segments_use = max(1, int(max_segments))
    return PreparedCscSpmv(
        data=data,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=shape,
        n_rows=n_rows,
        n_cols=n_cols,
        block_nnz=block_nnz_use,
        max_segments=max_segments_use,
        max_col_nnz=max_col_nnz,
        col_lengths=col_lengths,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
    )


def _validate_spmv_csc_x(x, prepared, op_code):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not _is_accel_tensor(x):
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    expected = prepared.n_rows if _spmv_csc_op_transposes(op_code) else prepared.n_cols
    if x.numel() != expected:
        raise ValueError(f"x length must be {expected}, got {x.numel()}")
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    return x.contiguous()


def _triton_spmv_csc_kernel(prepared, x, op_code):
    dtype = prepared.data.dtype
    trans = _spmv_csc_op_transposes(op_code)
    out_len = prepared.n_cols if trans else prepared.n_rows
    y = torch.zeros(out_len, dtype=dtype, device=prepared.data.device)
    if prepared.nnz == 0:
        return y
    if not trans:
        grid = (prepared.n_cols, prepared.max_segments)
        if _is_complex_dtype(dtype):
            data_ri = torch.view_as_real(prepared.data).reshape(-1)
            x_ri = torch.view_as_real(x).reshape(-1)
            y_ri = torch.zeros(out_len * 2, dtype=data_ri.dtype, device=y.device)
            _spmv_csc_non_complex_kernel[grid](
                data_ri,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                x_ri,
                y_ri,
                prepared.n_cols,
                BLOCK_NNZ=prepared.block_nnz,
            )
            y.copy_(torch.view_as_complex(y_ri.reshape(out_len, 2)))
            return y
        _spmv_csc_non_real_kernel[grid](
            prepared.data,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x,
            y,
            prepared.n_cols,
            BLOCK_NNZ=prepared.block_nnz,
        )
        return y
    grid = (prepared.n_cols,)
    if _is_complex_dtype(dtype):
        data_ri = torch.view_as_real(prepared.data).reshape(-1)
        x_ri = torch.view_as_real(x).reshape(-1)
        y_ri = torch.empty(out_len * 2, dtype=data_ri.dtype, device=y.device)
        _spmv_csc_trans_complex_kernel[grid](
            data_ri,
            prepared.kernel_indices,
            prepared.kernel_indptr,
            x_ri,
            y_ri,
            prepared.n_cols,
            BLOCK_NNZ=prepared.block_nnz,
            MAX_SEGMENTS=prepared.max_segments,
            CONJ=(op_code == SPMV_CSC_OP_CONJ_TRANS),
        )
        y.copy_(torch.view_as_complex(y_ri.reshape(out_len, 2)))
        return y
    _spmv_csc_trans_real_kernel[grid](
        prepared.data,
        prepared.kernel_indices,
        prepared.kernel_indptr,
        x,
        y,
        prepared.n_cols,
        BLOCK_NNZ=prepared.block_nnz,
        MAX_SEGMENTS=prepared.max_segments,
    )
    return y


def _spmv_csc_uses_int64_indices(prepared):
    return (
        prepared.kernel_indices.dtype == torch.int64
        or prepared.kernel_indptr.dtype == torch.int64
    )


def _spmv_csc_int32_fallback_blocker(prepared):
    if prepared.nnz > _INDEX_LIMIT_INT32:
        return f"nnz {prepared.nnz} cannot fit int32"
    if prepared.kernel_indices.numel() > 0:
        max_row = int(prepared.kernel_indices.max().item())
        if max_row > _INDEX_LIMIT_INT32:
            return f"row index {max_row} cannot fit int32"
    if prepared.kernel_indptr.numel() > 0:
        max_ptr = int(prepared.kernel_indptr[-1].item())
        if max_ptr > _INDEX_LIMIT_INT32:
            return f"indptr offset {max_ptr} cannot fit int32"
    return None


def _spmv_csc_prepared_with_int32_indices(prepared, reason):
    blocker = _spmv_csc_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(f"int32 fallback is unsafe: {blocker}") from reason
    return PreparedCscSpmv(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices.to(torch.int32).contiguous(),
        kernel_indptr=prepared.kernel_indptr.to(torch.int32).contiguous(),
        shape=prepared.shape,
        n_rows=prepared.n_rows,
        n_cols=prepared.n_cols,
        block_nnz=prepared.block_nnz,
        max_segments=prepared.max_segments,
        max_col_nnz=prepared.max_col_nnz,
        col_lengths=prepared.col_lengths,
        op=prepared.op,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
    )


def _run_spmv_csc_prepared_with_fallback(prepared, x, op_code):
    try:
        return _triton_spmv_csc_kernel(prepared, x, op_code)
    except RuntimeError as exc:
        if (
            prepared.index_fallback_policy != "auto"
            or not _spmv_csc_uses_int64_indices(prepared)
        ):
            raise
        fallback_prepared = _spmv_csc_prepared_with_int32_indices(prepared, exc)
        return _triton_spmv_csc_kernel(fallback_prepared, x, op_code)


def flagsparse_spmv_csc(
    data=None,
    indices=None,
    indptr=None,
    x=None,
    shape=None,
    block_nnz=256,
    max_segments=None,
    out=None,
    return_time=False,
    return_meta=False,
    prepared=None,
    transpose=None,
    op=None,
    index_fallback_policy="auto",
):
    """CSC SpMV using native Triton CSC kernels."""
    op_explicit = op is not None
    op_code = _normalize_spmv_csc_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmv_csc_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape)):
            raise ValueError(
                "data, indices, indptr, and shape are required when prepared is not provided"
            )
        prepared = prepare_spmv_csc(
            data,
            indices,
            indptr,
            shape,
            block_nnz=block_nnz,
            max_segments=max_segments,
            op=op_code,
            index_fallback_policy=index_fallback_policy,
        )
    else:
        if op_explicit and op_code != prepared.op:
            raise ValueError(
                f"op={_spmv_csc_op_to_name(op_code)} does not match prepared.op={_spmv_csc_op_to_name(prepared.op)}"
            )
        if (
            not op_explicit
            and transpose is not None
            and bool(transpose) != prepared.transpose
        ):
            raise ValueError(
                f"transpose={bool(transpose)} does not match prepared.transpose={prepared.transpose}"
            )
        if not op_explicit:
            op_code = prepared.op
    x = _validate_spmv_csc_x(x, prepared, op_code)
    do_timing = bool(return_time or return_meta)
    if do_timing:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    y = _run_spmv_csc_prepared_with_fallback(prepared, x, op_code)
    if do_timing:
        _ACCEL.synchronize()
        compute_ms = (time.perf_counter() - t0) * 1000.0
        op_total_ms = compute_ms
    else:
        compute_ms = None
        op_total_ms = None
    if out is not None:
        if not _is_accel_tensor(out):
            raise ValueError("out must be a CUDA tensor")
        if out.device != y.device:
            raise ValueError("out must be on the same CUDA device as the result")
        if out.shape != y.shape or out.dtype != y.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(y)
        y = out
    if return_meta:
        meta = {
            "op": _spmv_csc_op_to_name(op_code),
            "symbolic_ms": 0.0 if do_timing else None,
            "compute_ms": compute_ms,
            "op_total_ms": op_total_ms,
            "index_fallback_applied": prepared.index_fallback_applied,
            "index_fallback_reason": prepared.index_fallback_reason,
        }
        if return_time:
            return y, op_total_ms, meta
        return y, meta
    if return_time:
        return y, op_total_ms
    return y
