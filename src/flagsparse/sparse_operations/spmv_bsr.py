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

"""Native BSR SpMV kernels and public helpers."""

from ._common import *

import triton
import triton.language as tl


SUPPORTED_SPMV_BSR_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMV_BSR_OP_NON = 0
SPMV_BSR_OP_TRANS = 1
SPMV_BSR_OP_CONJ_TRANS = 2
SPMV_BSR_OP_NAMES = {
    SPMV_BSR_OP_NON: "non",
    SPMV_BSR_OP_TRANS: "trans",
    SPMV_BSR_OP_CONJ_TRANS: "conj",
}
SPMV_BSR_SUPPORTED_OP_NAMES = ("non", "trans", "conj")
_SPMV_BSR_OP_NAME_TO_CODE = {
    name: code for code, name in SPMV_BSR_OP_NAMES.items()
}
SPMV_BSR_ALG_BASE = "base"
SPMV_BSR_ALG_BLOCKROW_REDUCE = "blockrow_reduce"
SPMV_BSR_SUPPORTED_ALGORITHMS = (SPMV_BSR_ALG_BASE, SPMV_BSR_ALG_BLOCKROW_REDUCE)


def _spmv_bsr_dtype_error_message():
    return "BSR SpMV supports float32, float64, complex64, and complex128"


def _normalize_spmv_bsr_op(op=None, transpose=False):
    if op is None:
        return SPMV_BSR_OP_TRANS if bool(transpose) else SPMV_BSR_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMV_BSR_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMV_BSR_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMV_BSR_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmv_bsr_op_to_name(op):
    return SPMV_BSR_OP_NAMES[_normalize_spmv_bsr_op(op)]


def _spmv_bsr_op_transposes(op):
    return _normalize_spmv_bsr_op(op) in (
        SPMV_BSR_OP_TRANS,
        SPMV_BSR_OP_CONJ_TRANS,
    )


def _ensure_spmv_bsr_supported_op(op_code):
    _normalize_spmv_bsr_op(op_code)


def _normalize_spmv_bsr_algorithm(use_opt=False):
    if isinstance(use_opt, str):
        token = use_opt.strip().lower().replace("-", "_")
        if token in ("false", "0", "no", "base", "spmv_bsr_base"):
            return SPMV_BSR_ALG_BASE
        if token in (
            "true",
            "1",
            "yes",
            "opt",
            "auto",
            "blockrow_reduce",
            "spmv_bsr_blockrow_reduce",
        ):
            return SPMV_BSR_ALG_BLOCKROW_REDUCE
        raise ValueError("use_opt must be False, True, 'base', 'auto', or 'blockrow_reduce'")
    return SPMV_BSR_ALG_BLOCKROW_REDUCE if bool(use_opt) else SPMV_BSR_ALG_BASE


def _normalize_spmv_bsr_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


class PreparedBsrSpmv:
    """Prepared BSR metadata for repeated SpMV calls."""

    __slots__ = (
        "data",
        "kernel_indices",
        "kernel_indptr",
        "shape",
        "n_rows",
        "n_cols",
        "padded_n_rows",
        "padded_n_cols",
        "block_dim",
        "n_block_rows",
        "n_block_cols",
        "nnzb",
        "stored_nnz",
        "block_row_lengths",
        "max_block_row_nnz",
        "block_nnz",
        "max_segments",
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
        block_dim,
        n_block_rows,
        n_block_cols,
        block_nnz,
        max_segments,
        max_block_row_nnz,
        block_row_lengths=None,
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
        self.n_rows = int(shape[0])
        self.n_cols = int(shape[1])
        self.block_dim = int(block_dim)
        self.n_block_rows = int(n_block_rows)
        self.n_block_cols = int(n_block_cols)
        self.padded_n_rows = self.n_block_rows * self.block_dim
        self.padded_n_cols = self.n_block_cols * self.block_dim
        self.nnzb = int(data.shape[0])
        self.stored_nnz = int(data.numel())
        if block_row_lengths is None:
            block_row_lengths = kernel_indptr[1:] - kernel_indptr[:-1]
        self.block_row_lengths = block_row_lengths
        self.max_block_row_nnz = int(max_block_row_nnz)
        self.block_nnz = int(block_nnz)
        self.max_segments = int(max_segments)
        self.op = _normalize_spmv_bsr_op(op, transpose=transpose)
        self.transpose = _spmv_bsr_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason


@triton.jit
def _spmv_bsr_non_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_rows,
    n_cols,
    n_block_rows,
    BLOCK_DIM: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    if brow >= n_block_rows:
        return
    row = brow * BLOCK_DIM + inner_row
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    acc = tl.load(
        data_ptr + start * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM,
        mask=start < end,
        other=0.0,
    ) * 0
    for inner_col in tl.static_range(0, BLOCK_DIM):
        col = bcols * BLOCK_DIM + inner_col
        valid = mask
        vals = tl.load(
            data_ptr + offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col,
            mask=mask,
            other=0.0,
        )
        x_vals = tl.load(x_ptr + col, mask=valid, other=0.0)
        acc += tl.sum(tl.where(valid, vals * x_vals, 0.0))
    tl.atomic_add(y_ptr + row, acc)


@triton.jit
def _spmv_bsr_non_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_rows,
    n_cols,
    n_block_rows,
    BLOCK_DIM: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    if brow >= n_block_rows:
        return
    row = brow * BLOCK_DIM + inner_row
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    acc_re = tl.load(
        data_ri_ptr + (start * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM) * 2,
        mask=start < end,
        other=0.0,
    ) * 0
    acc_im = tl.load(
        data_ri_ptr + (start * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM) * 2 + 1,
        mask=start < end,
        other=0.0,
    ) * 0
    for inner_col in tl.static_range(0, BLOCK_DIM):
        col = bcols * BLOCK_DIM + inner_col
        valid = mask
        elem = offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col
        a_re = tl.load(data_ri_ptr + elem * 2, mask=mask, other=0.0)
        a_im = tl.load(data_ri_ptr + elem * 2 + 1, mask=mask, other=0.0)
        x_re = tl.load(x_ri_ptr + col * 2, mask=valid, other=0.0)
        x_im = tl.load(x_ri_ptr + col * 2 + 1, mask=valid, other=0.0)
        prod_re = a_re * x_re - a_im * x_im
        prod_im = a_re * x_im + a_im * x_re
        acc_re += tl.sum(tl.where(valid, prod_re, 0.0))
        acc_im += tl.sum(tl.where(valid, prod_im, 0.0))
    tl.atomic_add(y_ri_ptr + row * 2, acc_re)
    tl.atomic_add(y_ri_ptr + row * 2 + 1, acc_im)


@triton.jit
def _spmv_bsr_blockrow_reduce_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    rows_ptr,
    x_ptr,
    y_ptr,
    n_bucket_rows,
    BLOCK_DIM: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    DIRECT_STORE: tl.constexpr,
):
    row_pos = tl.program_id(0)
    seg = tl.program_id(1)
    if row_pos >= n_bucket_rows:
        return
    brow = tl.load(rows_ptr + row_pos)
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    seg_start = start + seg * BLOCK_NNZ
    offs = seg_start + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    for inner_row in tl.static_range(0, BLOCK_DIM):
        acc = tl.load(
            data_ptr + start * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM,
            mask=start < end,
            other=0.0,
        ) * 0
        for inner_col in tl.static_range(0, BLOCK_DIM):
            col = bcols * BLOCK_DIM + inner_col
            vals = tl.load(
                data_ptr + offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col,
                mask=mask,
                other=0.0,
            )
            x_vals = tl.load(x_ptr + col, mask=mask, other=0.0)
            acc += tl.sum(tl.where(mask, vals * x_vals, 0.0))
        out_row = brow * BLOCK_DIM + inner_row
        if DIRECT_STORE:
            tl.store(y_ptr + out_row, acc)
        else:
            tl.atomic_add(y_ptr + out_row, acc)


@triton.jit
def _spmv_bsr_blockrow_reduce_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    rows_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_bucket_rows,
    BLOCK_DIM: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    DIRECT_STORE: tl.constexpr,
):
    row_pos = tl.program_id(0)
    seg = tl.program_id(1)
    if row_pos >= n_bucket_rows:
        return
    brow = tl.load(rows_ptr + row_pos)
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    seg_start = start + seg * BLOCK_NNZ
    offs = seg_start + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    for inner_row in tl.static_range(0, BLOCK_DIM):
        acc_re = tl.load(
            data_ri_ptr + (start * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM) * 2,
            mask=start < end,
            other=0.0,
        ) * 0
        acc_im = tl.load(
            data_ri_ptr + (start * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM) * 2 + 1,
            mask=start < end,
            other=0.0,
        ) * 0
        for inner_col in tl.static_range(0, BLOCK_DIM):
            col = bcols * BLOCK_DIM + inner_col
            elem = offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col
            a_re = tl.load(data_ri_ptr + elem * 2, mask=mask, other=0.0)
            a_im = tl.load(data_ri_ptr + elem * 2 + 1, mask=mask, other=0.0)
            x_re = tl.load(x_ri_ptr + col * 2, mask=mask, other=0.0)
            x_im = tl.load(x_ri_ptr + col * 2 + 1, mask=mask, other=0.0)
            prod_re = a_re * x_re - a_im * x_im
            prod_im = a_re * x_im + a_im * x_re
            acc_re += tl.sum(tl.where(mask, prod_re, 0.0))
            acc_im += tl.sum(tl.where(mask, prod_im, 0.0))
        out_row = brow * BLOCK_DIM + inner_row
        if DIRECT_STORE:
            tl.store(y_ri_ptr + out_row * 2, acc_re)
            tl.store(y_ri_ptr + out_row * 2 + 1, acc_im)
        else:
            tl.atomic_add(y_ri_ptr + out_row * 2, acc_re)
            tl.atomic_add(y_ri_ptr + out_row * 2 + 1, acc_im)


@triton.jit
def _spmv_bsr_trans_real_kernel(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    x_ptr,
    y_ptr,
    n_block_rows,
    BLOCK_DIM: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    if brow >= n_block_rows:
        return
    row = brow * BLOCK_DIM + inner_row
    x_val = tl.load(x_ptr + row)
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    for inner_col in tl.static_range(0, BLOCK_DIM):
        col = bcols * BLOCK_DIM + inner_col
        vals = tl.load(
            data_ptr + offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col,
            mask=mask,
            other=0.0,
        )
        tl.atomic_add(y_ptr + col, vals * x_val, mask=mask)


@triton.jit
def _spmv_bsr_trans_complex_kernel(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    x_ri_ptr,
    y_ri_ptr,
    n_block_rows,
    BLOCK_DIM: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    SEG: tl.constexpr,
    CONJ: tl.constexpr,
):
    brow = tl.program_id(0)
    inner_row = tl.program_id(1)
    if brow >= n_block_rows:
        return
    row = brow * BLOCK_DIM + inner_row
    x_re = tl.load(x_ri_ptr + row * 2)
    x_im = tl.load(x_ri_ptr + row * 2 + 1)
    start = tl.load(indptr_ptr + brow)
    end = tl.load(indptr_ptr + brow + 1)
    offs = start + SEG * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
    mask = offs < end
    bcols = tl.load(indices_ptr + offs, mask=mask, other=0)
    for inner_col in tl.static_range(0, BLOCK_DIM):
        col = bcols * BLOCK_DIM + inner_col
        elem = offs * BLOCK_DIM * BLOCK_DIM + inner_row * BLOCK_DIM + inner_col
        a_re = tl.load(data_ri_ptr + elem * 2, mask=mask, other=0.0)
        a_im_raw = tl.load(data_ri_ptr + elem * 2 + 1, mask=mask, other=0.0)
        if CONJ:
            a_im = -a_im_raw
        else:
            a_im = a_im_raw
        prod_re = a_re * x_re - a_im * x_im
        prod_im = a_re * x_im + a_im * x_re
        tl.atomic_add(y_ri_ptr + col * 2, prod_re, mask=mask)
        tl.atomic_add(y_ri_ptr + col * 2 + 1, prod_im, mask=mask)


def _prepare_spmv_bsr_matrix(data, indices, indptr, shape, block_dim):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr)):
        raise TypeError("data, indices, indptr must all be torch.Tensor")
    if data.ndim != 3:
        raise ValueError("data must have shape (nnzb, block_dim, block_dim)")
    if indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("indices and indptr must be 1D tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    block_dim = int(block_dim)
    if block_dim <= 1:
        raise ValueError("block_dim must be greater than 1 for BSR SpMV")
    if data.shape[1] != block_dim or data.shape[2] != block_dim:
        raise ValueError("data block dimensions must match block_dim")
    n_block_rows = (n_rows + block_dim - 1) // block_dim
    n_block_cols = (n_cols + block_dim - 1) // block_dim
    if indptr.numel() != n_block_rows + 1:
        raise ValueError(
            f"indptr length must be n_block_rows+1={n_block_rows + 1}, got {indptr.numel()}"
        )
    if data.shape[0] != indices.numel():
        raise ValueError("data.shape[0] and indices length must both equal nnzb")
    if not all(_is_accel_tensor(t) for t in (data, indices, indptr)):
        raise ValueError("data, indices, indptr must be CUDA tensors")
    if not all(t.device == data.device for t in (indices, indptr)):
        raise ValueError("data, indices, indptr must be on the same CUDA device")
    if data.dtype not in SUPPORTED_SPMV_BSR_VALUE_DTYPES:
        raise TypeError(_spmv_bsr_dtype_error_message())
    if indices.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32 or torch.int64")
    if indptr.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32 or torch.int64")
    data = data.contiguous()
    indices = indices.contiguous()
    indptr = indptr.contiguous()
    if int(indptr[0].item()) != 0:
        raise ValueError("indptr must start at zero")
    if int(indptr[-1].item()) != data.shape[0]:
        raise ValueError("indptr[-1] must equal nnzb")
    if indptr.numel() > 1 and torch.any(indptr[1:] < indptr[:-1]).item():
        raise ValueError("indptr must be non-decreasing")
    if indices.numel() > 0:
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= n_block_cols:
            raise IndexError("indices out of range for n_block_cols")
    block_row_lengths = indptr[1:] - indptr[:-1]
    max_block_row_nnz = (
        int(block_row_lengths.max().item()) if n_block_rows > 0 else 0
    )
    return (
        data,
        indices,
        indptr,
        n_rows,
        n_cols,
        n_block_rows,
        n_block_cols,
        block_row_lengths,
        max_block_row_nnz,
    )


def prepare_spmv_bsr(
    data,
    indices,
    indptr,
    shape,
    block_dim,
    block_nnz=128,
    max_segments=None,
    transpose=False,
    op=None,
    use_opt=False,
    index_fallback_policy="auto",
):
    index_fallback_policy = _normalize_spmv_bsr_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_bsr_op(op, transpose=transpose)
    _ensure_spmv_bsr_supported_op(op_code)
    algorithm = _normalize_spmv_bsr_algorithm(use_opt)
    if algorithm == SPMV_BSR_ALG_BLOCKROW_REDUCE and op_code != SPMV_BSR_OP_NON:
        raise ValueError("spmv_bsr_blockrow_reduce only supports op='non'")
    if algorithm == SPMV_BSR_ALG_BLOCKROW_REDUCE and int(block_dim) not in (2, 4, 8, 16):
        raise ValueError("spmv_bsr_blockrow_reduce supports block_dim values 2, 4, 8, and 16")
    (
        data,
        indices,
        indptr,
        n_rows,
        n_cols,
        n_block_rows,
        n_block_cols,
        block_row_lengths,
        max_block_row_nnz,
    ) = _prepare_spmv_bsr_matrix(data, indices, indptr, shape, block_dim)
    block_nnz_use = int(block_nnz)
    if block_nnz_use <= 0:
        raise ValueError("block_nnz must be positive")
    if max_segments is None:
        max_segments_use = max((max_block_row_nnz + block_nnz_use - 1) // block_nnz_use, 1)
        while max_segments_use > 2048 and block_nnz_use < 65536:
            block_nnz_use *= 2
            max_segments_use = max(
                (max_block_row_nnz + block_nnz_use - 1) // block_nnz_use,
                1,
            )
    else:
        max_segments_use = max(1, int(max_segments))
    return PreparedBsrSpmv(
        data=data,
        kernel_indices=indices,
        kernel_indptr=indptr,
        shape=shape,
        block_dim=block_dim,
        n_block_rows=n_block_rows,
        n_block_cols=n_block_cols,
        block_nnz=block_nnz_use,
        max_segments=max_segments_use,
        max_block_row_nnz=max_block_row_nnz,
        block_row_lengths=block_row_lengths,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
    )


def _validate_spmv_bsr_x(x, prepared, op_code):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not _is_accel_tensor(x):
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    logical_expected = prepared.n_rows if _spmv_bsr_op_transposes(op_code) else prepared.n_cols
    padded_expected = (
        prepared.padded_n_rows
        if _spmv_bsr_op_transposes(op_code)
        else prepared.padded_n_cols
    )
    if x.numel() not in (logical_expected, padded_expected):
        raise ValueError(
            f"x length must be {logical_expected} or padded length {padded_expected}, got {x.numel()}"
        )
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    x = x.contiguous()
    if x.numel() == padded_expected:
        return x
    padded = torch.zeros(padded_expected, dtype=x.dtype, device=x.device)
    padded[: x.numel()].copy_(x)
    return padded


def _triton_spmv_bsr_kernel(prepared, x, op_code):
    _ensure_spmv_bsr_supported_op(op_code)
    dtype = prepared.data.dtype
    trans = _spmv_bsr_op_transposes(op_code)
    out_len = prepared.padded_n_cols if trans else prepared.padded_n_rows
    y = torch.zeros(out_len, dtype=dtype, device=prepared.data.device)
    if prepared.nnzb == 0:
        return y
    for seg in range(prepared.max_segments):
        grid = (prepared.n_block_rows, prepared.block_dim)
        if _is_complex_dtype(dtype):
            data_ri = torch.view_as_real(prepared.data).reshape(-1)
            x_ri = torch.view_as_real(x).reshape(-1)
            y_ri = torch.view_as_real(y).reshape(-1)
            if trans:
                _spmv_bsr_trans_complex_kernel[grid](
                    data_ri,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    x_ri,
                    y_ri,
                    prepared.n_block_rows,
                    BLOCK_DIM=prepared.block_dim,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                    CONJ=(op_code == SPMV_BSR_OP_CONJ_TRANS),
                )
            else:
                _spmv_bsr_non_complex_kernel[grid](
                    data_ri,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    x_ri,
                    y_ri,
                    prepared.padded_n_rows,
                    prepared.padded_n_cols,
                    prepared.n_block_rows,
                    BLOCK_DIM=prepared.block_dim,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                )
        else:
            if trans:
                _spmv_bsr_trans_real_kernel[grid](
                    prepared.data,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    x,
                    y,
                    prepared.n_block_rows,
                    BLOCK_DIM=prepared.block_dim,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                )
            else:
                _spmv_bsr_non_real_kernel[grid](
                    prepared.data,
                    prepared.kernel_indices,
                    prepared.kernel_indptr,
                    x,
                    y,
                    prepared.padded_n_rows,
                    prepared.padded_n_cols,
                    prepared.n_block_rows,
                    BLOCK_DIM=prepared.block_dim,
                    BLOCK_NNZ=prepared.block_nnz,
                    SEG=seg,
                )
    return y


def _spmv_bsr_blockrow_reduce_bucket_specs():
    return (
        ("le1", 0, 1, 1, True),
        ("le4", 2, 4, 4, True),
        ("le16", 5, 16, 16, True),
        ("le64", 17, 64, 64, True),
        ("gt64", 65, None, 128, False),
    )


def _build_spmv_bsr_blockrow_reduce_buckets(row_lengths, max_block_row_nnz):
    buckets = []
    lengths = row_lengths.to(torch.int64)
    for label, low, high, block_nnz, direct_store in _spmv_bsr_blockrow_reduce_bucket_specs():
        mask = lengths >= int(low)
        if high is not None:
            mask = mask & (lengths <= int(high))
        rows = torch.nonzero(mask, as_tuple=False).flatten().to(torch.int64).contiguous()
        if rows.numel() == 0:
            continue
        segments = 1 if direct_store else max(1, (int(max_block_row_nnz) + block_nnz - 1) // block_nnz)
        buckets.append(
            {
                "label": label,
                "rows": rows,
                "block_nnz": int(block_nnz),
                "segments": int(segments),
                "direct_store": bool(direct_store),
            }
        )
    return buckets


def _triton_spmv_bsr_blockrow_reduce_kernel(prepared, x, buckets=None):
    if prepared.op != SPMV_BSR_OP_NON:
        raise ValueError("spmv_bsr_blockrow_reduce only supports op='non'")
    dtype = prepared.data.dtype
    y = torch.zeros(prepared.padded_n_rows, dtype=dtype, device=prepared.data.device)
    if prepared.nnzb == 0:
        return y
    if buckets is None:
        buckets = _build_spmv_bsr_blockrow_reduce_buckets(
            prepared.block_row_lengths, prepared.max_block_row_nnz
        )
    if _is_complex_dtype(dtype):
        data_ri = torch.view_as_real(prepared.data).reshape(-1)
        x_ri = torch.view_as_real(x).reshape(-1)
        y_ri = torch.view_as_real(y).reshape(-1)
        for bucket in buckets:
            rows = bucket["rows"]
            if rows.numel() == 0:
                continue
            grid = (rows.numel(), bucket["segments"])
            _spmv_bsr_blockrow_reduce_complex_kernel[grid](
                data_ri,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                rows,
                x_ri,
                y_ri,
                rows.numel(),
                BLOCK_DIM=prepared.block_dim,
                BLOCK_NNZ=bucket["block_nnz"],
                DIRECT_STORE=bucket["direct_store"],
            )
    else:
        for bucket in buckets:
            rows = bucket["rows"]
            if rows.numel() == 0:
                continue
            grid = (rows.numel(), bucket["segments"])
            _spmv_bsr_blockrow_reduce_real_kernel[grid](
                prepared.data,
                prepared.kernel_indices,
                prepared.kernel_indptr,
                rows,
                x,
                y,
                rows.numel(),
                BLOCK_DIM=prepared.block_dim,
                BLOCK_NNZ=bucket["block_nnz"],
                DIRECT_STORE=bucket["direct_store"],
            )
    return y


def _run_spmv_bsr_blockrow_reduce_with_timing(prepared, x):
    _ACCEL.synchronize()
    process_start = _ACCEL.Event(enable_timing=True)
    process_end = _ACCEL.Event(enable_timing=True)
    process_start.record()
    buckets = _build_spmv_bsr_blockrow_reduce_buckets(
        prepared.block_row_lengths, prepared.max_block_row_nnz
    )
    process_end.record()
    _ACCEL.synchronize()
    process_gpu_ms = process_start.elapsed_time(process_end)
    compute_start = _ACCEL.Event(enable_timing=True)
    compute_end = _ACCEL.Event(enable_timing=True)
    compute_start.record()
    y = _triton_spmv_bsr_blockrow_reduce_kernel(prepared, x, buckets=buckets)
    compute_end.record()
    _ACCEL.synchronize()
    compute_ms = compute_start.elapsed_time(compute_end)
    return y, {
        "process_cpu_ms": 0.0,
        "process_gpu_ms": process_gpu_ms,
        "compute_ms": compute_ms,
        "bucket_counts": [
            {"bucket": bucket["label"], "rows": int(bucket["rows"].numel())}
            for bucket in buckets
        ],
        "launch_configs": [
            {
                "bucket": bucket["label"],
                "block_nnz": bucket["block_nnz"],
                "segments": bucket["segments"],
                "direct_store": bucket["direct_store"],
            }
            for bucket in buckets
        ],
    }


def _spmv_bsr_uses_int64_indices(prepared):
    return (
        prepared.kernel_indices.dtype == torch.int64
        or prepared.kernel_indptr.dtype == torch.int64
    )


def _spmv_bsr_int32_fallback_blocker(prepared):
    if prepared.nnzb > _INDEX_LIMIT_INT32:
        return f"nnzb {prepared.nnzb} cannot fit int32"
    if prepared.kernel_indices.numel() > 0:
        max_col = int(prepared.kernel_indices.max().item())
        if max_col > _INDEX_LIMIT_INT32:
            return f"block column index {max_col} cannot fit int32"
    if prepared.kernel_indptr.numel() > 0:
        max_ptr = int(prepared.kernel_indptr[-1].item())
        if max_ptr > _INDEX_LIMIT_INT32:
            return f"indptr offset {max_ptr} cannot fit int32"
    return None


def _spmv_bsr_prepared_with_int32_indices(prepared, reason):
    blocker = _spmv_bsr_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(f"int32 fallback is unsafe: {blocker}") from reason
    return PreparedBsrSpmv(
        data=prepared.data,
        kernel_indices=prepared.kernel_indices.to(torch.int32).contiguous(),
        kernel_indptr=prepared.kernel_indptr.to(torch.int32).contiguous(),
        shape=prepared.shape,
        block_dim=prepared.block_dim,
        n_block_rows=prepared.n_block_rows,
        n_block_cols=prepared.n_block_cols,
        block_nnz=prepared.block_nnz,
        max_segments=prepared.max_segments,
        max_block_row_nnz=prepared.max_block_row_nnz,
        block_row_lengths=prepared.block_row_lengths,
        op=prepared.op,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
    )


def _run_spmv_bsr_algorithm(prepared, x, op_code, algorithm, collect_timing=False):
    if algorithm == SPMV_BSR_ALG_BLOCKROW_REDUCE:
        if op_code != SPMV_BSR_OP_NON:
            raise ValueError("spmv_bsr_blockrow_reduce only supports op='non'")
        if collect_timing:
            return _run_spmv_bsr_blockrow_reduce_with_timing(prepared, x)
        return _triton_spmv_bsr_blockrow_reduce_kernel(prepared, x), {
            "process_cpu_ms": 0.0,
            "process_gpu_ms": None,
            "compute_ms": None,
            "bucket_counts": None,
            "launch_configs": None,
        }
    if collect_timing:
        _ACCEL.synchronize()
        compute_start = _ACCEL.Event(enable_timing=True)
        compute_end = _ACCEL.Event(enable_timing=True)
        compute_start.record()
        y = _triton_spmv_bsr_kernel(prepared, x, op_code)
        compute_end.record()
        _ACCEL.synchronize()
        compute_ms = compute_start.elapsed_time(compute_end)
        return y, {
            "process_cpu_ms": 0.0,
            "process_gpu_ms": 0.0,
            "compute_ms": compute_ms,
            "bucket_counts": None,
            "launch_configs": None,
        }
    return _triton_spmv_bsr_kernel(prepared, x, op_code), {
        "process_cpu_ms": 0.0,
        "process_gpu_ms": None,
        "compute_ms": None,
        "bucket_counts": None,
        "launch_configs": None,
    }


def _run_spmv_bsr_prepared_with_fallback(prepared, x, op_code, algorithm, collect_timing=False):
    try:
        return _run_spmv_bsr_algorithm(prepared, x, op_code, algorithm, collect_timing=collect_timing)
    except RuntimeError as exc:
        if (
            prepared.index_fallback_policy != "auto"
            or not _spmv_bsr_uses_int64_indices(prepared)
        ):
            raise
        fallback_prepared = _spmv_bsr_prepared_with_int32_indices(prepared, exc)
        return _run_spmv_bsr_algorithm(
            fallback_prepared, x, op_code, algorithm, collect_timing=collect_timing
        )


def flagsparse_spmv_bsr(
    data=None,
    indices=None,
    indptr=None,
    x=None,
    shape=None,
    block_dim=None,
    block_nnz=128,
    max_segments=None,
    out=None,
    return_time=False,
    return_meta=False,
    prepared=None,
    transpose=None,
    op=None,
    use_opt=False,
    index_fallback_policy="auto",
):
    """BSR SpMV using a native Triton BSR kernel."""
    op_explicit = op is not None
    op_code = _normalize_spmv_bsr_op(
        op,
        transpose=False if transpose is None else bool(transpose),
    )
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmv_bsr_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    _ensure_spmv_bsr_supported_op(op_code)
    algorithm = _normalize_spmv_bsr_algorithm(use_opt)
    if algorithm == SPMV_BSR_ALG_BLOCKROW_REDUCE and op_code != SPMV_BSR_OP_NON:
        raise ValueError("spmv_bsr_blockrow_reduce only supports op='non'")
    if prepared is None:
        if any(arg is None for arg in (data, indices, indptr, shape, block_dim)):
            raise ValueError(
                "data, indices, indptr, shape, and block_dim are required when prepared is not provided"
            )
        prepared = prepare_spmv_bsr(
            data,
            indices,
            indptr,
            shape,
            block_dim,
            block_nnz=block_nnz,
            max_segments=max_segments,
            op=op_code,
            use_opt=algorithm,
            index_fallback_policy=index_fallback_policy,
        )
    else:
        if op_explicit and op_code != prepared.op:
            raise ValueError(
                f"op={_spmv_bsr_op_to_name(op_code)} does not match prepared.op={_spmv_bsr_op_to_name(prepared.op)}"
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
    if algorithm == SPMV_BSR_ALG_BLOCKROW_REDUCE and op_code != SPMV_BSR_OP_NON:
        raise ValueError("spmv_bsr_blockrow_reduce only supports op='non'")
    if algorithm == SPMV_BSR_ALG_BLOCKROW_REDUCE and prepared.block_dim not in (2, 4, 8, 16):
        raise ValueError("spmv_bsr_blockrow_reduce supports block_dim values 2, 4, 8, and 16")
    x = _validate_spmv_bsr_x(x, prepared, op_code)
    do_timing = bool(return_time or return_meta)
    if do_timing:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    y, alg_timing = _run_spmv_bsr_prepared_with_fallback(
        prepared, x, op_code, algorithm, collect_timing=do_timing
    )
    if do_timing:
        _ACCEL.synchronize()
        wall_total_ms = (time.perf_counter() - t0) * 1000.0
        compute_ms = alg_timing["compute_ms"]
        process_cpu_ms = alg_timing["process_cpu_ms"]
        process_gpu_ms = alg_timing["process_gpu_ms"]
        op_total_ms = (
            process_cpu_ms + process_gpu_ms + compute_ms
            if process_gpu_ms is not None and compute_ms is not None
            else wall_total_ms
        )
    else:
        compute_ms = None
        process_cpu_ms = None
        process_gpu_ms = None
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
            "algorithm": algorithm,
            "op": _spmv_bsr_op_to_name(op_code),
            "block_dim": prepared.block_dim,
            "logical_shape": prepared.shape,
            "padded_shape": (prepared.padded_n_rows, prepared.padded_n_cols),
            "n_block_rows": prepared.n_block_rows,
            "n_block_cols": prepared.n_block_cols,
            "nnzb": prepared.nnzb,
            "stored_nnz": prepared.stored_nnz,
            "symbolic_ms": (process_cpu_ms or 0.0) + (process_gpu_ms or 0.0) if do_timing else None,
            "process_cpu_ms": process_cpu_ms,
            "process_gpu_ms": process_gpu_ms,
            "compute_ms": compute_ms,
            "op_total_ms": op_total_ms,
            "bucket_counts": alg_timing["bucket_counts"],
            "launch_configs": alg_timing["launch_configs"],
            "index_fallback_applied": prepared.index_fallback_applied,
            "index_fallback_reason": prepared.index_fallback_reason,
        }
        if return_time:
            return y, op_total_ms, meta
        return y, meta
    if return_time:
        return y, op_total_ms
    return y
