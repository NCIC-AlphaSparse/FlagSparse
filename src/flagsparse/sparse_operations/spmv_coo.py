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

"""COO SpMV **without CSR / indptr** (fp32/fp64/complex64/complex128).

- ``sort_by_row=True`` (default): lex-sort (row,col), build compact **row-run** offsets
  ``seg_starts`` (length #runs+1, not ``n_rows+1``), one Triton program per run — register
  reduction + single ``tl.store`` per output row (no atomics on ``y``).
- ``sort_by_row=False``: grid over NNZ with ``tl.atomic_add`` (slower / contentions).

Storage: sorted ``data, row, col`` plus optional ``seg_starts`` vector — never ``indptr``.
"""

from ._common import *

import time

import triton
import triton.language as tl

SUPPORTED_SPMV_COO_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)

SPMV_COO_OP_NON = 0
SPMV_COO_OP_TRANS = 1
SPMV_COO_OP_CONJ_TRANS = 2
SPMV_COO_OP_NAMES = {
    SPMV_COO_OP_NON: "non",
    SPMV_COO_OP_TRANS: "trans",
    SPMV_COO_OP_CONJ_TRANS: "conj",
}
_SPMV_COO_OP_NAME_TO_CODE = {name: code for code, name in SPMV_COO_OP_NAMES.items()}


def _spmv_coo_dtype_error_message():
    return "COO SpMV supports float32, float64, complex64, and complex128"


def _normalize_spmv_coo_op(op=None, transpose=False):
    if op is None:
        return SPMV_COO_OP_TRANS if bool(transpose) else SPMV_COO_OP_NON
    if isinstance(op, str):
        token = op.strip().lower()
        if token not in _SPMV_COO_OP_NAME_TO_CODE:
            raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
        return _SPMV_COO_OP_NAME_TO_CODE[token]
    try:
        op_code = int(op)
    except (TypeError, ValueError) as exc:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj") from exc
    if op_code not in SPMV_COO_OP_NAMES:
        raise ValueError("op must be one of: 0=non, 1=trans, 2=conj")
    return op_code


def _spmv_coo_op_to_name(op):
    op_code = _normalize_spmv_coo_op(op)
    return SPMV_COO_OP_NAMES[op_code]


def _spmv_coo_op_transposes(op):
    return _normalize_spmv_coo_op(op) in (
        SPMV_COO_OP_TRANS,
        SPMV_COO_OP_CONJ_TRANS,
    )


def _normalize_spmv_coo_index_fallback_policy(index_fallback_policy):
    policy = str(index_fallback_policy).lower()
    if policy not in ("auto", "strict"):
        raise ValueError("index_fallback_policy must be 'auto' or 'strict'")
    return policy


class PreparedCoo:
    """Prepared COO structure metadata that can serve non/trans/conj at runtime."""

    __slots__ = (
        "data_non",
        "row_non",
        "col_non",
        "seg_starts_non",
        "data_trans",
        "row_trans",
        "col_trans",
        "seg_starts_trans",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "sort_by_row",
        "op",
        "transpose",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
    )

    def __init__(
        self,
        data_non,
        row_non,
        col_non,
        shape,
        seg_starts_non=None,
        data_trans=None,
        row_trans=None,
        col_trans=None,
        seg_starts_trans=None,
        sort_by_row=True,
        transpose=False,
        op=None,
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
    ):
        self.data_non = data_non
        self.row_non = row_non
        self.col_non = col_non
        self.seg_starts_non = seg_starts_non
        self.data_trans = data_trans if data_trans is not None else data_non
        self.row_trans = row_trans if row_trans is not None else col_non
        self.col_trans = col_trans if col_trans is not None else row_non
        self.seg_starts_trans = seg_starts_trans
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows, self.n_cols = self.shape
        self.nnz = int(data_non.numel())
        self.sort_by_row = bool(sort_by_row)
        self.op = _normalize_spmv_coo_op(op, transpose=transpose)
        self.transpose = _spmv_coo_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason


class _PreparedCooLaunch:
    """Concrete launch view for one runtime op on top of structure-only prepare."""

    __slots__ = (
        "data",
        "row",
        "col",
        "shape",
        "n_rows",
        "n_cols",
        "nnz",
        "seg_starts",
        "n_segs",
        "use_seg_kernel",
        "op",
        "transpose",
        "index_fallback_policy",
        "index_fallback_applied",
        "index_fallback_reason",
    )

    def __init__(
        self,
        data,
        row,
        col,
        shape,
        seg_starts=None,
        op=SPMV_COO_OP_NON,
        index_fallback_policy="auto",
        index_fallback_applied=False,
        index_fallback_reason=None,
    ):
        self.data = data
        self.row = row
        self.col = col
        self.shape = (int(shape[0]), int(shape[1]))
        self.n_rows, self.n_cols = self.shape
        self.nnz = int(data.numel())
        self.seg_starts = seg_starts
        if seg_starts is None:
            self.n_segs = 0
            self.use_seg_kernel = False
        else:
            self.n_segs = int(seg_starts.numel()) - 1
            self.use_seg_kernel = self.n_segs > 0
        self.op = _normalize_spmv_coo_op(op)
        self.transpose = _spmv_coo_op_transposes(self.op)
        self.index_fallback_policy = str(index_fallback_policy).lower()
        self.index_fallback_applied = bool(index_fallback_applied)
        self.index_fallback_reason = index_fallback_reason


@triton.jit
def _spmv_coo_seg_f32(
    data_ptr,
    col_ptr,
    row_ptr,
    x_ptr,
    y_ptr,
    seg_starts_ptr,
    n_segs,
    BLOCK_INNER: tl.constexpr,
):
    seg = tl.program_id(0)
    if seg >= n_segs:
        return
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_id = tl.load(row_ptr + start)
    acc = tl.zeros((), dtype=tl.float32)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK_INNER)
        m = offs < end
        v = tl.load(data_ptr + offs, mask=m, other=0.0)
        c = tl.load(col_ptr + offs, mask=m, other=0)
        xv = tl.load(x_ptr + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, v * xv, 0.0))
        pos += BLOCK_INNER
    tl.store(y_ptr + row_id, acc)


@triton.jit
def _spmv_coo_seg_f64(
    data_ptr,
    col_ptr,
    row_ptr,
    x_ptr,
    y_ptr,
    seg_starts_ptr,
    n_segs,
    BLOCK_INNER: tl.constexpr,
):
    seg = tl.program_id(0)
    if seg >= n_segs:
        return
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_id = tl.load(row_ptr + start)
    acc = tl.zeros((), dtype=tl.float64)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK_INNER)
        m = offs < end
        v = tl.load(data_ptr + offs, mask=m, other=0.0)
        c = tl.load(col_ptr + offs, mask=m, other=0)
        xv = tl.load(x_ptr + c, mask=m, other=0.0)
        acc += tl.sum(tl.where(m, v * xv, 0.0))
        pos += BLOCK_INNER
    tl.store(y_ptr + row_id, acc)


@triton.jit
def _spmv_coo_seg_complex(
    data_ri_ptr,
    col_ptr,
    row_ptr,
    x_ri_ptr,
    y_ri_ptr,
    seg_starts_ptr,
    n_segs,
    BLOCK_INNER: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    seg = tl.program_id(0)
    if seg >= n_segs:
        return
    start = tl.load(seg_starts_ptr + seg)
    end = tl.load(seg_starts_ptr + seg + 1)
    row_id = tl.load(row_ptr + start)
    acc_re = tl.zeros((), dtype=ACC_DTYPE)
    acc_im = tl.zeros((), dtype=ACC_DTYPE)
    pos = start
    while pos < end:
        offs = pos + tl.arange(0, BLOCK_INNER)
        m = offs < end
        a_re = tl.load(data_ri_ptr + offs * 2, mask=m, other=0.0)
        a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=m, other=0.0)
        c = tl.load(col_ptr + offs, mask=m, other=0)
        x_re = tl.load(x_ri_ptr + c * 2, mask=m, other=0.0)
        x_im = tl.load(x_ri_ptr + c * 2 + 1, mask=m, other=0.0)
        prod_re = tl.where(m, a_re * x_re - a_im * x_im, 0.0)
        prod_im = tl.where(m, a_re * x_im + a_im * x_re, 0.0)
        acc_re += tl.sum(prod_re)
        acc_im += tl.sum(prod_im)
        pos += BLOCK_INNER
    tl.store(y_ri_ptr + row_id * 2, acc_re)
    tl.store(y_ri_ptr + row_id * 2 + 1, acc_im)


@triton.jit
def _spmv_coo_atomic_f32(
    data_ptr,
    row_ptr,
    col_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    v = tl.load(data_ptr + offs, mask=m, other=0.0)
    xv = tl.load(x_ptr + c, mask=m, other=0.0)
    contrib = tl.where(m, v * xv, 0.0).to(tl.float32)
    tl.atomic_add(y_ptr + r, contrib, mask=m, sem="relaxed")


@triton.jit
def _spmv_coo_atomic_complex(
    data_ri_ptr,
    row_ptr,
    col_ptr,
    x_ri_ptr,
    y_ri_ptr,
    nnz,
    BLOCK: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    a_re = tl.load(data_ri_ptr + offs * 2, mask=m, other=0.0)
    a_im = tl.load(data_ri_ptr + offs * 2 + 1, mask=m, other=0.0)
    x_re = tl.load(x_ri_ptr + c * 2, mask=m, other=0.0)
    x_im = tl.load(x_ri_ptr + c * 2 + 1, mask=m, other=0.0)
    prod_re = tl.where(m, a_re * x_re - a_im * x_im, 0.0).to(ACC_DTYPE)
    prod_im = tl.where(m, a_re * x_im + a_im * x_re, 0.0).to(ACC_DTYPE)
    tl.atomic_add(y_ri_ptr + r * 2, prod_re, mask=m, sem="relaxed")
    tl.atomic_add(y_ri_ptr + r * 2 + 1, prod_im, mask=m, sem="relaxed")


@triton.jit
def _spmv_coo_atomic_f64(
    data_ptr,
    row_ptr,
    col_ptr,
    x_ptr,
    y_ptr,
    nnz,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < nnz
    r = tl.load(row_ptr + offs, mask=m, other=0)
    c = tl.load(col_ptr + offs, mask=m, other=0)
    v = tl.load(data_ptr + offs, mask=m, other=0.0)
    xv = tl.load(x_ptr + c, mask=m, other=0.0)
    contrib = tl.where(m, v * xv, 0.0).to(tl.float64)
    tl.atomic_add(y_ptr + r, contrib, mask=m, sem="relaxed")


def _sort_coo_lex_inplace(data, row, col, n_cols):
    row64 = row.to(torch.int64)
    col64 = col.to(torch.int64)
    index_dtype = torch.promote_types(row.dtype, col.dtype)
    if data.numel() == 0:
        return (
            data.contiguous(),
            row.to(index_dtype).contiguous(),
            col.to(index_dtype).contiguous(),
        )
    key = row64 * max(1, int(n_cols)) + col64
    order = torch.argsort(key)
    return (
        data[order].contiguous(),
        row[order].to(index_dtype).contiguous(),
        col[order].to(index_dtype).contiguous(),
    )


def _seg_starts_from_sorted_rows(row, nnz, device):
    """Boundaries of constant-row runs in sorted COO."""
    if nnz == 0:
        return None
    index_dtype = row.dtype
    if nnz > _INDEX_LIMIT_INT32:
        index_dtype = torch.int64
    diff = row[1:] != row[:-1]
    breaks = torch.nonzero(diff, as_tuple=False).flatten().to(index_dtype) + 1
    return torch.cat(
        [
            torch.zeros(1, dtype=index_dtype, device=device),
            breaks,
            torch.tensor([nnz], dtype=index_dtype, device=device),
        ]
    )


def _prepare_coo_tensors(data, row, col, shape, sort_by_row):
    if not all(torch.is_tensor(t) for t in (data, row, col)):
        raise TypeError("data, row, col must all be torch.Tensor")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if not all(_is_accel_tensor(t) for t in (data, row, col)):
        raise ValueError("data, row, col must be CUDA tensors")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have the same length")
    if data.dtype not in SUPPORTED_SPMV_COO_VALUE_DTYPES:
        raise TypeError(_spmv_coo_dtype_error_message())
    if row.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("row dtype must be torch.int32 or torch.int64")
    if col.dtype not in SUPPORTED_INDEX_DTYPES:
        raise TypeError("col dtype must be torch.int32 or torch.int64")
    index_dtype = torch.promote_types(row.dtype, col.dtype)
    if sort_by_row:
        data, kr, kc = _sort_coo_lex_inplace(data, row, col, n_cols)
        seg = _seg_starts_from_sorted_rows(kr, data.numel(), data.device)
    else:
        data = data.contiguous()
        kr = row.to(index_dtype).contiguous()
        kc = col.to(index_dtype).contiguous()
        seg = None
    if data.numel() > 0:
        row64 = kr.to(torch.int64)
        col64 = kc.to(torch.int64)
        if int(row64.min().item()) < 0 or int(row64.max().item()) >= n_rows:
            raise IndexError("row indices out of range")
        if int(col64.min().item()) < 0 or int(col64.max().item()) >= n_cols:
            raise IndexError("col indices out of range")
    return data, kr, kc, seg


def prepare_spmv_coo(
    data,
    row,
    col,
    shape,
    sort_by_row=True,
    transpose=False,
    op=None,
    index_fallback_policy="auto",
):
    """Cache sorted COO + row-run ``seg_starts`` when ``sort_by_row``. No ``indptr``."""
    index_fallback_policy = _normalize_spmv_coo_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_coo_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_COO_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    shape = (int(shape[0]), int(shape[1]))
    data_non, row_non, col_non, seg_non = _prepare_coo_tensors(
        data, row, col, shape, sort_by_row
    )
    trans_shape = (shape[1], shape[0])
    if sort_by_row:
        data_trans, row_trans, col_trans, seg_trans = _prepare_coo_tensors(
            data,
            col,
            row,
            trans_shape,
            True,
        )
    else:
        data_trans = data_non
        row_trans = col_non
        col_trans = row_non
        seg_trans = None
    return PreparedCoo(
        data_non,
        row_non,
        col_non,
        shape,
        seg_starts_non=seg_non,
        data_trans=data_trans,
        row_trans=row_trans,
        col_trans=col_trans,
        seg_starts_trans=seg_trans,
        sort_by_row=sort_by_row,
        transpose=transpose,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
    )


def _prepare_spmv_coo_launch_from_raw(
    data,
    row,
    col,
    shape,
    sort_by_row=True,
    transpose=False,
    op=None,
    index_fallback_policy="auto",
):
    """Build one runtime launch view from raw COO inputs for benchmark-only timing."""
    index_fallback_policy = _normalize_spmv_coo_index_fallback_policy(
        index_fallback_policy
    )
    op_code = _normalize_spmv_coo_op(op, transpose=transpose)
    if op is not None and bool(transpose) and op_code == SPMV_COO_OP_NON:
        raise ValueError("transpose=True conflicts with op=non")
    shape = (int(shape[0]), int(shape[1]))
    launch_shape = shape
    launch_data = data
    launch_row = row
    launch_col = col
    if _spmv_coo_op_transposes(op_code):
        launch_shape = (shape[1], shape[0])
        launch_row = col
        launch_col = row
        if op_code == SPMV_COO_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
            launch_data = data.conj()
            if hasattr(launch_data, "resolve_conj"):
                launch_data = launch_data.resolve_conj()
    launch_data, launch_row, launch_col, launch_seg_starts = _prepare_coo_tensors(
        launch_data,
        launch_row,
        launch_col,
        launch_shape,
        sort_by_row,
    )
    return _PreparedCooLaunch(
        data=launch_data,
        row=launch_row,
        col=launch_col,
        shape=launch_shape,
        seg_starts=launch_seg_starts,
        op=op_code,
        index_fallback_policy=index_fallback_policy,
    )


def _resolve_spmv_coo_launch(prepared, op):
    op_code = _normalize_spmv_coo_op(op)
    if op_code == SPMV_COO_OP_NON:
        return _PreparedCooLaunch(
            data=prepared.data_non,
            row=prepared.row_non,
            col=prepared.col_non,
            shape=prepared.shape,
            seg_starts=prepared.seg_starts_non,
            op=op_code,
            index_fallback_policy=prepared.index_fallback_policy,
            index_fallback_applied=prepared.index_fallback_applied,
            index_fallback_reason=prepared.index_fallback_reason,
        )
    data = prepared.data_trans
    if op_code == SPMV_COO_OP_CONJ_TRANS and _is_complex_dtype(data.dtype):
        data = data.conj()
        if hasattr(data, "resolve_conj"):
            data = data.resolve_conj()
    return _PreparedCooLaunch(
        data=data,
        row=prepared.row_trans,
        col=prepared.col_trans,
        shape=(prepared.n_cols, prepared.n_rows),
        seg_starts=prepared.seg_starts_trans,
        op=op_code,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=prepared.index_fallback_applied,
        index_fallback_reason=prepared.index_fallback_reason,
    )


def _validate_x_coo(x, prepared):
    if x is None or not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if not _is_accel_tensor(x):
        raise ValueError("x must be a CUDA tensor")
    if x.dtype != prepared.data.dtype:
        raise TypeError("x dtype must match sparse matrix dtype")
    if x.numel() != prepared.n_cols:
        raise ValueError(f"x length must be n_cols={prepared.n_cols}, got {x.numel()}")
    if x.device != prepared.data.device:
        raise ValueError("x must be on the same device as sparse matrix data")
    return x.contiguous()


def _triton_spmv_coo_kernel(prepared, x, block_size, num_warps, block_inner):
    dtype = prepared.data.dtype
    y = torch.zeros(prepared.n_rows, dtype=dtype, device=prepared.data.device)
    nnz = prepared.nnz
    if nnz == 0:
        return y
    if _is_complex_dtype(dtype):
        data_ri = torch.view_as_real(prepared.data).reshape(-1)
        x_ri = torch.view_as_real(x).reshape(-1)
        y_ri = torch.zeros(prepared.n_rows * 2, dtype=data_ri.dtype, device=y.device)
        acc_dtype = tl.float64 if dtype == torch.complex128 else tl.float32
        if prepared.use_seg_kernel:
            grid = (prepared.n_segs,)
            _spmv_coo_seg_complex[grid](
                data_ri,
                prepared.col,
                prepared.row,
                x_ri,
                y_ri,
                prepared.seg_starts,
                prepared.n_segs,
                BLOCK_INNER=block_inner,
                ACC_DTYPE=acc_dtype,
                num_warps=1,
            )
        else:
            grid = (triton.cdiv(nnz, block_size),)
            _spmv_coo_atomic_complex[grid](
                data_ri,
                prepared.row,
                prepared.col,
                x_ri,
                y_ri,
                nnz,
                BLOCK=block_size,
                ACC_DTYPE=acc_dtype,
                num_warps=num_warps,
            )
        y.copy_(torch.view_as_complex(y_ri.reshape(prepared.n_rows, 2)))
        return y
    if prepared.use_seg_kernel:
        ker = _spmv_coo_seg_f64 if dtype == torch.float64 else _spmv_coo_seg_f32
        grid = (prepared.n_segs,)
        ker[grid](
            prepared.data,
            prepared.col,
            prepared.row,
            x,
            y,
            prepared.seg_starts,
            prepared.n_segs,
            BLOCK_INNER=block_inner,
            num_warps=1,
        )
        return y
    ker = _spmv_coo_atomic_f64 if dtype == torch.float64 else _spmv_coo_atomic_f32
    grid = (triton.cdiv(nnz, block_size),)
    ker[grid](
        prepared.data,
        prepared.row,
        prepared.col,
        x,
        y,
        nnz,
        BLOCK=block_size,
        num_warps=num_warps,
    )
    return y


def _spmv_coo_uses_int64_indices(prepared):
    return (
        prepared.row.dtype == torch.int64
        or prepared.col.dtype == torch.int64
        or (
            prepared.seg_starts is not None and prepared.seg_starts.dtype == torch.int64
        )
    )


def _spmv_coo_int32_fallback_blocker(prepared):
    for name, tensor in (("row", prepared.row), ("col", prepared.col)):
        if tensor.numel() > 0:
            min_index = int(tensor.min().item())
            max_index = int(tensor.max().item())
            if min_index < 0 or max_index > _INDEX_LIMIT_INT32:
                return (
                    f"{name} index range [{min_index}, {max_index}] cannot fit int32 "
                    f"for shape={prepared.shape}"
                )
    if prepared.seg_starts is not None and prepared.seg_starts.numel() > 0:
        max_offset = int(prepared.seg_starts[-1].item())
        if max_offset > _INDEX_LIMIT_INT32:
            return f"COO nnz offset {max_offset} cannot fit int32 for shape={prepared.shape}"
    if prepared.n_rows > _INDEX_LIMIT_INT32 or prepared.n_cols > _INDEX_LIMIT_INT32:
        return f"shape {prepared.shape} cannot fit int32 row/col metadata"
    return None


def _spmv_coo_prepared_with_int32_indices(prepared, reason):
    blocker = _spmv_coo_int32_fallback_blocker(prepared)
    if blocker is not None:
        raise RuntimeError(
            f"native int64 COO SpMV failed and int32 fallback is unsafe: {blocker}"
        )
    seg_starts = (
        None
        if prepared.seg_starts is None
        else prepared.seg_starts.to(torch.int32).contiguous()
    )
    return _PreparedCooLaunch(
        data=prepared.data,
        row=prepared.row.to(torch.int32).contiguous(),
        col=prepared.col.to(torch.int32).contiguous(),
        shape=prepared.shape,
        seg_starts=seg_starts,
        op=prepared.op,
        index_fallback_policy=prepared.index_fallback_policy,
        index_fallback_applied=True,
        index_fallback_reason=str(reason),
    )


def _run_spmv_coo_prepared_with_fallback(
    prepared, x, block_size, num_warps, block_inner
):
    try:
        return _triton_spmv_coo_kernel(
            prepared,
            x,
            block_size=block_size,
            num_warps=num_warps,
            block_inner=block_inner,
        )
    except Exception as exc:
        if (
            prepared.index_fallback_policy != "auto"
            or not _spmv_coo_uses_int64_indices(prepared)
        ):
            raise
        fallback_prepared = _spmv_coo_prepared_with_int32_indices(prepared, exc)
        return _triton_spmv_coo_kernel(
            fallback_prepared,
            x,
            block_size=block_size,
            num_warps=num_warps,
            block_inner=block_inner,
        )


def flagsparse_spmv_coo(
    data=None,
    row=None,
    col=None,
    x=None,
    shape=None,
    out=None,
    return_time=False,
    prepared=None,
    sort_by_row=True,
    block_size=256,
    num_warps=4,
    block_inner=128,
    transpose=None,
    op=None,
    index_fallback_policy="auto",
):
    """COO SpMV with no CSR indptr. See module docstring.

    ``block_inner``: tile for the row-run kernel (``sort_by_row=True``).
    ``block_size`` / ``num_warps``: grid over NNZ when ``sort_by_row=False`` (atomics).
    """
    transpose_flag = False if transpose is None else bool(transpose)
    op_explicit = op is not None
    op_code = _normalize_spmv_coo_op(op, transpose=transpose_flag)
    if (
        op_explicit
        and transpose is not None
        and bool(transpose) != _spmv_coo_op_transposes(op_code)
    ):
        raise ValueError("transpose conflicts with op")
    if prepared is None:
        if any(a is None for a in (data, row, col, x, shape)):
            raise ValueError("data, row, col, x, shape required when prepared is None")
        prepared = prepare_spmv_coo(
            data,
            row,
            col,
            shape,
            sort_by_row=sort_by_row,
            transpose=transpose_flag,
            op=op,
            index_fallback_policy=index_fallback_policy,
        )
    else:
        if x is None:
            raise TypeError("x is required when prepared is set")
        if shape is None:
            shape = prepared.shape
        sh = (int(shape[0]), int(shape[1]))
        if sh != prepared.shape:
            raise ValueError(
                f"shape {sh} does not match prepared.shape {prepared.shape}"
            )
        if not op_explicit and transpose is None:
            op_code = prepared.op
        elif not op_explicit and transpose is not None:
            op_code = _normalize_spmv_coo_op(None, transpose=transpose_flag)
    launch = _resolve_spmv_coo_launch(prepared, op_code)
    x = _validate_x_coo(x, launch)
    if num_warps not in (1, 2, 4, 8, 16, 32):
        raise ValueError("num_warps must be a power of 2 in [1, 32]")
    if block_inner <= 0 or (block_inner & (block_inner - 1)) != 0:
        raise ValueError("block_inner must be a positive power of 2")
    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    y = _run_spmv_coo_prepared_with_fallback(
        launch,
        x,
        block_size=block_size,
        num_warps=num_warps,
        block_inner=block_inner,
    )
    elapsed_ms = None
    if return_time:
        _ACCEL.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != y.shape or out.dtype != y.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(y)
        y = out
    if return_time:
        return y, elapsed_ms
    return y
