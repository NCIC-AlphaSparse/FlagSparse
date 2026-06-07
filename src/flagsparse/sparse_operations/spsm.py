"""Sparse triangular matrix-matrix solve (SpSM) for CSR/COO."""

import os
from collections import OrderedDict

from ._common import *


SUPPORTED_SPSM_VALUE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
)
SUPPORTED_SPSM_INDEX_DTYPES = (torch.int32,)
SPSM_NON_TRANS_PRIMARY_COMBOS = (
    ("csr", torch.float32, torch.int32),
    ("csr", torch.float64, torch.int32),
    ("csr", torch.complex64, torch.int32),
    ("csr", torch.complex128, torch.int32),
    ("coo", torch.float32, torch.int32),
    ("coo", torch.float64, torch.int32),
    ("coo", torch.complex64, torch.int32),
    ("coo", torch.complex128, torch.int32),
)
_SPSM_PREPROCESS_CACHE = OrderedDict()
_SPSM_PREPROCESS_CACHE_SIZE = 8


def _use_spsm_polling_path():
    token = os.environ.get("FLAGSPARSE_SPSM_USE_POLLING", "")
    return token.strip().lower() in ("1", "true", "yes", "on", "polling")


def _clear_spsm_preprocess_cache():
    _SPSM_PREPROCESS_CACHE.clear()


def _is_non_transpose(op):
    op_str = str(op).upper()
    return op_str in ("NON_TRANS", "NON_TRANSPOSE", "N")


def _validate_spsm_non_trans_combo(fmt_name, value_dtype, index_dtype):
    combo = (str(fmt_name).lower(), value_dtype, index_dtype)
    if combo not in SPSM_NON_TRANS_PRIMARY_COMBOS:
        raise TypeError(
            f"{fmt_name} SpSM currently supports NON_TRANS combinations: "
            "CSR/COO with float32/float64/complex64/complex128 and int32"
        )


def _validate_spsm_op_and_layout(opA, opB, major):
    if not _is_non_transpose(opA):
        raise NotImplementedError("Only op(A)=NON_TRANS is supported")
    if not _is_non_transpose(opB):
        raise NotImplementedError("Only op(B)=NON_TRANS is supported")
    if str(major).lower() != "row":
        raise NotImplementedError("Only row-major dense layout is supported")


def _prepare_spsm_csr_inputs(data, indices, indptr, B, shape, opA, opB, major):
    if not all(torch.is_tensor(t) for t in (data, indices, indptr, B)):
        raise TypeError("data, indices, indptr, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, indices, indptr, B)):
        raise ValueError("data, indices, indptr, B must all be CUDA tensors")
    if data.ndim != 1 or indices.ndim != 1 or indptr.ndim != 1:
        raise ValueError("data, indices, indptr must be 1D")
    if B.ndim != 2:
        raise ValueError("B must be a 2D dense matrix")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSM requires square A, got shape={shape}")
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError("data and indices must have the same length (nnz)")
    if B.shape[0] != n_rows:
        raise ValueError(f"B.shape[0] must equal n_rows={n_rows}")
    _validate_spsm_op_and_layout(opA, opB, major)
    if data.dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        raise TypeError("data dtype must be float32, float64, complex64, or complex128")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if indices.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("indices dtype must be torch.int32")
    if indptr.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("indptr dtype must be torch.int32")
    if indices.dtype != indptr.dtype:
        raise TypeError("indices and indptr dtype must match for SpSM CSR")
    input_index_dtype = indices.dtype
    _validate_spsm_non_trans_combo("csr", data.dtype, input_index_dtype)
    return data.contiguous(), indices.contiguous(), indptr.contiguous(), B.contiguous(), n_rows, n_cols


def _prepare_spsm_coo_inputs(data, row, col, B, shape, opA, opB, major):
    if not all(torch.is_tensor(t) for t in (data, row, col, B)):
        raise TypeError("data, row, col, B must all be torch.Tensor")
    if not all(t.is_cuda for t in (data, row, col, B)):
        raise ValueError("data, row, col, B must all be CUDA tensors")
    if data.ndim != 1 or row.ndim != 1 or col.ndim != 1:
        raise ValueError("data, row, col must be 1D")
    if data.numel() != row.numel() or data.numel() != col.numel():
        raise ValueError("data, row, col must have same length")
    if int(data.numel()) > _INDEX_LIMIT_INT32:
        raise ValueError("nnz exceeds int32 kernel range")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if n_rows != n_cols:
        raise ValueError(f"SpSM requires square A, got shape={shape}")
    if B.ndim != 2 or B.shape[0] != n_rows:
        raise ValueError("B must be 2D and B.shape[0] == n_rows")
    _validate_spsm_op_and_layout(opA, opB, major)
    if data.dtype not in SUPPORTED_SPSM_VALUE_DTYPES:
        raise TypeError("data dtype must be float32, float64, complex64, or complex128")
    if B.dtype != data.dtype:
        raise TypeError("B dtype must match data dtype")
    if row.dtype not in SUPPORTED_SPSM_INDEX_DTYPES or col.dtype not in SUPPORTED_SPSM_INDEX_DTYPES:
        raise TypeError("row/col dtype must be torch.int32")
    if row.dtype != col.dtype:
        raise TypeError("row and col dtype must match for SpSM COO")
    _validate_spsm_non_trans_combo("coo", data.dtype, row.dtype)
    return data.contiguous(), row.contiguous(), col.contiguous(), B.contiguous(), n_rows, n_cols


def _complex_interleaved_view(tensor):
    return torch.view_as_real(tensor.contiguous()).reshape(-1).contiguous()


def _tensor_cache_token(tensor):
    try:
        storage_ptr = int(tensor.untyped_storage().data_ptr())
    except Exception:
        storage_ptr = 0
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(v) for v in tensor.shape),
        int(tensor.numel()),
        storage_ptr,
        int(getattr(tensor, "_version", 0)),
    )


def _spsm_cache_get(cache, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _spsm_cache_put(cache, key, value, max_entries):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_entries:
        cache.popitem(last=False)


def _spsm_preprocess_cache_key(fmt_name, tensors, shape, lower, unit_diagonal):
    return (
        str(fmt_name).lower(),
        bool(lower),
        bool(unit_diagonal),
        int(shape[0]),
        int(shape[1]),
        *(_tensor_cache_token(t) for t in tensors),
    )


def _prepare_spsm_diag(data, indices64, indptr64, n_rows, unit_diagonal=False, row_ids=None):
    diag = torch.ones(n_rows, dtype=data.dtype, device=data.device)
    if unit_diagonal or n_rows == 0 or data.numel() == 0:
        return diag
    if row_ids is None:
        row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    diag_mask = indices64 == row_ids
    if bool(torch.any(diag_mask).item()):
        diag.scatter_(0, row_ids[diag_mask], data[diag_mask])
    return diag


def _prepare_spsm_inv_diag(diag):
    if diag.numel() == 0:
        return diag
    eps = 1e-12 if diag.dtype == torch.float64 else 1e-6
    safe_diag = torch.where(torch.abs(diag) < eps, torch.ones_like(diag), diag)
    return torch.reciprocal(safe_diag)


def _prepare_spsm_kernel_row_ptr(indptr64):
    if indptr64.numel() == 0:
        return indptr64.to(torch.int32)
    if int(indptr64[-1].item()) <= _INDEX_LIMIT_INT32:
        return indptr64.to(torch.int32)
    return indptr64


def _csr_row_ids_from_indptr(indptr64, n_rows):
    if n_rows == 0 or indptr64.numel() <= 1:
        return torch.empty(0, dtype=torch.int64, device=indptr64.device)
    return torch.repeat_interleave(
        torch.arange(n_rows, device=indptr64.device, dtype=torch.int64),
        indptr64[1:] - indptr64[:-1],
    )


def _csr_rows_sorted_by_col(indices64, indptr64, n_rows):
    if indices64.numel() <= 1:
        return True
    row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    if row_ids.numel() <= 1:
        return True
    same_row = row_ids[1:] == row_ids[:-1]
    if not bool(torch.any(same_row).item()):
        return True
    return bool(torch.all(indices64[1:][same_row] >= indices64[:-1][same_row]).item())


def _prepare_spsm_csr_sorted_view(data, indices64, indptr64, n_rows, n_cols):
    row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    if data.numel() <= 1:
        return data, indices64, row_ids
    if _csr_rows_sorted_by_col(indices64, indptr64, n_rows):
        return data, indices64, row_ids
    key = row_ids * max(1, n_cols) + indices64
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    return data[order].contiguous(), indices64[order].contiguous(), row_ids[order].contiguous()


def _prepare_spsm_csr_dependency_bounds(indices64, indptr64, n_rows, lower, row_ids=None):
    row_begin = indptr64[:-1].clone()
    row_end = indptr64[1:].clone()
    if n_rows == 0 or indices64.numel() == 0:
        return row_begin, row_begin if lower else row_end
    if row_ids is None:
        row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    dep_mask = indices64 < row_ids if lower else indices64 > row_ids
    dep_counts = torch.bincount(row_ids[dep_mask], minlength=n_rows)
    if lower:
        dep_begin = row_begin
        dep_end = row_begin + dep_counts
    else:
        dep_begin = row_end - dep_counts
        dep_end = row_end
    return dep_begin, dep_end


def _prepare_spsm_csr_dependency_view(
    data, indices64, indptr64, n_rows, lower, row_ids=None
):
    dep_begin64, dep_end64 = _prepare_spsm_csr_dependency_bounds(
        indices64, indptr64, n_rows, lower=lower, row_ids=row_ids
    )
    dep_counts = dep_end64 - dep_begin64
    dep_ptr64 = torch.zeros(n_rows + 1, dtype=torch.int64, device=indptr64.device)
    if n_rows > 0:
        dep_ptr64[1:] = torch.cumsum(dep_counts, dim=0)
    total_dep_nnz = int(dep_ptr64[-1].item()) if dep_ptr64.numel() > 0 else 0
    if total_dep_nnz == 0:
        empty_data = torch.empty(0, dtype=data.dtype, device=data.device)
        empty_indices = torch.empty(0, dtype=torch.int64, device=indices64.device)
        return empty_data, empty_indices, dep_ptr64

    if row_ids is None:
        row_ids = _csr_row_ids_from_indptr(indptr64, n_rows)
    dep_mask = indices64 < row_ids if lower else indices64 > row_ids
    dep_data = data[dep_mask].contiguous()
    dep_indices64 = indices64[dep_mask].contiguous()
    return dep_data, dep_indices64, dep_ptr64


def _alpha_is_one(alpha):
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            return False
        return bool((alpha.detach().cpu() == 1).item())
    return alpha == 1 or alpha == 1.0


def _alpha_to_host_scalar(alpha):
    if torch.is_tensor(alpha):
        if alpha.numel() != 1:
            raise ValueError("alpha tensor must be scalar")
        value = alpha.detach().cpu().item()
        if isinstance(value, complex):
            return value
        return float(value)
    if isinstance(alpha, complex):
        return alpha
    return float(alpha)


@triton.jit
def _spsm_extract_diag_kernel_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ptr,
    n_rows,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    row = tl.program_id(0)
    if USE_FP64_ACC:
        diag = tl.full((), 1.0, tl.float64)
    else:
        diag = tl.full((), 1.0, tl.float32)
    if not UNIT_DIAG:
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        p = start
        while p < end:
            col = tl.load(indices_ptr + p)
            if col == row:
                val = tl.load(data_ptr + p)
                if USE_FP64_ACC:
                    diag = val.to(tl.float64)
                else:
                    diag = val.to(tl.float32)
            p += 1
    tl.store(diag_ptr + row, diag)


@triton.jit
def _spsm_extract_diag_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ri_ptr,
    n_rows,
    UNIT_DIAG: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    row = tl.program_id(0)
    if USE_FP64_ACC:
        diag_re = tl.full((), 1.0, tl.float64)
        diag_im = tl.full((), 0.0, tl.float64)
    else:
        diag_re = tl.full((), 1.0, tl.float32)
        diag_im = tl.full((), 0.0, tl.float32)
    if not UNIT_DIAG:
        start = tl.load(indptr_ptr + row)
        end = tl.load(indptr_ptr + row + 1)
        p = start
        while p < end:
            col = tl.load(indices_ptr + p)
            if col == row:
                diag_re = tl.load(data_ri_ptr + p * 2)
                diag_im = tl.load(data_ri_ptr + p * 2 + 1)
                if USE_FP64_ACC:
                    diag_re = diag_re.to(tl.float64)
                    diag_im = diag_im.to(tl.float64)
                else:
                    diag_re = diag_re.to(tl.float32)
                    diag_im = diag_im.to(tl.float32)
            p += 1
    tl.store(diag_ri_ptr + row * 2, diag_re)
    tl.store(diag_ri_ptr + row * 2 + 1, diag_im)


@triton.jit
def _spsm_csr_diag_only_kernel_real(
    inv_diag_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    alpha,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs

    b_ptrs = b_ptr + row_i64 * stride_b0 + rhs_offsets_i64
    rhs = tl.load(b_ptrs, mask=rhs_mask, other=0.0)
    if USE_FP64_ACC:
        rhs = rhs.to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
    else:
        rhs = rhs.to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)

    inv_diag = tl.load(inv_diag_ptr + row)
    inv_diag = inv_diag.to(tl.float64) if USE_FP64_ACC else inv_diag.to(tl.float32)
    out = rhs * alpha_val * inv_diag
    out = tl.where(out == out, out, 0.0)

    out_ptrs = x_ptr + row_i64 * stride_x0 + rhs_offsets_i64
    tl.store(out_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_csr_level_kernel_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    inv_diag_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    alpha,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    if USE_FP64_ACC:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
    else:
        acc = tl.zeros((BLOCK_RHS,), dtype=tl.float32)

    for seg in range(MAX_SEGMENTS):
        nnz_offsets = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        nnz_mask = nnz_offsets < end
        a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
        col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)

        if USE_FP64_ACC:
            a = a.to(tl.float64)
        else:
            a = a.to(tl.float32)

        col_i64 = col.to(tl.int64)
        x_ptrs = x_ptr + col_i64[:, None] * stride_x0 + rhs_offsets_i64[None, :]
        x_mask = nnz_mask[:, None] & rhs_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        if USE_FP64_ACC:
            x_vals = x_vals.to(tl.float64)
        else:
            x_vals = x_vals.to(tl.float32)

        contrib = tl.where(nnz_mask[:, None], a[:, None] * x_vals, 0.0)
        acc += tl.sum(contrib, axis=0)

    b_ptrs = b_ptr + row_i64 * stride_b0 + rhs_offsets_i64
    rhs = tl.load(b_ptrs, mask=rhs_mask, other=0.0)
    if USE_FP64_ACC:
        rhs = rhs.to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
    else:
        rhs = rhs.to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)

    inv_diag = tl.load(inv_diag_ptr + row)
    inv_diag = inv_diag.to(tl.float64) if USE_FP64_ACC else inv_diag.to(tl.float32)
    out = (rhs * alpha_val - acc) * inv_diag
    out = tl.where(out == out, out, 0.0)

    out_ptrs = x_ptr + row_i64 * stride_x0 + rhs_offsets_i64
    tl.store(out_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_csr_level_kernel_single_segment_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    inv_diag_ptr,
    b_ptr,
    x_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    alpha,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    nnz_offsets = start + tl.arange(0, BLOCK_NNZ)
    nnz_mask = nnz_offsets < end

    a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
    col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)
    if USE_FP64_ACC:
        a = a.to(tl.float64)
    else:
        a = a.to(tl.float32)

    col_i64 = col.to(tl.int64)
    x_ptrs = x_ptr + col_i64[:, None] * stride_x0 + rhs_offsets_i64[None, :]
    x_mask = nnz_mask[:, None] & rhs_mask[None, :]
    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
    if USE_FP64_ACC:
        x_vals = x_vals.to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
    else:
        x_vals = x_vals.to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)
    contrib = tl.where(nnz_mask[:, None], a[:, None] * x_vals, 0.0)
    acc = tl.sum(contrib, axis=0)

    b_ptrs = b_ptr + row_i64 * stride_b0 + rhs_offsets_i64
    rhs = tl.load(b_ptrs, mask=rhs_mask, other=0.0)
    rhs = rhs.to(tl.float64) if USE_FP64_ACC else rhs.to(tl.float32)

    inv_diag = tl.load(inv_diag_ptr + row)
    inv_diag = inv_diag.to(tl.float64) if USE_FP64_ACC else inv_diag.to(tl.float32)
    out = (rhs * alpha_val - acc) * inv_diag
    out = tl.where(out == out, out, 0.0)

    out_ptrs = x_ptr + row_i64 * stride_x0 + rhs_offsets_i64
    tl.store(out_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_csr_level_kernel_staged_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    inv_diag_ptr,
    work_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_work0,
    alpha,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    RHS_TILE_GROUPS: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs_group = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    row_i64 = row.to(tl.int64)
    rhs_group_base = pid_rhs_group * BLOCK_RHS * RHS_TILE_GROUPS
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    if USE_FP64_ACC:
        inv_diag = tl.load(inv_diag_ptr + row).to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
        acc0 = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
        acc1 = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
        acc2 = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
        acc3 = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
    else:
        inv_diag = tl.load(inv_diag_ptr + row).to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)
        acc0 = tl.zeros((BLOCK_RHS,), dtype=tl.float32)
        acc1 = tl.zeros((BLOCK_RHS,), dtype=tl.float32)
        acc2 = tl.zeros((BLOCK_RHS,), dtype=tl.float32)
        acc3 = tl.zeros((BLOCK_RHS,), dtype=tl.float32)

    for seg in range(MAX_SEGMENTS):
        nnz_offsets = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        nnz_mask = nnz_offsets < end
        a = tl.load(data_ptr + nnz_offsets, mask=nnz_mask, other=0.0)
        col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)
        if USE_FP64_ACC:
            a = a.to(tl.float64)
        else:
            a = a.to(tl.float32)

        for tile_id in range(RHS_TILE_GROUPS):
            rhs_offsets = rhs_group_base + tile_id * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
            rhs_offsets_i64 = rhs_offsets.to(tl.int64)
            rhs_mask = rhs_offsets < n_rhs
            col_i64 = col.to(tl.int64)
            x_ptrs = work_ptr + col_i64[:, None] * stride_work0 + rhs_offsets_i64[None, :]
            x_mask = nnz_mask[:, None] & rhs_mask[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
            if USE_FP64_ACC:
                x_vals = x_vals.to(tl.float64)
            else:
                x_vals = x_vals.to(tl.float32)
            contrib = tl.where(nnz_mask[:, None], a[:, None] * x_vals, 0.0)
            tile_acc = tl.sum(contrib, axis=0)
            if tile_id == 0:
                acc0 += tile_acc
            elif tile_id == 1:
                acc1 += tile_acc
            elif tile_id == 2:
                acc2 += tile_acc
            else:
                acc3 += tile_acc

    for tile_id in range(RHS_TILE_GROUPS):
        rhs_offsets = rhs_group_base + tile_id * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
        rhs_offsets_i64 = rhs_offsets.to(tl.int64)
        rhs_mask = rhs_offsets < n_rhs
        work_ptrs = work_ptr + row_i64 * stride_work0 + rhs_offsets_i64
        rhs = tl.load(work_ptrs, mask=rhs_mask, other=0.0)
        rhs = rhs.to(tl.float64) if USE_FP64_ACC else rhs.to(tl.float32)
        if tile_id == 0:
            acc_tile = acc0
        elif tile_id == 1:
            acc_tile = acc1
        elif tile_id == 2:
            acc_tile = acc2
        else:
            acc_tile = acc3
        out = (rhs * alpha_val - acc_tile) * inv_diag
        out = tl.where(out == out, out, 0.0)
        tl.store(work_ptrs, out, mask=rhs_mask)


@triton.jit
def _spsm_csr_diag_only_kernel_complex(
    inv_diag_ri_ptr,
    b_ri_ptr,
    x_ri_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    alpha_re,
    alpha_im,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs
    b_base = (row_i64 * stride_b0 + rhs_offsets_i64) * 2

    rhs_re = tl.load(b_ri_ptr + b_base, mask=rhs_mask, other=0.0)
    rhs_im = tl.load(b_ri_ptr + b_base + 1, mask=rhs_mask, other=0.0)
    inv_re = tl.load(inv_diag_ri_ptr + row * 2)
    inv_im = tl.load(inv_diag_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        inv_re = inv_re.to(tl.float64)
        inv_im = inv_im.to(tl.float64)
        alpha_re_v = tl.full((BLOCK_RHS,), alpha_re, tl.float64)
        alpha_im_v = tl.full((BLOCK_RHS,), alpha_im, tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        inv_re = inv_re.to(tl.float32)
        inv_im = inv_im.to(tl.float32)
        alpha_re_v = tl.full((BLOCK_RHS,), alpha_re, tl.float32)
        alpha_im_v = tl.full((BLOCK_RHS,), alpha_im, tl.float32)

    scaled_re = rhs_re * alpha_re_v - rhs_im * alpha_im_v
    scaled_im = rhs_re * alpha_im_v + rhs_im * alpha_re_v
    out_re = scaled_re * inv_re - scaled_im * inv_im
    out_im = scaled_re * inv_im + scaled_im * inv_re
    out_re = tl.where(out_re == out_re, out_re, 0.0)
    out_im = tl.where(out_im == out_im, out_im, 0.0)

    x_base = (row_i64 * stride_x0 + rhs_offsets_i64) * 2
    tl.store(x_ri_ptr + x_base, out_re, mask=rhs_mask)
    tl.store(x_ri_ptr + x_base + 1, out_im, mask=rhs_mask)


@triton.jit
def _spsm_csr_level_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    inv_diag_ri_ptr,
    b_ri_ptr,
    x_ri_ptr,
    rows_ptr,
    n_level_rows,
    n_rhs,
    stride_b0,
    stride_x0,
    alpha_re,
    alpha_im,
    BLOCK_NNZ: tl.constexpr,
    BLOCK_RHS: tl.constexpr,
    MAX_SEGMENTS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if pid_row >= n_level_rows:
        return

    row = tl.load(rows_ptr + pid_row)
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs
    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)

    if USE_FP64_ACC:
        acc_re = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
        acc_im = tl.zeros((BLOCK_RHS,), dtype=tl.float64)
    else:
        acc_re = tl.zeros((BLOCK_RHS,), dtype=tl.float32)
        acc_im = tl.zeros((BLOCK_RHS,), dtype=tl.float32)

    for seg in range(MAX_SEGMENTS):
        nnz_offsets = start + seg * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        nnz_mask = nnz_offsets < end
        col = tl.load(indices_ptr + nnz_offsets, mask=nnz_mask, other=0)
        a_re = tl.load(data_ri_ptr + nnz_offsets * 2, mask=nnz_mask, other=0.0)
        a_im = tl.load(data_ri_ptr + nnz_offsets * 2 + 1, mask=nnz_mask, other=0.0)

        col_i64 = col.to(tl.int64)
        x_base = (col_i64[:, None] * stride_x0 + rhs_offsets_i64[None, :]) * 2
        x_mask = nnz_mask[:, None] & rhs_mask[None, :]
        x_re = tl.load(x_ri_ptr + x_base, mask=x_mask, other=0.0)
        x_im = tl.load(x_ri_ptr + x_base + 1, mask=x_mask, other=0.0)

        if USE_FP64_ACC:
            a_re = a_re.to(tl.float64)
            a_im = a_im.to(tl.float64)
            x_re = x_re.to(tl.float64)
            x_im = x_im.to(tl.float64)
        else:
            a_re = a_re.to(tl.float32)
            a_im = a_im.to(tl.float32)
            x_re = x_re.to(tl.float32)
            x_im = x_im.to(tl.float32)

        prod_re = a_re[:, None] * x_re - a_im[:, None] * x_im
        prod_im = a_re[:, None] * x_im + a_im[:, None] * x_re
        acc_re += tl.sum(tl.where(x_mask, prod_re, 0.0), axis=0)
        acc_im += tl.sum(tl.where(x_mask, prod_im, 0.0), axis=0)

    b_base = (row_i64 * stride_b0 + rhs_offsets_i64) * 2
    rhs_re = tl.load(b_ri_ptr + b_base, mask=rhs_mask, other=0.0)
    rhs_im = tl.load(b_ri_ptr + b_base + 1, mask=rhs_mask, other=0.0)
    inv_re = tl.load(inv_diag_ri_ptr + row * 2)
    inv_im = tl.load(inv_diag_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        inv_re = inv_re.to(tl.float64)
        inv_im = inv_im.to(tl.float64)
        alpha_re_v = tl.full((BLOCK_RHS,), alpha_re, tl.float64)
        alpha_im_v = tl.full((BLOCK_RHS,), alpha_im, tl.float64)
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        inv_re = inv_re.to(tl.float32)
        inv_im = inv_im.to(tl.float32)
        alpha_re_v = tl.full((BLOCK_RHS,), alpha_re, tl.float32)
        alpha_im_v = tl.full((BLOCK_RHS,), alpha_im, tl.float32)

    scaled_re = rhs_re * alpha_re_v - rhs_im * alpha_im_v
    scaled_im = rhs_re * alpha_im_v + rhs_im * alpha_re_v
    num_re = scaled_re - acc_re
    num_im = scaled_im - acc_im
    out_re = num_re * inv_re - num_im * inv_im
    out_im = num_re * inv_im + num_im * inv_re
    out_re = tl.where(out_re == out_re, out_re, 0.0)
    out_im = tl.where(out_im == out_im, out_im, 0.0)

    x_out_base = (row_i64 * stride_x0 + rhs_offsets_i64) * 2
    tl.store(x_ri_ptr + x_out_base, out_re, mask=rhs_mask)
    tl.store(x_ri_ptr + x_out_base + 1, out_im, mask=rhs_mask)


@triton.jit
def _spsm_csr_polling_kernel_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ptr,
    work_ptr,
    done_ptr,
    n_rows,
    n_rhs,
    stride_work0,
    alpha,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if LOWER:
        row = pid_row
    else:
        row = n_rows - 1 - pid_row
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs

    if USE_FP64_ACC:
        rhs = tl.load(work_ptr + row_i64 * stride_work0 + rhs_offsets_i64, mask=rhs_mask, other=0.0).to(tl.float64)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float64)
        local_sum = rhs * alpha_val
    else:
        rhs = tl.load(work_ptr + row_i64 * stride_work0 + rhs_offsets_i64, mask=rhs_mask, other=0.0).to(tl.float32)
        alpha_val = tl.full((BLOCK_RHS,), alpha, tl.float32)
        local_sum = rhs * alpha_val

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    flag_base = pid_rhs * n_rows

    if LOWER:
        p = start
        while p < end:
            col = tl.load(indices_ptr + p)
            if col < row:
                val = tl.load(data_ptr + p)
                if USE_FP64_ACC:
                    val = val.to(tl.float64)
                else:
                    val = val.to(tl.float32)
                dep_flag_ptr = done_ptr + flag_base + col
                ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                while ready == 0:
                    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                x_vals = tl.load(
                    work_ptr + col.to(tl.int64) * stride_work0 + rhs_offsets_i64,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                if USE_FP64_ACC:
                    x_vals = x_vals.to(tl.float64)
                else:
                    x_vals = x_vals.to(tl.float32)
                local_sum -= val * x_vals
            p += 1
    else:
        p = end - 1
        while p >= start:
            col = tl.load(indices_ptr + p)
            if col > row:
                val = tl.load(data_ptr + p)
                if USE_FP64_ACC:
                    val = val.to(tl.float64)
                else:
                    val = val.to(tl.float32)
                dep_flag_ptr = done_ptr + flag_base + col
                ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                while ready == 0:
                    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                x_vals = tl.load(
                    work_ptr + col.to(tl.int64) * stride_work0 + rhs_offsets_i64,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                if USE_FP64_ACC:
                    x_vals = x_vals.to(tl.float64)
                else:
                    x_vals = x_vals.to(tl.float32)
                local_sum -= val * x_vals
            p -= 1

    diag = tl.load(diag_ptr + row)
    if USE_FP64_ACC:
        diag = diag.to(tl.float64)
    else:
        diag = diag.to(tl.float32)
    if UNIT_DIAG:
        out = local_sum
    else:
        out = local_sum / diag
    out = tl.where(out == out, out, 0.0)
    tl.store(
        work_ptr + row_i64 * stride_work0 + rhs_offsets_i64,
        out,
        mask=rhs_mask,
        cache_modifier=".wt",
    )
    tl.atomic_or(done_ptr + flag_base + row, 1, sem="release")


@triton.jit
def _spsm_csr_polling_kernel_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    diag_ri_ptr,
    work_ri_ptr,
    done_ptr,
    n_rows,
    n_rhs,
    stride_work0,
    alpha_re,
    alpha_im,
    BLOCK_RHS: tl.constexpr,
    USE_FP64_ACC: tl.constexpr,
    LOWER: tl.constexpr,
    UNIT_DIAG: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_rhs = tl.program_id(1)
    if LOWER:
        row = pid_row
    else:
        row = n_rows - 1 - pid_row
    rhs_offsets = pid_rhs * BLOCK_RHS + tl.arange(0, BLOCK_RHS)
    row_i64 = row.to(tl.int64)
    rhs_offsets_i64 = rhs_offsets.to(tl.int64)
    rhs_mask = rhs_offsets < n_rhs
    base = (row_i64 * stride_work0 + rhs_offsets_i64) * 2

    rhs_re = tl.load(work_ri_ptr + base, mask=rhs_mask, other=0.0)
    rhs_im = tl.load(work_ri_ptr + base + 1, mask=rhs_mask, other=0.0)
    if USE_FP64_ACC:
        rhs_re = rhs_re.to(tl.float64)
        rhs_im = rhs_im.to(tl.float64)
        alpha_re_v = tl.full((BLOCK_RHS,), alpha_re, tl.float64)
        alpha_im_v = tl.full((BLOCK_RHS,), alpha_im, tl.float64)
        sum_re = rhs_re * alpha_re_v - rhs_im * alpha_im_v
        sum_im = rhs_re * alpha_im_v + rhs_im * alpha_re_v
    else:
        rhs_re = rhs_re.to(tl.float32)
        rhs_im = rhs_im.to(tl.float32)
        alpha_re_v = tl.full((BLOCK_RHS,), alpha_re, tl.float32)
        alpha_im_v = tl.full((BLOCK_RHS,), alpha_im, tl.float32)
        sum_re = rhs_re * alpha_re_v - rhs_im * alpha_im_v
        sum_im = rhs_re * alpha_im_v + rhs_im * alpha_re_v

    start = tl.load(indptr_ptr + row)
    end = tl.load(indptr_ptr + row + 1)
    flag_base = pid_rhs * n_rows

    if LOWER:
        p = start
        while p < end:
            col = tl.load(indices_ptr + p)
            if col < row:
                val_re = tl.load(data_ri_ptr + p * 2)
                val_im = tl.load(data_ri_ptr + p * 2 + 1)
                if USE_FP64_ACC:
                    val_re = val_re.to(tl.float64)
                    val_im = val_im.to(tl.float64)
                else:
                    val_re = val_re.to(tl.float32)
                    val_im = val_im.to(tl.float32)
                dep_flag_ptr = done_ptr + flag_base + col
                ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                while ready == 0:
                    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                x_base = (col.to(tl.int64) * stride_work0 + rhs_offsets_i64) * 2
                x_re = tl.load(
                    work_ri_ptr + x_base,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                x_im = tl.load(
                    work_ri_ptr + x_base + 1,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                if USE_FP64_ACC:
                    x_re = x_re.to(tl.float64)
                    x_im = x_im.to(tl.float64)
                else:
                    x_re = x_re.to(tl.float32)
                    x_im = x_im.to(tl.float32)
                sum_re -= val_re * x_re - val_im * x_im
                sum_im -= val_re * x_im + val_im * x_re
            p += 1
    else:
        p = end - 1
        while p >= start:
            col = tl.load(indices_ptr + p)
            if col > row:
                val_re = tl.load(data_ri_ptr + p * 2)
                val_im = tl.load(data_ri_ptr + p * 2 + 1)
                if USE_FP64_ACC:
                    val_re = val_re.to(tl.float64)
                    val_im = val_im.to(tl.float64)
                else:
                    val_re = val_re.to(tl.float32)
                    val_im = val_im.to(tl.float32)
                dep_flag_ptr = done_ptr + flag_base + col
                ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                while ready == 0:
                    ready = tl.atomic_or(dep_flag_ptr, 0, sem="acquire")
                x_base = (col.to(tl.int64) * stride_work0 + rhs_offsets_i64) * 2
                x_re = tl.load(
                    work_ri_ptr + x_base,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                x_im = tl.load(
                    work_ri_ptr + x_base + 1,
                    mask=rhs_mask,
                    other=0.0,
                    cache_modifier=".cv",
                )
                if USE_FP64_ACC:
                    x_re = x_re.to(tl.float64)
                    x_im = x_im.to(tl.float64)
                else:
                    x_re = x_re.to(tl.float32)
                    x_im = x_im.to(tl.float32)
                prod_re = val_re * x_re - val_im * x_im
                prod_im = val_re * x_im + val_im * x_re
                sum_re -= prod_re
                sum_im -= prod_im
            p -= 1

    diag_re = tl.load(diag_ri_ptr + row * 2)
    diag_im = tl.load(diag_ri_ptr + row * 2 + 1)
    if USE_FP64_ACC:
        diag_re = diag_re.to(tl.float64)
        diag_im = diag_im.to(tl.float64)
    else:
        diag_re = diag_re.to(tl.float32)
        diag_im = diag_im.to(tl.float32)
    denom = diag_re * diag_re + diag_im * diag_im
    if UNIT_DIAG:
        out_re = sum_re
        out_im = sum_im
    else:
        out_re = (sum_re * diag_re + sum_im * diag_im) / denom
        out_im = (sum_im * diag_re - sum_re * diag_im) / denom
    out_re = tl.where(out_re == out_re, out_re, 0.0)
    out_im = tl.where(out_im == out_im, out_im, 0.0)

    tl.store(work_ri_ptr + base, out_re, mask=rhs_mask, cache_modifier=".wt")
    tl.store(work_ri_ptr + base + 1, out_im, mask=rhs_mask, cache_modifier=".wt")
    tl.atomic_or(done_ptr + flag_base + row, 1, sem="release")


def _build_spsm_levels(indptr, indices, n_rows, lower=True):
    if n_rows == 0:
        return []
    indptr_h = indptr.to(torch.int64).cpu().tolist()
    indices_h = indices.to(torch.int64).cpu().tolist()
    levels = [0] * n_rows

    if lower:
        for i in range(n_rows):
            start = int(indptr_h[i])
            end = int(indptr_h[i + 1])
            level = 0
            for p in range(start, end):
                col = int(indices_h[p])
                if col < i:
                    level = max(level, levels[col] + 1)
            levels[i] = level
    else:
        for i in range(n_rows - 1, -1, -1):
            start = int(indptr_h[i])
            end = int(indptr_h[i + 1])
            level = 0
            for p in range(start, end):
                col = int(indices_h[p])
                if col > i:
                    level = max(level, levels[col] + 1)
            levels[i] = level

    max_level = max(levels)
    buckets = [[] for _ in range(max_level + 1)]
    for row, level in enumerate(levels):
        buckets[level].append(row)

    return [rows for rows in buckets if rows]


def _spsm_block_nnz_for_row_length(max_nnz_per_row):
    max_nnz_per_row = int(max_nnz_per_row)
    if max_nnz_per_row <= 32:
        return 32
    if max_nnz_per_row <= 64:
        return 64
    if max_nnz_per_row <= 128:
        return 128
    if max_nnz_per_row <= 256:
        return 256
    if max_nnz_per_row <= 512:
        return 512
    if max_nnz_per_row <= 1024:
        return 1024
    if max_nnz_per_row <= 4096:
        return 1024
    return 1024


def _spsm_single_segment_block_nnz(limit):
    return _spsm_block_nnz_for_row_length(limit)


def _bucketize_spsm_launch_groups(levels, indptr_like):
    if not levels:
        return []
    device = indptr_like.device
    row_lengths = (indptr_like[1:] - indptr_like[:-1]).to(torch.int64).cpu().tolist()
    bucket_limits = (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024)
    grouped = []
    for rows_lv in levels:
        rows = rows_lv if isinstance(rows_lv, list) else rows_lv.cpu().tolist()
        if not rows:
            continue
        lengths = [int(row_lengths[int(row)]) for row in rows]
        remaining = [True] * len(rows)
        zero_rows = [int(row) for row, length in zip(rows, lengths) if length == 0]
        if zero_rows:
            grouped.append(
                {
                    "rows": torch.tensor(zero_rows, dtype=torch.int32, device=device),
                    "diag_only": True,
                }
            )
            for i, length in enumerate(lengths):
                if length == 0:
                    remaining[i] = False
        for limit in bucket_limits:
            bucket_rows = [
                int(row)
                for row, length, keep in zip(rows, lengths, remaining)
                if keep and length <= limit
            ]
            if bucket_rows:
                block_nnz = _spsm_single_segment_block_nnz(limit)
                grouped.append(
                    {
                        "rows": torch.tensor(bucket_rows, dtype=torch.int32, device=device),
                        "block_nnz": block_nnz,
                        "max_segments": 1,
                        "diag_only": False,
                        "single_segment": False,
                        "staged": False,
                    }
                )
                for i, length in enumerate(lengths):
                    if remaining[i] and length <= limit:
                        remaining[i] = False
        if any(remaining):
            bucket_rows = [int(row) for row, keep in zip(rows, remaining) if keep]
            bucket_lengths = [length for length, keep in zip(lengths, remaining) if keep]
            bucket_max = max(bucket_lengths) if bucket_lengths else 0
            block_nnz = _spsm_block_nnz_for_row_length(bucket_max)
            max_segments = max((bucket_max + block_nnz - 1) // block_nnz, 1)
            grouped.append(
                {
                    "rows": torch.tensor(bucket_rows, dtype=torch.int32, device=device),
                    "block_nnz": block_nnz,
                    "max_segments": max_segments,
                    "diag_only": False,
                    "single_segment": False,
                    "staged": False,
                }
            )
    return grouped


def _auto_spsm_launch_config(indptr, block_nnz=None, max_segments=None):
    if indptr.numel() <= 1:
        max_nnz_per_row = 0
    else:
        row_lengths = indptr[1:] - indptr[:-1]
        max_nnz_per_row = int(row_lengths.max().item())

    auto_block = block_nnz is None
    if block_nnz is None:
        if max_nnz_per_row <= 64:
            block_nnz_use = 64
        elif max_nnz_per_row <= 256:
            block_nnz_use = 128
        elif max_nnz_per_row <= 1024:
            block_nnz_use = 256
        elif max_nnz_per_row <= 4096:
            block_nnz_use = 512
        else:
            block_nnz_use = 1024
    else:
        block_nnz_use = int(block_nnz)
        if block_nnz_use <= 0:
            raise ValueError("block_nnz must be positive")

    required_segments = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    if max_segments is None:
        max_segments_use = required_segments
        if auto_block:
            while max_segments_use > 2048 and block_nnz_use < 65536:
                block_nnz_use *= 2
                max_segments_use = max((max_nnz_per_row + block_nnz_use - 1) // block_nnz_use, 1)
    else:
        max_segments_use = int(max_segments)
        if max_segments_use <= 0:
            raise ValueError("max_segments must be positive")
        if max_segments_use < required_segments:
            raise ValueError(
                f"max_segments={max_segments_use} is too small; at least {required_segments} required"
            )
    return block_nnz_use, max_segments_use


def _auto_rhs_block(n_rhs):
    n_rhs = int(n_rhs)
    if n_rhs <= 4:
        return 4
    if n_rhs <= 8:
        return 8
    if n_rhs <= 16:
        return 16
    if n_rhs <= 32:
        return 32
    return 64


def _choose_group_block_rhs(
    n_rhs,
    *,
    block_nnz,
    max_segments,
    diag_only,
    single_segment,
    value_dtype,
):
    base = _auto_rhs_block(n_rhs)
    if diag_only:
        return min(base, 32 if value_dtype == torch.float32 else 16)
    if block_nnz <= 64 and single_segment:
        return min(base, 32 if value_dtype == torch.float32 else 16)
    if block_nnz <= 128 and single_segment:
        return min(base, 16 if value_dtype == torch.float32 else 8)
    if max_segments > 2 or block_nnz >= 512:
        return min(base, 8)
    if max_segments > 1 or block_nnz >= 256:
        return min(base, 16 if value_dtype == torch.float32 else 8)
    return base


def _choose_group_rhs_tile_groups(n_rhs, block_rhs_use, staged):
    if not staged:
        return 1
    if block_rhs_use <= 0:
        return 1
    if n_rhs >= block_rhs_use * 4:
        return 4
    if n_rhs >= block_rhs_use * 2:
        return 2
    return 1


def _prepare_spsm_rhs_work_buffer(rhs):
    # Library-main solves through a dedicated RHS work buffer. In our current
    # row-major NON_TRANS path that buffer already matches the final layout, so
    # we keep a contiguous in-place work copy rather than doing an extra transpose.
    return rhs.contiguous().clone()


def _coo_to_csr_sorted_unique(data, row64, col64, n_rows, n_cols):
    if data.numel() == 0:
        return (
            data,
            torch.empty(0, dtype=torch.int64, device=data.device),
            torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device),
        )
    key = row64 * max(1, n_cols) + col64
    if key.numel() == 1 or bool(torch.all(key[1:] > key[:-1]).item()):
        indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
        nnz_per_row = torch.bincount(row64.to(torch.int64), minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
        return data.contiguous(), col64.to(torch.int64).contiguous(), indptr
    try:
        order = torch.argsort(key, stable=True)
    except TypeError:
        order = torch.argsort(key)
    key_s = key[order]
    data_s = data[order]
    unique_key, inverse = torch.unique_consecutive(key_s, return_inverse=True)
    out_nnz = unique_key.numel()
    data_u = torch.zeros(out_nnz, dtype=data.dtype, device=data.device)
    data_u.scatter_add_(0, inverse, data_s)
    row_u = torch.div(unique_key, max(1, n_cols), rounding_mode="floor")
    col_u = unique_key - row_u * max(1, n_cols)
    indptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=data.device)
    if out_nnz > 0:
        nnz_per_row = torch.bincount(row_u, minlength=n_rows)
        indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    return data_u, col_u.to(torch.int64), indptr


def _coo_to_csr_allinone_view(data, row32, col32, n_rows):
    if data.numel() == 0:
        return (
            data,
            torch.empty(0, dtype=torch.int32, device=data.device),
            torch.zeros(n_rows + 1, dtype=torch.int32, device=data.device),
        )
    indptr = torch.zeros(n_rows + 1, dtype=torch.int32, device=data.device)
    counts = torch.bincount(row32.to(torch.int64), minlength=n_rows)
    indptr[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    return data, col32.contiguous(), indptr


def _prepare_spsm_csr_system(data, indices32, indptr32, n_rows, lower, unit_diagonal):
    if _use_spsm_polling_path():
        plan = {
            "kernel_dep_data": data,
            "kernel_dep_indices32": indices32,
            "kernel_dep_ptr": indptr32,
            "kernel_inv_diag": torch.empty(0, dtype=data.dtype, device=data.device),
            "launch_groups": "polling",
            "default_block_nnz": 0,
            "default_max_segments": 0,
            "lower_eff": bool(lower),
            "unit_diagonal": bool(unit_diagonal),
        }
        if torch.is_complex(data):
            plan["kernel_dep_data_ri"] = _complex_interleaved_view(data)
        return plan

    indices64 = indices32.to(torch.int64)
    indptr64 = indptr32.to(torch.int64)
    sorted_data, sorted_indices64, row_ids = _prepare_spsm_csr_sorted_view(
        data, indices64, indptr64, n_rows, n_rows
    )
    dep_data, dep_indices64, dep_ptr64 = _prepare_spsm_csr_dependency_view(
        sorted_data, sorted_indices64, indptr64, n_rows, lower=lower, row_ids=row_ids
    )
    levels = _build_spsm_levels(dep_ptr64, dep_indices64, n_rows, lower=lower)
    launch_groups = _bucketize_spsm_launch_groups(levels, dep_ptr64)
    diag = _prepare_spsm_diag(
        sorted_data,
        sorted_indices64,
        indptr64,
        n_rows,
        unit_diagonal=unit_diagonal,
        row_ids=row_ids,
    )
    inv_diag = _prepare_spsm_inv_diag(diag)
    default_block_nnz, default_max_segments = _auto_spsm_launch_config(dep_ptr64)
    plan = {
        "kernel_dep_data": dep_data,
        "kernel_dep_indices32": dep_indices64.to(torch.int32),
        "kernel_dep_ptr": _prepare_spsm_kernel_row_ptr(dep_ptr64),
        "kernel_inv_diag": inv_diag,
        "launch_groups": launch_groups,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "lower_eff": bool(lower),
        "unit_diagonal": bool(unit_diagonal),
    }
    if torch.is_complex(data):
        plan["kernel_dep_data_ri"] = _complex_interleaved_view(dep_data)
        plan["kernel_inv_diag_ri"] = _complex_interleaved_view(inv_diag)
    return plan


def _prepare_spsm_coo_system(data, row32, col32, n_rows, n_cols, lower, unit_diagonal):
    data_u, col_u64, row_ptr = _coo_to_csr_sorted_unique(
        data, row32.to(torch.int64), col32.to(torch.int64), n_rows, n_cols
    )
    if _use_spsm_polling_path():
        plan = {
            "kernel_dep_data": data_u,
            "kernel_dep_indices32": col_u64.to(torch.int32),
            "kernel_dep_ptr": _prepare_spsm_kernel_row_ptr(row_ptr),
            "kernel_inv_diag": torch.empty(0, dtype=data.dtype, device=data.device),
            "launch_groups": "polling",
            "default_block_nnz": 0,
            "default_max_segments": 0,
            "lower_eff": bool(lower),
            "unit_diagonal": bool(unit_diagonal),
        }
        if torch.is_complex(data):
            plan["kernel_dep_data_ri"] = _complex_interleaved_view(data_u)
        return plan

    dep_data, dep_indices64, dep_ptr64 = _prepare_spsm_csr_dependency_view(
        data_u, col_u64, row_ptr, n_rows, lower=lower
    )
    levels = _build_spsm_levels(dep_ptr64, dep_indices64, n_rows, lower=lower)
    launch_groups = _bucketize_spsm_launch_groups(levels, dep_ptr64)
    diag = _prepare_spsm_diag(
        data_u, col_u64, row_ptr, n_rows, unit_diagonal=unit_diagonal
    )
    inv_diag = _prepare_spsm_inv_diag(diag)
    default_block_nnz, default_max_segments = _auto_spsm_launch_config(dep_ptr64)
    plan = {
        "kernel_dep_data": dep_data,
        "kernel_dep_indices32": dep_indices64.to(torch.int32),
        "kernel_dep_ptr": _prepare_spsm_kernel_row_ptr(dep_ptr64),
        "kernel_inv_diag": inv_diag,
        "launch_groups": launch_groups,
        "default_block_nnz": default_block_nnz,
        "default_max_segments": default_max_segments,
        "lower_eff": bool(lower),
        "unit_diagonal": bool(unit_diagonal),
    }
    if torch.is_complex(data):
        plan["kernel_dep_data_ri"] = _complex_interleaved_view(dep_data)
        plan["kernel_inv_diag_ri"] = _complex_interleaved_view(inv_diag)
    return plan


def _resolve_spsm_csr_runtime(data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major):
    data, indices64, indptr64, B, n_rows, n_cols = _prepare_spsm_csr_inputs(
        data, indices, indptr, B, shape, opA, opB, major
    )
    cache_key = _spsm_preprocess_cache_key(
        "csr",
        (data, indices64, indptr64),
        shape,
        lower,
        unit_diagonal,
    )
    solve_plan = _spsm_cache_get(_SPSM_PREPROCESS_CACHE, cache_key)
    if solve_plan is None:
        solve_plan = _prepare_spsm_csr_system(
            data, indices64, indptr64, n_rows, lower, unit_diagonal
        )
        _spsm_cache_put(_SPSM_PREPROCESS_CACHE, cache_key, solve_plan, _SPSM_PREPROCESS_CACHE_SIZE)
    return data, B, n_rows, n_cols, solve_plan


def _resolve_spsm_coo_runtime(data, row, col, B, shape, lower, unit_diagonal, opA, opB, major):
    data, row64, col64, B, n_rows, n_cols = _prepare_spsm_coo_inputs(
        data, row, col, B, shape, opA, opB, major
    )
    cache_key = _spsm_preprocess_cache_key(
        "coo",
        (data, row64, col64),
        shape,
        lower,
        unit_diagonal,
    )
    solve_plan = _spsm_cache_get(_SPSM_PREPROCESS_CACHE, cache_key)
    if solve_plan is None:
        solve_plan = _prepare_spsm_coo_system(
            data, row64, col64, n_rows, n_cols, lower, unit_diagonal
        )
        _spsm_cache_put(_SPSM_PREPROCESS_CACHE, cache_key, solve_plan, _SPSM_PREPROCESS_CACHE_SIZE)
    return data, B, n_rows, n_cols, solve_plan


def _run_spsm_csr_core(
    data,
    indices32,
    indptr,
    inv_diag,
    rhs,
    n_rows,
    *,
    data_ri=None,
    inv_diag_ri=None,
    alpha=1.0,
    lower=True,
    block_nnz=None,
    max_segments=None,
    block_rhs=None,
    launch_groups=None,
    block_nnz_use=None,
    max_segments_use=None,
    unit_diagonal=False,
):
    if rhs.ndim != 2:
        raise ValueError("rhs must be 2D")
    rhs = rhs.contiguous()
    if rhs.shape[0] != n_rows:
        raise ValueError("rhs first dim must equal n_rows")
    n_rhs = int(rhs.shape[1])
    rhs_work = _prepare_spsm_rhs_work_buffer(rhs)
    if n_rows == 0 or n_rhs == 0:
        return rhs_work

    if launch_groups != "polling":
        if launch_groups is None:
            levels = _build_spsm_levels(indptr, indices32, n_rows, lower=lower)
            launch_groups = _bucketize_spsm_launch_groups(levels, indptr)
        if block_nnz_use is None or max_segments_use is None:
            block_nnz_use, max_segments_use = _auto_spsm_launch_config(
                indptr, block_nnz=block_nnz, max_segments=max_segments
            )

        is_complex = torch.is_complex(data)
        use_fp64 = data.dtype in (torch.float64, torch.complex128)
        if is_complex:
            data_ri = data_ri if data_ri is not None else _complex_interleaved_view(data)
            inv_diag_ri = (
                inv_diag_ri
                if inv_diag_ri is not None
                else _complex_interleaved_view(inv_diag)
            )
            rhs_work_ri = _complex_interleaved_view(rhs_work)
            alpha_re = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
            alpha_im = float(alpha.imag) if isinstance(alpha, complex) else 0.0

            for group in launch_groups:
                if isinstance(group, dict):
                    rows_lv = group["rows"]
                    diag_only = bool(group.get("diag_only", False))
                    block_nnz_lv = int(group.get("block_nnz", block_nnz_use))
                    max_segments_lv = int(group.get("max_segments", max_segments_use))
                    single_segment = False
                    staged = False
                else:
                    rows_lv = group
                    diag_only = False
                    block_nnz_lv = block_nnz_use
                    max_segments_lv = max_segments_use
                    single_segment = False
                    staged = False
                n_lv = rows_lv.numel()
                if n_lv == 0:
                    continue
                block_rhs_use = (
                    int(block_rhs)
                    if block_rhs is not None
                    else _choose_group_block_rhs(
                        n_rhs,
                        block_nnz=block_nnz_lv,
                        max_segments=max_segments_lv,
                        diag_only=diag_only,
                        single_segment=single_segment,
                        value_dtype=data.dtype,
                    )
                )
                if block_rhs_use <= 0:
                    raise ValueError("block_rhs must be positive")
                grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
                if diag_only:
                    _spsm_csr_diag_only_kernel_complex[grid](
                        inv_diag_ri,
                        rhs_work_ri,
                        rhs_work_ri,
                        rows_lv,
                        n_level_rows=n_lv,
                        n_rhs=n_rhs,
                        stride_b0=rhs_work.stride(0),
                        stride_x0=rhs_work.stride(0),
                        alpha_re=alpha_re,
                        alpha_im=alpha_im,
                        BLOCK_RHS=block_rhs_use,
                        USE_FP64_ACC=use_fp64,
                    )
                    continue
                if staged:
                    block_rhs_use = min(block_rhs_use, 8)
                    grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
                _spsm_csr_level_kernel_complex[grid](
                    data_ri,
                    indices32,
                    indptr,
                    inv_diag_ri,
                    rhs_work_ri,
                    rhs_work_ri,
                    rows_lv,
                    n_level_rows=n_lv,
                    n_rhs=n_rhs,
                    stride_b0=rhs_work.stride(0),
                    stride_x0=rhs_work.stride(0),
                    alpha_re=alpha_re,
                    alpha_im=alpha_im,
                    BLOCK_NNZ=block_nnz_lv,
                    BLOCK_RHS=block_rhs_use,
                    MAX_SEGMENTS=max_segments_lv,
                    USE_FP64_ACC=use_fp64,
                )
            return torch.view_as_complex(rhs_work_ri.reshape(n_rows, n_rhs, 2).contiguous())

        for group in launch_groups:
            if isinstance(group, dict):
                rows_lv = group["rows"]
                diag_only = bool(group.get("diag_only", False))
                single_segment = False
                staged = False
                block_nnz_lv = int(group.get("block_nnz", block_nnz_use))
                max_segments_lv = int(group.get("max_segments", max_segments_use))
            else:
                rows_lv = group
                diag_only = False
                single_segment = False
                staged = False
                block_nnz_lv = block_nnz_use
                max_segments_lv = max_segments_use
            n_lv = rows_lv.numel()
            if n_lv == 0:
                continue
            block_rhs_use = (
                int(block_rhs)
                if block_rhs is not None
                else _choose_group_block_rhs(
                    n_rhs,
                    block_nnz=block_nnz_lv,
                    max_segments=max_segments_lv,
                    diag_only=diag_only,
                    single_segment=single_segment,
                    value_dtype=data.dtype,
                )
            )
            if block_rhs_use <= 0:
                raise ValueError("block_rhs must be positive")
            rhs_tile_groups = _choose_group_rhs_tile_groups(n_rhs, block_rhs_use, staged)
            if diag_only:
                grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
                _spsm_csr_diag_only_kernel_real[grid](
                    inv_diag,
                    rhs_work,
                    rhs_work,
                    rows_lv,
                    n_level_rows=n_lv,
                    n_rhs=n_rhs,
                    stride_b0=rhs_work.stride(0),
                    stride_x0=rhs_work.stride(0),
                    alpha=alpha,
                    BLOCK_RHS=block_rhs_use,
                    USE_FP64_ACC=use_fp64,
                )
                continue
            grid = (n_lv, triton.cdiv(n_rhs, block_rhs_use))
            _spsm_csr_level_kernel_real[grid](
                data,
                indices32,
                indptr,
                inv_diag,
                rhs_work,
                rhs_work,
                rows_lv,
                n_level_rows=n_lv,
                n_rhs=n_rhs,
                stride_b0=rhs_work.stride(0),
                stride_x0=rhs_work.stride(0),
                alpha=alpha,
                BLOCK_NNZ=block_nnz_lv,
                BLOCK_RHS=block_rhs_use,
                MAX_SEGMENTS=max_segments_lv,
                USE_FP64_ACC=use_fp64,
            )
        return rhs_work

    if launch_groups == "polling":
        block_rhs_use = int(block_rhs) if block_rhs is not None else _auto_rhs_block(n_rhs)
        if block_rhs_use <= 0:
            raise ValueError("block_rhs must be positive")
        rhs_tiles = triton.cdiv(n_rhs, block_rhs_use)
        done = torch.empty((rhs_tiles, n_rows), dtype=torch.int32, device=rhs.device)
        done.zero_()
        is_complex = torch.is_complex(data)
        use_fp64 = data.dtype in (torch.float64, torch.complex128)
        grid = (n_rows, rhs_tiles)
        if is_complex:
            data_ri = data_ri if data_ri is not None else _complex_interleaved_view(data)
            rhs_work_ri = _complex_interleaved_view(rhs_work)
            diag_ri = torch.empty((n_rows * 2,), dtype=data_ri.dtype, device=rhs.device)
            _spsm_extract_diag_kernel_complex[(n_rows,)](
                data_ri,
                indices32,
                indptr,
                diag_ri,
                n_rows=n_rows,
                UNIT_DIAG=bool(unit_diagonal),
                USE_FP64_ACC=use_fp64,
            )
            alpha_re = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
            alpha_im = float(alpha.imag) if isinstance(alpha, complex) else 0.0
            _spsm_csr_polling_kernel_complex[grid](
                data_ri,
                indices32,
                indptr,
                diag_ri,
                rhs_work_ri,
                done,
                n_rows=n_rows,
                n_rhs=n_rhs,
                stride_work0=rhs_work.stride(0),
                alpha_re=alpha_re,
                alpha_im=alpha_im,
                BLOCK_RHS=block_rhs_use,
                USE_FP64_ACC=use_fp64,
                LOWER=bool(lower),
                UNIT_DIAG=bool(unit_diagonal),
            )
            return torch.view_as_complex(rhs_work_ri.reshape(n_rows, n_rhs, 2).contiguous())
        diag = torch.empty((n_rows,), dtype=data.dtype, device=rhs.device)
        _spsm_extract_diag_kernel_real[(n_rows,)](
            data,
            indices32,
            indptr,
            diag,
            n_rows=n_rows,
            UNIT_DIAG=bool(unit_diagonal),
            USE_FP64_ACC=use_fp64,
        )
        _spsm_csr_polling_kernel_real[grid](
            data,
            indices32,
            indptr,
            diag,
            rhs_work,
            done,
            n_rows=n_rows,
            n_rhs=n_rhs,
            stride_work0=rhs_work.stride(0),
            alpha=alpha,
            BLOCK_RHS=block_rhs_use,
            USE_FP64_ACC=use_fp64,
            LOWER=bool(lower),
            UNIT_DIAG=bool(unit_diagonal),
        )
        return rhs_work

    raise RuntimeError("SpSM solve plan must use the all-in-one polling path")


def flagsparse_spsm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    out=None,
    return_time=False,
):
    data, B, n_rows, _n_cols, solve_plan = _resolve_spsm_csr_runtime(
        data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major
    )
    alpha_value = 1.0 if _alpha_is_one(alpha) else _alpha_to_host_scalar(alpha)
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    x = _run_spsm_csr_core(
        solve_plan["kernel_dep_data"],
        solve_plan["kernel_dep_indices32"],
        solve_plan["kernel_dep_ptr"],
        solve_plan["kernel_inv_diag"],
        B,
        n_rows,
        data_ri=solve_plan.get("kernel_dep_data_ri"),
        inv_diag_ri=solve_plan.get("kernel_inv_diag_ri"),
        alpha=alpha_value,
        lower=solve_plan["lower_eff"],
        launch_groups=solve_plan["launch_groups"],
        block_nnz_use=solve_plan["default_block_nnz"],
        max_segments_use=solve_plan["default_max_segments"],
        unit_diagonal=solve_plan["unit_diagonal"],
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out
    if return_time:
        return x, elapsed_ms
    return x


def flagsparse_spsm_coo(
    data,
    row,
    col,
    B,
    shape,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    out=None,
    return_time=False,
):
    data, B, n_rows, _n_cols, solve_plan = _resolve_spsm_coo_runtime(
        data, row, col, B, shape, lower, unit_diagonal, opA, opB, major
    )
    alpha_value = 1.0 if _alpha_is_one(alpha) else _alpha_to_host_scalar(alpha)
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    x = _run_spsm_csr_core(
        solve_plan["kernel_dep_data"],
        solve_plan["kernel_dep_indices32"],
        solve_plan["kernel_dep_ptr"],
        solve_plan["kernel_inv_diag"],
        B,
        n_rows,
        data_ri=solve_plan.get("kernel_dep_data_ri"),
        inv_diag_ri=solve_plan.get("kernel_inv_diag_ri"),
        alpha=alpha_value,
        lower=solve_plan["lower_eff"],
        launch_groups=solve_plan["launch_groups"],
        block_nnz_use=solve_plan["default_block_nnz"],
        max_segments_use=solve_plan["default_max_segments"],
        unit_diagonal=solve_plan["unit_diagonal"],
    )
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if out is not None:
        if out.shape != x.shape or out.dtype != x.dtype:
            raise ValueError("out shape/dtype must match result")
        out.copy_(x)
        x = out
    if return_time:
        return x, elapsed_ms
    return x


def _analyze_spsm_csr(
    data,
    indices,
    indptr,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsm_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsm_csr_runtime(data, indices, indptr, B, shape, lower, unit_diagonal, opA, opB, major)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms
    return None


def _analyze_spsm_coo(
    data,
    row,
    col,
    B,
    shape,
    *,
    lower=True,
    unit_diagonal=False,
    opA="NON_TRANS",
    opB="NON_TRANS",
    major="row",
    clear_cache=False,
    return_time=False,
):
    if clear_cache:
        _clear_spsm_preprocess_cache()
    if return_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    _resolve_spsm_coo_runtime(data, row, col, B, shape, lower, unit_diagonal, opA, opB, major)
    if return_time:
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return elapsed_ms
    return None


def benchmark_spsm_case(
    fmt="csr",
    n_rows=1024,
    n_rhs=1024,
    nnz=8192,
    value_dtype=torch.float32,
    index_dtype=torch.int32,
    alpha=1.0,
    lower=True,
    unit_diagonal=False,
    warmup=10,
    iters=50,
):
    """Pure FlagSparse SpSM benchmark entry for one configuration."""
    device = torch.device("cuda")
    data, indices, indptr = _build_random_csr(
        n_rows, n_rows, nnz, value_dtype, index_dtype, device
    )
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr.to(torch.int64)[1:] - indptr.to(torch.int64)[:-1],
    )
    col_ids = indices.to(torch.int64)
    tri_mask = (col_ids <= row_ids) if lower else (col_ids >= row_ids)
    data = data[tri_mask]
    row_ids = row_ids[tri_mask]
    col_ids = col_ids[tri_mask]
    data, col_ids, indptr = _coo_to_csr_sorted_unique(data, row_ids, col_ids, n_rows, n_rows)
    row_ids = torch.repeat_interleave(
        torch.arange(n_rows, device=device, dtype=torch.int64),
        indptr[1:] - indptr[:-1],
    )
    diag_mask = col_ids == row_ids
    diag_present = torch.zeros(n_rows, dtype=torch.bool, device=device)
    if diag_mask.numel() > 0 and bool(torch.any(diag_mask).item()):
        diag_present[row_ids[diag_mask]] = True
    missing_diag = torch.nonzero(~diag_present, as_tuple=False).reshape(-1).to(torch.int64)
    if missing_diag.numel() > 0:
        diag_values = torch.ones(missing_diag.numel(), dtype=value_dtype, device=device)
        data = torch.cat([data, diag_values], dim=0)
        row_ids = torch.cat([row_ids, missing_diag], dim=0)
        col_ids = torch.cat([col_ids, missing_diag], dim=0)
        data, col_ids, indptr = _coo_to_csr_sorted_unique(data, row_ids, col_ids, n_rows, n_rows)
    B = torch.randn((n_rows, n_rhs), dtype=value_dtype, device=device).contiguous()
    shape = (n_rows, n_rows)

    if str(fmt).lower() == "coo":
        row_ids = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64),
            indptr[1:] - indptr[:-1],
        )
        analysis_ms = _analyze_spsm_coo(
            data,
            row_ids.to(index_dtype),
            col_ids.to(index_dtype),
            B,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            clear_cache=True,
            return_time=True,
        )
        solve_call = lambda: flagsparse_spsm_coo(
            data,
            row_ids.to(index_dtype),
            col_ids.to(index_dtype),
            B,
            shape,
            alpha=alpha,
            lower=lower,
            unit_diagonal=unit_diagonal,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    else:
        analysis_ms = _analyze_spsm_csr(
            data,
            col_ids.to(index_dtype),
            indptr.to(index_dtype),
            B,
            shape,
            lower=lower,
            unit_diagonal=unit_diagonal,
            clear_cache=True,
            return_time=True,
        )
        solve_call = lambda: flagsparse_spsm_csr(
            data,
            col_ids.to(index_dtype),
            indptr.to(index_dtype),
            B,
            shape,
            alpha=alpha,
            lower=lower,
            unit_diagonal=unit_diagonal,
            opA="NON_TRANS",
            opB="NON_TRANS",
            major="row",
        )
    C_fs, solve_ms = _benchmark_cuda_op(solve_call, warmup=warmup, iters=iters)
    return {
        "parameters": {
            "format": str(fmt).lower(),
            "n_rows": n_rows,
            "n_rhs": n_rhs,
            "nnz": int(data.numel()),
            "value_dtype": str(value_dtype),
            "index_dtype": str(index_dtype),
            "opA": "NON_TRANS",
            "opB": "NON_TRANS",
            "major": "row",
        },
        "performance": {
            "triton_analysis_ms": analysis_ms,
            "triton_solve_ms": solve_ms,
            "triton_time_total_ms": (
                analysis_ms + solve_ms if analysis_ms is not None and solve_ms is not None else None
            ),
        },
        "samples": {"flagsparse": C_fs},
    }
