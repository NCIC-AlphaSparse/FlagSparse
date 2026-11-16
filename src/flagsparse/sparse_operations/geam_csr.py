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

"""Sparse matrix addition C = alpha * A + beta * B, all CSR (cusparse<t>csrgeam2).

Both inputs need sorted, duplicate-free column indices in every row -- the CSR
cuSPARSE's csrgeam2 expects -- and C comes out the same way. No sort is run:
every output position is computed from ranks.

1. probe: for each stored entry of B, binary-search its column in A's row.
   That gives ``lb_a`` (how many A columns in the row are smaller) and whether
   the column is also in A (a duplicate).
2. an exclusive prefix sum over the duplicate flags gives, per row, how many B
   entries collapse into an A entry, hence C's row lengths and ``indptr``.
3. place A: entry ``i`` of a row with column ``c`` lands after ``i`` A entries
   and every B-only column smaller than ``c`` (a binary search in B's row plus
   the duplicate prefix sum).
4. place B: a B-only entry lands after ``lb_a`` A entries and the B-only entries
   before it; a duplicate adds into the slot step 3 wrote.

Each output slot has exactly one writer per step and step 4 runs after step 3,
so nothing is atomic. Complex values go through ``view_as_real``. Ascend takes a
torch_npu path that merges the already sorted CSR keys by binary-search ranks.
"""

from ._common import *

import time

import triton
import triton.language as tl

GEAM_VALUE_DTYPES = (torch.float32, torch.float64, torch.complex64, torch.complex128)


@triton.jit
def _lower_bound(cols_ptr, starts, ends, target, valid, ITERS):
    """Per lane: the first position in [start, end) whose column is >= target."""
    lo = starts
    hi = ends
    for _ in range(ITERS):
        active = valid & (lo < hi)
        mid = (lo + hi) // 2
        v = tl.load(cols_ptr + mid, mask=active, other=0)
        right = v < target
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & (~right), mid, hi)
    return lo


@triton.jit
def _geam_probe_kernel(
    b_ind, b_ptr, a_ind, a_ptr, lb_a_out, dup_out, n_rows, iters,
    ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    b_start = tl.load(b_ptr + rows, mask=rmask, other=0)
    b_end = tl.load(b_ptr + rows + 1, mask=rmask, other=0)
    a_start = tl.load(a_ptr + rows, mask=rmask, other=0)
    a_end = tl.load(a_ptr + rows + 1, mask=rmask, other=0)
    max_len = tl.max(b_end - b_start, axis=0)
    for j in range(0, max_len, BLOCK):
        offs = b_start[:, None] + j + tl.arange(0, BLOCK)[None, :]
        mask = offs < b_end[:, None]
        col = tl.load(b_ind + offs, mask=mask, other=0)
        starts = tl.broadcast_to(a_start[:, None], (ROWS, BLOCK))
        ends = tl.broadcast_to(a_end[:, None], (ROWS, BLOCK))
        pos = _lower_bound(a_ind, starts, ends, col, mask, iters)
        hit_mask = mask & (pos < ends)
        found = tl.load(a_ind + pos, mask=hit_mask, other=-1) == col
        tl.store(lb_a_out + offs, (pos - starts).to(tl.int32), mask=mask)
        tl.store(dup_out + offs, (found & hit_mask).to(tl.int32), mask=mask)


@triton.jit
def _geam_place_a_kernel(
    a_ind, a_ptr, a_data, b_ind, b_ptr, dup_prefix, c_ptr, c_ind, c_data,
    alpha_re, alpha_im, n_rows, iters,
    COMPLEX: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    a_start = tl.load(a_ptr + rows, mask=rmask, other=0)
    a_end = tl.load(a_ptr + rows + 1, mask=rmask, other=0)
    b_start = tl.load(b_ptr + rows, mask=rmask, other=0)
    b_end = tl.load(b_ptr + rows + 1, mask=rmask, other=0)
    c_start = tl.load(c_ptr + rows, mask=rmask, other=0)
    d_start = tl.load(dup_prefix + b_start, mask=rmask, other=0)
    max_len = tl.max(a_end - a_start, axis=0)
    for j in range(0, max_len, BLOCK):
        local = j + tl.arange(0, BLOCK)[None, :]
        offs = a_start[:, None] + local
        mask = offs < a_end[:, None]
        col = tl.load(a_ind + offs, mask=mask, other=0)
        starts = tl.broadcast_to(b_start[:, None], (ROWS, BLOCK))
        ends = tl.broadcast_to(b_end[:, None], (ROWS, BLOCK))
        lb = _lower_bound(b_ind, starts, ends, col, mask, iters)
        dups_before = tl.load(dup_prefix + lb, mask=mask, other=0) - d_start[:, None]
        pos = c_start[:, None] + local + (lb - starts) - dups_before
        tl.store(c_ind + pos, col, mask=mask)
        if COMPLEX:
            v_re = tl.load(a_data + 2 * offs, mask=mask, other=0.0)
            v_im = tl.load(a_data + 2 * offs + 1, mask=mask, other=0.0)
            tl.store(c_data + 2 * pos, v_re * alpha_re - v_im * alpha_im, mask=mask)
            tl.store(c_data + 2 * pos + 1, v_re * alpha_im + v_im * alpha_re, mask=mask)
        else:
            v = tl.load(a_data + offs, mask=mask, other=0.0)
            tl.store(c_data + pos, v * alpha_re, mask=mask)


@triton.jit
def _geam_place_b_kernel(
    b_ind, b_ptr, b_data, lb_a, dup, dup_prefix, c_ptr, c_ind, c_data,
    beta_re, beta_im, n_rows,
    COMPLEX: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rmask = rows < n_rows
    b_start = tl.load(b_ptr + rows, mask=rmask, other=0)
    b_end = tl.load(b_ptr + rows + 1, mask=rmask, other=0)
    c_start = tl.load(c_ptr + rows, mask=rmask, other=0)
    d_start = tl.load(dup_prefix + b_start, mask=rmask, other=0)
    max_len = tl.max(b_end - b_start, axis=0)
    for j in range(0, max_len, BLOCK):
        local = j + tl.arange(0, BLOCK)[None, :]
        offs = b_start[:, None] + local
        mask = offs < b_end[:, None]
        col = tl.load(b_ind + offs, mask=mask, other=0)
        rank_a = tl.load(lb_a + offs, mask=mask, other=0)
        is_dup = tl.load(dup + offs, mask=mask, other=0) != 0
        dups_before = tl.load(dup_prefix + offs, mask=mask, other=0) - d_start[:, None]
        pos = c_start[:, None] + rank_a + local - dups_before
        only_b = mask & (~is_dup)
        both = mask & is_dup
        tl.store(c_ind + pos, col, mask=only_b)
        if COMPLEX:
            v_re = tl.load(b_data + 2 * offs, mask=mask, other=0.0)
            v_im = tl.load(b_data + 2 * offs + 1, mask=mask, other=0.0)
            add_re = v_re * beta_re - v_im * beta_im
            add_im = v_re * beta_im + v_im * beta_re
            old_re = tl.load(c_data + 2 * pos, mask=both, other=0.0)
            old_im = tl.load(c_data + 2 * pos + 1, mask=both, other=0.0)
            tl.store(c_data + 2 * pos, old_re + add_re, mask=mask)
            tl.store(c_data + 2 * pos + 1, old_im + add_im, mask=mask)
        else:
            v = tl.load(b_data + offs, mask=mask, other=0.0)
            old = tl.load(c_data + pos, mask=both, other=0.0)
            tl.store(c_data + pos, old + v * beta_re, mask=mask)


def _validate_csr(data, indices, indptr, shape, tag, check):
    for tensor, label in ((data, "data"), (indices, "indices"), (indptr, "indptr")):
        if not torch.is_tensor(tensor):
            raise TypeError(f"{tag}_{label} must be a torch.Tensor")
        if tensor.ndim != 1:
            raise ValueError(f"{tag}_{label} must be a 1D tensor")
        if not _is_accel_tensor(tensor):
            raise ValueError(f"{tag}_{label} must be an accelerator tensor")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    if indptr.numel() != n_rows + 1:
        raise ValueError(f"{tag}_indptr length must be n_rows+1={n_rows + 1}")
    if data.numel() != indices.numel():
        raise ValueError(f"{tag}_data and {tag}_indices must have the same length")
    if data.dtype not in GEAM_VALUE_DTYPES:
        raise TypeError(f"{tag}_data dtype must be one of {GEAM_VALUE_DTYPES}")
    if indices.dtype != torch.int32:
        raise TypeError(f"{tag}_indices dtype must be torch.int32")
    if indptr.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{tag}_indptr dtype must be torch.int32 or torch.int64")
    if not check or not indices.numel():
        return
    ptr = indptr.to(torch.int64)
    rows = torch.repeat_interleave(
        torch.arange(n_rows, device=ptr.device), ptr[1:] - ptr[:-1]
    )
    same_row = rows[1:] == rows[:-1]
    unsorted = same_row & (indices[1:] <= indices[:-1])
    flags = torch.stack((
        (ptr[0] != 0) | (ptr[-1] != indices.numel()) | (ptr[1:] < ptr[:-1]).any(),
        (indices.min() < 0) | (indices.max() >= n_cols),
        unsorted.any(),
    )).cpu()
    if bool(flags[0]):
        raise ValueError(f"{tag}_indptr is not a valid CSR row pointer")
    if bool(flags[1]):
        raise IndexError(f"{tag}_indices must lie in [0, {n_cols})")
    if bool(flags[2]):
        raise ValueError(
            f"{tag}_indices must be sorted and duplicate-free within each row"
        )


def _cdiv(a, b):
    # triton.cdiv is a ConstexprFunction since Triton 3.6 (~1 us per host call).
    return -(-a // b)


def _row_tile(nnz, n_rows):
    mean = nnz / max(1, n_rows)
    if mean <= 8:
        return 32, 8
    if mean <= 32:
        return 8, 32
    return 2, 128


def _torch_geam(a_data, a_indices, a_ptr, b_data, b_indices, b_ptr, shape, alpha, beta, ptr_dtype):
    """Ascend path: rank both sorted inputs and scatter their union without sorting."""
    n_cols = int(shape[1])
    dev = a_data.device
    nnz_a, nnz_b = a_data.numel(), b_data.numel()

    def scaled_values(data, scalar):
        if data.is_complex():
            planes = torch.view_as_real(data.resolve_conj())
            scalar = complex(scalar)
            if scalar == 1:
                return planes
            return torch.stack((
                planes[:, 0] * scalar.real - planes[:, 1] * scalar.imag,
                planes[:, 0] * scalar.imag + planes[:, 1] * scalar.real,
            ), dim=-1)
        scalar = complex(scalar).real
        return data if scalar == 1 else data * scalar

    def restore_values(values):
        return torch.view_as_complex(values) if a_data.is_complex() else values

    if not nnz_a or not nnz_b:
        data, indices, pointer, scalar = (
            (a_data, a_indices, a_ptr, alpha) if nnz_a
            else (b_data, b_indices, b_ptr, beta)
        )
        return (
            restore_values(scaled_values(data, scalar).clone()),
            indices.clone(), pointer.to(ptr_dtype).clone(),
        )

    offsets_a = torch.arange(nnz_a, device=dev)
    offsets_b = torch.arange(nnz_b, device=dev)
    rows_a = torch.searchsorted(a_ptr[1:].contiguous(), offsets_a, right=True)
    rows_b = torch.searchsorted(b_ptr[1:].contiguous(), offsets_b, right=True)
    keys_a = rows_a * n_cols + a_indices
    keys_b = rows_b * n_cols + b_indices
    rank_a = torch.searchsorted(keys_a, keys_b)
    gather_a = rank_a.clamp(max=nnz_a - 1)
    duplicate = (rank_a < nnz_a) & (keys_a.index_select(0, gather_a) == keys_b)
    duplicate_prefix = torch.zeros(nnz_b + 1, dtype=torch.int64, device=dev)
    torch.cumsum(duplicate, 0, out=duplicate_prefix[1:])
    nnz_c = nnz_a + nnz_b - int(duplicate_prefix[-1].item())
    c_ptr = a_ptr + b_ptr - duplicate_prefix.index_select(0, b_ptr)
    rank_b = torch.searchsorted(keys_b, keys_a)
    positions_a = offsets_a + rank_b - duplicate_prefix.index_select(0, rank_b)
    positions_b = rank_a + offsets_b - duplicate_prefix[:-1]
    values_a = scaled_values(a_data, alpha)
    values_b = scaled_values(b_data, beta)
    if a_data.is_complex():
        duplicate = duplicate[:, None]
    merged_b = torch.where(duplicate, values_a.index_select(0, gather_a), 0) + values_b
    c_data = torch.empty((nnz_c, *values_a.shape[1:]), dtype=values_a.dtype, device=dev)
    c_indices = torch.empty(nnz_c, dtype=torch.int32, device=dev)
    c_data.index_put_((positions_a,), values_a)
    c_data.index_put_((positions_b,), merged_b)
    c_indices.index_put_((positions_a,), a_indices)
    c_indices.index_put_((positions_b,), b_indices)
    return restore_values(c_data), c_indices, c_ptr.to(ptr_dtype)


def flagsparse_geam_csr(
    a_data,
    a_indices,
    a_indptr,
    b_data,
    b_indices,
    b_indptr,
    shape,
    *,
    alpha=1.0,
    beta=1.0,
    validate=True,
    return_time=False,
):
    """``C = alpha * A + beta * B`` for two m x n CSR matrices (cusparse<t>csrgeam2).

    Returns ``(c_data, c_indices, c_indptr)``: int32 column indices, a row pointer
    of ``a_indptr``'s dtype, columns sorted within each row. Both inputs must have
    sorted, duplicate-free columns per row; ``validate=True`` (the default) checks
    that, the index ranges and the row pointers in one host sync. A column present
    in both inputs is stored once even when the sum is zero, as csrgeam2 does.
    float32, float64, complex64 and complex128; A and B share one dtype.
    """
    _validate_csr(a_data, a_indices, a_indptr, shape, "a", validate)
    _validate_csr(b_data, b_indices, b_indptr, shape, "b", validate)
    if b_data.dtype != a_data.dtype:
        raise TypeError("a_data and b_data must have the same dtype")
    if len({t.device for t in (a_data, a_indices, a_indptr, b_data, b_indices, b_indptr)}) != 1:
        raise ValueError("A and B tensors must be on the same device")
    n_rows = int(shape[0])
    nnz_a, nnz_b = int(a_data.numel()), int(b_data.numel())
    ptr_dtype = a_indptr.dtype
    if ptr_dtype == torch.int32 and nnz_a + nnz_b > _INDEX_LIMIT_INT32:
        ptr_dtype = torch.int64
    dtype, dev = a_data.dtype, a_data.device
    a_c, b_c = complex(alpha), complex(beta)
    if not dtype.is_complex and (a_c.imag or b_c.imag):
        raise TypeError("complex alpha/beta need complex A and B")
    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    if _is_ascend_runtime():
        result = _torch_geam(
            a_data, a_indices, a_indptr.to(torch.int64), b_data, b_indices,
            b_indptr.to(torch.int64), shape, alpha, beta, ptr_dtype,
        )
    else:
        result = _geam_triton(
            a_data.contiguous(), a_indices.contiguous(), a_indptr.contiguous(),
            b_data.contiguous(), b_indices.contiguous(), b_indptr.contiguous(),
            n_rows, nnz_a, nnz_b, a_c, b_c, ptr_dtype, dtype, dev,
        )
    if return_time:
        _ACCEL.synchronize()
        return result, (time.perf_counter() - t0) * 1000.0
    return result


def _geam_triton(a_data, a_ind, a_ptr, b_data, b_ind, b_ptr, n_rows, nnz_a, nnz_b,
                 alpha, beta, ptr_dtype, dtype, dev):
    a_len = a_ptr[1:] - a_ptr[:-1]
    b_len = b_ptr[1:] - b_ptr[:-1]
    if n_rows == 0:
        return (
            torch.empty(0, dtype=dtype, device=dev),
            torch.empty(0, dtype=torch.int32, device=dev),
            torch.zeros(1, dtype=ptr_dtype, device=dev),
        )
    longest = torch.stack((a_len.max(), b_len.max())).to(torch.int64).cpu()
    iters_a = int(longest[0]).bit_length()
    iters_b = int(longest[1]).bit_length()

    lb_a = torch.empty(nnz_b, dtype=torch.int32, device=dev)
    dup = torch.empty(nnz_b, dtype=torch.int32, device=dev)
    rows_b, block_b = _row_tile(nnz_b, n_rows)
    if nnz_b:
        _geam_probe_kernel[(_cdiv(n_rows, rows_b),)](
            b_ind, b_ptr, a_ind, a_ptr, lb_a, dup, n_rows, iters_a,
            ROWS=rows_b, BLOCK=block_b, num_warps=4,
        )
    dup_prefix = torch.zeros(nnz_b + 1, dtype=torch.int64, device=dev)
    if nnz_b:
        torch.cumsum(dup, 0, out=dup_prefix[1:])
    b_ptr64 = b_ptr.to(torch.int64)
    dup_rows = dup_prefix[b_ptr64[1:]] - dup_prefix[b_ptr64[:-1]]
    c_len = a_len.to(torch.int64) + b_len.to(torch.int64) - dup_rows
    c_ptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=dev)
    torch.cumsum(c_len, 0, out=c_ptr[1:])
    nnz_c = int(c_ptr[-1].item())
    c_ind = torch.empty(nnz_c, dtype=torch.int32, device=dev)
    c_data = torch.empty(nnz_c, dtype=dtype, device=dev)

    complex_ = dtype.is_complex
    view = (lambda t: torch.view_as_real(t).reshape(-1)) if complex_ else (lambda t: t)
    if nnz_a:
        rows_a, block_a = _row_tile(nnz_a, n_rows)
        _geam_place_a_kernel[(_cdiv(n_rows, rows_a),)](
            a_ind, a_ptr, view(a_data), b_ind, b_ptr, dup_prefix, c_ptr, c_ind, view(c_data),
            alpha.real, alpha.imag, n_rows, iters_b,
            COMPLEX=complex_, ROWS=rows_a, BLOCK=block_a, num_warps=4,
        )
    if nnz_b:
        _geam_place_b_kernel[(_cdiv(n_rows, rows_b),)](
            b_ind, b_ptr, view(b_data), lb_a, dup, dup_prefix, c_ptr, c_ind, view(c_data),
            beta.real, beta.imag, n_rows,
            COMPLEX=complex_, ROWS=rows_b, BLOCK=block_b, num_warps=4,
        )
    return c_data, c_ind, c_ptr.to(ptr_dtype)


# ---------------------------------------------------------------------------
# C API helpers. The C API runs the same probe/place kernels but has no torch for
# the two prefix sums and the row-length step, so these do them on the device.
# int32 throughout: csrgeam2's row pointers and nnz are int32.
# ---------------------------------------------------------------------------


@triton.jit
def _geam_scan_blocks_kernel(x_ptr, out_ptr, sums_ptr, n, BLOCK: tl.constexpr):
    """out = exclusive prefix sum of x within each BLOCK; sums[pid] = its total."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0)
    tl.store(out_ptr + offs, tl.cumsum(x, axis=0) - x, mask=mask)
    tl.store(sums_ptr + pid, tl.sum(x, axis=0))


@triton.jit
def _geam_scan_sums_kernel(sums_ptr, n_blocks, BLOCK: tl.constexpr):
    """Exclusive scan of the block totals in place (one program); the grand total
    is written after them, at sums[n_blocks]."""
    offs = tl.arange(0, BLOCK)
    mask = offs < n_blocks
    s = tl.load(sums_ptr + offs, mask=mask, other=0)
    tl.store(sums_ptr + offs, tl.cumsum(s, axis=0) - s, mask=mask)
    tl.store(sums_ptr + n_blocks, tl.sum(s, axis=0))


@triton.jit
def _geam_scan_add_kernel(out_ptr, sums_ptr, n, n_blocks, BLOCK: tl.constexpr):
    """Adds each block's offset; program 0 also writes the total at out[n]."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    base = tl.load(sums_ptr + pid)
    tl.store(out_ptr + offs, tl.load(out_ptr + offs, mask=mask, other=0) + base, mask=mask)
    if pid == 0:
        tl.store(out_ptr + n, tl.load(sums_ptr + n_blocks))


@triton.jit
def _geam_row_nnz_kernel(a_ptr, b_ptr, dup_prefix, c_len, n_rows, BLOCK: tl.constexpr):
    """C's row lengths: A's plus B's, less the B entries that collapse into A's."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < n_rows
    a0 = tl.load(a_ptr + rows, mask=mask, other=0)
    a1 = tl.load(a_ptr + rows + 1, mask=mask, other=0)
    b0 = tl.load(b_ptr + rows, mask=mask, other=0)
    b1 = tl.load(b_ptr + rows + 1, mask=mask, other=0)
    d0 = tl.load(dup_prefix + b0, mask=mask, other=0)
    d1 = tl.load(dup_prefix + b1, mask=mask, other=0)
    tl.store(c_len + rows, (a1 - a0) + (b1 - b0) - (d1 - d0), mask=mask)
