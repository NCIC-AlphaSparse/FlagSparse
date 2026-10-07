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

"""Sliced-ELLPACK SpMV (``cusparseSpMV`` with ``CUSPARSE_SPMV_SELL_ALG1``).

Storage is cuSPARSE's, and the same one ``flagsparse_spsv_sell`` takes: rows are
grouped into slices of ``slice_size``; slice ``s`` owns
``values[slice_offsets[s]:slice_offsets[s + 1]]``, stored column-major (slot-major),
so entry ``(row r of the slice, slot j)`` sits at
``slice_offsets[s] + j * slice_size + r``. Padding carries column index ``-1``.

Type rules (cuSPARSE SpMV): half types and int8 accumulate in float32 / int32.
``out_dtype`` picks the output type where cuSPARSE allows a different Y type:
``f16/bf16 -> f32`` and ``i8 -> i32`` (default) or ``i8 -> f32``.

Real programs group slices; complex programs use one warp or parallel slot lanes.
Both use runtime loops, keeping per-thread private memory flat (MetaX caps it at 4 KB).
Complex values run on interleaved real/imag planes; Ascend takes a torch_npu path.
"""

from ._common import *

import os
import time

import triton
import triton.language as tl

SPMV_SELL_VALUE_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.int8,
)
# value dtype -> output dtypes cuSPARSE accepts for Y; the first is the default.
_SPMV_SELL_OUT_DTYPES = {
    torch.float16: (torch.float16, torch.float32),
    torch.bfloat16: (torch.bfloat16, torch.float32),
    torch.float32: (torch.float32,),
    torch.float64: (torch.float64,),
    torch.complex64: (torch.complex64,),
    torch.complex128: (torch.complex128,),
    torch.int8: (torch.int32, torch.float32),
}


def _sell_acc_dtype(value_dtype, out_dtype):
    if value_dtype == torch.int8:
        return torch.int32 if out_dtype == torch.int32 else torch.float32
    if value_dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return value_dtype


_TL = {
    torch.float32: tl.float32,
    torch.float64: tl.float64,
    torch.int32: tl.int32,
}

@triton.jit
def _spmv_sell_real_kernel(
    y_ptr, values_ptr, cols_ptr, offsets_ptr, x_ptr, n_rows,
    SLICE: tl.constexpr, BLOCK_R: tl.constexpr, ACC: tl.constexpr,
    GROUP: tl.constexpr = 1,
):
    s = tl.program_id(0) * GROUP + tl.arange(0, GROUP)[:, None]
    r = tl.arange(0, BLOCK_R)[None, :]
    smask = s * SLICE < n_rows
    rmask = (r < SLICE) & smask
    start = tl.load(offsets_ptr + s, mask=smask, other=0)
    width = (tl.load(offsets_ptr + s + 1, mask=smask, other=0) - start) // SLICE
    acc = tl.zeros((GROUP, BLOCK_R), dtype=ACC)
    for j in range(0, tl.max(tl.max(width, 0), 0)):
        pos = start + j * SLICE + r
        col = tl.load(cols_ptr + pos, mask=rmask & (j < width), other=-1)
        valid = col >= 0
        v = tl.load(values_ptr + pos, mask=valid, other=0).to(ACC)
        xv = tl.load(x_ptr + col, mask=valid, other=0).to(ACC)
        acc += v * xv
    row = s * SLICE + r
    tl.store(y_ptr + row, acc.to(y_ptr.dtype.element_ty), mask=rmask & (row < n_rows))


@triton.jit
def _spmv_sell_complex_kernel(
    y_ri_ptr, values_ri_ptr, cols_ptr, offsets_ptr, x_ri_ptr, n_rows,
    SLICE: tl.constexpr, BLOCK_R: tl.constexpr, GROUP: tl.constexpr = 1,
):
    s = tl.program_id(0) * GROUP + tl.arange(0, GROUP)[:, None]
    r = tl.arange(0, BLOCK_R)[None, :]
    smask = s * SLICE < n_rows
    rmask = (r < SLICE) & smask
    start = tl.load(offsets_ptr + s, mask=smask, other=0)
    width = (tl.load(offsets_ptr + s + 1, mask=smask, other=0) - start) // SLICE
    acc_re = tl.zeros((GROUP, BLOCK_R), dtype=tl.float32).to(y_ri_ptr.dtype.element_ty)
    acc_im = tl.zeros((GROUP, BLOCK_R), dtype=tl.float32).to(y_ri_ptr.dtype.element_ty)
    for j in range(0, tl.max(tl.max(width, 0), 0)):
        pos = start + j * SLICE + r
        col = tl.load(cols_ptr + pos, mask=rmask & (j < width), other=-1)
        valid = col >= 0
        vr = tl.load(values_ri_ptr + 2 * pos, mask=valid, other=0.0)
        vi = tl.load(values_ri_ptr + 2 * pos + 1, mask=valid, other=0.0)
        xr = tl.load(x_ri_ptr + 2 * col, mask=valid, other=0.0)
        xi = tl.load(x_ri_ptr + 2 * col + 1, mask=valid, other=0.0)
        acc_re += vr * xr - vi * xi
        acc_im += vr * xi + vi * xr
    row = s * SLICE + r
    mask = rmask & (row < n_rows)
    tl.store(y_ri_ptr + 2 * row, acc_re, mask=mask)
    tl.store(y_ri_ptr + 2 * row + 1, acc_im, mask=mask)


@triton.jit
def _spmv_sell_complex_parallel_kernel(
    Y, V, C, P, X, M,
    SLICE: tl.constexpr, BLOCK_R: tl.constexpr, SLOTS: tl.constexpr,
):
    s = tl.program_id(0)
    r = tl.arange(0, BLOCK_R)
    slots = tl.arange(0, SLOTS)
    start = tl.load(P + s)
    width = (tl.load(P + s + 1) - start) // SLICE
    re = tl.zeros((SLOTS, BLOCK_R), tl.float32).to(Y.dtype.element_ty)
    im = tl.zeros((SLOTS, BLOCK_R), tl.float32).to(Y.dtype.element_ty)
    for base in range(0, width, SLOTS):
        j = base + slots
        pos = start + j[:, None] * SLICE + r[None, :]
        col = tl.load(C + pos, (j[:, None] < width) & (r[None, :] < SLICE), other=-1)
        valid = col >= 0
        vr = tl.load(V + 2 * pos, valid, other=0.)
        vi = tl.load(V + 2 * pos + 1, valid, other=0.)
        xr = tl.load(X + 2 * col, valid, other=0.)
        xi = tl.load(X + 2 * col + 1, valid, other=0.)
        re += vr * xr - vi * xi
        im += vr * xi + vi * xr
    rows = s * SLICE + r
    valid_rows = (r < SLICE) & (rows < M)
    tl.store(Y + 2 * rows, tl.sum(re, 0), valid_rows)
    tl.store(Y + 2 * rows + 1, tl.sum(im, 0), valid_rows)


def _sell_entry_rows(cols, offsets, slice_size):
    """Row of every stored entry (padding included); used by the torch path."""
    lengths = (offsets[1:] - offsets[:-1]).to(torch.int64)
    n_entries = int(cols.numel())
    slice_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=cols.device, dtype=torch.int64),
        lengths,
        output_size=n_entries,
    )
    starts = torch.repeat_interleave(offsets[:-1].to(torch.int64), lengths, output_size=n_entries)
    local = torch.arange(n_entries, device=cols.device, dtype=torch.int64) - starts
    return slice_ids * slice_size + local.remainder(slice_size)


def _torch_spmv_sell(values, cols, offsets, x, n_rows, slice_size, acc_dtype):
    rows = _sell_entry_rows(cols, offsets, slice_size)
    valid = (cols >= 0) & (rows < n_rows)
    rows = rows[valid]
    c = cols[valid].to(torch.int64)
    y = torch.zeros(n_rows, dtype=acc_dtype, device=x.device)
    if _is_complex_dtype(acc_dtype):
        v = torch.view_as_real(values)[valid]
        xv = torch.view_as_real(x).index_select(0, c)
        prod = torch.stack(
            (v[:, 0] * xv[:, 0] - v[:, 1] * xv[:, 1], v[:, 0] * xv[:, 1] + v[:, 1] * xv[:, 0]),
            dim=-1,
        )
        torch.view_as_real(y).index_add_(0, rows, prod)
        return y
    prod = values[valid].to(acc_dtype) * x.index_select(0, c).to(acc_dtype)
    y.index_add_(0, rows, prod)
    return y


def _validate_spmv_sell(values, cols, offsets, x, shape, slice_size, validate):
    for t, name in ((values, "values"), (cols, "col_indices"), (offsets, "slice_offsets"), (x, "x")):
        if not torch.is_tensor(t):
            raise TypeError(f"{name} must be a torch.Tensor")
        if t.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor")
        if not _is_accel_tensor(t):
            raise ValueError(f"{name} must be an accelerator tensor")
    if len({values.device, cols.device, offsets.device, x.device}) != 1:
        raise ValueError("SELL SpMV inputs must be on one device")
    if values.dtype not in SPMV_SELL_VALUE_DTYPES:
        names = ", ".join(str(d).replace("torch.", "") for d in SPMV_SELL_VALUE_DTYPES)
        raise TypeError(f"SELL SpMV supports value dtypes: {names}")
    if x.dtype != values.dtype:
        raise TypeError("x dtype must match values dtype")
    if cols.dtype not in SUPPORTED_INDEX_DTYPES or offsets.dtype != cols.dtype:
        raise TypeError("col_indices and slice_offsets must share int32 or int64 dtype")
    n_rows, n_cols = int(shape[0]), int(shape[1])
    slice_size = int(slice_size)
    if slice_size <= 0:
        raise ValueError("slice_size must be positive")
    n_slices = (n_rows + slice_size - 1) // slice_size
    if offsets.numel() != n_slices + 1:
        raise ValueError("slice_offsets must have ceil(n_rows / slice_size) + 1 entries")
    if values.numel() != cols.numel():
        raise ValueError("values and col_indices must have the same length")
    if x.numel() != n_cols:
        raise ValueError(f"x length {x.numel()} does not match n_cols {n_cols}")
    if validate and values.numel() > 0:
        lengths = offsets[1:] - offsets[:-1]
        if (
            int(offsets[0].item()) != 0
            or int(offsets[-1].item()) != values.numel()
            or bool(torch.any(lengths % slice_size != 0).item())
            or bool(torch.any(lengths < 0).item())
        ):
            raise ValueError("invalid SELL slice_offsets")
        if int(cols.min().item()) < -1 or int(cols.max().item()) >= n_cols:
            raise IndexError("SELL column index out of range (padding must be -1)")
    return n_rows, n_cols, slice_size, n_slices


def flagsparse_spmv_sell(
    values,
    col_indices,
    slice_offsets,
    x,
    shape,
    *,
    slice_size,
    out=None,
    out_dtype=None,
    op="non",
    validate=True,
    return_time=False,
):
    """``y = A @ x`` for a sliced-ELLPACK ``A`` (layout in the module docstring).

    ``out_dtype`` (or the dtype of ``out``) selects a wider output where cuSPARSE
    allows it; see ``_SPMV_SELL_OUT_DTYPES``. Only ``op="non"`` is supported, as for
    cuSPARSE's SELL SpMV entry the delivery list names.
    """
    if str(op).strip().lower() not in ("non", "n", "non_trans"):
        raise NotImplementedError("SELL SpMV supports op='non' only")
    n_rows, _n_cols, slice_size, n_slices = _validate_spmv_sell(
        values, col_indices, slice_offsets, x, shape, slice_size, validate
    )
    allowed = _SPMV_SELL_OUT_DTYPES[values.dtype]
    if out is not None:
        if out_dtype is not None and out.dtype != out_dtype:
            raise ValueError("out.dtype conflicts with out_dtype")
        out_dtype = out.dtype
        if out.shape != (n_rows,) or out.device != x.device:
            raise ValueError("out shape/device must match the SELL SpMV result")
    out_dtype = allowed[0] if out_dtype is None else out_dtype
    if out_dtype not in allowed:
        names = ", ".join(str(d).replace("torch.", "") for d in allowed)
        raise TypeError(f"{values.dtype} SELL SpMV can write: {names}")
    acc_dtype = _sell_acc_dtype(values.dtype, out_dtype)
    values = values.contiguous()
    cols = col_indices.contiguous()
    offsets = slice_offsets.contiguous()
    x = x.contiguous()

    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    if _is_ascend_runtime():
        y = _torch_spmv_sell(values, cols, offsets, x, n_rows, slice_size, acc_dtype)
        y = y if y.dtype == out_dtype else y.to(out_dtype)
    else:
        y = torch.empty(n_rows, dtype=out_dtype, device=x.device)
        block_r = triton.next_power_of_2(slice_size)
        # Group existing slices; changing SLICE would reinterpret the storage.
        group = max(1, min(8, (256 if _is_rocm_runtime() else 128) // block_r))
        # On gfx936, fp32 benefits from four slices in one wave while f16/int8
        # use the wider eight-slice tile.  The latter have lower per-entry work
        # and need two waves to hide the gather latency.  These values were
        # selected over the ten Q4 matrices; callers can still tune explicitly.
        if _is_rocm_runtime():
            if values.dtype == torch.float32:
                group, num_warps = 4, 1
            else:
                group, num_warps = 8, 2
        else:
            num_warps = 4
        # gfx936 wavefront occupancy is sensitive to how many slices one
        # program owns.  Keep an opt-in override for backend tuning.
        override = os.environ.get("FLAGSPARSE_SELL_ROCM_GROUP")
        if _is_rocm_runtime() and override:
            group = max(1, min(8, int(override)))
        warp_override = os.environ.get("FLAGSPARSE_SELL_ROCM_WARPS")
        if _is_rocm_runtime() and warp_override:
            num_warps = max(1, min(8, int(warp_override)))
        grid = (triton.cdiv(n_slices, group),)
        if n_slices:
            if _is_complex_dtype(values.dtype):
                complex_args = (
                    torch.view_as_real(y),
                    torch.view_as_real(values),
                    cols, offsets,
                    torch.view_as_real(x),
                    n_rows,
                )
                if block_r <= 32 and not (_is_rocm_runtime() or _is_maca_runtime()):
                    _spmv_sell_complex_kernel[(n_slices,)](
                        *complex_args, SLICE=slice_size, BLOCK_R=block_r, num_warps=1,
                    )
                else:
                    _spmv_sell_complex_parallel_kernel[(n_slices,)](
                        *complex_args, SLICE=slice_size, BLOCK_R=block_r,
                        SLOTS=(8 if _is_rocm_runtime() else 4),
                    )
            else:
                _spmv_sell_real_kernel[grid](
                    y, values, cols, offsets, x, n_rows,
                    SLICE=slice_size, BLOCK_R=block_r, ACC=_TL[acc_dtype], GROUP=group,
                    num_warps=num_warps,
                )
    if out is not None:
        out.copy_(y)
        y = out
    if return_time:
        _ACCEL.synchronize()
        return y, (time.perf_counter() - t0) * 1000.0
    return y


def csr_to_sell(data, indices, indptr, n_rows, slice_size, index_dtype=None):
    """Convert CSR arrays to cuSPARSE SELL arrays on their own device, vectorised.

    Each slice is as wide as its longest row. Returns ``(values, col_indices,
    slice_offsets)``; indices keep the CSR index dtype unless ``index_dtype`` is given.
    Built from torch primitives only (cumsum / repeat_interleave / scatter), so it
    runs on every backend without a kernel of its own.
    """
    n_rows = int(n_rows)
    slice_size = int(slice_size)
    if slice_size <= 0:
        raise ValueError("slice_size must be positive")
    index_dtype = indices.dtype if index_dtype is None else index_dtype
    device = data.device
    n_slices = (n_rows + slice_size - 1) // slice_size
    ptr = indptr.to(torch.int64)
    lengths = ptr[1:] - ptr[:-1]
    padded = torch.zeros(n_slices * slice_size, dtype=torch.int64, device=device)
    padded[:n_rows] = lengths
    widths = padded.view(n_slices, slice_size).amax(dim=1) if n_slices else padded[:0]
    offsets = torch.zeros(n_slices + 1, dtype=torch.int64, device=device)
    offsets[1:] = torch.cumsum(widths * slice_size, dim=0)
    total = int(offsets[-1].item()) if n_slices else 0
    values = torch.zeros(total, dtype=data.dtype, device=device)
    cols = torch.full((total,), -1, dtype=index_dtype, device=device)
    nnz = int(data.numel())
    if nnz:
        rows = torch.repeat_interleave(
            torch.arange(n_rows, device=device, dtype=torch.int64), lengths, output_size=nnz
        )
        slot = torch.arange(nnz, device=device, dtype=torch.int64) - ptr[rows]
        pos = offsets[rows // slice_size] + slot * slice_size + rows % slice_size
        values[pos] = data
        cols[pos] = indices.to(index_dtype)
    return values, cols, offsets.to(index_dtype)


def dense_to_sell(dense, slice_size, index_dtype=torch.int32):
    """Build cuSPARSE SELL arrays from a 2D dense tensor (test/benchmark helper).

    Each slice is as wide as its longest row; shorter rows are padded with
    column ``-1`` and value ``0``.
    """
    n_rows = dense.shape[0]
    n_slices = (n_rows + slice_size - 1) // slice_size
    values, cols, offsets = [], [], [0]
    for s in range(n_slices):
        rows = range(s * slice_size, min((s + 1) * slice_size, n_rows))
        per_row = [torch.nonzero(dense[r] != 0, as_tuple=True)[0].tolist() for r in rows]
        width = max([len(p) for p in per_row] + [0])
        block_v = torch.zeros(width, slice_size, dtype=dense.dtype)
        block_c = torch.full((width, slice_size), -1, dtype=torch.int64)
        for local, (r, nz) in enumerate(zip(rows, per_row)):
            for j, c in enumerate(nz):
                block_v[j, local] = dense[r, c]
                block_c[j, local] = c
        values.append(block_v.reshape(-1))
        cols.append(block_c.reshape(-1))
        offsets.append(offsets[-1] + width * slice_size)
    empty_v = torch.zeros(0, dtype=dense.dtype)
    return (
        torch.cat(values) if values else empty_v,
        torch.cat(cols).to(index_dtype) if cols else torch.zeros(0, dtype=index_dtype),
        torch.tensor(offsets, dtype=index_dtype),
    )
