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

from ._rocm_sparse_cache import TensorPlanCache
from ._musa_spmv_launch import launch as _launch_musa_spmv
from .mixed_spmx import _CUDA_DIRECT_LAUNCH

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
_ROCM_SELL_CSR_CACHE = TensorPlanCache(maxsize=16)
_MUSA_SELL_PLAN_CACHE = TensorPlanCache(maxsize=16)
_ILUVATAR_SELL_CSC_CACHE = TensorPlanCache(maxsize=16)


def _iluvatar_sell_csc_plan(values, cols, offsets, n_rows, n_cols, slice_size):
    sources = (values, cols, offsets)
    options = (int(n_rows), int(n_cols), int(slice_size))
    cached = _ILUVATAR_SELL_CSC_CACHE.get(sources, options)
    if cached is not None:
        return cached
    rows = _sell_entry_rows(cols, offsets, slice_size)
    valid = (cols >= 0) & (rows < n_rows)
    row_ids = rows[valid].to(torch.int64)
    col_ids = cols[valid].to(torch.int64)
    order = torch.argsort(col_ids, stable=True)
    row_ids = row_ids.index_select(0, order).to(cols.dtype)
    data = values[valid].index_select(0, order).contiguous()
    indices = row_ids.contiguous()
    counts = torch.bincount(col_ids, minlength=n_cols)
    indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(cols.dtype)
    plan = (data, indices, indptr)
    _ILUVATAR_SELL_CSC_CACHE.put(sources, options, plan)
    return plan


def _musa_sell_csr_plan(values, cols, offsets, n_rows, slice_size):
    """Compact heavily padded real SELL without retaining computed outputs."""
    rows = _sell_entry_rows(cols, offsets, slice_size)
    positions = torch.nonzero((cols >= 0) & (rows < n_rows)).flatten()
    plan = ()
    if values.numel() > max(positions.numel(), 1) * 2:
        rows = rows.index_select(0, positions)
        order = torch.argsort(rows, stable=True)
        rows = rows.index_select(0, order)
        positions = positions.index_select(0, order)
        boundaries = torch.arange(n_rows + 1, device=rows.device, dtype=rows.dtype)
        ptr = torch.searchsorted(rows, boundaries).to(cols.dtype)
        plan = (
            values.index_select(0, positions),
            cols.index_select(0, positions),
            ptr,
        )
    return plan


def _rocm_sell_csr_plan(values, cols, offsets, n_rows, slice_size):
    """Materialize a row-major view once for the ROCm real fast path."""
    sources = (values, cols, offsets)
    options = (int(n_rows), int(slice_size))
    cached = _ROCM_SELL_CSR_CACHE.get(sources, options)
    if cached is not None:
        return cached
    rows = _sell_entry_rows(cols, offsets, slice_size)
    valid = (cols >= 0) & (rows < n_rows)
    row_ids = rows[valid].to(torch.int64)
    order = torch.argsort(row_ids, stable=True)
    row_ids = row_ids.index_select(0, order)
    data = values[valid].index_select(0, order).contiguous()
    indices = cols[valid].index_select(0, order).contiguous()
    counts = torch.bincount(row_ids, minlength=n_rows)
    indptr = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(cols.dtype)
    plan = (data, indices, indptr, int(counts.max().item()) if n_rows else 0)
    _ROCM_SELL_CSR_CACHE.put(sources, options, plan)
    return plan


def _rocm_sell_real_kernel(values, cols, offsets, x, y, n_rows, slice_size):
    """Launch the tuned row-tile kernel over a cached row-major SELL view."""
    from . import _spmv_csr_kernels as kernels

    data, indices, indptr, max_row = _rocm_sell_csr_plan(
        values, cols, offsets, n_rows, slice_size
    )
    mean = data.numel() / max(n_rows, 1)
    if mean <= 4:
        rows_per, lanes, warps = 128, 4, 2
    elif mean <= 24:
        # The delivery matrices are dominated by rows shorter than one
        # wave.  A compact 8-lane row keeps masked work low; the third stage
        # hides the indirect x loads on gfx936.
        rows_per, lanes, warps = 32, 8, 1
    else:
        rows_per, lanes, warps = 16, 8, 1
    # Same accumulator as the generic SELL kernel: FP64 values keep FP64.
    acc = _TL[_sell_acc_dtype(values.dtype, y.dtype)]
    kernels.row_tile_real_kernel[(triton.cdiv(n_rows, rows_per),)](
        data, indices, indptr, x, y, indptr, n_rows,
        INDEXED=False,
        R=rows_per,
        V=lanes,
        STAGES=3,
        ACC=acc,
        POSITION_64=indices.dtype == torch.int64,
        FIXED_STEPS=0,
        num_warps=warps,
        enable_fp_fusion=(acc == tl.float32),
    )
    return y



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
def _musa_spmv_sell_kernel(
    Y, V, C, P, X, M,
    SLICE: tl.constexpr, ROWS: tl.constexpr, SLOTS: tl.constexpr,
    COMPLEX: tl.constexpr, ACC: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    slices = rows // SLICE
    start = tl.load(P + slices, rows < M, other=0)
    stop = tl.load(P + slices + 1, rows < M, other=0)
    width = (stop - start) // SLICE
    base = start + rows % SLICE
    slots = tl.arange(0, SLOTS)
    real = tl.zeros((ROWS, SLOTS), ACC)
    imag = tl.zeros((ROWS, SLOTS), ACC)
    for offset in range(0, tl.max(width, 0), SLOTS):
        j = offset + slots
        pos = base[:, None] + j[None, :] * SLICE
        active = (rows[:, None] < M) & (j[None, :] < width[:, None])
        col = tl.load(C + pos, active, other=-1)
        valid = active & (col >= 0)
        if COMPLEX:
            vr = tl.load(V + 2 * pos, valid, other=0).to(ACC)
            vi = tl.load(V + 2 * pos + 1, valid, other=0).to(ACC)
            xr = tl.load(X + 2 * col, valid, other=0).to(ACC)
            xi = tl.load(X + 2 * col + 1, valid, other=0).to(ACC)
            real += vr * xr - vi * xi
            imag += vr * xi + vi * xr
        else:
            value = tl.load(V + pos, valid, other=0).to(ACC)
            vector = tl.load(X + col, valid, other=0).to(ACC)
            real += value * vector
    if COMPLEX:
        tl.store(Y + 2 * rows, tl.sum(real, 1), rows < M)
        tl.store(Y + 2 * rows + 1, tl.sum(imag, 1), rows < M)
    else:
        tl.store(Y + rows, tl.sum(real, 1), rows < M)


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
    if values.dtype in SPMV_SELL_VALUE_DTYPES and x.dtype != values.dtype:
        raise TypeError("x dtype must match values dtype")
    dims = _validate_sell_arrays(values, cols, offsets, shape, slice_size, validate)
    if x.numel() != dims[1]:
        raise ValueError(f"x length {x.numel()} does not match n_cols {dims[1]}")
    return dims


def _validate_sell_arrays(values, cols, offsets, shape, slice_size, validate):
    """The matrix half of ``_validate_spmv_sell``; ``prepare_spmv_sell`` has no x."""
    for t, name in ((values, "values"), (cols, "col_indices"), (offsets, "slice_offsets")):
        if not torch.is_tensor(t):
            raise TypeError(f"{name} must be a torch.Tensor")
        if t.ndim != 1:
            raise ValueError(f"{name} must be a 1D tensor")
        if not _is_accel_tensor(t):
            raise ValueError(f"{name} must be an accelerator tensor")
    if len({values.device, cols.device, offsets.device}) != 1:
        raise ValueError("SELL SpMV inputs must be on one device")
    if values.dtype not in SPMV_SELL_VALUE_DTYPES:
        names = ", ".join(str(d).replace("torch.", "") for d in SPMV_SELL_VALUE_DTYPES)
        raise TypeError(f"SELL SpMV supports value dtypes: {names}")
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


def _musa_sell_plan(values, cols, offsets, x, shape, slice_size, validate):
    sources = (values, cols, offsets)
    options = (int(shape[0]), int(shape[1]), int(slice_size))
    state = _MUSA_SELL_PLAN_CACHE.get(sources, options)
    if state is None:
        dims = _validate_spmv_sell(values, cols, offsets, x, shape, slice_size, validate)
        state = {"dims": dims, "validated": bool(validate), "csr": None,
                 "arrays": (values.contiguous(), cols.contiguous(), offsets.contiguous())}
        _MUSA_SELL_PLAN_CACHE.put(sources, options, state)
    elif validate and not state["validated"]:
        _validate_spmv_sell(values, cols, offsets, x, shape, slice_size, True)
        state["validated"] = True
    else:
        if not torch.is_tensor(x):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 1:
            raise ValueError("x must be a 1D tensor")
        if not _is_accel_tensor(x):
            raise ValueError("x must be an accelerator tensor")
        if x.device != values.device:
            raise ValueError("SELL SpMV inputs must be on one device")
        if x.dtype != values.dtype:
            raise TypeError("x dtype must match values dtype")
        if x.numel() != state["dims"][1]:
            raise ValueError(f"x length {x.numel()} does not match n_cols {state['dims'][1]}")
    return state


def _musa_sell_tile(n_values, n_rows, n_slices, slice_size, complex_values):
    mean_width = n_values / max(1, n_slices * slice_size)
    if mean_width <= 4.5:
        return 128, 1
    if 9 < mean_width <= 16 and n_rows <= 524288:
        return 64, 2
    if not complex_values and 16 < mean_width <= 24 and n_rows >= 262144:
        return 128, 1
    return 32, 4


def flagsparse_spmv_sell(
    values=None,
    col_indices=None,
    slice_offsets=None,
    x=None,
    shape=None,
    *,
    slice_size=None,
    out=None,
    out_dtype=None,
    op="non",
    validate=True,
    return_time=False,
    prepared=None,
):
    """``y = A @ x`` for a sliced-ELLPACK ``A`` (layout in the module docstring).

    ``out_dtype`` (or the dtype of ``out``) selects a wider output where cuSPARSE
    allows it; see ``_SPMV_SELL_OUT_DTYPES``. Only ``op="non"`` is supported, as for
    cuSPARSE's SELL SpMV entry the delivery list names.

    ``prepared`` (from ``prepare_spmv_sell``) replaces the matrix arguments; the
    call then only checks ``x`` / ``out`` and launches the prepared kernel.
    """
    if str(op).strip().lower() not in ("non", "n", "non_trans"):
        raise NotImplementedError("SELL SpMV supports op='non' only")
    if prepared is not None:
        if any(a is not None for a in (values, col_indices, slice_offsets, shape, slice_size)):
            raise ValueError("pass either a prepared SELL plan or the SELL arrays, not both")
        if out_dtype is not None and out_dtype != prepared.out_dtype:
            raise ValueError("out_dtype does not match the prepared SELL plan")
        return spmv_sell_prepared(prepared, x, out=out, return_time=return_time)
    if any(a is None for a in (values, col_indices, slice_offsets, x, shape, slice_size)):
        raise TypeError(
            "flagsparse_spmv_sell needs values, col_indices, slice_offsets, x, shape "
            "and slice_size (or prepared=)"
        )
    musa = _is_mthreads_runtime()
    cached_musa = (
        musa and torch.is_tensor(values)
        and values.dtype in (
            torch.float16, torch.bfloat16, torch.float32, torch.int8, torch.complex64
        )
    )
    t0 = None
    if cached_musa:
        if return_time:
            _ACCEL.synchronize()
            t0 = time.perf_counter()
        state = _musa_sell_plan(values, col_indices, slice_offsets, x, shape, slice_size, validate)
        n_rows, _n_cols, slice_size, n_slices = state["dims"]
        values, cols, offsets = state["arrays"]
    else:
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
    if not cached_musa:
        values = values.contiguous()
        cols = col_indices.contiguous()
        offsets = slice_offsets.contiguous()
    x = x.contiguous()

    if return_time and not cached_musa:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    if musa:
        y = out if out is not None and out.is_contiguous() else torch.empty(
            n_rows, dtype=out_dtype, device=x.device
        )
        plan = ()
        if cached_musa:
            if state["csr"] is None:
                state["csr"] = ()
                if (
                    n_slices and not _is_complex_dtype(values.dtype)
                    and values.numel() > n_rows * 4.5
                ):
                    state["csr"] = _musa_sell_csr_plan(values, cols, offsets, n_rows, slice_size)
            plan = state["csr"]
        if plan:
            from .mixed_spmx import _musa_csr_spmv

            _musa_csr_spmv(*plan, x, n_rows, out_dtype, acc_dtype, y)
        elif n_slices:
            complex_values = _is_complex_dtype(values.dtype)
            rows, slots = _musa_sell_tile(
                values.numel(), n_rows, n_slices, slice_size, complex_values
            )
            target, matrix, vector = (
                (torch.view_as_real(t) for t in (y, values, x))
                if complex_values else (y, values, x)
            )
            _launch_musa_spmv(
                _musa_spmv_sell_kernel, ((n_rows + rows - 1) // rows,),
                (target, matrix, cols, offsets, vector), (n_rows,),
                (slice_size, rows, slots, complex_values,
                 tl.float64 if acc_dtype == torch.complex128 else (
                    tl.float32 if complex_values else _TL[acc_dtype]
                 )),
            )
    elif _is_ascend_runtime():
        y = _torch_spmv_sell(values, cols, offsets, x, n_rows, slice_size, acc_dtype)
        y = y if y.dtype == out_dtype else y.to(out_dtype)
    else:
        # Reuse a caller-provided output directly.  This keeps prepared
        # benchmark calls symmetric with cuSPARSE's preallocated DnVec and
        # avoids a full-device copy after every launch.
        # A strided ``out`` cannot be handed to the kernels, which index it
        # densely; it is filled by the copy below instead.
        y = out if out is not None and out.is_contiguous() else torch.empty(
            n_rows, dtype=out_dtype, device=x.device
        )
        if (
            _is_iluvatar_runtime()
            and values.dtype == torch.complex64
            and os.environ.get("FLAGSPARSE_SELL_USE_CSC") == "1"
        ):
            from .spmv_csc import flagsparse_spmv_csc

            csc_data, csc_rows, csc_ptr = _iluvatar_sell_csc_plan(
                values, cols, offsets, n_rows, _n_cols, slice_size
            )
            result = flagsparse_spmv_csc(
                csc_data, csc_rows, csc_ptr, x, (n_rows, _n_cols), op="non"
            )
            if out is not None:
                out.copy_(result)
                y = out
            else:
                y = result
            if return_time:
                _ACCEL.synchronize()
                return y, (time.perf_counter() - t0) * 1000.0
            return y
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
        elif _is_maca_runtime() and values.dtype in (torch.float16, torch.float32, torch.int8):
            # One slice per wave avoids C550's long serial slot loops and
            # improves occupancy for both real and integer SELL gathers.
            group, num_warps = 1, 1
        else:
            num_warps = 4
        # gfx936 wavefront occupancy is sensitive to how many slices one
        # program owns.  Keep an opt-in override for backend tuning.
        override = os.environ.get("FLAGSPARSE_SELL_ROCM_GROUP")
        if _is_rocm_runtime() and override:
            group = max(1, min(16, int(override)))
        warp_override = os.environ.get("FLAGSPARSE_SELL_ROCM_WARPS")
        if _is_rocm_runtime() and warp_override:
            num_warps = max(1, min(8, int(warp_override)))
        if _is_iluvatar_runtime() and not _is_complex_dtype(values.dtype):
            # BI-V benefits from one warp per slice until the launch grid gets
            # large enough that grouping two slices amortizes program overhead.
            group, num_warps = (1 if n_slices <= 32768 else 2), 1
            if slice_size <= 32:
                block_r = 64
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
                        num_warps=(1 if _is_maca_runtime() and values.dtype == torch.complex64 else 4),
                    )
            elif _is_rocm_runtime() and not _is_complex_dtype(values.dtype):
                _rocm_sell_real_kernel(
                    values, cols, offsets, x, y, n_rows, slice_size
                )
            else:
                _spmv_sell_real_kernel[grid](
                    y, values, cols, offsets, x, n_rows,
                    SLICE=slice_size, BLOCK_R=block_r, ACC=_TL[acc_dtype], GROUP=group,
                    num_warps=num_warps,
                )
    if out is not None:
        if y is not out:
            out.copy_(y)
        y = out
    if return_time:
        _ACCEL.synchronize()
        return y, (time.perf_counter() - t0) * 1000.0
    return y


# ---------------------------------------------------------------------------
# Prepared SELL-C-sigma
# ---------------------------------------------------------------------------

# 64-lane warps (BI-V150, C550, gfx936). On BI-V150 (2026-10-25, 10 matrices) real
# SELL was best at c=32, group=1, 1 warp with sigma=4096 (0.69-0.70x of the scaled
# H800 baseline); c=64 and group=2 both lost. Complex keeps c=64 for an explicit
# sigma route, but "auto" sends complex back to the native path there. Grouping
# slices makes the shorter one wait for the wider (5090: complex GROUP=2 doubled
# ASIC_680ks's time).
_SELL_PREP_WARP64 = _is_iluvatar_runtime() or _is_maca_runtime() or _is_rocm_runtime()
_SELL_PREP_K_DEFAULT = 4 if _is_iluvatar_runtime() else 1


def _sell_prep_defaults(complex_values):
    """(c, group, num_warps) before overrides: 5090-measured for CUDA, BI-V150 for 64-lane."""
    if _SELL_PREP_WARP64:
        return (64 if complex_values else 32), 1, 1
    return (32, 1, 1) if complex_values else (32, 4, 4)
_SELL_PREP_ROUTES = ("auto", "sigma", "call")
# sigma="auto": the smallest window (c, 2c, 4c, ... up to this) whose padding is
# within _SELL_AUTO_PADDING of nnz. Sorting rows reorders the x reads, which on a
# 5090 with the matrix L2-resident cost NACA0015 15% (39 -> 45 us at sigma=1024)
# for 1.30x -> 1.01x padding, so low-padding matrices keep (nearly) their order.
_SELL_AUTO_MAX_SIGMA = 4096
_SELL_AUTO_PADDING = 1.10
# reorder="auto": use reverse Cuthill-McKee only when it shrinks the median
# |col - row| at least this much from at least _SELL_RCM_MIN_SPREAD. On a 5090,
# RCM plus the per-call x permutation ran auto.mtx 2.17x and NACA0015 1.58x faster
# (median spread 35468 -> 2903 and 191584 -> 773); amazon0601 and ASIC_680ks
# (where RCM widens the spread) gained nothing.
_SELL_RCM_MIN_SPREAD = 4096
_SELL_RCM_MIN_GAIN = 4.0
_SELL_PREP_REORDERS = ("none", "rcm", "auto")


class PreparedSellSpmv:
    """Plan from ``prepare_spmv_sell``; run it with ``flagsparse_spmv_sell(prepared=...)``.

    ``route == "sigma"``: the caller's SELL arrays are re-encoded once into
    SELL-C-sigma -- rows sorted by length (descending) inside windows of ``sigma``
    rows, then cut into slices of ``c`` rows, stored slot-major like cuSPARSE -- so
    padding nearly disappears (ASIC_680ks 3.98x -> 1.17x of nnz at sigma=1024).
    Every load is masked by its lane's row length, so padding is never read (an
    inf/nan in ``x`` cannot leak through a padded zero), and ``perm`` writes each
    result back to its original row. ``route == "call"`` keeps the backend's own
    SELL path on the caller's arrays.

    ``reorder == "rcm"`` additionally relabels rows and columns of a square matrix
    by reverse Cuthill-McKee, so neighbouring rows read neighbouring ``x``; each run
    then permutes ``x`` into a plan-owned buffer first (one extra launch, timed), so
    one plan must not be run concurrently from several streams.

    The plan copies the values: when the source values change in place (torch's
    version counter), the next run refreshes them; when ``col_indices`` or
    ``slice_offsets`` change, it rebuilds the layout. Inference tensors have no
    version counter, so in-place changes to them are not seen -- prepare again.
    """

    __slots__ = (
        "shape", "n_rows", "n_cols", "slice_size", "device", "value_dtype", "out_dtype",
        "acc_dtype", "route", "sigma", "sigma_request", "c", "group", "num_warps", "paired",
        "reorder", "reorder_request", "k", "x_order", "x_buf", "y_buf", "y_order",
        "values", "cols", "offsets", "versions",
        "sell_values", "kernel_values", "sell_cols", "sell_offsets", "row_len", "perm", "kernel_perm",
        "gather", "n_slices", "n_values", "grid", "constants",
    )

    def __repr__(self):
        return (
            f"PreparedSellSpmv(route={self.route!r}, shape={self.shape}, "
            f"dtype={self.value_dtype}->{self.out_dtype}, sigma={self.sigma}, c={self.c}, "
            f"group={self.group}, num_warps={self.num_warps}, k={self.k}, reorder={self.reorder!r}, "
            f"n_values={self.n_values})"
        )


def _tensor_version(t):
    try:
        return t._version
    except RuntimeError:  # inference tensors carry no version counter
        return None


def _sell_prep_option(explicit, env, default, parse=int):
    if explicit is not None:
        return explicit
    raw = os.environ.get(env)
    return parse(raw) if raw not in (None, "") else default


def _sell_prep_k(explicit, complex_values, group):
    """Slice columns per sigma-kernel step (see ``prepare_spmv_sell``)."""
    if explicit is None:
        raw = os.environ.get("FLAGSPARSE_SELL_PREP_K")
        k = int(raw) if raw not in (None, "") else _SELL_PREP_K_DEFAULT
    else:
        k = int(explicit)
    if k < 1 or k > 16 or k & (k - 1):
        raise ValueError("k must be a power of two in [1, 16]")
    if k > 1 and (group > 1 or complex_values):
        if explicit is not None:
            raise ValueError("k > 1 requires group=1 and real-valued SELL data")
        # A process-wide FLAGSPARSE_SELL_PREP_K must not break complex / grouped plans.
        return 1
    return k


def _sell_plan_variant(p, *, k, out_dtype):
    """A sigma plan sharing ``p``'s layout, with another ``k`` and / or ``out_dtype``.

    K and the output dtype do not change the layout, so a caller choosing between
    them (``prepare_spmv_coo`` on Iluvatar) re-encodes the matrix once.
    """
    if p.route != "sigma":
        raise ValueError("only sigma plans can be varied")
    allowed = _SPMV_SELL_OUT_DTYPES[p.value_dtype]
    if out_dtype not in allowed:
        names = ", ".join(str(d).replace("torch.", "") for d in allowed)
        raise TypeError(f"{p.value_dtype} SELL SpMV can write: {names}")
    q = PreparedSellSpmv()
    for name in PreparedSellSpmv.__slots__:
        setattr(q, name, getattr(p, name))
    q.k = _sell_prep_k(k, _is_complex_dtype(p.value_dtype), p.group)
    q.out_dtype = out_dtype
    q.acc_dtype = _sell_acc_dtype(p.value_dtype, out_dtype)
    if not _is_complex_dtype(p.value_dtype):
        q.constants = (p.c, p.group, _TL[q.acc_dtype]) + tuple(p.constants[3:])
    if p.x_order is not None:
        q.y_buf = torch.empty(p.n_rows, dtype=out_dtype, device=p.device)
    return q


def _sell_window_order(lengths, sigma):
    """Row order: lengths descending inside consecutive windows of ``sigma`` rows."""
    n_rows = lengths.numel()
    longest = int(lengths.max().item()) if n_rows else 0
    window = torch.arange(n_rows, device=lengths.device, dtype=torch.int64) // sigma
    return torch.argsort(window * (longest + 1) + (longest - lengths), stable=True)


def _sell_padded_slots(sorted_lengths, c):
    n_slices = (sorted_lengths.numel() + c - 1) // c
    lane_len = torch.zeros(n_slices * c, dtype=torch.int64, device=sorted_lengths.device)
    lane_len[: sorted_lengths.numel()] = sorted_lengths
    return int(lane_len.view(n_slices, c).amax(dim=1).sum().item()) * c if n_slices else 0


def _sell_auto_sigma(lengths, c):
    nnz = max(1, int(lengths.sum().item()))
    sigma = c
    while True:
        order = (
            _sell_window_order(lengths, sigma) if sigma > c
            else torch.arange(lengths.numel(), device=lengths.device)
        )
        if (
            _sell_padded_slots(lengths[order], c) <= _SELL_AUTO_PADDING * nnz
            or sigma >= _SELL_AUTO_MAX_SIGMA
            or sigma >= lengths.numel()
        ):
            return sigma
        sigma *= 2


def _rcm_order(rows, cols, n):
    """Reverse Cuthill-McKee order (new -> old) of the symmetrised pattern, or None.

    The package does not depend on SciPy; only this opt-in step uses it.
    """
    try:
        import numpy as np
        import scipy.sparse as sp
        from scipy.sparse.csgraph import reverse_cuthill_mckee
    except ImportError:
        return None
    r, c = rows.cpu().numpy(), cols.cpu().numpy()
    pattern = sp.csr_matrix((np.ones(r.size, dtype=np.int8), (r, c)), shape=(n, n))
    order = reverse_cuthill_mckee((pattern + pattern.T).tocsr(), symmetric_mode=True)
    return torch.from_numpy(order.astype(np.int64)).to(rows.device)


def _median_spread(rows, cols):
    return float((cols - rows).abs().float().median().item()) if rows.numel() else 0.0


def _sell_sigma_layout(values, cols, offsets, n_rows, n_cols, slice_size, c, sigma, reorder):
    """Re-encode cuSPARSE SELL arrays as SELL-C-sigma (all on the device).

    ``sigma`` is a row count, or ``"auto"`` (see ``_SELL_AUTO_PADDING``); ``reorder``
    is ``"none"``, ``"rcm"`` or ``"auto"`` (see ``_SELL_RCM_MIN_GAIN``).
    """
    device = values.device
    n_in = int(cols.numel())
    widths = (offsets[1:] - offsets[:-1]).to(torch.int64)
    slice_ids = torch.repeat_interleave(
        torch.arange(widths.numel(), device=device, dtype=torch.int64), widths, output_size=n_in
    )
    starts = torch.repeat_interleave(offsets[:-1].to(torch.int64), widths, output_size=n_in)
    local = torch.arange(n_in, device=device, dtype=torch.int64) - starts
    rows = slice_ids * slice_size + local.remainder(slice_size)
    slots = local // slice_size
    src = torch.nonzero((cols >= 0) & (rows < n_rows)).flatten()
    rows, slots = rows[src], slots[src]
    # Row-major order of the stored entries, keeping each row's slot order.
    span = int(slots.max().item()) + 1 if src.numel() else 1
    order = torch.argsort(rows * span + slots)
    src, rows, slots = src[order], rows[order], slots[order]
    entry_cols = cols[src].to(torch.int64)
    nnz = int(src.numel())

    x_order = None
    if reorder != "none" and n_rows == n_cols and n_rows > 1 and nnz:
        rcm = _rcm_order(rows, entry_cols, n_rows)
        if rcm is None and reorder == "rcm":
            raise RuntimeError("reorder='rcm' needs SciPy (scipy.sparse.csgraph)")
        if rcm is not None:
            new_label = torch.empty_like(rcm)
            new_label[rcm] = torch.arange(n_rows, device=device, dtype=torch.int64)
            new_rows, new_cols = new_label[rows], new_label[entry_cols]
            before = _median_spread(rows, entry_cols)
            after = _median_spread(new_rows, new_cols)
            if reorder == "rcm" or (
                before >= _SELL_RCM_MIN_SPREAD and after * _SELL_RCM_MIN_GAIN <= before
            ):
                # Same per-row summation order, relabelled rows and columns.
                again = torch.argsort(new_rows * span + slots)
                src, rows, entry_cols = src[again], new_rows[again], new_cols[again]
                x_order = rcm
    lengths = torch.bincount(rows, minlength=n_rows)
    row_ptr = torch.zeros(n_rows + 1, dtype=torch.int64, device=device)
    row_ptr[1:] = torch.cumsum(lengths, 0)
    within = torch.arange(nnz, device=device, dtype=torch.int64) - row_ptr[rows]

    if sigma == "auto":
        sigma = _sell_auto_sigma(lengths, c) if n_rows else c
    if sigma > c and n_rows:
        row_order = _sell_window_order(lengths, sigma)
    else:
        row_order = torch.arange(n_rows, device=device, dtype=torch.int64)
    position = torch.empty(n_rows, dtype=torch.int64, device=device)
    position[row_order] = torch.arange(n_rows, device=device, dtype=torch.int64)

    n_slices = (n_rows + c - 1) // c
    lane_len = torch.zeros(n_slices * c, dtype=torch.int64, device=device)
    lane_len[:n_rows] = lengths[row_order]
    slice_width = lane_len.view(n_slices, c).amax(dim=1) if n_slices else lane_len[:0]
    sell_offsets = torch.zeros(n_slices + 1, dtype=torch.int64, device=device)
    sell_offsets[1:] = torch.cumsum(slice_width * c, 0)
    total = int(sell_offsets[-1].item()) if n_slices else 0

    lane = position[rows]
    dest = sell_offsets[lane // c] + within * c + lane.remainder(c)
    gather_dtype = torch.int32 if n_in < 2**31 else torch.int64
    gather = torch.zeros(total, dtype=gather_dtype, device=device)
    gather[dest] = src.to(gather_dtype)
    sell_cols = torch.zeros(total, dtype=torch.int32, device=device)
    sell_cols[dest] = entry_cols.to(torch.int32)
    sell_values = torch.zeros(total, dtype=values.dtype, device=device)
    if values.is_complex():
        # Index on the real planes: some backends reject complex index_put.
        torch.view_as_real(sell_values)[dest] = torch.view_as_real(values)[src]
    else:
        sell_values[dest] = values[src]
    perm = torch.full((n_slices * c,), -1, dtype=torch.int32, device=device)
    kernel_perm = torch.full((n_slices * c,), -1, dtype=torch.int32, device=device)
    # lane -> original row (through the RCM relabelling when there is one)
    perm[:n_rows] = (row_order if x_order is None else x_order[row_order]).to(torch.int32)
    kernel_perm[:n_rows] = row_order.to(torch.int32)
    offset_dtype = torch.int32 if total < 2**31 else torch.int64
    return {
        "sell_values": sell_values,
        "sell_cols": sell_cols,
        "sell_offsets": sell_offsets.to(offset_dtype),
        "row_len": lane_len.to(torch.int32),
        "perm": perm,
        "kernel_perm": kernel_perm,
        "gather": gather,
        "n_slices": n_slices,
        "n_values": total,
        "sigma": sigma,
        "x_order": x_order,
        "reorder": "none" if x_order is None else "rcm",
    }


def _install_sell_sigma(p):
    layout = _sell_sigma_layout(
        p.values, p.cols, p.offsets, p.n_rows, p.n_cols, p.slice_size, p.c,
        p.sigma_request, p.reorder_request,
    )
    for name, value in layout.items():
        setattr(p, name, value)
    p.x_buf = (
        None if p.x_order is None
        else torch.empty(p.n_cols, dtype=p.value_dtype, device=p.device)
    )
    p.y_buf = (
        None if p.x_order is None
        else torch.empty(p.n_rows, dtype=p.out_dtype, device=p.device)
    )
    p.y_order = (
        None if p.x_order is None
        else torch.empty_like(p.x_order)
    )
    if p.y_order is not None:
        p.y_order[p.x_order] = torch.arange(p.n_rows, device=p.device, dtype=p.x_order.dtype)
    p.kernel_values = (
        torch.view_as_real(p.sell_values) if p.sell_values.is_complex() else p.sell_values
    )
    p.grid = (triton.cdiv(p.n_slices, p.group),)
    p.versions = tuple(_tensor_version(t) for t in (p.values, p.cols, p.offsets))


def prepare_spmv_sell(
    values,
    col_indices,
    slice_offsets,
    shape,
    *,
    slice_size,
    out_dtype=None,
    route="auto",
    sigma=None,
    c=None,
    group=None,
    num_warps=None,
    k=None,
    paired=None,
    reorder=None,
    validate=True,
):
    """Plan ``y = A @ x`` for cuSPARSE SELL arrays once; all re-encoding happens here.

    ``route``: ``"sigma"`` re-encodes as SELL-C-sigma (see ``PreparedSellSpmv``),
    ``"call"`` keeps the backend's own SELL path, ``"auto"`` picks ``"sigma"`` on CUDA
    and Iluvatar and ``"call"`` elsewhere (Ascend has no Triton path for it).
    ``sigma`` is a window in rows or ``"auto"`` (the smallest window, doubling from
    ``c``, whose padding is within 10% of nnz; ``prepared.sigma`` reports it).
    ``k``: slice columns read per step of the sigma kernel -- the ``[k, c]`` tile is
    contiguous, so a program has ``k * c`` loads in flight instead of ``c``. A power of
    two in [1, 16]; ``k > 1`` needs real values and ``group == 1`` (an explicit value
    that breaks this raises; ``FLAGSPARSE_SELL_PREP_K`` and the default apply only
    where it holds). Defaults to 4 on Iluvatar (BI-V150 P9: 1.15-1.6x on most of the
    delivery matrices) and 1 elsewhere.
    ``reorder``: ``"none"`` (default), ``"rcm"`` (reverse Cuthill-McKee relabelling of
    a square matrix; needs SciPy) or ``"auto"`` (RCM only where it shrinks the median
    ``|col - row|`` 4x; ``prepared.reorder`` reports what was done).
    Tuning (``sigma``, ``c`` rows per slice, ``group`` slices per program,
    ``num_warps``, ``k``, ``paired`` complex loads, ``reorder``) falls back to the
    environment variables ``FLAGSPARSE_SELL_PREP_SIGMA / _C / _GROUP / _WARPS / _K /
    _PAIRED / _REORDER`` read here, then to backend defaults. Preparation synchronizes; time
    ``flagsparse_spmv_sell(prepared=...)``.
    """
    n_rows, n_cols, slice_size, _ = _validate_sell_arrays(
        values, col_indices, slice_offsets, shape, slice_size, validate
    )
    allowed = _SPMV_SELL_OUT_DTYPES[values.dtype]
    out_dtype = allowed[0] if out_dtype is None else out_dtype
    if out_dtype not in allowed:
        names = ", ".join(str(d).replace("torch.", "") for d in allowed)
        raise TypeError(f"{values.dtype} SELL SpMV can write: {names}")
    route = str(route or "auto").strip().lower()
    if route not in _SELL_PREP_ROUTES:
        raise ValueError(f"route must be one of {', '.join(_SELL_PREP_ROUTES)}")
    if route == "auto":
        route = "sigma" if (
            _backend_name() == "cuda"
            or (_is_iluvatar_runtime() and not _is_complex_dtype(values.dtype))
        ) else "call"
    if route == "sigma" and _is_ascend_runtime():
        raise ValueError("the SELL-C-sigma route needs Triton; Ascend keeps its torch_npu path")

    p = PreparedSellSpmv()
    p.shape, p.n_rows, p.n_cols, p.slice_size = (n_rows, n_cols), n_rows, n_cols, slice_size
    p.device, p.value_dtype, p.out_dtype = values.device, values.dtype, out_dtype
    p.acc_dtype = _sell_acc_dtype(values.dtype, out_dtype)
    p.route = route
    p.values = values.contiguous()
    p.cols = col_indices.contiguous()
    p.offsets = slice_offsets.contiguous()
    p.n_values = int(values.numel())
    for name in ("sigma", "sigma_request", "c", "group", "num_warps", "k", "paired", "reorder",
                 "reorder_request", "x_order", "x_buf", "y_buf", "y_order", "sell_values", "kernel_values",
                 "sell_cols", "sell_offsets", "row_len", "perm", "kernel_perm", "gather", "n_slices", "grid",
                 "constants", "versions"):
        setattr(p, name, None)
    if route == "call":
        if k is not None and int(k) != 1:
            raise ValueError("k > 1 needs route='sigma'")
        p.k = 1
        return p

    complex_values = _is_complex_dtype(values.dtype)
    default_c, default_group, default_warps = _sell_prep_defaults(complex_values)
    p.c = int(_sell_prep_option(c, "FLAGSPARSE_SELL_PREP_C", default_c))
    p.group = int(_sell_prep_option(group, "FLAGSPARSE_SELL_PREP_GROUP", default_group))
    p.num_warps = int(_sell_prep_option(num_warps, "FLAGSPARSE_SELL_PREP_WARPS", default_warps))
    p.k = _sell_prep_k(k, complex_values, p.group)
    p.paired = bool(_sell_prep_option(paired, "FLAGSPARSE_SELL_PREP_PAIRED", False, lambda v: v != "0"))
    p.reorder_request = str(_sell_prep_option(reorder, "FLAGSPARSE_SELL_PREP_REORDER", "none", str)).lower()
    if p.reorder_request not in _SELL_PREP_REORDERS:
        raise ValueError(f"reorder must be one of {', '.join(_SELL_PREP_REORDERS)}")
    if p.reorder_request == "rcm" and n_rows != n_cols:
        raise ValueError("reorder='rcm' relabels rows and columns together: square matrices only")
    sigma_default = 4096 if _is_iluvatar_runtime() and not complex_values else "auto"
    sigma = _sell_prep_option(sigma, "FLAGSPARSE_SELL_PREP_SIGMA", sigma_default, str)
    for name, value in (("c", p.c), ("group", p.group)):
        if value <= 0 or value & (value - 1) or value > 1024:
            raise ValueError(f"{name} must be a power of two in [1, 1024]")
    if p.num_warps not in (1, 2, 4, 8, 16):
        raise ValueError("num_warps must be 1, 2, 4, 8 or 16")
    if str(sigma).strip().lower() == "auto":
        p.sigma_request = "auto"
    else:
        sigma = int(sigma)
        if sigma <= 0:
            raise ValueError("sigma must be positive or 'auto'")
        # Windows of whole slices; sigma <= c keeps the caller's row order.
        p.sigma_request = sigma if sigma <= p.c else -(-sigma // p.c) * p.c
    if complex_values:
        acc = tl.float64 if values.dtype == torch.complex128 else tl.float32
    else:
        acc = _TL[p.acc_dtype]
    p.constants = (p.c, p.group, acc, complex_values, p.paired and complex_values)
    _install_sell_sigma(p)
    return p


@triton.jit
def _spmv_sell_sigma_kernel(
    y_ptr, v_ptr, c_ptr, off_ptr, len_ptr, perm_ptr, x_ptr, n_slices,
    C: tl.constexpr, GROUP: tl.constexpr, ACC: tl.constexpr,
    COMPLEX: tl.constexpr, PAIRED: tl.constexpr,
):
    s = tl.program_id(0) * GROUP + tl.arange(0, GROUP)[:, None]
    lane = tl.arange(0, C)[None, :]
    smask = s < n_slices
    start = tl.load(off_ptr + s, mask=smask, other=0)
    width = (tl.load(off_ptr + s + 1, mask=smask, other=0) - start) // C
    slot = s * C + lane
    # Each lane's row length masks every load: padding is never read, and the
    # value load does not wait for the column load.
    row_len = tl.load(len_ptr + slot, mask=smask, other=0)
    steps = tl.max(tl.max(width, 0), 0)
    row = tl.load(perm_ptr + slot, mask=smask, other=-1)
    if COMPLEX:
        re = tl.zeros((GROUP, C), dtype=ACC)
        im = tl.zeros((GROUP, C), dtype=ACC)
        if PAIRED:
            comp = tl.arange(0, 2)[None, None, :]
            sign = tl.where(comp == 0, 1.0, -1.0).to(ACC)
            for j in range(0, steps):
                m = j < row_len
                pos = start + j * C + lane
                col = tl.load(c_ptr + pos, mask=m, other=0)
                m3 = m[:, :, None]
                v = tl.load(v_ptr + 2 * pos[:, :, None] + comp, mask=m3, other=0.0)
                xv = tl.load(x_ptr + 2 * col[:, :, None] + comp, mask=m3, other=0.0)
                xs = tl.load(x_ptr + 2 * col[:, :, None] + (1 - comp), mask=m3, other=0.0)
                re += tl.sum(v * xv * sign, axis=2)
                im += tl.sum(v * xs, axis=2)
        else:
            for j in range(0, steps):
                m = j < row_len
                pos = start + j * C + lane
                col = tl.load(c_ptr + pos, mask=m, other=0)
                vr = tl.load(v_ptr + 2 * pos, mask=m, other=0.0)
                vi = tl.load(v_ptr + 2 * pos + 1, mask=m, other=0.0)
                xr = tl.load(x_ptr + 2 * col, mask=m, other=0.0)
                xi = tl.load(x_ptr + 2 * col + 1, mask=m, other=0.0)
                re += vr * xr - vi * xi
                im += vr * xi + vi * xr
        tl.store(y_ptr + 2 * row, re, mask=row >= 0)
        tl.store(y_ptr + 2 * row + 1, im, mask=row >= 0)
    else:
        acc = tl.zeros((GROUP, C), dtype=ACC)
        for j in range(0, steps):
            m = j < row_len
            pos = start + j * C + lane
            col = tl.load(c_ptr + pos, mask=m, other=0)
            v = tl.load(v_ptr + pos, mask=m, other=0).to(ACC)
            acc += v * tl.load(x_ptr + col, mask=m, other=0).to(ACC)
        tl.store(y_ptr + row, acc.to(y_ptr.dtype.element_ty), mask=row >= 0)


@triton.jit
def _spmv_sell_sigma_wide_kernel(
    y_ptr, v_ptr, c_ptr, off_ptr, len_ptr, perm_ptr, x_ptr, n_slices,
    C: tl.constexpr, K: tl.constexpr, ACC: tl.constexpr,
):
    s = tl.program_id(0)
    lane = tl.arange(0, C)[None, :]
    wide = tl.arange(0, K)[:, None]
    smask = s < n_slices
    start = tl.load(off_ptr + s, mask=smask, other=0)
    width = (tl.load(off_ptr + s + 1, mask=smask, other=0) - start) // C
    row_len = tl.load(len_ptr + s * C + lane, mask=smask, other=0)
    row = tl.load(perm_ptr + s * C + lane, mask=smask, other=-1)
    acc = tl.zeros((K, C), dtype=ACC)
    for j in range(0, width, K):
        m = (j + wide) < row_len
        pos = start + (j + wide) * C + lane
        col = tl.load(c_ptr + pos, mask=m, other=0)
        value = tl.load(v_ptr + pos, mask=m, other=0).to(ACC)
        vector = tl.load(x_ptr + col, mask=m, other=0).to(ACC)
        acc += value * vector
    tl.store(y_ptr + row, tl.sum(acc, axis=0)[None, :].to(y_ptr.dtype.element_ty), mask=row >= 0)


def _refresh_sell_sigma(p):
    versions = tuple(_tensor_version(t) for t in (p.values, p.cols, p.offsets))
    if versions == p.versions:
        return
    if versions[1:] != p.versions[1:]:
        _install_sell_sigma(p)  # structure changed: rebuild the layout
        return
    if p.n_values:
        p.sell_values.copy_(p.values.index_select(0, p.gather))
    p.versions = versions


def spmv_sell_prepared(prepared, x, *, out=None, return_time=False):
    """Run a ``prepare_spmv_sell`` plan: check ``x`` / ``out``, launch one kernel."""
    if not isinstance(prepared, PreparedSellSpmv):
        raise TypeError("prepared must be a PreparedSellSpmv from prepare_spmv_sell")
    p = prepared
    if not torch.is_tensor(x):
        raise TypeError("x must be a torch.Tensor")
    if x.ndim != 1:
        raise ValueError("x must be a 1D tensor")
    if x.dtype != p.value_dtype:
        raise TypeError("x dtype must match values dtype")
    if x.numel() != p.n_cols or x.device != p.device:
        raise ValueError(f"x must have n_cols={p.n_cols} entries on the plan's device")
    if out is not None and (
        out.shape != (p.n_rows,) or out.dtype != p.out_dtype or out.device != x.device
    ):
        raise ValueError("out shape/dtype/device must match the prepared SELL result")
    if p.route == "call":
        return flagsparse_spmv_sell(
            p.values, p.cols, p.offsets, x, p.shape, slice_size=p.slice_size,
            out=out, out_dtype=p.out_dtype, validate=False, return_time=return_time,
        )
    _refresh_sell_sigma(p)
    complex_values = p.sell_values.is_complex()
    x = (x.resolve_conj() if complex_values else x).contiguous()
    # A strided out cannot be indexed densely by the kernel; fill it by a copy.
    y = out if out is not None and out.is_contiguous() else torch.empty(
        p.n_rows, dtype=p.out_dtype, device=x.device
    )
    t0 = None
    if return_time:
        _ACCEL.synchronize()
        t0 = time.perf_counter()
    if p.x_order is not None:
        x = torch.index_select(x, 0, p.x_order, out=p.x_buf)
    contiguous_output = p.x_order is not None and os.environ.get(
        "FLAGSPARSE_SELL_CONTIGUOUS_OUTPUT", "0"
    ) == "1"
    kernel_y = p.y_buf if contiguous_output else y
    if p.n_slices:
        tensors = (
            torch.view_as_real(kernel_y) if complex_values else kernel_y,
            p.kernel_values, p.sell_cols, p.sell_offsets, p.row_len,
            p.kernel_perm if contiguous_output else p.perm,
            torch.view_as_real(x) if complex_values else x,
        )
        use_wide = p.k > 1  # prepare keeps k == 1 for complex / grouped plans
        if _CUDA_DIRECT_LAUNCH:
            _launch_musa_spmv(
                _spmv_sell_sigma_wide_kernel if use_wide else _spmv_sell_sigma_kernel,
                p.grid, tensors, (p.n_slices,),
                (p.c, p.k, p.constants[2]) if use_wide else p.constants,
                num_warps=p.num_warps, cuda=True,
            )
        else:
            if use_wide:
                _spmv_sell_sigma_wide_kernel[p.grid](
                    *tensors, p.n_slices, p.c, p.k, p.constants[2], num_warps=p.num_warps
                )
            else:
                _spmv_sell_sigma_kernel[p.grid](
                    *tensors, p.n_slices, *p.constants, num_warps=p.num_warps
                )
    if contiguous_output:
        torch.index_select(kernel_y, 0, p.y_order, out=y)
    if out is not None and y is not out:
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
        if (_is_ascend_runtime() or _is_mthreads_runtime()) and data.is_complex():
            # torch_npu's IndexPut implementation rejects complex tensors.  Store
            # the two real planes instead; IndexCopy supports the corresponding
            # floating dtype and view_as_complex restores the SELL value layout.
            parts = torch.zeros(
                (total, 2),
                dtype=torch.view_as_real(data).dtype,
                device=device,
            )
            parts.index_copy_(0, pos, torch.view_as_real(data))
            values = torch.view_as_complex(parts)
        else:
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
