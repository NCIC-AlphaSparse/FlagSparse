# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Row-owned SpMV with coalesced matrix loads on wave64 devices."""

import torch
import triton
import triton.language as tl

from ._rocm_launch import launch
from ._rocm_sparse_cache import TensorPlanCache

_PLANS = TensorPlanCache(maxsize=4)
_SLICE = 64


@triton.jit
def _pack(V, I, P, ORDER, ROWS, OFF, VAL, COL, M, INDIRECT: tl.constexpr,
          COMPLEX: tl.constexpr, R: tl.constexpr, K: tl.constexpr):
    group = tl.program_id(0)
    lanes = tl.arange(0, R)
    active = group * R + lanes < M
    row = tl.load(ROWS + group * R + lanes, active, 0)
    start = tl.load(P + row, active, 0)
    end = tl.load(P + row + 1, active, 0)
    first = tl.load(OFF + group)
    width = (tl.load(OFF + group + 1) - first) // R
    for base in range(0, width, K):
        slot = base + tl.arange(0, K)
        pos = start[None, :] + slot[:, None]
        valid = active[None, :] & (pos < end[None, :])
        source = tl.load(ORDER + pos, valid, 0) if INDIRECT else pos
        col = tl.load(I + pos, valid, -1)
        dst = first + slot[:, None] * R + lanes[None, :]
        keep = slot[:, None] < width
        tl.store(COL + dst, col, keep)
        if COMPLEX:
            tl.store(VAL + 2 * dst, tl.load(V + 2 * source, valid, 0.), keep)
            tl.store(VAL + 2 * dst + 1, tl.load(V + 2 * source + 1, valid, 0.), keep)
        else:
            tl.store(VAL + dst, tl.load(V + source, valid, 0), keep)


@triton.jit
def _multiply(V, I, OFF, ROWS, X, Y, M, STRIDE_X, CV: tl.constexpr,
              CX: tl.constexpr, CONJ: tl.constexpr, ACC: tl.constexpr,
              R: tl.constexpr, K: tl.constexpr):
    group = tl.program_id(0)
    lanes = tl.arange(0, R)
    first = tl.load(OFF + group)
    width = (tl.load(OFF + group + 1) - first) // R
    real = tl.zeros((K, R), ACC)
    imag = tl.zeros((K, R), ACC)
    for base in range(0, width, K):
        slot = base + tl.arange(0, K)
        pos = first + slot[:, None] * R + lanes[None, :]
        col = tl.load(I + pos, slot[:, None] < width, -1)
        active = col >= 0
        xoff = col.to(tl.int64) * STRIDE_X
        if CX:
            xr = tl.load(X + 2 * xoff, active, 0.).to(ACC)
            xi = tl.load(X + 2 * xoff + 1, active, 0.).to(ACC)
        else:
            xr = tl.load(X + xoff, active, 0).to(ACC)
        if CV:
            vr = tl.load(V + 2 * pos, active, 0.).to(ACC)
            vi = tl.load(V + 2 * pos + 1, active, 0.).to(ACC)
            if CONJ:
                vi = -vi
            real += vr * xr - vi * xi
            imag += vr * xi + vi * xr
        else:
            v = tl.load(V + pos, active, 0).to(ACC)
            real += v * xr
            if CX:
                imag += v * xi
    valid_rows = group * R + lanes < M
    output = tl.load(ROWS + group * R + lanes, valid_rows, 0)
    if CX:
        tl.store(Y + 2 * output, tl.sum(real, 0), valid_rows)
        tl.store(Y + 2 * output + 1, tl.sum(imag, 0), valid_rows)
    else:
        tl.store(Y + output, tl.sum(real, 0), valid_rows)


def _prepare(values, indices, ptr, shape, fmt):
    from . import _spmv_csr_transpose

    if fmt == "coo":
        # The two index arrays here are the output row and the RHS index.
        row, col = indices, ptr
        order = torch.argsort(row, stable=True)
        cols = col.index_select(0, order)
        lengths = torch.bincount(row.long(), minlength=shape[0])
        ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    elif fmt == "transpose":
        cols, ptr, order, *_ = _spmv_csr_transpose.prepare(indices, ptr, shape)
        lengths = ptr[1:] - ptr[:-1]
    else:
        cols, order = indices, None
        lengths = ptr[1:] - ptr[:-1]
    n = lengths.numel()
    rows = torch.argsort(lengths, descending=True, stable=True).int()
    widths = lengths.index_select(0, rows[::_SLICE].long()).long()
    offsets = torch.cat((widths.new_zeros(1), (widths * _SLICE).cumsum(0)))
    size = int(offsets[-1])
    # Pathological long rows can waste a whole slice of padding.
    if size > max(2 * values.numel(), 4096) or size >= 2**30:
        return ()
    offsets = offsets.int()
    packed = torch.empty(size, device=values.device, dtype=values.dtype)
    columns = torch.empty(size, device=values.device, dtype=cols.dtype)
    v = values.resolve_conj().resolve_neg().contiguous()
    if n:
        _pack[(widths.numel(),)](
            torch.view_as_real(v) if v.is_complex() else v, cols.contiguous(), ptr.contiguous(),
            order if order is not None else ptr, rows, offsets,
            torch.view_as_real(packed) if v.is_complex() else packed, columns,
            n, order is not None, v.is_complex(), _SLICE, 8, num_warps=4,
        )
    return packed, columns, offsets, rows


def multiply(values, indices, ptr, x, shape, fmt, dtype, *, conjugate=False, out=None):
    """Return a fresh result, or None if the compact CSR route is preferable."""
    if max(*shape, values.numel()) >= 2**30 or torch.cuda.is_current_stream_capturing():
        return None
    sources = values, indices, ptr
    options = (shape, fmt, torch.cuda.current_stream(values.device).cuda_stream)
    plan = _PLANS.get(sources, options)
    if plan is None:
        plan = _prepare(values, indices, ptr, shape, fmt)
        _PLANS.put(sources, options, plan)
    if not plan:
        return None
    packed, cols, offsets, rows = plan
    result = out
    if (out is None or not out.is_contiguous() or out.is_conj() or out.is_neg()
            or any(torch._C._overlaps(out, t) for t in (*sources, x))):
        result = torch.empty(rows.numel(), dtype=dtype, device=x.device)
    if rows.numel():
        x = x.resolve_conj().resolve_neg()
        acc = tl.int32 if dtype == torch.int32 else tl.float64
        tensors = (
            torch.view_as_real(packed) if packed.is_complex() else packed,
            cols, offsets, rows, torch.view_as_real(x) if x.is_complex() else x,
            torch.view_as_real(result) if x.is_complex() else result,
        )
        launch(_multiply, offsets.numel() - 1, tensors, (rows.numel(), x.stride(0)),
               (packed.is_complex(), x.is_complex(), conjugate, acc, _SLICE, 8))
    return result
