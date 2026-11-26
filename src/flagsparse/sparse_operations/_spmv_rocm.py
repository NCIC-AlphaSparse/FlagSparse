# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Row-owned SpMV with coalesced matrix loads on wave64 devices."""

import torch
import triton
import triton.language as tl

from ._rocm_launch import can_launch, launch
from ._rocm_sparse_cache import TensorPlanCache

_PLANS = TensorPlanCache(maxsize=4)


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
        xoff = col * STRIDE_X
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


def _prepare(values, indices, ptr, shape, fmt, complex_rhs, *, tile=None):
    from . import _spmv_csr_transpose

    if fmt == "coo":
        # The two index arrays here are the output row and the RHS index.
        row, col = indices, ptr
        order = torch.argsort(row, stable=True)
        cols = col.index_select(0, order)
        lengths = torch.bincount(row.long(), minlength=shape[0])
        if lengths.numel() != shape[0]:
            return ()
        ptr = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    elif fmt == "transpose":
        cols, ptr, order, *_ = _spmv_csr_transpose.prepare(indices, ptr, shape)
        lengths = ptr[1:] - ptr[:-1]
    else:
        cols, order = indices, None
        lengths = ptr[1:] - ptr[:-1]
    n = lengths.numel()
    rows_per_slice, slots, warps = tile or (64, 8, 4)
    if tile is None and complex_rhs and values.dtype != torch.float64:
        maximum = int(lengths.max()) if n else 0
        mean = values.numel() / max(1, n)
        if maximum > 512:
            rows_per_slice, slots, warps = 16, 16, 4
        elif mean >= 32:
            rows_per_slice, slots, warps = 16, 8, 2
        elif maximum > 64 and mean >= 16:
            rows_per_slice, slots, warps = 32, 4, 2
        elif maximum <= 16 or mean >= 8:
            slots, warps = 2, 2
        else:
            slots, warps = 8, 2
    rows = torch.argsort(lengths, descending=True, stable=True).int()
    widths = lengths.index_select(0, rows[::rows_per_slice].long()).long()
    offsets = torch.cat((widths.new_zeros(1), (widths * rows_per_slice).cumsum(0)))
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
            n, order is not None, v.is_complex(), rows_per_slice, 8, num_warps=4,
        )
    packed_arg = torch.view_as_real(packed) if packed.is_complex() else packed
    return (packed_arg, columns, offsets, rows), (rows_per_slice, slots, warps), {}


def bind_real_csr(prepared, x, out, *, acc):
    """Bind a wave64 row-tile launch for large CSR matrices with medium rows."""
    values = prepared.data
    cols, ptr = prepared.kernel_indices, prepared.kernel_indptr
    m, n = prepared.shape
    maximum = prepared.max_row_nnz
    if (
        prepared.transpose
        or prepared.alg_requested != "auto"
        or prepared.config_source == "explicit"
        or prepared.backend_caps.arch != "gfx936"
        or values.dtype not in (torch.float32, torch.float64)
        or cols.dtype != torch.int32 or ptr.dtype != torch.int32
        or not 1_000_000 <= values.numel() < 2**30
        or max(m, n) >= 2**30
        or not m or not 9 <= values.numel() / m <= 32
        or maximum > 128
        or values.is_neg() or x.is_neg()
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    mean = values.numel() / m
    if mean >= 20:
        tile = (64, 8, 1) if maximum <= 32 else (64, 16, 4)
    elif maximum <= 24:
        tile = (32, 4, 2)
    elif maximum <= 40:
        tile = (16, 8, 1)
    else:
        tile = (64, 16, 4)
    sources = (values, cols, ptr)
    stream = torch._C._cuda_getCurrentRawStream(values.device.index)
    options = (prepared.shape, "csr", False, tile, stream)
    plan = _PLANS.get(sources, options)
    if plan is None:
        plan = _prepare(*sources, prepared.shape, "csr", False, tile=tile)
        _PLANS.put(sources, options, plan)
    if not plan:
        return None
    matrix, (rows, slots, warps), _ = plan
    grid = (matrix[2].numel() - 1,)

    def run():
        _multiply[grid](
            *matrix, x, out, m, x.stride(0), False, False, False, acc,
            rows, slots, num_warps=warps, enable_fp_fusion=acc == tl.float32,
        )
        return out

    return run


def multiply(values, indices, ptr, x, shape, fmt, dtype, *, conjugate=False, out=None):
    """Return a fresh result, or None if the compact CSR route is preferable."""
    if (values.device.type != "cuda" or values.device != x.device or values.device != indices.device
            or values.device != ptr.device
            or max(*shape, values.numel(), x.numel() * x.stride(0)) >= 2**30):
        return None
    if torch._C._cuda_getDevice() != values.device.index:
        with torch.cuda.device(values.device):
            return multiply(values, indices, ptr, x, shape, fmt, dtype,
                            conjugate=conjugate, out=out)
    if torch.cuda.is_current_stream_capturing():
        return None
    sources = values, indices, ptr
    complex_rhs = x.is_complex()
    stream = torch._C._cuda_getCurrentRawStream(values.device.index)
    options = (shape, fmt, complex_rhs, stream)
    plan = _PLANS.get(sources, options)
    if plan is None:
        plan = _prepare(values, indices, ptr, shape, fmt, complex_rhs)
        _PLANS.put(sources, options, plan)
    if not plan:
        return None
    matrix, tile, runners = plan
    offsets, rows = matrix[2:]
    result = out
    if (out is None or out.shape != rows.shape or out.dtype != dtype or out.device != x.device
            or not out.is_contiguous() or out.is_conj() or out.is_neg()
            or any(torch._C._overlaps(out, t) for t in (*sources, x))):
        result = torch.empty(rows.numel(), dtype=dtype, device=x.device)
    if rows.numel():
        x = x.resolve_conj().resolve_neg()
        if dtype == torch.int32:
            acc = tl.int32
        elif values.dtype == x.dtype and values.dtype in (torch.float16, torch.int8):
            acc = tl.float32
        else:
            acc = tl.float64
        stride = x.stride(0)
        key = (x.dtype, dtype, stride, conjugate, x.data_ptr() % 16, result.data_ptr() % 16)
        runner = runners.get(key)
        if runner is not None and can_launch(_multiply):
            runner(*matrix, x, result, rows.numel(), stride, stream=stream)
        else:
            x_arg, y_arg = x, result
            if complex_rhs:
                x_arg, y_arg = torch.view_as_real(x), torch.view_as_real(result)
            rows_per_slice, slots, warps = tile
            runner = launch(
                _multiply, offsets.numel() - 1, (*matrix, x_arg, y_arg),
                (rows.numel(), stride),
                (values.is_complex(), complex_rhs, conjugate, acc, rows_per_slice, slots),
                num_warps=warps,
            )
            if runner is not None and len(runners) < 8:
                runners[key] = runner
    return result
