# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CUDA row-owned sparse products; cache topology, never numerical outputs."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._musa_spmv_launch import fast_path_allowed, launch
from ._rocm_sparse_cache import TensorPlanCache


_TOPOLOGY = TensorPlanCache(maxsize=8)
_ROWS = TensorPlanCache(maxsize=8)
_VALUES = TensorPlanCache(maxsize=8)
_SPMV_HOT = None
_COO_HOT = None
_CSC_HOT = None
_RAW_STREAM = None


def _stream(device):
    global _RAW_STREAM
    if _RAW_STREAM is None:
        from torch._C import _cuda_getCurrentRawStream

        _RAW_STREAM = _cuda_getCurrentRawStream
    return _RAW_STREAM(device.index)


def coo_topology(row, col, n_rows):
    """CSR indices and a permutation into the original, live COO values."""
    sources, options = (row, col), (n_rows, _stream(row.device))
    state = _TOPOLOGY.get(sources, options)
    if state is None:
        capturing = torch.cuda.is_current_stream_capturing()
        sorted_ = not capturing and (
            row.numel() < 2 or bool(torch.all(row[1:] >= row[:-1]))
        )
        order = None if sorted_ else torch.argsort(row, stable=True)
        columns = col if order is None else col.index_select(0, order).contiguous()
        if capturing:
            dtype = torch.int32 if col.numel() < 2**31 else torch.int64
            counts = torch.zeros(n_rows + 1, device=row.device, dtype=dtype)
            if row.numel():
                _coo_counts[((row.numel() + 255) // 256,)](
                    row, counts, row.numel(), 256
                )
            ptr = counts.cumsum(0, dtype=dtype)
        else:
            counts = torch.bincount(row.long(), minlength=n_rows)
            ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
            if col.numel() < 2**31:
                ptr = ptr.to(torch.int32)
        state = columns, ptr, order
        if not capturing:
            _TOPOLOGY.put(sources, options, state)
    return state


@triton.jit
def _coo_counts(ROWS, COUNTS, N, B: tl.constexpr):
    pos = tl.program_id(0) * B + tl.arange(0, B)
    row = tl.load(ROWS + pos, pos < N, 0)
    tl.atomic_add(COUNTS + row + 1, 1, pos < N, sem="relaxed")


def _row_order(ptr, n_rows):
    options = (n_rows, _stream(ptr.device))
    rows = _ROWS.get((ptr,), options)
    if rows is None:
        rows = torch.argsort(ptr[1:] - ptr[:-1], stable=True).to(ptr.dtype)
        if not torch.cuda.is_current_stream_capturing():
            _ROWS.put((ptr,), options, rows)
    return rows


def _ordered_values(values, order, dtype=None):
    dtype = dtype or values.dtype
    if order is None and dtype == values.dtype:
        return values
    if torch.cuda.is_current_stream_capturing():
        result = values if order is None else values.index_select(0, order)
        return result.to(dtype).contiguous()
    sources = (values,) if order is None else (values, order)
    options = (_stream(values.device), dtype)
    result = _VALUES.get(sources, options)
    if result is None:
        result = values if order is None else values.index_select(0, order)
        result = result.to(dtype).contiguous()
        _VALUES.put(sources, options, result)
    return result


def transpose_topology(cols, ptr, shape):
    sources, options = (cols, ptr), (tuple(shape), _stream(cols.device))
    state = _TOPOLOGY.get(sources, options)
    if state is None:
        rows = torch.repeat_interleave(
            torch.arange(shape[0], device=ptr.device, dtype=cols.dtype),
            (ptr[1:] - ptr[:-1]).long(),
            output_size=cols.numel(),
        )
        state = coo_topology(cols, rows, shape[1])
        if not torch.cuda.is_current_stream_capturing():
            _TOPOLOGY.put(sources, options, state)
    return state


@triton.jit
def _two_sum(ah, al, bh, bl):
    high = libdevice.add_rn(ah, bh)
    z = libdevice.sub_rn(high, ah)
    error = libdevice.add_rn(
        libdevice.sub_rn(ah, libdevice.sub_rn(high, z)), libdevice.sub_rn(bh, z)
    )
    return high, libdevice.add_rn(libdevice.add_rn(al, bl), error)


@triton.jit
def _spmv(
    V,
    COLS,
    P,
    ORDER,
    X,
    Y,
    M: tl.constexpr,
    VS: tl.constexpr,
    XS: tl.constexpr,
    YS: tl.constexpr,
    CV: tl.constexpr,
    CX: tl.constexpr,
    CONJ: tl.constexpr,
    INDIRECT: tl.constexpr,
    ACC: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    COMPENSATED: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    valid = rows < M
    start = tl.load(P + rows, valid, 0)
    end = tl.load(P + rows + 1, valid, 0)
    lanes = tl.arange(0, K)
    real = tl.zeros((R, K), ACC)
    imag = tl.zeros((R, K), ACC)
    low = tl.zeros((R, K), tl.float32)
    exceptional = tl.full((R, K), False, tl.int1)
    for base in range(0, tl.max(end - start, 0), K):
        pos = start[:, None] + base + lanes[None, :]
        mask = valid[:, None] & (pos < end[:, None])
        src = tl.load(ORDER + pos, mask, 0) if INDIRECT else pos
        col = tl.load(COLS + pos, mask, 0)
        if CV:
            vr = tl.load(V + 2 * src * VS, mask, 0).to(ACC)
            vi = tl.load(V + 2 * src * VS + 1, mask, 0).to(ACC)
            if CONJ:
                vi = -vi
        else:
            vr = tl.load(V + src * VS, mask, 0).to(ACC)
        if CX:
            xr = tl.load(X + 2 * col * XS, mask, 0).to(ACC)
            xi = tl.load(X + 2 * col * XS + 1, mask, 0).to(ACC)
            if CV:
                real += vr * xr - vi * xi
                imag += vr * xi + vi * xr
            else:
                real += vr * xr
                imag += vr * xi
        else:
            xr = tl.load(X + col * XS, mask, 0).to(ACC)
            if COMPENSATED:
                product = libdevice.mul_rn(vr, xr)
                error = tl.fma(vr, xr, -product)
                real, low = _two_sum(real, low, product, error)
                exceptional |= mask & (
                    (tl.abs(product) > 3.402823466e38)
                    | (product != product)
                    | ((tl.abs(product) < 1.175494351e-38) & (vr != 0) & (xr != 0))
                    | (tl.abs(real) > 3.402823466e38)
                    | (real != real)
                )
            else:
                real += vr * xr
    if CX:
        tl.store(Y + 2 * rows * YS, tl.sum(real, 1), valid)
        tl.store(Y + 2 * rows * YS + 1, tl.sum(imag, 1), valid)
    else:
        if COMPENSATED:
            high, correction = tl.reduce((real, low), 1, _two_sum)
            result = high + correction
            bad = tl.sum(tl.sum(exceptional.to(tl.int32), 1), 0) > 0
            bad |= (
                tl.sum(
                    ((tl.abs(result) > 3.402823466e38) | (result != result)).to(
                        tl.int32
                    ),
                    0,
                )
                > 0
            )
            if bad:
                # Rare overflow/subnormal products retain the original FP64
                # contract, including cancellation before the final FP32 cast.
                wide = tl.zeros((R, K), tl.float64)
                for base in range(0, tl.max(end - start, 0), K):
                    pos = start[:, None] + base + lanes[None, :]
                    mask = valid[:, None] & (pos < end[:, None])
                    src = tl.load(ORDER + pos, mask, 0) if INDIRECT else pos
                    col = tl.load(COLS + pos, mask, 0)
                    a = tl.load(V + src * VS, mask, 0).to(tl.float64)
                    b = tl.load(X + col * XS, mask, 0).to(tl.float64)
                    wide += a * b
                result = tl.sum(wide, 1).to(tl.float32)
            tl.store(Y + rows * YS, result, valid)
        else:
            tl.store(Y + rows * YS, tl.sum(real, 1), valid)


def csr_spmv(
    values,
    cols,
    ptr,
    x,
    n_rows,
    *,
    out=None,
    out_dtype=None,
    order=None,
    conjugate=False,
    checked_out=False,
):
    global _SPMV_HOT
    if values.device.index != torch._C._cuda_getDevice():
        with torch.cuda.device(values.device):
            return csr_spmv(
                values,
                cols,
                ptr,
                x,
                n_rows,
                out=out,
                out_dtype=out_dtype,
                order=order,
                conjugate=conjugate,
                checked_out=checked_out,
            )
    if out is not None and (
        out.is_conj()
        or out.is_neg()
        or (
            not checked_out
            and any(torch._C._overlaps(out, t) for t in (values, cols, ptr, x))
        )
    ):
        result = csr_spmv(
            values,
            cols,
            ptr,
            x,
            n_rows,
            out_dtype=out.dtype,
            order=order,
            conjugate=conjugate,
        )
        out.copy_(result)
        return out
    dtype = out.dtype if out is not None else (out_dtype or x.dtype)
    hot = _SPMV_HOT
    sources = (values, cols, ptr, x)
    options = (n_rows, dtype, id(order), conjugate)
    if (
        hot is not None
        and hot[0] == options
        and all(a is b for a, b in zip(sources, hot[1]))
        and hot[2] == _spmv_signature(sources)
        and fast_path_allowed(hot[3].kernel)
    ):
        bound, inputs, constants, addresses = hot[3:7]
        y = (
            out
            if out is not None
            else torch.empty(n_rows, dtype=dtype, device=x.device)
        )
        if y.stride(0) == constants[3] and y.data_ptr() % 16 == hot[7]:
            _launch_hot(bound, inputs, constants, addresses, y)
            return y
    y = out if out is not None else torch.empty(n_rows, dtype=dtype, device=x.device)
    if not n_rows:
        return y
    values, x = values.resolve_conj().resolve_neg(), x.resolve_conj().resolve_neg()
    mean = values.numel() / max(1, n_rows)
    k = 4 if mean <= 4 else 8 if mean <= 12 else 16
    r = 512 // k
    acc = {
        torch.float64: tl.float64,
        torch.complex128: tl.float64,
        torch.int32: tl.int32,
    }.get(dtype, tl.float32)
    compensated = values.dtype == x.dtype == dtype == torch.float32
    cv, cx = values.is_complex(), x.is_complex()
    tensors = (
        torch.view_as_real(values) if cv else values,
        cols,
        ptr,
        ptr if order is None else order,
        torch.view_as_real(x) if cx else x,
        torch.view_as_real(y) if cx else y,
    )
    constants = (
        n_rows,
        values.stride(0),
        x.stride(0),
        y.stride(0),
        cv,
        cx,
        conjugate,
        order is not None,
        acc,
        r,
        k,
        compensated,
    )
    bound = launch(
        _spmv,
        ((n_rows + r - 1) // r,),
        tensors,
        (),
        constants,
        num_warps=4,
        num_stages=1,
        cuda=True,
    )
    if (
        bound is not None
        and not torch.cuda.is_current_stream_capturing()
        and all(not t.is_conj() and not t.is_neg() for t in sources)
    ):
        _SPMV_HOT = (
            options,
            sources,
            _spmv_signature(sources),
            bound,
            tensors[:-1],
            constants,
            tuple(t.data_ptr() for t in tensors[:-1]),
            y.data_ptr() % 16,
        )
    return y


def _launch_hot(bound, inputs, constants, addresses, result):
    raw = bound.cuda_entry
    if raw is not None:
        entry, function, metadata = raw
        try:
            entry(
                *bound.grid,
                _stream(result.device),
                function,
                False,
                False,
                None,
                None,
                metadata,
                None,
                None,
                None,
                *addresses,
                result.data_ptr(),
                *constants,
            )
            return
        except TypeError:
            bound.cuda_entry = None
    target = torch.view_as_real(result) if result.is_complex() else result
    bound(
        (*inputs, target), (), constants, (*addresses, result.data_ptr()), checked=True
    )


def _spmv_signature(sources):
    a, b, c, d = sources
    return (
        (a.data_ptr(), a.dtype, a.device, a.shape, a.stride(), a.is_conj(), a.is_neg()),
        (b.data_ptr(), b.dtype, b.device, b.shape, b.stride(), b.is_conj(), b.is_neg()),
        (c.data_ptr(), c.dtype, c.device, c.shape, c.stride(), c.is_conj(), c.is_neg()),
        (d.data_ptr(), d.dtype, d.device, d.shape, d.stride(), d.is_conj(), d.is_neg()),
    )


def remember_coo(prepared, x, dtype):
    global _COO_HOT
    if torch.cuda.is_current_stream_capturing():
        return
    hot = _SPMV_HOT
    sources = (prepared.data_non, prepared.row_non, prepared.col_non, x)
    signature = _coo_signature(sources)
    if (
        hot is not None
        and signature is not None
        and hot[1][0] is sources[0]
        and hot[1][-1] is x
        and all(not t.is_conj() and not t.is_neg() for t in sources)
    ):
        _COO_HOT = (
            prepared,
            sources,
            signature,
            dtype,
            hot[3],
            hot[4],
            hot[5],
            _stream(x.device),
            hot[6],
        )


def try_coo(prepared, x, out_dtype):
    hot = _COO_HOT
    if hot is None or hot[0] is not prepared or x is not hot[1][-1]:
        return None
    sources = (prepared.data_non, prepared.row_non, prepared.col_non, x)
    if (
        sources[0] is not hot[1][0]
        or sources[1] is not hot[1][1]
        or sources[2] is not hot[1][2]
        or _coo_signature(sources) != hot[2]
        or prepared.n_rows != hot[6][0]
        or x.numel() != prepared.n_cols
        or (out_dtype is not None and out_dtype != hot[3])
        or (
            out_dtype is None
            and hot[3] != (torch.int32 if x.dtype == torch.int8 else x.dtype)
        )
        or _stream(x.device) != hot[7]
        or not fast_path_allowed(hot[4].kernel)
        or x.device.index != torch._C._cuda_getDevice()
    ):
        return None
    result = torch.empty(prepared.n_rows, dtype=hot[3], device=x.device)
    _launch_hot(hot[4], hot[5], hot[6], hot[8], result)
    return result


def _coo_signature(sources):
    # Numerical operands stay live; topology mutation must invalidate the plan.
    # Spell out the four operands to avoid generator/tuple conversions per call.
    a, b, c, d = sources
    try:
        return (
            (a.data_ptr(), a.dtype, a.device, a.shape, a.stride(), a.is_conj(), a.is_neg()),
            (b._version, b.data_ptr(), b.dtype, b.device, b.shape, b.stride()),
            (c._version, c.data_ptr(), c.dtype, c.device, c.shape, c.stride()),
            (d.data_ptr(), d.dtype, d.device, d.shape, d.stride(), d.is_conj(), d.is_neg()),
        )
    except RuntimeError:
        return None


def remember_csc(prepared, x, op):
    global _CSC_HOT
    if torch.cuda.is_current_stream_capturing():
        return
    hot = _SPMV_HOT
    sources = (prepared.data, prepared.kernel_indices, prepared.kernel_indptr, x)
    signature = _spmv_signature(sources)
    if (
        hot is not None
        and signature is not None
        and all(a is b for a, b in zip(sources, hot[1]))
        and all(not t.is_conj() and not t.is_neg() for t in sources)
    ):
        _CSC_HOT = (prepared, sources, signature, op, *hot[3:])


def try_csc(prepared, x, op):
    hot = _CSC_HOT
    if hot is None or hot[0] is not prepared or op != hot[3] or x is not hot[1][-1]:
        return None
    sources = (prepared.data, prepared.kernel_indices, prepared.kernel_indptr, x)
    if (
        not all(a is b for a, b in zip(sources, hot[1]))
        or _spmv_signature(sources) != hot[2]
        or prepared.n_cols != hot[6][0]
        or x.numel() != prepared.n_rows
        or not fast_path_allowed(hot[4].kernel)
        or x.device.index != torch._C._cuda_getDevice()
    ):
        return None
    result = torch.empty(prepared.n_cols, dtype=prepared.data.dtype, device=x.device)
    raw = hot[4].cuda_entry
    if raw is not None:
        entry, function, metadata = raw
        try:
            entry(
                *hot[4].grid,
                _stream(x.device),
                function,
                False,
                False,
                None,
                None,
                metadata,
                None,
                None,
                None,
                *hot[7],
                result.data_ptr(),
                *hot[6],
            )
            return result
        except TypeError:
            hot[4].cuda_entry = None
    target = torch.view_as_real(result) if result.is_complex() else result
    hot[4]((*hot[5], target), (), hot[6], (*hot[7], result.data_ptr()), checked=True)
    return result


@triton.jit
def _spmm(
    V,
    COLS,
    P,
    ORDER,
    ROWS,
    B,
    C,
    M: tl.constexpr,
    N: tl.constexpr,
    VS: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    CM: tl.constexpr,
    CN: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    INDIRECT: tl.constexpr,
    ACC: tl.constexpr,
    R: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
):
    ids = tl.program_id(0) * R + tl.arange(0, R)
    valid = ids < M
    rows = tl.load(ROWS + ids, valid, 0)
    start = tl.load(P + rows, valid, 0)
    end = tl.load(P + rows + 1, valid, 0)
    ns = tl.program_id(1) * T + tl.arange(0, T)
    ks = tl.arange(0, K)
    parts = tl.arange(0, 2)
    real = tl.zeros((R, K, T), ACC)
    imag = tl.zeros((R, K, T), ACC)
    for base in range(0, tl.max(end - start, 0), K):
        pos = start[:, None] + base + ks[None, :]
        mask = valid[:, None] & (pos < end[:, None])
        src = tl.load(ORDER + pos, mask, 0) if INDIRECT else pos
        col = tl.load(COLS + pos, mask, 0)
        bm = mask[:, :, None] & (ns[None, None, :] < N)
        bp = col[:, :, None] * BK + ns[None, None, :] * BN
        if COMPLEX:
            pair = tl.load(
                V + 2 * src[:, :, None] * VS + parts[None, None, :], mask[:, :, None], 0
            ).to(ACC)
            vr, vi = tl.split(pair)
            if CONJ:
                vi = -vi
            dense = tl.load(
                B + 2 * bp[:, :, :, None] + parts[None, None, None, :],
                bm[:, :, :, None],
                0,
            ).to(ACC)
            br, bi = tl.split(dense)
            real += vr[:, :, None] * br - vi[:, :, None] * bi
            imag += vr[:, :, None] * bi + vi[:, :, None] * br
        else:
            v = tl.load(V + src * VS, mask, 0).to(ACC)
            b = tl.load(B + bp, bm, 0).to(ACC)
            real += v[:, :, None] * b
    cp = rows[:, None] * CM + ns[None, :] * CN
    mask = valid[:, None] & (ns[None, :] < N)
    if COMPLEX:
        result = tl.join(tl.sum(real, 1), tl.sum(imag, 1))
        tl.store(
            C + 2 * cp[:, :, None] + parts[None, None, :], result, mask[:, :, None]
        )
    else:
        tl.store(C + cp, tl.sum(real, 1), mask)


def csr_spmm(
    values,
    cols,
    ptr,
    b,
    n_rows,
    *,
    out=None,
    out_dtype=None,
    order=None,
    conjugate=False,
):
    if values.device.index != torch._C._cuda_getDevice():
        with torch.cuda.device(values.device):
            return csr_spmm(
                values,
                cols,
                ptr,
                b,
                n_rows,
                out=out,
                out_dtype=out_dtype,
                order=order,
                conjugate=conjugate,
            )
    if out is not None and (
        out.is_conj()
        or out.is_neg()
        or torch._C._overlaps(out, values)
        or torch._C._overlaps(out, b)
    ):
        result = csr_spmm(
            values,
            cols,
            ptr,
            b,
            n_rows,
            out_dtype=out.dtype,
            order=order,
            conjugate=conjugate,
        )
        out.copy_(result)
        return out
    n = b.shape[1]
    dtype = out.dtype if out is not None else (out_dtype or b.dtype)
    c = (
        out
        if out is not None
        else torch.empty((n_rows, n), dtype=dtype, device=b.device)
    )
    if not n_rows or not n:
        return c
    values, b = values.resolve_conj().resolve_neg(), b.resolve_conj().resolve_neg()
    if b.stride(1) != 1 and n >= 16:
        b = b.contiguous()
    # Canonical COO half values contain fp32 duplicate sums. Keep those sums
    # intact and widen half dense loads in registers instead of rounding them.
    value_dtype = (
        torch.float32
        if b.dtype == torch.float16 and values.dtype == torch.float32
        else b.dtype
    )
    values = _ordered_values(values, order, value_dtype)
    order = None
    rows = _row_order(ptr, n_rows)
    complex_ = values.is_complex()
    acc = (
        tl.int32
        if dtype == torch.int32
        else tl.float64
        if dtype in (torch.float64, torch.complex128)
        else tl.float32
    )
    mean = values.numel() / max(1, n_rows)
    if complex_:
        r, k, warps = (4, 4, 4) if mean <= 4 else (4, 2, 2)
        if mean >= 24:
            r, k, warps = 4, 1, 4
    else:
        r, k, warps = (4, 4, 1) if mean <= 4 else (16, 1, 4)
    tile = min(32, triton.next_power_of_2(n))
    tensors = (
        torch.view_as_real(values) if complex_ else values,
        cols,
        ptr,
        ptr if order is None else order,
        rows,
        torch.view_as_real(b) if complex_ else b,
        torch.view_as_real(c) if complex_ else c,
    )
    constants = (
        n_rows,
        n,
        values.stride(0),
        *b.stride(),
        *c.stride(),
        complex_,
        conjugate,
        order is not None,
        acc,
        r,
        k,
        tile,
    )
    launch(
        _spmm,
        (triton.cdiv(n_rows, r), triton.cdiv(n, tile)),
        tensors,
        (),
        constants,
        num_warps=warps,
        num_stages=1,
        cuda=True,
    )
    return c


@triton.jit
def _sddmm(
    COLS,
    P,
    X,
    Y,
    V,
    OUT,
    M: tl.constexpr,
    K: tl.constexpr,
    XM: tl.constexpr,
    XK: tl.constexpr,
    YM: tl.constexpr,
    YK: tl.constexpr,
    ALPHA,
    BETA,
    HAS_IN: tl.constexpr,
    R: tl.constexpr,
    C: tl.constexpr,
    TK: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    valid = rows < M
    start = tl.load(P + rows, valid, 0)
    end = tl.load(P + rows + 1, valid, 0)
    ks = tl.arange(0, TK)
    parts = tl.arange(0, 2)
    xp = 2 * (rows[:, None] * XM + ks[None, :] * XK)
    x = tl.load(
        X + xp[:, :, None] + parts[None, None, :],
        (valid[:, None] & (ks[None, :] < K))[:, :, None],
        0,
    )
    xr, xi = tl.split(x)
    cs = tl.arange(0, C)
    for base in range(0, tl.max(end - start, 0), C):
        pos = start[:, None] + base + cs[None, :]
        mask = valid[:, None] & (pos < end[:, None])
        col = tl.load(COLS + pos, mask, 0)
        yp = 2 * (col[:, :, None] * YM + ks[None, None, :] * YK)
        y = tl.load(
            Y + yp[:, :, :, None] + parts[None, None, None, :],
            (mask[:, :, None] & (ks[None, None, :] < K))[:, :, :, None],
            0,
        )
        yr, yi = tl.split(y)
        re = tl.sum(xr[:, None, :] * yr - xi[:, None, :] * yi, 2) * ALPHA
        im = tl.sum(xr[:, None, :] * yi + xi[:, None, :] * yr, 2) * ALPHA
        if HAS_IN:
            value = tl.load(
                V + 2 * pos[:, :, None] + parts[None, None, :], mask[:, :, None], 0
            )
            vr, vi = tl.split(value)
            re += BETA * vr
            im += BETA * vi
        tl.store(
            OUT + 2 * pos[:, :, None] + parts[None, None, :],
            tl.join(re, im),
            mask[:, :, None],
        )


def sddmm_launch_config(x, y):
    # Row-major A @ B maps B to an N x K view with strided K loads.
    # Issue several sampled columns together instead of serialising every
    # nonzero. The wider tile improved all ten K=32 delivery matrices on CUDA.
    if (
        x.shape[1] == 32
        and x.stride(1) == 1
        and y.stride(0) == 1
        and y.stride(1) > 1
    ):
        return 8, 8, 4
    return 4, 1, 1


def sddmm(prepared, x, y, data, alpha, beta, out, *, config=None):
    rows, columns, warps = sddmm_launch_config(x, y) if config is None else config
    x, y = x.resolve_conj().resolve_neg(), y.resolve_conj().resolve_neg()
    if data is not None:
        data = data.resolve_conj().resolve_neg()
    inputs = (
        prepared.indices,
        prepared.indptr,
        torch.view_as_real(x),
        torch.view_as_real(y),
        torch.view_as_real(out if data is None else data),
        torch.view_as_real(out),
    )
    constants = (prepared.n_rows, x.shape[1], *x.stride(), *y.stride())
    # The two runtime scalars precede the remaining constexpr parameters.
    _sddmm[(triton.cdiv(prepared.n_rows, rows),)](
        *inputs,
        *constants,
        float(alpha),
        float(beta),
        data is not None,
        rows,
        columns,
        triton.next_power_of_2(x.shape[1]),
        num_warps=warps,
        num_stages=1,
    )
    return out
