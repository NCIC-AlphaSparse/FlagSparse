# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared transpose gather with version-aware values and a scatter fallback."""

import torch
import triton
import triton.language as tl

from ._common import _gather_values


def prepare(indices, indptr, shape):
    """Prepare transpose topology; values are refreshed lazily during compute."""
    m, n = shape
    lengths = (indptr[1:] - indptr[:-1]).long()
    source_rows = torch.repeat_interleave(
        torch.arange(m, device=indices.device, dtype=indices.dtype), lengths,
        output_size=indices.numel(),
    )
    order = torch.argsort(indices, stable=True).to(indices.dtype)
    cols = source_rows[order]
    counts = torch.bincount(indices.long(), minlength=n)
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0))).to(indptr.dtype)
    return make_plan(cols, ptr, order, n)


def make_plan(cols, ptr, order, n):
    counts = ptr[1:] - ptr[:-1]
    avg = cols.numel() / max(n, 1)
    maximum = int(counts.max()) if n else 0
    row_order = (torch.argsort(counts, stable=True) if avg <= 4 and maximum > 128
                 else torch.arange(n, device=ptr.device)).to(ptr.dtype)
    lanes = 4 if avg <= 4 else 8 if avg <= 8 else 16 if avg <= 24 else 32
    return cols, ptr, order, row_order, lanes, 512 // lanes, {}


@triton.jit
def _gather(V, I, P, ORDER, ROWS, X, Y, M: tl.constexpr,
            COMPLEX: tl.constexpr, CONJ: tl.constexpr, INDIRECT: tl.constexpr,
            R: tl.constexpr, K: tl.constexpr):
    ids = tl.program_id(0) * R + tl.arange(0, R)
    valid_rows = ids < M
    rows = tl.load(ROWS + ids, valid_rows, other=0)
    starts = tl.load(P + rows, valid_rows, other=0)
    ends = tl.load(P + rows + 1, valid_rows, other=0)
    slots = tl.arange(0, K)
    re = tl.zeros((R, K), tl.float32)
    im = tl.zeros((R, K), tl.float32)
    for base in range(0, tl.max(ends - starts, 0), K):
        pos = starts[:, None] + base + slots[None, :]
        valid = valid_rows[:, None] & (pos < ends[:, None])
        col = tl.load(I + pos, valid, other=0)
        source = tl.load(ORDER + pos, valid, other=0) if INDIRECT else pos
        if COMPLEX:
            vr = tl.load(V + 2 * source, valid, other=0.)
            vi = tl.load(V + 2 * source + 1, valid, other=0.)
            if CONJ:
                vi = -vi
            xr = tl.load(X + 2 * col, valid, other=0.)
            xi = tl.load(X + 2 * col + 1, valid, other=0.)
            re += vr * xr - vi * xi
            im += vr * xi + vi * xr
        else:
            v = tl.load(V + source, valid, other=0.).to(tl.float32)
            x = tl.load(X + col, valid, other=0.).to(tl.float32)
            re += v * x
    if COMPLEX:
        tl.store(Y + 2 * rows, tl.sum(re, 1), valid_rows)
        tl.store(Y + 2 * rows + 1, tl.sum(im, 1), valid_rows)
    else:
        tl.store(Y + rows, tl.sum(re, 1), valid_rows)


def gather(values, x, plan, n_rows, conjugate=False, out=None):
    y = out if out is not None else torch.empty(n_rows, dtype=values.dtype, device=values.device)
    if not n_rows:
        return y
    values = values.resolve_conj()
    x = x.resolve_conj().contiguous()
    complex_values = values.is_complex()
    cols, ptr, order, rows, lanes, batch, cache = plan
    # Torch in-place updates invalidate the materialized value order. Inference
    # tensors have no version counter and use live indirect loads instead.
    if order is not None:
        try:
            version = values._version
        except RuntimeError:
            version = None
        if version is not None:
            if cache.get("version") != version or cache.get("source") is not values:
                cache.update(
                    source=values,
                    version=version,
                    values=_gather_values(values, order).contiguous(),
                )
            values = cache["values"]
            order = None
    if complex_values:
        if cache.get("component_source") is not values:
            cache.update(component_source=values, component_view=torch.view_as_real(values))
        v, vector, target = cache["component_view"], torch.view_as_real(x), torch.view_as_real(y)
    else:
        v, vector, target = values, x, y
    _gather[(triton.cdiv(n_rows, batch),)](
        v, cols, ptr, order if order is not None else cols, rows, vector, target,
        n_rows, complex_values, conjugate, order is not None, batch, lanes, num_warps=4,
    )
    return y


@triton.jit
def _scatter(V, I, P, X, Y, M: tl.constexpr, COMPLEX: tl.constexpr,
             CONJ: tl.constexpr, R: tl.constexpr, K: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    starts = tl.load(P + rows, rows < M, other=0)
    ends = tl.load(P + rows + 1, rows < M, other=0)
    slots = tl.arange(0, K)
    if COMPLEX:
        xr = tl.load(X + 2 * rows, rows < M, other=0)
        xi = tl.load(X + 2 * rows + 1, rows < M, other=0)
    else:
        x = tl.load(X + rows, rows < M, other=0)
    for base in range(0, tl.max(ends - starts, 0), K):
        pos = starts[:, None] + base + slots[None, :]
        valid = (rows[:, None] < M) & (pos < ends[:, None])
        col = tl.load(I + pos, valid, other=0)
        if COMPLEX:
            vr = tl.load(V + 2 * pos, valid, other=0)
            vi = tl.load(V + 2 * pos + 1, valid, other=0)
            if CONJ:
                vi = -vi
            re = vr * xr[:, None] - vi * xi[:, None]
            im = vr * xi[:, None] + vi * xr[:, None]
            tl.atomic_add(Y + 2 * col, re, valid, sem="relaxed")
            tl.atomic_add(Y + 2 * col + 1, im, valid, sem="relaxed")
        else:
            v = tl.load(V + pos, valid, other=0)
            tl.atomic_add(Y + col, v * x[:, None], valid, sem="relaxed")


def compute(prepared, x, out=None):
    if prepared.transpose_plan is not None:
        return gather(prepared.data, x, prepared.transpose_plan, prepared.n_cols,
                      prepared.op == 2, out)
    y = out if out is not None else torch.empty(
        prepared.n_cols, dtype=prepared.data.dtype, device=prepared.data.device
    )
    y.zero_()
    if prepared.data.numel() and prepared.n_rows:
        values = prepared.data.resolve_conj().contiguous()
        x = x.resolve_conj().contiguous()
        complex_values = values.is_complex()
        if complex_values:
            values, x, target = (torch.view_as_real(t).reshape(-1) for t in (values, x, y))
        else:
            target = y
        rows_per_program = 8 if prepared.backend_caps.subgroup_width >= 64 else 4
        _scatter[(triton.cdiv(prepared.n_rows, rows_per_program),)](
            values, prepared.kernel_indices, prepared.kernel_indptr, x, target,
            prepared.n_rows, complex_values, prepared.op == 2, rows_per_program, 32,
            num_warps=4,
        )
    return y
