# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Prepared transpose gather with version-aware values and a scatter fallback."""

import os

import torch
import triton
import triton.language as tl

from ._common import (
    _gather_values,
    _is_iluvatar_runtime,
    _is_maca_runtime,
    _is_mthreads_runtime,
)


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
    if _is_mthreads_runtime():
        boundaries = torch.arange(n + 1, device=indices.device, dtype=indices.dtype)
        ptr = torch.searchsorted(indices[order], boundaries).to(indptr.dtype)
    else:
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
    if _is_iluvatar_runtime():
        if avg <= 3:
            lanes, batch = 4, 128
        elif avg <= 8:
            lanes, batch = 8, 64
        else:
            lanes, batch = 8, 32
        return cols, ptr, order, row_order, lanes, batch, {
            "max_row_nnz": maximum,
            "avg_row_nnz": avg,
        }
    if _is_maca_runtime():
        # C550 favors a small one-wave tile over the CUDA 512-element tile.
        lanes = 32 if avg >= 32 else 16
        return cols, ptr, order, row_order, lanes, 128 // lanes, {}
    if _is_mthreads_runtime():
        lanes, batch = (4, 128) if avg <= 4 else (8, 64) if avg <= 24 else (8, 16)
        return cols, ptr, order, row_order, lanes, batch, {}
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
        source = tl.load(ORDER + pos, valid, other=0) if INDIRECT else pos
        if COMPLEX:
            col = tl.load(I + pos, valid, other=0, cache_modifier=".cg")
            vr = tl.load(V + 2 * source, valid, other=0., cache_modifier=".cg")
            vi = tl.load(V + 2 * source + 1, valid, other=0., cache_modifier=".cg")
            if CONJ:
                vi = -vi
            xr = tl.load(X + 2 * col, valid, other=0., cache_modifier=".ca")
            xi = tl.load(X + 2 * col + 1, valid, other=0., cache_modifier=".ca")
            re += vr * xr - vi * xi
            im += vr * xi + vi * xr
        else:
            col = tl.load(I + pos, valid, other=0)
            v = tl.load(V + source, valid, other=0.).to(tl.float32)
            x = tl.load(X + col, valid, other=0.).to(tl.float32)
            re += v * x
    if COMPLEX:
        tl.store(Y + 2 * rows, tl.sum(re, 1), valid_rows)
        tl.store(Y + 2 * rows + 1, tl.sum(im, 1), valid_rows)
    else:
        tl.store(Y + rows, tl.sum(re, 1), valid_rows)


@triton.jit
def _gather_fixed(V, I, P, ORDER, ROWS, X, Y, M: tl.constexpr,
                  COMPLEX: tl.constexpr, CONJ: tl.constexpr, INDIRECT: tl.constexpr,
                  R: tl.constexpr, K: tl.constexpr, STEPS: tl.constexpr):
    ids = tl.program_id(0) * R + tl.arange(0, R)
    valid_rows = ids < M
    rows = tl.load(ROWS + ids, valid_rows, other=0)
    starts = tl.load(P + rows, valid_rows, other=0)
    ends = tl.load(P + rows + 1, valid_rows, other=0)
    slots = tl.arange(0, K)
    re = tl.zeros((R, K), tl.float32)
    im = tl.zeros((R, K), tl.float32)
    for step in range(STEPS):
        pos = starts[:, None] + step * K + slots[None, :]
        valid = valid_rows[:, None] & (pos < ends[:, None])
        source = tl.load(ORDER + pos, valid, other=0) if INDIRECT else pos
        if COMPLEX:
            col = tl.load(I + pos, valid, other=0, cache_modifier=".cg")
            vr = tl.load(V + 2 * source, valid, other=0., cache_modifier=".cg")
            vi = tl.load(V + 2 * source + 1, valid, other=0., cache_modifier=".cg")
            xr = tl.load(X + 2 * col, valid, other=0., cache_modifier=".ca")
            xi = tl.load(X + 2 * col + 1, valid, other=0., cache_modifier=".ca")
            if CONJ:
                vi = -vi
            re += vr * xr - vi * xi
            im += vr * xi + vi * xr
        else:
            col = tl.load(I + pos, valid, other=0)
            v = tl.load(V + source, valid, other=0.).to(tl.float32)
            x = tl.load(X + col, valid, other=0.).to(tl.float32)
            re += v * x
    if COMPLEX:
        tl.store(Y + 2 * rows, tl.sum(re, 1), valid_rows)
        tl.store(Y + 2 * rows + 1, tl.sum(im, 1), valid_rows)
    else:
        tl.store(Y + rows, tl.sum(re, 1), valid_rows)


@triton.jit
def _gather_split_fixed(VR, VI, I, P, ROWS, X, Y, M: tl.constexpr,
                        CONJ: tl.constexpr, R: tl.constexpr, K: tl.constexpr,
                        STEPS: tl.constexpr):
    ids = tl.program_id(0) * R + tl.arange(0, R)
    valid_rows = ids < M
    rows = tl.load(ROWS + ids, valid_rows, other=0)
    starts = tl.load(P + rows, valid_rows, other=0)
    ends = tl.load(P + rows + 1, valid_rows, other=0)
    slots = tl.arange(0, K)
    re = tl.zeros((R, K), tl.float32)
    im = tl.zeros((R, K), tl.float32)
    for step in range(STEPS):
        pos = starts[:, None] + step * K + slots[None, :]
        valid = valid_rows[:, None] & (pos < ends[:, None])
        col = tl.load(I + pos, valid, other=0, cache_modifier=".cg")
        vr = tl.load(VR + pos, valid, other=0., cache_modifier=".cg")
        vi = tl.load(VI + pos, valid, other=0., cache_modifier=".cg")
        xr = tl.load(X + 2 * col, valid, other=0., cache_modifier=".ca")
        xi = tl.load(X + 2 * col + 1, valid, other=0., cache_modifier=".ca")
        if CONJ:
            vi = -vi
        re += vr * xr - vi * xi
        im += vr * xi + vi * xr
    parts = tl.arange(0, 2)
    result = tl.join(tl.sum(re, 1), tl.sum(im, 1))
    tl.store(Y + 2 * rows[:, None] + parts[None, :], result, valid_rows[:, None])


@triton.jit
def _gather_split_dynamic(VR, VI, I, P, ROWS, X, Y, M: tl.constexpr,
                          CONJ: tl.constexpr, R: tl.constexpr, K: tl.constexpr):
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
        col = tl.load(I + pos, valid, other=0, cache_modifier=".cg")
        vr = tl.load(VR + pos, valid, other=0., cache_modifier=".cg")
        vi = tl.load(VI + pos, valid, other=0., cache_modifier=".cg")
        xr = tl.load(X + 2 * col, valid, other=0., cache_modifier=".ca")
        xi = tl.load(X + 2 * col + 1, valid, other=0., cache_modifier=".ca")
        if CONJ:
            vi = -vi
        re += vr * xr - vi * xi
        im += vr * xi + vi * xr
    parts = tl.arange(0, 2)
    result = tl.join(tl.sum(re, 1), tl.sum(im, 1))
    tl.store(Y + 2 * rows[:, None] + parts[None, :], result, valid_rows[:, None])


def gather(values, x, plan, n_rows, conjugate=False, out=None):
    if _is_mthreads_runtime() and values.dtype == torch.float16 and plan[2] is None:
        from .mixed_spmx import _musa_csr_spmv
        result = _musa_csr_spmv(values.contiguous(), plan[0], plan[1], x.contiguous(),
                                n_rows, values.dtype if out is None else out.dtype,
                                torch.float32, out)
        if out is not None and result is not out:
            out.copy_(result)
            return out
        return result
    y = out if out is not None else torch.empty(n_rows, dtype=values.dtype, device=values.device)
    if not n_rows:
        return y
    values = values.resolve_conj()
    x = x.resolve_conj().contiguous()
    complex_values = values.is_complex()
    cols, ptr, order, rows, lanes, batch, cache = plan
    maca = _is_maca_runtime()
    if maca and not complex_values:
        batch *= 2
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
    if _is_mthreads_runtime() and not complex_values and values.dtype == torch.float16:
        from .mixed_spmx import _musa_csr_spmv
        result = _musa_csr_spmv(values, cols, ptr, x, n_rows,
                                y.dtype, torch.float32, y)
        if result is not y:
            y.copy_(result)
        return y
    gather_warps = 1 if maca else 4
    gather_stages = None
    if _is_iluvatar_runtime() and complex_values:
        mean = values.numel() / max(1, n_rows)
        max_row_nnz = int(cache.get("max_row_nnz", 0))
        row_spread = max_row_nnz / max(mean, 1.0)
        if mean <= 4:
            if row_spread > 16.0:
                batch, lanes = 16, 8
            elif row_spread <= 8.0:
                batch, lanes = 64, 4
            else:
                batch, lanes = 32, 4
        elif mean <= 12:
            batch, lanes = 16, 8
        elif mean <= 16:
            batch, lanes = 8, 16
        else:
            batch, lanes = (4, 32) if row_spread <= 2.0 else (8, 16)
        gather_warps = 1
        gather_stages = 1
        split_warps = 1
        batch_override = os.environ.get("FLAGSPARSE_ILUVATAR_GATHER_BATCH")
        lanes_override = os.environ.get("FLAGSPARSE_ILUVATAR_GATHER_LANES")
        warps_override = os.environ.get("FLAGSPARSE_ILUVATAR_GATHER_WARPS")
        stages_override = os.environ.get("FLAGSPARSE_ILUVATAR_GATHER_STAGES")
        if batch_override:
            batch = max(1, int(batch_override))
        if lanes_override:
            lanes = max(1, int(lanes_override))
        if warps_override:
            gather_warps = max(1, min(8, int(warps_override)))
        if stages_override:
            gather_stages = max(1, min(8, int(stages_override)))
    fixed_steps = None
    if _is_iluvatar_runtime() and complex_values:
        max_row_nnz = int(cache.get("max_row_nnz", 0))
        avg_row_nnz = float(cache.get("avg_row_nnz", 0.0))
        if max_row_nnz <= 64 and max_row_nnz <= max(8, 5 * avg_row_nnz):
            fixed_steps = triton.cdiv(max_row_nnz, lanes)
    split_values = None
    if (
        _is_iluvatar_runtime()
        and complex_values
        and order is None
        and os.environ.get("FLAGSPARSE_ILUVATAR_GATHER_SPLIT", "1") != "0"
    ):
        if cache.get("split_source") is not values:
            cache.update(split_source=values)
        try:
            value_version = values._version
        except RuntimeError:
            value_version = None
        if cache.get("split_version") != value_version:
            cache.update(
                split_version=value_version,
                split_real=values.real.contiguous(),
                split_imag=values.imag.contiguous(),
            )
        split_values = cache["split_real"], cache["split_imag"]
    if split_values is not None and fixed_steps is not None:
        _gather_split_fixed[(triton.cdiv(n_rows, batch),)](
            split_values[0],
            split_values[1],
            cols,
            ptr,
            rows,
            vector,
            target,
            n_rows,
            conjugate,
            batch,
            lanes,
            fixed_steps,
            num_warps=split_warps,
            num_stages=gather_stages,
        )
        return y
    if split_values is not None:
        _gather_split_dynamic[(triton.cdiv(n_rows, batch),)](
            split_values[0],
            split_values[1],
            cols,
            ptr,
            rows,
            vector,
            target,
            n_rows,
            conjugate,
            batch,
            lanes,
            num_warps=split_warps,
            num_stages=gather_stages,
        )
        return y
    fixed_steps = None
    if maca and values.dtype == torch.complex64:
        from ._spmv_csr_maca import complex_gather
        complex_gather[(triton.cdiv(n_rows, batch),)](
            v, cols, ptr, order if order is not None else cols, rows, vector, target,
            n_rows, conjugate, order is not None, batch, lanes, num_warps=1,
        )
    else:
        launch = dict(num_warps=gather_warps)
        if gather_stages is not None:
            launch["num_stages"] = gather_stages
        args = (
            v, cols, ptr, order if order is not None else cols, rows, vector, target,
            n_rows, complex_values, conjugate, order is not None, batch, lanes,
        )
        if fixed_steps is None:
            _gather[(triton.cdiv(n_rows, batch),)](*args, **launch)
        else:
            _gather_fixed[(triton.cdiv(n_rows, batch),)](
                *args, STEPS=fixed_steps, **launch
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
