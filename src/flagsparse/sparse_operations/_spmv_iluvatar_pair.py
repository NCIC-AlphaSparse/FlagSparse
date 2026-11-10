"""Paired-component complex gathers for Iluvatar CSR row tiles."""

import os

import torch
import triton
import triton.language as tl

from ._common import _is_iluvatar_runtime
from ._spmv_iluvatar_ell import eligible as _ell_eligible


def eligible(n_rows, mean, maximum):
    if n_rows <= 0 or _ell_eligible(n_rows, mean, maximum):
        return False
    return ((mean <= 3 and maximum <= 16)
            or (mean > 4 and maximum <= 64 and (mean <= 8 or maximum > 16))
            or (mean > 32 and maximum <= 256))


@triton.jit
def _kernel(VR, VI, Cols, Ptr, X, Y, N: tl.constexpr,
            R: tl.constexpr, K: tl.constexpr, CONJ: tl.constexpr,
            STRIDE: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    starts = tl.load(Ptr + rows, rows < N, 0)
    ends = tl.load(Ptr + rows + 1, rows < N, 0)
    slots = tl.arange(0, K)
    parts = tl.arange(0, 2)
    total = tl.full((R, K, 2), 0., tl.float32)
    for base in range(0, tl.max(ends - starts, 0), K):
        pos = starts[:, None] + base + slots[None, :]
        valid = (rows[:, None] < N) & (pos < ends[:, None])
        col = tl.load(Cols + pos, valid, 0, cache_modifier=".cg")
        vr = tl.load(VR + pos, valid, 0., cache_modifier=".cg")
        vi = tl.load(VI + pos, valid, 0., cache_modifier=".cg")
        vector = tl.load(X + 2 * col[:, :, None] + parts[None, None, :],
                         valid[:, :, None], 0., cache_modifier=".ca")
        other = tl.gather(vector, tl.broadcast_to(1 - parts[None, None, :], (R, K, 2)), 2)
        sign = tl.where(parts == 0, -1., 1.)
        if CONJ:
            sign = -sign
        total += vr[:, :, None] * vector
        total += vi[:, :, None] * other * sign[None, None, :]
    result = tl.sum(total, 1)
    tl.store(Y + 2 * STRIDE * rows[:, None] + parts[None, :], result, rows[:, None] < N)


def run(values, cols, ptr, x, n_rows, cache, conjugate=False, out=None):
    if (not _is_iluvatar_runtime() or values.dtype != torch.complex64
            or x.dtype != torch.complex64 or not values.numel()):
        return None
    if (os.environ.get("FLAGSPARSE_ILUVATAR_GATHER_SPLIT", "1") == "0"
            or any(os.environ.get("FLAGSPARSE_ILUVATAR_GATHER_" + option)
                   for option in ("BATCH", "LANES", "WARPS", "STAGES"))):
        return None
    mean, maximum = cache.get("avg_row_nnz"), cache.get("max_row_nnz")
    if mean is None or maximum is None or not eligible(n_rows, mean, maximum):
        return None
    values = values.resolve_conj()
    try:
        version = values._version
    except RuntimeError:
        return None
    if cache.get("pair_source") is not values or cache.get("pair_version") != version:
        cache.update(pair_source=values, pair_version=version,
                     pair_real=values.real.contiguous(), pair_imag=values.imag.contiguous())
    x = x.resolve_conj().contiguous()
    target = out if out is not None else torch.empty(n_rows, dtype=values.dtype, device=values.device)
    rows, lanes = (32, 2) if mean <= 3 else (16, 8) if mean <= 12 else (8, 8)
    _kernel[(triton.cdiv(n_rows, rows),)](
        cache["pair_real"], cache["pair_imag"], cols, ptr,
        torch.view_as_real(x), torch.view_as_real(target), n_rows,
        rows, lanes, conjugate, target.stride(0), num_warps=1, num_stages=1,
    )
    return target


@triton.jit
def _mixed_kernel(V, Cols, Ptr, X, Y, N: tl.constexpr,
                  R: tl.constexpr, K: tl.constexpr, STEPS: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    starts = tl.load(Ptr + rows, rows < N, 0)
    ends = tl.load(Ptr + rows + 1, rows < N, 0)
    slots = tl.arange(0, K)
    parts = tl.arange(0, 2)
    total = tl.full((R, K, 2), 0., tl.float32)
    for step in tl.static_range(STEPS):
        pos = starts[:, None] + step * K + slots[None, :]
        valid = (rows[:, None] < N) & (pos < ends[:, None])
        col = tl.load(Cols + pos, valid, 0, cache_modifier=".cg")
        value = tl.load(V + pos, valid, 0., cache_modifier=".cg")
        vector = tl.load(X + 2 * col[:, :, None] + parts[None, None, :],
                         valid[:, :, None], 0., cache_modifier=".ca")
        total += value[:, :, None] * vector
    tl.store(Y + 2 * rows[:, None] + parts[None, :], tl.sum(total, 1), rows[:, None] < N)


def run_mixed(values, cols, ptr, x, out, n_rows, cache):
    if any(os.environ.get("FLAGSPARSE_ILUVATAR_MIXED_" + option)
           for option in ("ROWS", "BLOCK", "WARPS")):
        return False
    mean, maximum = cache.get("avg_row_nnz", 0), cache.get("max_row_nnz", 0)
    if not (8 < mean <= 16 and maximum <= 64 and n_rows > 262144):
        return False
    rows = 16 if mean <= 12 else 8
    _mixed_kernel[(triton.cdiv(n_rows, rows),)](
        values, cols, ptr, torch.view_as_real(x.resolve_conj()), torch.view_as_real(out),
        n_rows, rows, 8, triton.cdiv(maximum, 8), num_warps=1, num_stages=1,
    )
    return True
