# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA sparse-vector kernels; cached topology never freezes numerical inputs."""

import torch
import triton
import triton.language as tl

from ._rocm_sparse_cache import TensorPlanCache

_AXPBY_INDEX_CACHE = TensorPlanCache()


@triton.jit
def _spvv_complex_small(OUT, X, IDX, Y, NNZ,
                       CONJ: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.arange(0, BLOCK)
    components = tl.arange(0, 2)
    valid = pos < NNZ
    idx = tl.load(IDX + pos, valid, 0)
    x = tl.load(X + 2 * pos[:, None] + components[None, :], valid[:, None], 0.)
    y = tl.load(Y + 2 * idx[:, None] + components[None, :], valid[:, None], 0.)
    xr, xi = tl.split(x)
    yr, yi = tl.split(y)
    if CONJ:
        xi = -xi
    tl.store(OUT, tl.sum(xr * yr - xi * yi, 0))
    tl.store(OUT + 1, tl.sum(xr * yi + xi * yr, 0))


def spvv_complex(values, indices, y, conj):
    """One launch for a MACA complex64 dot with at most 4096 entries."""
    nnz = values.numel()
    block = max(64, triton.next_power_of_2(nnz))
    result = torch.empty((), dtype=torch.complex64, device=y.device)
    _spvv_complex_small[(1,)](
        torch.view_as_real(result), torch.view_as_real(values.resolve_conj().resolve_neg()),
        indices.contiguous(), torch.view_as_real(y.resolve_conj().resolve_neg()),
        nnz, conj, block, num_warps=4)
    return result


@triton.jit
def _inverse(OWNER, INDICES, NNZ, N, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(INDICES + pos, pos < NNZ, -1)
    tl.store(OWNER + row, pos, (pos < NNZ) & (row >= 0) & (row < N))


@triton.jit
def _axpby_complex(Y, V, OWNER, N, AR, AI, BR, BI,
                    COMPLEX_ALPHA: tl.constexpr, COMPLEX_BETA: tl.constexpr,
                    BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, 2)
    valid = row < N
    pos = tl.load(OWNER + row, valid, -1)
    values = tl.load(V + 2 * pos[:, None] + parts[None, :],
                     (valid & (pos >= 0))[:, None], 0.)
    previous = tl.load(Y + 2 * row[:, None] + parts[None, :], valid[:, None], 0.)
    xr, xi = tl.split(values)
    yr, yi = tl.split(previous)
    update_real = AR * xr
    update_imag = AR * xi
    if COMPLEX_ALPHA:
        update_real -= AI * xi
        update_imag += AI * xr
    # Missing sparse entries have no alpha contribution, even for NaN/Inf alpha.
    real = BR * yr + tl.where(pos >= 0, update_real, 0.)
    imag = BR * yi + tl.where(pos >= 0, update_imag, 0.)
    if COMPLEX_BETA:
        real -= BI * yi
        imag += BI * yr
    result = tl.join(real, imag)
    tl.store(Y + 2 * row[:, None] + parts[None, :], result, valid[:, None])


def axpby_complex(values, indices, y, ar, ai, br, bi):
    """One writer per dense element, using the API's unique-index contract.

    Sizes stay runtime arguments: a constexpr length recompiles for every new y.
    """
    n = y.numel()
    if not n or (not values.numel() and (br, bi) == (1., 0.)):
        return
    owner = _AXPBY_INDEX_CACHE.get((indices,), (n,))
    if owner is None:
        owner = torch.full((n,), -1, dtype=torch.int32, device=y.device)
        nnz = indices.numel()
        if nnz:
            _inverse[(triton.cdiv(nnz, 256),)](
                owner, indices.contiguous(), nnz, n, 256, num_warps=4)
        _AXPBY_INDEX_CACHE.put((indices,), (n,), owner)
    _axpby_complex[(triton.cdiv(n, 512),)](
        torch.view_as_real(y), torch.view_as_real(values), owner, n,
        ar, ai, br, bi, ai != 0., bi != 0., 512, num_warps=4)
