# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Batched row-owned SpMM for fp16 and complex64 short-row workloads."""
import torch
import triton
import triton.language as tl


@triton.jit
def _kernel(V, I, P, B, C, M, N, BK, BN, CM, CN,
            COMPLEX: tl.constexpr, R: tl.constexpr, K: tl.constexpr,
            TILE_N: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    valid = rows < M
    starts = tl.load(P + rows, valid, other=0)
    ends = tl.load(P + rows + 1, valid, other=0)
    ns = tl.program_id(1) * TILE_N + tl.arange(0, TILE_N)
    ks = tl.arange(0, K)
    real = tl.zeros((R, TILE_N), tl.float32)
    imag = tl.zeros((R, TILE_N), tl.float32)
    for base in range(0, tl.max(ends - starts, 0), K):
        pos = starts[:, None] + base + ks[None, :]
        mask = valid[:, None] & (pos < ends[:, None])
        cols = tl.load(I + pos, mask, other=0)
        dense_mask = mask[:, :, None] & (ns[None, None, :] < N)
        dense_pos = cols[:, :, None] * BK + ns[None, None, :] * BN
        if COMPLEX:
            vr = tl.load(V + 2 * pos, mask, other=0.).to(tl.float32)
            vi = tl.load(V + 2 * pos + 1, mask, other=0.).to(tl.float32)
            br = tl.load(B + 2 * dense_pos, dense_mask, other=0.).to(tl.float32)
            bi = tl.load(B + 2 * dense_pos + 1, dense_mask, other=0.).to(tl.float32)
            real += tl.sum(vr[:, :, None] * br - vi[:, :, None] * bi, 1)
            imag += tl.sum(vr[:, :, None] * bi + vi[:, :, None] * br, 1)
        else:
            v = tl.load(V + pos, mask, other=0.).to(tl.float32)
            b = tl.load(B + dense_pos, dense_mask, other=0.).to(tl.float32)
            real += tl.sum(v[:, :, None] * b, 1)
    out = rows[:, None] * CM + ns[None, :] * CN
    store_mask = valid[:, None] & (ns[None, :] < N)
    if COMPLEX:
        tl.store(C + 2 * out, real, store_mask)
        tl.store(C + 2 * out + 1, imag, store_mask)
    else:
        tl.store(C + out, real, store_mask)


def compute(values, cols, ptr, b, out):
    m, n = out.shape
    if not m or not n:
        return out
    is_complex = values.is_complex()
    # Preserve logical strides for row/column-major and padded dense operands.
    v, dense, target = (torch.view_as_real(t.resolve_conj()) for t in (values, b, out)) if is_complex else (values, b, out)
    mean = values.numel() / max(m, 1)
    batch, lanes = (8, 4) if mean <= 4 else (4, 8)
    tile_n = min(32, triton.next_power_of_2(n))
    _kernel[(triton.cdiv(m, batch), triton.cdiv(n, tile_n))](
        v, cols, ptr, dense, target, m, n, b.stride(0), b.stride(1),
        out.stride(0), out.stride(1), is_complex, batch, lanes, tile_n,
        num_warps=4, num_stages=1,
    )
    return out
