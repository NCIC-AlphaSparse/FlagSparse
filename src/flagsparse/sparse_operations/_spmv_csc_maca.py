# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA CSC integer product with exact int32 row reductions."""

import torch
import triton
import triton.language as tl


@triton.jit
def _int_gather(V, I, P, ORDER, ROWS, X, Y, N: tl.constexpr,
                INDIRECT: tl.constexpr, R: tl.constexpr, K: tl.constexpr,
                STRIDE: tl.constexpr):
    ids = tl.program_id(0) * R + tl.arange(0, R)
    rows = tl.load(ROWS + ids, ids < N, 0)
    starts = tl.load(P + rows, ids < N, 0)
    ends = tl.load(P + rows + 1, ids < N, 0)
    slots = tl.arange(0, K)
    acc = tl.zeros((R, K), tl.int32)
    for base in range(0, tl.max(ends - starts, 0), K):
        pos = starts[:, None] + base + slots[None, :]
        valid = (ids[:, None] < N) & (pos < ends[:, None])
        source = tl.load(ORDER + pos, valid, 0) if INDIRECT else pos
        col = tl.load(I + pos, valid, 0)
        a = tl.load(V + source, valid, 0).to(tl.int32)
        x = tl.load(X + col, valid, 0).to(tl.int32)
        acc += a * x
    tl.store(Y + rows * STRIDE, tl.sum(acc, 1), ids < N)


def int_gather(values, x, plan, n_rows, dtype, out=None):
    y = out if out is not None else torch.empty(n_rows, dtype=dtype, device=values.device)
    if n_rows:
        cols, ptr, order, rows, lanes, batch, _ = plan
        _int_gather[(triton.cdiv(n_rows, batch * 2),)](
            values.contiguous(), cols, ptr, order, rows, x.contiguous(), y,
            n_rows, order is not None, batch * 2, lanes, y.stride(0), num_warps=1)
    return y
