# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA CSR complex gather with contiguous component loads."""

import triton
import triton.language as tl


@triton.jit
def complex_gather(V, I, P, ORDER, ROWS, X, Y, M: tl.constexpr,
                   CONJ: tl.constexpr, INDIRECT: tl.constexpr,
                   R: tl.constexpr, K: tl.constexpr):
    ids = tl.program_id(0) * R + tl.arange(0, R)
    valid_rows = ids < M
    rows = tl.load(ROWS + ids, valid_rows, other=0)
    starts = tl.load(P + rows, valid_rows, other=0)
    ends = tl.load(P + rows + 1, valid_rows, other=0)
    slots = tl.arange(0, K)
    parts = tl.arange(0, 2)
    re = tl.zeros((R, K), tl.float32)
    im = tl.zeros((R, K), tl.float32)
    for base in range(0, tl.max(ends - starts, 0), K):
        pos = starts[:, None] + base + slots[None, :]
        valid = valid_rows[:, None] & (pos < ends[:, None])
        col = tl.load(I + pos, valid, other=0)
        source = tl.load(ORDER + pos, valid, other=0) if INDIRECT else pos
        # Loading the adjacent components together lets MACA vectorize the
        # memory operations instead of issuing separate component gathers.
        values = tl.load(V + 2 * source[:, :, None] + parts[None, None, :],
                         valid[:, :, None], other=0.)
        vector = tl.load(X + 2 * col[:, :, None] + parts[None, None, :],
                         valid[:, :, None], other=0.)
        vr, vi = tl.split(values)
        xr, xi = tl.split(vector)
        if CONJ:
            vi = -vi
        re += vr * xr - vi * xi
        im += vr * xi + vi * xr
    result = tl.join(tl.sum(re, 1), tl.sum(im, 1))
    tl.store(Y + 2 * rows[:, None] + parts[None, :], result, valid_rows[:, None])
