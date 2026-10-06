# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Batched output-row reduction for MUSA C API SpMV/SpMM.

Indirect addressing reads live values through a preprocessed topology. Direct
addressing batches ordinary CSR rows. Alpha/beta and complex conjugation are
fused; each output has one owner and requires no memset or atomic addition.
"""
import triton
import triton.language as tl


@triton.jit
def sparse_gather_f32(V, P, ORDER, COLS, B, C, ar, ai, br, bi, M, N,
                      SBK, SBN, SCM, SCN,
                      R: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
                      COMPLEX: tl.constexpr, CONJ: tl.constexpr,
                      INDIRECT: tl.constexpr, HAS_BETA: tl.constexpr,
                      WIDE: tl.constexpr = False):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    active = rows < M
    start = tl.load(P + rows, active, other=0)
    end = tl.load(P + rows + 1, active, other=0)
    if WIDE:
        start = start.to(tl.int64)
        end = end.to(tl.int64)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, K)
    re = tl.zeros((R, BN), tl.float32)
    if COMPLEX:
        im = tl.zeros((R, BN), tl.float32)
    for base in range(0, tl.max(end - start, 0), K):
        pos = start[:, None] + base + ks[None, :]
        valid = active[:, None] & (pos < end[:, None])
        if INDIRECT:
            src = tl.load(ORDER + pos, valid, other=0)
        else:
            src = pos
        col = tl.load(COLS + pos, valid, other=0)
        if WIDE:
            src = src.to(tl.int64)
            col = col.to(tl.int64)
        mask = valid[:, :, None] & (ns[None, None, :] < N)
        bp = B + col[:, :, None] * SBK + ns[None, None, :] * SBN
        if COMPLEX:
            vr = tl.load(V + 2 * src, valid, other=0.)
            vi = tl.load(V + 2 * src + 1, valid, other=0.)
            if CONJ:
                vi = -vi
            xr = tl.load(bp, mask, other=0.)
            xi = tl.load(bp + 1, mask, other=0.)
            re += tl.sum(vr[:, :, None] * xr - vi[:, :, None] * xi, 1)
            im += tl.sum(vr[:, :, None] * xi + vi[:, :, None] * xr, 1)
        else:
            value = tl.load(V + src, valid, other=0.)
            dense = tl.load(bp, mask, other=0.)
            re += tl.sum(value[:, :, None] * dense, 1)
    out = C + rows[:, None] * SCM + ns[None, :] * SCN
    mask = active[:, None] & (ns[None, :] < N)
    if COMPLEX:
        yr = ar * re - ai * im
        yi = ar * im + ai * re
        if HAS_BETA:
            cr = tl.load(out, mask, other=0.)
            ci = tl.load(out + 1, mask, other=0.)
            yr += br * cr - bi * ci
            yi += br * ci + bi * cr
        tl.store(out, yr, mask)
        tl.store(out + 1, yi, mask)
    else:
        result = ar * re
        if HAS_BETA:
            result += br * tl.load(out, mask, other=0.)
        tl.store(out, result, mask)


@triton.jit
def sparse_gather_spmv_f32(V, P, ORDER, COLS, B, C, ar, ai, br, bi, M, N,
                           SBK, SBN, SCM, SCN,
                           R: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
                           COMPLEX: tl.constexpr, CONJ: tl.constexpr,
                           INDIRECT: tl.constexpr, HAS_BETA: tl.constexpr,
                      WIDE: tl.constexpr = False):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    active = rows < M
    start = tl.load(P + rows, active, other=0)
    end = tl.load(P + rows + 1, active, other=0)
    if WIDE:
        start = start.to(tl.int64)
        end = end.to(tl.int64)
    lane = tl.arange(0, K)
    # Keep lane-local sums through the sparse loop: one reduction per output,
    # rather than a warp/shmem reduction in every iteration of a long row.
    re = tl.zeros((R, K), tl.float32)
    if COMPLEX:
        im = tl.zeros((R, K), tl.float32)
    for base in range(0, tl.max(end - start, 0), K):
        pos = start[:, None] + base + lane[None, :]
        valid = active[:, None] & (pos < end[:, None])
        if INDIRECT:
            src = tl.load(ORDER + pos, valid, other=0)
        else:
            src = pos
        col = tl.load(COLS + pos, valid, other=0)
        if WIDE:
            src = src.to(tl.int64)
            col = col.to(tl.int64)
        if COMPLEX:
            vr = tl.load(V + 2 * src, valid, other=0.)
            vi = tl.load(V + 2 * src + 1, valid, other=0.)
            if CONJ:
                vi = -vi
            xr = tl.load(B + 2 * col, valid, other=0.)
            xi = tl.load(B + 2 * col + 1, valid, other=0.)
            re += vr * xr - vi * xi
            im += vr * xi + vi * xr
        else:
            value = tl.load(V + src, valid, other=0.)
            re += value * tl.load(B + col, valid, other=0.)
    real = tl.sum(re, 1)
    if COMPLEX:
        imag = tl.sum(im, 1)
        yr = ar * real - ai * imag
        yi = ar * imag + ai * real
        if HAS_BETA:
            cr = tl.load(C + 2 * rows, active, other=0.)
            ci = tl.load(C + 2 * rows + 1, active, other=0.)
            yr += br * cr - bi * ci
            yi += br * ci + bi * cr
        tl.store(C + 2 * rows, yr, active)
        tl.store(C + 2 * rows + 1, yi, active)
    else:
        result = ar * real
        if HAS_BETA:
            result += br * tl.load(C + rows, active, other=0.)
        tl.store(C + rows, result, active)


@triton.jit
def sparse_gather_spmm_stream_f32(V, P, ORDER, COLS, B, C, ar, ai, br, bi, M, N,
                                  SBK, SBN, SCM, SCN,
                                  R: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
                                  COMPLEX: tl.constexpr, CONJ: tl.constexpr,
                                  INDIRECT: tl.constexpr, HAS_BETA: tl.constexpr,
                      WIDE: tl.constexpr = False):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    active = rows < M
    start = tl.load(P + rows, active, other=0)
    end = tl.load(P + rows + 1, active, other=0)
    if WIDE:
        start = start.to(tl.int64)
        end = end.to(tl.int64)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    re = tl.zeros((R, BN), tl.float32)
    if COMPLEX:
        im = tl.zeros((R, BN), tl.float32)
    for base in range(0, tl.max(end - start, 0)):
        pos = start + base
        valid = active & (pos < end)
        if INDIRECT:
            src = tl.load(ORDER + pos, valid, other=0)
        else:
            src = pos
        col = tl.load(COLS + pos, valid, other=0)
        if WIDE:
            src = src.to(tl.int64)
            col = col.to(tl.int64)
        mask = valid[:, None] & (ns[None, :] < N)
        bp = B + col[:, None] * SBK + ns[None, :] * SBN
        if COMPLEX:
            vr = tl.load(V + 2 * src, valid, other=0.)
            vi = tl.load(V + 2 * src + 1, valid, other=0.)
            if CONJ:
                vi = -vi
            xr = tl.load(bp, mask, other=0.)
            xi = tl.load(bp + 1, mask, other=0.)
            re += vr[:, None] * xr - vi[:, None] * xi
            im += vr[:, None] * xi + vi[:, None] * xr
        else:
            value = tl.load(V + src, valid, other=0.)
            re += value[:, None] * tl.load(bp, mask, other=0.)
    out = C + rows[:, None] * SCM + ns[None, :] * SCN
    mask = active[:, None] & (ns[None, :] < N)
    if COMPLEX:
        yr = ar * re - ai * im
        yi = ar * im + ai * re
        if HAS_BETA:
            cr = tl.load(out, mask, other=0.)
            ci = tl.load(out + 1, mask, other=0.)
            yr += br * cr - bi * ci
            yi += br * ci + bi * cr
        tl.store(out, yr, mask)
        tl.store(out + 1, yi, mask)
    else:
        result = ar * re
        if HAS_BETA:
            result += br * tl.load(out, mask, other=0.)
        tl.store(out, result, mask)


@triton.jit
def sparse_scatter_spmv_f32(V, SOURCE, TARGET, X, Y, alpha, NNZ, BLOCK: tl.constexpr):
    positions = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    active = positions < NNZ
    source = tl.load(SOURCE + positions, active, other=0)
    target = tl.load(TARGET + positions, active, other=0)
    value = tl.load(V + positions, active, other=0.)
    dense = tl.load(X + source, active, other=0.)
    tl.atomic_add(Y + target, alpha * value * dense, active, sem="relaxed")
