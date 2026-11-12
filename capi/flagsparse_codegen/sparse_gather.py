# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MUSA C API SpMV/SpMM reductions and wide COO transpose accumulation.

Indirect addressing reads live values through a preprocessed topology. Direct
addressing batches ordinary CSR rows. Alpha/beta and complex conjugation are
fused. COO transpose instead scatters wide products into an FP64 workspace
so short, irregular columns do not serialize an output-row loop.
"""
import triton
import triton.language as tl


@triton.jit
def musa_coo_transpose_zero(T, M, B: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tl.store(T + pos, 0., pos < M)


@triton.jit
def musa_coo_transpose_scatter_f64(V, ROWS, COLS, X, T, NNZ,
                                    B: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = pos < NNZ
    row = tl.load(ROWS + pos, valid, other=0).to(tl.int64)
    col = tl.load(COLS + pos, valid, other=0).to(tl.int64)
    value = tl.load(V + pos, valid, other=0.).to(tl.float64)
    rhs = tl.load(X + row, valid, other=0.).to(tl.float64)
    # Wide products and sums preserve small terms during FP32 cancellation.
    tl.atomic_add(T + col, value * rhs, valid, sem="relaxed")


@triton.jit
def musa_coo_transpose_finish(T, Y, M, ALPHA, BETA,
                              B: tl.constexpr, HAS_BETA: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = pos < M
    result = ALPHA.to(tl.float64) * tl.load(T + pos, valid, other=0.)
    if HAS_BETA:
        result += BETA.to(tl.float64) * tl.load(Y + pos, valid, other=0.).to(tl.float64)
    tl.store(Y + pos, result.to(tl.float32), valid)


@triton.jit
def musa_coo_transpose_scatter_c32_f64(V, ROWS, COLS, X, T, NNZ,
                                     B: tl.constexpr, CONJ: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = pos < NNZ
    row = tl.load(ROWS + pos, valid, other=0).to(tl.int64)
    col = tl.load(COLS + pos, valid, other=0).to(tl.int64)
    vr = tl.load(V + 2 * pos, valid, other=0.).to(tl.float64)
    vi = tl.load(V + 2 * pos + 1, valid, other=0.).to(tl.float64)
    if CONJ:
        vi = -vi
    xr = tl.load(X + 2 * row, valid, other=0.).to(tl.float64)
    xi = tl.load(X + 2 * row + 1, valid, other=0.).to(tl.float64)
    tl.atomic_add(T + 2 * col, vr * xr - vi * xi, valid, sem="relaxed")
    tl.atomic_add(T + 2 * col + 1, vr * xi + vi * xr, valid, sem="relaxed")


@triton.jit
def musa_coo_transpose_finish_c32_f64(T, Y, M, AR, AI, BR, BI,
                                    B: tl.constexpr, HAS_BETA: tl.constexpr):
    pos = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = pos < M
    real = tl.load(T + 2 * pos, valid, other=0.)
    imag = tl.load(T + 2 * pos + 1, valid, other=0.)
    yr = AR.to(tl.float64) * real - AI.to(tl.float64) * imag
    yi = AR.to(tl.float64) * imag + AI.to(tl.float64) * real
    if HAS_BETA:
        cr = tl.load(Y + 2 * pos, valid, other=0.).to(tl.float64)
        ci = tl.load(Y + 2 * pos + 1, valid, other=0.).to(tl.float64)
        yr += BR.to(tl.float64) * cr - BI.to(tl.float64) * ci
        yi += BR.to(tl.float64) * ci + BI.to(tl.float64) * cr
    tl.store(Y + 2 * pos, yr.to(tl.float32), valid)
    tl.store(Y + 2 * pos + 1, yi.to(tl.float32), valid)


@triton.jit
def musa_csr_spmv_trans_real_complex_f64(V, P, ORDER, ROWS, X, Y, M,
                                       R: tl.constexpr, K: tl.constexpr,
                                       WIDE: tl.constexpr = False):
    # One owner per output column. FP64 products/sums preserve small terms when
    # large FP32 contributions cancel, independently of source scheduling.
    row = (tl.program_id(0).to(tl.int64) if WIDE else tl.program_id(0)) * R + tl.arange(0, R)
    active = row < M
    start = tl.load(P + row, active, other=0)
    end = tl.load(P + row + 1, active, other=0)
    if WIDE:
        start = start.to(tl.int64)
        end = end.to(tl.int64)
    lane = tl.arange(0, K)
    re = tl.zeros((R, K), tl.float64)
    im = tl.zeros((R, K), tl.float64)
    for base in range(0, tl.max(end - start, 0), K):
        pos = start[:, None] + base + lane[None, :]
        valid = active[:, None] & (pos < end[:, None])
        src = tl.load(ORDER + pos, valid, other=0)
        col = tl.load(ROWS + pos, valid, other=0)
        if WIDE:
            src = src.to(tl.int64)
            col = col.to(tl.int64)
        value = tl.load(V + src, valid, other=0.).to(tl.float64)
        xr = tl.load(X + 2 * col, valid, other=0.).to(tl.float64)
        xi = tl.load(X + 2 * col + 1, valid, other=0.).to(tl.float64)
        re += value * xr
        im += value * xi
    tl.store(Y + 2 * row, tl.sum(re, 1).to(tl.float32), active)
    tl.store(Y + 2 * row + 1, tl.sum(im, 1).to(tl.float32), active)


@triton.jit
def sparse_gather_spmm_row_f32(V, P, ORDER, COLS, B, C, ar, ai, br, bi, M, N,
                              SBK, SBN, SCM, SCN,
                              R: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
                              COMPLEX: tl.constexpr, CONJ: tl.constexpr,
                              INDIRECT: tl.constexpr, HAS_BETA: tl.constexpr,
                              WIDE: tl.constexpr = False):
    # MUSA, real FP32, eight contiguous dense columns. Keep sparse lanes
    # independent through the loop and reduce once per output tile.
    row = (tl.program_id(0).to(tl.int64) if WIDE else tl.program_id(0)) * R + tl.arange(0, R)
    active = row < M
    start = tl.load(P + row, active, other=0)
    end = tl.load(P + row + 1, active, other=0)
    if WIDE:
        start = start.to(tl.int64)
        end = end.to(tl.int64)
    ns = tl.arange(0, BN)
    ks = tl.arange(0, K)
    acc = tl.zeros((R, K, BN), tl.float32)
    for offset in range(0, tl.max(end-start, 0), K):
        pos = start[:, None] + offset + ks[None, :]
        valid = active[:, None] & (pos < end[:, None])
        src = tl.load(ORDER + pos, valid, other=0) if INDIRECT else pos
        col = tl.load(COLS + pos, valid, other=0)
        if WIDE:
            src = src.to(tl.int64)
            col = col.to(tl.int64)
        value = tl.load(V + src, valid, other=0.)
        dense = tl.load(B + col[:, :, None] * 8 + ns[None, None, :],
                        valid[:, :, None] & (ns[None, None, :] < 8), other=0.)
        acc += value[:, :, None] * dense
    result = ar * tl.sum(acc, 1)
    out = C + row[:, None] * 8 + ns[None, :]
    mask = active[:, None] & (ns[None, :] < 8)
    if HAS_BETA:
        result += br * tl.load(out, mask, other=0.)
    tl.store(out, result, mask)


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
    if COMPLEX:
        # Widen the final tree so small terms survive cancellation between
        # lanes without widening the sparse loop's per-lane accumulators.
        real = tl.sum(re.to(tl.float64), 1).to(tl.float32)
        imag = tl.sum(im.to(tl.float64), 1).to(tl.float32)
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
        real = tl.sum(re, 1)
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
