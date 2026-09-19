"""Native CSR kernels. Runtime plans are built afresh on every invocation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _product(A, X, pos, col, mask, COMPLEX: tl.constexpr, ACC: tl.constexpr):
    if COMPLEX:
        ar = tl.load(A + 2 * pos, mask, 0).to(ACC)
        ai = tl.load(A + 2 * pos + 1, mask, 0).to(ACC)
        xr = tl.load(X + 2 * col, mask, 0).to(ACC)
        xi = tl.load(X + 2 * col + 1, mask, 0).to(ACC)
        return ar * xr - ai * xi, ar * xi + ai * xr
    else:
        a = tl.load(A + pos, mask, 0).to(ACC)
        x = tl.load(X + col, mask, 0).to(ACC)
        return a * x, tl.full(pos.shape, 0, ACC)


@triton.jit
def _store_result(Y, row, real, imag, mask, COMPLEX: tl.constexpr):
    if COMPLEX:
        tl.store(Y + row * 2, real, mask)
        tl.store(Y + row * 2 + 1, imag, mask)
    else:
        tl.store(Y + row, real, mask)


@triton.jit
def row_tile_kernel(
    A,
    CI,
    RP,
    X,
    Y,
    ROWS,
    N,
    INDEXED: tl.constexpr,
    R: tl.constexpr,
    V: tl.constexpr,
    STAGES: tl.constexpr,
    COMPLEX: tl.constexpr = False,
    ACC: tl.constexpr = tl.float64,
):
    ridx = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    valid = ridx < N
    if INDEXED:
        row = tl.load(ROWS + ridx, valid, 0)
    else:
        row = ridx
    start = tl.load(RP + row, valid, 0).to(tl.int64)
    end = tl.load(RP + row + 1, valid, 0).to(tl.int64)
    lane = tl.arange(0, V)
    acc = tl.zeros((R, V), ACC)
    imag = tl.zeros((R, V), ACC)
    steps = tl.max(tl.cdiv(end - start, V), 0)
    for step in tl.range(0, steps, num_stages=STAGES):
        pos = start[:, None] + step * V + lane[None, :]
        mask = valid[:, None] & (pos < end[:, None])
        col = tl.load(CI + pos, mask, 0).to(tl.int64)
        pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
        acc = acc + pr
        if COMPLEX:
            imag = imag + pi
    _store_result(Y, row, tl.sum(acc, 1), tl.sum(imag, 1), valid, COMPLEX)


@triton.jit
def row_vector_kernel(
    A,
    CI,
    RP,
    X,
    Y,
    ROWS,
    N,
    INDEXED: tl.constexpr,
    B: tl.constexpr,
    STAGES: tl.constexpr,
    COMPLEX: tl.constexpr = False,
    ACC: tl.constexpr = tl.float64,
):
    pid = tl.program_id(0).to(tl.int64)
    if INDEXED:
        row = tl.load(ROWS + pid)
    else:
        row = pid
    start = tl.load(RP + row).to(tl.int64)
    end = tl.load(RP + row + 1).to(tl.int64)
    lane = tl.arange(0, B)
    acc = tl.zeros((B,), ACC)
    imag = tl.zeros((B,), ACC)
    for step in tl.range(0, tl.cdiv(end - start, B), num_stages=STAGES):
        pos = start + step * B + lane
        mask = pos < end
        col = tl.load(CI + pos, mask, 0).to(tl.int64)
        pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
        acc = acc + pr
        if COMPLEX:
            imag = imag + pi
    _store_result(Y, row, tl.sum(acc, 0), tl.sum(imag, 0), True, COMPLEX)


@triton.jit
def bucket_rows_kernel(
    A,
    CI,
    RP,
    X,
    Y,
    ROWS,
    N,
    BATCH: tl.constexpr,
    B: tl.constexpr,
    MAX_SEGS: tl.constexpr,
    COMPLEX: tl.constexpr,
    ACC: tl.constexpr,
):
    """Legacy bucket scheduling, including sequential short-row batches."""
    pid = tl.program_id(0).to(tl.int64)
    lane = tl.arange(0, B)
    for batch in range(BATCH):
        index = pid * BATCH + batch
        active = index < N
        row = tl.load(ROWS + index, active, 0).to(tl.int64)
        start = tl.load(RP + row, active, 0).to(tl.int64)
        end = tl.load(RP + row + 1, active, 0).to(tl.int64)
        real = tl.zeros((B,), ACC)
        imag = tl.zeros((B,), ACC)
        for step in range(MAX_SEGS):
            pos = start + step * B + lane
            mask = active & (pos < end)
            col = tl.load(CI + pos, mask, 0).to(tl.int64)
            pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
            real += pr
            if COMPLEX:
                imag += pi
        _store_result(Y, row, tl.sum(real, 0), tl.sum(imag, 0), active, COMPLEX)


@triton.jit
def classify_kernel(
    RP,
    SHORT,
    MID,
    COUNTS,
    M,
    S: tl.constexpr,
    L: tl.constexpr,
    T: tl.constexpr,
    ADAPTIVE: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = r < M
    start = tl.load(RP + r, valid, 0).to(tl.int64)
    end = tl.load(RP + r + 1, valid, 0).to(tl.int64)
    length = end - start
    if ADAPTIVE:
        short = length <= S
        mid = (length > S) & (length <= L)
        counts = tl.where(length > L, tl.cdiv(length, T), 0)
    else:
        short = tl.full((B,), False, tl.int1)
        mid = tl.full((B,), False, tl.int1)
        counts = tl.cdiv(length, T)
    tl.store(SHORT + r, short.to(tl.int64), valid)
    tl.store(MID + r, mid.to(tl.int64), valid)
    tl.store(COUNTS + r, counts, valid)


@triton.jit
def compact_rows_kernel(SHORT, MID, SP, MP, SR, MR, M, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = r < M
    s = tl.load(SHORT + r, valid, 0) != 0
    m = tl.load(MID + r, valid, 0) != 0
    sp = tl.load(SP + r, valid, 0) - 1
    mp = tl.load(MP + r, valid, 0) - 1
    tl.store(SR + sp, r, valid & s)
    tl.store(MR + mp, r, valid & m)


@triton.jit
def _owner(PREFIX, index, M, valid):
    # First row whose inclusive segment offset is greater than index.
    lo = tl.full(index.shape, 0, tl.int64)
    hi = tl.full(index.shape, M, tl.int64)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        value = tl.load(PREFIX + mid, valid & (mid < M), 0)
        take = value <= index
        active = lo < hi
        lo = tl.where(active & take, mid + 1, lo)
        hi = tl.where(active & ~take, mid, hi)
    return lo


@triton.jit
def descriptors_kernel(
    PREFIX, RP, ROWS, STARTS, LENGTHS, TOTAL, M, T: tl.constexpr, B: tl.constexpr
):
    i = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = i < TOTAL
    row = _owner(PREFIX, i, M, valid)
    before = tl.load(PREFIX + row - 1, valid & (row > 0), 0)
    start = tl.load(RP + row, valid, 0).to(tl.int64) + (i - before) * T
    end = tl.load(RP + row + 1, valid, 0).to(tl.int64)
    tl.store(ROWS + i, row, valid)
    tl.store(STARTS + i, start, valid)
    tl.store(LENGTHS + i, tl.minimum(end - start, T), valid)


@triton.jit
def segment_kernel(
    A,
    CI,
    X,
    STARTS,
    LENGTHS,
    PARTIAL,
    B: tl.constexpr,
    STAGES: tl.constexpr,
    COMPLEX: tl.constexpr = False,
    ACC: tl.constexpr = tl.float64,
):
    pid = tl.program_id(0).to(tl.int64)
    start = tl.load(STARTS + pid)
    length = tl.load(LENGTHS + pid)
    lane = tl.arange(0, B)
    acc = tl.zeros((B,), ACC)
    imag = tl.zeros((B,), ACC)
    for step in tl.range(0, tl.cdiv(length, B), num_stages=STAGES):
        offset = step * B + lane
        mask = offset < length
        pos = start + offset
        col = tl.load(CI + pos, mask, 0).to(tl.int64)
        pr, pi = _product(A, X, pos, col, mask, COMPLEX, ACC)
        acc = acc + pr
        if COMPLEX:
            imag = imag + pi
    _store_result(PARTIAL, pid, tl.sum(acc, 0), tl.sum(imag, 0), True, COMPLEX)


@triton.jit
def reduce_level_kernel(
    PARTIAL,
    PREV_PREFIX,
    NEXT_PREFIX,
    OUTPUT,
    M,
    B: tl.constexpr,
    COMPLEX: tl.constexpr = False,
):
    pid = tl.program_id(0).to(tl.int64)
    # A one-element block lets the shared owner search serve scalar programs.
    ids = tl.full((1,), pid, tl.int64)
    row = tl.sum(_owner(NEXT_PREFIX, ids, M, tl.full((1,), True, tl.int1)), 0)
    group_base = tl.load(NEXT_PREFIX + row - 1, row > 0, 0)
    prev_base = tl.load(PREV_PREFIX + row - 1, row > 0, 0)
    prev_end = tl.load(PREV_PREFIX + row)
    pos = prev_base + (pid - group_base) * B + tl.arange(0, B)
    stride: tl.constexpr = 2 if COMPLEX else 1
    for component in tl.static_range(stride):
        value = tl.load(PARTIAL + pos * stride + component, pos < prev_end, 0)
        tl.store(OUTPUT + pid * stride + component, tl.sum(value, 0))


@triton.jit
def finish_kernel(
    PARTIAL,
    PREFIX,
    COUNTS,
    Y,
    M,
    WRITE_EMPTY: tl.constexpr,
    HAS_PARTIAL: tl.constexpr,
    B: tl.constexpr,
    COMPLEX: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = row < M
    count = tl.load(COUNTS + row, valid, 0)
    stride: tl.constexpr = 2 if COMPLEX else 1
    for component in tl.static_range(stride):
        if HAS_PARTIAL:
            pos = tl.load(PREFIX + row, valid, 0) - 1
            value = tl.load(PARTIAL + pos * stride + component, valid & (count > 0), 0)
        else:
            value = tl.full((B,), 0, PARTIAL.dtype.element_ty)
        mask = valid if WRITE_EMPTY else valid & (count > 0)
        tl.store(Y + row * stride + component, value, mask)


def build_plan(prepared, config, adaptive):
    """GPU membership/segment construction; no Python row classification or cache."""
    m, device = prepared.n_rows, prepared.data.device
    alloc = lambda n: torch.empty(n, dtype=torch.int64, device=device)
    short, mid, counts = alloc(m), alloc(m), alloc(m)
    split = config["row_split_reduce"]
    process = config["process"]
    block = process["block_size"]
    classify_kernel[(triton.cdiv(m, block),)](
        prepared.kernel_indptr,
        short,
        mid,
        counts,
        m,
        S=config["short_row_threshold"],
        L=config["split_row_threshold"],
        T=split["segment_nnz"],
        ADAPTIVE=adaptive,
        B=block,
        num_warps=process["num_warps"],
    )
    prefix = counts.cumsum(0)
    if adaptive:
        sp, mp = short.cumsum(0), mid.cumsum(0)
        ns, nm, total, maximum = torch.stack(
            (sp[-1], mp[-1], prefix[-1], counts.max())
        ).tolist()
        sr, mr = alloc(ns), alloc(nm)
        compact_rows_kernel[(triton.cdiv(m, block),)](
            short, mid, sp, mp, sr, mr, m, B=block, num_warps=process["num_warps"]
        )
    else:
        total, maximum = torch.stack((prefix[-1], counts.max())).tolist()
        sr = mr = None
    rows, starts, lengths = alloc(total), alloc(total), alloc(total)
    if total:
        descriptors_kernel[(triton.cdiv(total, block),)](
            prefix,
            prepared.kernel_indptr,
            rows,
            starts,
            lengths,
            total,
            m,
            T=split["segment_nnz"],
            B=block,
            num_warps=process["num_warps"],
        )
    levels = []
    next_counts = counts
    while maximum > 1:
        block = split["reduce_block_size"]
        next_counts = torch.div(next_counts + block - 1, block, rounding_mode="floor")
        next_prefix = next_counts.cumsum(0)
        next_total = int(next_prefix[-1].item())
        levels.append((next_prefix, next_total))
        maximum = triton.cdiv(maximum, block)
    return {
        "short_rows": sr,
        "mid_rows": mr,
        "counts": counts,
        "prefix": prefix,
        "segment_rows": rows,
        "starts": starts,
        "lengths": lengths,
        "total": total,
        "levels": levels,
    }


def compute(prepared, x, y, alg, config, plan=None):
    m = prepared.n_rows
    if not m:
        return y
    complex_input = prepared.data.is_complex()
    acc_dtype = (
        torch.float64
        if prepared.data.dtype in (torch.float32, torch.float64, torch.complex128)
        else torch.float32
    )
    acc = tl.float64 if acc_dtype == torch.float64 else tl.float32
    view = lambda t: (
        torch.view_as_real(t.resolve_conj()).reshape(-1) if complex_input else t
    )
    data, vector, output_view = view(prepared.data), view(x), view(y)
    args = (data, prepared.kernel_indices, prepared.kernel_indptr, vector, output_view)
    channels = 2 if complex_input else 1
    if alg in ("row_tile", "row_adaptive_split"):
        rows = None if plan is None else plan["short_rows"]
        n = m if rows is None else rows.numel()
        c = config["row_tile"]
        if n:
            row_tile_kernel[(triton.cdiv(n, c["rows_per_program"]),)](
                *args,
                prepared.kernel_indptr if rows is None else rows,
                n,
                INDEXED=rows is not None,
                R=c["rows_per_program"],
                V=c["lanes_per_row"],
                STAGES=c["loop_num_stages"],
                COMPLEX=complex_input,
                ACC=acc,
                num_warps=c["num_warps"],
                enable_fp_fusion=False,
            )
    if alg in ("row_vector", "row_adaptive_split"):
        rows = None if plan is None else plan["mid_rows"]
        n = m if rows is None else rows.numel()
        c = config["row_vector"]
        if n:
            row_vector_kernel[(n,)](
                *args,
                prepared.kernel_indptr if rows is None else rows,
                n,
                INDEXED=rows is not None,
                B=c["block_nnz"],
                STAGES=c["loop_num_stages"],
                COMPLEX=complex_input,
                ACC=acc,
                num_warps=c["num_warps"],
                enable_fp_fusion=False,
            )
    if plan is not None:
        c = config["row_split_reduce"]
        partial = torch.empty(
            plan["total"] * channels, dtype=acc_dtype, device=y.device
        )
        prefix = plan["prefix"]
        if plan["total"]:
            segment_kernel[(plan["total"],)](
                data,
                prepared.kernel_indices,
                vector,
                plan["starts"],
                plan["lengths"],
                partial,
                B=c["block_nnz"],
                STAGES=c["loop_num_stages"],
                COMPLEX=complex_input,
                ACC=acc,
                num_warps=c["num_warps"],
                enable_fp_fusion=False,
            )
            for next_prefix, total in plan["levels"]:
                output = torch.empty(total * channels, dtype=acc_dtype, device=y.device)
                reduce_level_kernel[(total,)](
                    partial,
                    prefix,
                    next_prefix,
                    output,
                    m,
                    B=c["reduce_block_size"],
                    COMPLEX=complex_input,
                    num_warps=c["reduce_num_warps"],
                )
                partial, prefix = output, next_prefix
        finish_block = config["process"]["block_size"]
        finish_kernel[(triton.cdiv(m, finish_block),)](
            partial,
            prefix,
            plan["counts"],
            output_view,
            m,
            WRITE_EMPTY=alg == "row_split_reduce",
            HAS_PARTIAL=plan["total"] > 0,
            B=finish_block,
            COMPLEX=complex_input,
            num_warps=c["reduce_num_warps"],
        )
    return y
