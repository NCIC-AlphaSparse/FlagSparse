# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""TLE-free MACA SpGEMM. Torch supplies storage, not sparse computation."""

import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(ai, ap, bp, work, pref, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(ap + row).to(tl.int64)
    stop = tl.load(ap + row + 1).to(tl.int64)
    total = tl.full((), 0, tl.int64)
    lane = tl.arange(0, BLOCK)
    for base in range(start, stop, BLOCK):
        pos = base + lane
        col = tl.load(ai + pos, pos < stop, 0)
        begin = tl.load(bp + col, pos < stop, 0).to(tl.int64)
        end = tl.load(bp + col + 1, pos < stop, 0).to(tl.int64)
        length = end - begin
        prefix = tl.cumsum(length, 0)
        tl.store(pref + pos, total + prefix - length, pos < stop)
        total += tl.sum(length, 0)
    tl.store(work + row, total)


@triton.jit
def _scan_blocks(counts, ptr, sums, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    pos = block * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(counts + pos, pos < N, 0).to(tl.int64)
    prefix = tl.cumsum(values, 0)
    tl.store(ptr + pos + 1, prefix, pos < N)
    tl.store(sums + block, tl.sum(values, 0))
    if block == 0:
        tl.store(ptr, 0)


@triton.jit
def _scan_add(ptr, block_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    pos = block * BLOCK + tl.arange(0, BLOCK)
    carry = tl.load(block_ptr + block)
    value = tl.load(ptr + pos + 1, pos < N, 0)
    tl.store(ptr + pos + 1, value + carry, pos < N)


def _prefix_ptr(counts):
    n = counts.numel()
    ptr = torch.empty(n + 1, dtype=torch.int64, device=counts.device)
    blocks = max(1, triton.cdiv(n, 1024))
    sums = torch.empty(blocks, dtype=torch.int64, device=counts.device)
    _scan_blocks[(blocks,)](counts, ptr, sums, n, 1024)
    if blocks > 1:
        block_ptr = _prefix_ptr(sums)
        _scan_add[(blocks,)](ptr, block_ptr, n, 1024)
    return ptr


@triton.jit
def _capacity(work, sizes, counts, N: tl.constexpr, COLS: tl.constexpr, LIMIT: tl.constexpr,
              BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(work + row, row < N, 0)
    # Long rows use a global hash table with guaranteed load <= 1/2. Short
    # rows need only their product bound for a compact intermediate result.
    h = tl.minimum(w, COLS) * 2
    h = tl.maximum(h, 1) - 1
    h = h | (h >> 1)
    h = h | (h >> 2)
    h = h | (h >> 4)
    h = h | (h >> 8)
    h = h | (h >> 16)
    h = h | (h >> 32)
    tl.store(sizes + row, tl.where(w <= LIMIT, w, h + 1), row < N)
    tl.store(counts + row, 0, row < N)


@triton.jit
def _products(ad, ai, ap, pref, bd, bi, bp, row, work, q, BLOCK: tl.constexpr):
    lo = tl.full((BLOCK,), tl.load(ap + row), tl.int64)
    stop = tl.load(ap + row + 1).to(tl.int64)
    hi = tl.full((BLOCK,), stop, tl.int64)
    while tl.sum((hi - lo > 1).to(tl.int32), 0) > 0:
        mid = (lo + hi) // 2
        p = tl.load(pref + mid, mid < stop, 0x7fffffffffffffff)
        take = p <= q
        lo = tl.where(take, mid, lo)
        hi = tl.where(take, hi, mid)
    col = tl.load(ai + lo, q < work, 0)
    p = tl.load(pref + lo, q < work, 0)
    bpos = tl.load(bp + col, q < work, 0) + q - p
    key = tl.load(bi + bpos, q < work, -1)
    av = tl.load(ad + lo, q < work, 0.0)
    bv = tl.load(bd + bpos, q < work, 0.0)
    return key, av * bv


@triton.jit
def _segment_add(ka, va, kb, vb):
    return kb, tl.where(ka == kb, va + vb, vb)


@triton.jit
def _scan_last_two(ka, next_a, na, kb, next_b, nb):
    return kb, tl.where(nb > 1, next_b, ka), na + nb


@triton.jit
def _sort_rows(ad, ai, ap, pref, bd, bi, bp, work, slots, keys, values, counts,
               LOWER: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    w = tl.load(work + row)
    if (w > LOWER) & (w <= BLOCK):
        q = tl.arange(0, BLOCK)
        key, _ = _products(ad, ai, ap, pref, bd, bi, bp, row, w, q, BLOCK)
        # MACA cannot lower dynamic cross-lane gather. Carry product positions
        # through scalar sort, then load operands directly in the sorted order.
        order = tl.sort((key.to(tl.uint32).to(tl.uint64) << 32) | q.to(tl.uint64), descending=False)
        key = (order >> 32).to(tl.int32)
        _, value = _products(ad, ai, ap, pref, bd, bi, bp, row, w,
                             (order & 0xffffffff).to(tl.int32), BLOCK)
        _, total = tl.associative_scan((key, value), 0, _segment_add)
        _, following, _ = tl.associative_scan(
            (key, key, tl.full((BLOCK,), 1, tl.int32)), 0, _scan_last_two, reverse=True
        )
        last = (key >= 0) & ((q == BLOCK - 1) | (key != following))
        rank = tl.cumsum(last.to(tl.int32), 0) - 1
        base = tl.load(slots + row)
        tl.store(keys + base + rank, key, last)
        tl.store(values + base + rank, total, last)
        tl.store(counts + row, tl.sum(last.to(tl.int32), 0))


@triton.jit
def _hash_init(work, slots, keys, values, LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(work + row) > LIMIT:
        start = tl.load(slots + row)
        end = tl.load(slots + row + 1)
        for base in range(start, end, BLOCK):
            pos = base + tl.arange(0, BLOCK)
            tl.store(keys + pos, -1, pos < end)
            tl.store(values + pos, 0.0, pos < end)


@triton.jit
def _hash_rows(ad, ai, ap, pref, bd, bi, bp, work, slots, keys, values, counts,
               LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    w = tl.load(work + row)
    if w > LIMIT:
        start = tl.load(slots + row)
        cap = tl.load(slots + row + 1) - start
        a_start = tl.load(ap + row).to(tl.int64)
        a_end = tl.load(ap + row + 1).to(tl.int64)
        count = 0
        for base in range(0, w, BLOCK):
            q = base + tl.arange(0, BLOCK)
            valid = q < w
            lo = tl.full((BLOCK,), a_start, tl.int64)
            hi = tl.full((BLOCK,), a_end, tl.int64)
            while tl.sum((hi - lo > 1).to(tl.int32), 0) > 0:
                mid = (lo + hi) // 2
                p = tl.load(pref + mid, mid < a_end, 0x7fffffffffffffff)
                take = p <= q
                lo = tl.where(take, mid, lo)
                hi = tl.where(take, hi, mid)
            col = tl.load(ai + lo, valid, 0)
            p = tl.load(pref + lo, valid, 0)
            bpos = tl.load(bp + col, valid, 0) + q - p
            key = tl.load(bi + bpos, valid, -1)
            av = tl.load(ad + lo, valid, 0.0)
            bv = tl.load(bd + bpos, valid, 0.0)
            prod = av * bv
            h = (key.to(tl.uint32) * 2654435761).to(tl.int64) & (cap - 1)
            done = ~valid
            while tl.sum((~done).to(tl.int32), 0) > 0:
                found = tl.atomic_cas(
                    keys + start + h, tl.full((BLOCK,), -1, tl.int32),
                    tl.where(done, -1, key).to(tl.int32), sem="relaxed"
                )
                hit = (found == -1) | (found == key)
                count += tl.sum(((~done) & (found == -1)).to(tl.int32), 0)
                tl.atomic_add(values + start + h, prod, (~done) & hit, sem="relaxed")
                done = done | hit
                h = (h + 1) & (cap - 1)
        tl.store(counts + row, count)


@triton.jit
def _compact(work, slots, keys, values, counts, ptr, ci, cd,
             LIMIT: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    base = tl.load(slots + row)
    dst = tl.load(ptr + row)
    count = tl.load(counts + row)
    if tl.load(work + row) <= LIMIT:
        for offset in range(0, count, BLOCK):
            lane = offset + tl.arange(0, BLOCK)
            key = tl.load(keys + base + lane, lane < count, 0)
            value = tl.load(values + base + lane, lane < count, 0.0)
            tl.store(ci + dst + lane, key, lane < count)
            tl.store(cd + dst + lane, value, lane < count)
    else:
        cap = tl.load(slots + row + 1) - base
        emitted = 0
        for offset in range(0, cap, BLOCK):
            lane = offset + tl.arange(0, BLOCK)
            key = tl.load(keys + base + lane, lane < cap, -1)
            value = tl.load(values + base + lane, lane < cap, 0.0)
            valid = (lane < cap) & (key >= 0)
            rank = tl.cumsum(valid.to(tl.int32), 0) - 1
            tl.store(ci + dst + emitted + rank, key, valid)
            tl.store(cd + dst + emitted + rank, value, valid)
            emitted += tl.sum(valid.to(tl.int32), 0)


def build_metadata(indices, indptr, b_indptr):
    work = torch.empty(indptr.numel() - 1, dtype=torch.int64, device=indices.device)
    pref = torch.empty(indices.numel(), dtype=torch.int64, device=indices.device)
    if work.numel():
        _metadata[(work.numel(),)](indices, indptr, b_indptr, work, pref, 32, num_warps=1)
    return work, pref


def compute(prepared):
    rows = prepared.n_rows
    device = prepared.a_data.device
    sizes = torch.empty(rows, dtype=torch.int64, device=device)
    counts = torch.empty(rows, dtype=torch.int64, device=device)
    limit = 4096
    if rows:
        _capacity[(triton.cdiv(rows, 256),)](
            prepared.a_row_work, sizes, counts, rows, prepared.n_cols, limit, 256
        )
    slots = _prefix_ptr(sizes)
    total = int(slots[-1].item())
    keys = torch.empty(total, dtype=torch.int32, device=device)
    values = torch.empty(total, dtype=prepared.a_data.dtype, device=device)
    args = (prepared.a_data, prepared.a_indices, prepared.a_indptr, prepared.a_pref,
            prepared.b_data, prepared.b_indices, prepared.b_indptr, prepared.a_row_work,
            slots, keys, values, counts)
    if rows:
        lower = 0
        for block, warps in ((64, 1), (256, 2), (1024, 4), (4096, 8)):
            _sort_rows[(rows,)](*args, lower, block, num_warps=warps, enable_fp_fusion=False)
            lower = block
        _hash_init[(rows,)](prepared.a_row_work, slots, keys, values, limit, 256)
        # C550 CAS/add loses lanes with a 128-element multi-warp layout; keep
        # each probing step within its native 64-lane warp (tested for FP64).
        _hash_rows[(rows,)](*args, limit, 64, num_warps=1, enable_fp_fusion=False)
    ptr = _prefix_ptr(counts)
    nnz = int(ptr[-1].item())
    data = torch.empty(nnz, dtype=prepared.a_data.dtype, device=device)
    indices = torch.empty(nnz, dtype=torch.int32, device=device)
    if rows:
        _compact[(rows,)](prepared.a_row_work, slots, keys, values, counts, ptr,
                         indices, data, limit, 256, num_warps=4)
    return data, indices, ptr
