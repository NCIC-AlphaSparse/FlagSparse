"""Ascend-native compaction of sorted row-local SpGEMM products."""

import torch
import triton
import triton.language as tl
from triton.language.extra.cann.extension import sort as native_sort


@triton.jit
def _compact_sorted(columns, lengths, source_positions, order, positions, compact, ends, counts,
                     N_ROWS, GRID, BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    while offset < N_ROWS:
        keys = tl.load(columns + offset * BLOCK + lane)
        length = tl.load(lengths + offset).to(tl.int32)
        following = tl.gather(keys, tl.minimum(lane + 1, BLOCK - 1), 0)
        tail = (lane < length) & ((lane == BLOCK - 1) | (keys != following))
        count = tl.sum(tail.to(tl.int32), 0)
        last = native_sort(tl.where(tail, lane, BLOCK - 1).to(tl.float32), descending=False).to(tl.int32)
        unique = tl.gather(keys, last, 0)
        source = tl.load(source_positions + offset * BLOCK + lane)
        permutation = tl.load(order + offset * BLOCK + lane).to(tl.int32)
        position = tl.gather(source.to(tl.float32, bitcast=True), permutation, 0).to(tl.int32, bitcast=True)
        tl.store(positions + offset * BLOCK + lane, position)
        tl.store(compact + offset * BLOCK + lane, unique)
        tl.store(ends + offset * BLOCK + lane, last)
        tl.store(counts + offset, count)
        offset += GRID


def compact_sorted(columns, lengths, source_positions, order, max_programs):
    positions = torch.empty_like(source_positions)
    compact = torch.empty_like(columns)
    ends = torch.empty_like(source_positions)
    counts = torch.empty(columns.shape[0], dtype=torch.int64, device=columns.device)
    grid = min(8192, max_programs, columns.shape[0])
    if grid:
        _compact_sorted[(grid,)](columns, lengths, source_positions, order, positions, compact, ends, counts,
                                 columns.shape[0], grid, columns.shape[1], multibuffer=False)
    return positions, compact, ends, counts
