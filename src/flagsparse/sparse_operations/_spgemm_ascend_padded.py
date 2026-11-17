"""Row-local Ascend SpGEMM with vectorized, padded right-row gathers."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pad_row(pointer, source, target, row, WIDTH: tl.constexpr, PARTS: tl.constexpr):
    lane = tl.arange(0, WIDTH * PARTS)
    start = tl.load(pointer + row).to(tl.int32)
    end = tl.load(pointer + row + 1).to(tl.int32)
    data = tl.load(source + start * PARTS + lane, lane < (end - start) * PARTS, 0)
    if PARTS == 2:
        columns = tl.arange(0, WIDTH)
        real = tl.gather(data, columns * 2, 0)
        imag = tl.gather(data, columns * 2 + 1, 0)
        tl.store(target + row * WIDTH * 2 + columns, real)
        tl.store(target + row * WIDTH * 2 + WIDTH + columns, imag)
    else:
        tl.store(target + row * WIDTH + lane, data.to(tl.float32))


@triton.jit
def _pad(pointer, source, target, N_ROWS, GRID, WIDTH: tl.constexpr, PARTS: tl.constexpr):
    row = tl.program_id(0)
    while row < N_ROWS:
        _pad_row(pointer, source, target, row, WIDTH, PARTS)
        row += GRID


@triton.jit
def _positions(pointer, prefix, row, BLOCK: tl.constexpr, DEGREE: tl.constexpr, WIDTH: tl.constexpr):
    start = tl.load(pointer + row).to(tl.int32)
    end = tl.load(pointer + row + 1).to(tl.int32)
    source_lane = tl.arange(0, DEGREE)
    preceding = tl.load(prefix + start + source_lane, source_lane < end - start, 0).to(tl.int32)
    boundary = tl.load(prefix + start + source_lane + 1, source_lane < end - start, 0).to(tl.int32)
    base = tl.load(prefix + start).to(tl.int32)
    stop = tl.load(prefix + end).to(tl.int32)
    lane = tl.arange(0, BLOCK)
    product = base + tl.minimum(lane, stop - base - 1)
    lower = tl.full((BLOCK,), 0, tl.int32)
    upper = tl.full((BLOCK,), end - start, tl.int32)
    for step in tl.static_range(DEGREE.bit_length()):
        middle = (lower + upper) // 2
        probe = tl.minimum(middle, end - start - 1)
        limit = tl.gather(boundary.to(tl.float32, bitcast=True), probe, 0).to(tl.int32, bitcast=True)
        advance = (middle < end - start) & (limit <= product)
        lower = tl.where(advance, middle + 1, lower)
        upper = tl.where(advance, upper, middle)
    selected_prefix = tl.gather(preceding.to(tl.float32, bitcast=True), lower, 0).to(tl.int32, bitcast=True)
    return start, end, lower, lower * WIDTH + product - selected_prefix, stop - base


@triton.jit
def _project_row(pointer, prefix, padded, rows, columns, source_positions, offset, WIDTH: tl.constexpr,
             DEGREE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.load(rows + offset)
    start, end, source, position, work = _positions(pointer, prefix, row, BLOCK, DEGREE, WIDTH)
    source_lane = tl.arange(0, DEGREE * WIDTH)
    data = tl.load(padded + start * WIDTH + source_lane, source_lane < (end - start) * WIDTH, 0.)
    result = tl.gather(data, position, 0)
    lane = tl.arange(0, BLOCK)
    tl.store(columns + offset * BLOCK + lane, tl.where(lane < work, result, 2147483647.))
    tl.store(source_positions + offset * BLOCK + lane, tl.where(lane < work, position, 0))


@triton.jit
def _project(pointer, prefix, padded, rows, columns, source_positions, N_ROWS, GRID, WIDTH: tl.constexpr,
             DEGREE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    while offset < N_ROWS:
        _project_row(pointer, prefix, padded, rows, columns, source_positions, offset, WIDTH, DEGREE, BLOCK)
        offset += GRID


@triton.jit
def _unique_row(columns, rows, work, compact_columns, compact_ends, counts, source_positions,
                order, positions, offset, BLOCK: tl.constexpr):
    row = tl.load(rows + offset)
    length = tl.load(work + row).to(tl.int32)
    lane = tl.arange(0, BLOCK)
    keys = tl.load(columns + offset * BLOCK + lane)
    previous = tl.gather(keys, tl.maximum(lane - 1, 0), 0)
    heads = (lane < length) & ((lane == 0) | (keys != previous))
    ranks = tl.cumsum(heads.to(tl.int32), 0) - 1
    count = tl.sum(heads.to(tl.int32), 0)
    lower = tl.full((BLOCK,), 0, tl.int32)
    upper = tl.full((BLOCK,), length, tl.int32)
    for step in tl.static_range(BLOCK.bit_length()):
        middle = (lower + upper) // 2
        probe = tl.minimum(middle, length - 1)
        rank = tl.gather(ranks.to(tl.float32, bitcast=True), probe, 0).to(tl.int32, bitcast=True)
        advance = (middle < length) & (rank <= lane)
        lower = tl.where(advance, middle + 1, lower)
        upper = tl.where(advance, upper, middle)
    ends = tl.maximum(lower - 1, 0)
    compact = tl.gather(keys, ends, 0)
    source = tl.load(source_positions + offset * BLOCK + lane)
    permutation = tl.load(order + offset * BLOCK + lane).to(tl.int32)
    position = tl.gather(source.to(tl.float32, bitcast=True), permutation, 0).to(tl.int32, bitcast=True)
    tl.store(positions + offset * BLOCK + lane, position)
    tl.store(compact_columns + offset * BLOCK + lane, compact)
    tl.store(compact_ends + offset * BLOCK + lane, ends)
    tl.store(counts + row, count)


@triton.jit
def _unique(columns, rows, work, compact_columns, compact_ends, counts, source_positions,
            order, positions, N_ROWS, GRID, BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    while offset < N_ROWS:
        _unique_row(columns, rows, work, compact_columns, compact_ends, counts,
                    source_positions, order, positions, offset, BLOCK)
        offset += GRID


@triton.jit
def _finalize_row(compact_columns, rows, pointer, output_columns, offset, BLOCK: tl.constexpr):
    row = tl.load(rows + offset)
    start = tl.load(pointer + row).to(tl.int32)
    end = tl.load(pointer + row + 1).to(tl.int32)
    lane = tl.arange(0, BLOCK)
    keys = tl.load(compact_columns + offset * BLOCK + lane)
    tl.store(output_columns + start + lane, keys.to(tl.int32), lane < end - start)


@triton.jit
def _finalize(compact_columns, rows, pointer, output_columns, N_ROWS, GRID, BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    while offset < N_ROWS:
        _finalize_row(compact_columns, rows, pointer, output_columns, offset, BLOCK)
        offset += GRID


@triton.jit
def _segment_add(left_key, left_value, right_key, right_value):
    return right_key, tl.where(left_key == right_key, left_value + right_value, right_value)


@triton.jit
def _numeric_row(pointer, prefix, padded, left, rows, columns, positions, compact_ends,
             output_pointer, output, N_OUTPUTS, offset, WIDTH: tl.constexpr, DEGREE: tl.constexpr,
             BLOCK: tl.constexpr, COMPLEX: tl.constexpr, REDUCE: tl.constexpr = True):
    row = tl.load(rows + offset)
    start = tl.load(pointer + row).to(tl.int32)
    end = tl.load(pointer + row + 1).to(tl.int32)
    work = (tl.load(prefix + end) - tl.load(prefix + start)).to(tl.int32)
    lane = tl.arange(0, BLOCK)
    position = tl.load(positions + offset * BLOCK + lane).to(tl.int32)
    source = position // WIDTH
    keys = tl.load(columns + offset * BLOCK + lane)
    output_start = tl.load(output_pointer + row).to(tl.int32)
    output_end = tl.load(output_pointer + row + 1).to(tl.int32)
    last = tl.load(compact_ends + offset * BLOCK + lane,
                   lane < output_end - output_start, 0)
    if COMPLEX:
        left_lane = tl.arange(0, DEGREE * 2)
        first = tl.load(left + start * 2 + left_lane, left_lane < (end - start) * 2, 0.)
        real_left = tl.gather(first, source * 2, 0)
        imag_left = tl.gather(first, source * 2 + 1, 0)
        if WIDTH == 1:
            source_lane = tl.arange(0, DEGREE * 2)
            right = tl.load(padded + start * 2 + source_lane, source_lane < (end - start) * 2, 0.)
            real_right = tl.gather(right, position * 2, 0)
            imag_right = tl.gather(right, position * 2 + 1, 0)
        elif DEGREE == 1:
            source_lane = tl.arange(0, WIDTH)
            right_real = tl.load(padded + start * WIDTH * 2 + source_lane)
            right_imag = tl.load(padded + start * WIDTH * 2 + WIDTH + source_lane)
        else:
            source_rows = tl.arange(0, DEGREE)
            source_lane = tl.arange(0, WIDTH)
            source_position = (start + source_rows[:, None]) * WIDTH * 2 + source_lane[None, :]
            right_real = tl.load(padded + source_position, source_rows[:, None] < end - start, 0.)
            right_real = tl.reshape(right_real, (DEGREE * WIDTH,))
            right_imag = tl.load(padded + source_position + WIDTH, source_rows[:, None] < end - start, 0.)
            right_imag = tl.reshape(right_imag, (DEGREE * WIDTH,))
        if WIDTH != 1:
            real_right = tl.gather(right_real, position, 0)
            imag_right = tl.gather(right_imag, position, 0)
        part = tl.program_id(1)
        product = tl.where(part == 0, real_left * real_right - imag_left * imag_right,
                           real_left * imag_right + imag_left * real_right)
        ordered = tl.where(lane < work, product, 0.)
        if REDUCE:
            _, total = tl.associative_scan((keys, ordered), 0, _segment_add)
            values = tl.gather(total, last, 0)
            tl.store(output + part * N_OUTPUTS + output_start + lane, values, lane < output_end - output_start)
        else:
            tl.store(output + part * N_OUTPUTS + offset * BLOCK + lane, ordered)
    else:
        source_lane = tl.arange(0, DEGREE * WIDTH)
        right = tl.load(padded + start * WIDTH + source_lane, source_lane < (end - start) * WIDTH, 0.)
        left_lane = tl.arange(0, DEGREE)
        first = tl.load(left + start + left_lane, left_lane < end - start, 0.)
        product = tl.gather(first, source, 0) * tl.gather(right, position, 0)
        ordered = tl.where(lane < work, product, 0.)
        if REDUCE:
            _, total = tl.associative_scan((keys, ordered), 0, _segment_add)
            values = tl.gather(total, last, 0)
            tl.store(output + output_start + lane, values, lane < output_end - output_start)
        else:
            tl.store(output + offset * BLOCK + lane, ordered)


@triton.jit
def _numeric(pointer, prefix, padded, left, rows, columns, positions, compact_ends,
             output_pointer, output, N_OUTPUTS, N_ROWS, GRID, WIDTH: tl.constexpr,
             DEGREE: tl.constexpr, BLOCK: tl.constexpr, COMPLEX: tl.constexpr):
    offset = tl.program_id(0)
    while offset < N_ROWS:
        _numeric_row(pointer, prefix, padded, left, rows, columns, positions, compact_ends,
                     output_pointer, output, N_OUTPUTS, offset, WIDTH, DEGREE, BLOCK, COMPLEX)
        offset += GRID


@triton.jit
def _products(pointer, prefix, padded, left, rows, columns, positions, compact_ends,
              output_pointer, output, N_OUTPUTS, N_ROWS, GRID, WIDTH: tl.constexpr,
              DEGREE: tl.constexpr, BLOCK: tl.constexpr, COMPLEX: tl.constexpr):
    offset = tl.program_id(0)
    while offset < N_ROWS:
        _numeric_row(pointer, prefix, padded, left, rows, columns, positions, compact_ends,
                     output_pointer, output, N_OUTPUTS, offset, WIDTH, DEGREE, BLOCK, COMPLEX, False)
        offset += GRID


@triton.jit
def _bounded_reduce(keys, products, ends, rows, pointer, output, N_OUTPUTS, N_ROWS, GRID,
                     BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    part = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    while offset < N_ROWS:
        row = tl.load(rows + offset)
        start = tl.load(pointer + row).to(tl.int32)
        stop = tl.load(pointer + row + 1).to(tl.int32)
        key = tl.load(keys + offset * BLOCK + lane)
        ordered = tl.load(products + part * N_ROWS * BLOCK + offset * BLOCK + lane)
        last = tl.load(ends + offset * BLOCK + lane, lane < stop - start, 0)
        previous = tl.gather(last.to(tl.float32, bitcast=True), tl.maximum(lane - 1, 0), 0).to(tl.int32, bitcast=True)
        first = tl.where(lane == 0, 0, previous + 1)
        maximum = tl.max(tl.where(lane < stop - start, last - first + 1, 0), 0)
        total = ordered
        distance = 1
        while distance < maximum:
            index = tl.maximum(lane - distance, 0)
            preceding = tl.gather(total, index, 0)
            preceding_key = tl.gather(key, index, 0)
            total = total + tl.where((lane >= distance) & (preceding_key == key), preceding, 0.)
            distance *= 2
        values = tl.gather(total, last, 0)
        tl.store(output + part * N_OUTPUTS + start + lane, values, lane < stop - start)
        offset += GRID


@triton.jit
def _scan_reduce(keys, products, ends, rows, pointer, output, N_OUTPUTS, N_ROWS, GRID,
                 BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    part = tl.program_id(1)
    lane = tl.arange(0, BLOCK)
    while offset < N_ROWS:
        row = tl.load(rows + offset)
        start = tl.load(pointer + row).to(tl.int32)
        stop = tl.load(pointer + row + 1).to(tl.int32)
        key = tl.load(keys + offset * BLOCK + lane)
        ordered = tl.load(products + part * N_ROWS * BLOCK + offset * BLOCK + lane)
        last = tl.load(ends + offset * BLOCK + lane, lane < stop - start, 0)
        unused, total = tl.associative_scan((key, ordered), 0, _segment_add)
        values = tl.gather(total, last, 0)
        tl.store(output + part * N_OUTPUTS + start + lane, values, lane < stop - start)
        offset += GRID

@triton.jit
def _project_wide(pointer, prefix, padded, rows, columns, positions, N_ROWS, GRID,
             WIDTH: tl.constexpr, DEGREE: tl.constexpr, BLOCK: tl.constexpr):
    CHUNK: tl.constexpr = 8192
    offset = tl.program_id(0)
    while offset < N_ROWS:
        row = tl.load(rows + offset)
        start, end, source, position, work = _positions(pointer, prefix, row, BLOCK, DEGREE, WIDTH)
        lane = tl.arange(0, BLOCK)
        source_lane = tl.arange(0, CHUNK)
        selected = tl.full((BLOCK,), 0., tl.float32)
        base = 0
        while base < (end - start) * WIDTH:
            data = tl.load(padded + start * WIDTH + base + source_lane,
                           base + source_lane < (end - start) * WIDTH, 0.)
            local = tl.minimum(tl.maximum(position - base, 0), CHUNK - 1)
            values = tl.gather(data, local, 0)
            selected = tl.where((position >= base) & (position < base + CHUNK), values, selected)
            base += CHUNK
        tl.store(columns + offset * BLOCK + lane, tl.where(lane < work, selected, 2147483647.))
        tl.store(positions + offset * BLOCK + lane, tl.where(lane < work, position, 0))
        offset += GRID

@triton.jit
def _products_wide(pointer, prefix, padded, left, rows, positions, output,
              N_ROWS, GRID, WIDTH: tl.constexpr, DEGREE: tl.constexpr,
              BLOCK: tl.constexpr, COMPLEX: tl.constexpr, CHUNK: tl.constexpr):
    PARTS: tl.constexpr = 2 if COMPLEX else 1
    offset = tl.program_id(0)
    part = tl.program_id(1)
    while offset < N_ROWS:
        row = tl.load(rows + offset)
        start = tl.load(pointer + row).to(tl.int32)
        end = tl.load(pointer + row + 1).to(tl.int32)
        work = (tl.load(prefix + end) - tl.load(prefix + start)).to(tl.int32)
        lane = tl.arange(0, BLOCK)
        position = tl.load(positions + offset * BLOCK + lane).to(tl.int32)
        source = position // WIDTH
        left_lane = tl.arange(0, DEGREE * PARTS)
        first = tl.load(left + start * PARTS + left_lane,
                        left_lane < (end - start) * PARTS, 0.)
        source_lane = tl.arange(0, CHUNK)
        base = tl.program_id(2) * CHUNK
        chunk_lane = base + source_lane
        if COMPLEX:
            if WIDTH == 1:
                address = (start + chunk_lane) * 2
                right_real = tl.load(padded + address + part, chunk_lane < end - start, 0.)
                right_imag = tl.load(padded + address + 1 - part, chunk_lane < end - start, 0.)
            else:
                source_rows = tl.arange(0, CHUNK // WIDTH)
                source_columns = tl.arange(0, WIDTH)
                selected_rows = start + base // WIDTH + source_rows
                address = selected_rows[:, None] * WIDTH * 2 + source_columns[None, :]
                right_real = tl.load(padded + address + part * WIDTH, selected_rows[:, None] < end, 0.)
                right_imag = tl.load(padded + address + (1 - part) * WIDTH, selected_rows[:, None] < end, 0.)
                right_real = tl.reshape(right_real, (CHUNK,))
                right_imag = tl.reshape(right_imag, (CHUNK,))
        else:
            right_real = tl.load(padded + start * WIDTH + chunk_lane,
                                 chunk_lane < (end - start) * WIDTH, 0.)
        local = tl.minimum(tl.maximum(position - base, 0), CHUNK - 1)
        active = (position >= base) & (position < base + CHUNK)
        selected_real = tl.gather(right_real, local, 0)
        if COMPLEX:
            selected_imag = tl.gather(right_imag, local, 0)
        if COMPLEX:
            first_real = tl.gather(first, source * 2, 0)
            first_imag = tl.gather(first, source * 2 + 1, 0)
            product = first_real * selected_real + tl.where(part == 0, -1., 1.) * first_imag * selected_imag
        else:
            product = tl.gather(first, source, 0) * selected_real
        output_base = (tl.program_id(2) * PARTS + tl.program_id(1)) * N_ROWS * BLOCK
        tl.store(output + output_base + offset * BLOCK + lane,
                 tl.where((lane < work) & active, product, 0.))
        offset += GRID


SCAN_BLOCK = 4096


SCAN_THRESHOLD = 524288


@triton.jit
def _block_sums(values, sums, SIZE, GRID, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    while block * BLOCK < SIZE:
        index = block * BLOCK + lane
        value = tl.load(values + index, index < SIZE, 0)
        tl.store(sums + block, tl.sum(value, 0))
        block += GRID


@triton.jit
def _scan_blocks(values, prefix, offsets, SIZE, GRID, BLOCK: tl.constexpr, MULTI: tl.constexpr):
    block = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    while block * BLOCK < SIZE:
        index = block * BLOCK + lane
        value = tl.load(values + index, index < SIZE, 0)
        scanned = tl.cumsum(value, 0)
        if MULTI:
            scanned += tl.load(offsets + block)
        tl.store(prefix + index + 1, scanned, index < SIZE)
        block += GRID
    if tl.program_id(0) == 0:
        tl.store(prefix, 0)


def _hierarchical_prefix(values, scratch):
    size = values.numel()
    blocks = triton.cdiv(size, SCAN_BLOCK)
    grid = min(8192, blocks)
    prefix = torch.empty(size + 1, dtype=values.dtype, device=values.device)
    if blocks > 1:
        totals = torch.empty(blocks, dtype=values.dtype, device=values.device)
        _block_sums[(grid,)](values, totals, size, grid, SCAN_BLOCK, multibuffer=False)
        offsets = _hierarchical_prefix(totals, scratch)
        scratch.extend((totals, offsets))
    else:
        offsets = prefix
    _scan_blocks[(grid,)](values, prefix, offsets, size, grid, SCAN_BLOCK, blocks > 1,
                         multibuffer=False)
    return prefix


def prefix_sum(values, scratch):
    if (values.numel() == 0 or values.numel() < SCAN_THRESHOLD
            or values.dtype != torch.int32 or not values.is_contiguous()):
        prefix = torch.zeros(values.numel() + 1, dtype=values.dtype, device=values.device)
        prefix[1:] = torch.cumsum(values, 0, dtype=values.dtype)
        return prefix
    return _hierarchical_prefix(values, scratch)


@triton.jit
def _select_rows(source, indices, target, N_ROWS, GRID, WIDTH: tl.constexpr, TILE: tl.constexpr):
    offset = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    column = tl.arange(0, WIDTH)
    while offset < N_ROWS:
        row = offset + row_lane
        index = tl.load(indices + row, row < N_ROWS, 0)
        values = tl.load(source + index[:, None] * WIDTH + column[None, :], row[:, None] < N_ROWS, 0.)
        tl.store(target + row[:, None] * WIDTH + column[None, :], values, row[:, None] < N_ROWS)
        offset += GRID * TILE


@triton.jit
def _pad_single_rows(pointer, source, target, N_ROWS, NNZ, GRID,
                    PARTS: tl.constexpr, TILE: tl.constexpr):
    row_base = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    source_lane = tl.arange(0, TILE * PARTS)
    while row_base < N_ROWS:
        row = row_base + row_lane
        start = tl.load(pointer + row, row < N_ROWS, NNZ).to(tl.int32)
        stop = tl.load(pointer + row + 1, row < N_ROWS, NNZ).to(tl.int32)
        base = tl.load(pointer + row_base).to(tl.int32)
        data = tl.load(source + base * PARTS + source_lane,
                       base * PARTS + source_lane < NNZ * PARTS, 0).to(tl.float32)
        position = tl.minimum((start - base) * PARTS, TILE * PARTS - PARTS)
        real = tl.gather(data, position, 0)
        if PARTS == 2:
            imag = tl.gather(data, position + 1, 0)
            values = tl.interleave(tl.where(start < stop, real, 0.), tl.where(start < stop, imag, 0.))
        else:
            values = tl.where(start < stop, real, 0.)
        tl.store(target + row_base * PARTS + source_lane, values,
                 row_base * PARTS + source_lane < N_ROWS * PARTS)
        row_base += GRID * TILE


@triton.jit
def _pad_rows(pointer, source, target, N_ROWS, NNZ, GRID,
             WIDTH: tl.constexpr, PARTS: tl.constexpr, TILE: tl.constexpr):
    row_base = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, WIDTH * PARTS)
    source_lane = tl.arange(0, TILE * WIDTH * PARTS)
    while row_base < N_ROWS:
        row = row_base + row_lane
        start = tl.load(pointer + row, row < N_ROWS, NNZ).to(tl.int32)
        stop = tl.load(pointer + row + 1, row < N_ROWS, NNZ).to(tl.int32)
        base = tl.load(pointer + row_base).to(tl.int32)
        data = tl.load(source + base * PARTS + source_lane,
                       base * PARTS + source_lane < NNZ * PARTS, 0).to(tl.float32)
        local = lane % WIDTH * PARTS + lane // WIDTH
        position = (start - base)[:, None] * PARTS + local[None, :]
        position = tl.minimum(position, TILE * WIDTH * PARTS - 1)
        values = tl.gather(data, tl.reshape(position, (TILE * WIDTH * PARTS,)), 0)
        values = tl.reshape(values, (TILE, WIDTH * PARTS))
        valid = local[None, :] < (stop - start)[:, None] * PARTS
        values = tl.where(valid, values, 0.)
        tl.store(target + row[:, None] * WIDTH * PARTS + lane[None, :], values, row[:, None] < N_ROWS)
        row_base += GRID * TILE


@triton.jit
def _project_batched(pointer, prefix, padded, rows, columns, positions, starts, stops,
              N_ROWS, GRID, WIDTH: tl.constexpr, DEGREE: tl.constexpr, BLOCK: tl.constexpr,
              TILE: tl.constexpr, SOURCE: tl.constexpr, NNZA: tl.constexpr):
    offset = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, BLOCK)
    prefix_lane = tl.arange(0, SOURCE * 2)
    source_lane = tl.arange(0, SOURCE * WIDTH)
    while offset < N_ROWS:
        base = tl.load(starts + offset).to(tl.int32)
        end = tl.load(stops + tl.minimum(offset + TILE - 1, N_ROWS - 1)).to(tl.int32)
        if end - base > SOURCE:
            local = offset
            while local < tl.minimum(offset + TILE, N_ROWS):
                _project_row(pointer, prefix, padded, rows, columns, positions, local, WIDTH, DEGREE, BLOCK)
                local += 1
        else:
            selected = offset + row_lane
            start = tl.load(starts + selected, selected < N_ROWS, base).to(tl.int32)
            stop = tl.load(stops + selected, selected < N_ROWS, base).to(tl.int32)
            source_prefix = tl.load(prefix + base + prefix_lane, base + prefix_lane <= NNZA, 0).to(tl.int32)
            preceding = tl.gather(source_prefix.to(tl.float32, bitcast=True), start - base, 0).to(tl.int32, bitcast=True)
            boundary = tl.gather(source_prefix.to(tl.float32, bitcast=True), stop - base, 0).to(tl.int32, bitcast=True)
            target = preceding[:, None] + tl.minimum(lane[None, :], tl.maximum((boundary - preceding - 1)[:, None], 0))
            target = tl.reshape(target, (TILE * BLOCK,))
            lower = tl.reshape(tl.broadcast_to((start - base)[:, None], (TILE, BLOCK)), (TILE * BLOCK,))
            upper = tl.reshape(tl.broadcast_to((stop - base)[:, None], (TILE, BLOCK)), (TILE * BLOCK,))
            for step in tl.static_range(DEGREE.bit_length()):
                middle = (lower + upper) // 2
                limit = tl.gather(source_prefix.to(tl.float32, bitcast=True), tl.minimum(middle + 1, SOURCE * 2 - 1), 0).to(tl.int32, bitcast=True)
                advance = (middle < upper) & (limit <= target)
                lower = tl.where(advance, middle + 1, lower)
                upper = tl.where(advance, upper, middle)
            first = tl.gather(source_prefix.to(tl.float32, bitcast=True), tl.minimum(lower, SOURCE * 2 - 1), 0).to(tl.int32, bitcast=True)
            position = tl.minimum(tl.maximum(lower * WIDTH + target - first, 0), SOURCE * WIDTH - 1)
            source = tl.load(padded + base * WIDTH + source_lane, base * WIDTH + source_lane < NNZA * WIDTH, 0.)
            key = tl.reshape(tl.gather(source, position, 0), (TILE, BLOCK))
            position = tl.reshape(position, (TILE, BLOCK)) - (start - base)[:, None] * WIDTH
            work = boundary - preceding
            valid = lane[None, :] < work[:, None]
            address = selected[:, None] * BLOCK + lane[None, :]
            tl.store(columns + address, tl.where(valid, key, 2147483647.), selected[:, None] < N_ROWS)
            tl.store(positions + address, tl.where(valid, position, 0), selected[:, None] < N_ROWS)
        offset += GRID * TILE


def _launch_project_batched(pointer, prefix, padded, rows, columns, positions, width, degree, block, max_programs):
    tile = min(8, max(1, 4096 // (degree * width * 2)))
    starts = pointer.index_select(0, rows)
    stops = pointer.index_select(0, rows + 1)
    grid = min(8192, max_programs, triton.cdiv(rows.numel(), tile))
    _project_batched[(grid,)](pointer, prefix, padded, rows, columns, positions, starts, stops,
                      rows.numel(), grid, width, degree, block, tile, degree * tile * 2,
                      padded.shape[0], multibuffer=False)


@triton.jit
def _batch_products(pointer, padded, left, rows, positions, output, starts, stops, lengths,
             N_ROWS, NNZA: tl.constexpr, GRID, WIDTH: tl.constexpr, DEGREE: tl.constexpr,
             BLOCK: tl.constexpr, TILE: tl.constexpr, SOURCE: tl.constexpr):
    offset = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, BLOCK)
    left_lane = tl.arange(0, SOURCE)
    right_lane = tl.arange(0, SOURCE * WIDTH)
    while offset < N_ROWS:
        base = tl.load(starts + offset).to(tl.int32)
        end = tl.load(stops + tl.minimum(offset + TILE - 1, N_ROWS - 1)).to(tl.int32)
        if end - base > SOURCE:
            local = offset
            while local < tl.minimum(offset + TILE, N_ROWS):
                local_row = tl.load(rows + local)
                local_start = tl.load(pointer + local_row).to(tl.int32)
                local_stop = tl.load(pointer + local_row + 1).to(tl.int32)
                local_work = tl.load(lengths + local)
                local_position = tl.load(positions + local * BLOCK + lane)
                local_left_lane = tl.arange(0, DEGREE)
                local_right_lane = tl.arange(0, DEGREE * WIDTH)
                local_first = tl.load(left + local_start + local_left_lane,
                                      local_left_lane < local_stop - local_start, 0.)
                local_second = tl.load(padded + local_start * WIDTH + local_right_lane,
                                       local_right_lane < (local_stop - local_start) * WIDTH, 0.)
                local_product = (tl.gather(local_first, local_position // WIDTH, 0)
                                 * tl.gather(local_second, local_position, 0))
                tl.store(output + local * BLOCK + lane, tl.where(lane < local_work, local_product, 0.))
                local += 1
        else:
            selected = offset + row_lane
            start = tl.load(starts + selected, selected < N_ROWS, base).to(tl.int32)
            work = tl.load(lengths + selected, selected < N_ROWS, 0)
            address = selected[:, None] * BLOCK + lane[None, :]
            position = tl.load(positions + address, selected[:, None] < N_ROWS, 0)
            source = tl.reshape((start - base)[:, None] * WIDTH + position, (TILE * BLOCK,))
            source = tl.minimum(tl.maximum(source, 0), SOURCE * WIDTH - 1)
            first = tl.load(left + base + left_lane, base + left_lane < NNZA, 0.)
            second = tl.load(padded + base * WIDTH + right_lane, base * WIDTH + right_lane < NNZA * WIDTH, 0.)
            product = tl.gather(first, source // WIDTH, 0) * tl.gather(second, source, 0)
            product = tl.reshape(product, (TILE, BLOCK))
            tl.store(output + address, tl.where(lane[None, :] < work[:, None], product, 0.), selected[:, None] < N_ROWS)
        offset += GRID * TILE


@triton.jit
def _batch_reduce_rows(keys, ordered, ends, compact, N_ROWS, GRID, BLOCK: tl.constexpr, TILE: tl.constexpr):
    offset = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, BLOCK)
    while offset < N_ROWS:
        row = offset + row_lane
        address = row[:, None] * BLOCK + lane[None, :]
        key = tl.load(keys + address, row[:, None] < N_ROWS, 2147483647.)
        total = tl.load(ordered + address, row[:, None] < N_ROWS, 0.)
        last = tl.load(ends + address, row[:, None] < N_ROWS, 0)
        following_lane = tl.broadcast_to(tl.minimum(lane + 1, BLOCK - 1)[None, :], (TILE, BLOCK))
        following = tl.gather(key, following_lane, 1)
        tails = ((lane[None, :] == BLOCK - 1) | (key != following)) & (key < 2147483647.)
        distance = 1
        running = True
        while (distance < BLOCK) & running:
            index = tl.broadcast_to(tl.maximum(lane - distance, 0)[None, :], (TILE, BLOCK))
            previous_key = tl.gather(key, index, 1)
            active = (lane[None, :] >= distance) & (previous_key == key)
            running = tl.sum(tl.sum((active & tails).to(tl.int32), 1), 0) != 0
            if running:
                total = total + tl.where(active, tl.gather(total, index, 1), 0.)
            distance *= 2
        values = tl.gather(total, last, 1)
        tl.store(compact + address, values, row[:, None] < N_ROWS)
        offset += GRID * TILE


@triton.jit
def _batch_reduce_bounded(keys, ordered, ends, counts, compact, N_ROWS, GRID, BLOCK: tl.constexpr, TILE: tl.constexpr):
    offset = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, BLOCK)
    while offset < N_ROWS:
        row = offset + row_lane
        address = row[:, None] * BLOCK + lane[None, :]
        key = tl.load(keys + address, row[:, None] < N_ROWS, 2147483647.)
        total = tl.load(ordered + address, row[:, None] < N_ROWS, 0.)
        last = tl.load(ends + address, row[:, None] < N_ROWS, 0)
        previous_lane = tl.broadcast_to(tl.maximum(lane - 1, 0)[None, :], (TILE, BLOCK))
        previous_last = tl.gather(last.to(tl.float32, bitcast=True), previous_lane, 1).to(tl.int32, bitcast=True)
        count = tl.load(counts + row, row < N_ROWS, 0).to(tl.int32)
        length = last - tl.where(lane[None, :] == 0, 0, previous_last + 1) + 1
        maximum = tl.max(tl.max(tl.where(lane[None, :] < count[:, None], length, 0), 1), 0)
        distance = 1
        while distance < maximum:
            index = tl.broadcast_to(tl.maximum(lane - distance, 0)[None, :], (TILE, BLOCK))
            previous_key = tl.gather(key, index, 1)
            active = (lane[None, :] >= distance) & (previous_key == key)
            total = total + tl.where(active, tl.gather(total, index, 1), 0.)
            distance *= 2
        values = tl.gather(total, last, 1)
        tl.store(compact + address, values, row[:, None] < N_ROWS)
        offset += GRID * TILE


@triton.jit
def _batch_finish(compact, rows, pointer, output, N_ROWS, GRID, BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    while offset < N_ROWS:
        row = tl.load(rows + offset)
        start = tl.load(pointer + row).to(tl.int32)
        stop = tl.load(pointer + row + 1).to(tl.int32)
        values = tl.load(compact + offset * BLOCK + lane)
        tl.store(output + start + lane, values, lane < stop - start)
        offset += GRID


@triton.jit
def _complex_products(pointer, prefix, padded, left, rows, positions, output,
                    N_ROWS, GRID, WIDTH: tl.constexpr, DEGREE: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    left_lane = tl.arange(0, DEGREE * 2)
    right_lane = tl.arange(0, DEGREE * WIDTH * 2)
    while offset < N_ROWS:
        row = tl.load(rows + offset)
        start = tl.load(pointer + row).to(tl.int32)
        stop = tl.load(pointer + row + 1).to(tl.int32)
        work = (tl.load(prefix + stop) - tl.load(prefix + start)).to(tl.int32)
        position = tl.load(positions + offset * BLOCK + lane)
        source = position // WIDTH
        first = tl.load(left + start * 2 + left_lane, left_lane < (stop - start) * 2, 0.)
        second = tl.load(padded + start * WIDTH * 2 + right_lane,
                          right_lane < (stop - start) * WIDTH * 2, 0.)
        real_left = tl.gather(first, source * 2, 0)
        imag_left = tl.gather(first, source * 2 + 1, 0)
        address = source * WIDTH * 2 + position % WIDTH
        real_right = tl.gather(second, address, 0)
        imag_right = tl.gather(second, address + WIDTH, 0)
        real = real_left * real_right - imag_left * imag_right
        imag = real_left * imag_right + imag_left * real_right
        tl.store(output + offset * BLOCK + lane, tl.where(lane < work, real, 0.))
        tl.store(output + N_ROWS * BLOCK + offset * BLOCK + lane, tl.where(lane < work, imag, 0.))
        offset += GRID


@triton.jit
def _complex_reduce(keys, products, ends, rows, pointer, output, N_OUTPUTS, N_ROWS, GRID,
                  BLOCK: tl.constexpr):
    offset = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    while offset < N_ROWS:
        row = tl.load(rows + offset)
        start = tl.load(pointer + row).to(tl.int32)
        stop = tl.load(pointer + row + 1).to(tl.int32)
        key = tl.load(keys + offset * BLOCK + lane)
        real = tl.load(products + offset * BLOCK + lane)
        imag = tl.load(products + N_ROWS * BLOCK + offset * BLOCK + lane)
        last = tl.load(ends + offset * BLOCK + lane, lane < stop - start, 0)
        previous = tl.gather(last.to(tl.float32, bitcast=True), tl.maximum(lane - 1, 0), 0).to(tl.int32, bitcast=True)
        first = tl.where(lane == 0, 0, previous + 1)
        maximum = tl.max(tl.where(lane < stop - start, last - first + 1, 0), 0)
        distance = 1
        while distance < maximum:
            index = tl.maximum(lane - distance, 0)
            previous_key = tl.gather(key, index, 0)
            active = (lane >= distance) & (previous_key == key)
            real = real + tl.where(active, tl.gather(real, index, 0), 0.)
            imag = imag + tl.where(active, tl.gather(imag, index, 0), 0.)
            distance *= 2
        real = tl.gather(real, last, 0)
        imag = tl.gather(imag, last, 0)
        tl.store(output + start + lane, real, lane < stop - start)
        tl.store(output + N_OUTPUTS + start + lane, imag, lane < stop - start)
        offset += GRID


@triton.jit
def _complex_batch_products(pointer, padded, left, rows, positions, output, starts, stops, lengths,
             N_ROWS, NNZA: tl.constexpr, GRID, WIDTH: tl.constexpr, DEGREE: tl.constexpr,
             BLOCK: tl.constexpr, TILE: tl.constexpr, SOURCE: tl.constexpr):
    offset = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, BLOCK)
    left_lane = tl.arange(0, SOURCE * 2)
    right_lane = tl.arange(0, SOURCE * WIDTH * 2)
    while offset < N_ROWS:
        base = tl.load(starts + offset).to(tl.int32)
        end = tl.load(stops + tl.minimum(offset + TILE - 1, N_ROWS - 1)).to(tl.int32)
        if end - base > SOURCE:
            local = offset
            while local < tl.minimum(offset + TILE, N_ROWS):
                local_row = tl.load(rows + local)
                local_start = tl.load(pointer + local_row).to(tl.int32)
                local_stop = tl.load(pointer + local_row + 1).to(tl.int32)
                local_work = tl.load(lengths + local).to(tl.int32)
                position = tl.load(positions + local * BLOCK + lane)
                source = position // WIDTH
                local_left_lane = tl.arange(0, DEGREE * 2)
                local_right_lane = tl.arange(0, DEGREE * WIDTH * 2)
                first = tl.load(left + local_start * 2 + local_left_lane,
                                local_left_lane < (local_stop - local_start) * 2, 0.)
                second = tl.load(padded + local_start * WIDTH * 2 + local_right_lane,
                                 local_right_lane < (local_stop - local_start) * WIDTH * 2, 0.)
                real_left = tl.gather(first, source * 2, 0)
                imag_left = tl.gather(first, source * 2 + 1, 0)
                address = source * WIDTH * 2 + position % WIDTH
                real_right = tl.gather(second, address, 0)
                imag_right = tl.gather(second, address + WIDTH, 0)
                real = real_left * real_right - imag_left * imag_right
                imag = real_left * imag_right + imag_left * real_right
                tl.store(output + local * BLOCK + lane, tl.where(lane < local_work, real, 0.))
                tl.store(output + N_ROWS * BLOCK + local * BLOCK + lane, tl.where(lane < local_work, imag, 0.))
                local += 1
        else:
            selected = offset + row_lane
            start = tl.load(starts + selected, selected < N_ROWS, base).to(tl.int32)
            work = tl.load(lengths + selected, selected < N_ROWS, 0).to(tl.int32)
            address = selected[:, None] * BLOCK + lane[None, :]
            position = tl.load(positions + address, selected[:, None] < N_ROWS, 0)
            source = tl.reshape((start - base)[:, None] * WIDTH + position, (TILE * BLOCK,))
            source = tl.minimum(tl.maximum(source, 0), SOURCE * WIDTH - 1)
            first = tl.load(left + base * 2 + left_lane, base * 2 + left_lane < NNZA * 2, 0.)
            second = tl.load(padded + base * WIDTH * 2 + right_lane,
                             base * WIDTH * 2 + right_lane < NNZA * WIDTH * 2, 0.)
            real_left = tl.gather(first, source // WIDTH * 2, 0)
            imag_left = tl.gather(first, source // WIDTH * 2 + 1, 0)
            right_position = source // WIDTH * WIDTH * 2 + source % WIDTH
            real_right = tl.gather(second, right_position, 0)
            imag_right = tl.gather(second, right_position + WIDTH, 0)
            real = tl.reshape(real_left * real_right - imag_left * imag_right, (TILE, BLOCK))
            imag = tl.reshape(real_left * imag_right + imag_left * real_right, (TILE, BLOCK))
            tl.store(output + address, tl.where(lane[None, :] < work[:, None], real, 0.), selected[:, None] < N_ROWS)
            tl.store(output + N_ROWS * BLOCK + address, tl.where(lane[None, :] < work[:, None], imag, 0.),
                     selected[:, None] < N_ROWS)
        offset += GRID * TILE


@triton.jit
def _schedule_maximum(first, second):
    return tl.maximum(first, second)


@triton.jit
def _schedule_build(keys, spans, bounds, N_ROWS, GRID, BLOCK: tl.constexpr, TILE: tl.constexpr):
    tile = tl.program_id(0)
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, BLOCK)
    while tile * TILE < N_ROWS:
        row = tile * TILE + row_lane
        address = row[:, None] * BLOCK + lane[None, :]
        key = tl.load(keys + address, row[:, None] < N_ROWS, 2147483647.)
        previous_lane = tl.broadcast_to(tl.maximum(lane - 1, 0)[None, :], (TILE, BLOCK))
        previous = tl.gather(key, previous_lane, 1)
        first = tl.where((lane[None, :] == 0) | (previous != key), lane[None, :], 0)
        beginning = tl.associative_scan(first, 1, _schedule_maximum)
        span = tl.where(key < 2147483647., lane[None, :] - beginning + 1, 0)
        tl.store(spans + address, span, row[:, None] < N_ROWS)
        maximum = tl.max(tl.max(span, 1), 0)
        tl.store(bounds + tile, maximum)
        tile += GRID


@triton.jit
def _schedule_reduce(spans, bounds, products, ends, compact, N_ROWS, GRID, BLOCK: tl.constexpr, TILE: tl.constexpr):
    tile = tl.program_id(0)
    part = tl.program_id(1)
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, BLOCK)
    while tile * TILE < N_ROWS:
        row = tile * TILE + row_lane
        address = row[:, None] * BLOCK + lane[None, :]
        span = tl.load(spans + address, row[:, None] < N_ROWS, 0).to(tl.int32)
        total = tl.load(products + part * N_ROWS * BLOCK + address, row[:, None] < N_ROWS, 0.)
        last = tl.load(ends + address, row[:, None] < N_ROWS, 0)
        maximum = tl.load(bounds + tile)
        distance = 1
        while distance < maximum:
            index = tl.broadcast_to(tl.maximum(lane - distance, 0)[None, :], (TILE, BLOCK))
            total = total + tl.where(span > distance, tl.gather(total, index, 1), 0.)
            distance *= 2
        values = tl.gather(total, last, 1)
        tl.store(compact + part * N_ROWS * BLOCK + address, values, row[:, None] < N_ROWS)
        tile += GRID


@triton.jit
def _schedule_layout(rows, pointer, output, N_ROWS, BLOCK: tl.constexpr):
    selected = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(rows + selected, selected < N_ROWS, 0)
    start = tl.load(pointer + row).to(tl.int32)
    stop = tl.load(pointer + row + 1).to(tl.int32)
    tl.store(output + selected, start, selected < N_ROWS)
    tl.store(output + N_ROWS + selected, stop - start, selected < N_ROWS)


@triton.jit
def _schedule_finish(compact, _schedule_layout, output, N_OUTPUTS, N_ROWS, GRID, BLOCK: tl.constexpr, PARTS: tl.constexpr):
    offset = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    while offset < N_ROWS:
        start = tl.load(_schedule_layout + offset)
        count = tl.load(_schedule_layout + N_ROWS + offset)
        real = tl.load(compact + offset * BLOCK + lane)
        tl.store(output + start + lane, real, lane < count)
        if PARTS == 2:
            imag = tl.load(compact + N_ROWS * BLOCK + offset * BLOCK + lane)
            tl.store(output + N_OUTPUTS + start + lane, imag, lane < count)
        offset += GRID


def _prepare_schedule(keys, rows, pointer, max_programs):
    count, block = keys.shape
    tile = min(32, max(1, 1024 // block))
    grid = min(8192, max_programs, triton.cdiv(count, tile))
    spans = torch.empty_like(keys, dtype=torch.uint8 if block <= 128 else torch.int16)
    bounds = torch.empty(triton.cdiv(count, tile), dtype=torch.int32, device=keys.device)
    destinations = torch.empty((2, count), dtype=torch.int32, device=keys.device)
    _schedule_build[(grid,)](keys, spans, bounds, count, grid, block, tile, multibuffer=False)
    _schedule_layout[(triton.cdiv(count, 1024),)](rows, pointer, destinations, count, 1024, multibuffer=False)
    return spans, bounds, destinations, tile


@triton.jit
def _initialize_output_map(mapping, N_OUTPUTS, BASE, PARTS: tl.constexpr, BLOCK: tl.constexpr):
    position = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = BASE + (position >> (PARTS - 1)) + (position & (PARTS - 1)) * N_OUTPUTS
    tl.store(mapping + position, source, position < N_OUTPUTS * PARTS)


@triton.jit
def _map_output(rows, pointer, mapping, N_ROWS, GRID, BASE, BLOCK: tl.constexpr, PARTS: tl.constexpr):
    selected = tl.program_id(0)
    lane = tl.arange(0, BLOCK)
    while selected < N_ROWS:
        row = tl.load(rows + selected)
        start = tl.load(pointer + row).to(tl.int32)
        count = tl.load(pointer + row + 1).to(tl.int32) - start
        source = BASE + selected * BLOCK + lane
        tl.store(mapping + (start + lane) * PARTS, source, lane < count)
        if PARTS == 2:
            tl.store(mapping + (start + lane) * 2 + 1, source + N_ROWS * BLOCK, lane < count)
        selected += GRID


def _prepare_output_map(groups, schedules, pointer, n_rows, complex_values, max_programs):
    eligible = [schedule is not None and group[1] <= 16 and group[0].numel() * group[1] >= 262144
                for group, schedule in zip(groups, schedules)]
    selected_rows = sum(group[0].numel() for group, selected in zip(groups, eligible) if selected)
    if n_rows < 65536 or selected_rows * 2 < n_rows:
        return None
    count = int(pointer[-1].item())
    selected_outputs = sum(int((pointer.index_select(0, group[0] + 1)
                                - pointer.index_select(0, group[0])).sum().item())
                           for group, selected in zip(groups, eligible) if selected)
    if selected_outputs * 2 < count:
        return None
    parts = 2 if complex_values else 1
    offsets = []
    base = 0
    for group, selected in zip(groups, eligible):
        offsets.append(base if selected else None)
        if selected:
            base += parts * group[0].numel() * group[1]
    if base + count * parts >= 2**31:
        return None
    mapping = torch.empty(count * parts, dtype=torch.int32, device=pointer.device)
    _initialize_output_map[(triton.cdiv(count * parts, 1024),)](mapping, count, base, parts, 1024, multibuffer=False)
    for group, offset in zip(groups, offsets):
        if offset is not None:
            rows, block = group[:2]
            grid = min(8192, max_programs, rows.numel())
            _map_output[(grid,)](rows, pointer, mapping, rows.numel(), grid, offset, block, parts, multibuffer=False)
    return dict(mapping=mapping, offsets=offsets, fallback_base=base, buffer_size=base + count * parts)


@triton.jit
def _fused_compact(real, imag, spans, ends, output, selected, N_ROWS,
             BLOCK: tl.constexpr, TILE: tl.constexpr, COMPLEX: tl.constexpr):
    lane = tl.arange(0, BLOCK)
    address = selected[:, None] * BLOCK + lane[None, :]
    span = tl.load(spans + address, selected[:, None] < N_ROWS, 0).to(tl.int32)
    last = tl.load(ends + address, selected[:, None] < N_ROWS, 0)
    maximum = tl.max(tl.max(span, 1), 0)
    distance = 1
    while distance < maximum:
        index = tl.broadcast_to(tl.maximum(lane - distance, 0)[None, :], (TILE, BLOCK))
        real = real + tl.where(span > distance, tl.gather(real, index, 1), 0.)
        if COMPLEX:
            imag = imag + tl.where(span > distance, tl.gather(imag, index, 1), 0.)
        distance *= 2
    real = tl.gather(real, last, 1)
    if COMPLEX:
        imag = tl.gather(imag, last, 1)
    tl.store(output + address, real, selected[:, None] < N_ROWS)
    if COMPLEX:
        tl.store(output + N_ROWS * BLOCK + address, imag, selected[:, None] < N_ROWS)


@triton.jit
def _fused_compute(pointer, padded, left, rows, positions, output, starts, stops, lengths,
            spans, ends, N_ROWS, NNZA: tl.constexpr, GRID,
            WIDTH: tl.constexpr, DEGREE: tl.constexpr, BLOCK: tl.constexpr,
            TILE: tl.constexpr, SOURCE: tl.constexpr, COMPLEX: tl.constexpr):
    offset = tl.program_id(0) * TILE
    row_lane = tl.arange(0, TILE)
    lane = tl.arange(0, BLOCK)
    parts: tl.constexpr = 2 if COMPLEX else 1
    left_lane = tl.arange(0, SOURCE * parts)
    right_lane = tl.arange(0, SOURCE * WIDTH * parts)
    while offset < N_ROWS:
        base = tl.load(starts + offset).to(tl.int32)
        end = tl.load(stops + tl.minimum(offset + TILE - 1, N_ROWS - 1)).to(tl.int32)
        if end - base > SOURCE:
            local = offset
            while local < tl.minimum(offset + TILE, N_ROWS):
                local_row = tl.load(rows + local)
                local_start = tl.load(pointer + local_row).to(tl.int32)
                local_stop = tl.load(pointer + local_row + 1).to(tl.int32)
                local_work = tl.load(lengths + local).to(tl.int32)
                position = tl.load(positions + local * BLOCK + lane)
                source = position >> (WIDTH.bit_length() - 1)
                local_left_lane = tl.arange(0, DEGREE * parts)
                local_right_lane = tl.arange(0, DEGREE * WIDTH * parts)
                first = tl.load(left + local_start * parts + local_left_lane,
                                local_left_lane < (local_stop - local_start) * parts, 0.)
                second = tl.load(padded + local_start * WIDTH * parts + local_right_lane,
                                 local_right_lane < (local_stop - local_start) * WIDTH * parts, 0.)
                if COMPLEX:
                    real_left = tl.gather(first, source * 2, 0)
                    imag_left = tl.gather(first, source * 2 + 1, 0)
                    address = source * WIDTH * 2 + (position & (WIDTH - 1))
                    real_right = tl.gather(second, address, 0)
                    imag_right = tl.gather(second, address + WIDTH, 0)
                    real = real_left * real_right - imag_left * imag_right
                    imag = real_left * imag_right + imag_left * real_right
                    imag = tl.reshape(tl.where(lane < local_work, imag, 0.), (1, BLOCK))
                else:
                    real = tl.gather(first, source, 0) * tl.gather(second, position, 0)
                    imag = tl.full((1, BLOCK), 0., tl.float32)
                real = tl.reshape(tl.where(lane < local_work, real, 0.), (1, BLOCK))
                _fused_compact(real, imag, spans, ends, output, tl.full((1,), local, tl.int32),
                         N_ROWS, BLOCK, 1, COMPLEX)
                local += 1
        else:
            selected = offset + row_lane
            start = tl.load(starts + selected, selected < N_ROWS, base).to(tl.int32)
            work = tl.load(lengths + selected, selected < N_ROWS, 0).to(tl.int32)
            address = selected[:, None] * BLOCK + lane[None, :]
            position = tl.load(positions + address, selected[:, None] < N_ROWS, 0)
            source = tl.reshape((start - base)[:, None] * WIDTH + position, (TILE * BLOCK,))
            source = tl.minimum(tl.maximum(source, 0), SOURCE * WIDTH - 1)
            first = tl.load(left + base * parts + left_lane, base * parts + left_lane < NNZA * parts, 0.)
            second = tl.load(padded + base * WIDTH * parts + right_lane,
                             base * WIDTH * parts + right_lane < NNZA * WIDTH * parts, 0.)
            left_source = source >> (WIDTH.bit_length() - 1)
            if COMPLEX:
                real_left = tl.gather(first, left_source * 2, 0)
                imag_left = tl.gather(first, left_source * 2 + 1, 0)
                right_source = left_source * WIDTH * 2 + (source & (WIDTH - 1))
                real_right = tl.gather(second, right_source, 0)
                imag_right = tl.gather(second, right_source + WIDTH, 0)
                real = tl.reshape(real_left * real_right - imag_left * imag_right, (TILE, BLOCK))
                imag = tl.reshape(real_left * imag_right + imag_left * real_right, (TILE, BLOCK))
                imag = tl.where(lane[None, :] < work[:, None], imag, 0.)
            else:
                real = tl.reshape(tl.gather(first, left_source, 0) * tl.gather(second, source, 0), (TILE, BLOCK))
                imag = tl.full((TILE, BLOCK), 0., tl.float32)
            real = tl.where(lane[None, :] < work[:, None], real, 0.)
            _fused_compact(real, imag, spans, ends, output, selected, N_ROWS, BLOCK, TILE, COMPLEX)
        offset += GRID * TILE


def _padded_gather(prepared, width, values, max_programs=65535):
    parts = 2 if values and prepared.a_data.dtype.is_complex else 1
    source = torch.view_as_real(prepared.b_data) if parts == 2 else prepared.b_data if values else prepared.b_indices
    target = torch.empty((prepared.b_shape[0], width * parts), dtype=torch.float32, device=source.device)
    if prepared.b_shape[0] < 256 or width in (2, 4):
        grid = min(8192, max_programs, prepared.b_shape[0])
        if grid:
            _pad[(grid,)](prepared.b_indptr, source, target, prepared.b_shape[0], grid, width, parts,
                          multibuffer=False)
    else:
        tile = min(256, max(1, 2048 // (width * parts)))
        grid = min(8192, max_programs, triton.cdiv(prepared.b_shape[0], tile))
        if width == 1:
            _pad_single_rows[(grid,)](prepared.b_indptr, source, target, prepared.b_shape[0],
                                     prepared.b_data.numel(), grid, parts, tile, multibuffer=False)
        else:
            _pad_rows[(grid,)](prepared.b_indptr, source, target, prepared.b_shape[0],
                              prepared.b_data.numel(), grid, width, parts, tile, multibuffer=False)
    if width == 16 and prepared.a_indices.numel() >= 262144:
        selected = torch.empty((prepared.a_indices.numel(), width * parts), dtype=target.dtype, device=target.device)
        tile = 8192 // (width * parts)
        grid = min(8192, max_programs, triton.cdiv(prepared.a_indices.numel(), tile))
        _select_rows[(grid,)](target, prepared.a_indices, selected, prepared.a_indices.numel(), grid,
                             width * parts, tile, multibuffer=False)
        return selected
    return torch.index_select(target, 0, prepared.a_indices)


@triton.jit
def _store_counts(rows, values, output, N_ROWS, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = tl.load(rows + offset, offset < N_ROWS, 0)
    value = tl.load(values + offset, offset < N_ROWS, 0)
    tl.store(output + row, value, offset < N_ROWS)


def _native_unique(columns, lengths, source_positions, order):
    block = columns.shape[1]
    if 512 <= block <= 2048:
        from ._spgemm_ascend import MAX_PROGRAMS
        from ._spgemm_ascend_sort import compact_sorted

        return compact_sorted(columns, lengths, source_positions, order, MAX_PROGRAMS)
    lane = torch.arange(block, dtype=torch.float32, device=columns.device)[None, :]
    following = torch.cat((columns[:, 1:], torch.full((columns.shape[0], 1), -1., device=columns.device)), dim=1)
    tails = (columns != following) & (lane < lengths[:, None])
    counts = tails.sum(dim=1)
    markers = torch.where(tails, lane, block - 1)
    ends = torch.sort(markers, dim=-1)[0]
    compact = torch.gather(columns, 1, ends.long())
    positions = torch.gather(source_positions, 1, order)
    return positions, compact, ends.int(), counts


def prepare(prepared):
    from ._spgemm_ascend import MAX_PROGRAMS

    plan = prepared.ascend_plan
    degrees = prepared.a_indptr[1:] - prepared.a_indptr[:-1]
    right_degrees = prepared.b_indptr[1:] - prepared.b_indptr[:-1]
    width = triton.next_power_of_2(max(1, int(right_degrees.max().item()) if right_degrees.numel() else 0))
    if (width > 256 or prepared.b_shape[0] * width > 256 * 1024 * 1024
            or prepared.a_data.numel() * width > 1024 * 1024 * 1024):
        return None
    groups = []
    for rows, block, unused in plan['groups']:
        degree = triton.next_power_of_2(max(1, int(degrees.index_select(0, rows).max().item())))
        if degree * width > 65536:
            return None
        groups.append((rows, block, degree))
    padded = _padded_gather(prepared, width, False, MAX_PROGRAMS) if groups else None
    counts = torch.zeros(prepared.n_rows, dtype=torch.int32, device=prepared.a_data.device)
    symbolic = []
    numeric_windows = []
    reduction_counts = []
    scratch = []
    for rows, block, degree in groups:
        grid = min(8192, MAX_PROGRAMS, rows.numel())
        expanded = torch.empty((rows.numel(), block), dtype=torch.float32, device=counts.device)
        source_positions = torch.empty_like(expanded, dtype=torch.int32)
        if width <= 16 and degree * width <= 1024 and expanded.numel() >= 262144:
            _launch_project_batched(prepared.a_indptr, plan['prefix'], padded, rows, expanded, source_positions,
                    width, degree, block, MAX_PROGRAMS)
        else:
            project = _project_wide if degree * width > 8192 else _project
            project[(grid,)](prepared.a_indptr, plan['prefix'], padded, rows, expanded, source_positions, rows.numel(), grid,
                             width, degree, block, multibuffer=False)
        columns, order = torch.sort(expanded, dim=-1)
        if columns.numel() >= 262144:
            positions, compact, ends, group_counts = _native_unique(
                columns, plan['work'].index_select(0, rows), source_positions, order,
            )
            _store_counts[(triton.cdiv(rows.numel(), 1024),)](
                rows, group_counts, counts, rows.numel(), 1024, multibuffer=False,
            )
        else:
            positions = torch.empty_like(source_positions)
            compact = torch.empty_like(columns)
            ends = torch.empty_like(columns, dtype=torch.int32)
            _unique[(grid,)](columns, rows, plan['work'], compact, ends, counts, source_positions,
                             order, positions, rows.numel(), grid, block,
                             multibuffer=False)
        symbolic.append((rows, block, degree, columns, positions, compact, ends))
        reduction_counts.append(group_counts if (not prepared.a_data.dtype.is_complex and block <= 128
                                                 and rows.numel() >= 32768 and columns.numel() >= 262144) else None)
        if (width <= 16 and degree * width <= 1024
                and columns.numel() >= 262144):
            numeric_windows.append((prepared.a_indptr.index_select(0, rows),
                                    prepared.a_indptr.index_select(0, rows + 1),
                                    plan['work'].index_select(0, rows)))
        else:
            numeric_windows.append(None)
        scratch.append((expanded, source_positions, order))
    pointer = prefix_sum(counts, scratch)
    nnz = int(pointer[-1].item())
    output_columns = torch.empty(nnz, dtype=torch.int32, device=counts.device)
    reduction_schedules = []
    for rows, block, degree, columns, positions, compact, ends in symbolic:
        grid = min(8192, MAX_PROGRAMS, rows.numel())
        _finalize[(grid,)](compact, rows, pointer, output_columns, rows.numel(), grid, block,
                           multibuffer=False)
        if rows.numel() >= 16384 and block <= (128 if prepared.a_data.dtype.is_complex else 1024):
            reduction_schedules.append(_prepare_schedule(columns, rows, pointer, MAX_PROGRAMS))
        else:
            reduction_schedules.append(None)
    output_map = _prepare_output_map(symbolic, reduction_schedules, pointer, prepared.n_rows,
                                    prepared.a_data.dtype.is_complex, MAX_PROGRAMS)
    torch.npu.synchronize()
    return dict(width=width, prefix=plan['prefix'], max_programs=MAX_PROGRAMS,
                groups=[(rows, block, degree, columns, positions, ends)
                        for rows, block, degree, columns, positions, compact, ends in symbolic],
                columns=output_columns, pointer=pointer.to(torch.int64), numeric_windows=numeric_windows,
                reduction_counts=reduction_counts, reduction_schedules=reduction_schedules, output_map=output_map)


def compute(prepared, plan):

    padded = _padded_gather(prepared, plan['width'], True, plan['max_programs']) if plan['groups'] else None
    complex_values = prepared.a_data.dtype.is_complex
    left = torch.view_as_real(prepared.a_data) if complex_values else prepared.a_data
    output_map = plan['output_map']
    if output_map is None:
        buffer = None
        output = torch.empty((2, plan['columns'].numel()) if complex_values else (plan['columns'].numel(),),
                             dtype=torch.float32, device=left.device)
    else:
        buffer = torch.empty(output_map['buffer_size'], dtype=torch.float32, device=left.device)
        output = buffer[output_map['fallback_base']:]
        if complex_values:
            output = output.view(2, -1)
    scratch = []
    for group_index, (rows, block, degree, columns, positions, ends) in enumerate(plan['groups']):
        parts = 2 if complex_values else 1
        grid = min(8192, max(1, plan['max_programs'] // parts), rows.numel())
        schedule = plan['reduction_schedules'][group_index]
        if complex_values and schedule is not None and plan['numeric_windows'][group_index] is not None:
            starts, stops, lengths = plan['numeric_windows'][group_index]
            spans, unused_bounds, destinations, unused_tile = schedule
            tile = min(32, max(1, 2048 // block), max(1, 4096 // (degree * plan['width'])))
            source = degree * tile
            product_grid = min(8192, plan['max_programs'], triton.cdiv(rows.numel(), tile))
            compact_offset = output_map['offsets'][group_index] if output_map is not None else None
            compact = (torch.empty((parts, rows.numel(), block), dtype=torch.float32, device=left.device)
                       if compact_offset is None else
                       buffer[compact_offset:compact_offset + parts * rows.numel() * block].view(parts, rows.numel(), block))
            _fused_compute[(product_grid,)](
                prepared.a_indptr, padded, left, rows, positions, compact,
                starts, stops, lengths, spans, ends,
                rows.numel(), prepared.a_data.numel(), product_grid,
                plan['width'], degree, block, tile, source, complex_values,
                enable_fp_fusion=False, multibuffer=False,
            )
            if compact_offset is None:
                _schedule_finish[(grid,)](
                    compact, destinations, output, plan['columns'].numel(), rows.numel(), grid, block, parts,
                    enable_fp_fusion=False, multibuffer=False,
                )
            scratch.append(compact)
            continue
        if columns.numel() >= 262144 or degree * plan['width'] > 8192:
            ordered = torch.empty((parts, rows.numel(), block), dtype=torch.float32, device=left.device)
            if degree * plan['width'] > 8192:
                chunk = 4096 if complex_values else 8192
                chunks = degree * plan['width'] // chunk
                product_grid = min(grid, max(1, plan['max_programs'] // (parts * chunks)))
                partial = torch.empty((chunks, parts, rows.numel(), block), dtype=torch.float32, device=left.device)
                _products_wide[(product_grid, parts, chunks)](
                    prepared.a_indptr, plan['prefix'], padded, left, rows, positions, partial,
                    rows.numel(), product_grid, plan['width'], degree, block, complex_values, chunk,
                    enable_fp_fusion=False, multibuffer=False,
                )
                ordered = partial.sum(dim=0)
                scratch.append(partial)
            elif plan['numeric_windows'][group_index] is not None:
                starts, stops, lengths = plan['numeric_windows'][group_index]
                tile_limit = 4096 if complex_values else 2048
                tile = min(16, max(1, tile_limit // (degree * plan['width'])))
                source = degree * tile * (1 if complex_values else 2)
                product_grid = min(8192, plan['max_programs'], triton.cdiv(rows.numel(), tile))
                product = _complex_batch_products if complex_values else _batch_products
                product[(product_grid,)](
                    prepared.a_indptr, padded, left, rows, positions, ordered, starts, stops, lengths,
                    rows.numel(), prepared.a_data.numel(), product_grid, plan['width'], degree, block,
                    tile, source, enable_fp_fusion=False, multibuffer=False,
                )
            elif complex_values and degree * plan['width'] <= 256:
                _complex_products[(grid,)](
                    prepared.a_indptr, plan['prefix'], padded, left, rows, positions, ordered,
                    rows.numel(), grid, plan['width'], degree, block,
                    enable_fp_fusion=False, multibuffer=False,
                )
            else:
                _products[(grid, parts)](prepared.a_indptr, plan['prefix'], padded, left, rows, columns, positions, ends,
                                     plan['pointer'], ordered, rows.numel() * block, rows.numel(), grid,
                                     plan['width'], degree, block, complex_values,
                                     enable_fp_fusion=False, multibuffer=False)
            if schedule is not None:
                spans, bounds, destinations, tile = schedule
                reduce_grid = min(8192, plan['max_programs'], triton.cdiv(rows.numel(), tile))
                compact_offset = output_map['offsets'][group_index] if output_map is not None else None
                compact = (torch.empty_like(ordered) if compact_offset is None else
                           buffer[compact_offset:compact_offset + ordered.numel()].view_as(ordered))
                _schedule_reduce[(reduce_grid, parts)](
                    spans, bounds, ordered, ends, compact, rows.numel(), reduce_grid, block, tile,
                    enable_fp_fusion=False, multibuffer=False,
                )
                if compact_offset is None:
                    _schedule_finish[(grid,)](
                        compact, destinations, output, plan['columns'].numel(), rows.numel(), grid, block, parts,
                        enable_fp_fusion=False, multibuffer=False,
                    )
                scratch.append(compact)
            elif not complex_values and block <= 128 and rows.numel() >= 8192:
                tile = min(32, max(1, 1024 // block))
                reduce_grid = min(8192, plan['max_programs'], triton.cdiv(rows.numel(), tile))
                compact = torch.empty_like(ordered)
                counts = plan['reduction_counts'][group_index]
                if counts is None:
                    _batch_reduce_rows[(reduce_grid,)](columns, ordered, ends, compact, rows.numel(), reduce_grid,
                                                 block, tile, enable_fp_fusion=False, multibuffer=False)
                else:
                    _batch_reduce_bounded[(reduce_grid,)](columns, ordered, ends, counts, compact, rows.numel(), reduce_grid,
                                                    block, tile, enable_fp_fusion=False, multibuffer=False)
                _batch_finish[(grid,)](compact, rows, plan['pointer'], output, rows.numel(), grid, block,
                                 enable_fp_fusion=False, multibuffer=False)
                scratch.append(compact)
            elif complex_values and block <= 1024:
                _complex_reduce[(grid,)](columns, ordered, ends, rows, plan['pointer'], output,
                                     plan['columns'].numel(), rows.numel(), grid, block,
                                     enable_fp_fusion=False, multibuffer=False)
            else:
                reduce = _scan_reduce if block > 2048 else _bounded_reduce
                reduce[(grid, parts)](columns, ordered, ends, rows, plan['pointer'], output,
                                     plan['columns'].numel(), rows.numel(), grid, block,
                                     enable_fp_fusion=False, multibuffer=False)
            scratch.append(ordered)
        else:
            _numeric[(grid, parts)](prepared.a_indptr, plan['prefix'], padded, left, rows, columns, positions, ends,
                                      plan['pointer'], output, plan['columns'].numel(), rows.numel(), grid,
                                      plan['width'], degree, block, complex_values,
                                      enable_fp_fusion=False, multibuffer=False)
    if output_map is not None:
        output = torch.index_select(buffer, 0, output_map['mapping'])
        if complex_values:
            output = torch.view_as_complex(output.view(-1, 2))
    elif complex_values:
        output = torch.view_as_complex(torch.stack((output[0], output[1]), dim=1))
    columns, pointer = plan['columns'].clone(), plan['pointer'].clone()
    torch.npu.synchronize()
    return output, columns, pointer
