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


def _padded_gather(prepared, width, values, max_programs=65535):
    parts = 2 if values and prepared.a_data.dtype.is_complex else 1
    source = torch.view_as_real(prepared.b_data) if parts == 2 else prepared.b_data if values else prepared.b_indices
    target = torch.empty((prepared.b_shape[0], width * parts), dtype=torch.float32, device=source.device)
    grid = min(8192, max_programs, prepared.b_shape[0])
    if grid:
        _pad[(grid,)](prepared.b_indptr, source, target, prepared.b_shape[0], grid, width, parts,
                      multibuffer=False)
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
    scratch = []
    for rows, block, degree in groups:
        grid = min(8192, MAX_PROGRAMS, rows.numel())
        expanded = torch.empty((rows.numel(), block), dtype=torch.float32, device=counts.device)
        source_positions = torch.empty_like(expanded, dtype=torch.int32)
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
        scratch.append((expanded, source_positions, order))
    pointer = torch.zeros(prepared.n_rows + 1, dtype=torch.int32, device=counts.device)
    pointer[1:] = torch.cumsum(counts, 0, dtype=torch.int32)
    nnz = int(pointer[-1].item())
    output_columns = torch.empty(nnz, dtype=torch.int32, device=counts.device)
    for rows, block, degree, columns, positions, compact, ends in symbolic:
        grid = min(8192, MAX_PROGRAMS, rows.numel())
        _finalize[(grid,)](compact, rows, pointer, output_columns, rows.numel(), grid, block,
                           multibuffer=False)
    torch.npu.synchronize()
    return dict(width=width, prefix=plan['prefix'], max_programs=MAX_PROGRAMS,
                groups=[(rows, block, degree, columns, positions, ends)
                        for rows, block, degree, columns, positions, compact, ends in symbolic],
                columns=output_columns, pointer=pointer.to(torch.int64))


def compute(prepared, plan):
    padded = _padded_gather(prepared, plan['width'], True, plan['max_programs']) if plan['groups'] else None
    complex_values = prepared.a_data.dtype.is_complex
    left = torch.view_as_real(prepared.a_data) if complex_values else prepared.a_data
    output = torch.empty((2, plan['columns'].numel()) if complex_values else (plan['columns'].numel(),),
                         dtype=torch.float32, device=left.device)
    scratch = []
    for rows, block, degree, columns, positions, ends in plan['groups']:
        parts = 2 if complex_values else 1
        grid = min(8192, max(1, plan['max_programs'] // parts), rows.numel())
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
            else:
                _products[(grid, parts)](prepared.a_indptr, plan['prefix'], padded, left, rows, columns, positions, ends,
                                     plan['pointer'], ordered, rows.numel() * block, rows.numel(), grid,
                                     plan['width'], degree, block, complex_values,
                                     enable_fp_fusion=False, multibuffer=False)
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
    if complex_values:
        output = torch.view_as_complex(torch.stack((output[0], output[1]), dim=1))
    columns, pointer = plan['columns'].clone(), plan['pointer'].clone()
    torch.npu.synchronize()
    return output, columns, pointer
