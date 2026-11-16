"""Bounded row-local symbolic plans and live numerical Ascend SpGEMM."""

import torch
import triton
import triton.language as tl


MAX_ROW_PRODUCTS = 4096
MAX_PRODUCTS = 128 * 1024 * 1024
MAX_PROGRAMS = 65535
NATIVE_NUMERIC_THRESHOLD = 4096
MIN_PADDED_PRODUCTS = 65536


@triton.jit
def _expand_rows(
    a_columns, a_pointer, b_columns, b_pointer,
    prefix, row_pointer, rows, sorted_columns, product_a, product_b,
    N_ROWS, SEARCH_STEPS: tl.constexpr,
    TILE_ROWS: tl.constexpr, BLOCK: tl.constexpr, LANES: tl.constexpr,
    SEARCH_BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    selected_rows = tl.minimum(row_offsets, N_ROWS - 1)
    row = tl.load(rows + selected_rows)
    start = tl.load(a_pointer + row).to(tl.int32)
    stop = tl.load(a_pointer + row + 1).to(tl.int32)
    base = tl.load(row_pointer + row).to(tl.int32)
    end = tl.load(row_pointer + row + 1).to(tl.int32)
    if SEARCH_BLOCK <= 256:
        source_lane = tl.arange(0, SEARCH_BLOCK)
        source_position = start[:, None] + source_lane[None, :]
        source_valid = source_position < stop[:, None]
        local_prefix = tl.load(prefix + source_position, source_valid, 0).to(tl.int32)
        local_boundary = tl.load(prefix + source_position + 1, source_valid, 0).to(tl.int32)
        local_columns = tl.load(a_columns + source_position, source_valid, 0)
        local_starts = tl.load(b_pointer + local_columns).to(tl.int32)
    lane = tl.program_id(1) * LANES + tl.arange(0, LANES)
    valid = (row_offsets[:, None] < N_ROWS) & (lane[None, :] < (end - base)[:, None])
    product = base[:, None] + tl.minimum(lane[None, :], (end - base - 1)[:, None])
    lower = tl.full((TILE_ROWS, LANES), 0, tl.int32)
    upper = tl.broadcast_to((stop - start)[:, None], (TILE_ROWS, LANES))
    for step in tl.static_range(SEARCH_STEPS):
        middle = (lower + upper) // 2
        probe = tl.minimum(middle, (stop - start - 1)[:, None])
        if SEARCH_BLOCK <= 256:
            boundary = tl.gather(local_boundary.to(tl.float32, bitcast=True), probe, 1).to(tl.int32, bitcast=True)
        else:
            boundary = tl.load(prefix + start[:, None] + probe + 1).to(tl.int32)
        advance = (middle < (stop - start)[:, None]) & (boundary <= product)
        lower = tl.where(advance, middle + 1, lower)
        upper = tl.where(advance, upper, middle)
    if SEARCH_BLOCK <= 256:
        second = tl.gather(local_starts.to(tl.float32, bitcast=True), lower, 1).to(tl.int32, bitcast=True)
        preceding = tl.gather(local_prefix.to(tl.float32, bitcast=True), lower, 1).to(tl.int32, bitcast=True)
    else:
        inner = tl.load(a_columns + start[:, None] + lower)
        second = tl.load(b_pointer + inner).to(tl.int32)
        preceding = tl.load(prefix + start[:, None] + lower).to(tl.int32)
    lower = lower + start[:, None]
    second = second + product - preceding
    column = tl.load(b_columns + second)
    column = tl.where(valid, column, 2147483647)
    tl.store(sorted_columns + selected_rows[:, None] * BLOCK + lane[None, :], column.to(tl.float32),
             (row_offsets[:, None] < N_ROWS) & (lane[None, :] < BLOCK))
    tl.store(product_a + base[:, None] + lane[None, :], lower, valid)
    tl.store(product_b + base[:, None] + lane[None, :], second, valid)


@triton.jit
def _unique_rows(
    row_pointer, rows, sorted_columns, sorted_positions, unique_columns, counts, unique_starts,
    source_a, source_b, product_a, product_b, product_destination,
    N_ROWS, TILE_ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    row_offsets = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    selected_rows = tl.minimum(row_offsets, N_ROWS - 1)
    row = tl.load(rows + selected_rows)
    base = tl.load(row_pointer + row).to(tl.int32)
    end = tl.load(row_pointer + row + 1).to(tl.int32)
    lane = tl.arange(0, BLOCK)
    valid = (row_offsets[:, None] < N_ROWS) & (lane[None, :] < (end - base)[:, None])
    ordered = tl.load(sorted_columns + selected_rows[:, None] * BLOCK + lane[None, :]).to(tl.int32)
    previous = tl.gather(ordered.to(tl.float32, bitcast=True),
                         tl.broadcast_to(tl.maximum(lane - 1, 0)[None, :], (TILE_ROWS, BLOCK)), 1).to(tl.int32, bitcast=True)
    heads = valid & ((lane[None, :] == 0) | (ordered != previous))
    ranks = tl.cumsum(heads.to(tl.int32), 1) - 1
    tl.store(unique_columns + base[:, None] + tl.maximum(ranks, 0), ordered, heads)
    tl.store(counts + row, tl.sum(heads.to(tl.int32), 1), row_offsets < N_ROWS)
    position = tl.load(sorted_positions + selected_rows[:, None] * BLOCK + lane[None, :]).to(tl.int32)
    local_first = tl.load(source_a + base[:, None] + lane[None, :], valid, 0)
    local_second = tl.load(source_b + base[:, None] + lane[None, :], valid, 0)
    first = tl.gather(local_first.to(tl.float32, bitcast=True), position, 1).to(tl.int32, bitcast=True)
    second = tl.gather(local_second.to(tl.float32, bitcast=True), position, 1).to(tl.int32, bitcast=True)
    tl.store(product_a + base[:, None] + lane[None, :], first, valid)
    tl.store(product_b + base[:, None] + lane[None, :], second, valid)
    tl.store(product_destination + base[:, None] + lane[None, :], ranks, valid)
    tl.store(unique_starts + base[:, None] + tl.maximum(ranks, 0), base[:, None] + lane[None, :], heads)


@triton.jit
def _unique_wide_row(
    row_pointer, rows, sorted_columns, sorted_positions, unique_columns, counts, unique_starts,
    source_a, source_b, product_a, product_b, product_destination,
    BLOCK: tl.constexpr,
):
    row_offset = tl.program_id(0)
    row = tl.load(rows + row_offset)
    base = tl.load(row_pointer + row).to(tl.int32)
    end = tl.load(row_pointer + row + 1).to(tl.int32)
    lane = tl.arange(0, BLOCK)
    valid = lane < end - base
    ordered = tl.load(sorted_columns + row_offset * BLOCK + lane, valid, 2147483647).to(tl.int32)
    previous = tl.gather(ordered.to(tl.float32, bitcast=True), tl.maximum(lane - 1, 0), 0).to(tl.int32, bitcast=True)
    heads = valid & ((lane == 0) | (ordered != previous))
    ranks = tl.cumsum(heads.to(tl.int32), 0) - 1
    tl.store(unique_columns + base + ranks, ordered, heads)
    tl.store(counts + row, tl.sum(heads.to(tl.int32), 0))
    position = tl.load(sorted_positions + row_offset * BLOCK + lane, valid, 0).to(tl.int32)
    local_first = tl.load(source_a + base + lane, valid, 0)
    local_second = tl.load(source_b + base + lane, valid, 0)
    first = tl.gather(local_first.to(tl.float32, bitcast=True), position, 0).to(tl.int32, bitcast=True)
    second = tl.gather(local_second.to(tl.float32, bitcast=True), position, 0).to(tl.int32, bitcast=True)
    tl.store(product_a + base + lane, first, valid)
    tl.store(product_b + base + lane, second, valid)
    tl.store(product_destination + base + lane, ranks, valid)
    tl.store(unique_starts + base + ranks, base + lane, heads)


@triton.jit
def _finalize_rows(
    row_pointer, rows, unique_columns, unique_starts, product_pointer, output_pointer, output_columns,
    product_destination, N_ROWS,
    TILE_ROWS: tl.constexpr, BLOCK: tl.constexpr, LANES: tl.constexpr,
):
    row_offsets = tl.program_id(0) * TILE_ROWS + tl.arange(0, TILE_ROWS)
    selected_rows = tl.minimum(row_offsets, N_ROWS - 1)
    row = tl.load(rows + selected_rows)
    base = tl.load(row_pointer + row).to(tl.int32)
    end = tl.load(row_pointer + row + 1).to(tl.int32)
    output_start = tl.load(output_pointer + row).to(tl.int32)
    output_end = tl.load(output_pointer + row + 1).to(tl.int32)
    lane = tl.program_id(1) * LANES + tl.arange(0, LANES)
    output_valid = (row_offsets[:, None] < N_ROWS) & (lane[None, :] < (output_end - output_start)[:, None])
    unique = tl.load(unique_columns + base[:, None] + lane[None, :], output_valid, 0)
    tl.store(output_columns + output_start[:, None] + lane[None, :], unique, output_valid)
    position = base[:, None] + lane[None, :]
    product_start = tl.load(unique_starts + position, output_valid, 0)
    tl.store(product_pointer + output_start[:, None] + lane[None, :], product_start, output_valid)
    product_valid = (row_offsets[:, None] < N_ROWS) & (lane[None, :] < (end - base)[:, None])
    destination = tl.load(product_destination + base[:, None] + lane[None, :], product_valid, 0)
    tl.store(product_destination + base[:, None] + lane[None, :],
             output_start[:, None] + destination, product_valid)


@triton.jit
def _numeric_segments(
    a_values, b_values, product_a, product_b, product_pointer, output_rows, output_values,
    N_OUTPUTS, COMPLEX: tl.constexpr, ROWS: tl.constexpr,
    LANES: tl.constexpr, STEPS: tl.constexpr,
):
    offsets = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    selected = tl.minimum(offsets, N_OUTPUTS - 1)
    destination = tl.load(output_rows + selected)
    start = tl.load(product_pointer + destination)
    end = tl.load(product_pointer + destination + 1)
    lane = tl.program_id(1) * LANES + tl.arange(0, LANES)
    position = start[:, None] + tl.minimum(lane[None, :], (end - start - 1)[:, None])
    valid = lane[None, :] < (end - start)[:, None]
    first = tl.load(product_a + position)
    second = tl.load(product_b + position)
    if COMPLEX:
        first_real = tl.load(a_values + first * 2)
        first_imag = tl.load(a_values + first * 2 + 1)
        second_real = tl.load(b_values + second * 2)
        second_imag = tl.load(b_values + second * 2 + 1)
        real = first_real * second_real - first_imag * second_imag
        imag = first_real * second_imag + first_imag * second_real
        result = tl.sum(tl.where(valid, real, 0.), 1)
        imaginary = tl.sum(tl.where(valid, imag, 0.), 1)
    else:
        first_value = tl.load(a_values + first)
        second_value = tl.load(b_values + second)
        result = tl.sum(tl.where(valid, first_value * second_value, 0.), 1)
    if STEPS > 1:
        destination = offsets * STEPS + tl.program_id(1)
    if COMPLEX:
        tl.store(output_values + destination * 2, result, offsets < N_OUTPUTS)
        tl.store(output_values + destination * 2 + 1, imaginary, offsets < N_OUTPUTS)
    else:
        tl.store(output_values + destination, result, offsets < N_OUTPUTS)


@triton.jit
def _finish_segments(partials, output_rows, output_values,
                     COMPLEX: tl.constexpr, STEPS: tl.constexpr):
    offset = tl.program_id(0)
    destination = tl.load(output_rows + offset)
    position = offset * STEPS + tl.arange(0, STEPS)
    if COMPLEX:
        real = tl.load(partials + position * 2)
        imag = tl.load(partials + position * 2 + 1)
        tl.store(output_values + destination * 2, tl.sum(real, 0))
        tl.store(output_values + destination * 2 + 1, tl.sum(imag, 0))
    else:
        values = tl.load(partials + position)
        tl.store(output_values + destination, tl.sum(values, 0))


def prepare_structure(a_columns, a_pointer, b_pointer):
    device = a_columns.device
    if max(int(a_pointer[-1].item()), int(b_pointer[-1].item())) >= 2**30:
        return None
    right_degrees = (b_pointer[1:] - b_pointer[:-1]).to(torch.int32)
    lengths = right_degrees.index_select(0, a_columns)
    total = int(torch.sum(lengths, dtype=torch.int64).item())
    if total > MAX_PRODUCTS:
        return None
    prefix = torch.zeros(a_columns.numel() + 1, dtype=torch.int32, device=device)
    prefix[1:] = torch.cumsum(lengths, 0, dtype=torch.int32)
    row_pointer = prefix.index_select(0, a_pointer)
    work = row_pointer[1:] - row_pointer[:-1]
    maximum = int(work.max().item()) if work.numel() else 0
    if maximum > MAX_ROW_PRODUCTS:
        return None
    groups = []
    lower = 0
    block = 16
    while lower < maximum:
        rows = torch.nonzero((work > lower) & (work <= block), as_tuple=False).flatten()
        if rows.numel():
            tile_rows = max(2, min(16, 512 // block))
            column_tiles = triton.cdiv(block, min(block, 256))
            chunk_rows = (MAX_PROGRAMS // column_tiles) * tile_rows
            if block >= 2048:
                chunk_rows = min(MAX_PROGRAMS, chunk_rows)
            for subset in rows.split(chunk_rows):
                groups.append((subset, block, tile_rows))
        lower, block = block, block * 2
    degrees = a_pointer[1:] - a_pointer[:-1]
    max_degree = int(degrees.max().item()) if degrees.numel() else 0
    return dict(prefix=prefix, row_pointer=row_pointer, work=work, groups=groups,
                total=total, search_steps=max_degree.bit_length(), search_block=triton.next_power_of_2(max(1, max_degree)))


def prepare_symbolic(prepared):
    plan = prepared.ascend_plan
    if plan['total'] >= MIN_PADDED_PRODUCTS:
        from ._spgemm_ascend_padded import prepare

        padded = prepare(prepared)
        if padded is not None:
            plan.update(padded=padded, output_columns=padded['columns'], output_pointer=padded['pointer'])
            return
    device = prepared.a_data.device
    unique_columns = torch.empty(plan["total"], dtype=torch.int32, device=device)
    counts = torch.zeros(prepared.n_rows, dtype=torch.int64, device=device)
    product_a = torch.empty(plan["total"], dtype=torch.int32, device=device)
    product_b = torch.empty_like(product_a)
    product_destination = torch.empty_like(product_a)
    unique_starts = torch.empty_like(product_a)
    source_a = torch.empty_like(product_a)
    source_b = torch.empty_like(product_a)
    scratch_buffers = []
    for rows, block, tile_rows in plan["groups"]:
        lanes = min(block, 256)
        expanded_columns = torch.empty((rows.numel(), block), dtype=torch.float32, device=device)
        _expand_rows[(triton.cdiv(rows.numel(), tile_rows), triton.cdiv(block, lanes))](
            prepared.a_indices, prepared.a_indptr, prepared.b_indices, prepared.b_indptr,
            plan["prefix"], plan["row_pointer"], rows, expanded_columns, source_a, source_b,
            rows.numel(), plan["search_steps"], tile_rows, block, lanes, plan["search_block"],
        )
        sorted_columns, sorted_positions = torch.sort(expanded_columns, dim=-1, descending=False)
        if block >= 2048:
            _unique_wide_row[(rows.numel(),)](
                plan["row_pointer"], rows, sorted_columns, sorted_positions,
                unique_columns, counts, unique_starts, source_a, source_b, product_a, product_b, product_destination, block,
            )
        else:
            _unique_rows[(triton.cdiv(rows.numel(), tile_rows),)](
                plan["row_pointer"], rows, sorted_columns, sorted_positions,
                unique_columns, counts, unique_starts, source_a, source_b, product_a, product_b, product_destination,
                rows.numel(), tile_rows, block,
            )
        scratch_buffers.append((expanded_columns, sorted_columns, sorted_positions))
        del expanded_columns, sorted_columns, sorted_positions
    output_pointer = torch.zeros(prepared.n_rows + 1, dtype=torch.int64, device=device)
    output_pointer[1:] = torch.cumsum(counts, 0)
    output_nnz = int(output_pointer[-1].item())
    scratch_buffers.clear()
    output_columns = torch.empty(output_nnz, dtype=torch.int32, device=device)
    product_pointer = torch.empty(output_nnz + 1, dtype=torch.int32, device=device)
    product_pointer[-1] = plan["total"]
    for rows, block, tile_rows in plan["groups"]:
        lanes = min(block, 256)
        _finalize_rows[(triton.cdiv(rows.numel(), tile_rows), triton.cdiv(block, lanes))](
            plan["row_pointer"], rows, unique_columns, unique_starts, product_pointer, output_pointer, output_columns,
            product_destination, rows.numel(), tile_rows, block, lanes,
        )
    multiplicity = product_pointer[1:] - product_pointer[:-1]
    numeric_groups = []
    lower, block = 0, 1
    maximum = int(multiplicity.max().item()) if output_nnz and plan["total"] < NATIVE_NUMERIC_THRESHOLD else 0
    while lower < maximum:
        outputs = torch.nonzero((multiplicity > lower) & (multiplicity <= block), as_tuple=False).flatten().to(torch.int32)
        tile_rows = max(2, min(512, 1024 // block))
        for subset in outputs.split(max(1, MAX_PROGRAMS // triton.cdiv(block, 256)) * tile_rows):
            if subset.numel():
                numeric_groups.append((subset, tile_rows, block))
        lower, block = block, block * 2
    torch.npu.current_stream(device).synchronize()
    plan.update(product_a=product_a, product_b=product_b, product_pointer=product_pointer,
                product_destination=product_destination if plan["total"] >= NATIVE_NUMERIC_THRESHOLD else None,
                numeric_groups=numeric_groups,
                output_columns=output_columns, output_pointer=output_pointer)


def _segmented_values(prepared):
    plan = prepared.ascend_plan
    device, dtype = prepared.a_data.device, prepared.a_data.dtype
    output_values = torch.empty(plan["output_columns"].numel(), dtype=dtype, device=device)
    if plan["total"]:
        complex_values = dtype.is_complex
        first = torch.view_as_real(prepared.a_data) if complex_values else prepared.a_data
        second = torch.view_as_real(prepared.b_data) if complex_values else prepared.b_data
        output = torch.view_as_real(output_values) if complex_values else output_values
        for outputs, tile_rows, block in plan["numeric_groups"]:
            steps = triton.cdiv(block, 256)
            partials = torch.empty(outputs.numel() * steps, dtype=dtype, device=device) if steps > 1 else output_values
            target = torch.view_as_real(partials) if complex_values else partials
            _numeric_segments[(triton.cdiv(outputs.numel(), tile_rows), steps)](
                first, second, plan["product_a"], plan["product_b"], plan["product_pointer"], outputs, target,
                outputs.numel(), complex_values, tile_rows, min(block, 256), steps,
                enable_fp_fusion=False,
            )
            if steps > 1:
                _finish_segments[(outputs.numel(),)](
                    target, outputs, output, complex_values, steps,
                )
                torch.npu.current_stream(device).synchronize()
    return output_values


def _native_values(prepared):
    plan = prepared.ascend_plan
    complex_values = prepared.a_data.dtype.is_complex
    first = torch.view_as_real(prepared.a_data) if complex_values else prepared.a_data
    second = torch.view_as_real(prepared.b_data) if complex_values else prepared.b_data
    first = torch.index_select(first, 0, plan["product_a"])
    second = torch.index_select(second, 0, plan["product_b"])
    if complex_values:
        products = torch.stack((first[:, 0] * second[:, 0] - first[:, 1] * second[:, 1],
                                first[:, 0] * second[:, 1] + first[:, 1] * second[:, 0]), dim=1)
        values = torch.zeros((plan["output_columns"].numel(), 2), dtype=torch.float32, device=first.device)
    else:
        products = first * second
        values = torch.zeros(plan["output_columns"].numel(), dtype=first.dtype, device=first.device)
    values.index_add_(0, plan["product_destination"], products)
    return torch.view_as_complex(values) if complex_values else values


def compute(prepared):
    plan = prepared.ascend_plan
    if 'padded' in plan:
        from ._spgemm_ascend_padded import compute as compute_padded

        return compute_padded(prepared, plan['padded'])
    output_values = _native_values(prepared) if plan["product_destination"] is not None else _segmented_values(prepared)
    output_columns = plan["output_columns"].clone()
    output_pointer = plan["output_pointer"].clone()
    torch.npu.current_stream(prepared.a_data.device).synchronize()
    return output_values, output_columns, output_pointer
