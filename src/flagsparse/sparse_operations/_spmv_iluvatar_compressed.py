"""Lossless value/index storage for versioned Iluvatar COO SELL tiles."""

import torch
import triton
import triton.language as tl


def _encode_real(values, plan, n_cols):
    real = values.real.contiguous()
    bits = real.view(torch.int32)
    columns, gather = plan[0], plan[4]
    colbits = max(1, (n_cols - 1).bit_length())
    if bool(torch.all(bits == bits[0])):
        encoding = 4 if int(bits[0]) == 0x3F800000 else 1
        return real[:1].clone(), columns, encoding, colbits
    if colbits <= 24:
        integer = real.to(torch.int8)
        if bool(torch.all(integer.float().view(torch.int32) == bits)):
            payload = columns.long() | (integer[gather].to(torch.uint8).long() << 24)
            return real[:1], payload.int(), 2, 24
    capacity = min(2048, 1 << max(0, 32 - colbits))
    if capacity > 1 and torch.unique(bits[:4096]).numel() <= capacity:
        table, inverse = torch.unique(bits, sorted=True, return_inverse=True)
        if table.numel() <= capacity and table.numel() < bits.numel():
            payload = columns.long() | (inverse[gather] << colbits)
            return table.view(torch.float32), payload.int(), 3, colbits
    return real[gather].contiguous(), columns, 0, colbits


@triton.jit
def _column_bounds(Cols, Ptr, Lengths, Bases, Spans,
                   N_COLS: tl.constexpr, HEIGHT: tl.constexpr, BLOCK: tl.constexpr):
    slice_id = tl.program_id(0)
    start = tl.load(Ptr + slice_id)
    end = tl.load(Ptr + slice_id + 1)
    positions = tl.arange(0, BLOCK)
    lengths = tl.load(Lengths + slice_id * HEIGHT + positions % HEIGHT)
    valid = (start + positions < end) & (positions // HEIGHT < lengths)
    columns = tl.load(Cols + start + positions, valid, 0)
    minimum = tl.min(tl.where(valid, columns, N_COLS), 0)
    maximum = tl.max(tl.where(valid, columns, 0), 0)
    tl.store(Bases + slice_id, tl.where(start < end, minimum, 0))
    tl.store(Spans + slice_id, maximum - minimum)


@triton.jit
def _pack_columns(Cols, Ptr, Bases, Short, BLOCK: tl.constexpr):
    slice_id = tl.program_id(0)
    start = tl.load(Ptr + slice_id)
    end = tl.load(Ptr + slice_id + 1)
    base = tl.load(Bases + slice_id)
    positions = start + tl.arange(0, BLOCK)
    columns = tl.load(Cols + positions, positions < end, 0)
    tl.store(Short + positions, columns - base, positions < end)


def _narrow_columns(plan, n_cols, height, maximum):
    columns, offsets, lengths, _rows, _gather = plan
    bases = torch.empty(offsets.numel() - 1, dtype=torch.int32, device=columns.device)
    spans = torch.empty_like(bases)
    block = triton.next_power_of_2(height * maximum)
    _column_bounds[(bases.numel(),)](
        columns, offsets, lengths, bases, spans, n_cols, height, block, num_warps=4,
    )
    narrow = spans <= 65535
    short_mode = True
    if not bool(torch.all(narrow)):
        if (height > 16 or maximum < 32 or not bool(
                torch.where(narrow, offsets[1:] - offsets[:-1], 0).sum() * 4 >= columns.numel() * 3)):
            return columns, offsets, False
        short_mode = 2
    short = torch.empty_like(columns, dtype=torch.uint16)
    _pack_columns[(bases.numel(),)](columns, offsets, bases, short, block, num_warps=4)
    if short_mode == 2:
        bases = torch.where(narrow, bases, -1)
    return short, bases, short_mode


def _prepare(values, plan, n_cols, height, maximum, conjugate=False):
    real, columns, encoding, colbits = _encode_real(values, plan, n_cols)
    short, bases = False, plan[1]
    if encoding in (0, 1, 4):
        columns, bases, short = _narrow_columns(plan, n_cols, height, maximum)
    metadata = ((plan[3].long() & 0xFFFFFF) | (plan[2].long() << 24)).int()
    imag = values.imag[plan[4]].contiguous()
    if conjugate:
        imag = -imag
    return {
        "real": real, "imag": imag,
        "cols": columns, "bases": bases, "metadata": metadata,
        "encoding": encoding, "colbits": colbits, "short": short,
    }


@triton.jit
def _kernel(Real, Imag, Cols, Bases, Ptr, Meta, Vector, Target, FullCols,
            STRIDE: tl.constexpr, HEIGHT: tl.constexpr, LANES: tl.constexpr,
            ENCODING: tl.constexpr, COLBITS: tl.constexpr, SHORT: tl.constexpr,
            FIRST: tl.constexpr, UNROLL: tl.constexpr):
    slice_id = tl.program_id(0)
    lanes = tl.arange(0, HEIGHT)
    slots = tl.arange(0, LANES)
    parts = tl.arange(0, 2)
    start = tl.load(Ptr + slice_id)
    metadata = tl.load(Meta + slice_id * HEIGHT + lanes).to(tl.uint32)
    lengths = (metadata >> 24).to(tl.int32)
    rows = (metadata & 0xFFFFFF).to(tl.int32)
    rows = tl.where(rows == 0xFFFFFF, -1, rows)
    if FIRST:
        width = tl.max(lengths, 0)
    else:
        width = (tl.load(Ptr + slice_id + 1) - start) // HEIGHT
    if SHORT:
        column_base = tl.load(Bases + slice_id)
    if ENCODING == 3:
        lanes = lanes[:, None]
        slots = slots[None, :]
        lengths = lengths[:, None]
        SHAPE: tl.constexpr = (HEIGHT, LANES)
        AXIS: tl.constexpr = 1
    else:
        lanes = lanes[None, :]
        slots = slots[:, None]
        lengths = lengths[None, :]
        SHAPE: tl.constexpr = (LANES, HEIGHT)
        AXIS: tl.constexpr = 0
    total_real = tl.full(SHAPE, 0., tl.float32)
    total_imag = tl.full(SHAPE, 0., tl.float32)
    imag_real = tl.full(SHAPE, 0., tl.float32)
    imag_imag = tl.full(SHAPE, 0., tl.float32)
    for base in tl.range(0, width, LANES, loop_unroll_factor=UNROLL):
        valid = base + slots < lengths
        positions = start + (base + slots) * HEIGHT + lanes
        if SHORT == 2:
            if column_base >= 0:
                payload = tl.load(Cols + positions, valid, 0, cache_modifier=".cg").to(tl.uint32)
            else:
                payload = tl.load(FullCols + positions, valid, 0, cache_modifier=".cg").to(tl.uint32)
        else:
            payload = tl.load(Cols + positions, valid, 0, cache_modifier=".cg").to(tl.uint32)
        if ENCODING == 2 or ENCODING == 3:
            columns = (payload & ((1 << COLBITS) - 1)).to(tl.int32)
        else:
            columns = payload.to(tl.int32)
        if SHORT:
            columns += tl.maximum(column_base, 0)
        vector = tl.load(Vector + 2 * columns[:, :, None] + parts[None, None, :],
                         valid[:, :, None], 0., cache_modifier=".ca")
        xr, xi = tl.split(vector)
        if ENCODING == 1:
            real = tl.where(valid, tl.load(Real), 0.)
        elif ENCODING == 4:
            real = 1.
        elif ENCODING == 2:
            real = (payload >> COLBITS).to(tl.int8).to(tl.float32)
        elif ENCODING == 3:
            code = (payload >> COLBITS).to(tl.int32)
            real = tl.load(Real + code, valid, 0., cache_modifier=".ca")
        else:
            real = tl.load(Real + positions, valid, 0., cache_modifier=".cg")
        imag = tl.load(Imag + positions, valid, 0., cache_modifier=".cg")
        total_real += real * xr
        total_imag += real * xi
        imag_real += imag * xi
        imag_imag += imag * xr
    result = tl.join(tl.sum(total_real - imag_real, AXIS), tl.sum(total_imag + imag_imag, AXIS))
    tl.store(Target + 2 * STRIDE * rows[:, None] + parts[None, :], result, rows[:, None] >= 0)


@triton.jit
def _integer_kernel(Imag, Cols, Ptr, Meta, Vector, Target,
                    STRIDE: tl.constexpr, SLICES: tl.constexpr):
    row_lanes = tl.arange(0, 64)
    slots = tl.arange(0, 2)[None, :]
    parts = tl.arange(0, 2)
    slice_ids = tl.program_id(0) * 4 + row_lanes // 16
    present = slice_ids < SLICES
    start = tl.load(Ptr + slice_ids, present, 0)[:, None]
    metadata = tl.load(Meta + tl.program_id(0) * 64 + row_lanes, present, 0xFFFFFF).to(tl.uint32)
    lengths = (metadata >> 24).to(tl.int32)
    width = tl.max(lengths, 0)
    rows = (metadata & 0xFFFFFF).to(tl.int32)
    lanes = (row_lanes % 16)[:, None]
    lengths = lengths[:, None]
    total_real = tl.full((64, 2), 0., tl.float32)
    total_imag = tl.full((64, 2), 0., tl.float32)
    imag_real = tl.full((64, 2), 0., tl.float32)
    imag_imag = tl.full((64, 2), 0., tl.float32)
    for base in range(0, width, 2):
        valid = base + slots < lengths
        positions = start + (base + slots) * 16 + lanes
        payload = tl.load(Cols + positions, valid, 0, cache_modifier=".cg").to(tl.uint32)
        columns = (payload & 0xFFFFFF).to(tl.int32)
        vector = tl.load(Vector + 2 * columns[:, :, None] + parts[None, None, :],
                         valid[:, :, None], 0., cache_modifier=".ca")
        xr, xi = tl.split(vector)
        real = (payload >> 24).to(tl.int8).to(tl.float32)
        imag = tl.load(Imag + positions, valid, 0., cache_modifier=".cg")
        total_real += real * xr
        total_imag += real * xi
        imag_real += imag * xi
        imag_imag += imag * xr
    result = tl.join(tl.sum(total_real - imag_real, 1), tl.sum(total_imag + imag_imag, 1))
    tl.store(Target + 2 * STRIDE * rows[:, None] + parts[None, :], result, rows[:, None] != 0xFFFFFF)


def run(values, plan, vector, target, n_rows, cache, height, lanes, maximum, version, conjugate=False):
    if (not values.numel() or n_rows >= 0xFFFFFF or maximum >= 256
            or vector.numel() > (1 << 30)):
        return False
    storage = cache.get("ell_compressed")
    if (storage is None or storage["plan"] is not plan
            or storage["source"] is not values or storage["version"] != version
            or storage["n_cols"] != vector.numel() or storage["conjugate"] != conjugate):
        storage = _prepare(values, plan, vector.numel(), height, maximum, conjugate)
        storage.update(plan=plan, source=values, version=version, n_cols=vector.numel(), conjugate=conjugate)
        cache["ell_compressed"] = storage
    if storage["encoding"] == 2 and height == 16 and maximum <= 32:
        _integer_kernel[(triton.cdiv(n_rows, 64),)](
            storage["imag"], storage["cols"], plan[1], storage["metadata"],
            torch.view_as_real(vector), torch.view_as_real(target),
            target.stride(0), triton.cdiv(n_rows, height), num_warps=1, num_stages=1,
        )
        return True
    _kernel[(triton.cdiv(n_rows, height),)](
        storage["real"], storage["imag"], storage["cols"], storage["bases"],
        plan[1], storage["metadata"], torch.view_as_real(vector), torch.view_as_real(target), plan[0],
        target.stride(0), height, lanes, storage["encoding"], storage["colbits"],
        storage["short"], FIRST=storage["encoding"] == 0 or lanes == 2,
        UNROLL=1 if lanes == 2 else 2, num_warps=1, num_stages=1,
    )
    return True
