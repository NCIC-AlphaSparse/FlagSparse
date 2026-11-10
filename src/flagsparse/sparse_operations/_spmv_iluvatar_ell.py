"""Versioned SELL tiles with jointly loaded complex vector components."""

import os

import torch
import triton
import triton.language as tl


def eligible(n_rows, mean, maximum):
    return 0 < n_rows <= 262144 and 12 < mean <= 32 and maximum <= 64


def _configuration(n_rows, mean, maximum, complex_values):
    if n_rows <= 0:
        return None
    if complex_values and mean <= 3 and maximum <= 16:
        return 32, 2
    if mean > (4 if complex_values else 8) and maximum <= 64:
        return 16, 4
    if mean > 12 and maximum <= 256:
        return 8, 8
    return None


def _layout(cols, ptr, n_rows, height=16, sigma=4096):
    lengths = (ptr[1:] - ptr[:-1]).long()
    maximum = int(lengths.max())
    window = torch.arange(n_rows, device=ptr.device) // sigma
    order = torch.argsort(window * (maximum + 1) + maximum - lengths, stable=True)
    position = torch.empty_like(order)
    position[order] = torch.arange(n_rows, device=ptr.device)
    n_slices = triton.cdiv(n_rows, height)
    padded_lengths = torch.zeros(n_slices * height, dtype=torch.int32, device=ptr.device)
    padded_lengths[:n_rows] = lengths[order].int()
    widths = padded_lengths.reshape(n_slices, height).amax(1).long()
    offsets = torch.cat((widths.new_zeros(1), (widths * height).cumsum(0)))
    rows = torch.repeat_interleave(torch.arange(n_rows, device=ptr.device), lengths,
                                   output_size=cols.numel())
    slot = torch.arange(cols.numel(), device=ptr.device) - ptr[rows]
    lane = position[rows]
    destination = offsets[lane // height] + slot * height + lane.remainder(height)
    total = int(offsets[-1])
    packed_cols = torch.zeros(total, dtype=cols.dtype, device=cols.device)
    packed_cols[destination] = cols
    gather = torch.zeros(total, dtype=torch.int64, device=cols.device)
    gather[destination] = torch.arange(cols.numel(), device=cols.device)
    permutation = torch.full((n_slices * height,), -1, dtype=torch.int32, device=ptr.device)
    permutation[:n_rows] = order.int()
    return packed_cols, offsets.int(), padded_lengths, permutation, gather


@triton.jit
def _kernel(Real, Imag, Cols, Ptr, Lengths, Perm, Vector, Target,
             COMPLEX: tl.constexpr, CONJUGATE: tl.constexpr, STRIDE: tl.constexpr,
             HEIGHT: tl.constexpr, LANES: tl.constexpr,
             PIPELINED: tl.constexpr = False):
    slice_id = tl.program_id(0)
    lanes = tl.arange(0, HEIGHT)
    slots = tl.arange(0, LANES)
    parts = tl.arange(0, 2)
    start = tl.load(Ptr + slice_id)
    width = (tl.load(Ptr + slice_id + 1) - start) // HEIGHT
    lengths = tl.load(Lengths + slice_id * HEIGHT + lanes)
    rows = tl.load(Perm + slice_id * HEIGHT + lanes)
    if COMPLEX:
        total_real = tl.full((LANES, HEIGHT), 0., tl.float32)
        total_imag = tl.full((LANES, HEIGHT), 0., tl.float32)
        if PIPELINED:
            imag_real = tl.full((LANES, HEIGHT), 0., tl.float32)
            imag_imag = tl.full((LANES, HEIGHT), 0., tl.float32)
    else:
        total = tl.full((LANES, HEIGHT, 2), 0., tl.float32)
    for base in tl.range(0, width, LANES, loop_unroll_factor=2 if PIPELINED else 1):
        valid = base + slots[:, None] < lengths[None, :]
        positions = start + (base + slots[:, None]) * HEIGHT + lanes[None, :]
        columns = tl.load(Cols + positions, valid, 0, cache_modifier=".cg")
        vector = tl.load(Vector + 2 * columns[:, :, None] + parts[None, None, :],
                         valid[:, :, None], 0., cache_modifier=".ca")
        real = tl.load(Real + positions, valid, 0., cache_modifier=".cg")
        if COMPLEX:
            xr, xi = tl.split(vector)
            imag = tl.load(Imag + positions, valid, 0., cache_modifier=".cg")
            if CONJUGATE:
                imag = -imag
            total_real += real * xr
            total_imag += real * xi
            if PIPELINED:
                imag_real += imag * xi
                imag_imag += imag * xr
            else:
                total_real -= imag * xi
                total_imag += imag * xr
        else:
            total += real[:, :, None] * vector
    if COMPLEX:
        if PIPELINED:
            total_real -= imag_real
            total_imag += imag_imag
        result = tl.join(tl.sum(total_real, 0), tl.sum(total_imag, 0))
    else:
        result = tl.sum(total, 0)
    tl.store(Target + 2 * STRIDE * rows[:, None] + parts[None, :], result, rows[:, None] >= 0)


def run(values, cols, ptr, x, out, n_rows, cache, conjugate=False):
    if values.dtype not in (torch.float32, torch.complex64) or x.dtype != torch.complex64:
        return False
    prefix = "GATHER" if values.is_complex() else "MIXED"
    options = ("BATCH", "LANES", "WARPS", "STAGES") if values.is_complex() else ("ROWS", "BLOCK", "WARPS")
    if (any(os.environ.get("FLAGSPARSE_ILUVATAR_" + prefix + "_" + option) for option in options)
            or (values.is_complex() and os.environ.get("FLAGSPARSE_ILUVATAR_GATHER_SPLIT", "1") == "0")):
        return False
    try:
        version = values._version
        topology_version = (cols._version, ptr._version)
    except RuntimeError:
        return False
    if (cache.get("ell_cols") is not cols or cache.get("ell_ptr") is not ptr
            or cache.get("ell_topology_version") != topology_version):
        cache.pop("ell_layout", None)
        cache.pop("ell_source", None)
        if "ell_topology_version" in cache:
            cache.pop("max_row_nnz", None)
            cache.pop("avg_row_nnz", None)
        cache.update(ell_cols=cols, ell_ptr=ptr, ell_topology_version=topology_version)
    mean = cache.get("avg_row_nnz")
    maximum = cache.get("max_row_nnz")
    if mean is None or maximum is None:
        mean = cols.numel() / max(1, n_rows)
        maximum = int((ptr[1:] - ptr[:-1]).max()) if n_rows else 0
        cache.update(avg_row_nnz=mean, max_row_nnz=maximum)
    configuration = _configuration(n_rows, mean, maximum, values.is_complex())
    if configuration is None:
        return False
    height, lanes = configuration
    x = x.resolve_conj().contiguous()
    plan = cache.get("ell_layout")
    if plan is None:
        sigma = 16384 if (
            cache.get("coo_c32_non") and values.is_complex() and not conjugate
            and n_rows <= 262144 and mean > 4 and maximum <= 64
        ) else 4096
        plan = _layout(cols, ptr, n_rows, height, sigma)
        cache["ell_layout"] = plan
    if values.is_complex() and (
            (cache.get("coo_c32_non") and not conjugate)
            or (cache.get("sell_c32_non") and not conjugate)
            or (cache.get("csc_c32_conj") and conjugate)):
        from . import _spmv_iluvatar_compressed

        if _spmv_iluvatar_compressed.run(
            values, plan, x, out, n_rows, cache, height, lanes, maximum, version,
            conjugate=conjugate,
        ):
            return True
    if cache.get("ell_source") is not values or cache.get("ell_version") != version:
        gather = plan[4]
        if values.is_complex():
            real = values.real[gather].contiguous()
            imag = values.imag[gather].contiguous()
        else:
            real = values[gather].contiguous()
            imag = real
        cache.update(ell_source=values, ell_version=version, ell_real=real, ell_imag=imag)
    _kernel[(triton.cdiv(n_rows, height),)](
        cache["ell_real"], cache["ell_imag"], *plan[:4],
        torch.view_as_real(x), torch.view_as_real(out), values.is_complex(), conjugate,
        out.stride(0), height, lanes, num_warps=1, num_stages=1,
        PIPELINED=bool(cache.get("coo_c32_non")) and values.is_complex() and not conjugate and lanes >= 4,
    )
    return True
