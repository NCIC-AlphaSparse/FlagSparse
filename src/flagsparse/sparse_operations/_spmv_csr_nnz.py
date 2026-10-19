"""Prepared NNZ-balanced CSR execution for short, irregular rows on gfx936."""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _segmented_add(left_row, left_value, right_row, right_value):
    return right_row, right_value + tl.where(left_row == right_row, left_value, 0.)


@triton.jit
def _musa_segmented_add(left_row, left_value, right_row, right_value):
    return right_row, right_value + tl.where(left_row == right_row, left_value, 0)


@triton.jit
def _nnz_kernel(A, CI, RP, ROW, X, Y, EMPTY, NE: tl.constexpr,
                NNZ: tl.constexpr, B: tl.constexpr, ACC: tl.constexpr,
                MUSA: tl.constexpr = False, COMPLEX_RHS: tl.constexpr = False):
    pid = tl.program_id(0)
    lane = tl.arange(0, B)
    pos = pid * B + lane
    empty_row = tl.load(EMPTY + pos, pos < NE, 0)
    if COMPLEX_RHS:
        tl.store(Y + 2 * empty_row, 0., pos < NE)
        tl.store(Y + 2 * empty_row + 1, 0., pos < NE)
    else:
        tl.store(Y + empty_row, 0., pos < NE)
    valid = pos < NNZ
    row = tl.load(ROW + pos, valid, -1)
    col = tl.load(CI + pos, valid, 0)
    value = tl.load(A + pos, valid, 0).to(ACC)
    vector = tl.load(X + (2 * col if COMPLEX_RHS else col), valid, 0).to(ACC)
    if MUSA:
        _, total = tl.associative_scan((row, value * vector), 0, _musa_segmented_add)
    else:
        _, total = tl.associative_scan((row, value * vector), 0, _segmented_add)
    if COMPLEX_RHS:
        vector_im = tl.load(X + 2 * col + 1, valid, 0).to(ACC)
        _, total_im = tl.associative_scan((row, value * vector_im), 0, _musa_segmented_add)
    next_row = tl.load(ROW + pos + 1, pos + 1 < NNZ, -1)
    # Only the block containing a row's last nonzero owns its output.
    boundary = valid & (row != next_row)
    first_row = tl.load(ROW + pid * B, pid * B < NNZ, 0)
    start = tl.load(RP + first_row)
    end = tl.load(RP + first_row + 1)
    prefix = tl.full((), 0., ACC)
    if COMPLEX_RHS:
        prefix_im = tl.full((), 0., ACC)
    if (start < pid * B) & (end <= (pid + 1) * B) & (pid * B < NNZ):
        # Recompute the short prefix crossing a block boundary, avoiding an
        # atomic update or a second launch to merge a partial-sum buffer.
        for offset in range(start, pid * B, 128):
            index = offset + tl.arange(0, 128)
            active = index < pid * B
            column = tl.load(CI + index, active, 0)
            a = tl.load(A + index, active, 0).to(ACC)
            x = tl.load(X + (2 * column if COMPLEX_RHS else column), active, 0).to(ACC)
            prefix += tl.sum(a * x, 0)
            if COMPLEX_RHS:
                xi = tl.load(X + 2 * column + 1, active, 0).to(ACC)
                prefix_im += tl.sum(a * xi, 0)
    if MUSA:
        total += tl.where(row == first_row, prefix, 0)
    else:
        total += tl.where(row == first_row, prefix, 0.)
    if COMPLEX_RHS:
        total_im += tl.where(row == first_row, prefix_im, 0)
        tl.store(Y + 2 * row, total, boundary)
        tl.store(Y + 2 * row + 1, total_im, boundary)
    else:
        tl.store(Y + row, total, boundary)


def prepare(prepared):
    """Build index-only metadata once; values and vectors remain live inputs."""
    device = prepared.data.device
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    rows = torch.repeat_interleave(
        torch.arange(prepared.n_rows, device=device, dtype=torch.int32),
        prepared.row_lengths,
        output_size=prepared.data.numel(),
    )
    empty = torch.nonzero(prepared.row_lengths == 0).flatten().to(torch.int32)
    torch.cuda.synchronize(device)
    return {
        "rows": rows,
        "empty": empty,
        "prepare_ms": (time.perf_counter() - start) * 1000,
        "metadata_bytes": (rows.numel() + empty.numel()) * 4,
    }


def bind(prepared, x, out):
    """Bind the same native launch used by the public API and benchmarks."""
    plan = prepared.rocm_nnz_plan
    nnz = prepared.data.numel()
    empty_count = plan["empty"].numel()
    block = 256
    grid = (triton.cdiv(max(nnz, empty_count), block),)
    acc = tl.float32 if prepared.data.dtype == torch.float32 else tl.float64

    def run():
        if grid[0]:
            _nnz_kernel[grid](
                prepared.data, prepared.kernel_indices, prepared.kernel_indptr,
                plan["rows"], x, out, plan["empty"], empty_count, nnz, block, acc,
                num_warps=2,
            )
        return out

    return run
