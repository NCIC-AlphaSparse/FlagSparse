# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MUSA narrow-input SpVV: independent, deterministic Triton reductions."""

import torch
import triton
import triton.language as tl

from ._musa_spmv_launch import launch


@triton.jit
def _spvv_half_part(OUT, X, Y, IDX, N, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    indices = tl.load(IDX + offsets, mask, other=0)
    lhs = tl.load(X + offsets, mask, other=0).to(tl.float32)
    rhs = tl.load(Y + indices, mask, other=0).to(tl.float32)
    tl.store(OUT + tl.program_id(0), tl.sum(lhs * rhs, 0))


@triton.jit
def _spvv_half_finish(OUT, PART, N, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    partials = tl.zeros((BLOCK,), tl.float32)
    for base in range(0, N, BLOCK):
        partials += tl.load(PART + base + offsets, base + offsets < N, other=0.)
    tl.store(OUT, tl.sum(partials, 0))


def spvv_half(values, indices, y):
    """Return a fresh FP32 scalar, reading all operands on the current stream."""
    nnz = values.numel()
    indices = indices.contiguous()
    result = torch.empty((), dtype=torch.float32, device=y.device)
    block = triton.next_power_of_2(max(1, nnz)) if nnz <= 16384 else 1024
    parts = max(1, triton.cdiv(nnz, block))
    partial = result if parts == 1 else torch.empty(parts, dtype=torch.float32, device=y.device)
    launch(_spvv_half_part, (parts,), (partial, values, y, indices),
           (nnz,), (block,), num_warps=8 if block >= 8192 else 4)
    if parts > 1:
        launch(_spvv_half_finish, (1,), (result, partial), (parts,),
               (min(1024, triton.next_power_of_2(parts)),), num_warps=4)
    return result


@triton.jit
def _spvv_int8_part(OUT, X, Y, IDX, N, ACC: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    indices = tl.load(IDX + offsets, mask, other=0)
    lhs = tl.load(X + offsets, mask, other=0).to(ACC)
    rhs = tl.load(Y + indices, mask, other=0).to(ACC)
    tl.store(OUT + tl.program_id(0), tl.sum(lhs * rhs, 0))


@triton.jit
def _spvv_int8_finish(OUT, PART, N, ACC: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    partials = tl.zeros((BLOCK,), ACC)
    for base in range(0, N, BLOCK):
        partials += tl.load(PART + base + offsets, base + offsets < N, other=0)
    tl.store(OUT, tl.sum(partials, 0))


def spvv_int8(values, indices, y, compute_dtype):
    """Keep int32/FP32 arithmetic through both stages and return a fresh scalar."""
    nnz = values.numel()
    indices = indices.contiguous()
    result = torch.empty((), dtype=compute_dtype, device=y.device)
    acc = tl.int32 if compute_dtype == torch.int32 else tl.float32
    block = triton.next_power_of_2(max(1, nnz)) if nnz <= 16384 else 1024
    parts = max(1, triton.cdiv(nnz, block))
    partial = result if parts == 1 else torch.empty(parts, dtype=compute_dtype, device=y.device)
    launch(_spvv_int8_part, (parts,), (partial, values, y, indices),
           (nnz,), (acc, block), num_warps=8 if block >= 8192 else 4)
    if parts > 1:
        launch(_spvv_int8_finish, (1,), (result, partial), (parts,),
               (acc, min(1024, triton.next_power_of_2(parts))), num_warps=4)
    return result
