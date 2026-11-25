# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Symbolic product maps and live numerical evaluation for CSR SpGEMM."""

import torch
import triton
import triton.language as tl


def build_symbolic(prepared, max_products):
    lengths = prepared.b_indptr[1:] - prepared.b_indptr[:-1]
    expansion = lengths[prepared.a_indices.long()].long()
    count = int(expansion.sum().item())
    if count > max_products:
        return ()
    device = expansion.device
    product_a = torch.repeat_interleave(
        torch.arange(expansion.numel(), device=device), expansion, output_size=count,
    )
    starts = torch.cumsum(expansion, 0) - expansion
    product_b = (
        prepared.b_indptr[prepared.a_indices[product_a].long()]
        + torch.arange(count, device=device) - starts[product_a]
    )
    row_a = torch.repeat_interleave(
        torch.arange(prepared.n_rows, device=device),
        (prepared.a_indptr[1:] - prepared.a_indptr[:-1]).long(),
        output_size=prepared.a_indices.numel(),
    )
    keys = row_a[product_a] * prepared.n_cols + prepared.b_indices[product_b]
    order = torch.argsort(keys, stable=True)
    keys, multiplicity = torch.unique_consecutive(keys[order], return_counts=True)
    product_ptr = torch.cat((multiplicity.new_zeros(1), multiplicity.cumsum(0)))
    rows = keys // max(prepared.n_cols, 1)
    counts = torch.bincount(rows, minlength=prepared.n_rows)
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    return (
        product_a[order].to(torch.int32), product_b[order].to(torch.int32),
        product_ptr, (keys % max(prepared.n_cols, 1)).to(torch.int32), ptr,
        count / max(keys.numel(), 1),
    )


@triton.jit
def _numeric(A, B, PA, PB, P, C, NNZ: tl.constexpr,
             R: tl.constexpr, K: tl.constexpr, COMPLEX: tl.constexpr):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    start = tl.load(P + rows, rows < NNZ, 0)
    end = tl.load(P + rows + 1, rows < NNZ, 0)
    lane = tl.arange(0, K)
    real = tl.zeros((R, K), tl.float32)
    imag = tl.zeros((R, K), tl.float32)
    for base in range(0, tl.max(end - start, 0), K):
        pos = start[:, None] + base + lane[None, :]
        mask = (rows[:, None] < NNZ) & (pos < end[:, None])
        left = tl.load(PA + pos, mask, 0)
        right = tl.load(PB + pos, mask, 0)
        if COMPLEX:
            ar = tl.load(A + 2 * left, mask, 0.)
            ai = tl.load(A + 2 * left + 1, mask, 0.)
            br = tl.load(B + 2 * right, mask, 0.)
            bi = tl.load(B + 2 * right + 1, mask, 0.)
            real += ar * br - ai * bi
            imag += ar * bi + ai * br
        else:
            real += tl.load(A + left, mask, 0.) * tl.load(B + right, mask, 0.)
    if COMPLEX:
        tl.store(C + 2 * rows, tl.sum(real, 1), rows < NNZ)
        tl.store(C + 2 * rows + 1, tl.sum(imag, 1), rows < NNZ)
    else:
        tl.store(C + rows, tl.sum(real, 1), rows < NNZ)


def evaluate_symbolic(prepared, symbolic):
    left, right, product_ptr, cols, ptr, mean_products = symbolic
    data = torch.empty(cols.numel(), dtype=prepared.a_data.dtype, device=prepared.a_data.device)
    complex_values = data.is_complex()
    a = prepared.a_data.resolve_neg()
    b = prepared.b_data.resolve_neg()
    matrix_a, matrix_b, target = (
        (torch.view_as_real(value.resolve_conj()) for value in (a, b, data))
        if complex_values else (a, b, data)
    )
    batch, lanes = (16, 4) if mean_products >= 8 else (32, 2)
    if cols.numel():
        _numeric[(triton.cdiv(cols.numel(), batch),)](
            matrix_a, matrix_b, left, right, product_ptr, target,
            cols.numel(), batch, lanes, complex_values, num_warps=1, enable_fp_fusion=False,
        )
    return data, cols.clone(), ptr.clone()
