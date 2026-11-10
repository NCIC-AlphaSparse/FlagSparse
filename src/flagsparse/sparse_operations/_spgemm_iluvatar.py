"""Bounded symbolic product maps with live numerical SpGEMM evaluation."""

import torch
import triton
import triton.language as tl

from ._rocm_sparse_cache import TensorPlanCache


_SYMBOLIC_CACHE = TensorPlanCache(maxsize=1)
_MAX_PRODUCTS = 128000000


def _build_symbolic(prepared):
    lengths = prepared.b_indptr[1:] - prepared.b_indptr[:-1]
    expansion = lengths[prepared.a_indices.long()].long()
    count = int(expansion.sum().item())
    if count > _MAX_PRODUCTS:
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


def compute(prepared):
    sources = prepared.symbolic_sources
    if sources is None:
        sources = (prepared.a_indices, prepared.a_indptr, prepared.b_indices, prepared.b_indptr)
    if (prepared.n_rows * prepared.n_cols >= 2**63
            or max(prepared.a_data.numel(), prepared.b_data.numel()) >= 2**31
            or _SYMBOLIC_CACHE._signature(sources) is None):
        return None
    options = (prepared.a_shape, prepared.b_shape)
    symbolic = _SYMBOLIC_CACHE.get(sources, options)
    if symbolic is None:
        symbolic = _build_symbolic(prepared)
        _SYMBOLIC_CACHE.put(sources, options, symbolic)
    if not symbolic:
        return None
    left, right, product_ptr, cols, ptr, mean_products = symbolic
    data = torch.empty(cols.numel(), dtype=prepared.a_data.dtype, device=prepared.a_data.device)
    complex_values = data.is_complex()
    matrix_a, matrix_b, target = (
        (torch.view_as_real(value.resolve_conj()) for value in (prepared.a_data, prepared.b_data, data))
        if complex_values else (prepared.a_data, prepared.b_data, data)
    )
    batch, lanes = (16, 4) if mean_products >= 8 else (32, 2)
    if cols.numel():
        _numeric[(triton.cdiv(cols.numel(), batch),)](
            matrix_a, matrix_b, left, right, product_ptr, target,
            cols.numel(), batch, lanes, complex_values, num_warps=1, enable_fp_fusion=False,
        )
    return data, cols.clone(), ptr.clone()
