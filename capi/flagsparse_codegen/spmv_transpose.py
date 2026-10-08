# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Atomic transpose/conjugate-transpose SpMV kernels for the C API.

CSR keeps its native row offsets and COO keeps its coordinate arrays.  Both
formats therefore scatter each contribution into the transposed output; the
C++ route applies beta before launch because no program owns an output entry.
"""

import triton
import triton.language as tl


@triton.jit
def spmv_csr_transpose_atomic_f32(
    values_ptr, cols_ptr, indptr_ptr, x_ptr, y_ptr, alpha, n_rows,
    BLOCK: tl.constexpr, SEGMENTS: tl.constexpr,
):
    row = tl.program_id(0)
    segment = tl.program_id(1)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row) + segment * BLOCK
    end = tl.load(indptr_ptr + row + 1)
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < end
    cols = tl.load(cols_ptr + offsets, mask=mask, other=0)
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + row)
    tl.atomic_add(y_ptr + cols, alpha * values * x, mask=mask, sem="relaxed")


@triton.jit
def spmv_csr_transpose_atomic_c32(
    values_ri_ptr, cols_ptr, indptr_ptr, x_ri_ptr, y_ri_ptr,
    alpha_re, alpha_im, n_rows,
    BLOCK: tl.constexpr, SEGMENTS: tl.constexpr, CONJ: tl.constexpr,
):
    row = tl.program_id(0)
    segment = tl.program_id(1)
    if row >= n_rows:
        return
    start = tl.load(indptr_ptr + row) + segment * BLOCK
    end = tl.load(indptr_ptr + row + 1)
    offsets = start + tl.arange(0, BLOCK)
    mask = offsets < end
    cols = tl.load(cols_ptr + offsets, mask=mask, other=0)
    a_re = tl.load(values_ri_ptr + offsets * 2, mask=mask, other=0.0)
    a_im = tl.load(values_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
    a_im = -a_im if CONJ else a_im
    x_re = tl.load(x_ri_ptr + row * 2)
    x_im = tl.load(x_ri_ptr + row * 2 + 1)
    prod_re = a_re * x_re - a_im * x_im
    prod_im = a_re * x_im + a_im * x_re
    out_re = alpha_re * prod_re - alpha_im * prod_im
    out_im = alpha_re * prod_im + alpha_im * prod_re
    tl.atomic_add(y_ri_ptr + cols * 2, out_re, mask=mask, sem="relaxed")
    tl.atomic_add(y_ri_ptr + cols * 2 + 1, out_im, mask=mask, sem="relaxed")


@triton.jit
def spmv_coo_transpose_atomic_f32(
    values_ptr, rows_ptr, cols_ptr, x_ptr, y_ptr, alpha, nnz,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < nnz
    rows = tl.load(rows_ptr + offsets, mask=mask, other=0)
    cols = tl.load(cols_ptr + offsets, mask=mask, other=0)
    values = tl.load(values_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + rows, mask=mask, other=0.0)
    tl.atomic_add(y_ptr + cols, alpha * values * x, mask=mask, sem="relaxed")


@triton.jit
def spmv_coo_transpose_atomic_c32(
    values_ri_ptr, rows_ptr, cols_ptr, x_ri_ptr, y_ri_ptr,
    alpha_re, alpha_im, nnz,
    BLOCK: tl.constexpr, CONJ: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < nnz
    rows = tl.load(rows_ptr + offsets, mask=mask, other=0)
    cols = tl.load(cols_ptr + offsets, mask=mask, other=0)
    a_re = tl.load(values_ri_ptr + offsets * 2, mask=mask, other=0.0)
    a_im = tl.load(values_ri_ptr + offsets * 2 + 1, mask=mask, other=0.0)
    a_im = -a_im if CONJ else a_im
    x_re = tl.load(x_ri_ptr + rows * 2, mask=mask, other=0.0)
    x_im = tl.load(x_ri_ptr + rows * 2 + 1, mask=mask, other=0.0)
    prod_re = a_re * x_re - a_im * x_im
    prod_im = a_re * x_im + a_im * x_re
    out_re = alpha_re * prod_re - alpha_im * prod_im
    out_im = alpha_re * prod_im + alpha_im * prod_re
    tl.atomic_add(y_ri_ptr + cols * 2, out_re, mask=mask, sem="relaxed")
    tl.atomic_add(y_ri_ptr + cols * 2 + 1, out_im, mask=mask, sem="relaxed")
