# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CSR SpMM transpose scatter kernel for the C API.

For ``A^T @ B`` the native CSR rows are source rows, while the output row is
the column index of each nonzero.  Programs therefore scatter their dense
tiles with atomics.  The C++ route applies beta to C before this kernel.
"""

import triton
import triton.language as tl


@triton.jit
def spmm_csr_transpose_atomic_f32(
    values_ptr,
    cols_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    alpha,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
):
    row = tl.program_id(0)
    n_block = tl.program_id(1)
    segment = tl.program_id(2)
    if row >= n_rows:
        return

    start = tl.load(indptr_ptr + row) + segment * BLOCK_NNZ
    end = tl.load(indptr_ptr + row + 1)
    nnz_offs = tl.arange(0, BLOCK_NNZ)
    positions = start + nnz_offs
    valid_nnz = positions < end
    cols = tl.load(cols_ptr + positions, mask=valid_nnz, other=0).to(tl.int64)
    values = tl.load(values_ptr + positions, mask=valid_nnz, other=0.0)

    n_offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = n_offs < n_dense_cols
    # B is indexed by the source CSR row.  Each A[row, col] contributes to
    # C[col, :] in the transposed product.
    b = tl.load(
        b_ptr + row * stride_bk + n_offs * stride_bn,
        mask=valid_n,
        other=0.0,
    )
    contribution = alpha * values[:, None] * b[None, :]
    out_ptr = c_ptr + cols[:, None] * stride_cm + n_offs[None, :] * stride_cn
    tl.atomic_add(out_ptr, contribution, mask=valid_nnz[:, None] & valid_n[None, :], sem="relaxed")


__all__ = ["spmm_csr_transpose_atomic_f32"]
