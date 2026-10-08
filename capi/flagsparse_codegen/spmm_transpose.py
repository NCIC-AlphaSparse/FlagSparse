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
    nnz,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    BALANCED: tl.constexpr,
):
    n_block = tl.program_id(1)
    # Upper-bound search handles empty rows and balances work by actual nnz,
    # avoiding rows * max_segments programs for skewed CSR matrices.
    if BALANCED:
        positions = tl.program_id(0) * BLOCK_NNZ + tl.arange(0, BLOCK_NNZ)
        valid_nnz = positions < nnz
        lo = tl.full((BLOCK_NNZ,), 0, tl.int64)
        hi = tl.full((BLOCK_NNZ,), n_rows, tl.int64)
        while tl.sum((lo < hi).to(tl.int32), 0) > 0:
            mid = (lo + hi) // 2
            boundary = tl.load(indptr_ptr + mid + 1, mask=mid < n_rows, other=nnz)
            right = boundary <= positions
            active = lo < hi
            lo = tl.where(active & right, mid + 1, lo)
            hi = tl.where(active & ~right, mid, hi)
        row = lo
    else:
        row = tl.program_id(0)
        positions = (tl.load(indptr_ptr + row) + tl.program_id(2) * BLOCK_NNZ
                     + tl.arange(0, BLOCK_NNZ))
        valid_nnz = positions < tl.load(indptr_ptr + row + 1)
    cols = tl.load(cols_ptr + positions, mask=valid_nnz, other=0).to(tl.int64)
    values = tl.load(values_ptr + positions, mask=valid_nnz, other=0.0)

    n_offs = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    valid_n = n_offs < n_dense_cols
    # B is indexed by the source CSR row.  Each A[row, col] contributes to
    # C[col, :] in the transposed product.
    if BALANCED:
        b = tl.load(
            b_ptr + row[:, None] * stride_bk + n_offs[None, :] * stride_bn,
            mask=valid_nnz[:, None] & valid_n[None, :], other=0.0,
        )
        contribution = alpha * values[:, None] * b
    else:
        b = tl.load(b_ptr + row * stride_bk + n_offs * stride_bn, valid_n, other=0.)
        contribution = alpha * values[:, None] * b[None, :]
    out_ptr = c_ptr + cols[:, None] * stride_cm + n_offs[None, :] * stride_cn
    tl.atomic_add(out_ptr, contribution, mask=valid_nnz[:, None] & valid_n[None, :], sem="relaxed")


@triton.jit
def spmm_csr_transpose_gather_f32(
    V, P, ORDER, ROWS, B, C, alpha, beta, M, N,
    SBK, SBN, SCM, SCN,
    R: tl.constexpr, K: tl.constexpr, BN: tl.constexpr,
):
    rows = tl.program_id(0) * R + tl.arange(0, R)
    valid_rows = rows < M
    start = tl.load(P + rows, valid_rows, other=0)
    end = tl.load(P + rows + 1, valid_rows, other=0)
    ns = tl.program_id(1) * BN + tl.arange(0, BN)
    ks = tl.arange(0, K)
    acc = tl.zeros((R, BN), tl.float32)
    for base in range(0, tl.max(end - start, 0), K):
        pos = start[:, None] + base + ks[None, :]
        valid = valid_rows[:, None] & (pos < end[:, None])
        src = tl.load(ORDER + pos, valid, other=0)
        col = tl.load(ROWS + pos, valid, other=0)
        value = tl.load(V + src, valid, other=0.)
        dense = tl.load(B + col[:, :, None] * SBK + ns[None, None, :] * SBN,
                        valid[:, :, None] & (ns[None, None, :] < N), other=0.)
        acc += tl.sum(value[:, :, None] * dense, 1)
    out = C + rows[:, None] * SCM + ns[None, :] * SCN
    mask = valid_rows[:, None] & (ns[None, :] < N)
    result = alpha * acc
    if beta != 0.:
        result += beta * tl.load(out, mask, other=0.)
    tl.store(out, result, mask)


__all__ = ["spmm_csr_transpose_atomic_f32", "spmm_csr_transpose_gather_f32"]
