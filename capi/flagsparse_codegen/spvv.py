# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Single-result mixed-precision SpVV kernels for the C API."""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402


@triton.jit
def _spvv_mixed_kernel(
    result_ptr, x_ptr, y_ptr, idx_ptr, nnz,
    ACC: tl.constexpr, BLOCK: tl.constexpr,
):
    acc = tl.zeros((BLOCK,), dtype=ACC)
    for base in range(0, nnz, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < nnz
        idx = tl.load(idx_ptr + offs, mask=mask, other=0)
        x = tl.load(x_ptr + offs, mask=mask, other=0).to(ACC)
        y = tl.load(y_ptr + idx, mask=mask, other=0).to(ACC)
        acc += x * y
    tl.store(result_ptr, tl.sum(acc, axis=0))


@triton.jit
def spvv_mixed_i32acc(result_ptr, x_ptr, y_ptr, idx_ptr, nnz,
                      BLOCK: tl.constexpr):
    _spvv_mixed_kernel(result_ptr, x_ptr, y_ptr, idx_ptr, nnz,
                       ACC=tl.int32, BLOCK=BLOCK)


@triton.jit
def spvv_mixed_f32acc(result_ptr, x_ptr, y_ptr, idx_ptr, nnz,
                      BLOCK: tl.constexpr):
    _spvv_mixed_kernel(result_ptr, x_ptr, y_ptr, idx_ptr, nnz,
                       ACC=tl.float32, BLOCK=BLOCK)


@triton.jit
def spvv_c32(result_ri_ptr, x_ri_ptr, y_ri_ptr, idx_ptr, nnz,
             CONJ: tl.constexpr, BLOCK: tl.constexpr):
    acc_re = tl.zeros((BLOCK,), dtype=tl.float32)
    acc_im = tl.zeros((BLOCK,), dtype=tl.float32)
    for base in range(0, nnz, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < nnz
        idx = tl.load(idx_ptr + offs, mask=mask, other=0)
        xr = tl.load(x_ri_ptr + 2 * offs, mask=mask, other=0.0)
        xi = tl.load(x_ri_ptr + 2 * offs + 1, mask=mask, other=0.0)
        if CONJ:
            xi = -xi
        yr = tl.load(y_ri_ptr + 2 * idx, mask=mask, other=0.0)
        yi = tl.load(y_ri_ptr + 2 * idx + 1, mask=mask, other=0.0)
        acc_re += xr * yr - xi * yi
        acc_im += xr * yi + xi * yr
    tl.store(result_ri_ptr, tl.sum(acc_re, axis=0))
    tl.store(result_ri_ptr + 1, tl.sum(acc_im, axis=0))
