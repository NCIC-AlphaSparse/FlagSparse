# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Mixed-precision / int8 SpMV and SpMM kernels -- re-exported from
src/flagsparse/sparse_operations/mixed_spmx.py, not copied.

``_csr_spmv_mixed_kernel`` and ``_csr_spmm_mixed_kernel`` carry an ``ACC:
tl.constexpr`` (the accumulate dtype: int32 for int8->int32, float32 for
int8->float32 and float16/bfloat16->float32). The raw-args signature parser
cannot spell a ``tl.dtype`` constexpr directly (same limitation noted in
spmm_csr.py's shim for its own kernel_names), so this file provides one
bool-constexpr wrapper per accumulate choice, each delegating straight to the
one real kernel. ``_csr_spmv_real_by_complex_kernel``/``_csr_spmm_real_by_complex_kernel``
(the real-A-times-complex-x/B case) have no such constexpr and are re-exported
directly.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from flagsparse.sparse_operations.mixed_spmx import (  # noqa: E402,F401
    _coo_spmm_mixed_kernel,
    _coo_spmv_atomic_kernel,
    _csr_spmm_mixed_kernel,
    _csr_spmv_mixed_kernel,
    _csr_spmv_real_by_complex_kernel,
)


@triton.jit
def csr_spmv_mixed_i32acc(
    y_ptr, data_ptr, cols_ptr, indptr_ptr, x_ptr, n_rows,
    ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    """int8 matrix/vector -> int32 (ACC=int32)."""
    _csr_spmv_mixed_kernel(y_ptr, data_ptr, cols_ptr, indptr_ptr, x_ptr, n_rows,
                           ACC=tl.int32, ROWS=ROWS, BLOCK=BLOCK)


@triton.jit
def csr_spmv_mixed_f32acc(
    y_ptr, data_ptr, cols_ptr, indptr_ptr, x_ptr, n_rows,
    ROWS: tl.constexpr, BLOCK: tl.constexpr,
):
    """int8->float32, or float16/bfloat16->float32 (ACC=float32)."""
    _csr_spmv_mixed_kernel(y_ptr, data_ptr, cols_ptr, indptr_ptr, x_ptr, n_rows,
                           ACC=tl.float32, ROWS=ROWS, BLOCK=BLOCK)


@triton.jit
def coo_spmv_atomic_i32acc(
    acc_ptr, data_ptr, rows_ptr, cols_ptr, x_ptr, nnz,
    BLOCK: tl.constexpr,
):
    _coo_spmv_atomic_kernel(acc_ptr, data_ptr, rows_ptr, cols_ptr, x_ptr, nnz,
                            ACC=tl.int32, BLOCK=BLOCK)


@triton.jit
def coo_spmv_atomic_f32acc(
    acc_ptr, data_ptr, rows_ptr, cols_ptr, x_ptr, nnz,
    BLOCK: tl.constexpr,
):
    _coo_spmv_atomic_kernel(acc_ptr, data_ptr, rows_ptr, cols_ptr, x_ptr, nnz,
                            ACC=tl.float32, BLOCK=BLOCK)


@triton.jit
def csr_spmm_mixed_i32acc(
    c_ptr, data_ptr, cols_ptr, indptr_ptr, b_ptr, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
):
    _csr_spmm_mixed_kernel(c_ptr, data_ptr, cols_ptr, indptr_ptr, b_ptr, n_dense,
                           stride_bk, stride_bn, stride_cm, stride_cn,
                           ACC=tl.int32, BLOCK_N=BLOCK_N)


@triton.jit
def csr_spmm_mixed_f32acc(
    c_ptr, data_ptr, cols_ptr, indptr_ptr, b_ptr, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
):
    _csr_spmm_mixed_kernel(c_ptr, data_ptr, cols_ptr, indptr_ptr, b_ptr, n_dense,
                           stride_bk, stride_bn, stride_cm, stride_cn,
                           ACC=tl.float32, BLOCK_N=BLOCK_N)


@triton.jit
def coo_spmm_mixed_i32acc(
    c_ptr, data_ptr, rows_ptr, cols_ptr, b_ptr, nnz, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
):
    _coo_spmm_mixed_kernel(c_ptr, data_ptr, rows_ptr, cols_ptr, b_ptr, nnz, n_dense,
                           stride_bk, stride_bn, stride_cm, stride_cn,
                           ACC=tl.int32, BLOCK_N=BLOCK_N)


@triton.jit
def coo_spmm_mixed_f32acc(
    c_ptr, data_ptr, rows_ptr, cols_ptr, b_ptr, nnz, n_dense,
    stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_N: tl.constexpr,
):
    _coo_spmm_mixed_kernel(c_ptr, data_ptr, rows_ptr, cols_ptr, b_ptr, nnz, n_dense,
                           stride_bk, stride_bn, stride_cm, stride_cn,
                           ACC=tl.float32, BLOCK_N=BLOCK_N)


__all__ = [
    "_csr_spmv_mixed_kernel",
    "_csr_spmv_real_by_complex_kernel",
    "_coo_spmv_atomic_kernel",
    "_csr_spmm_mixed_kernel",
    "_coo_spmm_mixed_kernel",
    "csr_spmv_mixed_i32acc",
    "csr_spmv_mixed_f32acc",
    "coo_spmv_atomic_i32acc",
    "coo_spmv_atomic_f32acc",
    "csr_spmm_mixed_i32acc",
    "csr_spmm_mixed_f32acc",
    "coo_spmm_mixed_i32acc",
    "coo_spmm_mixed_f32acc",
]
