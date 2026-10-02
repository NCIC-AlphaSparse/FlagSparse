# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""C API wrappers for Sliced-ELLPACK SpMV kernels."""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from flagsparse.sparse_operations.spmv_sell import _spmv_sell_real_kernel  # noqa: E402
from flagsparse.sparse_operations.spmv_sell import _spmv_sell_complex_kernel  # noqa: E402


@triton.jit
def spmv_sell_f32acc(
    y_ptr, values_ptr, cols_ptr, offsets_ptr, x_ptr, n_rows,
    SLICE: tl.constexpr, BLOCK_R: tl.constexpr,
):
    _spmv_sell_real_kernel(
        y_ptr, values_ptr, cols_ptr, offsets_ptr, x_ptr, n_rows,
        SLICE=SLICE, BLOCK_R=BLOCK_R, ACC=tl.float32,
    )


@triton.jit
def spmv_sell_i32acc(
    y_ptr, values_ptr, cols_ptr, offsets_ptr, x_ptr, n_rows,
    SLICE: tl.constexpr, BLOCK_R: tl.constexpr,
):
    _spmv_sell_real_kernel(
        y_ptr, values_ptr, cols_ptr, offsets_ptr, x_ptr, n_rows,
        SLICE=SLICE, BLOCK_R=BLOCK_R, ACC=tl.int32,
    )

