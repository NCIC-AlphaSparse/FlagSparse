# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Minimal fp16 sparse AXPBY kernels for the C API."""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import triton
import triton.language as tl

from flagsparse.sparse_operations.vector_ops import (  # noqa: E402
    _axpy_real_kernel,
    _scale_real_kernel,
)


@triton.jit
def axpby_scale_f16(y_ptr, n, beta, BLOCK: tl.constexpr):
    _scale_real_kernel(y_ptr, n, beta, COMPUTE=tl.float32, BLOCK=BLOCK)


@triton.jit
def axpby_add_f16(y_ptr, x_ptr, idx_ptr, nnz, alpha, BLOCK: tl.constexpr):
    _axpy_real_kernel(y_ptr, x_ptr, idx_ptr, nnz, alpha,
                      COMPUTE=tl.float32, BLOCK=BLOCK)
