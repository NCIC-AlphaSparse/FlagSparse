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

"""Dense matrix times sparse vector (gemvi) -- the kernels of
src/flagsparse/sparse_operations/gemvi.py, re-exported, not copied.

All three take ``ACC: tl.constexpr`` (the accumulate dtype), which the raw-args
signature parser cannot spell, so each gets one bool-constexpr wrapper
(``ACC_IS_FP64``) delegating straight to the real kernel.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from flagsparse.sparse_operations.gemvi import (  # noqa: E402,F401
    _gemvi_complex_kernel,
    _gemvi_real_kernel,
    _gemvi_reduce_kernel,
)


@triton.jit
def gemvi_real(
    y_ptr, a_ptr, x_ptr, idx_ptr, n_out, nnz,
    stride_out, stride_k, alpha, beta, k_chunk, part_stride,
    HAS_BETA: tl.constexpr, ACC_IS_FP64: tl.constexpr,
    BLOCK_OUT: tl.constexpr, BLOCK_K: tl.constexpr, OUT_INNER: tl.constexpr,
):
    _gemvi_real_kernel(
        y_ptr, a_ptr, x_ptr, idx_ptr, n_out, nnz,
        stride_out, stride_k, alpha, beta, k_chunk, part_stride,
        HAS_BETA, tl.float64 if ACC_IS_FP64 else tl.float32,
        BLOCK_OUT, BLOCK_K, OUT_INNER,
    )


@triton.jit
def gemvi_complex(
    y_ri_ptr, a_ri_ptr, x_ri_ptr, idx_ptr, n_out, nnz,
    stride_out, stride_k, alpha_re, alpha_im, beta_re, beta_im, k_chunk, part_stride,
    HAS_BETA: tl.constexpr, ACC_IS_FP64: tl.constexpr,
    BLOCK_OUT: tl.constexpr, BLOCK_K: tl.constexpr,
):
    _gemvi_complex_kernel(
        y_ri_ptr, a_ri_ptr, x_ri_ptr, idx_ptr, n_out, nnz,
        stride_out, stride_k, alpha_re, alpha_im, beta_re, beta_im, k_chunk, part_stride,
        HAS_BETA, tl.float64 if ACC_IS_FP64 else tl.float32, BLOCK_OUT, BLOCK_K,
    )


@triton.jit
def gemvi_reduce(
    y_ptr, part_ptr, n_out, n_split, alpha_re, alpha_im, beta_re, beta_im,
    HAS_BETA: tl.constexpr, COMPLEX: tl.constexpr, ACC_IS_FP64: tl.constexpr,
    BLOCK: tl.constexpr,
):
    _gemvi_reduce_kernel(
        y_ptr, part_ptr, n_out, n_split, alpha_re, alpha_im, beta_re, beta_im,
        HAS_BETA, COMPLEX, tl.float64 if ACC_IS_FP64 else tl.float32, BLOCK,
    )


__all__ = [
    "_gemvi_real_kernel",
    "_gemvi_complex_kernel",
    "_gemvi_reduce_kernel",
    "gemvi_real",
    "gemvi_complex",
    "gemvi_reduce",
]
