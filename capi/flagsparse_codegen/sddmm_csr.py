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

"""CSR SDDMM kernels -- re-exported from the Python operator package.

This is the one operator whose kernel already spoke cuSPARSE: it carries alpha,
beta and HAS_IN itself, so nothing had to be added for
``C = alpha * op(A) * op(B) . spy(C) + beta * C``. HAS_IN is beta != 0, and it is
constexpr for the usual reason -- beta == 0 must not read C at all.

Both dense operands are addressed by explicit strides, and the kernel wants
``y`` indexed [column][k], so op(B) = B^T is the two y strides swapped. Nothing
is materialised, same as SpMM.

``_row_ids_kernel`` expands the CSR indptr to one row id per nonzero by binary
search -- one launch instead of an arange/diff/repeat_interleave chain. The C
API asks for that array through flagsparseSDDMM_bufferSize (nnz int32) and fills
it in _preprocess. Its constexprs are ints, so it re-exports directly.

Real dtypes only: the operator package has no complex SDDMM kernel, so the
dispatch layer reports NOT_SUPPORTED for C_32F/C_64F rather than inventing one.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from flagsparse.sparse_operations.sddmm_csr import (  # noqa: E402,F401
    _row_ids_kernel,
    _sddmm_csr_real_kernel,
)


@triton.jit
def sddmm_csr_real(
    indices_ptr,
    row_ids_ptr,
    x_ptr,
    y_ptr,
    in_ptr,
    out_ptr,
    nnz,
    k_dim,
    stride_xm,
    stride_xk,
    stride_ym,
    stride_yk,
    alpha,
    beta,
    HAS_IN: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ACC_IS_FP64: tl.constexpr,
):
    """Bool-constexpr front for _sddmm_csr_real_kernel's tl.dtype ACC_DTYPE."""
    _sddmm_csr_real_kernel(
        indices_ptr,
        row_ids_ptr,
        x_ptr,
        y_ptr,
        in_ptr,
        out_ptr,
        nnz,
        k_dim,
        stride_xm,
        stride_xk,
        stride_ym,
        stride_yk,
        alpha,
        beta,
        HAS_IN,
        BLOCK_P,
        BLOCK_K,
        tl.float64 if ACC_IS_FP64 else tl.float32,
    )


__all__ = ["_row_ids_kernel", "_sddmm_csr_real_kernel", "sddmm_csr_real"]
