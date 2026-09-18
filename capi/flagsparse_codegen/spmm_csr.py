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

"""CSR SpMM kernels -- re-exported from the Python operator package, not copied.

Both dense operands are addressed through explicit strides, so row-major and
column-major (FLAGSPARSE_ORDER_ROW / _COL) are the same kernel with different
stride arguments -- no second code path, and no transpose materialised. opB =
TRANSPOSE is likewise just the two B strides swapped.

``alpha``/``beta``/``HAS_BETA`` live in the PYTHON package's kernels so that
cuSPARSE's ``C = alpha*op(A)*op(B) + beta*C`` is one launch; the operator
library passes 1/0 and ``HAS_BETA=False``, for which the generated code is
unchanged. One kernel, two front ends, nothing to drift.

The two ``@triton.jit`` functions below are the exception to "nothing here is
code": see the ACC_DTYPE note.
"""

# libtriton_jit loads this file with spec_from_file_location(path.stem, path):
# the module name carries no package, so a relative import would fail. Locate the
# bootstrap through __file__ instead.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402  -- import side effect puts flagsparse on sys.path

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

# Re-exported under the ORIGINAL names, underscore included. libtriton_jit's
# compile step names its cache artifacts from the JITFunction's own __name__, so
# an alias would make the C++ side ask for metadata under a name that was never
# written -- "Failed to load metadata for kernel: <alias>".
from flagsparse.sparse_operations.spmm_csr import (  # noqa: E402,F401
    _spmm_csr_complex_kernel,
    _spmm_csr_real_kernel,
)


# --------------------------------------------------------------------------
# ACC_DTYPE is a tl.dtype OBJECT in the kernels above, and libtriton_jit's
# raw-args signature parser understands only bool / int / float constexprs --
# anything else parses to None, and tl.zeros(dtype=None) raises. Rather than
# demote a precision knob to a bool in 28 kernels across the operator package
# (which would block a future tf32 accumulator) or patch the bridge, the C API
# calls a thin jit wrapper that takes the bool and picks the dtype.
#
# This costs nothing: Triton inlines jit-to-jit calls, so the wrapper does not
# appear in the generated PTX. The instruction streams differ in exactly one
# place -- the inlined early-return lowers to two instructions instead of four
# -- and measured timings were never slower than calling the kernel directly.
#
# Kernel bodies still live in ONE place. These functions contain no arithmetic;
# retuning or fixing SpMM means editing the Python package, as before.
# --------------------------------------------------------------------------


@triton.jit
def spmm_csr_real(
    data_ptr,
    indices_ptr,
    indptr_ptr,
    b_ptr,
    c_ptr,
    alpha,
    beta,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_IS_FP64: tl.constexpr,
    ACCURACY: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Bool-constexpr front for _spmm_csr_real_kernel's tl.dtype ACC_DTYPE."""
    _spmm_csr_real_kernel(
        data_ptr,
        indices_ptr,
        indptr_ptr,
        b_ptr,
        c_ptr,
        alpha,
        beta,
        n_rows,
        n_dense_cols,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_N,
        BLOCK_NNZ,
        tl.float64 if ACC_IS_FP64 else tl.float32,
        ACCURACY,
        HAS_BETA,
    )


@triton.jit
def spmm_csr_complex(
    data_ri_ptr,
    indices_ptr,
    indptr_ptr,
    b_ri_ptr,
    c_ri_ptr,
    alpha_re,
    alpha_im,
    beta_re,
    beta_im,
    n_rows,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_br,
    stride_cm,
    stride_cn,
    stride_cr,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_IS_FP64: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Bool-constexpr front for _spmm_csr_complex_kernel.

    Complex operands are interleaved real/imag pairs of the component dtype --
    Triton has no complex type, and this is what the Python package already
    does (torch.view_as_real). stride_br / stride_cr are the distance from a
    real part to its imaginary part, in elements of that component dtype.
    """
    _spmm_csr_complex_kernel(
        data_ri_ptr,
        indices_ptr,
        indptr_ptr,
        b_ri_ptr,
        c_ri_ptr,
        alpha_re,
        alpha_im,
        beta_re,
        beta_im,
        n_rows,
        n_dense_cols,
        stride_bk,
        stride_bn,
        stride_br,
        stride_cm,
        stride_cn,
        stride_cr,
        BLOCK_N,
        BLOCK_NNZ,
        tl.float64 if ACC_IS_FP64 else tl.float32,
        HAS_BETA,
    )


__all__ = [
    "_spmm_csr_real_kernel",
    "_spmm_csr_complex_kernel",
    "spmm_csr_real",
    "spmm_csr_complex",
]
