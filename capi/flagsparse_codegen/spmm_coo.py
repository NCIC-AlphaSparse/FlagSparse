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

"""COO SpMM kernels -- re-exported from the Python operator package, not copied.

The C API drives the *rowrun* route with ``SEG_IS_ROW=True``: ``seg_starts`` is a
full row-offsets array of length n_rows + 1 rather than the run-compressed form
the Python path builds, so the segment index is the row id, an empty row still
gets a program, and ``beta * C`` reaches every row without a second pass. That
offsets array is exactly a CSR indptr, which is why the C++ side asks for it
through ``flagsparseSpMM_bufferSize`` -- it is scratch the caller owns, not a
hidden allocation.

The other two COO routes in the operator package (atomic, alg1 bucket) are not
wired up here. The atomic one needs C pre-scaled by beta in a separate pass and
is non-deterministic by construction; the bucket one needs two prepare kernels.

``BLOCK_NNZ`` deserves a warning: the rowrun kernels unroll
``tl.static_range(0, BLOCK_NNZ)``, so the body is emitted BLOCK_NNZ times
whatever the row length is. The operator package's 30-matrix sweep landed on a
flat 4; the old default of 256 ran ~253 dead loads per useful one on a
short-rowed matrix and cost ~7x. The C++ side hardcodes 4 for the same reason --
do not "tune it up".
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

# Re-exported under the ORIGINAL names, underscore included: libtriton_jit names
# its cache artifacts from the JITFunction's own __name__.
from flagsparse.sparse_operations.spmm_coo import (  # noqa: E402,F401
    _spmm_coo_rowrun_complex_kernel,
    _spmm_coo_rowrun_real_kernel,
)


# See flagsparse_codegen/spmm_csr.py for why these wrappers exist: the raw-args
# signature parser handles only bool / int / float constexprs, and ACC_DTYPE is a
# tl.dtype object. Triton inlines the call away; the wrapper holds no arithmetic.


@triton.jit
def spmm_coo_real(
    data_ptr,
    row_ptr,
    col_ptr,
    b_ptr,
    c_ptr,
    seg_starts_ptr,
    alpha,
    beta,
    n_segs,
    n_dense_cols,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_NNZ: tl.constexpr,
    ACC_IS_FP64: tl.constexpr,
    SEG_IS_ROW: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Bool-constexpr front for _spmm_coo_rowrun_real_kernel."""
    _spmm_coo_rowrun_real_kernel(
        data_ptr,
        row_ptr,
        col_ptr,
        b_ptr,
        c_ptr,
        seg_starts_ptr,
        alpha,
        beta,
        n_segs,
        n_dense_cols,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_N,
        BLOCK_NNZ,
        tl.float64 if ACC_IS_FP64 else tl.float32,
        SEG_IS_ROW,
        HAS_BETA,
    )


@triton.jit
def spmm_coo_complex(
    data_ri_ptr,
    row_ptr,
    col_ptr,
    b_ri_ptr,
    c_ri_ptr,
    seg_starts_ptr,
    alpha_re,
    alpha_im,
    beta_re,
    beta_im,
    n_segs,
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
    SEG_IS_ROW: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Bool-constexpr front for _spmm_coo_rowrun_complex_kernel."""
    _spmm_coo_rowrun_complex_kernel(
        data_ri_ptr,
        row_ptr,
        col_ptr,
        b_ri_ptr,
        c_ri_ptr,
        seg_starts_ptr,
        alpha_re,
        alpha_im,
        beta_re,
        beta_im,
        n_segs,
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
        SEG_IS_ROW,
        HAS_BETA,
    )


__all__ = [
    "_spmm_coo_rowrun_real_kernel",
    "_spmm_coo_rowrun_complex_kernel",
    "spmm_coo_real",
    "spmm_coo_complex",
]
