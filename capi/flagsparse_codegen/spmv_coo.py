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

"""COO SpMV kernels -- re-exported from the Python operator package, not copied.

The C API drives the deterministic *segment* route with ``SEG_IS_ROW=True``, so
``seg_starts`` is a full row-offsets array (length n_rows + 1) rather than the
run-compressed form the Python path builds. Same reason as spmm_coo.py: a row
with no nonzeros must still run to pick up ``beta * y``. The array is the
caller's scratch, sized by flagsparseSpMV_bufferSize.

The real kernels are split by dtype (``_f32`` / ``_f64``) with the accumulator
baked in, so they re-export directly -- the dispatch layer picks the name. Only
the complex one takes ACC_DTYPE as a tl.dtype object and therefore needs the
bool-constexpr wrapper below; see flagsparse_codegen/spmm_csr.py for why.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from flagsparse.sparse_operations.spmv_coo import (  # noqa: E402,F401
    _spmv_coo_seg_complex,
    _spmv_coo_seg_f32,
    _spmv_coo_seg_f64,
)


@triton.jit
def spmv_coo_seg_complex(
    data_ri_ptr,
    col_ptr,
    row_ptr,
    x_ri_ptr,
    y_ri_ptr,
    seg_starts_ptr,
    alpha_re,
    alpha_im,
    beta_re,
    beta_im,
    n_segs,
    BLOCK_INNER: tl.constexpr,
    ACC_IS_FP64: tl.constexpr,
    SEG_IS_ROW: tl.constexpr,
    HAS_BETA: tl.constexpr,
):
    """Bool-constexpr front for _spmv_coo_seg_complex's tl.dtype ACC_DTYPE."""
    _spmv_coo_seg_complex(
        data_ri_ptr,
        col_ptr,
        row_ptr,
        x_ri_ptr,
        y_ri_ptr,
        seg_starts_ptr,
        alpha_re,
        alpha_im,
        beta_re,
        beta_im,
        n_segs,
        BLOCK_INNER,
        tl.float64 if ACC_IS_FP64 else tl.float32,
        SEG_IS_ROW,
        HAS_BETA,
    )


__all__ = [
    "_spmv_coo_seg_f32",
    "_spmv_coo_seg_f64",
    "_spmv_coo_seg_complex",
    "spmv_coo_seg_complex",
]
