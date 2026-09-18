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

"""BSR SpMV kernels -- re-exported, not copied. No wrappers needed.

Both directions accumulate with tl.atomic_add, so only alpha rides in the
kernel and ``beta * y`` comes from ``_dense.py``'s prologue. Being atomic, BSR
SpMV is not bit-reproducible across runs in fp32.

The C API launches with ``SEG_FROM_GRID=True``, which takes the segment index
from program_id(2) instead of the SEG constexpr. That matters: with SEG as a
constexpr the host has to loop over segments, and every distinct value compiles
its own kernel -- a matrix needing 20 segments pays 20 JIT compilations for one
solve. From the grid it is one compile and one launch. The Python path still
passes False and generates exactly the code it always did.

Blocks are read row-major within a block (``inner_row * BLOCK_DIM + inner_col``)
and BLOCK_DIM is used for both block extents, so the dispatch layer refuses
non-square blocks and FLAGSPARSE_ORDER_COL rather than reading them wrongly.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from flagsparse.sparse_operations.spmv_bsr import (  # noqa: E402,F401
    _spmv_bsr_non_complex_kernel,
    _spmv_bsr_non_real_kernel,
    _spmv_bsr_trans_complex_kernel,
    _spmv_bsr_trans_real_kernel,
)

__all__ = [
    "_spmv_bsr_non_real_kernel",
    "_spmv_bsr_non_complex_kernel",
    "_spmv_bsr_trans_real_kernel",
    "_spmv_bsr_trans_complex_kernel",
]
