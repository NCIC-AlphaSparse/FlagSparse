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

"""CSR SpGEMM kernels -- re-exported, not copied. No wrappers needed.

The shared-memory hash route: a program owns one row of C, builds a hash table
of that row's column indices in scratchpad memory, and reports either the row's
nnz (count phase) or the row itself (fill phase).

These kernels are defined under ``if _TLE_AVAILABLE`` in the operator package,
because the scratchpad allocation is a FlagTree TLE feature. Importing this
module therefore fails outright on a build without TLE rather than silently
serving a different route, which is the honest failure: the package's other
SpGEMM route (expand-sort-compress) is pure torch and has no kernels to
re-export at all.

The fill kernel comes in two flavours rather than taking an accumulator
constexpr -- TLE's smem alloc needs a literal dtype -- so the dispatch layer
picks by name, and no jit wrapper is involved.

Two contracts the dispatch layer owns:

* ``a_pref`` and ``rw`` are the product-count prefix per A nonzero and the
  product count per row. They are what flagsparseSpGEMM_workEstimation computes.
* ``CAP`` must be a power of two with 0.75 * CAP >= the row's product count, or
  the row overflows and sets its ovf flag. The C API turns any overflow into
  NOT_SUPPORTED, since the package's ESC fallback is not reachable from here.

Columns come out in hash-slot order, NOT sorted. cuSPARSE guarantees sorted CSR,
so flagsparseSpGEMM_copy sorts each row afterwards.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from flagsparse.sparse_operations.spgemm_csr import (  # noqa: E402,F401
    _spgemm_hash_count_kernel,
    _spgemm_hash_fill_kernel,
    _spgemm_hash_fill_kernel_f64,
)

__all__ = [
    "_spgemm_hash_count_kernel",
    "_spgemm_hash_fill_kernel",
    "_spgemm_hash_fill_kernel_f64",
]
