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

import triton
import triton.language as tl

from flagsparse.sparse_operations.spgemm_csr import (  # noqa: E402,F401
    _spgemm_hash_count_kernel,
    _spgemm_hash_fill_kernel,
    _spgemm_hash_fill_kernel_f64,
)


@triton.jit
def musa_spgemm_scan_rows(counts, overflow, indptr, summary, nrows, nblocks,
                          BLOCK: tl.constexpr):
    block = tl.program_id(0)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    count = tl.load(counts + rows, rows < nrows, 0)
    failed = tl.load(overflow + rows, rows < nrows, 0)
    tl.store(indptr + rows + 1, tl.cumsum(count, 0), rows < nrows)
    if block == 0:
        tl.store(indptr, 0)
    tl.store(summary + block, tl.sum(count.to(tl.int64), 0))
    tl.store(summary + nblocks + block, tl.max(count, 0).to(tl.int64))
    tl.store(summary + 2 * nblocks + block, tl.max(failed, 0).to(tl.int64))


@triton.jit
def musa_spgemm_scan_blocks(summary, offsets, metadata, nblocks,
                            BLOCK: tl.constexpr):
    block = tl.arange(0, BLOCK)
    count = tl.load(summary + block, block < nblocks, 0)
    maximum = tl.load(summary + nblocks + block, block < nblocks, 0)
    failed = tl.load(summary + 2 * nblocks + block, block < nblocks, 0)
    tl.store(offsets + block, tl.cumsum(count, 0) - count, block < nblocks)
    tl.store(metadata, tl.sum(count, 0))
    tl.store(metadata + 1, tl.max(maximum, 0))
    tl.store(metadata + 2, tl.max(failed, 0))


@triton.jit
def musa_spgemm_add_offsets(indptr, offsets, nrows, BLOCK: tl.constexpr):
    block = tl.program_id(0)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    local = tl.load(indptr + rows + 1, rows < nrows, 0)
    offset = tl.load(offsets + block)
    tl.store(indptr + rows + 1, (local.to(tl.int64) + offset).to(tl.int32), rows < nrows)


__all__ = [
    "_spgemm_hash_count_kernel",
    "_spgemm_hash_fill_kernel",
    "_spgemm_hash_fill_kernel_f64",
    "musa_spgemm_scan_rows",
    "musa_spgemm_scan_blocks",
    "musa_spgemm_add_offsets",
]
