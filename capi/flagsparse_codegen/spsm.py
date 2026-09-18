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

"""Triangular solve with multiple right-hand sides -- re-exported, no wrappers.

These kernels already carry alpha, and their accumulator type is a bool
constexpr, so nothing had to be added to them.

The polling route runs one program per (row, RHS tile) and spins on a per-(tile,
row) done flag. Unlike SpSV's chain-wave route there is no worker pool: the row
IS the block index, so progress depends on blocks being scheduled roughly in
launch order. That holds on every backend here, and it is why the row index is
mapped so that block 0 is always the row with no dependencies -- ``pid_row`` for
a lower triangle, ``n_rows - 1 - pid_row`` for an upper one.

Unlike SpSV this route does NOT care about column order: it filters each row by
``col < row`` / ``col > row`` and scans the whole row either way, so ascending
CSR works for both fill modes with no reordering and no backward-scan flag.

Three things the dispatch layer owns:

* ``work`` is a PACKED row-major n_rows x n_rhs array that holds B on entry and
  Y on exit -- the solve is in place. cuSPARSE hands over a separate B and C
  with arbitrary order and leading dimension, so the C API copies in and out
  with ``_dense.py``'s strided copy. ``stride_work0`` counts ELEMENTS, and the
  complex kernel doubles it itself.
* ``done`` (rhs_tiles x n_rows int32) must be zero at launch.
* ``diag`` is extracted once by ``_spsm_extract_diag_kernel_*``; it depends only
  on the matrix, so the C API does it in _analysis.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from flagsparse.sparse_operations.spsm import (  # noqa: E402,F401
    _spsm_csr_polling_kernel_complex,
    _spsm_csr_polling_kernel_real,
    _spsm_extract_diag_kernel_complex,
    _spsm_extract_diag_kernel_real,
)

__all__ = [
    "_spsm_csr_polling_kernel_real",
    "_spsm_csr_polling_kernel_complex",
    "_spsm_extract_diag_kernel_real",
    "_spsm_extract_diag_kernel_complex",
]
