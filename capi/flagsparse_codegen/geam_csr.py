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

"""CSR sparse matrix addition (csrgeam2) -- re-exported from
src/flagsparse/sparse_operations/geam_csr.py, not copied.

None of these kernels takes a ``tl.dtype`` constexpr (``COMPLEX`` is a bool), so
all of them are re-exported directly. The probe and place kernels are the ones
flagsparse_geam_csr runs; the scan and row-length kernels stand in for the torch
ops that function uses between them.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from flagsparse.sparse_operations.geam_csr import (  # noqa: E402,F401
    _geam_place_a_kernel,
    _geam_place_b_kernel,
    _geam_probe_kernel,
    _geam_row_nnz_kernel,
    _geam_scan_add_kernel,
    _geam_scan_blocks_kernel,
    _geam_scan_sums_kernel,
)

__all__ = [
    "_geam_probe_kernel",
    "_geam_place_a_kernel",
    "_geam_place_b_kernel",
    "_geam_row_nnz_kernel",
    "_geam_scan_blocks_kernel",
    "_geam_scan_sums_kernel",
    "_geam_scan_add_kernel",
]
