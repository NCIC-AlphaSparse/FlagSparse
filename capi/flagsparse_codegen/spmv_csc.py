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

"""CSC SpMV kernels -- re-exported, not copied. No wrappers needed here.

CSC has two structurally different directions and the C API uses both:

* op = TRANSPOSE / CONJUGATE_TRANSPOSE -> the ``_trans_`` kernels. One program
  per column, deterministic, every column written exactly once, so alpha AND
  beta fold into the store. CONJ is a plain bool constexpr, so conjugate
  transpose costs nothing extra.
* op = NON_TRANSPOSE -> the ``_non_`` kernels, which scatter into y with
  atomics. Only alpha can live in the kernel; ``beta * y`` is applied first by
  ``_dense.py``'s prologue. Being atomic, this direction is not bit-reproducible
  across runs in fp32 -- same as cuSPARSE's own atomic SpMV routes.

None of these kernels has a tl.dtype constexpr (the accumulator type follows the
operands), so unlike spmm_csr.py this module needs no jit wrappers.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from flagsparse.sparse_operations.spmv_csc import (  # noqa: E402,F401
    _spmv_csc_non_complex_kernel,
    _spmv_csc_non_real_kernel,
    _spmv_csc_trans_complex_kernel,
    _spmv_csc_trans_real_kernel,
)

__all__ = [
    "_spmv_csc_non_real_kernel",
    "_spmv_csc_non_complex_kernel",
    "_spmv_csc_trans_real_kernel",
    "_spmv_csc_trans_complex_kernel",
]
