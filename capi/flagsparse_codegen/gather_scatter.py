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

"""Gather / scatter kernels -- re-exported, not copied.

libtriton_jit loads this file by path and picks the kernel out by name, so a
plain re-export is enough: the object it gets is the one
``flagsparse.sparse_operations.gather_scatter`` defines, and Triton compiles it
from that file. Nothing here can drift from the Python package because nothing
here is a second copy.

Re-exported under the ORIGINAL names, underscore included. libtriton_jit's
compile step names its cache artifacts from the JITFunction's own ``__name__``,
so an alias makes the C++ side ask for metadata under a name that was never
written -- "Failed to load metadata for kernel: <alias>". The leading underscore
is the Python package's private-name convention; from the C API's side it is
simply the kernel's identity.

Semantics (cuSPARSE):
    gather  : X_values[i] = Y[X_indices[i]]
    scatter : Y[X_indices[i]] = X_values[i]

The ``*_complex`` variants take the same buffers reinterpreted as interleaved
real/imag pairs of the component dtype -- Triton has no complex type, so this is
the only representation there is.
"""

# libtriton_jit loads this file with spec_from_file_location(path.stem, path):
# the module name carries no package, so a relative import would fail. Locate the
# bootstrap through __file__ instead.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402  -- import side effect puts flagsparse on sys.path

from flagsparse.sparse_operations.gather_scatter import (  # noqa: E402,F401
    _gather_complex_kernel,
    _gather_real_kernel,
    _scatter_complex_kernel,
    _scatter_real_kernel,
)

__all__ = [
    "_gather_real_kernel",
    "_gather_complex_kernel",
    "_scatter_real_kernel",
    "_scatter_complex_kernel",
]
