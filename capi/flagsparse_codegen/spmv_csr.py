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

"""CSR SpMV kernels -- re-exported from the Python operator package, not copied.

``_spmv_csr_real_kernel`` carries ``alpha``/``beta``/``HAS_BETA`` so that
cuSPARSE's ``y = alpha*A*x + beta*y`` is one launch. Those parameters live in the
PYTHON package's kernel, not in a C-side fork: the operator library passes 1/0
(``HAS_BETA=False`` folds the beta term away at compile time, leaving exactly the
code it always generated), and the C++ layer passes the caller's values. One
kernel, two front ends, nothing to drift.

Re-exported under the ORIGINAL names, underscore included. libtriton_jit's
compile step names its cache artifacts from the JITFunction's own ``__name__``,
so an alias makes the C++ side ask for metadata under a name that was never
written -- "Failed to load metadata for kernel: <alias>". The leading underscore
is the Python package's private-name convention; from the C API's side it is
simply the kernel's identity.

"""

# libtriton_jit loads this file with spec_from_file_location(path.stem, path):
# the module name carries no package, so a relative import would fail. Locate the
# bootstrap through __file__ instead.
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402  -- import side effect puts flagsparse on sys.path

from flagsparse.sparse_operations.spmv_csr import (  # noqa: E402,F401
    _spmv_csr_complex_kernel,
    _spmv_csr_real_kernel,
)

# The complex kernel is a separate function rather than a dtype constexpr on the
# real one, so no jit wrapper is involved: the dispatch layer picks by name.
# Complex operands are interleaved real/imag pairs of the component dtype, and
# alpha/beta arrive split the same way -- Triton has no complex type, and this is
# what the Python package already does (torch.view_as_real).
__all__ = ["_spmv_csr_real_kernel", "_spmv_csr_complex_kernel"]
