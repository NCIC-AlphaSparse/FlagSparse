# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""CSC SpMM kernels re-exported for the C API."""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from flagsparse.sparse_operations.spmm_csc import (  # noqa: E402,F401
    _spmm_csc_non_complex_kernel,
    _spmm_csc_non_real_kernel,
)
