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

"""The dense prologue shared by every accumulate-style route -- re-exported.

Routes that build their result with tl.atomic_add (CSC op="non", BSR, the COO
atomic variants) cannot fold ``beta * y`` into a store, so the C API applies it
first with this kernel. All its constexprs are bool/int, so no wrapper is
needed: it is a plain re-export.

``_dense_copy_kernel`` is here for the same reason from the other direction:
SpSM's solver works in place on a packed row-major work array, while cuSPARSE
hands it a separate right-hand side and destination with arbitrary order and
leading dimension.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bootstrap  # noqa: F401,E402

from flagsparse.sparse_operations._common import (  # noqa: E402,F401
    _dense_copy_kernel,
    _dense_scale_kernel,
)

__all__ = ["_dense_scale_kernel", "_dense_copy_kernel"]
