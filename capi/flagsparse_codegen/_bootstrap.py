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

"""Make the ``flagsparse`` Python package importable from the embedded runtime.

Every module in this package RE-EXPORTS kernels from the Python operator library
instead of copying them.  That is deliberate: a copied kernel is a second source
of truth, and the two drift the moment someone retunes a BLOCK size or fixes a
numerical bug on one side only.  Re-exporting makes drift impossible -- the
object the C++ layer launches IS the object the Python package uses, and Triton
records the original file as the kernel's source.

The package location is not derivable from here (this directory ships with the C
library, the Python package does not), so the C++ layer passes it in as
``FLAGSPARSE_PYTHON_SRC``, baked at configure time and overridable at runtime.
"""

import os
import sys


def ensure_flagsparse_importable():
    """Put the configured operator package at the FRONT of sys.path, then import.

    Precedence matters and is the opposite of the obvious "try import first, fall
    back to the configured path": an installed ``flagsparse`` in site-packages
    will satisfy a bare import and silently win over the tree the build was
    configured against. That is not hypothetical -- a stale dist-packages copy on
    this machine still had the pre-alpha/beta SpMV kernel, and the only symptom
    was ``number of argument mismatch: Actual(11), Function Definition(8)`` from
    deep inside the JIT, which says nothing about which copy got loaded.

    So: when FLAGSPARSE_PYTHON_SRC is set it wins, full stop. Only when it is
    unset do we accept whatever is importable.

    Returns the resolved package file, for diagnostics.
    """
    candidate = os.environ.get("FLAGSPARSE_PYTHON_SRC", "").strip()
    if candidate:
        if not os.path.isdir(candidate):
            raise ImportError(f"FLAGSPARSE_PYTHON_SRC={candidate!r} is not a directory")
        # Front of the path, and drop any already-imported copy: this module may
        # be executed after something else has pulled in the installed package.
        if sys.path[:1] != [candidate]:
            while candidate in sys.path:
                sys.path.remove(candidate)
            sys.path.insert(0, candidate)
        for name in [m for m in sys.modules if m == "flagsparse" or m.startswith("flagsparse.")]:
            del sys.modules[name]

    try:
        import flagsparse
    except ImportError as exc:
        raise ImportError(
            "the flagsparse Python package is not importable.\n"
            "The C++ dispatch layer launches the SAME kernel objects that package "
            "defines, so it must be reachable. Set FLAGSPARSE_PYTHON_SRC to its "
            "`src` directory, or put it on PYTHONPATH."
        ) from exc

    resolved = getattr(flagsparse, "__file__", "<unknown>")
    if candidate and not os.path.abspath(resolved).startswith(os.path.abspath(candidate)):
        # Defensive: if the configured path did not actually win, say so loudly
        # rather than compiling against a copy nobody asked for.
        raise ImportError(
            f"FLAGSPARSE_PYTHON_SRC={candidate!r} was requested but flagsparse "
            f"resolved to {resolved!r}. Something else is shadowing it."
        )
    return resolved


ensure_flagsparse_importable()
