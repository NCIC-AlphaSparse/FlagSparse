# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Resolve per-backend OVERRIDES for the shared Triton operators.

The operators live once, at the top of this package. `backends/<name>/` holds
only the modules that genuinely diverge for one backend, and is empty by default;
see backends/__init__.py for why a full copy per backend was a mistake.

HOW AN OVERRIDE TAKES EFFECT. `install_overrides()` runs once, from this
package's __init__, BEFORE the `from .spmv_csr import ...` lines. For each
operator it looks for `backends/<selected>/<operator>.py`; when one exists it is
imported and registered in `sys.modules` under the shared module's name, so every
later import -- the package's own, a caller's, and the C API's path-loaded shims
-- sees the override. When none exists nothing happens and the shared module is
imported normally.

Doing it through sys.modules rather than an import hook keeps the mechanism
visible: one function, called from one place, at a point you can read.
"""

import importlib.util
import os
import sys
from importlib import import_module

# Every backend `_common._BACKEND_SPECS` knows about maps to a directory here.
# The two must stay in step: a backend present there but missing here used to
# raise RuntimeError instead of falling back, which is how `gcu` and `mlu` were
# lost when this tree briefly held one full operator copy per backend.
_IMPLEMENTATION_BACKENDS = {
    "cuda": "cuda",
    "rocm": "rocm",
    "metax": "maca",
    "mthreads": "musa",
    "ascend": "ascend",
    "xpu": "xpu",
    "gcu": "gcu",
    "mlu": "mlu",
}

# The operator modules an override may shadow. Listed explicitly rather than
# globbed so that a stray .py in a backend directory cannot silently replace
# something -- an override is a deliberate act and reads like one here.
_OPERATORS = (
    "_common",
    "_alpha_spmm_alg1_common",
    "alpha_spmm_alg1",
    "benchmarks",
    "gather_scatter",
    "sddmm_csr",
    "spgemm_csr",
    "spmm_bell",
    "spmm_bsr",
    "spmm_coo",
    "spmm_csc",
    "spmm_csr",
    "spmm_csr_opt_alg2",
    "spmv_bsr",
    "spmv_coo",
    "spmv_csc",
    "spmv_csr",
    "spsm",
    "spsv",
)


def selected_backend():
    """The override package for the running backend.

    Unknown backends fall back to the shared implementation rather than raising:
    a platform without tuned kernels still runs, which is what the registry in
    _common means by declaring a slot.
    """
    # Delaying this import keeps _common usable during package initialization.
    from ._common import _backend_name

    return _IMPLEMENTATION_BACKENDS.get(_backend_name())


def override_path(operator, backend=None):
    """Path of the override module for `operator`, or None when there is none."""
    backend = backend or selected_backend()
    if not backend:
        return None
    path = os.path.join(os.path.dirname(__file__), "backends", backend,
                        f"{operator}.py")
    return path if os.path.isfile(path) else None


def operator_module(operator):
    """The module that implements `operator`: the override if any, else shared."""
    backend = selected_backend()
    if backend and override_path(operator, backend):
        return import_module(f"{__package__}.backends.{backend}.{operator}")
    return import_module(f"{__package__}.{operator}")


def installed_overrides():
    """{operator: backend module name} for the overrides currently in effect."""
    backend = selected_backend()
    if not backend:
        return {}
    return {
        op: f"{__package__}.backends.{backend}.{op}"
        for op in _OPERATORS
        if override_path(op, backend)
    }


def install_overrides():
    """Register any per-backend overrides ahead of the shared modules.

    Returns the mapping that was installed (empty in the normal case). Called
    once from this package's __init__; calling it again is harmless.
    """
    backend = selected_backend()
    if not backend:
        return {}

    installed = {}
    for operator in _OPERATORS:
        path = override_path(operator, backend)
        if path is None:
            continue
        shared_name = f"{__package__}.{operator}"
        if shared_name in sys.modules:
            # The shared module is already imported, so replacing it now would
            # leave two live copies of the same operator -- the override for new
            # importers, the shared one for whoever already holds a reference.
            # Refuse loudly instead: install_overrides() belongs before the first
            # operator import, and being called late is a wiring bug, not a
            # condition to paper over.
            raise RuntimeError(
                f"{shared_name} was imported before install_overrides(); the "
                f"{backend} override at {path} cannot take effect. Call "
                "install_overrides() at the top of sparse_operations/__init__."
            )
        module = import_module(f"{__package__}.backends.{backend}.{operator}")
        sys.modules[shared_name] = module
        installed[operator] = module.__name__
    return installed


def export_backend_module(namespace, operator):
    """Populate a module namespace from its resolved implementation.

    Kept for modules that are thin re-export shims. The shared operators are no
    longer shims, so this has no callers inside the package today; it stays
    because the C API's flagsparse_codegen/ re-exports rely on the same idea and
    a future override may want it.
    """
    implementation = operator_module(operator)
    for name, value in vars(implementation).items():
        if not name.startswith("__"):
            namespace[name] = value
    namespace["__all__"] = getattr(
        implementation,
        "__all__",
        tuple(name for name in vars(implementation) if not name.startswith("_")),
    )
    namespace["__backend__"] = selected_backend()
    namespace["__implementation_module__"] = implementation.__name__
