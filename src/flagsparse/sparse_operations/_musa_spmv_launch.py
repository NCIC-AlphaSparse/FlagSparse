# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse compiled MUSA SpMV launchers, never operands or computed outputs."""

from collections import OrderedDict


_LAUNCHERS = OrderedDict()
_MAX_LAUNCHERS = 64


def launch(kernel, grid, tensors, scalars, constants):
    """Keep the pointer types, alignment and aliases of Triton's specialization.

    Sparse plan caches still validate tensor versions/storage on every call.
    This cache only bypasses repeated JIT argument binding for the same kernel
    signature. CompiledKernel's public launcher resolves the current stream
    and runs Triton's profiling hooks on every invocation.
    """
    from triton import knobs

    if (
        kernel.pre_run_hooks or knobs.runtime.debug
        or knobs.compilation.instrumentation_mode
        or knobs.runtime.add_stages_inspection_hook is not None
    ):
        return kernel[grid](*tensors, *scalars, *constants, num_warps=4, num_stages=1)
    addresses = tuple(t.data_ptr() for t in tensors)
    signature = tuple((t.dtype, t.device, p % 16) for t, p in zip(tensors, addresses))
    aliases = ()
    if len(set(addresses)) != len(addresses):
        aliases = tuple((i, j) for j, p in enumerate(addresses) if p
                        for i in range(j) if addresses[i] == p)
    grid = tuple(grid) + (1,) * (3 - len(grid))
    key = (kernel, grid, signature, aliases, scalars, constants, kernel.debug)
    runner = _LAUNCHERS.get(key)
    if runner is None:
        compiled = kernel[grid](*tensors, *scalars, *constants, num_warps=4, num_stages=1)
        if compiled is not None:
            _LAUNCHERS[key] = compiled[grid]
            while len(_LAUNCHERS) > _MAX_LAUNCHERS:
                _LAUNCHERS.popitem(last=False)
        return compiled
    _LAUNCHERS.move_to_end(key)
    runner(*tensors, *scalars, *constants)
