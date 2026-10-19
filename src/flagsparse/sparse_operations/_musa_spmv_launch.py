# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse compiled MUSA Triton launchers, never computed outputs."""

from collections import OrderedDict
import os


_LAUNCHERS = OrderedDict()
_MAX_LAUNCHERS = 64
_RAW_STREAM = None

try:
    _NUM_WARPS = max(1, min(8, int(os.environ.get("FLAGSPARSE_MUSA_TRITON_WARPS", "4"))))
except (TypeError, ValueError):
    _NUM_WARPS = 4


def _has_hook(hook):
    # FlagTree installs empty HookChains instead of None even without a profiler.
    return hook is not None and (not hasattr(hook, "calls") or bool(hook.calls))


def fast_path_allowed(kernel):
    # Imported here, not at module level: every backend imports this module via
    # gather_scatter/mixed_spmx/spmv_sell, and older Triton builds (CoreX 3.2.3
    # ships Triton 2.3.1) have no triton.knobs. Only MUSA ever reaches this call.
    from triton import knobs

    return not (
        kernel.pre_run_hooks or knobs.runtime.debug
        or knobs.compilation.instrumentation_mode
        or knobs.runtime.add_stages_inspection_hook is not None
        or _has_hook(knobs.runtime.launch_enter_hook)
        or _has_hook(knobs.runtime.launch_exit_hook)
    )


class _Launch:
    """Own the compiled module but resolve pointers/streams on every launch."""

    def __init__(self, kernel, compiled, grid, device):
        self.kernel = kernel
        self.compiled = compiled
        self.grid = grid
        self.device = device
        self.runner = compiled[grid]
        self.direct = hasattr(compiled, "run") and device.type == "musa"

    def __call__(self, tensors, scalars, constants, addresses=None):
        global _RAW_STREAM
        if not fast_path_allowed(self.kernel):
            return self.kernel[self.grid](
                *tensors, *scalars, *constants, num_warps=_NUM_WARPS, num_stages=1,
            )
        if not self.direct:
            return self.runner(*tensors, *scalars, *constants)
        if _RAW_STREAM is None:
            import torch_musa

            _RAW_STREAM = torch_musa._MUSAC._musa_getCurrentRawStream
        if addresses is None:
            addresses = tuple(t.data_ptr() for t in tensors)
        # This is Triton's normal compiled launcher, including scratch handling.
        # Only pointer validation and empty profiling metadata are redundant.
        self.compiled.run(
            *self.grid, _RAW_STREAM(self.device.index), self.compiled.function,
            self.compiled.packed_metadata, None, None, None,
            *addresses, *scalars, *constants,
        )


def launch(kernel, grid, tensors, scalars, constants):
    """Cache dtype/device/alignment/alias specializations, not tensor addresses.

    Returns the bound specialization after executing it. Callers may reuse that
    object only while validating the same specialization themselves. Debugging,
    instrumentation and profiling always use the full JIT path.
    """
    if not fast_path_allowed(kernel):
        kernel[grid](*tensors, *scalars, *constants, num_warps=_NUM_WARPS, num_stages=1)
        return None
    addresses = tuple(t.data_ptr() for t in tensors)
    signature = tuple((t.dtype, t.device, p % 16) for t, p in zip(tensors, addresses))
    aliases = ()
    if len(set(addresses)) != len(addresses):
        aliases = tuple((i, j) for j, p in enumerate(addresses) if p
                        for i in range(j) if addresses[i] == p)
    grid = tuple(grid) + (1,) * (3 - len(grid))
    key = (id(kernel), getattr(kernel, "hash", None), grid, signature, aliases,
           scalars, constants, kernel.debug, _NUM_WARPS)
    bound = _LAUNCHERS.get(key)
    if bound is None:
        compiled = kernel[grid](
            *tensors, *scalars, *constants, num_warps=_NUM_WARPS, num_stages=1,
        )
        if compiled is None:
            return None
        key = (id(kernel), getattr(kernel, "hash", None), *key[2:])
        bound = _Launch(kernel, compiled, grid, tensors[0].device)
        _LAUNCHERS[key] = bound
        while len(_LAUNCHERS) > _MAX_LAUNCHERS:
            _LAUNCHERS.popitem(last=False)
    else:
        _LAUNCHERS.move_to_end(key)
        bound(tensors, scalars, constants, addresses)
    return bound
