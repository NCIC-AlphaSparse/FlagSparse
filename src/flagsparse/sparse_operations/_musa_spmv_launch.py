# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse compiled MUSA / CUDA Triton launchers, never computed outputs.

Triton's JIT dispatch costs more host time than a short SpMV kernel runs:
16 us of a 39 us mixed CSR call on a 5090, against a 7 us direct launch.
"""

from collections import OrderedDict
import os

from ._rocm_sparse_cache import TensorPlanCache


class MusaTensorPlanCache(TensorPlanCache):
    """Keep the usual mutation checks without rebuilding a hot identity key."""

    def __init__(self, maxsize=8):
        super().__init__(maxsize)
        self._hot = None

    @staticmethod
    def _signature(tensors):
        if len(tensors) != 3:
            return TensorPlanCache._signature(tensors)
        # CSR/COO/SELL all have three source arrays. Retain every metadata
        # check, avoiding a generator frame and duplicate shape tuples.
        a, b, c = tensors
        try:
            return (
                (a._version, a.data_ptr(), a.dtype, a.device, a.shape, a.stride()),
                (b._version, b.data_ptr(), b.dtype, b.device, b.shape, b.stride()),
                (c._version, c.data_ptr(), c.dtype, c.device, c.shape, c.stride()),
            )
        except RuntimeError:
            return None

    def get(self, tensors, options=()):
        hot = self._hot
        if hot is not None:
            sources, previous_options, signature, plan = hot
            if (options == previous_options and len(tensors) == 3
                    and sources[0] is tensors[0] and sources[1] is tensors[1]
                    and sources[2] is tensors[2]):
                if signature == self._signature(tensors):
                    return plan
                self._hot = None
        plan = super().get(tensors, options)
        if plan is not None:
            entry = self._entries[(tuple(id(t) for t in tensors), options)]
            self._hot = (entry[0], options, entry[1], plan)
        return plan

    def put(self, tensors, options, plan):
        super().put(tensors, options, plan)
        entry = self._entries.get((tuple(id(t) for t in tensors), options))
        self._hot = ((entry[0], options, entry[1], plan)
                     if entry is not None else None)


_LAUNCHERS = OrderedDict()
_MAX_LAUNCHERS = 64
_RAW_STREAM = None
_CUDA_RAW_STREAM = None

try:
    _NUM_WARPS = max(1, min(8, int(os.environ.get("FLAGSPARSE_MUSA_TRITON_WARPS", "4"))))
except (TypeError, ValueError):
    _NUM_WARPS = 4


def _has_hook(hook):
    # FlagTree installs empty HookChains instead of None even without a profiler.
    return hook is not None and (not hasattr(hook, "calls") or bool(hook.calls))


_KNOBS = None


def fast_path_allowed(kernel):
    # Imported here, not at module level: every backend imports this module via
    # gather_scatter/mixed_spmx/spmv_sell, and older Triton builds (CoreX 3.2.3
    # ships Triton 2.3.1) have no triton.knobs. Only MUSA and CUDA reach this call.
    global _KNOBS
    if _KNOBS is None:
        from triton import knobs as _KNOBS
    knobs = _KNOBS

    return not (
        kernel.pre_run_hooks or knobs.runtime.debug
        or knobs.compilation.instrumentation_mode
        or knobs.runtime.add_stages_inspection_hook is not None
        or _has_hook(knobs.runtime.launch_enter_hook)
        or _has_hook(knobs.runtime.launch_exit_hook)
    )


class _Launch:
    """Own the compiled module but resolve pointers/streams on every launch."""

    def __init__(self, kernel, compiled, grid, device, options, cuda=False):
        self.kernel = kernel
        self.compiled = compiled
        self.grid = grid
        self.device = device
        self.options = options
        self.runner = compiled[grid]
        self.cuda = cuda
        self.direct = hasattr(compiled, "run") and (device.type == "musa" or cuda)
        # MUSA's compiled entry and module metadata are immutable. Resolving
        # these lazy properties on every short call adds host launch overhead.
        self.musa_entry = None
        if self.direct and not cuda:
            entry = compiled.run
            # Use the compiled Triton launcher directly only when its Python
            # wrapper has no descriptor expansion or scratch allocation to do.
            if (hasattr(entry, "launch")
                    and getattr(entry, "_needs_runtime_expansion", True) is False
                    and getattr(entry, "global_scratch_size", None) == 0):
                entry = entry.launch
            self.musa_entry = (entry, compiled.function, compiled.packed_metadata)

    def __call__(self, tensors, scalars, constants, addresses=None, checked=False):
        global _RAW_STREAM, _CUDA_RAW_STREAM
        if not checked and not fast_path_allowed(self.kernel):
            return self.kernel[self.grid](*tensors, *scalars, *constants, **self.options)
        if not self.direct:
            return self.runner(*tensors, *scalars, *constants)
        if self.cuda:
            if _CUDA_RAW_STREAM is None:
                # What Triton's own CUDA driver uses for the current stream.
                from torch._C import _cuda_getCurrentRawStream

                _CUDA_RAW_STREAM = _cuda_getCurrentRawStream
            stream = _CUDA_RAW_STREAM(self.device.index)
        else:
            if _RAW_STREAM is None:
                import torch_musa

                _RAW_STREAM = torch_musa._MUSAC._musa_getCurrentRawStream
            stream = _RAW_STREAM(self.device.index)
        if addresses is None:
            addresses = tuple(t.data_ptr() for t in tensors)
        # This is Triton's normal compiled launcher, including scratch handling.
        # Only pointer validation and empty profiling metadata are redundant.
        try:
            if self.musa_entry is not None:
                entry, function, metadata = self.musa_entry
            else:
                entry, function, metadata = (self.compiled.run, self.compiled.function,
                                              self.compiled.packed_metadata)
            entry(
                *self.grid, stream, function,
                metadata, None, None, None,
                *addresses, *scalars, *constants,
            )
        except TypeError:
            # A Triton whose launcher takes other arguments rejects them while
            # parsing, before anything is launched: use its JIT path from now on.
            if not self.cuda:
                raise
            self.direct = False
            self.runner = lambda *args: self.kernel[self.grid](*args, **self.options)
            self.runner(*tensors, *scalars, *constants)


def launch(kernel, grid, tensors, scalars, constants, num_warps=None, num_stages=None,
           cuda=False):
    """Cache dtype/device/alignment/alias specializations, not tensor addresses.

    Returns the bound specialization after executing it. Callers may reuse that
    object only while validating the same specialization themselves. Debugging,
    instrumentation and profiling always use the full JIT path.

    ``num_warps``/``num_stages`` default to the MUSA settings (Triton's default
    stages under ``cuda=True``). ``cuda=True``
    enables the direct launch on a device that presents as torch.cuda; pass it
    only for NVIDIA CUDA (ROCm, MACA and Iluvatar ship their own Triton launchers).
    """
    options = {"num_warps": _NUM_WARPS if num_warps is None else num_warps}
    if num_stages is not None or not cuda:  # CUDA keeps Triton's default
        options["num_stages"] = 1 if num_stages is None else num_stages
    if not fast_path_allowed(kernel):
        kernel[grid](*tensors, *scalars, *constants, **options)
        return None
    addresses = tuple(t.data_ptr() for t in tensors)
    signature = tuple((t.dtype, t.device, p % 16) for t, p in zip(tensors, addresses))
    aliases = ()
    if len(set(addresses)) != len(addresses):
        aliases = tuple((i, j) for j, p in enumerate(addresses) if p
                        for i in range(j) if addresses[i] == p)
    grid = tuple(grid) + (1,) * (3 - len(grid))
    key = (id(kernel), getattr(kernel, "hash", None), grid, signature, aliases,
           scalars, constants, kernel.debug, tuple(sorted(options.items())))
    bound = _LAUNCHERS.get(key)
    if bound is None:
        compiled = kernel[grid](*tensors, *scalars, *constants, **options)
        if compiled is None:
            return None
        key = (id(kernel), getattr(kernel, "hash", None), *key[2:])
        bound = _Launch(kernel, compiled, grid, tensors[0].device, options, cuda)
        _LAUNCHERS[key] = bound
        while len(_LAUNCHERS) > _MAX_LAUNCHERS:
            _LAUNCHERS.popitem(last=False)
    else:
        _LAUNCHERS.move_to_end(key)
        bound(tensors, scalars, constants, addresses, checked=True)
    return bound
