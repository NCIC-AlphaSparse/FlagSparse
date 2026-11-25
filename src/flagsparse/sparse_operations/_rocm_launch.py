# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse compiled Triton kernels for repeated ROCm launches."""

from collections import OrderedDict
from functools import lru_cache

import torch
import triton

from ._musa_spmv_launch import fast_path_allowed

_RUNNERS = OrderedDict()


@lru_cache(maxsize=1)
def _launcher_version():
    return tuple(triton.__version__.split(".")[:2])


def can_launch(kernel):
    version = _launcher_version()
    if kernel.pre_run_hooks or kernel.debug:
        return False
    if version == ("3", "1"):
        return True
    hcu = getattr(getattr(triton, "knobs", None), "hcu", None)
    return (version == ("3", "6") and hcu is not None
            and not hcu.use_buffer_ops
            and fast_path_allowed(kernel))


def _bind_compiled(compiled, grid, constants, device_index, pointer_count):
    run = compiled[grid]
    if _launcher_version() == ("3", "1"):
        return run

    entry = compiled.run
    if (type(entry).__module__ == "triton.backends.hcu.driver"
            and callable(getattr(entry, "launch", None))
            and getattr(entry, "profile_scratch_size", None) == 0
            and getattr(entry, "launch_cooperative_grid", None) is False):
        raw = entry.launch
    else:
        raw = None

    # HCU's launcher accepts device addresses and the original constexpr slots.
    def bound(*args, stream=None):
        nonlocal raw
        if stream is None:
            stream = torch._C._cuda_getCurrentRawStream(device_index)
        if raw is not None:
            addresses = tuple(t.data_ptr() for t in args[:pointer_count])
            try:
                raw(False, *grid, stream, compiled.function, None,
                    compiled.packed_metadata, None, None, None,
                    *addresses, *args[pointer_count:], *constants)
                return
            except TypeError:
                # A different HCU launcher ABI rejects the arguments before launch.
                raw = None
        run(*args, *constants, stream=stream)

    return bound


def launch(kernel, blocks, tensors, scalars, constants, *, num_warps=4):
    device = tensors[0].device
    if torch.cuda.current_device() != device.index:
        with torch.cuda.device(device):
            return launch(kernel, blocks, tensors, scalars, constants, num_warps=num_warps)
    grid = (blocks, 1, 1)
    args = (*tensors, *scalars)
    if (not can_launch(kernel)
            or any(t.device != device for t in tensors[1:])):
        kernel[grid](*args, *constants, num_warps=num_warps)
        return
    addresses = tuple(t.data_ptr() for t in tensors)
    signature = tuple((t.dtype, p % 16, addresses.index(p)) for t, p in zip(tensors, addresses))
    scalar_signature = tuple((type(value), value) for value in scalars)
    key = kernel, device, grid, signature, scalar_signature, constants, num_warps
    runner = _RUNNERS.get(key)
    if runner is None:
        compiled = kernel[grid](*args, *constants, num_warps=num_warps)
        if compiled is None:
            return None
        runner = _bind_compiled(compiled, grid, constants, device.index, len(tensors))
        _RUNNERS[key] = runner
        if len(_RUNNERS) > 64:
            _RUNNERS.popitem(last=False)
    else:
        runner(*args)
    return runner
