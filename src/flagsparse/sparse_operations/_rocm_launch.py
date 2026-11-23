# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Reuse FlagTree 3.1 compiled kernels for repeated ROCm launches."""

from collections import OrderedDict

import torch
import triton

_RUNNERS = OrderedDict()


def launch(kernel, blocks, tensors, scalars, constants):
    if torch.cuda.current_device() != tensors[0].device.index:
        with torch.cuda.device(tensors[0].device):
            return launch(kernel, blocks, tensors, scalars, constants)
    grid = (blocks, 1, 1)
    args = (*tensors, *scalars)
    # Triton 3.1's public compiled launcher takes the non-constexpr arguments.
    # Keep JIT dispatch for other ABIs and when pre-launch hooks are attached.
    if (not triton.__version__.startswith("3.1.") or kernel.pre_run_hooks
            or kernel.debug):
        kernel[grid](*args, *constants, num_warps=4)
        return
    addresses = tuple(t.data_ptr() for t in tensors)
    signature = tuple((t.dtype, p % 16, addresses.index(p)) for t, p in zip(tensors, addresses))
    key = kernel, tensors[0].device, grid, signature, scalars, constants
    runner = _RUNNERS.get(key)
    if runner is None:
        compiled = kernel[grid](*args, *constants, num_warps=4)
        _RUNNERS[key] = compiled[grid]
        if len(_RUNNERS) > 64:
            _RUNNERS.popitem(last=False)
    else:
        runner(*args)
    return _RUNNERS[key]
