# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Bounded, mutation-aware reuse of ROCm sparse preprocessing."""

from collections import OrderedDict


class TensorPlanCache:
    def __init__(self, maxsize=8):
        self.maxsize = maxsize
        self._entries = OrderedDict()

    @staticmethod
    def _signature(tensors):
        try:
            return tuple(
                (
                    t._version,
                    t.data_ptr(),
                    t.dtype,
                    t.device,
                    tuple(t.shape),
                    t.stride(),
                )
                for t in tensors
            )
        except RuntimeError:
            # Inference tensors have no mutation counter; never reuse their plans.
            return None

    def get(self, tensors, options=()):
        key = (tuple(id(t) for t in tensors), options)
        entry = self._entries.get(key)
        if entry is None:
            return None
        sources, signature, plan = entry
        if not all(
            a is b for a, b in zip(sources, tensors)
        ) or signature != self._signature(tensors):
            del self._entries[key]
            return None
        self._entries.move_to_end(key)
        return plan

    def put(self, tensors, options, plan):
        signature = self._signature(tensors)
        if signature is None:
            return
        key = (tuple(id(t) for t in tensors), options)
        # Retaining the sources prevents Python id/storage reuse until eviction.
        self._entries[key] = (tuple(tensors), signature, plan)
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)
