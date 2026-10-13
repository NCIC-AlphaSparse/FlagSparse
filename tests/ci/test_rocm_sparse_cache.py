# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Sparse plan reuse must never hide tensor mutation or grow without bound."""

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
_path = (
    Path(__file__).resolve().parents[2]
    / "src/flagsparse/sparse_operations/_rocm_sparse_cache.py"
)
_spec = importlib.util.spec_from_file_location("rocm_sparse_cache", _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
TensorPlanCache = _module.TensorPlanCache


def test_cache_checks_identity_options_and_all_input_versions():
    cache = TensorPlanCache()
    inputs = tuple(torch.arange(4) for _ in range(3))
    plan = object()
    cache.put(inputs, (4, 4), plan)
    assert cache.get(inputs, (4, 4)) is plan
    assert cache.get(inputs, (4, 5)) is None
    assert cache.get(tuple(t.clone() for t in inputs), (4, 4)) is None
    for tensor in inputs:
        cache.put(inputs, (4, 4), plan)
        tensor.add_(1)
        assert cache.get(inputs, (4, 4)) is None


def test_cache_checks_storage_replacement_and_lru_limit():
    cache = TensorPlanCache(maxsize=2)
    a, b, c = (torch.arange(4) for _ in range(3))
    cache.put((a,), (), "a")
    cache.put((b,), (), "b")
    assert cache.get((a,)) == "a"
    cache.put((c,), (), "c")
    assert cache.get((b,)) is None
    a.data = a.clone()
    assert cache.get((a,)) is None
    assert len(cache._entries) <= 2


def test_inference_tensors_are_not_cached():
    cache = TensorPlanCache()
    with torch.inference_mode():
        tensor = torch.ones(4)
        cache.put((tensor,), (), object())
        assert cache.get((tensor,)) is None
        assert not cache._entries
