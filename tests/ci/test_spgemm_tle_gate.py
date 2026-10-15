# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""SpGEMM keeps the TLE shared-memory hash path off the backends that cannot run it."""

import importlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spgemm = importlib.import_module("flagsparse.sparse_operations.spgemm_csr")

BACKENDS = ("_is_mthreads_runtime", "_is_xpu_runtime", "_is_iluvatar_runtime")


def _route(monkeypatch, backend):
    for name in BACKENDS:
        monkeypatch.setattr(spgemm, name, lambda value=(name == backend): value)
    monkeypatch.setattr(spgemm, "_TLE_AVAILABLE", True)
    monkeypatch.setattr(spgemm, "_spgemm_hash_hybrid_compute", lambda p: "hash")
    monkeypatch.setattr(spgemm, "_spgemm_esc_compute_safe", lambda p: "esc")
    prepared = SimpleNamespace(a_data=torch.zeros(1, dtype=torch.float32))
    return spgemm._spgemm_compute(prepared)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_without_working_tle_use_esc(monkeypatch, backend):
    # Iluvatar: "iluvatar TLE alloc does not support nv_mma_shared_layout=True"
    # failed every BI-V150 SpGEMM row (2026-10-09).
    assert _route(monkeypatch, backend) == "esc"


def test_other_backends_keep_the_hash_path(monkeypatch):
    assert _route(monkeypatch, None) == "hash"
