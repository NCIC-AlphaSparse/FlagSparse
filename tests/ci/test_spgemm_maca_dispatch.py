# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Only MACA dispatches to its TLE-free Triton product implementation."""

import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spgemm = importlib.import_module("flagsparse.sparse_operations.spgemm_csr")


@pytest.mark.parametrize("maca", [True, False])
@pytest.mark.parametrize("tle", [True, False])
def test_triton_dispatch_is_maca_only(monkeypatch, maca, tle):
    monkeypatch.setattr(spgemm, "_is_maca_runtime", lambda: maca)
    for name in ("_is_mthreads_runtime", "_is_xpu_runtime", "_is_iluvatar_runtime"):
        monkeypatch.setattr(spgemm, name, lambda: False)
    monkeypatch.setattr(spgemm, "_TLE_AVAILABLE", tle)
    monkeypatch.setattr(spgemm, "_maca_triton_spgemm_compute", lambda p: "maca_triton")
    monkeypatch.setattr(spgemm, "_spgemm_esc_compute_safe", lambda p: "esc")
    monkeypatch.setattr(spgemm, "_spgemm_hash_hybrid_compute", lambda p: "hash")
    prepared = object.__new__(spgemm.SpGEMMPrepared)
    prepared.a_data = SimpleNamespace(dtype=torch.float64)
    assert spgemm._spgemm_compute(prepared) == (
        "maca_triton" if maca else "hash" if tle else "esc"
    )


def test_maca_module_does_not_compute_with_torch():
    path = Path(spgemm.__file__).with_name("_spgemm_maca.py")
    tree = ast.parse(path.read_text())
    # Storage allocation is allowed; numerical work belongs to Triton kernels.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            root = node.func
            attributes = []
            while isinstance(root, ast.Attribute):
                attributes.append(root.attr)
                root = root.value
            if isinstance(root, ast.Name) and root.id == "torch":
                assert attributes == ["empty"], ast.unparse(node.func)
            if node.func.attr in (
                "sort",
                "argsort",
                "cumsum",
                "index_add_",
                "scatter_add_",
            ):
                assert isinstance(root, ast.Name) and root.id == "tl", ast.unparse(
                    node.func
                )
    assert not hasattr(spgemm, "_maca_torch_spgemm_compute")
