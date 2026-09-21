# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA's CSR SpMV baseline is timed on the PyTorch sparse route it can run.

On MetaX/MACA there is no cuSPARSE binding, so the delivery run selects PyTorch
(``FLAGSPARSE_MACA_VENDOR=torch``) as the vendor baseline. ``measure_vendor`` then
has to time the sparse format ``_pytorch_sparse_mm`` found to work (int32 CSR or
COO) rather than report ``N/A``.
"""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]

benchmark_spec = importlib.util.spec_from_file_location(
    "flagsparse.sparse_operations._spmv_csr_benchmark_contract",
    ROOT / "src/flagsparse/sparse_operations/_spmv_csr_benchmark.py",
)
benchmark = importlib.util.module_from_spec(benchmark_spec)
sys.modules[benchmark_spec.name] = benchmark
benchmark_spec.loader.exec_module(benchmark)


def test_csr_runner_times_the_pytorch_baseline_selected_for_maca(monkeypatch):
    data = torch.tensor([2.0], dtype=torch.float32)
    indices = torch.tensor([0], dtype=torch.int32)
    indptr = torch.tensor([0, 1], dtype=torch.int32)
    x = torch.tensor([3.0], dtype=torch.float32)
    matrix = torch.sparse_coo_tensor(torch.tensor([[0], [0]]), data, (1, 1)).coalesce()

    monkeypatch.setattr(
        benchmark.common,
        "_spmv_csr_sparse_ref_backend",
        lambda *_args, **_kwargs: ("torch", None),
    )
    monkeypatch.setattr(
        benchmark.common,
        "_pytorch_sparse_mm",
        lambda *_args, **_kwargs: (torch.tensor([[6.0]]), "COO"),
    )
    monkeypatch.setattr(
        benchmark.common,
        "_pytorch_sparse_coo_matrix",
        lambda *_args, **_kwargs: matrix,
    )
    monkeypatch.setattr(
        benchmark,
        "event_benchmark",
        lambda fn, _warmup, _iters: (fn(), 2.0),
    )
    monkeypatch.setattr(
        benchmark, "_spmv_device_context", lambda _device: nullcontext()
    )

    result = benchmark.measure_vendor(
        data, indices, indptr, x, (1, 1), "non", warmup=0, iters=1
    )

    assert result["vendor_backend"] == "torch"
    assert result["vendor_alg"] == "torch_sparse_coo_matvec"
    assert result["vendor_ms"] == 2.0
    assert torch.equal(result["values"], torch.tensor([6.0]))


def test_maca_selects_torch_as_the_csr_spmv_baseline_only_when_asked(monkeypatch):
    """`none` means no baseline; `torch` means PyTorch. Only the latter is timed."""
    common = benchmark.common
    monkeypatch.setattr(common, "_IS_MACA_RUNTIME", True)

    monkeypatch.setattr(common, "_vendor_sparse_library", lambda: "torch")
    assert common._spmv_csr_sparse_ref_backend("float32", "int32", "non") == (
        "torch",
        None,
    )

    monkeypatch.setattr(common, "_vendor_sparse_library", lambda: None)
    backend, reason = common._spmv_csr_sparse_ref_backend("float32", "int32", "non")
    assert backend is None and "not wired" in reason


def test_torch_is_not_a_csr_spmv_baseline_off_maca(monkeypatch):
    """Other backends that resolve to `torch` keep their existing behaviour."""
    common = benchmark.common
    monkeypatch.setattr(common, "_IS_MACA_RUNTIME", False)
    monkeypatch.setattr(common, "_vendor_sparse_library", lambda: "torch")
    backend, reason = common._spmv_csr_sparse_ref_backend("float32", "int32", "non")
    assert backend is None and "not wired" in reason
