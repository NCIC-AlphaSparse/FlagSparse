# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Complex Axpby fusion is restricted to MACA and bounded dense sizes."""

import importlib

import pytest

torch = pytest.importorskip("torch")
vector = importlib.import_module("flagsparse.sparse_operations.vector_ops")
maca_vector = importlib.import_module("flagsparse.sparse_operations._vector_maca")


@pytest.mark.parametrize("maca", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128, torch.float32])
def test_fused_backend_and_dtype_scope(monkeypatch, maca, dtype):
    monkeypatch.setattr(vector, "_is_maca_runtime", lambda: maca)
    for name in (
        "_is_mthreads_runtime",
        "_is_iluvatar_runtime",
        "_is_rocm_runtime",
        "_is_ascend_runtime",
    ):
        monkeypatch.setattr(vector, name, lambda: False)
    monkeypatch.setattr(vector, "_is_accel_tensor", lambda tensor: True)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append("generic")

    for name in (
        "_scale_complex_kernel",
        "_axpy_complex_kernel",
        "_scale_real_kernel",
        "_axpy_real_kernel",
    ):
        monkeypatch.setattr(vector, name, Kernel())
    monkeypatch.setattr(
        maca_vector, "axpby_complex", lambda *args: calls.append("maca")
    )
    y = torch.ones(7, dtype=dtype)
    assert (
        vector.flagsparse_axpby(
            torch.ones(1, dtype=dtype),
            torch.tensor([3], dtype=torch.int32),
            y,
            1.5,
            0.5,
            validate=False,
        )
        is y
    )
    assert calls == (
        ["maca"] if maca and dtype == torch.complex64 else ["generic", "generic"]
    )


@pytest.mark.parametrize("size", [4 * 1024 * 1024, 4 * 1024 * 1024 + 1])
def test_dense_map_size_bound(monkeypatch, size):
    monkeypatch.setattr(vector, "_is_maca_runtime", lambda: True)
    for name in (
        "_is_mthreads_runtime",
        "_is_iluvatar_runtime",
        "_is_rocm_runtime",
        "_is_ascend_runtime",
    ):
        monkeypatch.setattr(vector, name, lambda: False)
    monkeypatch.setattr(vector, "_is_accel_tensor", lambda tensor: True)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append("generic")

    monkeypatch.setattr(vector, "_scale_complex_kernel", Kernel())
    monkeypatch.setattr(vector, "_axpy_complex_kernel", Kernel())
    monkeypatch.setattr(
        maca_vector, "axpby_complex", lambda *args: calls.append("maca")
    )
    y = torch.empty(size, dtype=torch.complex64)
    vector.flagsparse_axpby(
        torch.ones(1, dtype=torch.complex64),
        torch.tensor([0], dtype=torch.int32),
        y,
        1.5,
        0.5,
        validate=False,
    )
    assert calls == (["maca"] if size <= 4 * 1024 * 1024 else ["generic", "generic"])
