# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MUSA half-input specialization must leave every other dtype/backend alone."""

import importlib

import pytest

torch = pytest.importorskip("torch")
vector = importlib.import_module("flagsparse.sparse_operations.vector_ops")
musa_vector = importlib.import_module("flagsparse.sparse_operations._vector_musa")


@pytest.mark.parametrize("musa", [False, True])
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.int8,
    ],
)
@pytest.mark.parametrize("op", ["non", "conj"])
def test_half_backend_scope(monkeypatch, musa, dtype, op):
    monkeypatch.setattr(vector, "_is_mthreads_runtime", lambda: musa)
    for name in (
        "_is_maca_runtime",
        "_is_rocm_runtime",
        "_is_ascend_runtime",
        "_is_iluvatar_runtime",
    ):
        monkeypatch.setattr(vector, name, lambda: False)
    monkeypatch.setattr(vector, "_backend_name", lambda: "cuda")
    monkeypatch.setattr(vector, "_is_accel_tensor", lambda tensor: True)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def run(*args, **kwargs):
                calls.append("generic")

            return run

    monkeypatch.setattr(vector, "_spvv_real_kernel", Kernel())
    monkeypatch.setattr(vector, "_spvv_complex_kernel", Kernel())

    def specialized(values, indices, y):
        calls.append("musa")
        assert values.dtype == torch.float16
        return torch.zeros((), dtype=torch.float32)

    monkeypatch.setattr(musa_vector, "spvv_half", specialized)
    vector.flagsparse_spvv(
        torch.ones(2, dtype=dtype),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(3, dtype=dtype),
        op=op,
        validate=False,
    )
    assert calls == (["musa"] if musa and dtype == torch.float16 else ["generic"])
