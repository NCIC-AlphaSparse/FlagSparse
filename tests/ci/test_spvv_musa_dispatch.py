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
    "dtype,out_dtype",
    [
        (torch.float16, None),
        (torch.bfloat16, None),
        (torch.float32, None),
        (torch.float64, None),
        (torch.complex64, None),
        (torch.int8, None),
        (torch.int8, torch.int32),
        (torch.int8, torch.float32),
    ],
)
@pytest.mark.parametrize("op", ["non", "conj"])
def test_narrow_backend_scope(monkeypatch, musa, dtype, out_dtype, op):
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

    def specialized_int8(values, indices, y, compute):
        calls.append("musa_int8")
        assert values.dtype == torch.int8
        assert compute == (out_dtype or torch.int32)
        return torch.zeros((), dtype=compute)

    monkeypatch.setattr(musa_vector, "spvv_int8", specialized_int8)
    vector.flagsparse_spvv(
        torch.ones(2, dtype=dtype),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(3, dtype=dtype),
        op=op,
        validate=False,
        out_dtype=out_dtype,
    )
    expected = "generic"
    if musa and dtype == torch.float16:
        expected = "musa"
    elif musa and dtype == torch.int8:
        expected = "musa_int8"
    assert calls == [expected]
