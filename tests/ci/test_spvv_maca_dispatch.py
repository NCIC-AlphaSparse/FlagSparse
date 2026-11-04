# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA SpVV dtype/size scope and deterministic reduction routing."""

import importlib

import pytest

torch = pytest.importorskip("torch")
vector = importlib.import_module("flagsparse.sparse_operations.vector_ops")
maca_vector = importlib.import_module("flagsparse.sparse_operations._vector_maca")


@pytest.mark.parametrize("maca", [False, True])
@pytest.mark.parametrize(
    "dtype,out_dtype",
    [
        (torch.complex64, None),
        (torch.complex128, None),
        (torch.float32, None),
        (torch.float64, None),
        (torch.float16, None),
        (torch.int8, None),
        (torch.int8, torch.float32),
    ],
)
@pytest.mark.parametrize("op", ["non", "conj"])
def test_backend_and_dtype_scope(monkeypatch, maca, dtype, out_dtype, op):
    monkeypatch.setattr(vector, "_is_maca_runtime", lambda: maca)
    monkeypatch.setattr(vector, "_is_rocm_runtime", lambda: False)
    monkeypatch.setattr(vector, "_is_ascend_runtime", lambda: False)
    monkeypatch.setattr(vector, "_is_iluvatar_runtime", lambda: False)
    monkeypatch.setattr(vector, "_backend_name", lambda: "cuda")
    monkeypatch.setattr(vector, "_is_accel_tensor", lambda tensor: True)
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append("generic")

    monkeypatch.setattr(vector, "_spvv_complex_kernel", Kernel())
    monkeypatch.setattr(vector, "_spvv_real_kernel", Kernel())

    def specialized(values, indices, y, conj):
        calls.append("maca")
        assert conj == (op == "conj")
        return torch.zeros((), dtype=dtype)

    monkeypatch.setattr(maca_vector, "spvv", specialized)
    vector.flagsparse_spvv(
        torch.ones(2, dtype=dtype),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(3, dtype=dtype),
        op=op,
        validate=False,
        out_dtype=out_dtype,
    )
    assert calls == (
        ["maca"] if maca and dtype in (torch.float32, torch.complex64) else ["generic"]
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("nnz", [4096, 4097, 65536, 65537])
@pytest.mark.parametrize("deterministic", [False, True])
def test_size_and_determinism_gate(monkeypatch, dtype, nnz, deterministic):
    monkeypatch.setattr(vector, "_is_maca_runtime", lambda: True)
    monkeypatch.setattr(vector, "_is_rocm_runtime", lambda: False)
    monkeypatch.setattr(vector, "_is_ascend_runtime", lambda: False)
    monkeypatch.setattr(vector, "_is_accel_tensor", lambda tensor: True)
    monkeypatch.setattr(
        torch, "are_deterministic_algorithms_enabled", lambda: deterministic
    )

    class GenericReached(Exception):
        pass

    def real_dtype(dtype):
        raise GenericReached

    monkeypatch.setattr(vector, "_real_dtype", real_dtype)
    marker = torch.zeros((), dtype=dtype)
    monkeypatch.setattr(maca_vector, "spvv", lambda *args: marker)
    values = torch.ones(nnz, dtype=dtype)
    indices = torch.zeros(nnz, dtype=torch.int32)
    y = torch.ones(1, dtype=dtype)
    if nnz <= 65536 and (nnz <= 4096 or not deterministic):
        assert vector.flagsparse_spvv(values, indices, y, validate=False) is marker
    else:
        with pytest.raises(GenericReached):
            vector.flagsparse_spvv(values, indices, y, validate=False)
