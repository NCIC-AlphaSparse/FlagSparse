# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep complex SpVV specialization inside the MACA complex64 size gate."""

import importlib

import pytest

torch = pytest.importorskip("torch")
vector = importlib.import_module("flagsparse.sparse_operations.vector_ops")
maca_vector = importlib.import_module("flagsparse.sparse_operations._vector_maca")


@pytest.mark.parametrize("maca", [False, True])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128, torch.float32])
@pytest.mark.parametrize("op", ["non", "conj"])
def test_backend_and_dtype_scope(monkeypatch, maca, dtype, op):
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

    monkeypatch.setattr(maca_vector, "spvv_complex", specialized)
    vector.flagsparse_spvv(
        torch.ones(2, dtype=dtype),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(3, dtype=dtype),
        op=op,
        validate=False,
    )
    assert calls == (["maca"] if maca and dtype == torch.complex64 else ["generic"])


@pytest.mark.parametrize("nnz", [4096, 4097])
def test_size_gate(monkeypatch, nnz):
    monkeypatch.setattr(vector, "_is_maca_runtime", lambda: True)
    monkeypatch.setattr(vector, "_is_rocm_runtime", lambda: False)
    monkeypatch.setattr(vector, "_is_ascend_runtime", lambda: False)
    monkeypatch.setattr(vector, "_is_accel_tensor", lambda tensor: True)

    class GenericReached(Exception):
        pass

    def real_dtype(dtype):
        raise GenericReached

    monkeypatch.setattr(vector, "_real_dtype", real_dtype)
    marker = torch.zeros((), dtype=torch.complex64)
    monkeypatch.setattr(maca_vector, "spvv_complex", lambda *args: marker)
    values = torch.ones(nnz, dtype=torch.complex64)
    indices = torch.zeros(nnz, dtype=torch.int32)
    y = torch.ones(1, dtype=torch.complex64)
    if nnz <= 4096:
        assert vector.flagsparse_spvv(values, indices, y, validate=False) is marker
    else:
        with pytest.raises(GenericReached):
            vector.flagsparse_spvv(values, indices, y, validate=False)
