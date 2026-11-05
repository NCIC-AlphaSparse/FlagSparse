# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""MACA-only CSC transpose selection; CUDA/ROCm retain their existing routes."""

import importlib

import pytest

torch = pytest.importorskip("torch")
csc = importlib.import_module("flagsparse.sparse_operations.spmv_csc")
gather = importlib.import_module("flagsparse.sparse_operations._spmv_csr_transpose")


@pytest.mark.parametrize("maca", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64, torch.float64])
@pytest.mark.parametrize("op", ["non", "trans", "conj"])
def test_transpose_gather_selection(monkeypatch, maca, dtype, op):
    monkeypatch.setattr(csc, "_is_maca_runtime", lambda: maca)
    monkeypatch.setattr(csc, "_is_rocm_runtime", lambda: False)
    monkeypatch.setattr(csc, "_is_ascend_runtime", lambda: False)
    monkeypatch.setattr(csc, "_SPMV_CSC_CUDA", False)
    monkeypatch.setattr(
        csc,
        "_get_device_backend_info",
        lambda device: {
            "backend": "metax" if maca else "other",
            "device_warp_size": 64,
        },
    )
    marker = object()
    calls = []

    def plan(indices, ptr, order, n):
        calls.append((indices, ptr, order, n))
        return marker

    monkeypatch.setattr(gather, "make_plan", plan)
    values = torch.ones(1, dtype=dtype)
    indices = torch.zeros(1, dtype=torch.int32)
    ptr = torch.tensor([0, 1], dtype=torch.int32)
    prepared = csc.PreparedCscSpmv(values, indices, ptr, (1, 1), 1, 1, 256, 1, 1, op=op)
    selected = maca and op != "non" and dtype != torch.float64
    assert (prepared.maca_trans_plan is marker) == selected
    assert len(calls) == int(selected)
    if selected:
        result = object()

        def compute(data, x, plan, n, conjugate=False):
            assert data is values and plan is marker and n == 1
            assert conjugate == (op == "conj")
            return result

        monkeypatch.setattr(gather, "gather", compute)
        assert csc._triton_spmv_csc_kernel(prepared, values, prepared.op) is result
